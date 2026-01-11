import os
import time
import warnings
from pathlib import Path
from typing import Callable

import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F  # noqa: N812

from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.model_loader import load_text_encoder
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.hqq_quantization import quantize_model_hqq
from ltx_trainer.quantization import quantize_model
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.utils import get_gpu_memory_gb, open_image_as_srgb, save_image
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import read_video, save_video

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


class LtxvTrainer:
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._cached_validation_embeddings = self._load_text_encoder_and_cache_embeddings()
        self._load_models()
        self._setup_accelerator()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths = []
        self._init_wandb()

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
        profile_steps: bool = True,  # Profile first few steps for timing analysis
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        self._profile_enabled = profile_steps
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        # Use the same seed for all processes and ensure deterministic operations
        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()
        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=cfg.optimization.steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        self._transformer.train()
        self._global_step = 0

        peak_mem_during_training = start_mem

        sampled_videos_paths = None

        with progress:
            # Initial validation before training starts
            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                sampled_videos_paths = self._sample_videos(progress)
                if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                    self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

            self._accelerator.wait_for_everyone()

            for step in range(cfg.optimization.steps * cfg.optimization.gradient_accumulation_steps):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start_time = time.time()

                # Profiling: track timing breakdown
                _profile_enabled = getattr(self, '_profile_enabled', False) and step < 20
                _t = {}

                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    if is_optimization_step:
                        self._global_step += 1

                    if _profile_enabled:
                        torch.cuda.synchronize()
                        _t['forward_start'] = time.perf_counter()

                    loss = self._training_step(batch)

                    if _profile_enabled:
                        torch.cuda.synchronize()
                        _t['forward_end'] = time.perf_counter()

                    self._accelerator.backward(loss)

                    if _profile_enabled:
                        torch.cuda.synchronize()
                        _t['backward_end'] = time.perf_counter()

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()

                    if _profile_enabled:
                        torch.cuda.synchronize()
                        _t['optimizer_end'] = time.perf_counter()

                    # Custom optimizer step for FP8 parameters (only on actual optimizer steps, not accumulation)
                    if self._accelerator.sync_gradients:
                        current_lr = self._lr_scheduler.get_last_lr()[0] if self._lr_scheduler else self._config.optimization.learning_rate
                        self._fp8_optimizer_step(current_lr)

                        if _profile_enabled:
                            torch.cuda.synchronize()
                            _t['fp8_opt_end'] = time.perf_counter()
                            # Log timing breakdown
                            fwd = _t['forward_end'] - _t['forward_start']
                            bwd = _t['backward_end'] - _t['forward_end']
                            opt = _t['optimizer_end'] - _t['backward_end']
                            fp8 = _t['fp8_opt_end'] - _t['optimizer_end']
                            total = _t['fp8_opt_end'] - _t['forward_start']
                            logger.info(f"⏱️ Step {self._global_step} timing: fwd={fwd:.2f}s bwd={bwd:.2f}s opt={opt:.2f}s fp8={fp8:.2f}s total={total:.2f}s")

                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        if self._accelerator.distributed_type == DistributedType.FSDP:
                            # FSDP: All processes must participate in validation
                            sampled_videos_paths = self._sample_videos(progress)
                            if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)
                        # DDP: Only main process runs validation
                        elif IS_MAIN_PROCESS:
                            sampled_videos_paths = self._sample_videos(progress)
                            if sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        self._save_checkpoint()

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress and log metrics
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps

                    progress.update_training(
                        loss=loss.item(),
                        lr=current_lr,
                        step_time=step_time,
                        advance=is_optimization_step,
                    )

                    # Log metrics to W&B (only on main process and optimization steps)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        self._log_metrics(
                            {
                                "train/loss": loss.item(),
                                "train/learning_rate": current_lr,
                                "train/step_time": step_time,
                                "train/global_step": self._global_step,
                            }
                        )

                    # Fallback logging when progress bars are disabled
                    if disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:
                        elapsed = time.time() - train_start_time
                        progress_percentage = self._global_step / cfg.optimization.steps
                        if progress_percentage > 0:
                            total_estimated = elapsed / progress_percentage
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."
                        logger.info(
                            f"Step {self._global_step}/{cfg.optimization.steps} - "
                            f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, "
                            f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                        )

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = cfg.optimization.steps / total_time_seconds

        samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        saved_path = self._save_checkpoint()

        # Verify FP8 weight changes (compare original vs trained)
        if IS_MAIN_PROCESS and getattr(self, '_has_fp8_training', False):
            self._verify_weight_changes(saved_path)

        if IS_MAIN_PROCESS:
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

        return saved_path, stats

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> Tensor:
        """Perform a single training step using the configured strategy."""
        # Apply embedding connectors to transform pre-computed text embeddings
        conditions = batch["conditions"]
        video_embeds, audio_embeds, attention_mask = self._text_encoder._run_connectors(
            conditions["prompt_embeds"], conditions["prompt_attention_mask"]
        )
        conditions["video_prompt_embeds"] = video_embeds
        conditions["audio_prompt_embeds"] = audio_embeds
        conditions["prompt_attention_mask"] = attention_mask

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)

        return loss

    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        """Load text encoder, computes and returns validation embeddings."""

        # This method:
        #   1. Loads the text encoder on GPU
        #   2. If validation prompts are configured, computes and caches their embeddings
        #   3. Unloads the heavy Gemma model while keeping the lightweight embedding connectors
        #   The text encoder is kept (as self._text_encoder) but with model/tokenizer/feature_extractor
        #   set to None. Only the embedding connectors remain for use during training.

        # Load text encoder on GPU
        logger.debug("Loading text encoder...")
        if self._config.acceleration.load_text_encoder_in_8bit:
            logger.warning(
                "⚠️  load_text_encoder_in_8bit is set to True but 8-bit text encoder loading "
                "is not currently implemented. The text encoder will be loaded in bfloat16 precision."
            )

        self._text_encoder = load_text_encoder(
            checkpoint_path=self._config.model.model_path,
            gemma_model_path=self._config.model.text_encoder_path,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Cache validation embeddings if prompts are configured
        cached_embeddings = None
        if self._config.validation.prompts:
            logger.info(f"Pre-computing embeddings for {len(self._config.validation.prompts)} validation prompts...")
            cached_embeddings = []
            with torch.inference_mode():
                for prompt in self._config.validation.prompts:
                    v_ctx_pos, a_ctx_pos, _ = self._text_encoder(prompt)
                    v_ctx_neg, a_ctx_neg, _ = self._text_encoder(self._config.validation.negative_prompt)

                    cached_embeddings.append(
                        CachedPromptEmbeddings(
                            video_context_positive=v_ctx_pos.cpu(),
                            audio_context_positive=a_ctx_pos.cpu(),
                            video_context_negative=v_ctx_neg.cpu() if v_ctx_neg is not None else None,
                            audio_context_negative=a_ctx_neg.cpu() if a_ctx_neg is not None else None,
                        )
                    )

        # Unload Gemma model to free VRAM (~17B params = ~34GB in bf16)
        # Keep embedding connectors on GPU (~400MB) - needed for _run_connectors during training
        del self._text_encoder.model
        del self._text_encoder.tokenizer
        del self._text_encoder.feature_extractor_linear
        self._text_encoder.model = None
        self._text_encoder.tokenizer = None
        self._text_encoder.feature_extractor_linear = None
        # Force garbage collection to free GPU memory immediately
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        vram_after = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"Validation prompt embeddings cached. Gemma unloaded, connectors kept (VRAM: {vram_after:.2f} GB)")
        return cached_embeddings

    def _load_models(self) -> None:
        """Load the LTX-2 model components."""
        # Load audio components if:
        # 1. Training strategy requires audio (training the audio branch), OR
        # 2. Validation is configured to generate audio (even if not training audio)
        load_audio = self._training_strategy.requires_audio or self._config.validation.generate_audio

        # Check if we need VAE encoder (for image or reference video conditioning)
        need_vae_encoder = (
            self._config.validation.images is not None or self._config.validation.reference_videos is not None
        )

        # Load all model components (except text encoder - already handled)
        # For FP8 models, preserve original dtypes to keep FP8 weights
        preserve_fp8 = self._config.acceleration.load_fp8_model
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=need_vae_encoder,  # Needed for image conditioning
            with_video_vae_decoder=True,  # Needed for validation sampling
            with_audio_vae_decoder=load_audio,
            with_vocoder=load_audio,
            with_text_encoder=False,  # Text encoder handled separately
            preserve_dtype=preserve_fp8,
        )

        # Extract components
        self._transformer = components.transformer
        self._vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(dtype=torch.bfloat16)
        self._scheduler = components.scheduler
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder
        # Note: self._text_encoder was set in _load_text_encoder_and_cache_embeddings

        # Determine initial dtype based on training mode.
        # Note: For FSDP + LoRA, we'll cast to FP32 later in _prepare_models_for_training()
        # after the accelerator is set up, and we can detect FSDP.
        # For FP8 models: preserve dtypes for LoRA (frozen weights), but set up masters for full finetuning
        if not preserve_fp8:
            transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
            self._transformer = self._transformer.to(dtype=transformer_dtype)
        elif self._config.model.training_mode == "lora":
            # LoRA mode - FP8 weights stay frozen, only LoRA adapters are trained
            logger.info("FP8 model loaded - preserving mixed dtypes for LoRA training (frozen base weights)")
        # Note: FP8 full finetuning setup is done after move_model_to_cpu below

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quanto quantization is not supported in full training mode. Use HQQ quantization instead.")

            logger.warning(f"Quantizing model with quanto precision: {self._config.acceleration.quantization}")
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )
            # Ensure all quantized weights stay on CPU (quanto might use GPU internally)
            self._transformer = self._transformer.to("cpu")
            torch.cuda.empty_cache()

        # HQQ quantization for memory-efficient full model training
        if self._config.acceleration.hqq_quantization is not None:
            logger.info(f"Quantizing model with HQQ {self._config.acceleration.hqq_quantization} for memory-efficient training")
            self._transformer = quantize_model_hqq(
                self._transformer,
                precision=self._config.acceleration.hqq_quantization,
            )

        # CPU offloading - keeps parameters in CPU RAM, transfers to GPU on-demand
        # Uses forward method hijacking to preserve nn.Linear class for PEFT compatibility
        # Order of operations (following AI Toolkit):
        # 1. Move Linear layer weights to CPU (after quantization)
        # 2. Apply memory manager (hijack forward methods)
        # 3. Apply PEFT LoRA (later, in _setup_lora)
        if self._config.acceleration.ramtorch_offload:
            from ltx_trainer.ramtorch_offload import move_model_to_cpu, apply_cpu_offload

            offload_pct = self._config.acceleration.ramtorch_offload_percent

            # Step 1: Move Linear layer weights to CPU with pinned memory
            logger.info("Moving transformer Linear layers to CPU for memory-efficient training...")
            self._transformer = move_model_to_cpu(self._transformer, offload_percent=offload_pct)

            # Step 1.5: For FP8 full finetuning, create bf16 masters AFTER weights are on CPU
            if preserve_fp8 and self._config.model.training_mode == "full":
                logger.info("Setting up FP8 training with bf16 master weights...")
                self._setup_fp8_training()

            # Step 2: Apply memory manager (hijack forward methods to bounce weights CPU<->GPU)
            logger.info(f"Applying CPU offload manager (offload_percent={offload_pct:.0%})...")
            self._transformer = apply_cpu_offload(self._transformer, offload_percent=offload_pct)
        elif preserve_fp8 and self._config.model.training_mode == "full":
            # FP8 full finetuning without CPU offload - still need master weights
            logger.info("Setting up FP8 training with bf16 master weights (no CPU offload)...")
            self._setup_fp8_training()

        # Freeze all models. We later unfreeze the transformer based on training mode.
        # Note: embedding_connectors are already frozen (they come from the frozen text encoder)
        self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

    def _setup_fp8_training(self) -> None:
        """Set up FP8 training with gradient accumulation in separate buffers.

        FP8 doesn't support gradient accumulation directly, so we:
        1. Keep FP8 weights as-is (memory efficient)
        2. Mark FP8 params as requires_grad=False (autograd skips them)
        3. For CPU-offloaded FP8: Ramtorch computes gradients and stores in registry
        4. For GPU-resident FP8: We add hooks here to compute gradients
        5. Custom optimizer step reads from registry, updates FP8 weights
        """
        from ltx_trainer.ramtorch_offload import _FP8_GRADIENT_REGISTRY

        # Collect FP8 params
        self._fp8_params: list[tuple[str, torch.nn.Parameter]] = [
            (name, param) for name, param in self._transformer.named_parameters()
            if param.dtype == torch.float8_e4m3fn
        ]

        # Mark FP8 params as not requiring grad (autograd can't handle FP8)
        for name, param in self._fp8_params:
            param.requires_grad_(False)

        # Find GPU-resident FP8 Linear modules (not marked for offload)
        # These need manual gradient computation via hooks
        gpu_fp8_modules = []
        for name, module in self._transformer.named_modules():
            if module.__class__.__name__ == "Linear":
                if hasattr(module, "weight") and module.weight is not None:
                    if module.weight.dtype == torch.float8_e4m3fn:
                        # Check if NOT offloaded (no _should_offload flag)
                        if not getattr(module, "_should_offload", False):
                            gpu_fp8_modules.append((name, module))

        # For each GPU-resident FP8 module, add hooks for gradient computation
        self._fp8_gpu_hooks = []
        self._fp8_saved_inputs: dict[int, torch.Tensor] = {}

        # Create dedicated CUDA stream for async gradient D2H transfers
        self._fp8_grad_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        # Track pending async gradient transfers: param_id -> (grad_gpu_tensor, event)
        self._fp8_pending_grads: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}

        def make_forward_hook(module_id: int):
            def hook(module, args, output):
                # Save input for backward gradient computation
                self._fp8_saved_inputs[module_id] = args[0].detach()
                return output
            return hook

        def make_backward_hook(module, module_id: int):
            def hook(mod, grad_input, grad_output):
                # Compute weight gradient: grad_weight = grad_output.T @ input
                if module_id not in self._fp8_saved_inputs:
                    return

                saved_input = self._fp8_saved_inputs.pop(module_id)
                grad_out = grad_output[0]

                if grad_out is None:
                    return

                # Compute weight gradient in bf16 (use view instead of flatten for efficiency)
                x_flat = saved_input.view(-1, saved_input.shape[-1]).to(torch.bfloat16)
                grad_flat = grad_out.view(-1, grad_out.shape[-1]).to(torch.bfloat16)
                grad_weight = grad_flat.T @ x_flat

                # Async D2H transfer: queue on separate stream to not block backward compute
                param_id = id(module.weight)
                if self._fp8_grad_stream is not None:
                    # Async transfer to CPU - doesn't block backward pass
                    with torch.cuda.stream(self._fp8_grad_stream):
                        grad_cpu = grad_weight.to("cpu", non_blocking=True)
                        # Store in pending grads dict (will be flushed before optimizer step)
                        if param_id in self._fp8_pending_grads:
                            # Need to sync for accumulation
                            self._fp8_grad_stream.synchronize()
                            self._fp8_pending_grads[param_id] = self._fp8_pending_grads[param_id] + grad_cpu
                        else:
                            self._fp8_pending_grads[param_id] = grad_cpu
                else:
                    # CPU fallback - sync transfer
                    grad_cpu = grad_weight.cpu()
                    if param_id in _FP8_GRADIENT_REGISTRY and _FP8_GRADIENT_REGISTRY[param_id] is not None:
                        _FP8_GRADIENT_REGISTRY[param_id] = _FP8_GRADIENT_REGISTRY[param_id] + grad_cpu
                    else:
                        _FP8_GRADIENT_REGISTRY[param_id] = grad_cpu

            return hook

        for name, module in gpu_fp8_modules:
            module_id = id(module)
            fwd_hook = module.register_forward_hook(make_forward_hook(module_id))
            bwd_hook = module.register_full_backward_hook(make_backward_hook(module, module_id))
            self._fp8_gpu_hooks.extend([fwd_hook, bwd_hook])

        logger.info(f"Set up FP8 training for {len(self._fp8_params)} parameters "
                    f"({len(gpu_fp8_modules)} GPU-resident with hooks)")
        self._has_fp8_training = len(self._fp8_params) > 0

    def _fp8_optimizer_step(self, lr: float) -> None:
        """Custom optimizer step for FP8 parameters.

        Merges pending async grads into registry, then processes all.
        """
        from ltx_trainer.ramtorch_offload import _FP8_GRADIENT_REGISTRY, get_fp8_gradient, clear_fp8_gradients

        if not getattr(self, '_has_fp8_training', False):
            return

        # First, sync and merge pending async grads into the registry
        if hasattr(self, '_fp8_pending_grads') and self._fp8_pending_grads:
            if self._fp8_grad_stream is not None:
                self._fp8_grad_stream.synchronize()

            for param_id, grad_cpu in self._fp8_pending_grads.items():
                if param_id in _FP8_GRADIENT_REGISTRY and _FP8_GRADIENT_REGISTRY[param_id] is not None:
                    _FP8_GRADIENT_REGISTRY[param_id] = _FP8_GRADIENT_REGISTRY[param_id] + grad_cpu
                else:
                    _FP8_GRADIENT_REGISTRY[param_id] = grad_cpu
            self._fp8_pending_grads.clear()

        # Process all grads from registry
        updated = 0
        with torch.no_grad():
            for name, param in self._fp8_params:
                grad = get_fp8_gradient(param)
                if grad is None:
                    continue

                # Update on same device as param
                weight_bf16 = param.data.to(torch.bfloat16)
                grad_bf16 = grad.to(device=weight_bf16.device, dtype=torch.bfloat16)
                weight_bf16.sub_(grad_bf16, alpha=lr)
                param.data = weight_bf16.to(torch.float8_e4m3fn)
                updated += 1

        # Clear gradient registry for next iteration
        clear_fp8_gradients()

        if updated > 0:
            logger.debug(f"FP8 optimizer step: updated {updated} parameters")

    def _sync_fp8_from_masters(self) -> None:
        """Copy bf16 master weights back to FP8 parameters after optimizer step.

        Note: We DON'T actually sync back to FP8 because we're using the bf16 masters
        for forward passes now (see ramtorch _bouncing_forward). The FP8 weights
        are kept for memory efficiency during checkpointing only.
        """
        # With the current design, the bf16 masters are used directly for forward pass,
        # so no sync is needed during training. The FP8 weights are essentially frozen.
        # We only need to convert back to FP8 when saving checkpoints.
        pass

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
            # FP8 params were set to requires_grad=False in _setup_fp8_training
            # They're trained via custom optimizer step, not standard autograd
            if getattr(self, '_has_fp8_training', False):
                for _, param in self._fp8_params:
                    param.requires_grad_(False)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        # Collect trainable params from transformer (excludes FP8 which are handled separately)
        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]

        # Log counts
        trainable_count = sum(p.numel() for p in self._trainable_params)
        fp8_count = sum(p.numel() for _, p in getattr(self, '_fp8_params', []))
        logger.debug(f"Trainable params (autograd): {trainable_count:,}, FP8 params (custom): {fp8_count:,}")

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config)

        # When using CPU offload, the LoRA adapter weights (lora_A, lora_B) are created
        # on CPU. We need to move them to GPU for training.
        if self._config.acceleration.ramtorch_offload:
            lora_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            lora_moved = 0
            for name, module in self._transformer.named_modules():
                # PEFT's LoRA modules are nn.Linear with names containing 'lora_'
                if "lora_" in name and isinstance(module, torch.nn.Linear):
                    module.to(lora_device)
                    lora_moved += 1
            logger.debug(f"Moved {lora_moved} LoRA adapter modules to {lora_device}")

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config."""
        if not self._config.model.load_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            return

        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        state_dict = load_file(checkpoint_path)

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, state_dict)

        logger.info("✅ LoRA checkpoint loaded successfully")

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # For FSDP + LoRA: Ensure uniform dtype across all parameters.
        # PEFT creates LoRA params in FP32 by default, but FSDP wants uniform dtype.
        # Instead of upcasting base model to FP32 (wasteful), downcast LoRA to bf16.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting LoRA parameters to bf16 for uniform dtype")
            for name, param in self._transformer.named_parameters():
                if "lora_" in name and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        # Keep frozen models on CPU for memory efficiency (only loaded to GPU during validation)
        self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to("cpu")
        if self._audio_vae is not None:
            self._audio_vae = self._audio_vae.to("cpu")
        if self._vocoder is not None:
            self._vocoder = self._vocoder.to("cpu")
        torch.cuda.empty_cache()

        # noinspection PyTypeChecker
        # When using CPU offload, disable device placement to prevent Accelerate
        # from moving the memory-managed model to GPU
        device_placement = [not self._config.acceleration.ramtorch_offload]
        self._transformer = self._accelerator.prepare(self._transformer, device_placement=device_placement)

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._training_strategy.get_data_sources()

            self._dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, data_sources=data_sources)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        # Add scheduler initialization
        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = self._config.optimization.scheduler_params or {}

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),  # First restart cycle length
                T_mult=params.pop("T_mult", 1),  # Multiplicative factor for cycle lengths
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            logger.info(f"Local batch size: {local_batch}, global batch size: {global_batch}")

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

        # DeepSpeed-specific validation and logging
        if self._accelerator.distributed_type == DistributedType.DEEPSPEED:
            logger.info("DeepSpeed distributed training enabled")

            # DeepSpeed + quantization is not supported (DeepSpeed calls .to() which quantized models don't support)
            if self._config.acceleration.quantization is not None:
                raise ValueError(
                    "DeepSpeed is not compatible with pre-quantized models (quanto). "
                    "DeepSpeed ZeRO-3 with CPU offload provides similar memory savings through parameter sharding. "
                    "Remove 'quantization' from your config when using DeepSpeed."
                )

            # DeepSpeed ZeRO-3 + LoRA has known bugs (adapter weights stay zero)
            if self._config.model.training_mode == "lora":
                logger.warning(
                    "DeepSpeed ZeRO-3 with LoRA has known compatibility issues where adapter weights may not train properly. "
                    "This configuration is recommended only for full finetuning. "
                    "Consider using FSDP for LoRA training instead."
                )

            # DeepSpeed + HQQ is not compatible (use one or the other)
            if self._config.acceleration.hqq_quantization is not None:
                raise ValueError(
                    "DeepSpeed and HQQ quantization cannot be used together. "
                    "Use either DeepSpeed ZeRO-3 with CPU offload (BF16) or HQQ quantization (INT4/INT8), not both."
                )

        # HQQ-specific validation
        if self._config.acceleration.hqq_quantization is not None:
            if self._config.acceleration.quantization is not None:
                raise ValueError(
                    "Cannot use both HQQ quantization and quanto quantization simultaneously. "
                    "Choose one quantization method."
                )
            if self._config.model.training_mode == "lora":
                raise ValueError(
                    "HQQ quantization is incompatible with LoRA training. "
                    "HQQ replaces nn.Linear with HQQLinear, which PEFT cannot wrap with LoRA adapters. "
                    "Use quanto quantization for LoRA training, or use HQQ only for full finetuning."
                )
            logger.info(f"HQQ {self._config.acceleration.hqq_quantization} quantization enabled for memory-efficient training")

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation
    @torch.no_grad()
    def _sample_videos(self, progress: TrainingProgress) -> list[Path] | None:
        """Run validation by generating videos from validation prompts."""
        use_images = self._config.validation.images is not None
        use_reference_videos = self._config.validation.reference_videos is not None
        generate_audio = self._config.validation.generate_audio
        inference_steps = self._config.validation.inference_steps

        # Free up GPU memory before validation sampling.
        # Zero gradients and empty the cache to reclaim memory.
        self._optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        # Start sampling progress tracking
        sampling_ctx = progress.start_sampling(
            num_prompts=len(self._config.validation.prompts),
            num_steps=inference_steps,
        )

        # Create validation sampler with loaded models and progress tracking
        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        width, height, num_frames = self._config.validation.video_dims

        for prompt_idx, prompt in enumerate(self._config.validation.prompts):
            # Update progress to show current video
            sampling_ctx.start_video(prompt_idx)

            # Load conditioning image if provided
            condition_image = None
            if use_images:
                image_path = self._config.validation.images[prompt_idx]
                image = open_image_as_srgb(image_path)
                # Convert PIL image to tensor [C, H, W] in [0, 1]
                condition_image = F.to_tensor(image)

            # Load reference video if provided (for IC-LoRA)
            reference_video = None
            if use_reference_videos:
                ref_video_path = self._config.validation.reference_videos[prompt_idx]
                # read_video returns [F, C, H, W] in [0, 1]
                reference_video, _ = read_video(ref_video_path, max_frames=num_frames)

            # Get cached embeddings for this prompt if available
            cached_embeddings = (
                self._cached_validation_embeddings[prompt_idx]
                if self._cached_validation_embeddings is not None
                else None
            )

            # Create generation config
            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                generate_audio=generate_audio,
                include_reference_in_output=self._config.validation.include_reference_in_output,
                cached_embeddings=cached_embeddings,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
            )

            # Generate sample
            video, audio = sampler.generate(
                config=gen_config,
                device=self._accelerator.device,
            )

            # Save output (image for single frame, video otherwise)
            if IS_MAIN_PROCESS:
                ext = "png" if num_frames == 1 else "mp4"
                output_path = output_dir / f"step_{self._global_step:06d}_{prompt_idx + 1}.{ext}"
                if num_frames == 1:
                    save_image(video, output_path)
                else:
                    save_video(
                        video_tensor=video,
                        output_path=output_path,
                        fps=self._config.validation.frame_rate,
                        audio=audio,
                        audio_sample_rate=self._vocoder.output_sample_rate if audio is not None else None,
                    )
                video_paths.append(output_path)

        # Clean up progress tasks
        sampling_ctx.cleanup()

        # Clear GPU cache after validation
        torch.cuda.empty_cache()

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return video_paths

    def _verify_weight_changes(self, checkpoint_path: Path | None) -> None:
        """Verify FP8 weights changed during training by comparing with original model."""
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning("Cannot verify weight changes: checkpoint not found")
            return

        original_path = Path(self._config.model.model_path)
        if not original_path.exists():
            logger.warning(f"Cannot verify weight changes: original model not found at {original_path}")
            return

        try:
            from safetensors.torch import load_file

            logger.info("🔍 Verifying weight changes (comparing checkpoint vs original)...")

            # Load original and trained state dicts
            original_sd = load_file(str(original_path))
            trained_sd = load_file(str(checkpoint_path))

            fp8_changes = []
            other_changes = []

            for key in trained_sd:
                if key not in original_sd:
                    continue

                orig_tensor = original_sd[key]
                trained_tensor = trained_sd[key]

                if orig_tensor.shape != trained_tensor.shape:
                    continue

                # Convert to float for comparison
                orig_float = orig_tensor.float()
                trained_float = trained_tensor.float()

                # Calculate difference
                diff = (trained_float - orig_float).abs()
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()

                is_fp8 = orig_tensor.dtype == torch.float8_e4m3fn
                if is_fp8:
                    fp8_changes.append((key, mean_diff, max_diff))
                elif mean_diff > 1e-9:  # Only track if actually changed
                    other_changes.append((key, mean_diff, max_diff))

            # Report FP8 weight changes
            if fp8_changes:
                avg_mean_diff = sum(c[1] for c in fp8_changes) / len(fp8_changes)
                max_mean_diff = max(c[1] for c in fp8_changes)
                changed_count = sum(1 for c in fp8_changes if c[1] > 1e-9)
                logger.info(
                    f"✅ FP8 weight verification: {changed_count}/{len(fp8_changes)} tensors changed "
                    f"(avg diff: {avg_mean_diff:.6f}, max avg diff: {max_mean_diff:.6f})"
                )
            else:
                logger.warning("⚠️ No FP8 weights found to compare")

            # Report other changes
            if other_changes:
                logger.debug(f"Other changed tensors: {len(other_changes)}")

        except Exception as e:
            logger.warning(f"Weight verification failed: {e}")

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP
        is_deepspeed = self._accelerator.distributed_type == DistributedType.DEEPSPEED

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()

        # DeepSpeed ZeRO-3 requires special handling for checkpoint saving
        if is_deepspeed:
            # For DeepSpeed, unwrap=True gathers the sharded parameters
            full_state_dict = self._accelerator.get_state_dict(self._transformer, unwrap=True)
        else:
            full_state_dict = self._accelerator.get_state_dict(self._transformer)

        if not IS_MAIN_PROCESS:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        # For LoRA: extract only adapter weights; for full: use as-is
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            # For FSDP/DeepSpeed, pass full_state_dict since model params aren't directly accessible
            state_dict = get_peft_model_state_dict(
                unwrapped, state_dict=full_state_dict if (is_fsdp or is_deepspeed) else None
            )

            # Remove "base_model.model." prefix added by PEFT
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

            # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

            # Save to disk
            save_file(state_dict, saved_weights_path)
        else:
            # Log dtype distribution for verification (especially important for FP8)
            dtype_counts: dict[str, int] = {}
            for name, tensor in full_state_dict.items():
                dtype_str = str(tensor.dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            dtype_summary = ", ".join(f"{dtype}: {count}" for dtype, count in sorted(dtype_counts.items()))
            logger.debug(f"Checkpoint dtype distribution: {dtype_summary}")

            # Save to disk
            self._accelerator.save(full_state_dict, saved_weights_path)

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")

        # Keep track of checkpoint paths, and cleanup old checkpoints if needed
        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()
        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoints: {old_checkpoint}")
            # Update the list to only contain kept checkpoints
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_config.tags,
            config=self._config.model_dump(),
        )
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        """Log validation samples (videos or images) to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Determine if outputs are images or videos based on file extension
        is_image = sample_paths and sample_paths[0].suffix.lower() in (".png", ".jpg", ".jpeg", ".heic", ".webp")
        media_cls = wandb.Image if is_image else wandb.Video

        samples = [media_cls(str(path), caption=prompt) for path, prompt in zip(sample_paths, prompts, strict=True)]
        self._wandb_run.log({"validation_samples": samples}, step=self._global_step)
