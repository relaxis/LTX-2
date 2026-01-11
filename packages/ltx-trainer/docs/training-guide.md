# Training Guide

This guide covers how to run training jobs, from basic single-GPU training to advanced distributed setups and automatic
model uploads.

## ⚡ Basic Training (Single GPU)

After preprocessing your dataset and preparing a configuration file, you can start training using the trainer script:

```bash
uv run python scripts/train.py configs/ltx2_av_lora.yaml
```

The trainer will:

1. **Load your configuration** and validate all parameters
2. **Initialize models** and apply optimizations
3. **Run the training loop** with progress tracking
4. **Generate validation videos** (if configured)
5. **Save the trained weights** in your output directory

### Output Files

**For LoRA training:**

- `lora_weights.safetensors` - Main LoRA weights file
- `training_config.yaml` - Copy of training configuration
- `validation_samples/` - Generated validation videos (if enabled)

**For full model fine-tuning:**

- `model_weights.safetensors` - Full model weights
- `training_config.yaml` - Copy of training configuration
- `validation_samples/` - Generated validation videos (if enabled)

## 🖥️ Distributed / Multi-GPU Training

We use Hugging Face 🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) for multi-GPU DDP and FSDP.

### Configure Accelerate

Run the interactive wizard once to set up your environment (DDP / FSDP, GPU count, etc.):

```bash
uv run accelerate config
```

This stores your preferences in `~/.cache/huggingface/accelerate/default_config.yaml`.

### Use the Provided Accelerate Configs (Recommended)

We include ready-to-use Accelerate config files in `configs/accelerate/`:

- [ddp.yaml](../configs/accelerate/ddp.yaml) — Standard DDP
- [ddp_compile.yaml](../configs/accelerate/ddp_compile.yaml) — DDP with `torch.compile` (Inductor)
- [fsdp.yaml](../configs/accelerate/fsdp.yaml) — Standard FSDP (auto-wraps `BasicAVTransformerBlock`)
- [fsdp_compile.yaml](../configs/accelerate/fsdp_compile.yaml) — FSDP with `torch.compile` (Inductor)

Launch with a specific config using `--config_file`:

```bash
# DDP (2 GPUs shown as example)
CUDA_VISIBLE_DEVICES=0,1 \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml \
  scripts/train.py configs/ltx2_av_lora.yaml

# DDP + torch.compile
CUDA_VISIBLE_DEVICES=0,1 \
uv run accelerate launch --config_file configs/accelerate/ddp_compile.yaml \
  scripts/train.py configs/ltx2_av_lora.yaml

# FSDP (4 GPUs shown as example)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run accelerate launch --config_file configs/accelerate/fsdp.yaml \
  scripts/train.py configs/ltx2_av_lora.yaml

# FSDP + torch.compile
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run accelerate launch --config_file configs/accelerate/fsdp_compile.yaml \
  scripts/train.py configs/ltx2_av_lora.yaml
```

**Notes:**

- The number of processes is taken from the Accelerate config (`num_processes`). Override with `--num_processes X` or
  restrict GPUs with `CUDA_VISIBLE_DEVICES`.
- The compile variants enable `torch.compile` with the Inductor backend via Accelerate's `dynamo_config`.
- FSDP configs auto-wrap the transformer blocks (`fsdp_transformer_layer_cls_to_wrap: BasicAVTransformerBlock`).

### Launch with Your Default Accelerate Config

If you prefer to use your default Accelerate profile:

```bash
# Use settings from your default accelerate config
uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml

# Override number of processes on the fly (e.g., 2 GPUs)
uv run accelerate launch --num_processes 2 scripts/train.py configs/ltx2_av_lora.yaml

# Select specific GPUs
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml
```

> [!TIP]
> You can disable the in-terminal progress bars with `--disable-progress-bars` flag in the trainer CLI if desired.

### Benefits of Distributed Training

- **Faster training**: Distribute workload across multiple GPUs
- **Larger effective batch sizes**: Combine gradients from multiple GPUs
- **Memory efficiency**: Each GPU handles a portion of the batch

> [!NOTE]
> Distributed training requires that all GPUs have sufficient memory for the model and batch size. The effective batch
> size becomes `batch_size × num_processes`.

## 🤗 Pushing Models to Hugging Face Hub

You can automatically push your trained models to the Hugging Face Hub by adding the following to your configuration:

```yaml
hub:
  push_to_hub: true
  hub_model_id: "your-username/your-model-name"
```

### Prerequisites

Before pushing, make sure you:

1. **Have a Hugging Face account** - Sign up at [huggingface.co](https://huggingface.co)
2. **Are logged in** via `huggingface-cli login` or have set the `HUGGING_FACE_HUB_TOKEN` environment variable
3. **Have write access** to the specified repository (it will be created if it doesn't exist)

### Login Options

**Option 1: Interactive login**

```bash
uv run huggingface-cli login
```

**Option 2: Environment variable**

```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### What Gets Uploaded

The trainer will automatically:

- **Create a model card** with training details and sample outputs
- **Upload model weights**
- **Push sample videos as GIFs** in the model card
- **Include training configuration and prompts**

## 📊 Weights & Biases Logging

Enable experiment tracking with W&B by adding to your configuration:

```yaml
wandb:
  enabled: true
  project: "ltx-2-trainer"
  entity: null  # Your W&B username or team
  tags: [ "ltx2", "lora" ]
  log_validation_videos: true
```

This will log:

- Training loss and learning rate
- Validation videos
- Model configuration
- Training progress

## 🚀 Next Steps

After training completes:

- **Run inference with your trained LoRA** - The [`ltx-pipelines`](../../ltx-pipelines/) package provides
  production-ready inference
  pipelines that support loading custom LoRAs. Available pipelines include text-to-video, image-to-video,
  IC-LoRA video-to-video, and more. See the [`ltx-pipelines`](../../ltx-pipelines/) package for usage details.
- **Test your model** with validation prompts
- **Iterate and improve** based on validation results
- **Share your results** by pushing to Hugging Face Hub

## 💾 CPU Offloading for Memory-Constrained Training

When training with high LoRA ranks (64+) on limited VRAM (24-32GB), CPU offloading enables training that would
otherwise not fit in GPU memory. This section covers how to configure and use the RamTorch CPU offloading feature.

### When to Use CPU Offloading

Use CPU offloading when:
- Training with LoRA rank 64+ on GPUs with 24-32GB VRAM
- You're hitting OOM errors during training even with gradient checkpointing enabled
- You want to train larger LoRA ranks without quantization overhead

### Configuration

Enable CPU offloading in your training config:

```yaml
acceleration:
  mixed_precision_mode: "bf16"
  # CPU offloading settings
  ramtorch_offload: true
  ramtorch_offload_percent: 0.65  # 65% of linear layers offloaded to CPU
  # Disable quantization when using CPU offload (avoids dequantization overhead)
  quantization: null
```

**Key parameters:**

| Parameter                 | Description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `ramtorch_offload`        | Enable RamTorch CPU offloading (moves Linear layers to CPU RAM)                               |
| `ramtorch_offload_percent`| Percentage of Linear layers to offload (0.0-1.0). Lower = faster but more VRAM usage          |

### Finding the Right Offload Percentage

The optimal offload percentage depends on your GPU VRAM and LoRA rank:

| GPU VRAM | LoRA Rank | Recommended `ramtorch_offload_percent` | Expected Speed |
|----------|-----------|---------------------------------------|----------------|
| 24GB     | 64        | 0.75-0.85                             | ~1.5-2x slower |
| 32GB     | 64        | 0.60-0.70                             | ~1.3-1.5x slower |
| 32GB     | 128       | 0.75-0.85                             | ~1.5-2x slower |
| 48GB     | 64        | 0.40-0.50                             | ~1.2x slower   |

> [!TIP]
> Start with a higher offload percentage (more offloading) and decrease it if training is stable.
> The goal is to use as much GPU memory as possible without OOM errors.

### Running Training with CPU Offload

For best results with CPU offloading, use the `--disable-progress-bars` flag to get proper step-by-step logging:

```bash
# Set memory allocator optimizations
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True'

# Run training with progress bars disabled for better logging
uv run python scripts/train.py configs/your_config.yaml --disable-progress-bars
```

### Memory Allocator Settings

The `PYTORCH_CUDA_ALLOC_CONF` environment variable helps reduce memory fragmentation during CPU<->GPU transfers:

| Setting                         | Description                                           |
|---------------------------------|-------------------------------------------------------|
| `max_split_size_mb:128`         | Limits allocation splitting to reduce fragmentation   |
| `garbage_collection_threshold:0.6` | Triggers GC earlier to free unused memory           |
| `expandable_segments:True`      | Allows memory segments to grow, reducing fragmentation|

### Performance Expectations

With 65% CPU offloading on a 32GB GPU:
- **Setup time**: ~20-30 seconds for model loading and offload setup
- **Training speed**: ~16-17 seconds per gradient accumulation step (vs ~12s native)
- **VRAM usage**: ~16-17GB after setup, ~28-30GB during training
- **RAM usage**: ~30-40GB for offloaded weights

### Validation with CPU Offload

Validation inference works with CPU offloading but will be significantly slower due to the weight transfers.
For efficient training:

1. **Disable automatic validation during training** by setting a very high interval:
   ```yaml
   validation:
     interval: 999999  # Effectively disable validation during training
   ```

2. **Run validation manually** after training checkpoints by using the inference script:
   ```bash
   uv run python scripts/inference.py \
     --model_path /path/to/checkpoint/lora_weights.safetensors \
     --prompt "Your validation prompt"
   ```

3. **For occasional validation checks**, set a moderate interval (e.g., every 500-1000 steps):
   ```yaml
   validation:
     interval: 500
     skip_initial_validation: true  # Skip step 0 validation
   ```

> [!NOTE]
> Each validation run with CPU offload will take several minutes due to the denoising loop
> running 30 inference steps with weight transfers.

### Troubleshooting CPU Offload

**OOM during backward pass:**
- Increase `ramtorch_offload_percent` to offload more layers
- Ensure the allocator settings are configured correctly

**Training too slow:**
- Decrease `ramtorch_offload_percent` to keep more layers on GPU
- Verify GPU utilization is high (90%+) during training

**No training output in logs:**
- Use `--disable-progress-bars` flag - the rich progress bar doesn't log to files properly

**Validation OOM:**
- Validation runs the full 30-step denoising loop which requires more memory
- Consider disabling validation (`interval: 999999`) and validating manually after training

## 🧊 FP8 Full Finetuning

For memory-efficient full finetuning of the 19B parameter LTX-2 model on a single GPU, you can use a pre-quantized FP8 model file combined with CPU offloading.

### Overview

FP8 (float8_e4m3fn) quantization reduces model memory by ~50% compared to bf16 while maintaining better precision than INT8/INT4. Combined with CPU offloading, this enables full finetuning on GPUs with 24-32GB VRAM.

**Key features:**
- Uses pre-quantized FP8 model files (no runtime quantization overhead)
- Custom gradient computation for FP8 parameters (PyTorch doesn't support FP8 autograd)
- Simple SGD optimizer for FP8 params (memory-efficient, no momentum states)
- All 19B parameters are trainable

### Configuration

Create a configuration file for FP8 full finetuning:

```yaml
model:
  # Pre-quantized FP8 model file
  model_path: "/path/to/ltx-2-19b-dev-fp8.safetensors"
  text_encoder_path: "/path/to/gemma"
  training_mode: "full"

# No LoRA for full finetuning
lora: null

optimization:
  learning_rate: 1e-6  # Lower LR for full finetuning
  steps: 5000
  batch_size: 1
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  # Use regular AdamW (adamw8bit not compatible with CPU offload)
  optimizer_type: "adamw"
  scheduler_type: "cosine"
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: null  # Model is already FP8, no runtime quantization
  # CPU offload for memory-efficient training
  ramtorch_offload: true
  ramtorch_offload_percent: 0.70  # 70% of layers on CPU
  # Enable FP8 model loading (preserves FP8 weights)
  load_fp8_model: true
```

### Key Settings Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `load_fp8_model` | `true` | Preserves FP8 weights from checkpoint (doesn't convert to bf16) |
| `ramtorch_offload` | `true` | Enables CPU offloading for linear layers |
| `ramtorch_offload_percent` | `0.65-0.75` | % of layers to offload; higher = less VRAM, slower |
| `optimizer_type` | `"adamw"` | Regular AdamW (8-bit variants incompatible with CPU params) |
| `training_mode` | `"full"` | Full model finetuning (all params trainable) |

### How FP8 Training Works

Since PyTorch doesn't support gradient computation for FP8 tensors directly, the trainer implements custom gradient handling:

1. **FP8 parameters** are set to `requires_grad=False` (autograd skips them)
2. **CPU-offloaded FP8 layers**: Gradients computed in bf16 during the ramtorch backward pass
3. **GPU-resident FP8 layers**: Gradients computed via hooks (saved activations for backward)
4. **Custom optimizer step**: FP8 → bf16 → gradient update → FP8

All 19B parameters receive gradients and are updated each step.

### Memory Requirements

| GPU VRAM | Offload % | VRAM Usage | RAM Usage | Speed |
|----------|-----------|------------|-----------|-------|
| 24GB | 85-90% | ~20GB | ~80GB | ~15-20s/step |
| 32GB | 70-75% | ~27GB | ~60GB | ~10-12s/step |
| 48GB | 50-60% | ~35GB | ~40GB | ~8-10s/step |

> [!IMPORTANT]
> FP8 full finetuning requires significant system RAM (60-100GB) for offloaded weights
> and gradient storage. Ensure adequate swap space as backup.

### Running FP8 Training

```bash
# Set memory allocator optimizations
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True'

# Run training
uv run python scripts/train.py configs/your_fp8_config.yaml

# Or with logging to file (progress bars don't capture well)
nohup uv run python scripts/train.py configs/your_fp8_config.yaml > train.log 2>&1 &
```

### Finding the Right Offload Percentage

Start with a higher offload percentage and decrease until you find the balance:

1. **Start at 80-85%**: Safe starting point, will be slower
2. **Monitor VRAM**: Check `nvidia-smi` during training
3. **Decrease by 5%**: If VRAM is stable with headroom (5+ GB free)
4. **Stop when**: VRAM peaks within 2-3GB of your GPU max

The optimal percentage uses most of your VRAM without OOM errors during backward passes.

### Troubleshooting FP8 Training

**OOM during backward pass:**
- Increase `ramtorch_offload_percent`
- Reduce `batch_size` to 1
- Enable `enable_gradient_checkpointing: true`

**Training too slow:**
- Decrease `ramtorch_offload_percent` to keep more on GPU
- Check GPU utilization with `nvidia-smi` (should be 60-80%)

**"FP8 optimizer step: updated 0 parameters":**
- Ensure `load_fp8_model: true` is set
- Verify model file has FP8 weights (`.safetensors` should have mixed dtypes)

**Gradients all zero / loss not decreasing:**
- Check that all FP8 parameters report updates in logs
- FP8 models may need lower learning rates (1e-6 to 1e-7)

### Verifying Checkpoints

When saving checkpoints, the trainer logs dtype distribution:

```
DEBUG: Checkpoint dtype distribution: torch.bfloat16: 194, torch.float32: 50, torch.float8_e4m3fn: 1176
```

This confirms:
- **1176 FP8 tensors**: The trained FP8 weights
- **194 bf16 tensors**: Non-FP8 layers (embeddings, some normalization)
- **50 f32 tensors**: Other parameters

If you don't see FP8 tensors in the distribution, the checkpoint may not include trained FP8 weights.

## 💡 Tips for Successful Training

- **Start small**: Begin with a small dataset and a few hundred steps to verify everything works
- **Monitor validation**: Keep an eye on validation samples to catch overfitting
- **Adjust learning rate**: Lower learning rates often produce better results
- **Use gradient checkpointing**: Essential for training with limited GPU memory
- **Save checkpoints**: Regular checkpoints help recover from interruptions

## Need Help?

If you encounter issues during training, see the [Troubleshooting Guide](troubleshooting.md).

Join our [Discord community](https://discord.gg/ltxplatform) for real-time help!
