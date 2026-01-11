"""
CPU offloading for LTX-2 training via forward method hijacking.

This module implements AI Toolkit's approach to CPU offloading:
- Model parameters should already be on CPU (moved there after quantization)
- Hijack forward() method to stream weights to GPU on-demand via async CUDA streams
- Preserve the nn.Linear module class so PEFT can still inject LoRA adapters

Key insight: Unlike RamTorch's replace_linear_with_ramtorch() which replaces nn.Linear
with CPUBouncingLinear (breaking PEFT), this approach only hijacks forward() while
keeping the module type as nn.Linear.

Heavily inspired by AI Toolkit's memory management:
https://github.com/ostris/ai-toolkit/blob/main/toolkit/memory_management/manager_modules.py
Original RamTorch concept by Lodestone-Rock:
https://github.com/lodestone-rock/RamTorch
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import has_torch_function_unary

from ltx_trainer import logger

if TYPE_CHECKING:
    from typing import Dict, Any

# Per-device global state for CUDA streams and buffers
_DEVICE_STATE: "Dict[torch.device, Dict[str, Any]]" = {}


def _get_device_state(device: torch.device) -> "Dict[str, Any]":
    """Get or initialize per-device CUDA state for async transfers."""
    if isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        if device not in _DEVICE_STATE:
            _DEVICE_STATE[device] = {}
        return _DEVICE_STATE[device]

    if device not in _DEVICE_STATE:
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                # CUDA streams for async H2D/D2H transfers
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                # Events for synchronization
                "transfer_forward_finished_event": torch.cuda.Event(),
                "compute_forward_start_event": torch.cuda.Event(),
                "transfer_backward_finished_event": torch.cuda.Event(),
                "transfer_weight_backward_finished_event": torch.cuda.Event(),
                "compute_backward_start_event": torch.cuda.Event(),
                "compute_backward_finished_event": torch.cuda.Event(),
                # Ping-pong buffers for overlapping transfers
                "w_buffers": [None, None],
                "b_buffers": [None, None],
                "w_bwd_buffers": [None, None],
                # Device-side staging for gradients
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                # Clocks for buffer alternation
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE[device]


def _is_ao_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    """Detect torchao wrapper tensors."""
    if t is None:
        return False
    try:
        if has_torch_function_unary(t):
            return t.__class__.__module__.startswith("torchao.")
    except Exception:
        pass
    for attr in ("_scale", "_scales", "_zero_point", "_zp", "_block_size", "_group_size", "_pack_dim"):
        if hasattr(t, attr):
            return True
    return False


# Global registry for FP8 gradients (since FP8 can't accumulate gradients directly)
_FP8_GRADIENT_REGISTRY: dict[int, torch.Tensor] = {}


def get_fp8_gradient(param: torch.Tensor) -> torch.Tensor | None:
    """Get accumulated gradient for an FP8 parameter."""
    return _FP8_GRADIENT_REGISTRY.get(id(param))


def clear_fp8_gradients():
    """Clear all FP8 gradients (call after optimizer step)."""
    _FP8_GRADIENT_REGISTRY.clear()


def _is_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    """Check if tensor is quantized (torch quantized, torchao, FP8, or non-floating-point)."""
    if t is None:
        return False
    try:
        if torch.is_quantized(t):
            return True
    except Exception:
        pass
    if _is_ao_quantized_tensor(t):
        return True
    # FP8 tensors - they're floating point but can't be pinned the normal way
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    # packed/int formats
    return not t.dtype.is_floating_point


def _ensure_cpu_pinned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Ensure tensor is on CPU with pinned memory (if possible)."""
    if t is None:
        return None
    if t.device.type != "cpu":
        try:
            t = t.to("cpu", copy=True)
        except Exception:
            t = t.to("cpu")
    # Don't attempt to pin quantized tensors; many backends don't support it
    if _is_quantized_tensor(t):
        return t
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t


def _materialize_weight_for_compute(cpu_w: torch.Tensor, device: torch.device, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Move weight from CPU to GPU, handling quantized tensors.

    For quantized tensors: move to GPU, dequantize, cast to target dtype.
    For float tensors: move to GPU, cast to target dtype.
    """
    if _is_quantized_tensor(cpu_w):
        # Move quantized wrapper to GPU -> dequantize on GPU -> cast on GPU
        w_q_gpu = cpu_w.to(device, non_blocking=True)
        try:
            w_fp_gpu = w_q_gpu.dequantize()
        except Exception:
            w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
        if w_fp_gpu.dtype != target_dtype:
            w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
        return w_fp_gpu

    # Float path: move and cast
    w_gpu = cpu_w.to(device, non_blocking=True)
    if w_gpu.dtype != target_dtype and target_dtype in (torch.bfloat16, torch.float16, torch.float32):
        w_gpu = w_gpu.to(target_dtype, non_blocking=True)
    return w_gpu


class _BouncingLinearFn(torch.autograd.Function):
    """
    Autograd function that streams weights from CPU to GPU on-demand.

    Supports both quantized and non-quantized weights.
    Uses async CUDA streams to overlap weight transfers with computation.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight_cpu: torch.Tensor, bias_cpu: Optional[torch.Tensor], device: torch.device):
        # Match activation dtype for consistent math
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # CPU fallback path
        if device.type != "cuda":
            x_cpu = x.to("cpu", dtype=target_dtype)
            w_cpu = _materialize_weight_for_compute(weight_cpu, torch.device("cpu"), target_dtype)
            b_cpu = None
            if bias_cpu is not None:
                b_cpu = bias_cpu.to("cpu")
                if b_cpu.dtype != target_dtype:
                    b_cpu = b_cpu.to(target_dtype)
            out = F.linear(x_cpu, w_cpu, b_cpu)
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.device = torch.device("cpu")
            ctx.target_dtype = target_dtype
            return out.to(x.device, dtype=x.dtype)

        # GPU path with async transfers
        state = _get_device_state(device)
        ts = state["transfer_stream"]
        w_bufs, b_bufs = state["w_buffers"], state["b_buffers"]
        ev_tx_f = state["transfer_forward_finished_event"]
        ev_cu_s = state["compute_forward_start_event"]
        idx = state["forward_clk"]

        # Async H2D transfer on dedicated stream
        with torch.cuda.stream(ts):
            ts.wait_event(ev_cu_s)
            w_bufs[idx] = _materialize_weight_for_compute(weight_cpu, device, target_dtype)
            if bias_cpu is not None:
                b_dev = bias_cpu.to(device, non_blocking=True)
                if b_dev.dtype != target_dtype:
                    b_dev = b_dev.to(target_dtype, non_blocking=True)
                b_bufs[idx] = b_dev
            else:
                b_bufs[idx] = None
            state["forward_clk"] ^= 1
            ev_tx_f.record()

        # Wait for transfer and compute
        torch.cuda.current_stream().wait_event(ev_tx_f)
        ev_cu_s.record()
        out = F.linear(x, w_bufs[idx], b_bufs[idx])

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.target_dtype = target_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        target_dtype = getattr(ctx, "target_dtype", grad_out.dtype)

        # CPU fallback path
        if device.type != "cuda":
            go_cpu = grad_out.to("cpu", dtype=target_dtype)
            x_cpu = x.to("cpu", dtype=target_dtype)
            w_mat = _materialize_weight_for_compute(weight_cpu, torch.device("cpu"), target_dtype)
            grad_input = go_cpu @ w_mat
            grad_weight = (
                go_cpu.flatten(0, -2).T @ x_cpu.flatten(0, -2)
                if getattr(weight_cpu, "requires_grad", False) and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go_cpu.sum(dim=tuple(range(go_cpu.ndim - 1)))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return grad_input.to(grad_out.device), grad_weight, grad_bias, None

        # GPU path
        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_bwd_buffers = state["w_bwd_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]
        ev_tx_b = state["transfer_backward_finished_event"]
        ev_tx_w_bwd_done = state["transfer_weight_backward_finished_event"]
        ev_cu_b_start = state["compute_backward_start_event"]
        ev_cu_b_finish = state["compute_backward_finished_event"]
        idx = state["backward_clk"]

        # Transfer weights for backward (with dequantization if needed)
        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(ev_cu_b_start)
            w_bwd_buffers[idx] = _materialize_weight_for_compute(weight_cpu, device, target_dtype)
            state["backward_clk"] ^= 1
            ev_tx_b.record()

        torch.cuda.current_stream().wait_event(ev_tx_b)
        ev_cu_b_start.record()

        # Compute grad wrt input
        grad_input = grad_out.to(dtype=target_dtype) @ w_bwd_buffers[idx]

        # Wait for previous grad transfer
        torch.cuda.current_stream().wait_event(ev_tx_w_bwd_done)

        # Compute weight/bias gradients
        grad_weight = None
        grad_bias = None
        is_fp8_weight = weight_cpu.dtype == torch.float8_e4m3fn
        needs_weight_grad = getattr(weight_cpu, "requires_grad", False) or is_fp8_weight

        if needs_weight_grad:
            # Ensure same dtype for matmul (x may be different dtype from grad_out)
            x_for_grad = x.to(dtype=target_dtype) if x.dtype != target_dtype else x
            grad_out_for_weight = grad_out.to(dtype=target_dtype) if grad_out.dtype != target_dtype else grad_out
            w_grad_buffers[idx] = grad_out_for_weight.flatten(0, -2).T @ x_for_grad.flatten(0, -2)
        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
            reduce_dims = tuple(range(grad_out.ndim - 1))
            b_grad_buffers[idx] = grad_out.sum(dim=reduce_dims)

        ev_cu_b_finish.record()

        # Async D2H transfer for gradients
        # Determine target device for gradients (match the original weight's device)
        weight_grad_device = weight_cpu.device

        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(ev_cu_b_finish)
            if is_fp8_weight:
                # FP8 weights: store gradient in registry (can't use .grad directly)
                grad_on_cpu = w_grad_buffers[idx].to("cpu", non_blocking=True)
                param_id = id(weight_cpu)
                if param_id in _FP8_GRADIENT_REGISTRY and _FP8_GRADIENT_REGISTRY[param_id] is not None:
                    # Accumulate gradients
                    _FP8_GRADIENT_REGISTRY[param_id] = _FP8_GRADIENT_REGISTRY[param_id] + grad_on_cpu
                else:
                    _FP8_GRADIENT_REGISTRY[param_id] = grad_on_cpu
                # Don't return grad for autograd (FP8 can't handle it)
            elif getattr(weight_cpu, "requires_grad", False):
                # Regular weights: return gradient for autograd
                grad_weight = w_grad_buffers[idx].to(weight_grad_device, non_blocking=True)
            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = b_grad_buffers[idx].to(weight_cpu.device, non_blocking=True)
            state["transfer_weight_backward_finished_event"].record()

        return grad_input.to(dtype=grad_out.dtype), grad_weight, grad_bias, None


# Module types to offload
LINEAR_MODULES = ["Linear", "LoRACompatibleLinear", "QLinear"]

# Module types to NOT offload (keep on GPU for efficiency)
UNMANAGED_MODULES = [
    "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "Embedding", "EmbeddingBag", "RNNBase", "LSTM", "GRU", "RNN", "Conv3d"
]
UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


class CPUOffloadManager:
    """
    Manages CPU offloading for a model by hijacking forward methods.

    IMPORTANT: The model should already be on CPU before calling attach().
    This is typically done by moving the model to CPU after quantization.
    """

    def __init__(self, module: nn.Module, device: torch.device):
        self.module = module
        self.device = device
        self.unmanaged_modules: list[nn.Module] = []
        self._managed_count = 0

    @classmethod
    def attach(
        cls,
        module: nn.Module,
        device: torch.device,
        offload_percent: float = 1.0,
    ) -> "CPUOffloadManager":
        """
        Attach CPU offloading to all Linear layers in a module.

        The model should already be on CPU. This method hijacks forward()
        to stream weights to GPU on-demand.

        Args:
            module: The model (already on CPU) to attach offloading to
            device: The compute device (cuda)
            offload_percent: Fraction of layers to offload (0.0-1.0)

        Returns:
            The CPUOffloadManager instance
        """
        if hasattr(module, "_cpu_offload_manager"):
            return module._cpu_offload_manager

        manager = cls(module, device)
        module._cpu_offload_manager = manager

        # Override the .to() method to handle memory management
        module._mm_to = module.to
        module.to = manager._memory_managed_to

        import random
        modules_processed = []

        # Find and manage all eligible layers
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                class_name = child_module.__class__.__name__

                if class_name in LINEAR_MODULES and child_module not in modules_processed:
                    # Skip if already managed
                    if hasattr(child_module, "_is_cpu_offloaded"):
                        modules_processed.append(child_module)
                        continue

                    # Check if this layer was marked for offloading by move_model_to_cpu()
                    # If not marked and partial offload is enabled, skip it
                    should_offload = getattr(child_module, "_should_offload", offload_percent >= 1.0)
                    if not should_offload:
                        manager.unmanaged_modules.append(child_module)
                        modules_processed.append(child_module)
                        continue

                    # Verify weights are on CPU (they should be!)
                    if hasattr(child_module, "weight") and child_module.weight is not None:
                        if child_module.weight.device.type != "cpu":
                            logger.warning(
                                f"Module {name}.{child_name} weight is on {child_module.weight.device}, "
                                "expected CPU. Moving to CPU."
                            )
                            with torch.no_grad():
                                child_module.weight.data = _ensure_cpu_pinned(child_module.weight.data)
                                if child_module.bias is not None:
                                    child_module.bias.data = _ensure_cpu_pinned(child_module.bias.data)

                    # Store original forward and hijack
                    original_forward = child_module.forward

                    def make_bouncing_forward(mod, orig_fwd, mgr):
                        def _bouncing_forward(x, *args, **kwargs):
                            if args or kwargs:
                                return orig_fwd(x, *args, **kwargs)
                            weight_cpu = mod.weight
                            bias_cpu = getattr(mod, "bias", None)
                            return _BouncingLinearFn.apply(x, weight_cpu, bias_cpu, mgr.device)
                        return _bouncing_forward

                    child_module.forward = make_bouncing_forward(child_module, original_forward, manager)
                    child_module._is_cpu_offloaded = True
                    child_module._cpu_offload_device = device
                    manager._managed_count += 1
                    modules_processed.append(child_module)

                elif class_name in UNMANAGED_MODULES or any(inc in class_name for inc in UNMANAGED_MODULES_INCLUDES):
                    manager.unmanaged_modules.append(child_module)

        logger.info(f"CPU offload attached to {manager._managed_count} Linear layers")
        logger.info(f"Unmanaged modules (kept on device): {len(manager.unmanaged_modules)}")

        # Move unmanaged modules to GPU - they need to be on device for forward pass
        for unmanaged in manager.unmanaged_modules:
            try:
                unmanaged.to(device)
            except Exception:
                pass  # Some modules might not have parameters

        return manager

    def _memory_managed_to(self, *args, **kwargs):
        """Custom .to() method that handles memory-managed modules."""
        # Move unmanaged modules normally
        for module in self.unmanaged_modules:
            if isinstance(module, nn.Parameter):
                module.data = module.data.to(*args, **kwargs)
            else:
                module.to(*args, **kwargs)

        # Handle dtype changes for managed modules
        dtype = kwargs.get("dtype")
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break

        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module


def move_model_to_cpu(model: torch.nn.Module, offload_percent: float = 1.0) -> torch.nn.Module:
    """
    Move a model's Linear layer weights to CPU with pinned memory.

    This should be called AFTER quantization but BEFORE applying CPU offload.
    Non-Linear layers (norms, embeddings) stay on GPU.

    Args:
        model: The model to process
        offload_percent: Fraction of layers to move to CPU (0.0-1.0)

    Returns:
        The model with specified percentage of Linear weights on CPU
    """
    import random
    random.seed(42)  # Deterministic for reproducibility

    linear_count = 0
    moved_count = 0

    # First pass: identify all Linear layers and determine which to offload
    all_linear_modules = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name in LINEAR_MODULES:
            all_linear_modules.append((name, module))

    linear_count = len(all_linear_modules)

    # Randomly select which layers to offload based on percentage
    if offload_percent < 1.0:
        num_to_offload = int(linear_count * offload_percent)
        offload_indices = set(random.sample(range(linear_count), num_to_offload))
    else:
        offload_indices = set(range(linear_count))

    # Second pass: move selected layers to CPU with pinned memory
    for idx, (name, module) in enumerate(all_linear_modules):
        if idx in offload_indices:
            if hasattr(module, "weight") and module.weight is not None:
                with torch.no_grad():
                    weight = module.weight.data

                    # For quantized tensors (including FP8), just ensure they're on CPU (no pinning)
                    if _is_quantized_tensor(weight):
                        if weight.device.type != "cpu":
                            module.weight.data = weight.to("cpu")
                    else:
                        # For float tensors: ensure on CPU with pinned memory
                        # Only copy if necessary (mmap'd or on GPU)
                        if weight.device.type != "cpu":
                            new_weight = weight.to("cpu")
                        elif not weight.is_pinned():
                            # Already on CPU but need to force out of mmap to pinned memory
                            new_weight = weight.clone().pin_memory()
                        else:
                            new_weight = weight  # Already pinned, no action needed

                        if new_weight is not weight:
                            module.weight.data = new_weight

                    if module.bias is not None:
                        bias = module.bias.data
                        if bias.device.type != "cpu":
                            module.bias.data = bias.to("cpu")
                        elif not _is_quantized_tensor(bias) and not bias.is_pinned():
                            try:
                                module.bias.data = bias.clone().pin_memory()
                            except RuntimeError:
                                pass

                module._should_offload = True  # Mark for forward hijacking
                moved_count += 1

    logger.info(f"Moved {moved_count}/{linear_count} Linear layers to CPU ({offload_percent:.0%} offload)")
    torch.cuda.empty_cache()
    return model


def apply_cpu_offload(model: torch.nn.Module, device: torch.device = None, offload_percent: float = 1.0) -> torch.nn.Module:
    """
    Apply CPU offloading to a model.

    IMPORTANT: Call move_model_to_cpu() first to mark layers for offloading.
    Only layers marked with _should_offload=True will have forward hijacking applied.

    This hijacks forward() methods while preserving nn.Linear module classes,
    allowing PEFT to still inject LoRA adapters.

    Args:
        model: The model (with weights on CPU) to apply offloading to
        device: The compute device (defaults to cuda:0)
        offload_percent: Only used for logging; actual selection done in move_model_to_cpu

    Returns:
        The same model with CPU offloading applied
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    linear_count = sum(1 for m in model.modules() if m.__class__.__name__ in LINEAR_MODULES)
    offload_count = sum(1 for m in model.modules() if getattr(m, '_should_offload', False))
    logger.info(f"Applying CPU offload to model with {linear_count} Linear layers ({offload_count} to offload)...")

    CPUOffloadManager.attach(model, device, offload_percent=offload_percent)

    logger.info("CPU offload applied (forward methods hijacked, module classes preserved)")
    return model
