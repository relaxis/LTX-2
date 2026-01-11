"""
HQQ (Half-Quadratic Quantization) for memory-efficient full model training.

HQQ provides fast, calibration-free quantization that's compatible with training.
Unlike quanto which freezes weights, HQQ allows gradients to flow through
dequantized weights during training.

Reference: https://github.com/mobiusml/hqq
"""

from typing import Literal

import torch

from ltx_trainer import logger

HQQQuantizationOptions = Literal["int4", "int8"]

# Layers to exclude from quantization (critical for model quality)
EXCLUDE_PATTERNS = [
    "proj_in",
    "proj_out",
    "patchify_proj",
    "time_embed",
    "caption_projection",
    "adaln",
    "rope",
    "norm",
]


def quantize_model_hqq(
    model: torch.nn.Module,
    precision: HQQQuantizationOptions,
) -> torch.nn.Module:
    """
    Quantize a model using HQQ for memory-efficient training.

    Unlike quanto, HQQ:
    - Does not require calibration data
    - Is faster to quantize (minutes vs hours)
    - Supports training through dequantized weights

    Args:
        model: The model to quantize.
        precision: The precision level ("int4" or "int8").

    Returns:
        The quantized model.
    """
    try:
        from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
    except ImportError as e:
        raise ImportError(
            "HQQ is required for HQQ quantization. Install it with: pip install hqq"
        ) from e

    nbits = 4 if precision == "int4" else 8
    group_size = 64 if precision == "int4" else 128  # Smaller groups for int4

    quant_config = BaseQuantizeConfig(
        nbits=nbits,
        group_size=group_size,
    )

    # Set PyTorch backend for training compatibility (supports backprop)
    HQQLinear.set_backend(HQQBackend.PYTORCH)

    quantized_count = 0
    skipped_count = 0

    from tqdm import tqdm

    modules_list = list(model.named_modules())
    for name, module in tqdm(modules_list, desc=f"HQQ {precision} quantization", unit="layer"):
        # Skip excluded layers
        if any(pattern in name.lower() for pattern in EXCLUDE_PATTERNS):
            skipped_count += 1
            continue

        # Only quantize Linear layers
        if isinstance(module, torch.nn.Linear):
            try:
                # Get parent module and attribute name
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                # Get device as string (HQQ expects "cpu" or "cuda")
                device = module.weight.device
                device_str = "cuda" if device.type == "cuda" else "cpu"

                # Replace with HQQ quantized linear
                quantized_linear = HQQLinear(
                    linear_layer=module,
                    quant_config=quant_config,
                    compute_dtype=torch.bfloat16,
                    device=device_str,
                    initialize=True,
                )
                setattr(parent, attr_name, quantized_linear)
                quantized_count += 1
            except Exception as e:
                logger.debug(f"Could not quantize {name}: {e}")
                skipped_count += 1

    logger.info(
        f"HQQ {precision} quantization complete: "
        f"{quantized_count} layers quantized, {skipped_count} layers skipped"
    )

    # Validate that quantization actually succeeded
    if quantized_count == 0:
        raise RuntimeError(
            f"HQQ quantization failed: 0 layers were quantized ({skipped_count} skipped). "
            "This may indicate an incompatible model architecture or all layers matching exclude patterns."
        )

    return model


def estimate_memory_savings(model: torch.nn.Module, precision: HQQQuantizationOptions) -> dict:
    """
    Estimate memory savings from HQQ quantization.

    Args:
        model: The model to analyze.
        precision: The target precision.

    Returns:
        Dict with memory estimates.
    """
    total_params = sum(p.numel() for p in model.parameters())
    original_bytes = total_params * 2  # Assuming BF16 original

    bits_per_param = 4 if precision == "int4" else 8
    # HQQ adds overhead for scales/zeros (~10-15%)
    overhead_factor = 1.15
    quantized_bytes = (total_params * bits_per_param / 8) * overhead_factor

    return {
        "original_gb": original_bytes / 1e9,
        "quantized_gb": quantized_bytes / 1e9,
        "savings_percent": (1 - quantized_bytes / original_bytes) * 100,
    }
