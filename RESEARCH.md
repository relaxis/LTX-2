# The definitive guide to training LTX-2 LoRAs on LTX-Video 2.0

Lightricks' LTX-Video family represents the first DiT-based (Diffusion Transformer) video generation architecture, with **LTX-2 being a 19-billion parameter audio-video foundation model** capable of generating synchronized audio and video in a single pass at up to 4K resolution. Training LoRAs on these models requires specific configurations, substantial VRAM, and careful attention to dataset preparation—but can achieve impressive results in under an hour with proper setup.

The official **LTX-Video-Trainer** repository from Lightricks provides the most comprehensive training pipeline, supporting LoRA, full fine-tuning, and IC-LoRA (In-Context LoRA) for video-to-video transformations. This guide synthesizes official documentation, community best practices, and empirical training studies to provide a complete technical reference.

---

## Model architecture and variant selection

LTX-Video uses a Diffusion Transformer architecture inspired by DiT and PixArt-alpha. Understanding which model variant to train on is critical for balancing quality, speed, and hardware requirements.

### Available model variants

| Model | Parameters | Best For | Training Compatibility |
|-------|------------|----------|----------------------|
| **ltxv-2b-0.9.6-dev** | 2B | Consumer GPUs, rapid iteration | ✅ Full LoRA support |
| **ltxv-13b-0.9.7-dev** | 13B | Production quality | ✅ Requires 24GB+ VRAM |
| **ltx-2-19b-dev** | 19B | Audio+video, 4K output | ✅ Requires 80GB+ VRAM |
| **Distilled variants** | Same | Fast inference only | ❌ Not for training |

**Critical distinction**: Always train on the **dev (non-distilled)** base models. Distilled versions are optimized for 8-step inference with CFG=1 and will not produce proper LoRA results. The distilled models lack the full denoising capacity needed for fine-tuning.

---

## Dataset requirements and preparation

Dataset quality determines LoRA quality more than any hyperparameter. The preprocessing pipeline must respect LTX-Video's strict dimensional requirements while providing sufficient diversity.

### Video specifications

Resolution and frame count follow specific mathematical constraints:

- **Resolution**: Must be divisible by **32** in both dimensions
- **Frame count**: Must follow the pattern **(8n + 1)** — valid counts include 9, 17, 25, 33, 41, 49, 65, 81, 97, 121, 153, 257
- **Optimal resolution**: Under **720×1280** for best performance
- **FPS**: 24-30 FPS standard (LTX-2 supports up to 50 FPS)

### Recommended dataset sizes

| LoRA Type | Minimum Videos/Images | Optimal Range | Notes |
|-----------|----------------------|---------------|-------|
| Style transfer | 5-10 images | 20-50 images | Image-only works well |
| Character/subject | 15-30 clips | 50-100 clips | Include pose diversity |
| Motion/effect | 20-50 clips | 100+ clips | Requires temporal variety |
| IC-LoRA (control) | 100+ pairs | 500+ pairs | Paired input/output needed |

### Video length guidelines

Training clips are automatically truncated to approximately **4-5 seconds** during preprocessing. For optimal results:

- **Minimum clip length**: 2 seconds (to capture meaningful motion)
- **Maximum clip length**: 6-8 seconds (longer clips get split)
- **Scene detection**: Use PySceneDetect or similar tools to split at natural cuts
- **Consistency**: Ensure caption matches the actual clip content after splitting

### Captioning requirements

Detailed captioning is non-negotiable for quality results. LTX-Video responds best to **chronological, motion-aware descriptions** under 200 words.

**Caption structure** (in order):
1. Main action and movement
2. Character/object appearances
3. Background and environment
4. Camera movement and angle
5. Lighting conditions
6. Temporal events and transitions

**Example effective caption**:
> "A ceramic coffee mug transforms into chocolate cake, melting from the rim downward in smooth flowing motion. The white porcelain surface ripples and darkens to rich brown as frosting forms in swirling patterns. Steam wisps become chocolate drizzle. Shot on wooden table with warm window light from the left. Camera holds static medium close-up throughout the 4-second transformation."

**Captioning tools**: Florence-2 for images, or run video frames through a Vision LLM. Standard image captioners miss motion descriptions, which critically hurts training quality.

### Preprocessing commands

```bash
# Using LTX-Video-Trainer
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x49" \
    --caption-column "caption" \
    --video-column "media_path"
```

---

## Training hyperparameters in depth

These parameters represent the synthesis of official Lightricks configurations and community empirical testing.

### LoRA architecture settings

| Parameter | Recommended Value | Range | Impact |
|-----------|------------------|-------|--------|
| **Rank (r)** | 128 | 32-128 | Higher = more capacity, more VRAM |
| **Alpha** | Same as rank | 32-128 | Scaling factor; matching rank is standard |
| **Target modules** | `to_q to_k to_v to_out.0` | — | Attention layers only |
| **Dropout** | 0.0 | 0.0-0.1 | Rarely needed for video LoRAs |

**Rank selection guidance**: Use **rank 32** for small datasets (under 20 samples) or subtle style adjustments. Use **rank 128** for dramatic transformations, complex characters, or datasets with 50+ samples. Higher rank with insufficient data causes overparameterization—manifesting as extra limbs or grotesque artifacts.

### Learning rate and schedule

```yaml
lr: 0.0002              # 2e-4 is the official default
lr_scheduler: "linear"   # Best for most cases
lr_warmup_steps: 100     # 5-10% of total steps
lr_num_cycles: 1         # For cosine schedulers
```

**Learning rate ranges by use case**:
- **Style LoRA**: 1e-4 to 2e-4
- **Character LoRA**: 5e-5 to 1e-4
- **Motion/effect LoRA**: 1e-4 to 2e-4
- **Fine detail work**: 1e-5 to 5e-5

The community consensus strongly favors **lower learning rate + more steps** over higher learning rate with fewer steps. A linear schedule with warmup provides the most consistent results.

### Training steps recommendations

| Dataset Size | Recommended Steps | Checkpoint Interval |
|--------------|------------------|---------------------|
| Single image/style | 700-1,400 | Every 100 steps |
| Small (5-20 samples) | 1,000-1,500 | Every 200 steps |
| Medium (20-50 samples) | 1,500-2,500 | Every 300 steps |
| Large (50+ samples) | 2,500-3,500 | Every 500 steps |

**Empirical observations from training studies**:
- **700 steps**: Early style transfer visible
- **1,400 steps**: Good balance of likeness and motion preservation
- **2,400+ steps**: Maximum fidelity but may reduce motion in single-image LoRAs

### Optimizer configuration

```yaml
optimizer: adamw          # Or "adamw8bit" for VRAM savings
beta1: 0.9
beta2: 0.95
epsilon: 1e-8
weight_decay: 0.001
max_grad_norm: 1.0
```

The **adamw8bit** optimizer reduces VRAM by approximately 30% with negligible quality impact—essential for 24GB cards.

### Batch size and gradient accumulation

| VRAM | Batch Size | Gradient Accumulation | Effective Batch |
|------|------------|----------------------|-----------------|
| 24GB | 1 | 4 | 4 |
| 48GB | 2-4 | 2 | 4-8 |
| 80GB | 4-8 | 1 | 4-8 |

For single-image training on high-VRAM systems, batch sizes up to **28** have been used successfully. Always enable **gradient_checkpointing: true**—it's mandatory for fitting training into consumer VRAM.

---

## Complete configuration examples

### Standard LoRA training (24GB VRAM)

```yaml
# ltxv_13b_lora_standard.yaml
model_source: "LTXV_13B_097_DEV"
training_mode: "lora"
mixed_precision: bf16

# LoRA settings
rank: 128
lora_alpha: 128
target_modules: "to_q to_k to_v to_out.0"

# Optimization
lr: 0.0002
lr_scheduler: "linear"
lr_warmup_steps: 100
optimizer: adamw
beta1: 0.9
beta2: 0.95
epsilon: 1e-8
max_grad_norm: 1

# Data
batch_size: 1
gradient_accumulation_steps: 4
caption_dropout_p: 0.05
caption_dropout_technique: "empty"

# Memory optimization
gradient_checkpointing: true
enable_slicing: true
enable_tiling: true

# Checkpointing
checkpointing_steps: 200
checkpointing_limit: 10
```

### Low-VRAM configuration (16-24GB)

```yaml
# ltxv_2b_lora_low_vram.yaml
model_source: "LTXV_2B_0.9.6_DEV"
training_mode: "lora"
mixed_precision: bf16

# LoRA settings (lower rank for memory)
rank: 32
lora_alpha: 32
target_modules: "to_q to_k to_v to_out.0"

# Optimization
lr: 0.0002
lr_scheduler: "linear"
lr_warmup_steps: 100
optimizer: adamw8bit

# Data
batch_size: 1
gradient_accumulation_steps: 1

# Critical memory optimizations
gradient_checkpointing: true
enable_slicing: true
enable_tiling: true
quantization: "fp8-quanto"
load_text_encoder_in_8bit: true
```

### finetrainers shell script

```bash
#!/bin/bash
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1

GPU_IDS="0"
DATA_ROOT="/path/to/dataset"
OUTPUT_DIR="/path/to/output"
ID_TOKEN="MYSTYLE"

accelerate launch --config_file accelerate_configs/uncompiled_1.yaml \
    --gpu_ids $GPU_IDS train.py \
    --model_name ltx_video \
    --pretrained_model_name_or_path Lightricks/LTX-Video \
    --data_root $DATA_ROOT \
    --video_column videos.txt \
    --caption_column prompts.txt \
    --id_token $ID_TOKEN \
    --video_resolution_buckets 49x512x768 \
    --caption_dropout_p 0.05 \
    --training_type lora \
    --seed 42 \
    --mixed_precision bf16 \
    --batch_size 1 \
    --train_steps 2000 \
    --rank 128 \
    --lora_alpha 128 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 100
```

---

## Hardware requirements breakdown

### VRAM requirements by task

| Task | Minimum VRAM | Recommended | Optimal |
|------|-------------|-------------|---------|
| **LTXV-2B LoRA** | 16GB | 24GB | 48GB |
| **LTXV-13B LoRA** | 24GB (quantized) | 48GB | 80GB |
| **LTX-2 LoRA** | 48GB | 80GB | 80GB+ |
| **Full fine-tuning** | 80GB | 2×80GB | 8×80GB |

### GPU recommendations

- **Entry-level training**: RTX 4090 (24GB) with fp8-quanto quantization
- **Standard training**: A6000 (48GB) or dual RTX 3090s
- **Production training**: H100 (80GB) or A100 (80GB)
- **LTX-2 full training**: 80GB+ highly recommended; the official docs state this explicitly

### Inference-only requirements

| Configuration | VRAM | Notes |
|--------------|------|-------|
| Full model bf16 | 24GB+ | RTX 3090/4090 |
| FP8 quantized | 8-12GB | RTX 4060/4070 |
| Minimum viable | 6GB | 512×512, quantized text encoder |

---

## Inference specifications for testing LoRAs

### Resolution and duration capabilities

| Model | Max Native Resolution | Max Duration | Max FPS |
|-------|----------------------|--------------|---------|
| **LTXV-2B/13B** | 720×1280 | 60 seconds | 30 |
| **LTX-2** | 3840×2160 (4K) | 10-15 seconds | 50 |

### Recommended inference settings

```python
# Testing LoRA quality
inference_steps: 40-50          # For quality evaluation
cfg_scale: 3.0-3.5              # Standard models
lora_strength: 0.55-0.75        # Sweet spot for most LoRAs
frames: 97+                     # Minimum for accurate style assessment
```

**CFG scale guidance**:
- **Standard models**: 3.0-3.5 optimal
- **Image-to-video**: 4.0-5.0
- **Distilled models**: 1.0 (CFG not required)
- **Below 2.0**: Causes "excessive floating" artifacts
- **Above 5.0**: Overbakes output, loses quality

### Loading LoRAs in diffusers

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.load_lora_weights("path/to/lora", adapter_name="my-lora")
pipe.set_adapters(["my-lora"], [0.75])

video = pipe(
    prompt="your prompt here",
    num_inference_steps=40,
    guidance_scale=3.5,
    num_frames=97
).frames[0]

export_to_video(video, "output.mp4", fps=24)
```

---

## Training frameworks comparison

### Lightricks/LTX-Video-Trainer (recommended)

The official trainer from Lightricks provides the most complete and tested pipeline.

**Installation**:
```bash
git clone https://github.com/Lightricks/LTX-Video-Trainer
cd LTX-Video-Trainer
pip install -e .
```

**Strengths**:
- Official support, actively maintained
- IC-LoRA support for video-to-video control
- Pre-built configs for all model variants
- Automated preprocessing pipeline

### HuggingFace/finetrainers

A diffusers-native alternative with broad model support.

**Installation**:
```bash
git clone https://github.com/huggingface/finetrainers
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
```

**Strengths**:
- Tight diffusers integration
- Lower minimum VRAM (5GB with FP8)
- GUI available (finetrainers-ui)

### tdrussell/diffusion-pipe

DeepSpeed-based trainer with pipeline parallelism support.

**Strengths**:
- Multi-GPU scaling with DeepSpeed
- TOML configuration format
- Outputs ComfyUI-compatible LoRAs directly

| Feature | LTX-Video-Trainer | finetrainers | diffusion-pipe |
|---------|------------------|--------------|----------------|
| Official support | ✅ | Partial | Community |
| IC-LoRA | ✅ | ❌ | ❌ |
| Min VRAM | ~16GB | ~5GB (FP8) | ~24GB |
| Multi-GPU | ✅ (distributed) | ✅ (FSDP) | ✅ (DeepSpeed) |
| Config format | YAML | CLI/Shell | TOML |

---

## Best practices and quality optimization

### Critical success factors

1. **Caption quality trumps quantity**: Five videos with excellent motion-aware captions outperform fifty with generic descriptions

2. **Match rank to dataset size**: Overparameterization (high rank + small dataset) causes artifacts; underparameterization (low rank + large dataset) limits capacity

3. **Enable all memory optimizations**: Gradient checkpointing, slicing, and tiling have minimal quality impact but massive VRAM savings

4. **Test with consistent seeds**: Compare checkpoints using identical prompts, seeds, and frame counts to isolate training progress

5. **Use validation during training**: First-frame conditioning (image-to-video) provides more reliable validation than text-to-video

### Common mistakes and solutions

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Overfitting** | Only works with training-similar prompts | Reduce steps, add caption dropout, diversify data |
| **Overparameterization** | Extra limbs, grotesque features | Lower rank, or add more training data |
| **Static output** | Videos barely move | Blur input images slightly, add motion to prompts |
| **Caption mismatch** | Inconsistent results | Verify captions match clips after scene splitting |
| **Poor generalization** | Works on training subjects only | Add pose/background diversity to dataset |

### Detecting overfitting

**Early signs**:
- Validation loss plateaus while training loss decreases
- LoRA only produces good results with prompts nearly identical to training captions
- Increasing LoRA strength beyond 0.8 causes artifacts without improving likeness

**Testing method**: Set LoRA strength to 2.0+. If the subject appears (despite heavy artifacting), training is working. If nothing recognizable appears, training hasn't converged. If the output looks nearly identical to training frames, overfitting has occurred.

### When to stop training

Monitor these checkpoints:
- **700 steps**: First visible results; continue if quality is poor
- **1,400 steps**: Evaluate likeness vs. motion trade-off
- **2,400 steps**: Maximum fidelity checkpoint; motion may decrease
- **3,000+ steps**: Rarely beneficial; high overfitting risk

The relationship between steps and quality is **non-linear**—performance can regress between checkpoints before improving again. Save checkpoints every 100-200 steps and test multiple.

---

## Data augmentation strategies

Augmentation helps prevent overfitting but requires caution with video data.

**Safe augmentations**:
- Horizontal flips (when subject is symmetric)
- Subtle brightness/contrast adjustments (±10%)
- Minor saturation shifts

**Risky augmentations**:
- Temporal speed changes (can confuse motion learning)
- Aggressive cropping (alters composition semantics)
- Color space transformations (may cause artifacts)

**Recommended config**:
```yaml
caption_dropout_p: 0.05
caption_dropout_technique: "empty"
# Apply spatial augmentation in preprocessing, not during training
```

---

## Conclusion

Training quality LTX-2 LoRAs requires balancing three key factors: **dataset quality** (motion-aware captions, appropriate clip lengths, sufficient diversity), **architecture choices** (rank matched to dataset size, correct base model selection), and **optimization settings** (conservative learning rates with adequate warmup, appropriate step counts with frequent checkpointing).

The **Lightricks LTX-Video-Trainer** repository provides the most robust starting point, with pre-configured YAML files for common scenarios. For consumer hardware, the low-VRAM configuration using fp8-quanto quantization and 8-bit text encoders enables training on 24GB cards. For production work, 48-80GB VRAM unlocks the full 13B model and higher batch sizes.

The most impactful improvement for most users comes from **caption quality**, not hyperparameter tuning. Invest time in creating detailed, chronological, motion-aware descriptions for every training clip. A well-captioned dataset of 20 videos will consistently outperform a poorly-captioned dataset of 200.