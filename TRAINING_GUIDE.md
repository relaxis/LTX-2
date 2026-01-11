# LTX-2 LoRA Training Guide

A practical guide for training LTX-2 audio-video LoRAs on consumer GPUs (24-32GB VRAM).

**This fork adds:**
- RamTorch CPU offloading for training on consumer GPUs
- Memory-efficient configurations tested on RTX 5090
- Practical guidance from real-world training experience

## Table of Contents
1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Download Models](#3-download-models)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Captioning with Qwen + Whisper](#5-captioning-with-qwen--whisper)
6. [Preprocessing (Latent Computation)](#6-preprocessing-latent-computation)
7. [Training Configuration](#7-training-configuration)
8. [Running Training](#8-running-training)
9. [Testing Checkpoints](#9-testing-checkpoints)
10. [Parameter Reference](#10-parameter-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 24GB VRAM | 32GB VRAM |
| System RAM | 32GB | 64GB+ |
| Storage | 100GB | 200GB+ |

### Software

- **OS**: Linux (required - triton is Linux-only)
- **Python**: 3.10+
- **CUDA**: 12.0+

### Tested Configurations

| GPU | VRAM | RAM | Config | Speed |
|-----|------|-----|--------|-------|
| RTX 5090 | 32GB | 64GB | BF16 + 60% ramtorch | ~7s/step |
| A100 | 80GB | 128GB | BF16, no offload | ~4s/step |

---

## 2. Installation

### Clone the Repository

```bash
git clone https://github.com/relaxis/LTX-2.git
cd LTX-2
```

### Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Environment and Install Dependencies

```bash
uv sync
source .venv/bin/activate
```

### Setup Weights & Biases (Recommended)

For loss tracking and training visualization:

```bash
uv run wandb login
```

---

## 3. Download Models

Create a `models/` directory:

```bash
mkdir -p models
cd models
```

### LTX-2 Model (Required)

Download the **BF16 model** for training (FP8 doesn't support autograd):

```bash
wget https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors
```

### Gemma Text Encoder (Required)

**Option A: Standard Gemma 3**
```bash
git lfs install
git clone https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized gemma-3
```

**Option B: Abliterated Gemma (for uncensored content)**

For NSFW or unrestricted content generation, use an abliterated Gemma model with safety refusals removed. Search HuggingFace for "gemma abliterated" models.

---

## 4. Dataset Preparation

### Video Requirements

| Requirement | Specification |
|-------------|---------------|
| **Frame count** | Must satisfy `frames % 8 == 1` (valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121) |
| **Resolution** | Width and height must be divisible by 32 |
| **Duration** | Minimum 5 seconds recommended |
| **Format** | MP4, AVI, MOV, or other ffmpeg-compatible formats |
| **Audio** | Required for audio-video training (will be converted to stereo 44.1kHz) |

### Recommended Resolutions by VRAM

| VRAM | Resolution | Frames | Notes |
|------|------------|--------|-------|
| 24GB | 512x512 | 49-89 | Testing, simple concepts |
| 32GB | 704x384 | 97 | Standard training |
| 48GB | 768x512 | 97-121 | Higher quality |
| 80GB | 1024x768 | 121+ | Production quality |

### Dataset Size Guidelines

| Dataset Size | Recommended Steps | Effective Epochs | Notes |
|--------------|-------------------|------------------|-------|
| 10-25 videos | 500-1000 | 20-100 | Small concept/style |
| 25-50 videos | 1000-2000 | 20-80 | Standard concept |
| 50-100 videos | 2000-3000 | 20-60 | Complex concept |
| 100+ videos | 2000-4000 | 20-40 | Large-scale training |

**Formula:** `effective_epochs = (steps * gradient_accumulation) / dataset_size`

### Directory Structure

```
datasets/my_dataset/
  videos/
    video1.mp4
    video2.mp4
    ...
  dataset.jsonl
```

### dataset.jsonl Format

Each line is a JSON object:

```json
{"media_path": "videos/video1.mp4", "prompt": "[VISUAL]: trigger. Description... [SPEECH]: None [SOUNDS]: ambient [TEXT]: None"}
```

---

## 5. Captioning with Qwen + Whisper

For NSFW or uncensored content, use an abliterated/uncensored VLM model.

### Required Models
- **Visual captioning:** `huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated` (or similar uncensored)
- **Audio transcription:** `openai/whisper-large-v3`

### Caption Format (LTX-2)

LTX-2 expects captions with explicit section markers:

```
[VISUAL]: trigger_token. Detailed visual description of the scene...
[SPEECH]: Transcribed speech or "None"
[SOUNDS]: Description of ambient/non-speech sounds
[TEXT]: Any on-screen text or "None"
```

### Example Captioning Script

```python
#!/usr/bin/env python3
"""
Caption videos using Qwen2.5-VL + Whisper for LTX-2 training
"""

import os
import torch
import json
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

# Qwen2.5-VL imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Whisper imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# === CONFIGURATION ===
VIDEO_DIR = "/path/to/your/videos"
OUTPUT_JSONL = "/path/to/output/dataset.jsonl"
TRIGGER_TOKEN = "your_trigger"  # e.g., "mystyle", "mychar"

QWEN_MODEL = "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated"
WHISPER_MODEL = "openai/whisper-large-v3"

CAPTION_PROMPT = """Describe this video in detail for AI training.
Focus on: actions, movements, camera angles, lighting, subjects.
Be descriptive and specific. Just provide the description."""


def extract_audio(video_path: Path, output_path: Path) -> bool:
    """Extract audio from video using ffmpeg"""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(output_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"  Audio extraction failed: {e}")
        return False


def transcribe_audio(whisper_model, whisper_processor, audio_path: Path, device: str) -> str:
    """Transcribe audio using Whisper"""
    try:
        audio, sr = librosa.load(str(audio_path), sr=16000)
        inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=torch.float16) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = whisper_model.generate(
                inputs["input_features"],
                max_length=448,
                language="en",
                task="transcribe"
            )

        transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f"  Transcription failed: {e}")
        return ""


def caption_video(qwen_model, qwen_processor, video_path: Path, device: str) -> str:
    """Caption a video using Qwen2.5-VL"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": 360 * 420,
                    "nframes": 16,
                },
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_dir = Path(VIDEO_DIR)
    video_files = sorted(list(video_dir.glob("**/*.mp4")))
    print(f"Found {len(video_files)} videos")

    # Load models
    print("Loading Whisper...")
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    print("Loading Qwen2.5-VL...")
    qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )

    results = []
    for video_path in tqdm(video_files, desc="Captioning"):
        # Transcribe audio
        speech_text = ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            if extract_audio(video_path, tmp_path):
                speech_text = transcribe_audio(whisper_model, whisper_processor, tmp_path, device)

        # Caption video
        try:
            visual_caption = caption_video(qwen_model, qwen_processor, video_path, device)
        except Exception as e:
            print(f"  Caption failed: {e}")
            visual_caption = "video content"

        # Build LTX-2 format caption
        speech_line = speech_text if speech_text else "None"
        full_caption = f"[VISUAL]: {TRIGGER_TOKEN}. {visual_caption} [SPEECH]: {speech_line} [SOUNDS]: ambient sounds [TEXT]: None"

        results.append({
            "prompt": full_caption,
            "media_path": str(video_path.relative_to(video_dir))
        })

    # Save JSONL
    output_path = Path(OUTPUT_JSONL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Saved {len(results)} captions to {output_path}")


if __name__ == "__main__":
    main()
```

### Dependencies for Captioning

```bash
pip install transformers accelerate qwen-vl-utils librosa torch
pip install flash-attn --no-build-isolation  # Optional, faster inference
```

---

## 6. Preprocessing (Latent Computation)

Preprocessing converts videos and captions into latent representations for efficient training.

### Step 1: Process Videos to VAE Latents

```bash
cd packages/ltx-trainer

uv run python scripts/process_videos.py \
  --input-dir /path/to/datasets/my_dataset \
  --output-dir /path/to/datasets/my_dataset/.precomputed \
  --model-path /path/to/models/ltx-2-19b-dev.safetensors
```

### Step 2: Process Captions to Text Embeddings

```bash
uv run python scripts/process_captions.py \
  --input-dir /path/to/datasets/my_dataset \
  --output-dir /path/to/datasets/my_dataset/.precomputed \
  --text-encoder-path /path/to/models/gemma-3
```

### All-in-One Processing (Alternative)

```bash
uv run python scripts/process_dataset.py /path/to/dataset.jsonl \
    --resolution-buckets "704x384x97" \
    --model-path /path/to/models/ltx-2-19b-dev.safetensors \
    --text-encoder-path /path/to/models/gemma-3 \
    --with-audio
```

### Output Structure

```
dataset_folder/.precomputed/
├── conditions/          # Text embeddings (.pt files)
│   └── videos/
├── latents/            # Video latents (.pt files)
│   └── videos/
└── audio_latents/      # Audio latents (.pt files)
    └── videos/
```

**Important:** All three directories must have matching files. If any video fails to process (too short, no audio), remove it from all directories.

---

## 7. Training Configuration

### Memory-Efficient Config (24-32GB GPU)

```yaml
# configs/my_lora.yaml

model:
  model_path: "/path/to/models/ltx-2-19b-dev.safetensors"
  text_encoder_path: "/path/to/models/gemma-3"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: 64
  alpha: 64
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"
    - "ff.net.0.proj"
    - "ff.net.2"
    - "audio_ff.net.0.proj"
    - "audio_ff.net.2"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.3
  with_audio: true

optimization:
  learning_rate: 5e-5
  steps: 3000
  batch_size: 1
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "constant"
  scheduler_params: {}
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: null  # Don't use - slower than ramtorch
  hqq_quantization: null
  load_text_encoder_in_8bit: true
  ramtorch_offload: true
  ramtorch_offload_percent: 0.60  # Increase to 0.70 for 24GB GPU
  load_fp8_model: false

data:
  preprocessed_data_root: "/path/to/datasets/my_dataset/.precomputed"
  num_dataloader_workers: 2

validation:
  prompts:
    - "[VISUAL]: trigger. Your validation prompt here... [SPEECH]: None [SOUNDS]: ambient [TEXT]: None"
  negative_prompt: "worst quality, blurry, distorted"
  video_dims: [768, 512, 97]
  frame_rate: 30.0
  seed: 42
  inference_steps: 30
  interval: 999999  # Disabled - validation OOMs with ramtorch
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: true
  skip_initial_validation: true

checkpoints:
  interval: 100
  keep_last_n: 30

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {}

wandb:
  enabled: true
  project: "ltx-2-training"
  entity: null
  tags: ["ltx2", "lora"]
  log_validation_videos: false

seed: 42
output_dir: "/path/to/outputs/my_lora"
```

### Target Modules Reference

**For audio-video training (recommended):**
```yaml
target_modules:
  - "to_k"
  - "to_q"
  - "to_v"
  - "to_out.0"
  - "ff.net.0.proj"
  - "ff.net.2"
  - "audio_ff.net.0.proj"
  - "audio_ff.net.2"
```

**For video-only training:**
```yaml
target_modules:
  - "attn1.to_k"
  - "attn1.to_q"
  - "attn1.to_v"
  - "attn1.to_out.0"
```

---

## 8. Running Training

### Start Training

```bash
cd /path/to/LTX-2/packages/ltx-trainer

PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py configs/my_lora.yaml
```

### Monitor Training

- **W&B Dashboard**: Visit https://wandb.ai and find your project
- **Terminal**: Watch loss values in the progress bar
- **Checkpoints**: Saved to `output_dir/checkpoint-{step}/`

### Expected Training Time

| Steps | Time (32GB GPU) |
|-------|-----------------|
| 1000 | ~2 hours |
| 3000 | ~6 hours |
| 6000 | ~12 hours |

---

## 9. Testing Checkpoints

Since validation is disabled during training (OOMs with ramtorch), test checkpoints manually:

```bash
cd packages/ltx-pipelines

uv run python scripts/inference.py \
  --model-path /path/to/models/ltx-2-19b-dev.safetensors \
  --text-encoder-path /path/to/models/gemma-3 \
  --lora-path /path/to/outputs/my_lora/checkpoint-1000/lora_weights.safetensors \
  --prompt "[VISUAL]: trigger. Your test prompt... [SPEECH]: None [SOUNDS]: ambient [TEXT]: None" \
  --output-path test_output.mp4
```

Test multiple checkpoints (e.g., 500, 1000, 1500, 2000) to find the best one.

---

## 10. Parameter Reference

### Learning Rate by Training Type

| Training Type | LR Range | Recommended |
|--------------|----------|-------------|
| Style/aesthetic | 5e-5 to 1e-4 | 5e-5 |
| Character/subject | 1e-5 to 5e-5 | 3e-5 |
| Motion/action | 5e-5 to 2e-4 | 1e-4 |
| Complex concept | 1e-5 to 5e-5 | 5e-5 |

### LoRA Rank Guidelines

| Rank | Trainable Params | Use Case |
|------|------------------|----------|
| 8-16 | ~25-50M | Simple style transfer |
| 32 | ~100M | Standard (recommended) |
| 64-128 | ~200-400M | Complex concepts, high fidelity |

### Ramtorch Offload Settings

| GPU VRAM | Offload % | Notes |
|----------|-----------|-------|
| 24GB | 0.70-0.75 | May be slow |
| 32GB | 0.60 | Recommended |
| 40GB | 0.40-0.50 | Faster |
| 48GB+ | 0.20-0.30 | Much faster |

---

## 11. Troubleshooting

### CUDA Out of Memory

1. Increase `ramtorch_offload_percent` (e.g., 0.70)
2. Reduce `lora.rank` (e.g., 32)
3. Ensure `enable_gradient_checkpointing: true`
4. Use `PYTORCH_ALLOC_CONF=expandable_segments:True`

### Validation OOM

The validation sampler doesn't support ramtorch offloading. Set `validation.interval: 999999` to disable and test checkpoints manually.

### Slow Training

| Approach | Speed | Notes |
|----------|-------|-------|
| BF16 + ramtorch 60% | ~7s/step | Recommended |
| INT8-quanto + ramtorch | ~8s/step | Slower - dequant overhead |
| FP8-quanto + ramtorch | ~10s/step | Even slower |

**Don't use quantization** - the dequantization overhead is slower than ramtorch CPU transfers.

### FP8 Model Errors

```
NotImplementedError: "ufunc_add_CUDA" not implemented for 'Float8_e4m3fn'
```

FP8 models don't support autograd. Use the BF16 model (`ltx-2-19b-dev.safetensors`).

### Audio Not Being Learned

- Ensure `with_audio: true` in training_strategy
- Use short target_modules patterns: `"to_k"` not `"attn1.to_k"`
- Verify `audio_latents/` exists and has matching files

### Mismatched Latent Files

If preprocessing fails for some videos, you may have mismatched counts:
```
latents/videos/: 50 files
audio_latents/videos/: 47 files
conditions/videos/: 50 files
```

Remove videos that failed from all three directories to ensure matching counts.

### Loss Spikes or NaN

- Reduce learning rate
- Enable gradient clipping: `max_grad_norm: 1.0`
- Check for corrupted training samples

---

## Quick Start Checklist

1. [ ] Install dependencies: `uv sync && source .venv/bin/activate`
2. [ ] Download BF16 model: `ltx-2-19b-dev.safetensors`
3. [ ] Download Gemma text encoder
4. [ ] Prepare videos (frame count % 8 == 1, resolution % 32 == 0)
5. [ ] Create dataset.jsonl with LTX-2 format captions
6. [ ] Run preprocessing (videos + captions)
7. [ ] Verify matching file counts in `.precomputed/`
8. [ ] Create training config with ramtorch offload
9. [ ] Start training with wandb enabled
10. [ ] Test checkpoints manually to find best one
