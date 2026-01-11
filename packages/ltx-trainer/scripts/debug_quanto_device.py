#!/usr/bin/env python3
"""Debug script to understand quanto tensor device behavior with PEFT."""

import torch
import torch.nn as nn

# Create a simple model on CPU
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)

    def forward(self, x):
        return self.linear2(self.linear1(x))

print("Creating model on CPU...")
model = SimpleModel()
model = model.to("cpu", dtype=torch.bfloat16)

print(f"Before quanto - linear1.weight device: {model.linear1.weight.device}")
print(f"Before quanto - linear1.weight.data device: {model.linear1.weight.data.device}")

# Quantize with quanto
print("\nQuantizing with quanto...")
from optimum.quanto import quantize, freeze, qint8

quantize(model, weights=qint8)
freeze(model)

print(f"After quanto freeze - linear1.weight type: {type(model.linear1.weight)}")
print(f"After quanto freeze - linear1.weight device: {model.linear1.weight.device}")

# Check if the weight has _data or other internal tensors
if hasattr(model.linear1.weight, '_data'):
    print(f"linear1.weight._data device: {model.linear1.weight._data.device}")
if hasattr(model.linear1.weight, '_scale'):
    print(f"linear1.weight._scale device: {model.linear1.weight._scale.device}")

# Try calling .to("cpu") explicitly
print("\nCalling model.to('cpu')...")
model = model.to("cpu")
print(f"After .to('cpu') - linear1.weight device: {model.linear1.weight.device}")

# Now test with PEFT
print("\n--- Testing PEFT ---")
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["linear1", "linear2"],
    lora_dropout=0.0,
)

try:
    print("Calling get_peft_model...")
    peft_model = get_peft_model(model, lora_config)
    print("SUCCESS! PEFT model created.")

    # Check LoRA adapter devices
    for name, module in peft_model.named_modules():
        if "lora_" in name:
            if hasattr(module, 'weight'):
                print(f"  {name}: weight device = {module.weight.device}")
except Exception as e:
    print(f"PEFT FAILED with error: {type(e).__name__}: {e}")

print("\n--- Now testing with CUDA model ---")

# Create model on CUDA
model_cuda = SimpleModel().cuda().to(torch.bfloat16)
print(f"CUDA model - linear1.weight device: {model_cuda.linear1.weight.device}")

# Quantize on CUDA
quantize(model_cuda, weights=qint8)
freeze(model_cuda)
print(f"CUDA quanto - linear1.weight device: {model_cuda.linear1.weight.device}")

# Move to CPU
model_cuda = model_cuda.to("cpu")
print(f"After .to('cpu') - linear1.weight device: {model_cuda.linear1.weight.device}")

# Test PEFT
try:
    print("Calling get_peft_model on CUDA-quantized-then-CPU model...")
    peft_model_cuda = get_peft_model(model_cuda, lora_config)
    print("SUCCESS!")
except Exception as e:
    print(f"PEFT FAILED: {type(e).__name__}: {e}")
