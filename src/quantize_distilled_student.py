
import json
import time
from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

SOURCE_DIR = Path("models/distilled_student")
OUTPUT_DIR = Path("models/distilled_student_int8")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)

print("Loading trained distilled student...")

model = AutoModelForSequenceClassification.from_pretrained(
    str(SOURCE_DIR)
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    str(SOURCE_DIR)
)

print("Applying dynamic INT8 quantization...")

start = time.perf_counter()

quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

quantization_time = time.perf_counter() - start

# A quantized PyTorch model is saved as a stateful
# PyTorch object, not as a standard HF safetensors model.
# Load it only from files you trust.
model_path = OUTPUT_DIR / "quantized_model.pt"
torch.save(quantized_model, model_path)

tokenizer.save_pretrained(str(OUTPUT_DIR))

original_size = sum(
    p.stat().st_size
    for p in SOURCE_DIR.rglob("*")
    if p.is_file()
) / (1024 ** 2)

quantized_size = sum(
    p.stat().st_size
    for p in OUTPUT_DIR.rglob("*")
    if p.is_file()
) / (1024 ** 2)

print("\nQUANTIZATION RESULTS")
print("=" * 55)
print("Original model directory (MB):", round(original_size, 2))
print("Quantized directory (MB):", round(quantized_size, 2))

if original_size > 0:
    reduction = (
        1 - quantized_size / original_size
    ) * 100
    print("Storage reduction (%):", round(reduction, 2))

print("Quantization time (seconds):", round(
    quantization_time, 2
))

# Verify the quantized model can still predict.
examples = [
    "My refrigerator is broken.",
    "Buzdolabım bozuldu."
]

inputs = tokenizer(
    examples,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = quantized_model(**inputs)

print("Output shape:", tuple(outputs.logits.shape))

assert outputs.logits.shape == (2, 6)

print("Quantized model forward pass: PASSED")
print("Saved:", model_path)

with open(
    OUTPUT_DIR / "quantization_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump({
        "source": str(SOURCE_DIR),
        "method": "dynamic_int8",
        "quantized_layers": "torch.nn.Linear",
        "original_directory_mb": round(original_size, 2),
        "quantized_directory_mb": round(quantized_size, 2),
        "quantization_seconds": round(
            quantization_time, 2
        )
    }, file, indent=2)
