
import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_NAME = "distilbert-base-multilingual-cased"

LABELS = [
    "delivery_question",
    "out_of_scope",
    "refund_request",
    "repair_request",
    "technical_issue",
    "warranty_question"
]

print("Loading multilingual DistilBERT student...")
start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label={i: label for i, label in enumerate(LABELS)},
    label2id={label: i for i, label in enumerate(LABELS)}
)

model.eval()

examples = [
    "My refrigerator is not cooling.",
    "Buzdolabım soğutmuyor."
]

encoded = tokenizer(
    examples,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**encoded)

print("\nDISTILBERT STUDENT VERIFICATION")
print("=" * 55)
print("Architecture:", model.config.model_type)
print("Parameters:", f"{model.num_parameters():,}")
print("Number of labels:", model.config.num_labels)
print("Output shape:", tuple(outputs.logits.shape))
print("Loading and verification time:", round(
    time.perf_counter() - start, 2
), "seconds")

assert outputs.logits.shape == (2, 6)

print("\nStudent forward pass: PASSED")
print("NOTE: Classification head is not trained yet.")
