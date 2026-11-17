
import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_NAME = "bert-base-multilingual-cased"

LABELS = [
    "delivery_question",
    "out_of_scope",
    "refund_request",
    "repair_request",
    "technical_issue",
    "warranty_question"
]

print("Loading multilingual BERT...")
print("Model:", MODEL_NAME)
print("Device: CPU")

start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label={i: label for i, label in enumerate(LABELS)},
    label2id={label: i for i, label in enumerate(LABELS)}
)

print("\nMULTILINGUAL BERT LOADED")
print("=" * 55)

print("Tokenizer vocabulary:", tokenizer.vocab_size)
print("Model parameters:", f"{model.num_parameters():,}")
print("Number of labels:", model.config.num_labels)
print("Load time (seconds):", round(
    time.perf_counter() - start, 2
))

# Check English and Turkish tokenization.
examples = [
    "My washing machine is leaking water.",
    "Çamaşır makinem su sızdırıyor."
]

for text in examples:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0]
    )

    print("\nMessage:", text)
    print("Tokens:", tokens)

# Verify the model can perform a forward pass.
model.eval()

with torch.no_grad():
    encoded = tokenizer(
        examples,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    outputs = model(**encoded)

print("\nForward pass: PASSED")
print("Logits shape:", tuple(outputs.logits.shape))

print("\nNOTE:")
print("Classification head is not trained yet.")
print("Predictions are not meaningful until fine-tuning.")
