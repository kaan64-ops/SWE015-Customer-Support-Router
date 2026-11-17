
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "models/multilingual_bert"

print("Checking saved BERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model, loading_info = (
    AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        output_loading_info=True
    )
)

print("\nSAVED BERT VERIFICATION")
print("=" * 55)
print("Number of labels:", model.config.num_labels)

print("\nMissing keys:")
print(loading_info.get("missing_keys", []))

print("\nUnexpected keys:")
print(loading_info.get("unexpected_keys", []))

print("\nTokenizer vocabulary:", tokenizer.vocab_size)

if (
    not loading_info.get("missing_keys")
    and not loading_info.get("unexpected_keys")
):
    print("\nSaved checkpoint loading: PASSED")
else:
    print("\nSaved checkpoint loading: NEEDS REVIEW")
