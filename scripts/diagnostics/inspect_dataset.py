
from datasets import load_dataset
from collections import Counter

print("Loading Turkish MASSIVE dataset...")

dataset = load_dataset(
    "SetFit/amazon_massive_intent_tr-TR"
)

print("\nDATASET SPLITS")
print("=" * 50)

for split in dataset.keys():
    print(f"{split}: {len(dataset[split])} examples")

print("\nDATASET COLUMNS")
print(dataset["train"].column_names)

print("\nFIRST FIVE TRAINING EXAMPLES")
print("=" * 50)

for example in dataset["train"].select(range(5)):
    print("Text:", example["text"])
    print("Intent:", example["label_text"])
    print("-" * 40)

print("\nMOST FREQUENT INTENTS")
print("=" * 50)

counts = Counter(dataset["train"]["label_text"])

print("Number of intent classes:", len(counts))

for intent, count in counts.most_common(20):
    print(f"{intent}: {count}")

print("\nDATASET INSPECTION COMPLETE")
