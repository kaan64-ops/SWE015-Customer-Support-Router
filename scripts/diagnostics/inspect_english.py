
from datasets import load_dataset
from collections import Counter

print("Loading English MASSIVE dataset...")

dataset = load_dataset(
    "SetFit/amazon_massive_intent_en-US"
)

print("\nDATASET SPLITS")
for split in dataset.keys():
    print(f"{split}: {len(dataset[split])} examples")

print("\nDATASET COLUMNS")
print(dataset["train"].column_names)

print("\nFIRST FIVE EXAMPLES")
for example in dataset["train"].select(range(5)):
    print("Text:", example["text"])
    print("Intent:", example["label_text"])
    print("-" * 40)

counts = Counter(dataset["train"]["label_text"])

print("\nNUMBER OF INTENT CLASSES:", len(counts))
print("\nMOST FREQUENT INTENTS")

for intent, count in counts.most_common(15):
    print(f"{intent}: {count}")

print("\nENGLISH DATASET INSPECTION COMPLETE")
