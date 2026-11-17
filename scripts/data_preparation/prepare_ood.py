
from datasets import load_dataset
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

sources = {
    "en": "SetFit/amazon_massive_intent_en-US",
    "tr": "SetFit/amazon_massive_intent_tr-TR",
}

all_splits = []

for language, source in sources.items():
    print(f"Loading {language}: {source}")
    dataset = load_dataset(source)

    for split in ["train", "validation", "test"]:
        df = dataset[split].to_pandas()

        df = df[["id", "text", "label_text"]].copy()
        df["language"] = language
        df["source"] = "MASSIVE"
        df["original_intent"] = df["label_text"]
        df["intent"] = "out_of_scope"
        df["split"] = split

        all_splits.append(df)

combined = pd.concat(all_splits, ignore_index=True)

# MASSIVE translations share IDs. Preserve them so
# English/Turkish pairs stay in the same split.
combined["group_id"] = (
    "massive_" + combined["id"].astype(str)
)

columns = [
    "text", "intent", "language", "source",
    "original_intent", "split", "group_id"
]

combined = combined[columns]

# Export a manageable, reproducible sample.
# Do not alter the original train/validation/test split.
sample_sizes = {
    "train": 500,
    "validation": 100,
    "test": 100,
}

samples = []

for split, size in sample_sizes.items():
    split_df = combined[combined["split"] == split]

    # Sample shared IDs so bilingual pairs stay together.
    ids = (
        split_df["group_id"]
        .drop_duplicates()
        .sample(n=size, random_state=42)
    )

    selected = split_df[
        split_df["group_id"].isin(ids)
    ]

    samples.append(selected)

result = pd.concat(samples, ignore_index=True)

result.to_csv(
    OUTPUT_DIR / "massive_ood_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nExport completed!")
print(result.groupby(["split", "language"]).size())
print("\nSaved: data/massive_ood_candidates.csv")
