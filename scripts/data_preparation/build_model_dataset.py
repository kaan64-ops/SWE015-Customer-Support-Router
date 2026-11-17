
from pathlib import Path
import pandas as pd

support = pd.read_csv(
    "data/support_split_100.csv",
    dtype=str,
    encoding="utf-8-sig"
).fillna("")

ood = pd.read_csv(
    "data/massive_ood_clean.csv",
    dtype=str,
    encoding="utf-8-sig"
).fillna("")

# Use only manually approved OOS groups.
ood = ood[ood["review_decision"] == "keep"].copy()

required = {"text", "intent", "language", "split"}
for name, df in [("support", support), ("ood", ood)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

# Create a common schema.
support_data = support[
    ["text", "intent", "language", "split"]
].copy()

ood_data = ood[
    ["text", "language", "split"]
].copy()
ood_data["intent"] = "out_of_scope"

combined = pd.concat(
    [
        support_data,
        ood_data[["text", "intent", "language", "split"]]
    ],
    ignore_index=True
)

combined["text"] = combined["text"].str.strip()
combined["intent"] = combined["intent"].str.strip()
combined["language"] = combined["language"].str.strip()
combined["split"] = combined["split"].str.strip()

if combined["text"].eq("").any():
    raise ValueError("Empty message detected.")

if not combined["split"].isin(
    ["train", "validation", "test"]
).all():
    raise ValueError("Unexpected split detected.")

if combined.duplicated(
    ["language", "text", "split"]
).any():
    raise ValueError("Exact duplicate detected.")

output = Path("data/model_dataset_verified.csv")
combined.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("VERIFIED MODEL DATASET CREATED")
print("=" * 55)
print("Total messages:", len(combined))

print("\nIntent distribution:")
print(combined["intent"].value_counts())

print("\nSplit distribution:")
print(pd.crosstab(combined["split"], combined["intent"]))

print("\nLanguage distribution:")
print(pd.crosstab(combined["split"], combined["language"]))

print("\nSaved:", output)
