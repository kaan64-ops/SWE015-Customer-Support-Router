
from pathlib import Path
import pandas as pd

ood = pd.read_csv(
    "data/massive_ood_review_status.csv",
    dtype=str,
    encoding="utf-8-sig"
).fillna("")

support = pd.read_csv(
    "data/support_split_100.csv",
    dtype=str,
    encoding="utf-8-sig"
).fillna("")

# Exclude rejected bilingual groups.
clean = ood[
    ood["review_decision"] != "exclude"
].copy()

# Remove the second duplicate bilingual group.
# Both duplicates are in training, so test data stays unchanged.
duplicate_group = "massive_4946"

if duplicate_group not in set(clean["group_id"]):
    raise ValueError("Expected duplicate group was not found.")

clean = clean[
    clean["group_id"] != duplicate_group
].copy()

def normalize(series):
    return (
        series.astype(str)
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

clean["normalized_text"] = normalize(clean["text"])

# Confirm no duplicates remain within the same language.
if clean.duplicated(
    ["language", "normalized_text"]
).any():
    raise ValueError("OOS duplicates still exist.")

# Check overlap with the support dataset.
support_texts = set(normalize(support["text"]))

overlap = clean[
    clean["normalized_text"].isin(support_texts)
].copy()

Path("results").mkdir(exist_ok=True)

overlap.to_csv(
    "results/ood_support_overlap.csv",
    index=False,
    encoding="utf-8-sig"
)

print("CLEAN OOS DATASET PREPARED")
print("=" * 55)
print("Remaining OOS messages:", len(clean))
print("Remaining bilingual groups:", clean["group_id"].nunique())

print("\nReview status by group:")
print(
    clean.drop_duplicates("group_id")[
        "review_decision"
    ].value_counts()
)

print("\nOOS/support exact overlaps:", len(overlap))

if len(overlap):
    print(
        overlap[
            ["group_id", "language", "text", "split"]
        ].to_string(index=False)
    )

# Save only if there is no exact overlap.
if len(overlap) == 0:
    clean = clean.drop(columns=["normalized_text"])

    output = "data/massive_ood_clean.csv"

    clean.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nData integrity: PASSED")
    print("Saved:", output)
else:
    print("\nReview overlap before saving the clean dataset.")
