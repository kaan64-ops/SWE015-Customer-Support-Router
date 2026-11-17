
from pathlib import Path
import pandas as pd

first = pd.read_csv(
    "data/ood_manual_review.csv",
    dtype=str
).fillna("")

second = pd.read_csv(
    "data/ood_remaining_88_recovered.csv",
    dtype=str
).fillna("")

candidates = pd.read_csv(
    "data/massive_ood_candidates.csv",
    dtype=str
).fillna("")

# Only retain reviewed decisions from each batch.
first = first[first["review_decision"] != "pending"].copy()

decisions = pd.concat(
    [
        first[["group_id", "review_decision"]],
        second[["group_id", "review_decision"]]
    ],
    ignore_index=True
)

if len(decisions) != 121:
    raise ValueError(
        f"Expected 121 reviewed groups, found {len(decisions)}"
    )

if decisions["group_id"].duplicated().any():
    raise ValueError("Duplicate reviewed group IDs.")

if not decisions["review_decision"].isin(
    ["keep", "exclude"]
).all():
    raise ValueError("Unresolved review decisions detected.")

if not set(decisions["group_id"]).issubset(
    set(candidates["group_id"])
):
    raise ValueError("Unknown MASSIVE group ID detected.")

# Attach decisions to the original candidate messages.
merged = candidates.merge(
    decisions,
    on="group_id",
    how="left",
    validate="many_to_one"
)

# Groups not selected for manual review remain unreviewed.
merged["review_decision"] = (
    merged["review_decision"].fillna("unreviewed")
)

# Verify that bilingual pairs remain together.
group_sizes = merged.groupby("group_id").size()

if not group_sizes.eq(2).all():
    raise ValueError("Incomplete bilingual groups detected.")

if merged.groupby("group_id")["split"].nunique().max() != 1:
    raise ValueError("Cross-split group leakage detected.")

output = Path("data/massive_ood_review_status.csv")

merged.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("OOS REVIEW DECISIONS MERGED")
print("=" * 55)

print("Total candidate messages:", len(merged))
print("Total bilingual groups:", merged["group_id"].nunique())

print("\nReview decisions by group:")
print(
    merged.drop_duplicates("group_id")[
        "review_decision"
    ].value_counts()
)

print("\nReview decisions by message:")
print(merged["review_decision"].value_counts())

print("\nSaved:", output)
