
from pathlib import Path
import pandas as pd

original_path = Path("data/ood_remaining_88.csv")
reviewed_path = Path("data/ood_remaining_88_reviewed.csv")
output_path = Path("data/ood_remaining_88_recovered.csv")

# Original file preserves the correct bilingual text.
original = pd.read_csv(
    original_path,
    dtype=str,
    encoding="utf-8-sig"
).fillna("")

# Excel-exported file contains the manual decisions.
reviewed = pd.read_csv(
    reviewed_path,
    sep=None,
    engine="python",
    encoding="cp1254",
    dtype=str
).fillna("")

# Excel created generic column names and included
# the original header as the first data row.
if "Column1" not in reviewed.columns:
    raise ValueError("Unexpected Excel export structure.")

reviewed = reviewed[
    reviewed["Column1"] != "group_id"
].copy()

reviewed = reviewed.rename(columns={
    "Column1": "group_id"
})

# We only need the group ID and the manual decision.
decisions = reviewed[
    ["group_id", "review_decision"]
].copy()

decisions["group_id"] = decisions["group_id"].str.strip()
decisions["review_decision"] = (
    decisions["review_decision"].str.strip().str.lower()
)

if len(decisions) != 88:
    raise ValueError(
        f"Expected 88 decisions, found {len(decisions)}"
    )

if decisions["group_id"].duplicated().any():
    raise ValueError("Duplicate group IDs detected.")

if not decisions["review_decision"].isin(
    ["keep", "exclude", "pending"]
).all():
    raise ValueError("Invalid review decisions detected.")

if set(decisions["group_id"]) != set(original["group_id"]):
    raise ValueError(
        "Reviewed group IDs do not match the original file."
    )

# Remove the original pending decisions before merging.
recovered = original.drop(
    columns=["review_decision"]
).merge(
    decisions,
    on="group_id",
    how="left",
    validate="one_to_one"
)

# Restore the original column order.
recovered = recovered[original.columns]

if recovered["review_decision"].isna().any():
    raise ValueError("Some decisions were not recovered.")

# Confirm the original bilingual text is unchanged.
reference = original.set_index("group_id")
result = recovered.set_index("group_id")

for column in ["english_text", "turkish_text"]:
    if not reference[column].sort_index().equals(
        result[column].sort_index()
    ):
        raise ValueError(
            f"Original {column} was modified."
        )

recovered.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("OOS REVIEW RECOVERY SUCCESSFUL")
print("=" * 55)
print("Recovered groups:", len(recovered))

print("\nReview decisions:")
print(recovered["review_decision"].value_counts())

print("\nOriginal bilingual texts: PRESERVED")
print("Group IDs: VERIFIED")
print("Data integrity: PASSED")

print("\nSaved:", output_path)
