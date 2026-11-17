
from pathlib import Path
import pandas as pd

source = Path("data/ood_manual_review.csv")
output = Path("data/ood_remaining_88.csv")

df = pd.read_csv(source, dtype=str).fillna("")

pending = df[
    df["review_decision"] == "pending"
].copy()

if len(pending) != 88:
    raise ValueError(
        f"Expected 88 pending groups, found {len(pending)}"
    )

if pending["group_id"].duplicated().any():
    raise ValueError("Duplicate group IDs detected.")

pending.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("REMAINING OOS REVIEW FILE CREATED")
print("=" * 55)
print("Pending bilingual groups:", len(pending))

print("\nOriginal intent distribution:")
print(
    pending["original_intent"]
    .value_counts()
    .to_string()
)

print("\nSaved:", output)
