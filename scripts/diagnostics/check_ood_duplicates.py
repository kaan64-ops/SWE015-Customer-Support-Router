
from pathlib import Path
import pandas as pd

source = Path("data/massive_ood_review_status.csv")
output = Path("results/ood_duplicate_audit.csv")

df = pd.read_csv(source, dtype=str).fillna("")

required = {
    "group_id", "split", "language", "text",
    "original_intent", "review_decision"
}

if not required.issubset(df.columns):
    raise ValueError("Required columns are missing.")

# Normalize text for duplicate detection.
df["normalized_text"] = (
    df["text"]
    .str.casefold()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Compare duplicates only within the same language.
duplicate_key = ["language", "normalized_text"]

duplicates = df[
    df.duplicated(subset=duplicate_key, keep=False)
].copy()

duplicates = duplicates.sort_values(
    ["language", "normalized_text", "group_id"]
)

# Identify duplicate texts appearing in multiple splits.
cross_split = (
    df.groupby(duplicate_key)["split"]
    .nunique()
)

cross_split_count = int((cross_split > 1).sum())

output.parent.mkdir(parents=True, exist_ok=True)

duplicates.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("OOS DUPLICATE AUDIT COMPLETE")
print("=" * 55)

print("Total messages:", len(df))
print("Duplicate message rows:", len(duplicates))
print("Duplicate text keys:", duplicates.groupby(
    duplicate_key
).ngroups if len(duplicates) else 0)

print("Cross-split duplicate keys:", cross_split_count)

print("\nDuplicate details:")

if duplicates.empty:
    print("No exact normalized duplicates found.")
else:
    print(
        duplicates[
            [
                "group_id", "split", "language",
                "text", "review_decision"
            ]
        ].to_string(index=False)
    )

print("\nSaved:", output)
