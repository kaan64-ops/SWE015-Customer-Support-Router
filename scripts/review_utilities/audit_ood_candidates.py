
from pathlib import Path
import pandas as pd

path = Path("data/massive_ood_candidates.csv")
df = pd.read_csv(path, dtype=str).fillna("")

required = {
    "text", "intent", "language", "source",
    "original_intent", "split", "group_id"
}

if not required.issubset(df.columns):
    raise ValueError("Required columns are missing.")

print("MASSIVE OOS DATASET AUDIT")
print("=" * 60)

print("\nOriginal MASSIVE intents:")
print(
    df["original_intent"]
    .value_counts()
    .to_string()
)

print("\nBilingual group sizes:")
print(
    df.groupby("group_id")
    .size()
    .value_counts()
    .sort_index()
)

print("\nGroups appearing in multiple splits:")
leaking_groups = (
    df.groupby("group_id")["split"]
    .nunique()
)

print((leaking_groups > 1).sum())

print("\nGroups missing an English or Turkish message:")
languages = df.groupby("group_id")["language"].apply(set)
print((languages != {"en", "tr"}).sum())

print("\nExact duplicate messages:")
duplicates = df[
    df.duplicated(subset=["text"], keep=False)
].sort_values("text")

if duplicates.empty:
    print("None")
else:
    print(
        duplicates[
            ["text", "language", "original_intent",
             "split", "group_id"]
        ].to_string(index=False)
    )

print("\nPotentially ambiguous original intents:")

keywords = [
    "iot",
    "cleaning",
    "shopping",
    "order",
    "lists",
    "recommendation"
]

mask = df["original_intent"].str.lower().apply(
    lambda value: any(
        keyword in value for keyword in keywords
    )
)

ambiguous = df[mask]

print("Candidate rows:", len(ambiguous))
print("Candidate groups:", ambiguous["group_id"].nunique())

print("\nExamples:")
print(
    ambiguous[
        ["original_intent", "language", "text"]
    ].head(20).to_string(index=False)
)

output = Path("results/ood_audit_by_intent.csv")

df.groupby("original_intent").agg(
    messages=("text", "size"),
    groups=("group_id", "nunique")
).reset_index().to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", output)
