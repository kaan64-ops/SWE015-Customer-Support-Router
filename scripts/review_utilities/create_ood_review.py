
from pathlib import Path
import pandas as pd

df = pd.read_csv(
    "data/massive_ood_candidates.csv",
    dtype=str
).fillna("")

keywords = [
    "iot",
    "cleaning",
    "shopping",
    "order",
    "lists",
    "recommendation",
    "takeaway"
]

ambiguous_mask = df["original_intent"].str.lower().apply(
    lambda value: any(
        keyword in value for keyword in keywords
    )
)

duplicate_mask = df.duplicated(
    subset=["text"],
    keep=False
)

review_groups = set(
    df.loc[ambiguous_mask, "group_id"]
) | set(
    df.loc[duplicate_mask, "group_id"]
)

selected = df[
    df["group_id"].isin(review_groups)
].copy()

english = selected[
    selected["language"] == "en"
][
    ["group_id", "text", "original_intent", "split"]
].rename(columns={"text": "english_text"})

turkish = selected[
    selected["language"] == "tr"
][
    ["group_id", "text"]
].rename(columns={"text": "turkish_text"})

review = english.merge(
    turkish,
    on="group_id",
    how="inner",
    validate="one_to_one"
)

review["review_decision"] = "pending"
review["review_note"] = ""

review = review.sort_values(
    ["original_intent", "group_id"]
).reset_index(drop=True)

output = Path("data/ood_review.csv")

review.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("OOS REVIEW FILE CREATED")
print("=" * 55)

print("Groups selected for review:", len(review))

print("\nGroups by original intent:")
print(
    review["original_intent"]
    .value_counts()
    .to_string()
)

print("\nFirst 10 review rows:")
print(
    review.head(10).to_string(index=False)
)

print("\nSaved:", output)
print("\nAllowed review decisions:")
print("keep")
print("exclude")
print("pending")
