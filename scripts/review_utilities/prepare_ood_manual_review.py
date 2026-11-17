
from pathlib import Path
import pandas as pd

source = Path("data/ood_review_with_suggestions.csv")
output = Path("data/ood_manual_review.csv")

df = pd.read_csv(source, dtype=str).fillna("")

required = {
    "group_id",
    "english_text",
    "turkish_text",
    "original_intent",
    "split",
    "suggested_decision",
    "review_decision",
}

if not required.issubset(df.columns):
    raise ValueError("Missing required review columns.")

columns = [
    "group_id",
    "split",
    "original_intent",
    "english_text",
    "turkish_text",
    "suggested_decision",
    "suggestion_reason",
    "review_decision",
    "review_note",
]

df = df[columns]

# Show the potentially difficult examples first.
df["review_priority"] = df["suggested_decision"].map({
    "review_carefully": 0,
    "likely_keep": 1,
}).fillna(2)

df = df.sort_values(
    ["review_priority", "original_intent", "group_id"]
).drop(columns=["review_priority"])

df.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("MANUAL OOS REVIEW FILE READY")
print("=" * 55)
print("Total bilingual groups:", len(df))
print(
    "Groups requiring careful review:",
    (df["suggested_decision"] == "review_carefully").sum()
)
print(
    "Groups suggested for keeping:",
    (df["suggested_decision"] == "likely_keep").sum()
)
print("\nSaved:", output)
print("\nNext: Open this CSV in Excel and inspect the messages.")
