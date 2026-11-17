
from pathlib import Path
import pandas as pd

path = Path("data/ood_review.csv")

df = pd.read_csv(path, dtype=str).fillna("")

# These are review suggestions, NOT approved labels.
review_carefully = {
    "iot_cleaning",
    "iot_coffee",
    "iot_wemo_off",
    "takeaway_order",
    "takeaway_query",
}

for index, row in df.iterrows():
    intent = row["original_intent"]

    if intent in review_carefully:
        suggestion = "review_carefully"
        reason = (
            "Contains appliance, operation, or delivery-related "
            "vocabulary; verify that it is not a support request."
        )
    elif intent.startswith("iot_"):
        suggestion = "likely_keep"
        reason = (
            "Appears to be a smart-home command rather than "
            "a customer-support ticket."
        )
    elif intent.startswith("lists_"):
        suggestion = "likely_keep"
        reason = (
            "Appears to concern personal list management."
        )
    elif intent.startswith("recommendation_"):
        suggestion = "likely_keep"
        reason = (
            "Appears to request recommendations rather "
            "than appliance support."
        )
    else:
        suggestion = "review_carefully"
        reason = "Check the message meaning manually."

    df.loc[index, "suggested_decision"] = suggestion
    df.loc[index, "suggestion_reason"] = reason

# Preserve review_decision as pending.
# Suggested decisions must not be treated as approvals.

output = Path("data/ood_review_with_suggestions.csv")

df.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("OOS REVIEW SUGGESTIONS CREATED")
print("=" * 55)

print("Total groups:", len(df))

print("\nSuggestion distribution:")
print(df["suggested_decision"].value_counts())

print("\nActual review decisions:")
print(df["review_decision"].value_counts())

print("\nSaved:", output)
