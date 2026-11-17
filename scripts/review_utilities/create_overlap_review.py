
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

seed = pd.read_csv("data/support_seed_grouped.csv")
inventory = pd.read_csv("data/scenario_inventory.csv")

def similarity(a, b):
    return SequenceMatcher(
        None,
        str(a).lower().strip(),
        str(b).lower().strip()
    ).ratio()

rows = []

for _, example in seed.iterrows():
    candidates = inventory[
        inventory["intent"] == example["intent"]
    ].copy()

    candidates["score"] = candidates[
        "scenario_description"
    ].apply(
        lambda description: similarity(
            example["text"], description
        )
    )

    best = candidates.sort_values(
        "score", ascending=False
    ).iloc[0]

    rows.append({
        "seed_group_id": example["group_id"],
        "language": example["language"],
        "seed_text": example["text"],
        "intent": example["intent"],
        "suggested_scenario_id": best["scenario_id"],
        "suggested_description":
            best["scenario_description"],
        "similarity": round(best["score"], 3),
        "review_decision": "pending",
        "final_scenario_id": "",
    })

review = pd.DataFrame(rows)

output = Path("data/overlap_review.csv")
output.parent.mkdir(exist_ok=True)

review.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("OVERLAP REVIEW SHEET CREATED")
print("=" * 55)
print("Rows:", len(review))
print("Pending reviews:",
      (review["review_decision"] == "pending").sum())

print("\nPreview:")
print(
    review[
        [
            "language",
            "seed_text",
            "suggested_scenario_id",
            "similarity"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved:", output)
