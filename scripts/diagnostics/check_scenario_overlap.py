
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

inventory = pd.read_csv(
    "data/scenario_inventory.csv"
)

seed = pd.read_csv(
    "data/support_seed_grouped.csv"
)

def similarity(a, b):
    return SequenceMatcher(
        None,
        str(a).lower().strip(),
        str(b).lower().strip()
    ).ratio()

results = []

for _, scenario in inventory.iterrows():
    for _, example in seed.iterrows():

        if scenario["intent"] != example["intent"]:
            continue

        score = similarity(
            scenario["scenario_description"],
            example["text"]
        )

        results.append({
            "scenario_id": scenario["scenario_id"],
            "scenario_description":
                scenario["scenario_description"],
            "seed_text": example["text"],
            "seed_language": example["language"],
            "seed_group_id": example["group_id"],
            "similarity": round(score, 3),
        })

report = pd.DataFrame(results)

report = report.sort_values(
    "similarity",
    ascending=False
)

output = Path("results/scenario_overlap_report.csv")
output.parent.mkdir(exist_ok=True)

report.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("SCENARIO OVERLAP INSPECTION")
print("=" * 60)

print("\nTop 15 possible overlaps:\n")

print(
    report[
        [
            "scenario_id",
            "seed_text",
            "similarity"
        ]
    ].head(15).to_string(index=False)
)

print("\nReport saved:", output)

print(
    "\nNote: String similarity is only a screening tool."
    "\nSemantic overlap still requires manual review."
)
