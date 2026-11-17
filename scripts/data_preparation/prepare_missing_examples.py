
from pathlib import Path
import pandas as pd

template = pd.read_csv(
    "data/bilingual_scenario_template.csv"
).fillna("")

missing = template[
    (template["english_seed"] == "")
    | (template["turkish_seed"] == "")
].copy()

output = missing[
    ["scenario_id", "intent", "scenario_description"]
].copy()

output["english_message"] = ""
output["turkish_message"] = ""
output["review_status"] = "pending"
output["source"] = "manually_authored"

path = Path("data/missing_bilingual_examples.csv")

output.to_csv(
    path,
    index=False,
    encoding="utf-8-sig"
)

print("MISSING BILINGUAL EXAMPLES TEMPLATE")
print("=" * 55)

print("Scenarios needing examples:", len(output))

print("\nBy intent:")
print(output["intent"].value_counts().sort_index())

print("\nSaved:", path)
