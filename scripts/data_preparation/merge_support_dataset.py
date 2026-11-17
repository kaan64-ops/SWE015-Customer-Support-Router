
from pathlib import Path
import pandas as pd

template = pd.read_csv(
    "data/bilingual_scenario_template.csv",
    dtype=str
).fillna("")

missing = pd.read_csv(
    "data/missing_bilingual_examples.csv",
    dtype=str
).fillna("")

if len(template) != 50:
    raise ValueError("Expected exactly 50 scenarios.")

if len(missing) != 35:
    raise ValueError("Expected exactly 35 missing scenarios.")

if template["scenario_id"].duplicated().any():
    raise ValueError("Duplicate scenario IDs in template.")

if missing["scenario_id"].duplicated().any():
    raise ValueError("Duplicate scenario IDs in missing examples.")

missing_ids = set(missing["scenario_id"])
template_ids = set(template["scenario_id"])

if not missing_ids.issubset(template_ids):
    raise ValueError("Unknown scenario IDs in missing examples.")

missing_lookup = missing.set_index("scenario_id")

rows = []

for _, scenario in template.iterrows():
    scenario_id = scenario["scenario_id"]
    intent = scenario["intent"]

    english = scenario["english_seed"].strip()
    turkish = scenario["turkish_seed"].strip()

    if scenario_id in missing_ids:
        new_row = missing_lookup.loc[scenario_id]

        if not english:
            english = new_row["english_message"].strip()

        if not turkish:
            turkish = new_row["turkish_message"].strip()

    if not english or not turkish:
        raise ValueError(
            f"Missing bilingual message for {scenario_id}"
        )

    for language, message in [
        ("en", english),
        ("tr", turkish)
    ]:
        rows.append({
            "scenario_id": scenario_id,
            "intent": intent,
            "language": language,
            "text": message,
            "source": "manually_authored",
        })

dataset = pd.DataFrame(rows)

if len(dataset) != 100:
    raise ValueError("Expected exactly 100 messages.")

if dataset["text"].str.strip().eq("").any():
    raise ValueError("Empty messages detected.")

if dataset["text"].duplicated().any():
    raise ValueError("Duplicate message texts detected.")

if not dataset.groupby("scenario_id")["language"].apply(
    lambda x: set(x) == {"en", "tr"}
).all():
    raise ValueError("Incomplete bilingual scenario pairs.")

output = Path("data/support_bilingual_100.csv")

dataset.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("BILINGUAL SUPPORT DATASET MERGED")
print("=" * 55)
print("Total messages:", len(dataset))
print("Unique scenarios:", dataset["scenario_id"].nunique())

print("\nMessages by language:")
print(dataset["language"].value_counts().sort_index())

print("\nMessages by intent:")
print(dataset["intent"].value_counts().sort_index())

print("\nDuplicate texts:", dataset["text"].duplicated().sum())
print("Empty texts:", dataset["text"].str.strip().eq("").sum())

print("\nSaved:", output)
