
from pathlib import Path
import pandas as pd

inventory = pd.read_csv("data/scenario_inventory.csv")

approved = pd.read_csv(
    "data/overlap_review_approved.csv"
)

# Ensure each scenario has at most one seed
# example per language.
duplicates = approved.duplicated(
    subset=["final_scenario_id", "language"]
)

if duplicates.any():
    raise ValueError(
        "Duplicate seed mappings found. "
        "Review them before continuing."
    )

english = approved[
    approved["language"] == "en"
][["final_scenario_id", "seed_text"]].rename(
    columns={"seed_text": "english_seed"}
)

turkish = approved[
    approved["language"] == "tr"
][["final_scenario_id", "seed_text"]].rename(
    columns={"seed_text": "turkish_seed"}
)

template = inventory.merge(
    english,
    left_on="scenario_id",
    right_on="final_scenario_id",
    how="left"
).drop(columns=["final_scenario_id"])

template = template.merge(
    turkish,
    left_on="scenario_id",
    right_on="final_scenario_id",
    how="left"
).drop(columns=["final_scenario_id"])

template["english_seed"] = (
    template["english_seed"].fillna("")
)

template["turkish_seed"] = (
    template["turkish_seed"].fillna("")
)

template["review_status"] = "pending"

output = Path("data/bilingual_scenario_template.csv")

template.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("BILINGUAL SCENARIO TEMPLATE CREATED")
print("=" * 55)

print("Total scenarios:", len(template))

print(
    "Scenarios with English seeds:",
    (template["english_seed"] != "").sum()
)

print(
    "Scenarios with Turkish seeds:",
    (template["turkish_seed"] != "").sum()
)

print(
    "Scenarios needing new examples:",
    (
        (template["english_seed"] == "")
        | (template["turkish_seed"] == "")
    ).sum()
)

print("\nSaved:", output)
