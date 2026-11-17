
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = Path("data/support_bilingual_100.csv")
OUTPUT = Path("data/support_split_100.csv")
MANIFEST = Path("results/support_split_manifest.csv")

df = pd.read_csv(INPUT, dtype=str).fillna("")

required = {"scenario_id", "intent", "language", "text"}

if not required.issubset(df.columns):
    raise ValueError("Required dataset columns are missing.")

scenario_info = (
    df[["scenario_id", "intent"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

if scenario_info["scenario_id"].duplicated().any():
    raise ValueError(
        "A scenario has more than one intent label."
    )

if len(scenario_info) != 50:
    raise ValueError("Expected 50 unique scenarios.")

# First reserve 20% of scenarios for the test set.
train_val, test = train_test_split(
    scenario_info,
    test_size=0.20,
    random_state=42,
    stratify=scenario_info["intent"]
)

# Of the remaining 40 scenarios, reserve 10 for validation.
train, validation = train_test_split(
    train_val,
    test_size=0.25,
    random_state=42,
    stratify=train_val["intent"]
)

train = train.copy()
validation = validation.copy()
test = test.copy()

train["split"] = "train"
validation["split"] = "validation"
test["split"] = "test"

manifest = pd.concat(
    [train, validation, test],
    ignore_index=True
)

df = df.merge(
    manifest[["scenario_id", "split"]],
    on="scenario_id",
    how="left",
    validate="many_to_one"
)

if df["split"].isna().any():
    raise ValueError("Some scenarios were not assigned a split.")

# Ensure each scenario belongs to exactly one split.
if df.groupby("scenario_id")["split"].nunique().max() != 1:
    raise ValueError("Scenario leakage detected.")

# Ensure both translations remain in the same split.
if not df.groupby("scenario_id")["language"].apply(
    lambda x: set(x) == {"en", "tr"}
).all():
    raise ValueError("Incomplete bilingual scenario detected.")

expected = {
    "train": 60,
    "validation": 20,
    "test": 20
}

if df["split"].value_counts().to_dict() != expected:
    raise ValueError("Unexpected split sizes.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")

print("SCENARIO-LEVEL DATASET SPLIT CREATED")
print("=" * 55)

print("\nMessages by split:")
print(df["split"].value_counts().sort_index())

print("\nIntent distribution:")
print(pd.crosstab(df["intent"], df["split"]))

print("\nLanguage distribution:")
print(pd.crosstab(df["language"], df["split"]))

print("\nScenario leakage:",
      df.groupby("scenario_id")["split"].nunique().gt(1).sum())

print("\nSaved:", OUTPUT)
print("Saved:", MANIFEST)
