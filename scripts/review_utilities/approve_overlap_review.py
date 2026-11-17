
import pandas as pd
from pathlib import Path

input_path = Path("data/overlap_review.csv")
output_path = Path("data/overlap_review_approved.csv")

df = pd.read_csv(input_path)
df["final_scenario_id"] = df["final_scenario_id"].astype("string")
df["review_decision"] = df["review_decision"].astype("string")

# Each list follows the existing three English/Turkish
# seed messages in their original order.
corrected = {
    "technical_issue": [1, 2, 3],
    "repair_request": [3, 2, 9],
    "warranty_question": [1, 2, 3],
    "delivery_question": [1, 2, 3],
    "refund_request": [1, 2, 3],
}

for intent, scenario_numbers in corrected.items():
    for language in ["en", "tr"]:
        mask = (
            (df["intent"] == intent)
            & (df["language"] == language)
        )

        indices = df.index[mask].tolist()

        if len(indices) != 3:
            raise ValueError(
                f"Expected 3 examples for {intent}/{language}"
            )

        for index, number in zip(indices, scenario_numbers):
            scenario_id = f"{intent}_{number:03d}"

            df.loc[index, "final_scenario_id"] = scenario_id
            df.loc[index, "review_decision"] = "approved"

# Check all messages have approved mappings.
assert len(df) == 30
assert (df["review_decision"] == "approved").all()
assert df["final_scenario_id"].notna().all()

# Check that all referenced scenarios exist.
inventory = pd.read_csv("data/scenario_inventory.csv")
valid_ids = set(inventory["scenario_id"])

assert set(df["final_scenario_id"]).issubset(valid_ids)

# Check paired English/Turkish messages map together.
for group_id, group in df.groupby("seed_group_id"):
    assert group["final_scenario_id"].nunique() == 1, (
        f"Language mismatch in {group_id}"
    )

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("MANUAL REVIEW MAPPINGS SAVED")
print("=" * 50)
print("Total reviewed:", len(df))
print("Approved:", (df["review_decision"] == "approved").sum())
print("Unique scenarios:", df["final_scenario_id"].nunique())

print("\nCorrected mappings:")
print(
    df[
        [
            "language",
            "seed_text",
            "final_scenario_id"
        ]
    ].to_string(index=False)
)

print("\nSaved:", output_path)
