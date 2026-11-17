
from pathlib import Path
import pandas as pd

INPUT = Path("data/support_seed.csv")
OUTPUT = Path("data/support_seed_grouped.csv")

df = pd.read_csv(INPUT)

required = {"text", "intent", "language"}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

if len(df) != 30:
    raise ValueError(
        f"Expected 30 seed examples, found {len(df)}"
    )

# These are the five support categories.
intents = [
    "technical_issue",
    "repair_request",
    "warranty_question",
    "delivery_question",
    "refund_request",
]

# Group related examples by intent and position.
# This is only a preliminary grouping, not a semantic
# duplicate detector. We will review the groups later.
df["group_id"] = ""

for intent in intents:
    en_indices = df.index[
        (df["intent"] == intent)
        & (df["language"] == "en")
    ].tolist()

    tr_indices = df.index[
        (df["intent"] == intent)
        & (df["language"] == "tr")
    ].tolist()

    if len(en_indices) != 3 or len(tr_indices) != 3:
        raise ValueError(
            f"Unexpected count for {intent}"
        )

    for number, (en_idx, tr_idx) in enumerate(
        zip(en_indices, tr_indices), start=1
    ):
        group = f"support_{intent}_{number:03d}"

        df.loc[en_idx, "group_id"] = group
        df.loc[tr_idx, "group_id"] = group

df["split"] = "unassigned"

OUTPUT.parent.mkdir(exist_ok=True)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("Grouped support seed created!")
print("\nTotal messages:", len(df))
print("Unique groups:", df["group_id"].nunique())

print("\nExamples per language:")
print(df["language"].value_counts())

print("\nFirst six grouped examples:")
print(
    df[
        ["text", "intent", "language", "group_id"]
    ].head(6).to_string(index=False)
)

print("\nSaved:", OUTPUT)
