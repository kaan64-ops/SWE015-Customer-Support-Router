
from pathlib import Path
import pandas as pd

path = Path("data/massive_ood_candidates.csv")

df = pd.read_csv(path, dtype=str).fillna("")

print("OUT-OF-SCOPE CANDIDATE INSPECTION")
print("=" * 60)

print("\nTotal rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nMissing values in key fields:")
for column in ["text", "language", "label_text"]:
    if column in df.columns:
        print(
            column,
            "empty:",
            df[column].str.strip().eq("").sum()
        )

if "language" in df.columns:
    print("\nLanguage distribution:")
    print(df["language"].value_counts())

if "split" in df.columns:
    print("\nSplit distribution:")
    print(df["split"].value_counts())

if "label_text" in df.columns:
    print("\nTop 20 original MASSIVE intents:")
    print(df["label_text"].value_counts().head(20))

print("\nExact duplicate messages:",
      df["text"].duplicated().sum()
      if "text" in df.columns else "N/A")
