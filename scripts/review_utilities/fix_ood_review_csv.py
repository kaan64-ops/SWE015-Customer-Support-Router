
from pathlib import Path
import shutil
import pandas as pd

path = Path("data/ood_manual_review.csv")
backup = Path("data/ood_manual_review_semicolon_backup.csv")

# Read the semicolon-separated file saved by Excel.
df = pd.read_csv(
    path,
    sep=";",
    dtype=str
).fillna("")

expected_columns = [
    "group_id",
    "split",
    "original_intent",
    "english_text",
    "turkish_text",
    "suggested_decision",
    "suggestion_reason",
    "review_decision",
    "review_note",
]

if df.columns.tolist() != expected_columns:
    raise ValueError("Unexpected CSV column structure.")

if len(df) != 121:
    raise ValueError("Unexpected number of review groups.")

allowed = {"keep", "exclude", "pending"}

if not set(df["review_decision"]).issubset(allowed):
    raise ValueError("Unexpected review decision found.")

# Preserve the original file before changing it.
if backup.exists():
    print("Existing backup found. Keeping it unchanged.")
else:
    shutil.copy2(path, backup)
    print("Backup created successfully.")

# Save as standard comma-separated CSV.
df.to_csv(
    path,
    index=False,
    encoding="utf-8-sig",
    sep=","
)

# Verify the converted file.
verified = pd.read_csv(
    path,
    dtype=str
).fillna("")

if not verified.equals(df):
    raise ValueError("Converted CSV does not match original data.")

print("OOS REVIEW CSV CONVERTED SUCCESSFULLY")
print("=" * 55)
print("Rows:", len(verified))
print("Columns:", len(verified.columns))

print("\nReview decisions:")
print(verified["review_decision"].value_counts())

print("\nBackup:", backup)
print("Converted file:", path)

print("\nData integrity check: PASSED")
