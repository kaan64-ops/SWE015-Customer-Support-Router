
from pathlib import Path
import pandas as pd

MODEL_FILES = {
    "rule_based": [
        "src/train_rule_baseline.py"
    ],
    "tfidf_logreg": [
        "models/classical_tfidf_logreg.joblib"
    ],
    "feedforward_fasttext": [
        "models/feedforward_fasttext.pt",
        "models/feedforward_preprocessing.joblib",
        "embeddings/bilingual_compact_vectors.npz"
    ],
    "multilingual_bert": [
        "models/multilingual_bert"
    ],
    "distilled_student": [
        "models/distilled_student"
    ],
    "distilled_student_int8": [
        "models/distilled_student_int8"
    ]
}

def get_size(path_string):
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        return path.stat().st_size

    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )

rows = []

for model_name, paths in MODEL_FILES.items():
    total_bytes = sum(get_size(p) for p in paths)

    rows.append({
        "model": model_name,
        "size_mb": round(total_bytes / (1024 ** 2), 3),
        "size_bytes": total_bytes
    })

results = pd.DataFrame(rows)

Path("results").mkdir(exist_ok=True)

results.to_csv(
    "results/model_sizes.csv",
    index=False
)

print("\nMODEL STORAGE COMPARISON")
print("=" * 60)
print(results.to_string(index=False))

original = results.loc[
    results["model"] == "distilled_student",
    "size_bytes"
].iloc[0]

quantized = results.loc[
    results["model"] == "distilled_student_int8",
    "size_bytes"
].iloc[0]

reduction = (original - quantized) / original * 100

print(f"\nINT8 storage reduction: {reduction:.2f}%")
print("\nSaved: results/model_sizes.csv")
