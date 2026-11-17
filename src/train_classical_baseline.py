
from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

data = pd.read_csv(
    "data/model_dataset_verified.csv",
    dtype=str,
    encoding="utf-8-sig"
)

train = data[data["split"] == "train"]
validation = data[data["split"] == "validation"]

# Word features capture meaningful terms.
# Character features help with Turkish suffixes
# and variations in appliance terminology.
features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True
        )
    ),
    (
        "character",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True
        )
    )
])

model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])

print("Training classical baseline...")

start = time.perf_counter()

model.fit(
    train["text"],
    train["intent"]
)

training_seconds = time.perf_counter() - start

predictions = model.predict(validation["text"])

accuracy = accuracy_score(
    validation["intent"],
    predictions
)

print("\nCLASSICAL BASELINE RESULTS")
print("=" * 55)
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Validation accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        validation["intent"],
        predictions,
        zero_division=0
    )
)

labels = sorted(data["intent"].unique())

print("\nConfusion matrix:")
print(
    pd.DataFrame(
        confusion_matrix(
            validation["intent"],
            predictions,
            labels=labels
        ),
        index=labels,
        columns=labels
    )
)

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/classical_tfidf_logreg.joblib"
)

pd.DataFrame({
    "text": validation["text"].values,
    "language": validation["language"].values,
    "true_intent": validation["intent"].values,
    "predicted_intent": predictions
}).to_csv(
    "results/classical_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nModel saved: models/classical_tfidf_logreg.joblib")
print(
    "Predictions saved: "
    "results/classical_validation_predictions.csv"
)
