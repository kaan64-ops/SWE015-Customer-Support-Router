
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

torch.set_num_threads(4)
Path("results").mkdir(exist_ok=True)

DATA_PATH = "data/model_dataset_verified.csv"
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

data = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig"
).fillna("")

test = data[data["split"] == "test"].copy()
labels = sorted(data["intent"].unique())

texts = test["text"].tolist()
languages = test["language"].tolist()
truth = test["intent"].tolist()

print("FINAL HELD-OUT TEST")
print("=" * 60)
print("Test messages:", len(test))
print("Labels:", labels)
print("Languages:", test["language"].value_counts().to_dict())
print("Test intent distribution:")
print(test["intent"].value_counts().to_string())

assert len(test) == 58, "Unexpected test-set size."
assert len(set(labels)) == 6

all_predictions = {}
summary = []

def evaluate(name, predictions):
    predictions = list(predictions)

    assert len(predictions) == len(test)
    assert set(predictions).issubset(set(labels))

    accuracy = accuracy_score(truth, predictions)
    macro_f1 = f1_score(
        truth,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0
    )

    support_indices = [
        i for i, label in enumerate(truth)
        if label != "out_of_scope"
    ]
    oos_indices = [
        i for i, label in enumerate(truth)
        if label == "out_of_scope"
    ]

    support_accuracy = np.mean([
        predictions[i] == truth[i]
        for i in support_indices
    ])

    oos_recall = np.mean([
        predictions[i] == "out_of_scope"
        for i in oos_indices
    ])

    en_indices = [
        i for i, language in enumerate(languages)
        if language == "en"
    ]
    tr_indices = [
        i for i, language in enumerate(languages)
        if language == "tr"
    ]

    en_accuracy = np.mean([
        predictions[i] == truth[i]
        for i in en_indices
    ])
    tr_accuracy = np.mean([
        predictions[i] == truth[i]
        for i in tr_indices
    ])

    metrics = {
        "model": name,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "in_scope_accuracy": round(
            float(support_accuracy), 4
        ),
        "oos_recall": round(float(oos_recall), 4),
        "english_accuracy": round(
            float(en_accuracy), 4
        ),
        "turkish_accuracy": round(
            float(tr_accuracy), 4
        )
    }

    summary.append(metrics)
    all_predictions[name] = predictions

    print(f"\n{name}")
    print("-" * 60)
    print(json.dumps(metrics, indent=2))

    print(
        classification_report(
            truth,
            predictions,
            labels=labels,
            zero_division=0
        )
    )

# --------------------------------------------------
# 1. TF-IDF + Logistic Regression
# --------------------------------------------------

print("\nLoading TF-IDF classifier...")

classical = joblib.load(
    "models/classical_tfidf_logreg.joblib"
)

classical_predictions = classical.predict(texts)
evaluate("tfidf_logreg", classical_predictions)

# --------------------------------------------------
# 2. Feedforward neural network
# --------------------------------------------------

print("\nLoading feedforward network...")

checkpoint = torch.load(
    "models/feedforward_fasttext.pt",
    map_location="cpu",
    weights_only=True
)

preprocessing = joblib.load(
    "models/feedforward_preprocessing.joblib"
)

archive = np.load(
    "embeddings/bilingual_compact_vectors.npz"
)

vector_lookup = dict(zip(
    archive["tokens"].tolist(),
    archive["vectors"]
))

class FeedforwardClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.network(x)

feedforward = FeedforwardClassifier(
    checkpoint["input_dim"],
    checkpoint["hidden_dim"],
    len(checkpoint["labels"])
)

feedforward.load_state_dict(
    checkpoint["model_state"]
)
feedforward.eval()

def sentence_vector(text, language):
    tokens = TOKEN_PATTERN.findall(str(text).casefold())

    vectors = [
        vector_lookup[f"{language}:{token}"]
        for token in tokens
        if f"{language}:{token}" in vector_lookup
    ]

    if vectors:
        return np.mean(vectors, axis=0)

    return np.zeros(
        checkpoint["input_dim"],
        dtype=np.float32
    )

features = np.vstack([
    sentence_vector(text, language)
    for text, language in zip(texts, languages)
])

features = (
    features - preprocessing["mean"]
) / preprocessing["std"]

with torch.no_grad():
    logits = feedforward(
        torch.tensor(features, dtype=torch.float32)
    )

    predicted_ids = logits.argmax(dim=1).tolist()

feedforward_labels = checkpoint["labels"]

feedforward_predictions = [
    feedforward_labels[i] for i in predicted_ids
]

evaluate("feedforward_fasttext", feedforward_predictions)

# --------------------------------------------------
# Transformer evaluation helper
# --------------------------------------------------

def transformer_predictions(
    model_path,
    quantized=False
):
    tokenizer_path = (
        "models/distilled_student"
        if quantized else model_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path
    )

    if quantized:
        # Load only our own trusted local model file.
        model = torch.load(
            "models/distilled_student_int8/quantized_model.pt",
            map_location="cpu",
            weights_only=False
        )
    else:
        model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_path)
        )

    model.eval()

    model_labels = [
        model.config.id2label[i]
        for i in range(model.config.num_labels)
    ]

    predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), 8):
            batch = texts[start:start + 8]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )

            logits = model(**inputs).logits
            ids = logits.argmax(dim=1).tolist()

            predictions.extend(
                model_labels[i] for i in ids
            )

    del model
    return predictions

# --------------------------------------------------
# 3. Fine-tuned multilingual BERT
# --------------------------------------------------

print("\nEvaluating multilingual BERT...")

bert_predictions = transformer_predictions(
    "models/multilingual_bert"
)

evaluate("multilingual_bert", bert_predictions)

# --------------------------------------------------
# 4. Knowledge-distilled student
# --------------------------------------------------

print("\nEvaluating distilled student...")

student_predictions = transformer_predictions(
    "models/distilled_student"
)

evaluate("distilled_student", student_predictions)

# --------------------------------------------------
# 5. INT8 quantized distilled student
# --------------------------------------------------

print("\nEvaluating quantized student...")

int8_predictions = transformer_predictions(
    "models/distilled_student_int8",
    quantized=True
)

evaluate("distilled_student_int8", int8_predictions)

# --------------------------------------------------
# Save reproducible evaluation outputs
# --------------------------------------------------

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "results/final_test_metrics.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "text": texts,
    "language": languages,
    "true_intent": truth,
    **all_predictions
})

predictions_df.to_csv(
    "results/final_test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFINAL TEST SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

print("\nSaved:")
print("results/final_test_metrics.csv")
print("results/final_test_predictions.csv")
print("\nDo not tune models on this test set.")
