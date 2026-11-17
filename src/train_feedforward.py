
from pathlib import Path
import copy
import random
import re
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).casefold())

# Load compact aligned embeddings.
archive = np.load(
    "embeddings/bilingual_compact_vectors.npz"
)

vector_lookup = dict(zip(
    archive["tokens"].tolist(),
    archive["vectors"]
))

data = pd.read_csv(
    "data/model_dataset_verified.csv",
    encoding="utf-8-sig"
).fillna("")

labels = sorted(data["intent"].unique())
label_to_id = {label: i for i, label in enumerate(labels)}

def sentence_vector(text, language):
    vectors = [
        vector_lookup[f"{language}:{token}"]
        for token in tokenize(text)
        if f"{language}:{token}" in vector_lookup
    ]

    if not vectors:
        return np.zeros(300, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)

def prepare_split(split):
    subset = data[data["split"] == split].copy()

    X = np.stack([
        sentence_vector(row.text, row.language)
        for row in subset.itertuples()
    ])

    y = np.array([
        label_to_id[label]
        for label in subset["intent"]
    ], dtype=np.int64)

    return subset, X, y

train_df, X_train, y_train = prepare_split("train")
val_df, X_val, y_val = prepare_split("validation")

# Standardize using training data only.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std = np.maximum(std, 1e-6)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

class FeedforwardClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, classes)
        )

    def forward(self, x):
        return self.network(x)

model = FeedforwardClassifier(classes=len(labels))

# Weight classes to reduce OOS dominance.
counts = np.bincount(
    y_train,
    minlength=len(labels)
)

weights = len(y_train) / (len(labels) * counts)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32)
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4
)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

best_loss = float("inf")
best_state = None
patience = 15
epochs_without_improvement = 0

history = []
start = time.perf_counter()

for epoch in range(1, 151):
    model.train()
    total_loss = 0.0

    for features, targets in loader:
        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)

    train_loss = total_loss / len(train_dataset)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_loss = criterion(
            val_logits,
            torch.tensor(y_val, dtype=torch.long)
        ).item()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss
    })

    if val_loss < best_loss - 1e-4:
        best_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} | "
            f"validation loss {val_loss:.4f}"
        )

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    logits = model(X_val_tensor)
    predictions = logits.argmax(dim=1).numpy()

predicted_labels = [labels[i] for i in predictions]

print("\nFEEDFORWARD NEURAL NETWORK RESULTS")
print("=" * 55)
print(
    "Training time (seconds):",
    round(time.perf_counter() - start, 2)
)
print(
    "Validation accuracy:",
    round(accuracy_score(y_val, predictions), 4)
)
print("\nClassification report:")
print(
    classification_report(
        val_df["intent"],
        predicted_labels,
        labels=labels,
        zero_division=0
    )
)

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

torch.save({
    "model_state": model.state_dict(),
    "labels": labels,
    "input_dim": 300,
    "hidden_dim": 128
}, "models/feedforward_fasttext.pt")

joblib.dump({
    "mean": mean,
    "std": std,
    "labels": labels
}, "models/feedforward_preprocessing.joblib")

pd.DataFrame(history).to_csv(
    "results/feedforward_training_history.csv",
    index=False
)

pd.DataFrame({
    "text": val_df["text"].values,
    "language": val_df["language"].values,
    "true_intent": val_df["intent"].values,
    "predicted_intent": predicted_labels
}).to_csv(
    "results/feedforward_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved model and preprocessing files.")
print("Saved training history and validation predictions.")
