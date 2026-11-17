
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

SEED = 42
TEMPERATURE = 2.0
ALPHA = 0.3
EPOCHS = 3
BATCH_SIZE = 4
MAX_LENGTH = 64

TEACHER_PATH = "models/multilingual_bert"
STUDENT_NAME = "distilbert-base-multilingual-cased"
OUTPUT_PATH = "models/distilled_student"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

device = torch.device("cpu")

data = pd.read_csv(
    "data/model_dataset_verified.csv",
    encoding="utf-8-sig"
).fillna("")

labels = sorted(data["intent"].unique())
label_to_id = {
    label: i for i, label in enumerate(labels)
}

train_df = data[data["split"] == "train"].copy()
val_df = data[data["split"] == "validation"].copy()

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Labels:", labels)

teacher_tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_PATH
)
student_tokenizer = AutoTokenizer.from_pretrained(
    STUDENT_NAME
)

teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_PATH
).to(device)

student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_NAME,
    num_labels=len(labels),
    id2label={i: label for i, label in enumerate(labels)},
    label2id=label_to_id
).to(device)

teacher.eval()
student.train()

class TicketDataset(Dataset):
    def __init__(self, frame):
        self.texts = frame["text"].tolist()
        self.targets = [
            label_to_id[x] for x in frame["intent"]
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.targets[index]

def collate_batch(batch):
    texts, targets = zip(*batch)

    teacher_inputs = teacher_tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    student_inputs = student_tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return (
        teacher_inputs,
        student_inputs,
        torch.tensor(targets, dtype=torch.long)
    )

train_loader = DataLoader(
    TicketDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

val_loader = DataLoader(
    TicketDataset(val_df),
    batch_size=8,
    shuffle=False,
    collate_fn=collate_batch
)

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=2e-5,
    weight_decay=0.01
)

best_macro_f1 = -1.0
best_state = None
history = []

print("\nStarting knowledge distillation...")
start = time.perf_counter()

for epoch in range(1, EPOCHS + 1):
    student.train()
    running_loss = 0.0

    for teacher_inputs, student_inputs, targets in train_loader:
        teacher_inputs = {
            k: v.to(device)
            for k, v in teacher_inputs.items()
        }
        student_inputs = {
            k: v.to(device)
            for k, v in student_inputs.items()
        }
        targets = targets.to(device)

        with torch.no_grad():
            teacher_logits = teacher(
                **teacher_inputs
            ).logits

        student_logits = student(
            **student_inputs
        ).logits

        ce_loss = F.cross_entropy(
            student_logits,
            targets
        )

        kd_loss = F.kl_div(
            F.log_softmax(
                student_logits / TEMPERATURE,
                dim=-1
            ),
            F.softmax(
                teacher_logits / TEMPERATURE,
                dim=-1
            ),
            reduction="batchmean"
        ) * (TEMPERATURE ** 2)

        loss = (
            (1 - ALPHA) * ce_loss
            + ALPHA * kd_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            student.parameters(), 1.0
        )
        optimizer.step()

        running_loss += loss.item()

    student.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for _, student_inputs, targets in val_loader:
            student_inputs = {
                k: v.to(device)
                for k, v in student_inputs.items()
            }

            logits = student(
                **student_inputs
            ).logits

            predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )
            references.extend(targets.tolist())

    accuracy = accuracy_score(
        references, predictions
    )
    macro_f1 = f1_score(
        references,
        predictions,
        labels=list(range(len(labels))),
        average="macro",
        zero_division=0
    )

    average_loss = running_loss / len(train_loader)

    history.append({
        "epoch": epoch,
        "training_loss": average_loss,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1
    })

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train loss {average_loss:.4f} | "
        f"val accuracy {accuracy:.4f} | "
        f"val macro F1 {macro_f1:.4f}",
        flush=True
    )

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in student.state_dict().items()
        }

student.load_state_dict(best_state)
student.eval()

final_predictions = []
final_references = []

with torch.no_grad():
    for _, student_inputs, targets in val_loader:
        student_inputs = {
            k: v.to(device)
            for k, v in student_inputs.items()
        }

        logits = student(**student_inputs).logits

        final_predictions.extend(
            logits.argmax(dim=1).cpu().tolist()
        )
        final_references.extend(targets.tolist())

elapsed = time.perf_counter() - start

print("\nDISTILLED STUDENT RESULTS")
print("=" * 55)
print("Training time (seconds):", round(elapsed, 2))
print("Best validation macro F1:", round(best_macro_f1, 4))

print(
    classification_report(
        final_references,
        final_predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0
    )
)

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

student.save_pretrained(OUTPUT_PATH)
student_tokenizer.save_pretrained(OUTPUT_PATH)

pd.DataFrame(history).to_csv(
    "results/distillation_training_history.csv",
    index=False
)

pd.DataFrame({
    "text": val_df["text"].values,
    "language": val_df["language"].values,
    "true_intent": val_df["intent"].values,
    "predicted_intent": [
        labels[i] for i in final_predictions
    ]
}).to_csv(
    "results/distilled_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    "results/distillation_config.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump({
        "teacher": TEACHER_PATH,
        "student": STUDENT_NAME,
        "temperature": TEMPERATURE,
        "alpha": ALPHA,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "seed": SEED
    }, file, indent=2)

print("\nSaved distilled student:", OUTPUT_PATH)
print("Saved training history and predictions.")
