
from pathlib import Path
import inspect
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

SEED = 42
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "models/multilingual_bert_checkpoints"
FINAL_DIR = "models/multilingual_bert"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

data = pd.read_csv(
    "data/model_dataset_verified.csv",
    encoding="utf-8-sig"
).fillna("")

labels = sorted(data["intent"].unique())
label_to_id = {
    label: index for index, label in enumerate(labels)
}

train_df = data[data["split"] == "train"].copy()
val_df = data[data["split"] == "validation"].copy()

print("Training messages:", len(train_df))
print("Validation messages:", len(val_df))
print("Labels:", labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def make_dataset(frame):
    dataset = Dataset.from_dict({
        "text": frame["text"].tolist(),
        "labels": [
            label_to_id[x] for x in frame["intent"]
        ]
    })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=64
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
    )

    return dataset

train_dataset = make_dataset(train_df)
val_dataset = make_dataset(val_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label={i: x for i, x in enumerate(labels)},
    label2id=label_to_id
)

def compute_metrics(prediction):
    logits, references = prediction
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(
            references, predictions
        ),
        "macro_f1": f1_score(
            references,
            predictions,
            labels=list(range(len(labels))),
            average="macro",
            zero_division=0
        )
    }

# Transformers versions may differ in argument names.
args_parameters = inspect.signature(
    TrainingArguments.__init__
).parameters

arguments = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,
    seed=SEED,
    use_cpu=True
)

if "eval_strategy" in args_parameters:
    arguments["eval_strategy"] = "epoch"
elif "evaluation_strategy" in args_parameters:
    arguments["evaluation_strategy"] = "epoch"
else:
    raise RuntimeError(
        "No supported evaluation strategy argument."
    )

training_args = TrainingArguments(**arguments)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print("\nStarting CPU fine-tuning...")
start = time.perf_counter()

trainer.train()

elapsed = time.perf_counter() - start

pred_output = trainer.predict(val_dataset)
pred_ids = np.argmax(pred_output.predictions, axis=1)
pred_labels = [labels[i] for i in pred_ids]

print("\nMULTILINGUAL BERT VALIDATION RESULTS")
print("=" * 55)
print("Training time (seconds):", round(elapsed, 2))
print(
    "Validation accuracy:",
    round(
        accuracy_score(val_df["intent"], pred_labels),
        4
    )
)

print("\nClassification report:")
print(
    classification_report(
        val_df["intent"],
        pred_labels,
        labels=labels,
        zero_division=0
    )
)

Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

pd.DataFrame({
    "text": val_df["text"].values,
    "language": val_df["language"].values,
    "true_intent": val_df["intent"].values,
    "predicted_intent": pred_labels
}).to_csv(
    "results/bert_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    "results/bert_training_log.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        trainer.state.log_history,
        file,
        indent=2
    )

print("\nSaved model:", FINAL_DIR)
print("Saved predictions: results/bert_validation_predictions.csv")
print("Saved training log: results/bert_training_log.json")
