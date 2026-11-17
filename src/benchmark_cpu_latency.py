
import ast
import gc
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

torch.set_num_threads(4)

RUNS = 120
WARMUP = 10
MAX_LENGTH = 64

data = pd.read_csv(
    "data/model_dataset_verified.csv",
    encoding="utf-8-sig"
)

# Benchmark representative development examples,
# not the held-out test set.
examples = data[
    data["split"] == "validation"
]["text"].tolist()

languages = data[
    data["split"] == "validation"
]["language"].tolist()

assert len(examples) > 0

Path("results").mkdir(exist_ok=True)

results = []

def benchmark(name, predict):
    timings = []

    # Warm up before measuring.
    for i in range(WARMUP):
        index = i % len(examples)
        predict(examples[index], languages[index])

    for i in range(RUNS):
        index = i % len(examples)

        start = time.perf_counter()

        predict(examples[index], languages[index])

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        timings.append(elapsed_ms)

    metrics = {
        "model": name,
        "runs": RUNS,
        "mean_ms": round(float(np.mean(timings)), 3),
        "median_ms": round(float(np.median(timings)), 3),
        "p95_ms": round(
            float(np.percentile(timings, 95)), 3
        )
    }

    results.append(metrics)

    print(
        f"{name:25s} "
        f"mean={metrics['mean_ms']:9.3f} ms | "
        f"median={metrics['median_ms']:9.3f} ms | "
        f"p95={metrics['p95_ms']:9.3f} ms"
    )

# --------------------------------------------------
# 1. Original rule-based classifier
# --------------------------------------------------

source = Path(
    "src/train_rule_baseline.py"
).read_text(encoding="utf-8-sig")

tree = ast.parse(source)

selected = []

for node in tree.body:
    if isinstance(node, ast.Assign):
        names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]

        if any(
            name in {"RULES", "APPLIANCE_TERMS"}
            for name in names
        ):
            selected.append(node)

    elif isinstance(node, ast.FunctionDef):
        if node.name == "predict_intent":
            selected.append(node)

assert len(selected) == 3

namespace = {"re": re}

exec(
    compile(
        ast.Module(
            body=selected,
            type_ignores=[]
        ),
        "src/train_rule_baseline.py",
        "exec"
    ),
    namespace
)

rule_predict = namespace["predict_intent"]

print("\nCPU INFERENCE LATENCY BENCHMARK")
print("=" * 75)
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"Timed runs per model: {RUNS}")
print(f"Warmup runs: {WARMUP}\n")

benchmark(
    "rule_based",
    lambda text, language: rule_predict(text)
)

# --------------------------------------------------
# 2. TF-IDF + Logistic Regression
# --------------------------------------------------

classical = joblib.load(
    "models/classical_tfidf_logreg.joblib"
)

benchmark(
    "tfidf_logreg",
    lambda text, language: classical.predict([text])[0]
)

del classical
gc.collect()

# --------------------------------------------------
# 3. Feedforward + fastText
# --------------------------------------------------

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

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

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

def feedforward_predict(text, language):
    tokens = TOKEN_PATTERN.findall(str(text).casefold())

    vectors = [
        vector_lookup[f"{language}:{token}"]
        for token in tokens
        if f"{language}:{token}" in vector_lookup
    ]

    if vectors:
        vector = np.mean(vectors, axis=0)
    else:
        vector = np.zeros(
            checkpoint["input_dim"],
            dtype=np.float32
        )

    vector = (
        vector - preprocessing["mean"]
    ) / preprocessing["std"]

    tensor = torch.tensor(
        vector,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.inference_mode():
        output = feedforward(tensor)
        return output.argmax(dim=1).item()

benchmark(
    "feedforward_fasttext",
    feedforward_predict
)

del feedforward
del archive
del vector_lookup
gc.collect()

# --------------------------------------------------
# Transformer models
# --------------------------------------------------

def benchmark_transformer(name, path, quantized=False):
    tokenizer_path = (
        "models/distilled_student"
        if quantized else path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path
    )

    if quantized:
        # Only load our trusted locally generated checkpoint.
        model = torch.load(
            "models/distilled_student_int8/quantized_model.pt",
            map_location="cpu",
            weights_only=False
        )
    else:
        model = (
            AutoModelForSequenceClassification
            .from_pretrained(path)
        )

    model.eval()

    def predict(text, language):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        with torch.inference_mode():
            logits = model(**inputs).logits
            return logits.argmax(dim=1).item()

    benchmark(name, predict)

    del model
    del tokenizer
    gc.collect()

benchmark_transformer(
    "multilingual_bert",
    "models/multilingual_bert"
)

benchmark_transformer(
    "distilled_student",
    "models/distilled_student"
)

benchmark_transformer(
    "distilled_student_int8",
    "models/distilled_student_int8",
    quantized=True
)

# --------------------------------------------------
# Save benchmark results
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(
    "results/cpu_latency_benchmark.csv",
    index=False
)

print("\nSaved: results/cpu_latency_benchmark.csv")

# Merge accuracy, model size, and latency.
metrics = pd.read_csv(
    "results/final_test_metrics.csv"
)

sizes = pd.read_csv(
    "results/model_sizes.csv"
)

comparison = (
    metrics
    .merge(sizes[["model", "size_mb"]], on="model")
    .merge(
        df[["model", "mean_ms", "median_ms", "p95_ms"]],
        on="model"
    )
)

comparison.to_csv(
    "results/final_model_comparison.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print("=" * 100)
print(
    comparison[
        [
            "model",
            "accuracy",
            "macro_f1",
            "in_scope_accuracy",
            "oos_recall",
            "size_mb",
            "mean_ms",
            "p95_ms"
        ]
    ].to_string(index=False)
)

print("\nSaved: results/final_model_comparison.csv")
