
import ast
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

torch.set_num_threads(4)

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

MODEL_NAMES = {
    "rule_based": "Rule-Based Baseline",
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "feedforward_fasttext": "Feedforward + fastText",
    "multilingual_bert": "Multilingual BERT",
    "distilled_student": "Distilled Student",
    "distilled_student_int8": "INT8 Quantized Student"
}


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


def load_rule_predictor():
    source_path = Path("src/train_rule_baseline.py")
    source = source_path.read_text(encoding="utf-8-sig")

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
            ast.Module(body=selected, type_ignores=[]),
            str(source_path),
            "exec"
        ),
        namespace
    )

    return namespace["predict_intent"]


def load_model(model_name):
    if model_name == "rule_based":
        return {
            "type": "rules",
            "predict": load_rule_predictor()
        }

    if model_name == "tfidf_logreg":
        return {
            "type": "classical",
            "model": joblib.load(
                "models/classical_tfidf_logreg.joblib"
            )
        }

    if model_name == "feedforward_fasttext":
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

        network = FeedforwardClassifier(
            checkpoint["input_dim"],
            checkpoint["hidden_dim"],
            len(checkpoint["labels"])
        )

        network.load_state_dict(
            checkpoint["model_state"]
        )
        network.eval()

        return {
            "type": "feedforward",
            "model": network,
            "labels": checkpoint["labels"],
            "preprocessing": preprocessing,
            "vectors": vector_lookup,
            "input_dim": checkpoint["input_dim"]
        }

    if model_name in {
        "multilingual_bert",
        "distilled_student",
        "distilled_student_int8"
    }:
        if model_name == "distilled_student_int8":
            model_path = "models/distilled_student_int8"

            # Only load our trusted, locally generated model.
            network = torch.load(
                f"{model_path}/quantized_model.pt",
                map_location="cpu",
                weights_only=False
            )
        else:
            model_path = f"models/{model_name}"

            network = (
                AutoModelForSequenceClassification
                .from_pretrained(model_path)
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        network.eval()

        return {
            "type": "transformer",
            "model": network,
            "tokenizer": tokenizer
        }

    raise ValueError(f"Unknown model: {model_name}")


def predict_message(model_bundle, text, language):
    model_type = model_bundle["type"]

    if model_type == "rules":
        prediction = model_bundle["predict"](text)

        return {
            "intent": prediction,
            "confidence": None
        }

    if model_type == "classical":
        model = model_bundle["model"]

        probabilities = model.predict_proba([text])[0]

        index = int(np.argmax(probabilities))

        return {
            "intent": str(model.classes_[index]),
            "confidence": float(probabilities[index])
        }

    if model_type == "feedforward":
        vectors = model_bundle["vectors"]

        tokens = TOKEN_PATTERN.findall(
            str(text).casefold()
        )

        matches = [
            vectors[f"{language}:{token}"]
            for token in tokens
            if f"{language}:{token}" in vectors
        ]

        if matches:
            vector = np.mean(matches, axis=0)
        else:
            vector = np.zeros(
                model_bundle["input_dim"],
                dtype=np.float32
            )

        preprocessing = model_bundle["preprocessing"]

        vector = (
            vector - preprocessing["mean"]
        ) / preprocessing["std"]

        inputs = torch.tensor(
            vector,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.inference_mode():
            logits = model_bundle["model"](inputs)

            probabilities = torch.softmax(
                logits,
                dim=-1
            )[0]

            index = int(probabilities.argmax())

        return {
            "intent": model_bundle["labels"][index],
            "confidence": float(probabilities[index])
        }

    if model_type == "transformer":
        tokenizer = model_bundle["tokenizer"]
        model = model_bundle["model"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        )

        with torch.inference_mode():
            logits = model(**inputs).logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            )[0]

            index = int(probabilities.argmax())

        return {
            "intent": model.config.id2label[index],
            "confidence": float(probabilities[index])
        }

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )
