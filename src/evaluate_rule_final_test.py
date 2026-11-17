
import ast
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

SOURCE = Path("src/train_rule_baseline.py")
DATASET = Path("data/model_dataset_verified.csv")
RESULTS = Path("results")

RESULTS.mkdir(exist_ok=True)

# Read the original rules and function without executing
# the validation evaluation at the bottom of the script.
source_text = SOURCE.read_text(encoding="utf-8-sig")
tree = ast.parse(source_text)

selected = []

for node in tree.body:
    if isinstance(node, ast.Assign):
        variable_names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]

        if any(
            name in {"RULES", "APPLIANCE_TERMS"}
            for name in variable_names
        ):
            selected.append(node)

    elif isinstance(node, ast.FunctionDef):
        if node.name == "predict_intent":
            selected.append(node)

assert len(selected) == 3, (
    "Could not locate the original rule definitions "
    "and prediction function."
)

# Execute only the extracted definitions.
# The source is our own trusted local project file.
import re

namespace = {"re": re}

definitions = ast.Module(
    body=selected,
    type_ignores=[]
)

exec(
    compile(definitions, str(SOURCE), "exec"),
    namespace
)

predict_intent = namespace["predict_intent"]

# Evaluate the original untouched test partition.
data = pd.read_csv(DATASET, encoding="utf-8-sig")
test = data[data["split"] == "test"].copy()

assert len(test) == 58, "Unexpected test-set size."

truth = test["intent"].tolist()
predictions = [
    predict_intent(text)
    for text in test["text"]
]

labels = sorted(data["intent"].unique())

support_mask = test["intent"] != "out_of_scope"
oos_mask = test["intent"] == "out_of_scope"

correct = pd.Series(
    [
        predicted == actual
        for predicted, actual in zip(predictions, truth)
    ],
    index=test.index
)

metrics = {
    "model": "rule_based",
    "accuracy": accuracy_score(truth, predictions),
    "macro_f1": f1_score(
        truth,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0
    ),
    "in_scope_accuracy": correct[support_mask].mean(),
    "oos_recall": sum(
        prediction == "out_of_scope"
        for prediction, is_oos in zip(
            predictions,
            oos_mask
        )
        if is_oos
    ) / int(oos_mask.sum()),
    "english_accuracy": correct[
        test["language"] == "en"
    ].mean(),
    "turkish_accuracy": correct[
        test["language"] == "tr"
    ].mean()
}

print("\nRULE-BASED FINAL TEST RESULTS")
print("=" * 60)

for key, value in metrics.items():
    print(
        f"{key}: {value:.4f}"
        if isinstance(value, (int, float))
        else f"{key}: {value}"
    )

print("\nClassification report:")
print(
    classification_report(
        truth,
        predictions,
        labels=labels,
        zero_division=0
    )
)

# Append rule-based results to existing metrics.
metrics_path = RESULTS / "final_test_metrics.csv"
metrics_df = pd.read_csv(metrics_path)

metrics_df = metrics_df[
    metrics_df["model"] != "rule_based"
].copy()

metrics_df = pd.concat(
    [pd.DataFrame([metrics]), metrics_df],
    ignore_index=True
)

metrics_df.to_csv(metrics_path, index=False)

# Append rule-based predictions to existing predictions.
predictions_path = RESULTS / "final_test_predictions.csv"
predictions_df = pd.read_csv(
    predictions_path,
    encoding="utf-8-sig"
)

assert len(predictions_df) == len(test)
assert predictions_df["text"].tolist() == (
    test["text"].tolist()
), "Prediction row order does not match test data."

predictions_df["rule_based"] = predictions

predictions_df.to_csv(
    predictions_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nUpdated:")
print(metrics_path)
print(predictions_path)

print("\nAll six model variants are now evaluated.")
