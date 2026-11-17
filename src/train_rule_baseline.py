
import re
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv(
    "data/model_dataset_verified.csv",
    encoding="utf-8-sig"
)

validation = data[data["split"] == "validation"].copy()

RULES = {
    "refund_request": [
        r"\brefund\b", r"\bmoney back\b", r"\breimburse",
        r"\biade\b", r"\bparamı geri\b",
        r"\bpara iadesi\b"
    ],
    "delivery_question": [
        r"\bdeliver", r"\bshipping\b", r"\bshipment\b",
        r"\btracking\b", r"\barriv",
        r"\bteslimat\b", r"\bkargo\b",
        r"\bgönderi\b", r"\bne zaman gelecek\b"
    ],
    "warranty_question": [
        r"\bwarrant", r"\bguarantee\b",
        r"\bgaranti\b", r"\bgarantisi\b",
        r"\bcoverage\b"
    ],
    "repair_request": [
        r"\brepair\b", r"\btechnician\b",
        r"\bservice appointment\b",
        r"\btamir\b", r"\bonarım\b",
        r"\bservis\b", r"\bteknisyen\b"
    ],
    "technical_issue": [
        r"\bnot working\b", r"\bdoesn't work\b",
        r"\bwon't start\b", r"\berror\b",
        r"\bleak", r"\bnoise\b",
        r"\bçalışmıyor\b", r"\baçılmıyor\b",
        r"\bhata\b", r"\bsızdır", r"\bses çıkar"
    ]
}

# Appliance context helps avoid classifying unrelated
# questions as customer-support tickets.
APPLIANCE_TERMS = [
    "washing machine", "dishwasher", "refrigerator",
    "fridge", "oven", "dryer", "freezer",
    "çamaşır makinesi", "bulaşık makinesi",
    "buzdolabı", "fırın", "kurutma makinesi",
    "dondurucu"
]

def predict_intent(text):
    text = str(text).casefold()

    has_appliance = any(
        term in text for term in APPLIANCE_TERMS
    )

    matches = []

    for intent, patterns in RULES.items():
        if any(re.search(pattern, text) for pattern in patterns):
            matches.append(intent)

    # Conservative fallback when evidence is insufficient.
    if not matches:
        return "out_of_scope"

    # Avoid treating unrelated messages as appliance support.
    if not has_appliance:
        return "out_of_scope"

    # Fixed priority resolves messages matching multiple rules.
    return matches[0]

validation["predicted_intent"] = (
    validation["text"].apply(predict_intent)
)

print("RULE-BASED BASELINE RESULTS")
print("=" * 55)

print(
    "Validation accuracy:",
    round(
        accuracy_score(
            validation["intent"],
            validation["predicted_intent"]
        ),
        4
    )
)

print("\nClassification report:")
print(
    classification_report(
        validation["intent"],
        validation["predicted_intent"],
        zero_division=0
    )
)

validation[
    ["text", "language", "intent", "predicted_intent"]
].to_csv(
    "results/rule_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSaved: results/rule_validation_predictions.csv"
)
