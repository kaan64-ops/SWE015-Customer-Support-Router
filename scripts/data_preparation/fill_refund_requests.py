
from pathlib import Path
import pandas as pd

path = Path("data/missing_bilingual_examples.csv")
df = pd.read_csv(path, dtype=str).fillna("")

examples = {
    "refund_request_004": (
        "My new refrigerator arrived damaged. I want a full refund.",
        "Yeni buzdolabım hasarlı geldi. Ücretimin tamamını geri istiyorum."
    ),
    "refund_request_005": (
        "How long will it take to receive my refund for the oven?",
        "Fırınımın ücret iadesini almam ne kadar sürer?"
    ),
    "refund_request_006": (
        "I returned my washing machine two weeks ago, but I still haven't received my refund.",
        "Çamaşır makinemi iki hafta önce iade ettim ama ücretim hâlâ geri ödenmedi."
    ),
    "refund_request_007": (
        "Can you refund the money to my original payment card?",
        "Ücret iadesini ödeme yaptığım karta gönderebilir misiniz?"
    ),
    "refund_request_008": (
        "I bought a microwave but never opened the box. Can I return it?",
        "Bir mikrodalga fırın aldım ama kutusunu hiç açmadım. İade edebilir miyim?"
    ),
    "refund_request_009": (
        "The appliance has a minor defect. Can I get a partial refund?",
        "Cihazda küçük bir kusur var. Kısmi ücret iadesi alabilir miyim?"
    ),
    "refund_request_010": (
        "My dishwasher has failed repeatedly despite several repairs. I would like a refund.",
        "Bulaşık makinem birkaç kez tamir edilmesine rağmen sürekli arızalanıyor. Ücret iadesi istiyorum."
    ),
}

expected_ids = set(examples)
actual_ids = set(
    df.loc[
        df["intent"] == "refund_request",
        "scenario_id"
    ]
)

if expected_ids != actual_ids:
    raise ValueError(
        "Refund scenario IDs do not match the inventory."
    )

for scenario_id, (english, turkish) in examples.items():
    mask = df["scenario_id"] == scenario_id

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one row for {scenario_id}"
        )

    if (
        df.loc[mask, "english_message"].iloc[0]
        or df.loc[mask, "turkish_message"].iloc[0]
    ):
        raise ValueError(
            f"Messages already exist for {scenario_id}"
        )

    df.loc[mask, "english_message"] = english
    df.loc[mask, "turkish_message"] = turkish
    df.loc[mask, "review_status"] = "drafted"

df.to_csv(
    path,
    index=False,
    encoding="utf-8-sig"
)

refund = df[df["intent"] == "refund_request"]

print("REFUND REQUEST EXAMPLES CREATED")
print("=" * 55)

print("Refund scenarios:", len(refund))
print(
    "English messages filled:",
    (refund["english_message"] != "").sum()
)
print(
    "Turkish messages filled:",
    (refund["turkish_message"] != "").sum()
)

remaining = (
    (df["english_message"] == "")
    | (df["turkish_message"] == "")
).sum()

print("\nRemaining incomplete scenarios:", remaining)
print("\nSaved:", path)

if remaining != 0:
    raise ValueError(
        "Some bilingual scenarios are still incomplete."
    )

print("\nALL 35 MISSING BILINGUAL PAIRS DRAFTED!")
