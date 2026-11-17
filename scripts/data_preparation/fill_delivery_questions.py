
from pathlib import Path
import pandas as pd

path = Path("data/missing_bilingual_examples.csv")
df = pd.read_csv(path, dtype=str).fillna("")

examples = {
    "delivery_question_004": (
        "My washing machine delivery is three days late. What happened?",
        "Çamaşır makinemin teslimatı üç gün gecikti. Ne oldu?"
    ),
    "delivery_question_005": (
        "Can I change the delivery address for my refrigerator?",
        "Buzdolabımın teslimat adresini değiştirebilir miyim?"
    ),
    "delivery_question_006": (
        "I need to reschedule my dishwasher delivery for next Monday.",
        "Bulaşık makinemin teslimatını gelecek pazartesiye ertelemek istiyorum."
    ),
    "delivery_question_007": (
        "How much does delivery cost for a new oven?",
        "Yeni bir fırının teslimat ücreti ne kadar?"
    ),
    "delivery_question_008": (
        "My appliance shipment arrived, but one accessory is missing.",
        "Cihazım teslim edildi ancak bir aksesuar eksik."
    ),
    "delivery_question_009": (
        "Can you confirm the delivery time window for my washing machine?",
        "Çamaşır makinemin teslim edileceği saat aralığını doğrulayabilir misiniz?"
    ),
    "delivery_question_010": (
        "Do you offer international delivery for refrigerators?",
        "Buzdolapları için uluslararası teslimat yapıyor musunuz?"
    ),
}

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

delivery = df[df["intent"] == "delivery_question"]

print("DELIVERY QUESTION EXAMPLES CREATED")
print("=" * 55)
print("Delivery scenarios:", len(delivery))
print(
    "English messages filled:",
    (delivery["english_message"] != "").sum()
)
print(
    "Turkish messages filled:",
    (delivery["turkish_message"] != "").sum()
)
print(
    "\nRemaining scenarios to fill:",
    (df["english_message"] == "").sum()
)
print("\nSaved:", path)
