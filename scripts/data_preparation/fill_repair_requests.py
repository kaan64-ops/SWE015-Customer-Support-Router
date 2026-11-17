
from pathlib import Path
import pandas as pd

path = Path("data/missing_bilingual_examples.csv")
df = pd.read_csv(path, dtype=str).fillna("")

examples = {
    "repair_request_001": (
        "Could you send a technician to fix my washing machine?",
        "Çamaşır makinemi tamir etmek için teknisyen gönderebilir misiniz?"
    ),
    "repair_request_004": (
        "I need to arrange a repair for my refrigerator.",
        "Buzdolabım için tamir servisi ayarlamak istiyorum."
    ),
    "repair_request_005": (
        "Can I change the date of my technician appointment?",
        "Teknisyen randevumun tarihini değiştirebilir miyim?"
    ),
    "repair_request_006": (
        "Please cancel my scheduled repair appointment.",
        "Planlanmış tamir randevumu iptal eder misiniz?"
    ),
    "repair_request_007": (
        "I need an urgent repair visit because my freezer has stopped working.",
        "Derin dondurucum çalışmadığı için acil tamir servisi istiyorum."
    ),
    "repair_request_008": (
        "Do you have any technicians available this weekend?",
        "Bu hafta sonu müsait teknisyeniniz var mı?"
    ),
    "repair_request_010": (
        "I requested a repair last week. Can you check its status?",
        "Geçen hafta tamir talebi oluşturdum. Durumunu kontrol edebilir misiniz?"
    ),
}

for scenario_id, (english, turkish) in examples.items():
    mask = df["scenario_id"] == scenario_id

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one row for {scenario_id}"
        )

    # Protect existing examples from accidental overwrites.
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

repair = df[df["intent"] == "repair_request"]

print("REPAIR REQUEST EXAMPLES CREATED")
print("=" * 55)
print("Repair scenarios:", len(repair))
print(
    "English messages filled:",
    (repair["english_message"] != "").sum()
)
print(
    "Turkish messages filled:",
    (repair["turkish_message"] != "").sum()
)
print(
    "\nRemaining scenarios to fill:",
    (df["english_message"] == "").sum()
)
print("\nSaved:", path)
