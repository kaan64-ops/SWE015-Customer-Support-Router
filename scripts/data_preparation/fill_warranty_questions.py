
from pathlib import Path
import pandas as pd

path = Path("data/missing_bilingual_examples.csv")
df = pd.read_csv(path, dtype=str).fillna("")

examples = {
    "warranty_question_004": (
        "Can I purchase an extended warranty for my dishwasher?",
        "Bulaşık makinem için ek garanti satın alabilir miyim?"
    ),
    "warranty_question_005": (
        "How do I register my new washing machine for warranty?",
        "Yeni çamaşır makinemi garantiye nasıl kaydettirebilirim?"
    ),
    "warranty_question_006": (
        "Does the warranty cover accidental damage to my oven?",
        "Fırınımın kazara hasar görmesi garanti kapsamında mı?"
    ),
    "warranty_question_007": (
        "Is my refrigerator still covered after a previous repair?",
        "Buzdolabım daha önce tamir edildi. Garantisi hâlâ geçerli mi?"
    ),
    "warranty_question_008": (
        "Could you send me a copy of my warranty certificate?",
        "Garanti belgemin bir kopyasını gönderebilir misiniz?"
    ),
    "warranty_question_009": (
        "If I sell my washing machine, can the warranty transfer to the buyer?",
        "Çamaşır makinemi satarsam garanti yeni sahibine devredilir mi?"
    ),
    "warranty_question_010": (
        "Are replacement parts included in the appliance warranty?",
        "Cihazın garanti kapsamına yedek parçalar da dahil mi?"
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

warranty = df[df["intent"] == "warranty_question"]

print("WARRANTY QUESTION EXAMPLES CREATED")
print("=" * 55)
print("Warranty scenarios:", len(warranty))
print(
    "English messages filled:",
    (warranty["english_message"] != "").sum()
)
print(
    "Turkish messages filled:",
    (warranty["turkish_message"] != "").sum()
)
print(
    "\nRemaining scenarios to fill:",
    (df["english_message"] == "").sum()
)
print("\nSaved:", path)
