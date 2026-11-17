
from pathlib import Path
import pandas as pd

path = Path("data/missing_bilingual_examples.csv")
df = pd.read_csv(path).fillna("")

examples = {
    "technical_issue_004": (
        "My oven turns on, but it doesn't heat up.",
        "Fırınım açılıyor ama ısınmıyor."
    ),
    "technical_issue_005": (
        "My vacuum cleaner has lost most of its suction power.",
        "Elektrikli süpürgem artık eskisi gibi çekmiyor."
    ),
    "technical_issue_006": (
        "My washing machine makes a loud banging noise during spinning.",
        "Çamaşır makinem sıkma sırasında yüksek sesle vuruyor."
    ),
    "technical_issue_007": (
        "Water keeps leaking from underneath my dishwasher.",
        "Bulaşık makinemin altından sürekli su sızıyor."
    ),
    "technical_issue_008": (
        "My refrigerator has started making a strange buzzing sound.",
        "Buzdolabım garip bir uğultu sesi çıkarmaya başladı."
    ),
    "technical_issue_009": (
        "The display on my microwave is completely blank.",
        "Mikrodalga fırınımın ekranında hiçbir şey görünmüyor."
    ),
    "technical_issue_010": (
        "My tumble dryer stops before the drying cycle is finished.",
        "Çamaşır kurutma makinem program bitmeden duruyor."
    ),
}

for scenario_id, (english, turkish) in examples.items():
    mask = df["scenario_id"] == scenario_id

    if mask.sum() != 1:
        raise ValueError(
            f"Expected one row for {scenario_id}"
        )

    df.loc[mask, "english_message"] = english
    df.loc[mask, "turkish_message"] = turkish
    df.loc[mask, "review_status"] = "drafted"

df.to_csv(
    path,
    index=False,
    encoding="utf-8-sig"
)

technical = df[
    df["intent"] == "technical_issue"
]

print("TECHNICAL ISSUE EXAMPLES CREATED")
print("=" * 55)

print("Technical scenarios:", len(technical))

print(
    "English messages filled:",
    (technical["english_message"] != "").sum()
)

print(
    "Turkish messages filled:",
    (technical["turkish_message"] != "").sum()
)

print("\nRemaining scenarios to fill:")
print(
    (df["english_message"] == "").sum()
)

print("\nSaved:", path)
