
from pathlib import Path
import pandas as pd

examples = [
    # Technical issues
    ("My washing machine won't start.", "technical_issue", "en"),
    ("My dishwasher is not draining water.", "technical_issue", "en"),
    ("My refrigerator is not cooling properly.", "technical_issue", "en"),
    ("Çamaşır makinem çalışmıyor.", "technical_issue", "tr"),
    ("Bulaşık makinem su boşaltmıyor.", "technical_issue", "tr"),
    ("Buzdolabım yeterince soğutmuyor.", "technical_issue", "tr"),

    # Repair requests
    ("I need a technician to repair my oven.", "repair_request", "en"),
    ("Can I schedule a dishwasher repair?", "repair_request", "en"),
    ("Please arrange a service appointment.", "repair_request", "en"),
    ("Fırınım için teknisyen istiyorum.", "repair_request", "tr"),
    ("Bulaşık makinesi için tamir randevusu alabilir miyim?", "repair_request", "tr"),
    ("Servis randevusu oluşturmak istiyorum.", "repair_request", "tr"),

    # Warranty questions
    ("Is my washing machine still under warranty?", "warranty_question", "en"),
    ("How long is the refrigerator warranty?", "warranty_question", "en"),
    ("Does the warranty cover motor failure?", "warranty_question", "en"),
    ("Çamaşır makinemin garantisi devam ediyor mu?", "warranty_question", "tr"),
    ("Buzdolabının garanti süresi ne kadar?", "warranty_question", "tr"),
    ("Motor arızası garanti kapsamında mı?", "warranty_question", "tr"),

    # Delivery questions
    ("When will my dishwasher arrive?", "delivery_question", "en"),
    ("Where is my refrigerator order?", "delivery_question", "en"),
    ("Can I track my oven delivery?", "delivery_question", "en"),
    ("Bulaşık makinem ne zaman teslim edilecek?", "delivery_question", "tr"),
    ("Buzdolabı siparişim nerede?", "delivery_question", "tr"),
    ("Fırınımın teslimatını takip edebilir miyim?", "delivery_question", "tr"),

    # Refund requests
    ("I want a refund for my vacuum cleaner.", "refund_request", "en"),
    ("How can I return my dishwasher?", "refund_request", "en"),
    ("Please refund my cancelled order.", "refund_request", "en"),
    ("Elektrikli süpürgemi iade etmek istiyorum.", "refund_request", "tr"),
    ("Bulaşık makinesini nasıl iade edebilirim?", "refund_request", "tr"),
    ("İptal ettiğim siparişin ücretini geri istiyorum.", "refund_request", "tr"),
]

df = pd.DataFrame(
    examples,
    columns=["text", "intent", "language"]
)

df["source"] = "manually_authored_seed"
df["split"] = "development_seed"

Path("data").mkdir(exist_ok=True)

df.to_csv(
    "data/support_seed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Support seed dataset created!")
print("\nExamples per intent and language:")
print(df.groupby(["intent", "language"]).size())

print("\nTotal examples:", len(df))
print("\nSaved: data/support_seed.csv")
