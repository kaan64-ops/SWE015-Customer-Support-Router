
from pathlib import Path
import pandas as pd

scenarios = {
    "technical_issue": [
        "Washing machine does not start",
        "Dishwasher does not drain",
        "Refrigerator does not cool",
        "Oven does not heat",
        "Vacuum cleaner loses suction",
        "Washing machine makes loud noises",
        "Dishwasher leaks water",
        "Refrigerator makes unusual sounds",
        "Microwave display is not working",
        "Dryer stops during a cycle",
    ],
    "repair_request": [
        "Book a washing machine technician",
        "Schedule dishwasher repair",
        "Request an oven service appointment",
        "Arrange refrigerator repair",
        "Reschedule a technician visit",
        "Cancel an existing repair appointment",
        "Request urgent repair service",
        "Ask about technician availability",
        "Request an appliance inspection visit",
        "Follow up on an existing repair request",
    ],
    "warranty_question": [
        "Check washing machine warranty validity",
        "Ask about refrigerator warranty duration",
        "Check whether motor damage is covered",
        "Ask about extended warranty options",
        "Request warranty registration",
        "Ask whether accidental damage is covered",
        "Check warranty after a previous repair",
        "Request a copy of warranty documentation",
        "Ask about warranty transfer after resale",
        "Check coverage for replacement parts",
    ],
    "delivery_question": [
        "Ask when a dishwasher will arrive",
        "Track a refrigerator order",
        "Check oven delivery status",
        "Report delayed appliance delivery",
        "Change the delivery address",
        "Reschedule a delivery",
        "Ask about delivery charges",
        "Report missing items from a shipment",
        "Confirm a delivery time window",
        "Ask about international delivery",
    ],
    "refund_request": [
        "Request refund for a vacuum cleaner",
        "Return a dishwasher",
        "Ask for refund after order cancellation",
        "Request refund for a damaged appliance",
        "Ask about refund processing time",
        "Report a missing refund",
        "Request refund to original payment method",
        "Return an unopened appliance",
        "Ask about partial refunds",
        "Request refund after repeated product failures",
    ],
}

rows = []

for intent, descriptions in scenarios.items():
    for number, description in enumerate(
        descriptions, start=1
    ):
        rows.append({
            "scenario_id": f"{intent}_{number:03d}",
            "intent": intent,
            "scenario_description": description,
            "review_status": "pending",
        })

df = pd.DataFrame(rows)

assert len(df) == 50
assert df["scenario_id"].is_unique

output = Path("data/scenario_inventory.csv")
output.parent.mkdir(exist_ok=True)

df.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("Scenario inventory created!")
print("\nScenarios per intent:")
print(df["intent"].value_counts().sort_index())

print("\nTotal scenarios:", len(df))
print("Unique IDs:", df["scenario_id"].nunique())

print("\nSaved:", output)
