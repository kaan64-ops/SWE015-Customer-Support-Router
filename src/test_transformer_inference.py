
import gc

from src.model_inference import (
    load_model,
    predict_message
)

models = [
    "multilingual_bert",
    "distilled_student",
    "distilled_student_int8"
]

message = "My washing machine is leaking water."

for model_name in models:
    print("\n" + "=" * 60)
    print("MODEL:", model_name)

    model = load_model(model_name)

    result = predict_message(
        model,
        message,
        "en"
    )

    print("RESULT:", result)

    del model
    gc.collect()

print("\nALL TRANSFORMER INFERENCE TESTS COMPLETED")
