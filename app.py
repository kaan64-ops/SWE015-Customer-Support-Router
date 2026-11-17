
import streamlit as st

from src.message_processing import (
    detect_language,
    extract_product_codes,
    normalize_message,
    get_small_talk_reply,
    correct_appliance_names,
)

from src.model_inference import (
    MODEL_NAMES,
    load_model,
    predict_message
)

st.set_page_config(
    page_title="Bilingual Customer Support Router",
    page_icon="🛠️",
    layout="centered"
)

INTENT_NAMES = {
    "technical_issue": "Technical Issue",
    "repair_request": "Repair Request",
    "warranty_question": "Warranty Question",
    "delivery_question": "Delivery Question",
    "refund_request": "Refund Request",
    "out_of_scope": "Out of Scope"
}

@st.cache_resource(show_spinner="Loading selected model...")
def get_model(model_name):
    return load_model(model_name)


st.title("🛠️ Bilingual Customer Support Router")

st.write(
    "Classify English and Turkish home-appliance "
    "support messages using six different model variants."
)

selected_model = st.selectbox(
    "Choose classification model",
    options=list(MODEL_NAMES.keys()),
    index=1,
    format_func=lambda name: MODEL_NAMES[name]
)

threshold = st.slider(
    "Automatic routing confidence threshold",
    min_value=0.30,
    max_value=0.95,
    value=0.60,
    step=0.05,
    format="%.2f"
)






# Initialize customer message storage
if "customer_message" not in st.session_state:
    st.session_state.customer_message = ""

# Save the submitted message before clearing the input
def clear_customer_message():
    st.session_state.submitted_message = (
        st.session_state.customer_message
    )
    st.session_state.customer_message = ""

# Customer message input
message = st.text_area(
    "Customer message",
    key="customer_message",
    placeholder=(
        "My washing machine is leaking water.\n"
        "Buzdolabımın garantisi devam ediyor mu?"
    ),
    height=140
)



if st.button(
    "Classify Message",
    type="primary",
    on_click=clear_customer_message
):
    message = st.session_state.get(
        "submitted_message", ""
    )

    if not message.strip():

        st.warning("Please enter a message.")

    else:
        cleaned_message = normalize_message(message)
        corrected_message, spelling_corrections = correct_appliance_names(cleaned_message)
        language = detect_language(cleaned_message)
        product_codes = extract_product_codes(cleaned_message)

        st.subheader("Message Analysis")

        if spelling_corrections:
            st.write("**Spelling corrections suggested:**")
            for original, corrected in spelling_corrections:
                st.write(f"- {original} → {corrected}")

            st.write("**Corrected message:**", corrected_message)
        else:
            st.write("**Spelling corrections:** None needed")

        st.write(
            "**Detected language:**",
            "Turkish" if language == "tr" else "English"
        )

        st.write(
            "**Product model code:**",
            ", ".join(product_codes)
            if product_codes
            else "Not detected"
        )
        small_talk_reply = get_small_talk_reply(cleaned_message, language)
        is_small_talk = small_talk_reply is not None

        if small_talk_reply is not None:
            st.subheader("Rule-Based Small-Talk Response")
            st.success(small_talk_reply)
            st.caption(
                "ELIZA-style rule-based response. "
                "This reply is generated without a trained model."
            )

        with st.spinner("Classifying message..."):
            model_bundle = get_model(selected_model)

            result = predict_message(model_bundle, corrected_message, language)

        prediction = result["intent"]
        confidence = result["confidence"]

        st.subheader("Side-by-Side Model Comparison")

        comparison_models = [
            "rule_based",
            "tfidf_logreg",
            "feedforward_fasttext",
            "multilingual_bert",
            "distilled_student",
            "distilled_student_int8",
        ]

        comparison_rows = []

        with st.spinner("Comparing all six models..."):
            for model_name in comparison_models:
                comparison_bundle = get_model(model_name)

                if model_name == selected_model:
                    comparison_result = result
                else:
                    comparison_result = predict_message(
                        comparison_bundle,
                        corrected_message,
                        language,
                    )

                comparison_confidence = comparison_result["confidence"]

                comparison_rows.append({
                    "Model": MODEL_NAMES[model_name],
                    "Predicted intent": INTENT_NAMES[comparison_result["intent"]],
                    "Confidence": (
                        f"{comparison_confidence * 100:.1f}%"
                        if comparison_confidence is not None
                        else "N/A (rule-based)"
                    ),
                })

        st.dataframe(
            comparison_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Classification Result")

        st.write(
            "**Model:**",
            MODEL_NAMES[selected_model]
        )

        st.write(
            "**Predicted intent:**",
            INTENT_NAMES[prediction]
        )
        if is_small_talk:
            st.success("Small talk handled automatically — no human review required.")
            st.write("**Routing decision:** Handled automatically — small-talk response")
        elif confidence is None:
            st.info(
                "The rule-based classifier does not "
                "produce probabilistic confidence scores."
            )

            if prediction == "out_of_scope":
                st.warning(
                    "Rule-based OOS prediction — "
                    "send to human review."
                )
                st.write(
                    "**Routing decision:** Human support queue"
                )
            else:
                st.success(
                    "Rule-based category match found."
                )
                st.write(
                    "**Routing decision:**",
                    INTENT_NAMES[prediction],
                    "support queue"
                )

        else:
            st.metric("Model confidence", f"{confidence * 100:.3f}%")

            if confidence < threshold:
                st.warning(
                    "Human Review Required — model confidence "
                    "is below the selected threshold."
                )

                st.write(
                    "**Routing decision:** Human support queue"
                )

            elif prediction == "out_of_scope":
                st.warning("Out-of-Scope — human agent review required.")
                st.write("**Routing decision:** Human support queue")

            else:
                st.success("Automatic Routing Approved")

                st.write(
                    "**Routing decision:**",
                    INTENT_NAMES[prediction],
                    "support queue"
                )

        
        # Support ticket summary
        st.subheader("Support Ticket Summary")
        if is_small_talk:
            confidence_text = (
                "Not applicable (rule-based small-talk response)"
            )
            routing_queue = "Handled automatically — small-talk response"
        elif confidence is None:
            confidence_text = "Not available (rule-based)"

            if prediction == "out_of_scope":
                routing_queue = "Human support queue"
            else:
                routing_queue = (
                    f"{INTENT_NAMES[prediction]} support queue"
                )

        else:
            confidence_text = f"{confidence * 100:.3f}%"

            if confidence < threshold:
                routing_queue = "Human support queue"
            elif prediction == "out_of_scope":
                routing_queue = "Human support queue"
            else:
                routing_queue = (
                    f"{INTENT_NAMES[prediction]} support queue"
                )

        ticket_summary = {
            "Customer message": cleaned_message,
            "Corrected message": corrected_message,
            "Spelling corrections": [
                {"original": original, "corrected": corrected}
                for original, corrected in spelling_corrections
            ],
            "Detected language": (
                "Turkish" if language == "tr" else "English"
            ),
            "Product model codes": (
                ", ".join(product_codes)
                if product_codes else "Not detected"
            ),
            "Classification model": MODEL_NAMES[selected_model],
            "Predicted intent": INTENT_NAMES[prediction],
            "Model confidence": confidence_text,
            "Routing decision": routing_queue
        }

        st.json(ticket_summary, expanded=True)

        import json

        json_data = json.dumps(
            ticket_summary,
            ensure_ascii=False,
            indent=2
        )

        st.download_button(
            "📥 Download Ticket Summary (JSON)",
            data=json_data,
            file_name="support_ticket_summary.json",
            mime="application/json",
            key="download_ticket"
        )





st.divider()

st.caption(
    "SWE015 — Introduction to Large Language Models | "
    "Bilingual Customer Support Ticket Router"
)

st.caption(
    "Confidence values are uncalibrated model scores. "
    "Language and product-code detection use heuristics."
)

