import streamlit as st
from src.prediction_explain import explain_prediction
from src.model_utils import load_model


def predict_text(text, model, vectorizer):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return prediction


def predict_text_with_confidence(text, model, vectorizer):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_tfidf)[0]
        classes = model.classes_

        predicted_index = list(classes).index(prediction)
        confidence = probabilities[predicted_index]

        return prediction, confidence

    return prediction, None


st.set_page_config(
    page_title="Review Prediction",
    page_icon="⭐",
    layout="centered"
)

st.title("⭐ Review Prediction")
st.write(
    "Predict either a 1–5 star rating or binary sentiment from review text."
)

mode = st.radio(
    "Choose prediction mode:",
    ["Rating 1–5", "Sentiment"]
)

review = st.text_area(
    "Write a product review:",
    height=180,
    placeholder="Example: This product works perfectly and arrived fast..."
)

if mode == "Rating 1–5":
    model, vectorizer = load_model(
        "models/rating_model.joblib",
        "models/rating_vectorizer.joblib"
    )
else:
    model, vectorizer = load_model(
        "models/sentiment_model.joblib",
        "models/sentiment_vectorizer.joblib"
    )

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please write a review first.")
    else:
        prediction, confidence = predict_text_with_confidence(
            review, model, vectorizer
        )

        st.subheader("Prediction")

        if mode == "Rating 1–5":
            st.write(f"## {prediction} ⭐")
            st.info(
                "Rating prediction is a fine-grained task, so predictions may be less reliable "
                "for mixed or short reviews."
            )
        else:
            if prediction == "positive":
                st.success("Positive 🙂")
            else:
                st.error("Negative 🙁")

            if confidence is not None:
                st.write(f"Confidence: **{confidence * 100:.2f}%**")
                st.progress(float(confidence))

        explanation = explain_prediction(review, model, vectorizer)

        if explanation:
            st.subheader("Why this prediction?")

            for feature, score in explanation:
                st.write(f"- **{feature}** → contribution: `{score:.4f}`")