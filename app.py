import streamlit as st

from src.model_utils import load_model


def predict_text(text, model, vectorizer):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return prediction


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
        prediction = predict_text(review, model, vectorizer)

        st.subheader("Prediction")

        if mode == "Rating 1–5":
            st.write(f"## {prediction} ⭐")
        else:
            if prediction == "positive":
                st.success("Positive 🙂")
            else:
                st.error("Negative 🙁")