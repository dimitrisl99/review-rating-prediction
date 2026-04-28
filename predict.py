from src.model_utils import load_model


def predict_review(review_text, model, vectorizer):
    review_tfidf = vectorizer.transform([review_text])
    prediction = model.predict(review_tfidf)[0]
    return prediction


model, vectorizer = load_model()

while True:
    review = input("\nWrite a review or type 'exit': ")

    if review.lower() == "exit":
        print("Exiting...")
        break

    predicted_rating = predict_review(review, model, vectorizer)

    print(f"Predicted rating: {predicted_rating} stars")