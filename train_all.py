import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import clean_rating_data, clean_sentiment_data
from src.features import create_tfidf
from src.train import train_svm, train_logistic
from src.evaluate import evaluate
from src.model_utils import save_model


def train_rating_model():
    raw_df = load_data()
    rating_df = clean_rating_data(raw_df)

    X = rating_df["review"]
    y = rating_df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = create_tfidf()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = train_svm(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print("\nRating model results:")
    evaluate(y_test, y_pred)

    save_model(
        model,
        vectorizer,
        "models/rating_model.joblib",
        "models/rating_vectorizer.joblib"
    )


def train_sentiment_model():
    raw_df = load_data()
    sentiment_df = clean_sentiment_data(raw_df)

    custom_df = pd.read_csv("data/custom_sentiment_examples.csv")

    sentiment_df = pd.concat(
        [sentiment_df, custom_df],
        ignore_index=True
    )

    sentiment_df = sentiment_df.dropna(subset=["review", "sentiment"])
    sentiment_df["sentiment"] = sentiment_df["sentiment"].str.strip().str.lower()
    sentiment_df = sentiment_df[
        sentiment_df["sentiment"].isin(["positive", "negative"])
    ]

    print("\nFinal sentiment dataset distribution:")
    print(sentiment_df["sentiment"].value_counts())

    X = sentiment_df["review"]
    y = sentiment_df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = create_tfidf()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = train_logistic(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print("\nSentiment model results:")
    evaluate(y_test, y_pred)

    save_model(
        model,
        vectorizer,
        "models/sentiment_model.joblib",
        "models/sentiment_vectorizer.joblib"
    )


train_rating_model()
train_sentiment_model()

print("\nBoth models saved successfully.")