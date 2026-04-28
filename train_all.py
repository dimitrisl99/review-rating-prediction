from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import clean_rating_data, clean_sentiment_data
from src.features import create_tfidf
from src.train import train_svm
from src.evaluate import evaluate
from src.model_utils import save_model


def train_rating_model():
    df = load_data()
    df = clean_rating_data(df)

    X = df["review"]
    y = df["rating"]

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
    df = load_data()
    df = clean_sentiment_data(df)

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = create_tfidf()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = train_svm(X_train_tfidf, y_train)
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