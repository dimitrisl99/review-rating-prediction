from src.data_loader import load_data
from src.preprocess import clean_data
from src.features import create_tfidf
from src.train import train_svm, train_logistic
from src.evaluate import evaluate, plot_confusion_matrix
from src.error_analysis import show_misclassified_examples
from src.explain import show_top_features_per_class
from sklearn.model_selection import train_test_split
from src.model_utils import save_model

MODEL_NAME = "svm"   # επιλογές: "svm" ή "logistic"


df = load_data()
df = clean_data(df)

X = df["review"]
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = create_tfidf()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

if MODEL_NAME == "svm":
    model = train_svm(X_train_tfidf, y_train)
elif MODEL_NAME == "logistic":
    model = train_logistic(X_train_tfidf, y_train)
else:
    raise ValueError("MODEL_NAME must be either 'svm' or 'logistic'")

save_model(model, vectorizer)
print("Model and vectorizer saved successfully.")

y_pred = model.predict(X_test_tfidf)

evaluate(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)
show_misclassified_examples(X_test, y_test, y_pred, n=10)
show_top_features_per_class(model, vectorizer, n=15)

