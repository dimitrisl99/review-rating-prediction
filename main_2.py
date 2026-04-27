import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
url = "https://huggingface.co/datasets/hugginglearners/amazon-reviews-sentiment-analysis/resolve/main/amazon_reviews.csv?download=true"
df = pd.read_csv(url)

print("Original shape:")
print(df.shape)


# 2. Keep only the columns we need
df = df[["reviewText", "overall"]] #αυτές τις 2 στήλες θα χρησιμοποιήσουμε (προς το παρόν)

print("\nShape after keeping only needed columns:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# 3. Remove rows with missing values
df = df.dropna(subset=["reviewText", "overall"]) #πέταξε τις γραμμές οπου λείπει review ή rating

print("\nShape after dropping missing values:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# 4. Rename columns to something simpler
df = df.rename(columns={ #Ranemae (optinal αλλά θα βοηθήσει)
    "reviewText": "review",
    "overall": "rating"
})

print("\nColumns after renaming:")
print(df.columns)


# 5. Convert rating to integer
df["rating"] = df["rating"].astype(int) #κάνουμε τα rating ακέραιους (π.χ. 5.0 --> 5)

print("\nRating data type:")
print(df["rating"].dtype)


# 6. Check class distribution
print("\nRating distribution:")
print(df["rating"].value_counts().sort_index()) #βλέπουμε το distribution ποσα έχουν 1 αστέρι πόσα 2 ...


# 7. Define X and y
X = df["review"] #όλα τα review texts
y = df["rating"] #όλα τα αντίστοιχα ratings

print("\nExample review:")
print(X.iloc[0])

print("\nExample label:")
print(y.iloc[0])


# 8. Split into train and test
X_train, X_test, y_train, y_test = train_test_split( #κάνουμε το split
    X,
    y,
    test_size=0.2, #20% test
    random_state=42
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))


# 9. Convert text to TF-IDF features
vectorizer = TfidfVectorizer( #κρατάει μονο τις 5000 πιο σημαντικές λέξεις
    max_features=5000, #λιγότερο noise , πιο σταθερό μοντέλο
    ngram_range=(1,2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF train shape:")
print(X_train_tfidf.shape)

print("TF-IDF test shape:")
print(X_test_tfidf.shape)


# 10. Train model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)


# 11. Predict
y_pred = model.predict(X_test_tfidf)

print("\nFirst 20 predictions:")
print(y_pred[:20])

print("\nFirst 20 true labels:")
print(y_test.values[:20])


# 12. Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

classes = np.unique(y_test)
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()