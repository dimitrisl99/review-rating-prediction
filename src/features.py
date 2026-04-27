from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf():
    return TfidfVectorizer(max_features=5000, ngram_range=(1,2))