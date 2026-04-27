import pandas as pd

def load_data():

    url = "https://huggingface.co/datasets/hugginglearners/amazon-reviews-sentiment-analysis/resolve/main/amazon_reviews.csv?download=true"
    df = pd.read_csv(url)
    return df