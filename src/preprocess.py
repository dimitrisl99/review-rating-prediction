import pandas as pd

def clean_rating_data(df):
    df = df[["reviewText", "overall"]]
    df = df.dropna(subset=["reviewText", "overall"])
    df = df.rename(columns={"reviewText": "review", "overall": "rating"})
    df["rating"] = df["rating"].astype(int)
    return df


def clean_sentiment_data(df):
    df = clean_rating_data(df)

    # Drop neutral / ambiguous reviews
    df = df[df["rating"] != 3]

    # 1-2 = negative, 4-5 = positive
    df["sentiment"] = df["rating"].apply(
        lambda rating: "negative" if rating <= 2 else "positive"
    )

    # Balance dataset: same number of positive and negative examples
    negative_df = df[df["sentiment"] == "negative"]
    positive_df = df[df["sentiment"] == "positive"]

    min_count = min(len(negative_df), len(positive_df))

    negative_sample = negative_df.sample(n=min_count, random_state=42)
    positive_sample = positive_df.sample(n=min_count, random_state=42)

    balanced_df = (
        pd.concat([negative_sample, positive_sample])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    print("\nBalanced sentiment distribution:")
    print(balanced_df["sentiment"].value_counts())

    return balanced_df[["review", "sentiment"]]


# Για να μη σπάσει το υπάρχον main.py
def clean_data(df):
    return clean_rating_data(df)