def clean_data(df):
    df = df[["reviewText", "overall"]]
    df = df.dropna(subset=["reviewText", "overall"])
    df = df.rename(columns={"reviewText": "review", "overall": "rating"})
    df["rating"] = df["rating"].astype(int)

    return df