import pandas as pd

def clean_dataset(df):

    # Drop Domain column (string)
    if "Domain" in df.columns:
        df = df.drop("Domain", axis=1)

    df = df.drop_duplicates()
    df = df.fillna(0)

    return df


def separate_features_label(df):

    X = df.drop("Label", axis=1)
    y = df["Label"]

    return X, y
