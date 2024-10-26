import pandas as pd


def load_data(file_path="data/OnlineNewsPopularity.csv"):
    df = pd.read_csv(file_path, index_col=False).drop('url', axis=1)
    df.columns = df.columns.str.strip()
    return df
