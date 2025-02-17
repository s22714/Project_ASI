import pandas as pd


def load_data(df):
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['url', 'timedelta'])
    return df
