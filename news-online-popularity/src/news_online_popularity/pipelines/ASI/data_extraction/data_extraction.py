import pandas as pd


def load_data(df):
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['url', 'timedelta'])
    #pd.set_option('display.max_columns', None)
    print(df.head())
    print(df.describe())
    return df
