from sklearn.model_selection import train_test_split


def prepare_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)