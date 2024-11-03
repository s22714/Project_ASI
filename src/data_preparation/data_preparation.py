from sklearn.model_selection import train_test_split


def prepare_data(df):

    _int_features = ['n_tokens_title', 'n_tokens_content', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',
                     'num_keywords']
    for f in _int_features:
        df[f] = df[f].astype('int64')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(df.head())
    print(df.info())
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
