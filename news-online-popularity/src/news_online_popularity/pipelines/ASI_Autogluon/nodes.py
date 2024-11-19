"""
This is a boilerplate pipeline 'ASI_Autogluon'
generated using Kedro 0.19.9
"""

import pandas as pd
import autogluon
from autogluon.tabular import TabularDataset, TabularPredictor

def data_split(df):

    data = TabularDataset(df)
    train_size = int(39645 * 0.8)
    seed = 1080
    train_set = data.sample(train_size, random_state=seed)
    test_set = data.drop(train_set.index)

    return train_set, test_set

def create_test_predictor(train_set, test_set):
    train_data = TabularDataset(train_set)
    predictor = TabularPredictor(label='shares', eval_metric="root_mean_squared_error", path='bestModel').fit(train_data, presets="medium_quality", excluded_model_types=['NN_TORCH', 'FASTAI'], fit_weighted_ensemble=False)
    
    test_data = TabularDataset(test_set)

    predictions = predictor.predict(test_data)
    print(predictions)

    leaderboard = predictor.leaderboard()
    print(leaderboard)

    print(predictor.evaluate(train_data))