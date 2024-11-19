"""
This is a boilerplate pipeline 'ASI_Autogluon'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import data_split, create_test_predictor

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=data_split,
            inputs="news_data_prepared",
            outputs=["train_set", "test_set"],
            name="data_split",
        ),
        node(
            func=create_test_predictor,
            inputs=["train_set", "test_set"],
            outputs=None,
            name="create_test_predict",
        )
    ])
