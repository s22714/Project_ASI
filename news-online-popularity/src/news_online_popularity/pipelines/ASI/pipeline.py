"""
This is a boilerplate pipeline 'ASI'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from news_online_popularity.pipelines.ASI.data_extraction import load_data
from news_online_popularity.pipelines.ASI.data_preparation import prepare_data, split_data
from news_online_popularity.pipelines.ASI.model_training import (
    train_linear_regression,
    train_decision_tree,
    get_predictions
)
from news_online_popularity.pipelines.ASI.model_evaluation import calculate_metrics, print_metrics
from news_online_popularity.pipelines.ASI.data_analysis import analyze_data, plot_results
from news_online_popularity.pipelines.ASI.wandb_logging import wandb_logging, end_wandb_logging

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_data,
            inputs="news_data_table",
            outputs=["X","y"],
            name="data_praparation_node",
        ),
        node(
            func=split_data,
            inputs=["X","y","params:test_size","params:random_state"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="data_split_node",
        ),
        node(
            func=train_linear_regression,
            inputs=["X_train","y_train", "y_test", "X_test","X","y"],
            outputs="linear_regression",
            name="linear_model_node",
        ),
        node(
            func=get_predictions,
            inputs=["linear_regression","X_test"],
            outputs="linear_predictions",
            name="linear_predictions_node",
        ),
        node(
            func=calculate_metrics,
            inputs=["linear_regression", "X_train", "y_train","y_test","linear_predictions"],
            outputs="linear_metrics",
            name="linear_metrics_node",
        ),
        node(
            func=print_metrics,
            inputs=["linear_metrics","params:linear_regression_name"],
            outputs=None,
            name="linear_metrics_print_node",
        ),
        node(
            func=train_decision_tree,
            inputs=["X_train","y_train", "y_test", "X_test","X","y"],
            outputs="decision_tree",
            name="tree_model_node",
        ),
        node(
            func=get_predictions,
            inputs=["decision_tree","X_test"],
            outputs="tree_predictions",
            name="tree_predictions_node",
        ),
        node(
            func=calculate_metrics,
            inputs=["decision_tree", "X_train", "y_train","y_test","tree_predictions"],
            outputs="tree_metrics",
            name="tree_metrics_node",
        ),
        node(
            func=print_metrics,
            inputs=["tree_metrics","params:decision_tree_name"],
            outputs=None,
            name="tree_metrics_print_node",
        ),
    ])
