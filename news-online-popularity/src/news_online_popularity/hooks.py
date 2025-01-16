from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node
from typing import Any
import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances, plot_learning_curve
import pandas as pd

class WandBCallHook:

    @hook_impl
    def after_node_run(self, node: Node, inputs: dict[str, Any], outputs: dict[str, Any]):
        if node.name == "data_praparation_node":
            wandb.init(project="ASI project")
            print("########################################################### logging started ########################################")

        if node.name == "data_praparation_node":
            print(outputs["X"])
            print(outputs["y"])
            print("################################################################## X y ########################################")

        if node.name == "tree_metrics_print_node":
            wandb.finish()
            print("########################################################### logging finished ########################################")
        