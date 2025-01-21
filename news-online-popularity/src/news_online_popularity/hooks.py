from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node
from typing import Any
import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances, plot_learning_curve
import pandas as pd
import yaml
import os

class WandBCallHook:

    @hook_impl
    def after_node_run(self, node: Node, inputs: dict[str, Any], outputs: dict[str, Any]):

        if node.name == "data_praparation_node":

            with open(os.path.join('news-online-popularity','conf','local','credentials.yml'), 'r') as file:
                conn_str_service = yaml.safe_load(file)

            wandb.login(key=conn_str_service['wandbapikey'])
            wandb.init(project=conn_str_service['wandbprojectname'])
            df = pd.DataFrame(inputs['news_data_table'])
            df.columns = [str(col) for col in df.columns]
            wandb.run.log({"Dataset": wandb.Table(dataframe=df)})

            print("########################################################### wandb logging started ########################################")

        if node.name == "data_split_node":
            with open(os.path.join('news-online-popularity','conf','base','parameters.yml'), 'r') as file:
                param_service = yaml.safe_load(file)
            wandb.config.update({"test_size": param_service['test_size'],
                         "train_len": len(outputs['X_train']),
                         "test_len": len(outputs['X_test'])})
            
        if node.name == "linear_model_node":
            plot_learning_curve(outputs['linear_regression'], inputs['X'], inputs['y'])
            wandb.sklearn.plot_regressor(outputs['linear_regression'], inputs['X_train'], inputs['X_test'], inputs['y_train'], inputs['y_test'], "linear regression")
            plot_feature_importances(outputs['linear_regression'])

            wandb.sklearn.plot_outlier_candidates(outputs['linear_regression'], inputs['X'], inputs['y'])
            wandb.sklearn.plot_residuals(outputs['linear_regression'], inputs['X'], inputs['y'])
            wandb.sklearn.plot_summary_metrics(outputs['linear_regression'], inputs['X'], inputs['y'], inputs['X_test'], inputs['y_test'])
            
        if node.name == "tree_model_node":
            plot_learning_curve(outputs['decision_tree'], inputs['X'], inputs['y'])
            wandb.sklearn.plot_regressor(outputs['decision_tree'], inputs['X_train'], inputs['X_test'], inputs['y_train'], inputs['y_test'], "decision tree")
            plot_feature_importances(outputs['decision_tree'])

            wandb.sklearn.plot_outlier_candidates(outputs['decision_tree'], inputs['X'], inputs['y'])
            wandb.sklearn.plot_residuals(outputs['decision_tree'], inputs['X'], inputs['y'])
            wandb.sklearn.plot_summary_metrics(outputs['decision_tree'], inputs['X'], inputs['y'], inputs['X_test'], inputs['y_test'])
            
        if node.name == "tree_metrics_print_node":

            wandb.finish()

            print("########################################################### wandb logging finished ########################################")
            
        