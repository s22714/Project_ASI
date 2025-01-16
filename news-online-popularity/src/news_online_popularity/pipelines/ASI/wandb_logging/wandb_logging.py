import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances, plot_learning_curve
import pandas as pd

def wandb_logging(sqlData, X, y, X_train, X_test, y_train, y_test, model, project_name):
    df = pd.DataFrame(sqlData)
    # create wandb session
    wandb.init(project=project_name, config=model.get_params())

    wandb.config.update({"test_size": 0.2,
                         "train_len": len(X_train),
                         "test_len": len(X_test)})

    # wandb Logs
    wandb.run.log({"Dataset": wandb.Table(dataframe=df)})

    plot_learning_curve(model, X, y)
    wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test, "modello")
    plot_feature_importances(model)

    wandb.sklearn.plot_outlier_candidates(model, X, y)
    wandb.sklearn.plot_residuals(model, X, y)
    wandb.sklearn.plot_summary_metrics(model, X, y, X_test, y_test)


def end_wandb_logging(sqlData):
    wandb.finish()