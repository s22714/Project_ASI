from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_extraction import load_data
from src.data_preparation import prepare_data, split_data
from src.model_training import (
    train_linear_regression,
    train_decision_tree,
    get_predictions
)
from src.model_evaluation import calculate_metrics, print_metrics
from src.data_analysis import analyze_data, plot_results

import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances, plot_learning_curve


def wandb_logging(test_size, df, X, y, X_train, X_test, y_train, y_test, model, project_name):
    # create wandb session
    wandb.init(project=project_name, config=model.get_params())

    wandb.config.update({"test_size": test_size,
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


def main():

    # Data extraction
    df = load_data()

    # Data preparation
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Model training
    linear_model = train_linear_regression(X_train, y_train)
    decision_tree_model = train_decision_tree(X_train, y_train)

    # Get predictions
    linear_predictions = get_predictions(linear_model, X_test)
    dec_tree_preds = get_predictions(decision_tree_model, X_test)

    # Model evaluation
    wandb_logging(0.2, df, X, y, X_train, X_test, y_train, y_test, linear_model, "ASI_linear_regression_2")
    linear_metrics = calculate_metrics(y_test, linear_predictions)
    wandb.finish()
    wandb_logging(0.2, df, X, y, X_train, X_test, y_train, y_test, decision_tree_model, "ASI_tree_model_2")
    tree_metrics = calculate_metrics(y_test, dec_tree_preds)
    wandb.finish()
    # Print evaluation results
    print_metrics(linear_metrics, "linear regression")
    print_metrics(tree_metrics, "decision trees")

    # Data analysis
    min_outliers, max_outliers, feature_importance, correlation_matrix = analyze_data(df)
    plot_results(min_outliers, max_outliers, feature_importance, correlation_matrix)




if __name__ == "__main__":
    main()
