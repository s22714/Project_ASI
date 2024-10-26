from src.data_extraction import load_data
from src.data_preparation import prepare_data, split_data
from src.model_training import (
    train_linear_regression,
    train_decision_tree,
    get_predictions
)
from src.model_evaluation import calculate_metrics, print_metrics
from src.data_analysis import analyze_data, plot_results


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
    linear_metrics = calculate_metrics(y_test, linear_predictions)
    tree_metrics = calculate_metrics(y_test, dec_tree_preds)

    # Print evaluation results
    print_metrics(linear_metrics, "linear regression")
    print_metrics(tree_metrics, "decision trees")

    # Data analysis
    min_outliers, max_outliers, feature_importance, correlation_matrix = analyze_data(df)
    plot_results(min_outliers, max_outliers, feature_importance, correlation_matrix)


if __name__ == "__main__":
    main()
