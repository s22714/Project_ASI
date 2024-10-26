import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    explained_variance_score
)


def calculate_metrics(y_test, predictions):
    mse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    abs_error = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    medae = median_absolute_error(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)

    return {
        'mse': mse,
        'r2': r2,
        'abs_error': abs_error,
        'rmse': rmse,
        'medae': medae,
        'evs': evs
    }


def print_metrics(metrics, model_name):
    print(f"\n================================== {model_name} =============================\n")
    print(f"Mean Squared Error (MSE): {metrics['mse']}")
    print(f"R-squared (R2 Score): {metrics['r2']}")
    print(f"Absolute Error: {metrics['abs_error']}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']}")
    print(f"Median Absolute Error (MedAE): {metrics['medae']}")
    print(f"Explained Variance Score (EVS): {metrics['evs']}")
