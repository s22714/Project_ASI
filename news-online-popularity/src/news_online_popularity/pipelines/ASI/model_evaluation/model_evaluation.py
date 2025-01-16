import numpy as np
from sklearn.model_selection import cross_val_score
import wandb
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    explained_variance_score
)

def calculate_metrics(model, X_train, y_train, y_test, predictions):
    mse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    abs_error = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    medae = median_absolute_error(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)

    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
    print(scores)
    
    wandb.log({"scores" : scores})
    wandb.log({"mean squared error" : mse})
    wandb.log({"r2": r2})
    wandb.log({"mean absolute error": abs_error})
    wandb.log({"root mean squared error": rmse})
    wandb.log({"median absolute error": medae})
    wandb.log({"explained variance score": evs})

    return {
        'mse': mse,
        'r2': r2,
        'abs_error': abs_error,
        'rmse': rmse,
        'medae': medae,
        'evs': evs,
        'scores': scores
    }


def print_metrics(metrics, model_name):
    print(f"\n================================== {model_name} =============================\n")
    print(f"Mean Squared Error (MSE): {metrics['mse']}")
    print(f"R-squared (R2 Score): {metrics['r2']}")
    print(f"Absolute Error: {metrics['abs_error']}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']}")
    print(f"Median Absolute Error (MedAE): {metrics['medae']}")
    print(f"Explained Variance Score (EVS): {metrics['evs']}")
    print(f"Cross val scores (scores): {metrics['scores']}")
