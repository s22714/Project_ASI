import wandb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def train_linear_regression(X_train, y_train):
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model


def get_predictions(model, X_test):
    return model.predict(X_test)
