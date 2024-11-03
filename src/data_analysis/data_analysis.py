import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression


def analyze_data(df):
    # Find outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    min_outliers = df[df < lower_bound].min()
    max_outliers = df[df > upper_bound].max()

    upper_array = df[df > upper_bound]
    print("Upper Bound:", max_outliers)
    print(upper_array.sum())

    lower_array = df[df < lower_bound]
    print("Lower Bound:", min_outliers)
    print(lower_array.sum())

    # Calculate feature importance (F-test)
    X = df.drop('shares', axis=1)
    y = df['shares']
    f_values, p_values = f_regression(X, y)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'F-value': f_values,
        'p-value': p_values
    })
    feature_importance = feature_importance.sort_values(by='F-value', ascending=False)

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    return min_outliers, max_outliers, feature_importance, correlation_matrix


def plot_results(min_outliers, max_outliers, feature_importance, correlation_matrix):
    # Plot outliers
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=min_outliers.index, y=min_outliers.values)
    plt.xticks(rotation=90)
    plt.title('Minimalne wartości odstające')
    plt.subplot(1, 2, 2)
    sns.barplot(x=max_outliers.index, y=max_outliers.values)
    plt.xticks(rotation=90)
    plt.title('Maksymalne wartości odstające')
    plt.tight_layout()
    plt.show()

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='F-value', y='Feature', data=feature_importance.head(20), palette='viridis')
    plt.title('Top 20 cech według istotności (F-value)')
    plt.xlabel('F-value')
    plt.ylabel('Feature')
    plt.show()

    # Plot correlation matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title('Macierz korelacji')
    plt.show()

