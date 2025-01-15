import pandas as pd
import streamlit as st
import sqlalchemy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression
st.set_page_config(page_title="Database")

connection_string = 'mysql://root:qwerty@localhost:3306/asi_project'

engine = sqlalchemy.create_engine(connection_string)

query = "SELECT * FROM newspop"
news_data = pd.read_sql(query, engine)

st.dataframe(news_data)

def analyze_data(df):
    # Find outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    min_outliers = df[df < lower_bound].min()
    max_outliers = df[df > upper_bound].max()

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
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=min_outliers.index, y=min_outliers.values)
    plt.xticks(rotation=90)
    plt.title('Minimalne wartości odstające')
    plt.subplot(1, 2, 2)
    sns.barplot(x=max_outliers.index, y=max_outliers.values)
    plt.xticks(rotation=90)
    plt.title('Maksymalne wartości odstające')
    plt.tight_layout()
    st.pyplot(fig)

    

    # Plot feature importance
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(x='F-value', y='Feature', data=feature_importance.head(20), palette='viridis')
    plt.title('Top 20 cech według istotności (F-value)')
    plt.xlabel('F-value')
    plt.ylabel('Feature')
    st.pyplot(fig)

    # Plot correlation matrix
    fig = plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title('Macierz korelacji')
    st.pyplot(fig)

min_outliers, max_outliers, feature_importance, correlation_matrix = analyze_data(news_data)
plot_results(min_outliers, max_outliers, feature_importance, correlation_matrix)
