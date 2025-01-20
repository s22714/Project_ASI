import pandas as pd
import streamlit as st
import sqlalchemy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression
import yaml

st.set_page_config(page_title="Database")

with open(f'news-online-popularity/conf/local/credentials.yml', 'r') as file:
    prime_service = yaml.safe_load(file)

connection_string = prime_service['my_mysql_creds']['con']
if not connection_string:
    st.text("Could not read data")
else:
    engine = sqlalchemy.create_engine(connection_string)

    query = "SELECT * FROM newspop"
    news_data = pd.read_sql(query, engine)



    edited_data = st.data_editor(news_data)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('Save data'):
            edited_data.to_sql(con=engine, name='newspop',if_exists="replace",index=False)
            st.rerun()
    with col2:
        if st.button('Remove rows with empty values'):
            edited_data = edited_data.dropna()
            edited_data.to_sql(con=engine, name='newspop',if_exists="replace",index=False)
            st.rerun()
    with col3:
        if st.button('Put 0 into empty fields'):
            edited_data = edited_data.fillna(0)
            edited_data.to_sql(con=engine, name='newspop',if_exists="replace",index=False)
            st.rerun()
    with col4:
        if st.button('Remove duplicates'):
            edited_data = edited_data.drop_duplicates()
            edited_data.to_sql(con=engine, name='newspop',if_exists="replace",index=False)
            st.rerun()

    st.text(f'Number of rows {len(news_data.index)}')
    st.text(f'Number of Null cells {news_data.isnull().sum().sum()}')

    new_data = st.file_uploader("Upload file.", type=["csv"])

    if new_data is not None:
        if st.button('Upload data'):
            newdf = pd.read_csv(new_data)
            newdf.columns = newdf.columns.str.strip()
            newdf = newdf.drop(columns=['url', 'timedelta'], errors='ignore')
            newdf = newdf.dropna()
            newdf.to_sql(con=engine, name='newspop',if_exists="append",index=False)
            st.rerun()



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
