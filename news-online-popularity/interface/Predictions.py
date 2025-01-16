import streamlit as st
import requests
import pandas as pd
import pickle
import kedro
import sqlalchemy
import yaml

#Picking option
st.sidebar.title("Select an option")
selected_option = st.sidebar.selectbox(
    "Select a category", ["CSV", "Picking"]
)


        
if selected_option == "CSV":
    st.title("News Popularity Prediction")

    file_to_predict = st.file_uploader("Upload file.", type=["csv"])
    if file_to_predict is not None:

        df = pd.read_csv(file_to_predict, delimiter=';')

        with open('news-online-popularity\\conf\\base\\parameters.yml', 'r') as file:
            param_service = yaml.safe_load(file)
            MODEL_PATH = rf"news-online-popularity/data/06_models/{param_service['model_name']}/{param_service['model_version']}/{param_service['model_name']}"
            #MODEL_PATH = r"news-online-popularity/data/06_models/decision_tree.pickle/2024-11-03T22.19.38.878Z/decision_tree.pickle"
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict(df)

        df['predictions'] = prediction
        csv = df.to_csv(index=False).encode("utf-8")
        
        st.dataframe(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="preds.csv",
            mime="text/csv",
        )
#Picking
if selected_option == "Picking":

    st.title("Category Input Form")
    st.write("Enter the values for each category below:")
    prediction = .0

    #List of categories
    categories = [
        "n_tokens_title", 
        "n_tokens_content", 
        "n_unique_tokens", 
        "n_non_stop_words", 
        "n_non_stop_unique_tokens", 
        "num_hrefs", 
        "num_self_hrefs", 
        "num_imgs", 
        "num_videos", 
        "average_token_length", 
        "num_keywords", 
        "data_channel_is_lifestyle", 
        "data_channel_is_entertainment", 
        "data_channel_is_bus", 
        "data_channel_is_socmed", 
        "data_channel_is_tech", 
        "data_channel_is_world", 
        "kw_min_min", 
        "kw_max_min", 
        "kw_avg_min", 
        "kw_min_max", 
        "kw_max_max", 
        "kw_avg_max", 
        "kw_min_avg", 
        "kw_max_avg", 
        "kw_avg_avg", 
        "self_reference_min_shares", 
        "self_reference_max_shares", 
        "self_reference_avg_sharess", 
        "weekday_is_monday", 
        "weekday_is_tuesday", 
        "weekday_is_wednesday", 
        "weekday_is_thursday", 
        "weekday_is_friday", 
        "weekday_is_saturday", 
        "weekday_is_sunday", 
        "is_weekend", 
        "LDA_00", 
        "LDA_01", 
        "LDA_02", 
        "LDA_03", 
        "LDA_04", 
        "global_subjectivity", 
        "global_sentiment_polarity", 
        "global_rate_positive_words", 
        "global_rate_negative_words", 
        "rate_positive_words", 
        "rate_negative_words", 
        "avg_positive_polarity", 
        "min_positive_polarity", 
        "max_positive_polarity", 
        "avg_negative_polarity", 
        "min_negative_polarity", 
        "max_negative_polarity", 
        "title_subjectivity", 
        "title_sentiment_polarity", 
        "abs_title_subjectivity",
        "abs_title_sentiment_polarity"
        
    ]

   
    input_values = {}

    #Creating input fields for each category
    cols = st.columns(3)
    for index, category in enumerate(categories):
        with cols[index % 3]:
            input_values[category] = st.text_input(f"{category}")

    #Submit button
    if st.button("Submit"):
        #Set empty fields as 0
        features = {key: (float(value) if value else 0) for key, value in input_values.items()}

        #Create a DataFrame from the input values
        df = pd.DataFrame([features])
        st.write("Entered Values")
        st.dataframe(df)

        #Sending to API
        url = "http://127.0.0.1:8000/predict"
        response = requests.post(url, json={"features": features})
        response.raise_for_status()
        prediction = response.json().get("prediction")
        st.write(f"Prediction: {prediction}")

    if st.button("Add to database"):
        with open('news-online-popularity\\conf\\local\\credentials.yml', 'r') as file:
            prime_service = yaml.safe_load(file)

        connection_string = prime_service['my_mysql_creds']['con']

        engine = sqlalchemy.create_engine(connection_string)
        features = {key: (float(value) if value else 0) for key, value in input_values.items()}
        features.update({'shares' : prediction})
        #Create a DataFrame from the input values
        df = pd.DataFrame([features])
        df.to_sql(con=engine, name='newspop',if_exists="append",index=False)