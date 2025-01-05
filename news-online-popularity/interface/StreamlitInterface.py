import streamlit as st
import requests

category = [
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
    "abs_title_subjectivity"
]



if "selected_options" not in st.session_state:
    st.session_state["selected_options"] = {}



st.title("News Popularity Prediction")

#Picking category
st.sidebar.title("Select an option")
selected_option = st.sidebar.selectbox(
    "Select a category", category
)

#Picking number for picked category
with st.form("add_item_form"):
    st.write("Enter a new value for the selected category.")
    value = st.number_input("Enter a value:", value=0.0, step=0.1)
    submitted = st.form_submit_button("Update")

    if submitted:
        #Update the value if category already in list
        st.session_state["selected_options"][selected_option] = value

#Showing the list
st.header("Selected Options")
if st.session_state["selected_options"]:
    for i, (category_name, category_value) in enumerate(st.session_state["selected_options"].items()):
        st.write(f"{i + 1}. {category_name}: {category_value}")
else:
    st.write("No options selected.")

#Adding missing categories so it won't freakout
def get_features():

    for cat in category:
        if cat not in st.session_state["selected_options"]:
            st.session_state["selected_options"][cat] = 0.0

    return st.session_state["selected_options"]



#Prediction button (not done properly yet)
if st.button("Predict"):
    features = get_features()
    if features:
        #sending to api
        url = "http://127.0.0.1:8000/predict"
        response = requests.post(url, json={"features": features})
        prediction = response.json().get("prediction")
        st.write(f"Prediction: {prediction}")

   