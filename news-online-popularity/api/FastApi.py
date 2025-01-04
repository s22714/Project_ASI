import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import streamlit as st

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




MODEL_PATH = r"../data/06_models/decision_tree.pickle/2024-11-03T22.19.38.878Z/decision_tree.pickle"


st.title("News popularity prediction")
st.write("Click 'Predict' to get a prediction.")



import streamlit as st
 
 
st.sidebar.title("Select option")

whatLasts = category; 

selected_option = st.sidebar.selectbox("Select option", whatLasts)




   
 


st.title("selected_option")


if "option_list" not in st.session_state:
    st.session_state["option_list"] = []
with st.form("add_item_form"):
    item = st.text_input("Dodaj opcje:")
    submitted = st.form_submit_button("Dodaj")
   
    # Dodanie opcje do listy 
    if submitted and item:
        st.session_state["option_list"].append(item)
        whatLasts.remove(selected_option) 
        st.success(f"Dodano: {selected_option} {item}")
        st.experimental_rerun()
 
# Wyświetlanie listy zakupów
    st.header("Your options")
    if st.session_state["option_list"]:
        for i, product in enumerate(st.session_state["option_list"]):
            col1, col2 = st.columns([4, 1])
            col1.write(f"{i + 1}. {product}")
            # Przycisk do usunięcia konkretnej opcji
            if col2.button("Usuń", key=f"delete_{i}"):
                st.session_state["option_list"].pop(i)
                st.experimental_rerun()  # Odśwież aplikację po usunięciu elementu
    else:
        st.write("Lista jest pusta.")
 









app = FastAPI(title="Single Model Predictor")

class PredictionInput(BaseModel):
    features: dict = Field(..., description="Dictionary of input features for prediction")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        df = pd.DataFrame([input_data.features])

        prediction = model.predict(df)[0]

        return {"prediction": str(prediction)}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Model Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
