import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

MODEL_PATH = r"../data/06_models/decision_tree.pickle/2024-11-03T22.19.38.878Z/decision_tree.pickle"

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
