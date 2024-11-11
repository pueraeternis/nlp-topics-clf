from dataclasses import asdict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.src.predict import predict

MODEL_PATH = "app/models/logreg_tfidf_v_0_1_1.joblib"

app = FastAPI()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")


class TextInput(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    probabilities: dict[str, float]


@app.post("/predict", response_model=PredictionResponse)
def predict_text(input_data: TextInput):
    try:
        prediction = predict(input_data.text, model)
        return PredictionResponse(**asdict(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")
