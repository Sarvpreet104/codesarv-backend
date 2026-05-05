# imports

## fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

## services
from services.load_model import MODELS
from services.predict import make_predictions
from services.sample_data import take_one_sample

## pydantic
from pydantic import BaseModel


# type class
class PredictionInput(BaseModel):
    data: list[dict] | dict

# app
app = FastAPI()

# middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# get routes

## home
@app.get("/")
def home():
    return {"message": "codeSarv API is running."}

## pick data
@app.get("/pick-data/{model_name}")
def pick_data(model_name: str):
    test_data_path = MODELS.get(model_name).get("test_data_path")
    sample = take_one_sample(test_data_path=test_data_path)

    return {"test_data": sample}

# post routes

## predict dynamically
@app.post("/predict/{model_name}")
def predict(model_name: str, payload: PredictionInput):
    model = MODELS.get(model_name).get("model")

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    prediction = make_predictions(model, payload.data)

    return {
        "model": model_name,
        "prediction": prediction
    }