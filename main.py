# main.py

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load your saved model
full_model = joblib.load('hall_id_full_model.pkl')

# 2. Create FastAPI app
app = FastAPI()

# 3. Define input format
class ModelInput(BaseModel):
    Date: str
    Start_time: str
    End_time: str
    proj: int
    computers: int
    students: int

# 4. Home endpoint
@app.get("/")
def home():
    return {"message": "Hello World from FastAPI!"}

# 5. Prediction endpoint
@app.post("/predict")
async def predict(input: ModelInput):
    input_data = input.dict()
    df_input = pd.DataFrame([input_data])

    prediction = full_model.predict(df_input)

    return {"predicted_hall_id": prediction.tolist()}