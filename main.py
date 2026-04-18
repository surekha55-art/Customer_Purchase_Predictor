from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ================================
# CUSTOMER PURCHASE PREDICTOR
# Step 2: FastAPI Backend
# ================================

# Load trained model
model = joblib.load("purchase_model.pkl")

app = FastAPI(title="Customer Purchase Prediction API")


class CustomerFeatures(BaseModel):
    age: int
    income: float
    visits: int
    time_on_site: float


@app.get("/")
def home():
    return {"message": "Customer Purchase Predictor API is running!"}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    # Use DataFrame to avoid feature name warnings
    features = pd.DataFrame([[
        customer.age,
        customer.income,
        customer.visits,
        customer.time_on_site
    ]], columns=["age", "income", "visits", "time_on_site"])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Confidence scoring logic
    if probability >= 0.75:
        confidence = "High"
    elif probability >= 0.50:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "will_buy": bool(prediction),
        "probability": round(float(probability), 4),
        "confidence": confidence,
        "message": "Customer will BUY!" if prediction == 1 else "Customer will NOT buy."
    }
