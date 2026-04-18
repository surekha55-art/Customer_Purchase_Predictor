import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Train model if not already saved
if not os.path.exists("purchase_model.pkl"):
    data = {
        "age":          [25, 45, 35, 50, 23, 40, 60, 28, 33, 55],
        "income":       [30000, 80000, 50000, 120000, 25000, 70000, 
                         95000, 40000, 60000, 110000],
        "visits":       [1, 5, 3, 8, 1, 4, 6, 2, 3, 7],
        "time_on_site": [2, 15, 8, 20, 1, 12, 18, 5, 9, 17],
        "will_buy":     [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    X = df.drop("will_buy", axis=1)
    y = df["will_buy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "purchase_model.pkl")

# Load model
model = joblib.load("purchase_model.pkl")

# Prediction function
def predict_customer(age, income, visits, time_on_site):
    features = pd.DataFrame([[age, income, visits, time_on_site]],
                            columns=["age", "income", 
                                     "visits", "time_on_site"])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if probability >= 0.75:
        confidence = "High"
    elif probability >= 0.50:
        confidence = "Medium"
    else:
        confidence = "Low"

    will_buy = "Yes" if prediction == 1 else "No"
    message = "Customer will BUY!" if prediction == 1 else "Customer will NOT buy."

    return will_buy, f"{probability:.2%}", confidence, message

# Gradio UI
demo = gr.Interface(
    fn=predict_customer,
    inputs=[
        gr.Number(label="Age", value=35),
        gr.Number(label="Annual Income (£)", value=50000),
        gr.Number(label="Number of Visits", value=3),
        gr.Number(label="Time on Site (mins)", value=10)
    ],
    outputs=[
        gr.Text(label="Will Buy?"),
        gr.Text(label="Probability"),
        gr.Text(label="Confidence Level"),
        gr.Text(label="Result")
    ],
    title="Customer Purchase Predictor",
    description="""
    Predict whether a customer will make a purchase based on 
    their demographic and behavioural data.
    Built with Random Forest Classifier + FastAPI + Gradio.
    """
)

demo.launch()
