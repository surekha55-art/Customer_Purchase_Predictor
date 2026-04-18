import gradio as gr
import requests

# ================================
# CUSTOMER PURCHASE PREDICTOR
# Step 3: Gradio Frontend UI
# ================================

def predict_customer(age, income, visits, time_on_site):
    customer = {
        "age": int(age),
        "income": float(income),
        "visits": int(visits),
        "time_on_site": float(time_on_site)
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=customer)
    result = response.json()

    return (
        f"Will Buy: {result['will_buy']}",
        f"Probability: {result['probability']}",
        f"Confidence: {result['confidence']}",
        f"Message: {result['message']}"
    )


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
        gr.Text(label="Confidence"),
        gr.Text(label="Message")
    ],
    title="🛒 Customer Purchase Predictor",
    description="Enter customer details to predict whether they will make a purchase."
)

demo.launch(share=True)
