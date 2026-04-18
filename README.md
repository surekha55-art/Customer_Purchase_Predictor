## 📌 Overview

An end-to-end Machine Learning web application that predicts whether a customer is likely to make a purchase, based on their demographic and behavioural data.

This project demonstrates a full ML pipeline — from model training to REST API deployment and an interactive web interface — built to showcase practical data science and software engineering skills.

---

## 🎯 Problem Statement

Retail businesses need to identify which customers are most likely to make a purchase. By predicting purchasing behaviour, businesses can:

- Target the right customers at the right time
- Optimise marketing spend and promotions
- Improve conversion rates
- Personalise customer experiences

This is highly relevant in eCommerce and retail analytics environments like NEXT, where data-driven decisions directly impact revenue.

---

## 🏗️ Project Architecture

```
Customer Data (Input)
        ↓
Random Forest Classifier (ML Model)
        ↓
FastAPI REST API (Backend)
        ↓
Gradio Web Interface (Frontend)
        ↓
Prediction Output + Confidence Score
```

---

## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Library:** Scikit-learn
- **Model saved as:** `purchase_model.pkl` (Joblib serialisation)

### Why Random Forest?
- Handles mixed data types (age, income, visits, time)
- Resistant to overfitting on small datasets
- Produces probability scores, not just binary predictions
- Provides feature importance for business insights

---

## 📥 Input Features

| Feature | Type | Description |
|---|---|---|
| Age | Integer | Customer age in years |
| Income | Float | Annual income (£) |
| Visits | Integer | Number of website visits |
| Time on Site | Float | Minutes spent on site per session |

---

## 📤 Output

| Output | Description |
|---|---|
| Will Buy | True / False binary prediction |
| Probability | Score between 0.0 and 1.0 |
| Confidence | High / Medium / Low rating |
| Message | Human-readable result summary |

### Confidence Score Logic

| Probability | Confidence Level |
|---|---|
| ≥ 0.75 | 🟢 High |
| ≥ 0.50 | 🟡 Medium |
| < 0.50 | 🔴 Low |

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Machine Learning | Scikit-learn | Model training and prediction |
| Data Processing | Pandas, NumPy | Data manipulation |
| Model Serialisation | Joblib | Save and load trained model |
| REST API Backend | FastAPI | Expose prediction endpoint |
| API Server | Uvicorn | Run the FastAPI server |
| Frontend UI | Gradio | Interactive web interface |
| Development Environment | Google Colab | Notebook-based development |
| Language | Python 3.12 | Core language |

---

## 📁 Project Structure

```
customer-purchase-predictor/
│
├── customer_purchase_predictor.ipynb  # Main Colab notebook (full pipeline)
├── main.py                            # FastAPI backend API
├── purchase_model.pkl                 # Serialised trained ML model
└── README.md                          # Project documentation
```

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/surekha55-art/customer-purchase-predictor
cd customer-purchase-predictor
```

### Step 2 — Install dependencies
```bash
pip install pandas scikit-learn fastapi uvicorn gradio joblib numpy
```

### Step 3 — Open in Google Colab
- Upload `customer_purchase_predictor.ipynb` to Google Colab
- Run all cells in order from top to bottom
- The Gradio interface will launch automatically with a public link

### Step 4 — Use the API directly (optional)
```bash
uvicorn main:app --reload
```
Then send a POST request to `http://127.0.0.1:8000/predict`

```json
{
  "age": 45,
  "income": 80000,
  "visits": 5,
  "time_on_site": 15
}
```

---

## 💡 How It Works — Step by Step

1. **Data Preparation** — Customer features are structured into a Pandas DataFrame
2. **Model Training** — Random Forest Classifier is trained using Scikit-learn
3. **Model Saving** — Trained model is serialised to `purchase_model.pkl` using Joblib
4. **API Layer** — FastAPI exposes a `/predict` POST endpoint that loads the saved model
5. **Confidence Logic** — Raw probability scores are mapped to High / Medium / Low confidence
6. **UI Layer** — Gradio provides a user-friendly interactive web interface for real-time predictions

---

## 📊 Sample Predictions

| Customer | Age | Income | Visits | Time on Site | Prediction | Probability | Confidence |
|---|---|---|---|---|---|---|---|
| Customer 1 | 45 | £80,000 | 5 | 15 mins | ✅ Will BUY | 0.95 | 🟢 High |
| Customer 2 | 23 | £25,000 | 1 | 2 mins | ❌ Will NOT buy | 0.00 | 🔴 Low |
| Customer 3 | 35 | £60,000 | 3 | 9 mins | ❌ Will NOT buy | 0.18 | 🔴 Low |

---

## 🌐 API Endpoints

### GET /
Health check — confirms the API is running.

**Response:**
```json
{
  "message": "API is running!"
}
```

### POST /predict
Accepts customer features and returns a purchase prediction.

**Request Body:**
```json
{
  "age": 45,
  "income": 80000,
  "visits": 5,
  "time_on_site": 15
}
```

**Response:**
```json
{
  "will_buy": true,
  "probability": 0.95,
  "confidence": "High",
  "message": "Customer will BUY!"
}
```

---

## ⚠️ Note on Dataset

This project uses a small synthetic demo dataset to demonstrate the end-to-end ML pipeline architecture. The primary focus is on building a production-style system including:

- Model training and serialisation with Joblib
- REST API deployment with FastAPI and Uvicorn
- Interactive user interface with Gradio
- Probability-based confidence scoring logic

A production version would use a full customer transaction dataset for robust model training and evaluation.

---

## 🔮 Future Improvements

- [ ] Train on a larger real-world retail dataset (e.g. from Kaggle)
- [ ] Add feature importance visualisation chart
- [ ] Add model performance metrics dashboard
- [ ] Deploy permanently on Hugging Face Spaces
- [ ] Add SHAP values for explainability
- [ ] Add customer segmentation using clustering

---

## 🛠️ Skills Demonstrated

`Python` `Machine Learning` `Random Forest Classifier` `FastAPI` `REST API Design` `Gradio` `Scikit-learn` `Pandas` `NumPy` `Joblib` `Model Deployment` `End-to-End ML Pipeline` `Google Colab`

---

## 👩‍💻 Author

**Surekha Chinthala**
- 💼 Data Scientist | Sales Consultant at NEXT
- 🔗 LinkedIn: www.linkedin.com/in/surekha-chinthala-304915275/
- 📧 surekha.datascientist55@gmail.com

---

*Built as part of my personal data science portfolio while transitioning into a UK-based data science / software engineering role.*

