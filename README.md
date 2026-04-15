# ♻️ SustainMe — Lifestyle Sustainability Recommendation System

A machine learning project that predicts your sustainability score 
and gives personalized recommendations based on people similar to you.

## 🌟 Features
- Predicts sustainability score (1–5) using XGBoost
- Content Based Filtering recommendations
- Personalized advice with Easy / Medium / Hard steps
- Clean Streamlit web UI

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- XGBoost (prediction model)
- Scikit-learn (cosine similarity, preprocessing)
- Streamlit (web UI)

## 🚀 How to Run
1. Install dependencies:
pip install streamlit xgboost scikit-learn pandas numpy lightgbm

2. Run the app:
streamlit run sustainme_app.py

## 📊 Model Performance
- R2 Score : 0.70
- MAE      : 0.58
- RMSE     : 0.86