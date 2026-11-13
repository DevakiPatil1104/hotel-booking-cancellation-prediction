# 🏨 Hotel Booking Cancellation Prediction System

A production-ready machine learning web application built with **Streamlit**, **Scikit-Learn**, **Pandas**, and **Plotly** to predict the likelihood of hotel reservation cancellations in real time and provide actionable revenue management insights.

---

## 📌 Project Overview

Booking cancellations significantly impact hotel operational planning, revenue forecasting, and inventory management. This application leverages historical reservation data to classify booking profiles into **High Cancellation Risk** vs. **Low Cancellation Risk**, allowing hotel operators to optimize overbooking buffers, refund policies, and customer engagement strategies.

### Core Features
* **🏠 Interactive Home Page:** High-level project summary and system architecture overview.
* **📊 Data Explorer:** Interactive raw dataset inspection, metadata viewer, and summary statistics.
* **📈 Exploratory Data Analysis (EDA):** Dynamic visual analysis (lead time distributions, pricing impact, market segment volatility).
* **⚡ Model Performance:** Detailed comparative analysis across 4 trained classifiers (Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting) with accuracy, precision, recall, F1-score, and ROC-AUC curves.
* **🔮 Real-Time Prediction:** Interactive form to input customer booking details and receive immediate prediction results with confidence scores.
* **💡 Feature Importance & Strategic Insights:** Breakdown of top predictor features and operational recommendations for revenue managers.

---

## 📁 Repository & Project Structure

```text
DSA_CP_Hotel_Reservation_Cancelation_Prediction_System/
│
├── data/
│   └── Hotel Reservations.csv        # Dataset source
│
├── models/                           # Serialized model artifacts (.pkl)
│   ├── all_models.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── model_metrics.pkl
│
├── pages/                            # Multi-page Streamlit application
│   ├── 1_🏠_Home.py
│   ├── 2_📊_Data_Explorer.py
│   ├── 3_📈_EDA_Analysis.py
│   ├── 4_⚡_Model_Performance.py
│   ├── 5_🔮_Predict_Cancellation.py
│   └── 6_💡_Model_Insights.py
│
├── src/                              # Core python modules
│   ├── data_loader.py                # Data ingestion logic
│   ├── preprocessing.py             # Feature engineering & scaling
│   ├── model_manager.py              # Serialization & artifact loading
│   └── visualization.py              # Custom Plotly chart routines
│
├── app.py                            # Streamlit main entrypoint
├── train_and_save.py                 # Automated training & artifact export script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation