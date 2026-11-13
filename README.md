# Hotel Booking Cancellation Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hotel-cancellation-prediction-1104.streamlit.app/)

A production-ready machine learning web application built with **Streamlit**, **Scikit-Learn**, **Pandas**, and **Plotly** to predict the likelihood of hotel reservation cancellations in real time and provide actionable revenue management insights.

🔗 **Live Demo:** [https://hotel-cancellation-prediction-1104.streamlit.app/](https://hotel-cancellation-prediction-1104.streamlit.app/)

---

## Project Overview

Booking cancellations significantly impact hotel operational planning, revenue forecasting, and inventory management. This application leverages historical reservation data to classify booking profiles into **High Cancellation Risk** vs. **Low Cancellation Risk**, allowing hotel operators to optimize overbooking buffers, refund policies, and customer engagement strategies.

### Core Features
- *Interactive Home Page:* High-level project summary and system architecture overview.
- *Data Explorer:* Interactive raw dataset inspection, metadata viewer, and summary statistics.
- *Exploratory Data Analysis (EDA):* Dynamic visual analysis (lead time distributions, pricing impact, market segment volatility).
- *Model Performance:* Detailed comparative analysis across 4 trained classifiers (Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting) with accuracy, precision, recall, F1-score, and ROC-AUC curves.
- *Real-Time Prediction:* Interactive form to input customer booking details and receive immediate prediction results with confidence scores.
- *Feature Importance & Strategic Insights:* Breakdown of top predictor features and operational recommendations for revenue managers.

---

## Repository & Project Structure

```text
hotel-booking-cancellation-prediction/
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
│   ├── 1_Home.py
│   ├── 2_Data_Explorer.py
│   ├── 3_EDA_Analysis.py
│   ├── 4_Model_Performance.py
│   ├── 5_Predict_Cancellation.py
│   └── 6_Model_Insights.py
│
├── src/                              # Core python modules
│   ├── data_loader.py                # Data ingestion logic
│   ├── preprocessing.py              # Feature engineering & scaling
│   ├── model_manager.py              # Serialization & artifact loading
│   └── visualization.py              # Custom Plotly chart routines
│
├── Hotel_Booking_Cancellation_Prediction.ipynb  # Exploratory & modeling notebook
├── app.py                            # Streamlit main entrypoint
├── train_and_save.py                 # Automated training & artifact export script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/DevakiPatil1104/hotel-booking-cancellation-prediction.git](https://github.com/DevakiPatil1104/hotel-booking-cancellation-prediction.git)
   cd hotel-booking-cancellation-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
