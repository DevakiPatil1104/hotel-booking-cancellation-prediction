import datetime
import pandas as pd
import streamlit as st
from src.model_manager import load_artifacts
from src.preprocessing import preprocess_single_input

st.set_page_config(page_title="Predict Cancellation Risk", layout="wide")

st.header("Real-Time Cancellation Risk Prediction")

try:
  models, scaler, feature_columns, _ = load_artifacts()

  selected_model_name = st.selectbox(
      "Select Classification Model:", list(models.keys())
  )
  model = models[selected_model_name]

  with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
      adults = st.number_input(
          "Number of Adults", min_value=1, max_value=10, value=2
      )
      children = st.number_input(
          "Number of Children", min_value=0, max_value=10, value=0
      )
      lead_time = st.number_input(
          "Lead Time (Days)", min_value=0, max_value=500, value=30
      )
      arrival_date = st.date_input("Arrival Date", value=datetime.date.today())

    with col2:
      weekend_nights = st.number_input(
          "Weekend Nights", min_value=0, max_value=10, value=1
      )
      week_nights = st.number_input(
          "Week Nights", min_value=0, max_value=20, value=2
      )
      price = st.number_input(
          "Average Room Price (€)",
          min_value=1.0,
          max_value=1000.0,
          value=100.0,
      )
      market_segment = st.selectbox(
          "Market Segment",
          ["Online", "Offline", "Corporate", "Aviation", "Complementary"],
      )

    with col3:
      meal_plan = st.selectbox(
          "Meal Plan",
          ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
      )
      room_type = st.selectbox(
          "Room Type",
          [
              "Room_Type 1",
              "Room_Type 2",
              "Room_Type 3",
              "Room_Type 4",
              "Room_Type 5",
              "Room_Type 6",
              "Room_Type 7",
          ],
      )
      parking = st.selectbox("Requires Car Parking?", [0, 1])
      repeated = st.selectbox("Repeated Guest?", [0, 1])
      prev_canc = st.number_input(
          "Previous Cancellations", min_value=0, max_value=20, value=0
      )
      prev_not_canc = st.number_input(
          "Previous Non-Cancellations", min_value=0, max_value=50, value=0
      )
      special_req = st.number_input(
          "Special Requests", min_value=0, max_value=5, value=0
      )

    submitted = st.form_submit_button("Predict Cancellation Risk")

  if submitted:
    input_data = {
        "no_of_adults": adults,
        "no_of_children": children,
        "no_of_weekend_nights": weekend_nights,
        "no_of_week_nights": week_nights,
        "type_of_meal_plan": meal_plan,
        "required_car_parking_space": parking,
        "room_type_reserved": room_type,
        "lead_time": lead_time,
        "market_segment_type": market_segment,
        "repeated_guest": repeated,
        "no_of_previous_cancellations": prev_canc,
        "no_of_previous_bookings_not_canceled": prev_not_canc,
        "avg_price_per_room": price,
        "no_of_special_requests": special_req,
        "arrival_date": arrival_date,
    }

    X_processed = preprocess_single_input(input_data, feature_columns, scaler)
    prediction = model.predict(X_processed)[0]
    probs = (
        model.predict_proba(X_processed)[0]
        if hasattr(model, "predict_proba")
        else None
    )

    st.markdown("---")
    st.subheader("Assessment Result")

    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
      if prediction == 1:
        st.error(
            "High Cancellation Risk: This reservation exhibits structural"
            " indicators of potential cancellation."
        )
      else:
        st.success(
            "Low Cancellation Risk: This reservation exhibits high commitment"
            " and is likely to honor the stay."
        )

    with res_col2:
      if probs is not None:
        canc_prob = probs[1] * 100
        conf_prob = probs[0] * 100
        st.metric("Cancellation Probability", f"{canc_prob:.1f}%")
        st.metric("Confirmation Probability", f"{conf_prob:.1f}%")

except FileNotFoundError:
  st.warning(
      "Model artifacts not found in `models/`. Execute `python"
      " train_and_save.py` to build the required artifacts."
  )
except Exception as e:
  st.error(f"An unexpected error occurred: {e}")