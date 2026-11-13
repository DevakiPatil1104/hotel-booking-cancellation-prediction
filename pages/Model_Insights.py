import pandas as pd
import plotly.express as px
import streamlit as st
from src.model_manager import load_artifacts

st.set_page_config(page_title="Model Insights", layout="wide")

st.header("Feature Importance and Strategic Recommendations")

try:
  models, _, feature_columns, _ = load_artifacts()

  rf_model = (
      models.get("Random Forest")
      or models.get("RandomForest")
      or models.get("random_forest")
  )

  if rf_model is not None and hasattr(rf_model, "feature_importances_"):
    importances = (
        pd.DataFrame({
            "Feature": feature_columns,
            "Importance": rf_model.feature_importances_,
        })
        .sort_values(by="Importance", ascending=True)
        .tail(10)
    )

    fig = px.bar(
        importances,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Predictors of Reservation Cancellation (Random Forest)",
        color="Importance",
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig.update_layout(
        xaxis_title="Relative Importance Weight",
        yaxis_title="Feature Name",
        height=480,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
  else:
    st.info(
        "Random Forest model feature importances are not available in the"
        " loaded artifacts."
    )

  st.markdown("---")

  st.subheader("Operational and Revenue Management Takeaways")

  col_a, col_b = st.columns(2)

  with col_a:
    st.markdown("""
        **1. Lead Time Controls**
        * **Finding:** Lead time is consistently the highest-weighted feature driving cancellation probability.
        * **Strategy:** Implement non-refundable deposit terms or automated re-confirmation checks for reservations made more than 60 days in advance.
        """)

    st.markdown("""
        **2. Booking Channel Calibration**
        * **Finding:** Online travel agency (OTA) bookings exhibit higher cancellation volatility than corporate direct channels.
        * **Strategy:** Adjust overbooking allowances independently by channel based on historical retention rates.
        """)

  with col_b:
    st.markdown("""
        **3. Guest Preference Engagement**
        * **Finding:** Guests logging 1 or more special requests demonstrate significantly higher commitment and lower cancellation rates.
        * **Strategy:** Encourage preference selection during checkout to strengthen booking intent.
        """)

    st.markdown("""
        **4. Pricing Tier Thresholds**
        * **Finding:** High average daily room rates correlate with elevated cancellation risk during off-peak periods.
        * **Strategy:** Structure dynamic pricing rules that offer moderate early-bird discounts in exchange for non-refundable terms.
        """)

except FileNotFoundError:
  st.warning(
      "Model artifacts not found in `models/`. Please run `python"
      " train_and_save.py` to generate model files."
  )
except Exception as e:
  st.error(f"An unexpected error occurred while generating insights: {e}")