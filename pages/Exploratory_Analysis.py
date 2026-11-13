import pandas as pd
import plotly.express as px
import plotly.graph_objects as bg
import streamlit as st
from src.data_loader import load_raw_data
from src.preprocessing import clean_and_transform_raw

st.set_page_config(page_title="EDA Analysis", layout="wide")

st.header("Exploratory Data Analysis")

try:
  raw_df = load_raw_data()
  df = clean_and_transform_raw(raw_df)

  # 1. Target Distribution & Lead Time
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Target Distribution: Booking Status")
    status_counts = (
        df["booking_status"]
        .value_counts()
        .reset_index()
        .replace({"booking_status": {1: "Canceled", 0: "Not Canceled"}})
    )
    status_counts.columns = ["Status", "Count"]

    fig_target = px.pie(
        status_counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Canceled": "#EF4444", "Not Canceled": "#10B981"},
        hole=0.4,
        template="plotly_dark",
    )
    fig_target.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_target, use_container_width=True)

  with col2:
    st.subheader("Lead Time Distribution by Cancellation Status")
    df_plot = df.copy()
    df_plot["Status"] = df_plot["booking_status"].map(
        {1: "Canceled", 0: "Not Canceled"}
    )

    fig_lead = px.box(
        df_plot,
        x="Status",
        y="lead_time",
        color="Status",
        color_discrete_map={"Canceled": "#EF4444", "Not Canceled": "#10B981"},
        points=False,
        template="plotly_dark",
    )
    fig_lead.update_layout(
        yaxis_title="Lead Time (Days)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_lead, use_container_width=True)

  st.markdown("---")

  # 2. Market Segment & Special Requests Analysis
  col3, col4 = st.columns(2)

  with col3:
    st.subheader("Cancellation Rate by Market Segment")
    segment_df = (
        df.groupby("market_segment_type")["booking_status"]
        .agg(["count", "mean"])
        .reset_index()
    )
    segment_df["Cancellation Rate (%)"] = segment_df["mean"] * 100
    segment_df = segment_df.sort_values(
        by="Cancellation Rate (%)", ascending=False
    )

    fig_seg = px.bar(
        segment_df,
        x="market_segment_type",
        y="Cancellation Rate (%)",
        color="Cancellation Rate (%)",
        color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig_seg.update_layout(
        xaxis_title="Market Segment", height=380, coloraxis_showscale=False
    )
    st.plotly_chart(fig_seg, use_container_width=True)

  with col4:
    st.subheader("Impact of Special Requests on Cancellation")
    req_df = (
        df.groupby("no_of_special_requests")["booking_status"]
        .agg(["count", "mean"])
        .reset_index()
    )
    req_df["Cancellation Rate (%)"] = req_df["mean"] * 100

    fig_req = px.line(
        req_df,
        x="no_of_special_requests",
        y="Cancellation Rate (%)",
        markers=True,
        template="plotly_dark",
    )
    fig_req.update_traces(
        line_color="#3B82F6", line_width=3, marker_size=8
    )
    fig_req.update_layout(
        xaxis_title="Number of Special Requests",
        yaxis_title="Cancellation Rate (%)",
        height=380,
    )
    st.plotly_chart(fig_req, use_container_width=True)

  st.markdown("---")

  # 3. Monthly Trends
  st.subheader("Monthly Booking Volatility and Cancellation Rates")
  monthly_df = (
      df.groupby("arrival_month")["booking_status"]
      .agg(["count", "mean"])
      .reset_index()
  )
  monthly_df["Cancellation Rate (%)"] = monthly_df["mean"] * 100
  month_names = {
      1: "Jan",
      2: "Feb",
      3: "Mar",
      4: "Apr",
      5: "May",
      6: "Jun",
      7: "Jul",
      8: "Aug",
      9: "Sep",
      10: "Oct",
      11: "Nov",
      12: "Dec",
  }
  monthly_df["Month"] = monthly_df["arrival_month"].map(month_names)

  fig_month = px.bar(
      monthly_df,
      x="Month",
      y="Cancellation Rate (%)",
      color="Cancellation Rate (%)",
      color_continuous_scale="Viridis",
      template="plotly_dark",
  )
  fig_month.update_layout(height=380)
  st.plotly_chart(fig_month, use_container_width=True)

except Exception as e:
  st.error(f"Error executing EDA pipeline: {e}")