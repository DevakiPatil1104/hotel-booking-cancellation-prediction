from io import StringIO
import pandas as pd
import plotly.express as px
import streamlit as st
from src.model_manager import load_artifacts

st.set_page_config(page_title="Model Performance", layout="wide")

st.header("Classifier Evaluation and Comparative Metrics")

try:
  models, scaler, feature_columns, raw_metrics = load_artifacts()

  # 1. Parse raw_metrics dictionary
  if isinstance(raw_metrics, dict) and "metrics" in raw_metrics:
    metrics_data = raw_metrics["metrics"]
    if isinstance(metrics_data, str):
      # Parse the formatted string table directly into a DataFrame
      metrics_df = pd.read_csv(StringIO(metrics_data), sep=r"\s+")
    elif isinstance(metrics_data, pd.DataFrame):
      metrics_df = metrics_data
    else:
      metrics_df = pd.DataFrame(metrics_data)
  elif isinstance(raw_metrics, pd.DataFrame):
    metrics_df = raw_metrics
  else:
    metrics_df = pd.DataFrame(raw_metrics)

  # Cleanup unnamed index columns if present
  metrics_df = metrics_df.loc[
      :, ~metrics_df.columns.str.contains("^Unnamed")
  ]

  # Clean column names (normalize F1-Score / F1 Score)
  metrics_df.columns = [col.replace("-", " ") for col in metrics_df.columns]

  # 2. Performance Overview Table
  st.subheader("Comparative Benchmark Metrics")

  display_df = metrics_df.copy()
  for col in display_df.columns:
    if col != "Model":
      display_df[col] = pd.to_numeric(display_df[col], errors="ignore")
      display_df[col] = display_df[col].apply(
          lambda x: (
              f"{x * 100:.2f}%"
              if isinstance(x, (int, float)) and x <= 1.0
              else str(x)
          )
      )

  st.dataframe(display_df, use_container_width=True)

  st.markdown("---")

  # 3. Metric Comparative Bar Visualizer
  st.subheader("Metric Comparison Across Models")

  metric_cols = [col for col in metrics_df.columns if col != "Model"]
  for col in metric_cols:
    metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

  if metric_cols:
    metric_choice = st.selectbox(
        "Select Metric to Compare:", metric_cols, index=0
    )

    fig_metric = px.bar(
        metrics_df,
        x="Model",
        y=metric_choice,
        color=metric_choice,
        color_continuous_scale="Viridis",
        text_auto=".3f",
        template="plotly_dark",
    )
    fig_metric.update_layout(
        yaxis_title=metric_choice, height=400, coloraxis_showscale=False
    )
    st.plotly_chart(fig_metric, use_container_width=True)

  st.markdown("---")

  # 4. Best Model Summary Card
  primary_sort_col = (
      "F1 Score"
      if "F1 Score" in metrics_df.columns
      else (metric_cols[0] if metric_cols else None)
  )

  if primary_sort_col:
    best_model_row = metrics_df.sort_values(
        by=primary_sort_col, ascending=False
    ).iloc[0]

    st.subheader("Champion Model Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
      st.metric("Top Model", str(best_model_row.get("Model", "N/A")))
    with col2:
      acc_val = best_model_row.get("Accuracy", "N/A")
      st.metric(
          "Accuracy",
          f"{acc_val*100:.2f}%"
          if isinstance(acc_val, (int, float))
          else str(acc_val),
      )
    with col3:
      f1_val = best_model_row.get("F1 Score", "N/A")
      st.metric(
          "F1 Score",
          f"{f1_val:.3f}" if isinstance(f1_val, (int, float)) else str(f1_val),
      )
    with col4:
      auc_val = best_model_row.get("ROC-AUC", "N/A")
      st.metric(
          "ROC-AUC",
          f"{auc_val:.3f}"
          if isinstance(auc_val, (int, float))
          else str(auc_val),
      )

except Exception as e:
  st.error(f"Error loading model performance metrics: {e}")