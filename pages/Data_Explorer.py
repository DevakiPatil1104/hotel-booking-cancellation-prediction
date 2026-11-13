import streamlit as st
import pandas as pd
from src.data_loader import load_raw_data

st.set_page_config(page_title="Data Explorer", layout="wide")

st.header("Dataset Inspection and Quality Assurance")

try:
    df = load_raw_data()
    
    # 1. High-Level Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{df.shape[1]}")
    with col3:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_count}")
    with col4:
        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicate_count}")

    st.markdown("---")

    # 2. Interactive Data Viewers
    tab1, tab2, tab3 = st.tabs(["Raw Dataset", "Data Structure & Types", "Summary Statistics"])

    with tab1:
        st.subheader("Raw Booking Records")
        rows_to_display = st.slider("Select number of rows to preview:", min_value=5, max_value=100, value=10, step=5)
        st.dataframe(df.head(rows_to_display), use_container_width=True)

    with tab2:
        st.subheader("Feature Schema and Null Analysis")
        data_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Unique Values": [df[col].nunique() for col in df.columns]
        })
        st.dataframe(data_info, use_container_width=True)

    with tab3:
        st.subheader("Numerical Summary Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

except Exception as e:
    st.error(f"Error loading dataset: {e}")