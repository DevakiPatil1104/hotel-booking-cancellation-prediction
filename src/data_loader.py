import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_raw_data(data_path: str = "data/Hotel Reservations.csv") -> pd.DataFrame:
    """Loads raw dataset from CSV with Streamlit caching."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    return pd.read_csv(data_path)