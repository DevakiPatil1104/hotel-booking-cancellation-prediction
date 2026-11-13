import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CAT_COLS = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
SCALE_COLS = ['lead_time', 'avg_price_per_room']


def clean_and_transform_raw(df: pd.DataFrame) -> pd.DataFrame:
  """Cleans raw hotel dataframe and constructs temporal features."""
  data = df.copy()

  # Drop unique ID if present
  if 'Booking_ID' in data.columns:
    data.drop(columns=['Booking_ID'], inplace=True)

  # Safe date parsing
  date_cols = ['arrival_year', 'arrival_month', 'arrival_date']
  if all(col in data.columns for col in date_cols):
    date_str = (
        data['arrival_year'].astype(str)
        + '-'
        + data['arrival_month'].astype(str).str.zfill(2)
        + '-'
        + data['arrival_date'].astype(str).str.zfill(2)
    )
    data['date_of_arrival'] = pd.to_datetime(
        date_str, format='%Y-%m-%d', errors='coerce'
    )
    data.drop(columns=date_cols, inplace=True)

  # Remove invalid NaT dates and zero-adult stays
  data.dropna(subset=['date_of_arrival'], inplace=True)
  data = data[data['no_of_adults'] > 0].reset_index(drop=True)

  # Handle zero prices safely
  valid_mean = data.loc[
      data['avg_price_per_room'] > 0, 'avg_price_per_room'
  ].mean()
  data['avg_price_per_room'] = data['avg_price_per_room'].replace(0, valid_mean)

  # Temporal feature extraction
  data['arrival_month'] = data['date_of_arrival'].dt.month
  data['arrival_day'] = data['date_of_arrival'].dt.day
  data['arrival_dayofweek'] = data['date_of_arrival'].dt.dayofweek
  data.drop(columns=['date_of_arrival'], inplace=True)

  # Binary target encoding: Canceled -> 1, Not_Canceled -> 0
  if 'booking_status' in data.columns:
    data['booking_status'] = data['booking_status'].apply(
        lambda x: 1 if str(x).strip().lower() == 'canceled' else 0
    )

  return data


def build_feature_pipeline(df: pd.DataFrame):
  """Encodes categorical data and splits X, y."""
  cleaned_df = clean_and_transform_raw(df)

  df_encoded = pd.get_dummies(cleaned_df, columns=CAT_COLS, drop_first=True)

  X = df_encoded.drop(columns=['booking_status'])
  y = df_encoded['booking_status']
  return X, y


def preprocess_single_input(
    input_dict: dict, feature_columns: list, scaler: StandardScaler
) -> pd.DataFrame:
  """Preprocesses a single user input for inference, matching training features exactly."""
  df_single = pd.DataFrame([input_dict])

  # Extract temporal components using standard Python datetime methods
  arrival_date = input_dict['arrival_date']
  df_single['arrival_month'] = arrival_date.month
  df_single['arrival_day'] = arrival_date.day

  # Handle both datetime.date (.weekday()) and pandas Timestamp (.dayofweek)
  if hasattr(arrival_date, 'weekday'):
    df_single['arrival_dayofweek'] = arrival_date.weekday()
  else:
    df_single['arrival_dayofweek'] = arrival_date.dayofweek

  df_single.drop(columns=['arrival_date'], inplace=True)

  # One-hot encode single record
  df_encoded = pd.get_dummies(df_single, columns=CAT_COLS)

  # Reindex columns to match exact training features, filling missing dummy columns with 0
  df_reindexed = df_encoded.reindex(columns=feature_columns, fill_value=0)

  # Apply fitted scaler to numerical features
  df_reindexed[SCALE_COLS] = scaler.transform(df_reindexed[SCALE_COLS])

  return df_reindexed