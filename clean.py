"""
clean.py
--------
Cleans and preprocesses the raw telecom churn dataset:
 - Fixes data types
 - Handles missing values
 - Encodes categorical variables
 - Scales numerical variables
 - Saves the cleaned dataset + encoders/scaler for later use (train.py, app.py)
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

from read_data import load_raw_data

# ---------------- CONFIG ----------------
DATA_DIR = "data"
MODELS_DIR = "models"
CLEANED_FILE_PATH = os.path.join(DATA_DIR, "cleaned_churn.csv")
ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Columns that should be treated as numeric
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Target column
TARGET_COL = "Churn"

# Columns not useful for modeling
DROP_COLS = ["customerID"]


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges sometimes comes as a string with blank spaces for new
    customers (tenure = 0). Convert to numeric and fill missing with 0.
    """
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_categorical(df: pd.DataFrame):
    """
    Label-encodes all categorical (object type) columns.
    Returns the encoded dataframe + a dict of fitted LabelEncoders
    (needed later to encode new/live input in app.py).
    """
    encoders = {}
    df_encoded = df.copy()

    categorical_cols = df_encoded.select_dtypes(include="object").columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders


def scale_numeric(df: pd.DataFrame, numeric_cols=NUMERIC_COLS):
    """
    Standard-scales numeric columns. Returns scaled df + fitted scaler.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled, scaler


def clean_pipeline(save: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    raw data -> fix types -> drop useless cols -> encode -> scale -> save
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_raw_data()

    df = fix_total_charges(df)
    df = drop_useless_columns(df)

    # Drop exact duplicate rows if any
    df = df.drop_duplicates()

    # Encode target separately (Yes/No -> 1/0) for readability
    if df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # Encode remaining categorical features
    df_encoded, encoders = encode_categorical(df)

    # Scale numeric columns (target excluded automatically since it's not in NUMERIC_COLS)
    df_final, scaler = scale_numeric(df_encoded)

    if save:
        df_final.to_csv(CLEANED_FILE_PATH, index=False)
        joblib.dump(encoders, ENCODERS_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✅ Cleaned data saved to: {CLEANED_FILE_PATH}")
        print(f"✅ Encoders saved to:     {ENCODERS_PATH}")
        print(f"✅ Scaler saved to:       {SCALER_PATH}")

    return df_final


if __name__ == "__main__":
    cleaned_df = clean_pipeline()
    print("\n----- CLEANED DATA SAMPLE -----")
    print(cleaned_df.head())
