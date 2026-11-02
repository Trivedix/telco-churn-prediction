"""
read_data.py
------------
Responsible for loading the raw Telecom Customer Churn dataset.

Dataset expected (standard Telco Customer Churn columns):
customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges, Churn

Download link (if you don't already have the CSV):
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV inside the `data/` folder as `telecom_churn.csv`
"""

import os
import pandas as pd

# ---------------- CONFIG ----------------
DATA_DIR = "data"
RAW_FILE_NAME = "telecom_churn.csv"
RAW_FILE_PATH = os.path.join(DATA_DIR, RAW_FILE_NAME)


def load_raw_data(path: str = RAW_FILE_PATH) -> pd.DataFrame:
    """
    Loads the raw CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Dataset not found at '{path}'.\n"
            f"Please download the Telco Customer Churn dataset and place it "
            f"inside the '{DATA_DIR}/' folder as '{RAW_FILE_NAME}'."
        )

    df = pd.read_csv(path)
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Prints a quick summary of the dataset — useful for sanity checks."""
    print("\n----- HEAD -----")
    print(df.head())

    print("\n----- INFO -----")
    print(df.info())

    print("\n----- NULL VALUES -----")
    print(df.isnull().sum())

    print("\n----- SHAPE -----")
    print(df.shape)

    if "Churn" in df.columns:
        print("\n----- CHURN DISTRIBUTION -----")
        print(df["Churn"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    data = load_raw_data()
    basic_info(data)
