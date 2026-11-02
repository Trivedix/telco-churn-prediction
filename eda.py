"""
eda.py
------
Exploratory Data Analysis for the telecom churn dataset.
Generates and saves useful plots inside the `plots/` folder:
 - Churn distribution
 - Correlation heatmap
 - Tenure vs Churn
 - Monthly Charges vs Churn
 - Contract type vs Churn
"""

import os
import matplotlib
matplotlib.use("Agg")  # so it works without a display (server-safe)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from read_data import load_raw_data

PLOTS_DIR = os.path.join("static", "plots")  # saved here so Flask can serve them directly
sns.set_style("whitegrid")


def churn_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x="Churn", data=df, palette="Set2")
    ax.set_title("Customer Churn Distribution")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')
    plt.savefig(os.path.join(PLOTS_DIR, "churn_distribution.png"), bbox_inches="tight")
    plt.close()


def correlation_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    numeric_df = df.copy()
    numeric_df["Churn_num"] = numeric_df["Churn"].map({"Yes": 1, "No": 0})
    corr = numeric_df.select_dtypes(include=["int64", "float64"]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()


def tenure_vs_churn(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, palette="Set1")
    plt.title("Tenure Distribution by Churn")
    plt.savefig(os.path.join(PLOTS_DIR, "tenure_vs_churn.png"), bbox_inches="tight")
    plt.close()


def monthly_charges_vs_churn(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set3")
    plt.title("Monthly Charges vs Churn")
    plt.savefig(os.path.join(PLOTS_DIR, "monthlycharges_vs_churn.png"), bbox_inches="tight")
    plt.close()


def contract_vs_churn(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2")
    plt.title("Contract Type vs Churn")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(PLOTS_DIR, "contract_vs_churn.png"), bbox_inches="tight")
    plt.close()


def run_eda():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_raw_data()

    # TotalCharges cleanup just for plotting purposes
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    churn_distribution(df)
    correlation_heatmap(df)
    tenure_vs_churn(df)
    monthly_charges_vs_churn(df)
    contract_vs_churn(df)

    print(f"✅ All EDA plots saved inside '{PLOTS_DIR}/' folder.")


if __name__ == "__main__":
    run_eda()
