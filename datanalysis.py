"""
CreditScore - Part 1: Data Extraction & Basic Analytics
Use the German Credit Dataset (UCI, 1000 records, 20 features).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits",
    "job", "liable_people", "telephone", "foreign_worker", "credit_risk",
]
#Separated using Claude <-> Criterion: not a raw number, e.g. "checking account"
CATEGORICAL = [               
    "checking_account", "credit_history", "purpose", "savings",
    "employment", "personal_status", "other_debtors", "property",
    "other_installments", "housing", "job", "telephone", "foreign_worker",
]
#Separated using Claude <-> Criterion: raw number, e.g. "age"
NUMERIC = [
    "duration", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "liable_people",
]

def load_data():
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/statlog/german/german.data"
    )
    df = pd.read_csv(url, sep=" ", header=None, names=COLUMNS)
    #Maybe it's better to implement try/except here

    return df

def preprocess(df):
    # get target
    target = []
    for val in df["credit_risk"]:
        if val == 2:
            target.append(1)
        else:
            target.append(0)
    y = np.array(target)

    df = df.drop("credit_risk", axis=1)

    # encode categoricals
    for col in CATEGORICAL:
        unique_vals = list(df[col].unique())
        df[col] = df[col].apply(lambda x: unique_vals.index(x))

    # scale manually
    X = df[NUMERIC + CATEGORICAL].values.astype(float)
    for i in range(X.shape[1]):
        col_mean = X[:, i].mean()
        col_std  = X[:, i].std()
        if col_std != 0:
            X[:, i] = (X[:, i] - col_mean) / col_std

    return X, y

#Histograms
def plot_eda(df: pd.DataFrame, save_path="credit_eda.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Part 1 — Credit Data EDA", fontsize=14, fontweight="bold")

    colors = {1: "#1D9E75", 2: "#D85A30"}
    labels = {1: "Good", 2: "Bad"}

    # Class distribution
    ax = axes[0, 0]
    counts = df["credit_risk"].value_counts()
    ax.bar([labels[k] for k in counts.index], counts.values,
           color=[colors[k] for k in counts.index], width=0.4)
    ax.set_title("Class Distribution")
    ax.spines[["top","right"]].set_visible(False)

    # Credit amount by class
    ax = axes[0, 1]
    for cls in [1, 2]:
        ax.hist(df[df.credit_risk == cls]["credit_amount"], bins=25,
                alpha=0.6, color=colors[cls], label=labels[cls])
    ax.set_title("Credit Amount by Risk")
    ax.set_xlabel("Amount (DM)")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # Duration by class
    ax = axes[1, 0]
    for cls in [1, 2]:
        ax.hist(df[df.credit_risk == cls]["duration"], bins=20,
                alpha=0.6, color=colors[cls], label=labels[cls])
    ax.set_title("Loan Duration by Risk")
    ax.set_xlabel("Months")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # Age by class
    ax = axes[1, 1]
    for cls in [1, 2]:
        ax.hist(df[df.credit_risk == cls]["age"], bins=20,
                alpha=0.6, color=colors[cls], label=labels[cls])
    ax.set_title("Age by Risk")
    ax.set_xlabel("Age")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"EDA plot saved → {save_path}")


if __name__ == "__main__":
    df = load_data()
    print(df[["age", "credit_amount", "duration", "credit_risk"]].describe().round(1))
    X, y = preprocess(df)
    print(f"\nX: {X.shape} | Bad-credit rate: {y.mean()*100:.1f}%")
    plot_eda(df)
    
