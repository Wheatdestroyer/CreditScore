"""
CreditScore - Part 1: Data Extraction & Basic Analytics
Use the German Credit Dataset (UCI, 1000 records, 20 features).
"""

import numpy as np
import pandas as pd

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits",
    "job", "liable_people", "telephone", "foreign_worker", "credit_risk",
]
#Separated using Claude <-> Criterion: not a raw number, e.g. "checking account"
Categorical = [               
    "checking_account", "credit_history", "purpose", "savings",
    "employment", "personal_status", "other_debtors", "property",
    "other_installments", "housing", "job", "telephone", "foreign_worker",
]
#Separated using Claude <-> Criterion: raw number, e.g. "age"
Numerical = [
    "duration", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "liable_people",
]

def load_data() -> pd.DataFrame:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/statlog/german/german.data"
    )
    df = pd.read_csv(url, sep=" ", header=None, names=COLUMNS)
    #Maybe it's better to implement try/except here

    return df
def preprocess() ->
