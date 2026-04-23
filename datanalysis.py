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

def load_data() -> pd.DataFrame:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/statlog/german/german.data"
    )
    df = pd.read_csv(url, sep=" ", header=None, names=COLUMNS)
    #Maybe it's better to implement try/except here

    return df

