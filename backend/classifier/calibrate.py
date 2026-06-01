"""
Extract IEEE-CIS statistical distributions for synthetic data calibration.
Run ONCE: python backend/classifier/calibrate.py
Paste output into CALIBRATION dict in generate_training_data.py
"""

import json
import os

import numpy as np
import pandas as pd

CSV_PATH = os.getenv("IEEE_CSV_PATH", "./data/train_transaction.csv")


def calibrate():
    print("Loading IEEE-CIS data...")
    df = pd.read_csv(CSV_PATH, usecols=[
        "TransactionAmt", "isFraud", "card1", "addr1", "addr2",
        "C1", "C2", "D1", "D4", "P_emaildomain"
    ])

    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0]

    log_legit_amt = np.log(legit["TransactionAmt"].clip(1))
    log_fraud_amt = np.log(fraud["TransactionAmt"].clip(1))

    legit_vel = legit["C1"].dropna()
    fraud_vel = fraud["C1"].dropna()

    legit_geo = (legit["addr1"] != legit["addr2"]).mean()
    fraud_geo = (fraud["addr1"] != fraud["addr2"]).mean()

    risky_domains = ["protonmail", "mail.com", "temp", "guerrilla", "throwam"]
    def is_risky(domain):
        if pd.isna(domain):
            return False
        return any(r in str(domain).lower() for r in risky_domains)

    legit_risky_email = legit["P_emaildomain"].apply(is_risky).mean()
    fraud_risky_email = fraud["P_emaildomain"].apply(is_risky).mean()

    result = {
        "fraud_rate":              round(df["isFraud"].mean(), 4),
        "legit_amt_mu":            round(log_legit_amt.mean(), 3),
        "legit_amt_sigma":         round(log_legit_amt.std(), 3),
        "fraud_amt_mu":            round(log_fraud_amt.mean(), 3),
        "fraud_amt_sigma":         round(log_fraud_amt.std(), 3),
        "legit_velocity_lambda":   round(legit_vel.mean(), 3),
        "fraud_velocity_lambda":   round(fraud_vel.mean(), 3),
        "legit_geo_mismatch_prob": round(float(legit_geo), 3),
        "fraud_geo_mismatch_prob": round(float(fraud_geo), 3),
        "legit_risky_email_prob":  round(float(legit_risky_email), 4),
        "fraud_risky_email_prob":  round(float(fraud_risky_email), 4),
    }

    print("\n── CALIBRATION VALUES ──────────────────────────────")
    print(json.dumps(result, indent=2))
    print("\nPaste these into CALIBRATION dict in generate_training_data.py")
    return result


if __name__ == "__main__":
    calibrate()
