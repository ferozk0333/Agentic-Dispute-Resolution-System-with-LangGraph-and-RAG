"""
ML Classifier evaluation — runs the trained Decision Tree directly on the
IEEE-CIS held-out test set (data/ieee_eval_set.csv).

No pipeline calls; this is pure model inference.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_PATH = "backend/classifier/fraud_dt.joblib"
META_PATH  = "backend/classifier/model_metadata.json"
EVAL_PATH  = os.getenv("EVAL_CSV_PATH", "data/ieee_eval_set.csv")

FEATURES = [
    "transaction_amt",
    "amt_vs_typical_ratio",
    "velocity_24h",
    "velocity_amt_24h",
    "geo_mismatch",
    "merchant_risk_score",
    "is_high_risk_category",
    "email_domain_risk",
    "unique_locations_24h",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_amt"]       = df["TransactionAmt"]
    df["amt_vs_typical_ratio"]  = (df["TransactionAmt"] / 500).clip(0, 10)
    df["velocity_24h"]          = df["C1"].fillna(1).clip(1, 20).astype(int)
    df["velocity_amt_24h"]      = df["C2"].fillna(df["TransactionAmt"])
    df["merchant_risk_score"]   = df["V12"].fillna(0.5).clip(0, 1)
    df["is_high_risk_category"] = (df["ProductCD"].isin(["S", "H"])).astype(int)
    df["unique_locations_24h"]  = df["D4"].fillna(1).clip(1, 5).astype(int)

    if "addr2" in df.columns:
        df["geo_mismatch"] = (df["addr1"] != df["addr2"]).astype(int)
    else:
        df["geo_mismatch"] = 0

    if "P_emaildomain" in df.columns:
        risky = ["protonmail", "temp", "mail.com", "guerrilla"]
        df["email_domain_risk"] = df["P_emaildomain"].apply(
            lambda x: 1 if pd.notna(x) and any(r in str(x) for r in risky) else 0
        )
    else:
        df["email_domain_risk"] = 0

    return df


def eval_classifier() -> dict:
    if not os.path.exists(EVAL_PATH):
        raise FileNotFoundError(
            f"Eval CSV not found at {EVAL_PATH!r}. "
            "Run: python3 backend/data/ingest_ieee.py first."
        )

    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(EVAL_PATH)
    df    = _build_features(df)
    df    = df.dropna(subset=FEATURES + ["isFraud"])

    X = df[FEATURES]
    y = df["isFraud"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    f1        = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall    = recall_score(y, y_pred)
    auc       = roc_auc_score(y, y_prob)

    print("\n── CLASSIFIER EVALUATION ──────────────────────────")
    print(classification_report(y, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"ROC-AUC: {auc:.4f}")

    return {
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "roc_auc":   round(auc, 4),
        "n_eval":    int(len(df)),
    }
