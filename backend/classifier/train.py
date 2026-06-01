"""
Train, evaluate, and serialize the fraud Decision Tree classifier.
Run: python backend/classifier/train.py
Output: backend/classifier/fraud_dt.joblib  (model)
        backend/classifier/model_metadata.json  (feature names, threshold, eval metrics)
"""

import json
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH  = "backend/classifier/training_data.csv"
MODEL_PATH = "backend/classifier/fraud_dt.joblib"
META_PATH  = "backend/classifier/model_metadata.json"

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


def train():
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    weight_dict   = {0: class_weights[0], 1: class_weights[1]}

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        min_samples_split=20,
        class_weight=weight_dict,
        random_state=42,
        criterion="gini",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_prob)
    cv_f1     = cross_val_score(model, X, y, cv=5, scoring="f1").mean()

    print("\n── EVALUATION RESULTS ──────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"ROC-AUC:        {auc:.4f}")
    print(f"CV F1 (5-fold): {cv_f1:.4f}")

    importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\n── FEATURE IMPORTANCES ─────────────────────────────")
    for feat, imp in importances:
        print(f"  {feat:<28} {imp:.4f}")

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "features":     FEATURES,
        "max_depth":    6,
        "eval": {
            "f1":        round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "roc_auc":   round(auc, 4),
            "cv_f1":     round(cv_f1, 4),
        },
        "feature_importances": {f: round(i, 4) for f, i in importances},
        "fraud_threshold": 0.5,
        "training_rows":   len(df),
        "fraud_rows":      int(y.sum()),
    }
    with open(META_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nModel saved     → {MODEL_PATH}")
    print(f"Metadata saved  → {META_PATH}")
    return model, metadata


if __name__ == "__main__":
    train()
