"""
Tool called by Agent 2 to get DT fraud prediction + interpretable decision path.
Runs in < 1ms. No API call needed.
"""

import json
import os

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = os.getenv("DT_MODEL_PATH", "backend/classifier/fraud_dt.joblib")
META_PATH  = "backend/classifier/model_metadata.json"

_model: DecisionTreeClassifier = None
_meta:  dict = None


def _load():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(META_PATH) as fh:
            _meta = json.load(fh)


def extract_decision_path(model, feature_names: list, feature_values: dict) -> list[str]:
    """Walk the decision tree path for this sample and return human-readable rules."""
    from sklearn.tree import _tree

    tree_     = model.tree_
    feature   = tree_.feature
    threshold = tree_.threshold

    import pandas as pd
    x              = pd.DataFrame([{f: feature_values[f] for f in feature_names}])
    node_indicator = model.decision_path(x)
    node_ids       = node_indicator.indices

    path = []
    for node_id in node_ids:
        if tree_.feature[node_id] == _tree.TREE_UNDEFINED:
            leaf_class = np.argmax(tree_.value[node_id])
            path.append("FRAUD" if leaf_class == 1 else "LEGITIMATE")
        else:
            feat_name = feature_names[feature[node_id]]
            thresh    = round(threshold[node_id], 3)
            actual    = feature_values[feat_name]
            direction = "<=" if actual <= thresh else ">"
            path.append(f"{feat_name} {direction} {thresh} (actual: {actual})")
    return path


def run_fraud_classifier(
    transaction_amt: float,
    velocity_24h: int,
    velocity_amt_24h: float,
    geo_mismatch: int,
    merchant_risk_score: float,
    is_high_risk_category: int,
    email_domain_risk: int,
    unique_locations_24h: int,
    typical_amt_max: float = 500.0,
) -> dict:
    """
    Run the Decision Tree fraud classifier.
    All inputs are derived from existing Agent 2 tool call results — no new API call.
    """
    _load()

    amt_vs_typical = round(transaction_amt / max(typical_amt_max, 1.0), 4)

    feature_values = {
        "transaction_amt":       transaction_amt,
        "amt_vs_typical_ratio":  amt_vs_typical,
        "velocity_24h":          velocity_24h,
        "velocity_amt_24h":      velocity_amt_24h,
        "geo_mismatch":          geo_mismatch,
        "merchant_risk_score":   merchant_risk_score,
        "is_high_risk_category": is_high_risk_category,
        "email_domain_risk":     email_domain_risk,
        "unique_locations_24h":  unique_locations_24h,
    }

    features = _meta["features"]
    import pandas as pd
    X = pd.DataFrame([{f: feature_values[f] for f in features}])
    fraud_prob = float(_model.predict_proba(X)[0][1])
    prediction = "fraud" if fraud_prob >= _meta["fraud_threshold"] else "legitimate"
    confidence = "high" if fraud_prob > 0.75 or fraud_prob < 0.25 else "medium"

    importances  = _meta["feature_importances"]
    top_features = sorted(
        [
            {
                "feature":    feat,
                "value":      round(feature_values[feat], 4),
                "importance": importances.get(feat, 0.0),
            }
            for feat in features
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )[:4]

    decision_path = extract_decision_path(_model, features, feature_values)

    return {
        "fraud_probability": round(fraud_prob, 4),
        "prediction":        prediction,
        "confidence":        confidence,
        "top_features":      top_features,
        "decision_path":     decision_path,
        "model_version":     "dt_v1",
    }
