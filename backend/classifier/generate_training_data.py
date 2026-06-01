"""
Generate 5,000 synthetic training rows calibrated to IEEE-CIS distributions.
Run: python backend/classifier/generate_training_data.py
Output: backend/classifier/training_data.csv
"""

import os

import numpy as np
import pandas as pd

# ── Calibration constants from IEEE-CIS analysis ─────────────────────────────
# Derived by running calibrate.py against train_transaction.csv.
# The velocity and geo values below reflect the C1 address-count proxy and
# addr1/addr2 mismatch proxy — both are reasonable stand-ins for the actual
# velocity_24h and geo_mismatch features used at inference time.

CALIBRATION = {
    "fraud_rate": 0.035,

    "legit_amt_mu":    3.8,   # exp(3.8) ≈ $44
    "legit_amt_sigma": 1.4,

    "fraud_amt_mu":    4.6,   # exp(4.6) ≈ $99
    "fraud_amt_sigma": 1.8,

    "legit_velocity_lambda": 1.2,
    "fraud_velocity_lambda": 4.5,

    "legit_geo_mismatch_prob": 0.08,
    "fraud_geo_mismatch_prob": 0.41,

    "legit_merchant_risk_alpha": 1.2,
    "legit_merchant_risk_beta":  8.0,
    "fraud_merchant_risk_alpha": 2.5,
    "fraud_merchant_risk_beta":  3.0,

    "legit_high_risk_cat_prob": 0.06,
    "fraud_high_risk_cat_prob": 0.28,

    "legit_risky_email_prob": 0.02,
    "fraud_risky_email_prob": 0.19,

    "legit_unique_loc_probs": [0.88, 0.10, 0.02],   # 1, 2, 3 unique countries
    "fraud_unique_loc_probs": [0.52, 0.31, 0.17],
}


def generate_row(is_fraud: int, rng: np.random.Generator) -> dict:
    c = CALIBRATION
    f = bool(is_fraud)

    mu    = c["fraud_amt_mu"]    if f else c["legit_amt_mu"]
    sigma = c["fraud_amt_sigma"] if f else c["legit_amt_sigma"]
    amt   = float(np.clip(rng.lognormal(mu, sigma), 1.0, 25000.0))

    lam      = c["fraud_velocity_lambda"] if f else c["legit_velocity_lambda"]
    velocity = int(np.clip(rng.poisson(lam), 1, 20))

    # Velocity amount is correlated with velocity and amount distribution
    velocity_amt = float(np.clip(
        rng.lognormal(mu + 0.3 * velocity, sigma * 0.8), 1.0, 50000.0
    ))

    geo_prob     = c["fraud_geo_mismatch_prob"] if f else c["legit_geo_mismatch_prob"]
    geo_mismatch = int(rng.random() < geo_prob)

    alpha = c["fraud_merchant_risk_alpha"] if f else c["legit_merchant_risk_alpha"]
    beta  = c["fraud_merchant_risk_beta"]  if f else c["legit_merchant_risk_beta"]
    merchant_risk = float(np.clip(rng.beta(alpha, beta), 0.0, 1.0))

    # typical_amt_max is used to compute amt_vs_typical_ratio
    typical_amt_max = max(50.0, rng.lognormal(3.5, 0.8))
    amt_vs_typical  = float(amt / typical_amt_max)

    hr_prob          = c["fraud_high_risk_cat_prob"] if f else c["legit_high_risk_cat_prob"]
    is_high_risk_cat = int(rng.random() < hr_prob)

    em_prob           = c["fraud_risky_email_prob"] if f else c["legit_risky_email_prob"]
    email_domain_risk = int(rng.random() < em_prob)

    loc_probs   = c["fraud_unique_loc_probs"] if f else c["legit_unique_loc_probs"]
    unique_locs = int(rng.choice([1, 2, 3], p=loc_probs))

    return {
        "transaction_amt":       round(amt, 2),
        "amt_vs_typical_ratio":  round(amt_vs_typical, 4),
        "velocity_24h":          velocity,
        "velocity_amt_24h":      round(velocity_amt, 2),
        "geo_mismatch":          geo_mismatch,
        "merchant_risk_score":   round(merchant_risk, 4),
        "is_high_risk_category": is_high_risk_cat,
        "email_domain_risk":     email_domain_risk,
        "unique_locations_24h":  unique_locs,
        "is_fraud":              is_fraud,
    }


def generate_dataset(n: int = 5000, fraud_rate: float = 0.035, seed: int = 42) -> pd.DataFrame:
    rng     = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_legit = n - n_fraud

    rows = (
        [generate_row(1, rng) for _ in range(n_fraud)] +
        [generate_row(0, rng) for _ in range(n_legit)]
    )
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("backend/classifier", exist_ok=True)
    df = generate_dataset(n=5000, fraud_rate=0.20)
    out = "backend/classifier/training_data.csv"
    df.to_csv(out, index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"Generated {len(df):,} rows — {fraud_count} fraud ({fraud_count/len(df)*100:.1f}%), "
          f"{len(df)-fraud_count} legitimate")
    print(f"Saved to {out}")
    print(df.describe().to_string())
