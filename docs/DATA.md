# DATA.md — Data Layer Setup

## Overview

Three data sources power this system:

| Source | Format | Used by |
|---|---|---|
| Hand-crafted demo disputes (25 records) | SQLite `demo_disputes` | Live demo — Agent 2 tool calls |
| IEEE-CIS Fraud Detection dataset | SQLite `ieee_transactions` | Batch eval script — F1 scoring only |
| Visa Core Rules PDF | ChromaDB | RAG tool — compliance rule retrieval |

Set up all three before implementing any agents. Run scripts in the order listed below.

---

## Part 1 — Demo disputes dataset

### Why hand-crafted

IEEE-CIS is an ML benchmark dataset — it has no merchant names, no readable timestamps, no customer narratives, and opaque engineered features (V1–V339). Agent 2 can't produce meaningful, explainable dispute decisions from it. The hand-crafted dataset is designed to make every agent step readable and demonstrable.

### Schema

```sql
CREATE TABLE IF NOT EXISTS demo_disputes (
    transaction_id      TEXT PRIMARY KEY,
    transaction_date    TEXT,               -- ISO 8601: "2024-03-18"
    transaction_amt     REAL,
    merchant_name       TEXT,
    merchant_category   TEXT,               -- "electronics" | "travel" | "retail" etc
    card_id             TEXT,               -- masked: "card_****8821"
    card_country        TEXT,               -- country card was issued in
    txn_country         TEXT,               -- country transaction occurred in
    product_type        TEXT,
    p_email_domain      TEXT,
    is_fraud            INTEGER,            -- ground truth: 0 or 1
    dispute_outcome     TEXT,               -- expected: "approve"|"reject"|"escalate"|"policy_violation"
    dispute_scenario    TEXT                -- human label: "geo_mismatch"|"duplicate"|"high_value" etc
);

CREATE TABLE IF NOT EXISTS card_velocity (
    card_id             TEXT,
    transaction_date    TEXT,
    transaction_amt     REAL,
    txn_country         TEXT,
    merchant_name       TEXT,
    is_fraud            INTEGER
);

CREATE TABLE IF NOT EXISTS merchant_risk (
    merchant_name       TEXT PRIMARY KEY,
    risk_score          REAL,               -- 0.0 (safe) to 1.0 (high risk)
    risk_category       TEXT,               -- "low" | "medium" | "high"
    fraud_rate          REAL,
    total_txn_count     INTEGER,
    typical_amt_min     REAL,
    typical_amt_max     REAL
);

CREATE INDEX IF NOT EXISTS idx_card_velocity ON card_velocity(card_id, transaction_date);
```

### 25 hand-crafted dispute records

Cover all key scenarios the demo needs to show. Script: `backend/data/seed_demo_data.py`

```python
DEMO_DISPUTES = [
    # ── APPROVE scenarios (8 records) ──────────────────────────────────────
    {
        "transaction_id": "TXN-2024-0318-8821",
        "transaction_date": "2024-03-18",
        "transaction_amt": 1240.00,
        "merchant_name": "ElectroMart Inc.",
        "merchant_category": "electronics",
        "card_id": "card_****8821",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "laptop",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "clean_unauthorized",
    },
    {
        "transaction_id": "TXN-2024-0201-4432",
        "transaction_date": "2024-02-01",
        "transaction_amt": 89.99,
        "merchant_name": "StreamFlix Pro",
        "merchant_category": "digital_goods",
        "card_id": "card_****4432",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "subscription",
        "p_email_domain": "yahoo.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "unauthorized_subscription",
    },
    {
        "transaction_id": "TXN-2024-0310-6601",
        "transaction_date": "2024-03-10",
        "transaction_amt": 342.50,
        "merchant_name": "TechGadgets Online",
        "merchant_category": "electronics",
        "card_id": "card_****6601",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "headphones",
        "p_email_domain": "outlook.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "clean_unauthorized",
    },
    {
        "transaction_id": "TXN-2024-0225-3310",
        "transaction_date": "2024-02-25",
        "transaction_amt": 215.00,
        "merchant_name": "AirTravel Bookings",
        "merchant_category": "travel",
        "card_id": "card_****3310",
        "card_country": "US",
        "txn_country": "UK",
        "product_type": "flight",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "geo_mismatch",
    },
    {
        "transaction_id": "TXN-2024-0305-7743",
        "transaction_date": "2024-03-05",
        "transaction_amt": 499.00,
        "merchant_name": "LuxuryWatch Co.",
        "merchant_category": "jewelry",
        "card_id": "card_****7743",
        "card_country": "US",
        "txn_country": "FR",
        "product_type": "watch",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "geo_mismatch",
    },
    {
        "transaction_id": "TXN-2024-0101-1122",
        "transaction_date": "2024-01-01",
        "transaction_amt": 75.00,
        "merchant_name": "GroceryHub",
        "merchant_category": "grocery",
        "card_id": "card_****1122",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "groceries",
        "p_email_domain": "icloud.com",
        "is_fraud": 0,
        "dispute_outcome": "reject",
        "dispute_scenario": "legitimate_charge",
    },
    {
        "transaction_id": "TXN-2024-0312-9910",
        "transaction_date": "2024-03-12",
        "transaction_amt": 188.00,
        "merchant_name": "FashionForward",
        "merchant_category": "retail",
        "card_id": "card_****9910",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "clothing",
        "p_email_domain": "hotmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "clean_unauthorized",
    },
    {
        "transaction_id": "TXN-2024-0220-5588",
        "transaction_date": "2024-02-20",
        "transaction_amt": 920.00,
        "merchant_name": "HomeAppliances Direct",
        "merchant_category": "electronics",
        "card_id": "card_****5588",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "appliance",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "high_velocity",
    },

    # ── REJECT scenarios (5 records) ───────────────────────────────────────
    {
        "transaction_id": "TXN-2024-0101-1199",
        "transaction_date": "2024-01-01",
        "transaction_amt": 130.00,
        "merchant_name": "GroceryHub",
        "merchant_category": "grocery",
        "card_id": "card_****1199",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "groceries",
        "p_email_domain": "gmail.com",
        "is_fraud": 0,
        "dispute_outcome": "reject",
        "dispute_scenario": "legitimate_charge",
    },
    {
        "transaction_id": "TXN-2024-0308-2234",
        "transaction_date": "2024-03-08",
        "transaction_amt": 45.00,
        "merchant_name": "CafeRoast",
        "merchant_category": "food",
        "card_id": "card_****2234",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "food",
        "p_email_domain": "gmail.com",
        "is_fraud": 0,
        "dispute_outcome": "reject",
        "dispute_scenario": "legitimate_charge",
    },
    {
        "transaction_id": "TXN-2024-0318-8821-DUP",
        "transaction_date": "2024-03-19",
        "transaction_amt": 1240.00,
        "merchant_name": "ElectroMart Inc.",
        "merchant_category": "electronics",
        "card_id": "card_****8821",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "laptop",
        "p_email_domain": "gmail.com",
        "is_fraud": 0,
        "dispute_outcome": "reject",
        "dispute_scenario": "duplicate_dispute",
    },
    {
        "transaction_id": "TXN-2024-0215-6677",
        "transaction_date": "2024-02-15",
        "transaction_amt": 320.00,
        "merchant_name": "AutoRepairPro",
        "merchant_category": "automotive",
        "card_id": "card_****6677",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "service",
        "p_email_domain": "outlook.com",
        "is_fraud": 0,
        "dispute_outcome": "reject",
        "dispute_scenario": "legitimate_charge",
    },
    {
        "transaction_id": "TXN-2023-0901-4421",
        "transaction_date": "2023-09-01",
        "transaction_amt": 650.00,
        "merchant_name": "FurnitureWorld",
        "merchant_category": "home",
        "card_id": "card_****4421",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "furniture",
        "p_email_domain": "yahoo.com",
        "is_fraud": 0,
        "dispute_outcome": "reject",
        "dispute_scenario": "outside_time_window",    # > 120 days ago
    },

    # ── POLICY VIOLATION scenarios (4 records) ─────────────────────────────
    {
        "transaction_id": "TXN-2024-0316-9934",
        "transaction_date": "2024-03-16",
        "transaction_amt": 7800.00,
        "merchant_name": "LuxuryGoods Direct",
        "merchant_category": "luxury",
        "card_id": "card_****9934",
        "card_country": "US",
        "txn_country": "AE",
        "product_type": "jewelry",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "policy_violation",
        "dispute_scenario": "exceeds_refund_threshold",  # > $5,000
    },
    {
        "transaction_id": "TXN-2024-0310-1155",
        "transaction_date": "2024-03-10",
        "transaction_amt": 12500.00,
        "merchant_name": "PremiumAutos",
        "merchant_category": "automotive",
        "card_id": "card_****1155",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "vehicle_deposit",
        "p_email_domain": "corporate.com",
        "is_fraud": 1,
        "dispute_outcome": "policy_violation",
        "dispute_scenario": "exceeds_refund_threshold",
    },
    {
        "transaction_id": "TXN-2024-0314-7766",
        "transaction_date": "2024-03-14",
        "transaction_amt": 980.00,
        "merchant_name": "OnlineCasino888",
        "merchant_category": "gambling",
        "card_id": "card_****7766",
        "card_country": "US",
        "txn_country": "MT",
        "product_type": "gambling",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "policy_violation",
        "dispute_scenario": "prohibited_merchant_category",
    },
    {
        "transaction_id": "TXN-2024-0315-3388",
        "transaction_date": "2024-03-15",
        "transaction_amt": 2200.00,
        "merchant_name": "CryptoExchange Pro",
        "merchant_category": "crypto",
        "card_id": "card_****3388",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "crypto_purchase",
        "p_email_domain": "protonmail.com",
        "is_fraud": 1,
        "dispute_outcome": "policy_violation",
        "dispute_scenario": "reason_code_mismatch",
    },

    # ── ESCALATE scenarios (4 records) ─────────────────────────────────────
    {
        "transaction_id": "TXN-2024-0317-4499",
        "transaction_date": "2024-03-17",
        "transaction_amt": 3400.00,
        "merchant_name": "MedicalSupplies Inc.",
        "merchant_category": "medical",
        "card_id": "card_****4499",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "medical_equipment",
        "p_email_domain": "hospital.org",
        "is_fraud": 0,
        "dispute_outcome": "escalate",
        "dispute_scenario": "ambiguous_signals",       # low confidence, needs human
    },
    {
        "transaction_id": "TXN-2024-0313-6622",
        "transaction_date": "2024-03-13",
        "transaction_amt": 4800.00,
        "merchant_name": "InternationalShipping Co.",
        "merchant_category": "logistics",
        "card_id": "card_****6622",
        "card_country": "US",
        "txn_country": "CN",
        "product_type": "shipping",
        "p_email_domain": "gmail.com",
        "is_fraud": 0,
        "dispute_outcome": "escalate",
        "dispute_scenario": "ambiguous_geo",
    },
    {
        "transaction_id": "TXN-2024-0301-8833",
        "transaction_date": "2024-03-01",
        "transaction_amt": 1650.00,
        "merchant_name": "ArtGallery Boutique",
        "merchant_category": "art",
        "card_id": "card_****8833",
        "card_country": "US",
        "txn_country": "IT",
        "product_type": "artwork",
        "p_email_domain": "icloud.com",
        "is_fraud": 0,
        "dispute_outcome": "escalate",
        "dispute_scenario": "ambiguous_signals",
    },
    {
        "transaction_id": "TXN-2024-0311-2277",
        "transaction_date": "2024-03-11",
        "transaction_amt": 540.00,
        "merchant_name": "UnknownMerchant XZ",
        "merchant_category": "unknown",
        "card_id": "card_****2277",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "unknown",
        "p_email_domain": "tempmail.com",
        "is_fraud": 1,
        "dispute_outcome": "escalate",
        "dispute_scenario": "unknown_merchant",
    },

    # ── HIGH VELOCITY scenarios (4 records) ────────────────────────────────
    {
        "transaction_id": "TXN-2024-0318-5511",
        "transaction_date": "2024-03-18",
        "transaction_amt": 399.00,
        "merchant_name": "GameStore Digital",
        "merchant_category": "gaming",
        "card_id": "card_****5511",
        "card_country": "US",
        "txn_country": "US",
        "product_type": "game_keys",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "high_velocity",
    },
    {
        "transaction_id": "TXN-2024-0318-5512",
        "transaction_date": "2024-03-18",
        "transaction_amt": 399.00,
        "merchant_name": "GameStore Digital",
        "merchant_category": "gaming",
        "card_id": "card_****5511",       # same card as above — velocity signal
        "card_country": "US",
        "txn_country": "US",
        "product_type": "game_keys",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "high_velocity",
    },
    {
        "transaction_id": "TXN-2024-0318-5513",
        "transaction_date": "2024-03-18",
        "transaction_amt": 399.00,
        "merchant_name": "GameStore Digital",
        "merchant_category": "gaming",
        "card_id": "card_****5511",       # same card — third hit in 2 hours
        "card_country": "US",
        "txn_country": "US",
        "product_type": "game_keys",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "high_velocity",
    },
    {
        "transaction_id": "TXN-2024-0318-9901",
        "transaction_date": "2024-03-18",
        "transaction_amt": 2750.00,
        "merchant_name": "DiamondJewelers",
        "merchant_category": "jewelry",
        "card_id": "card_****9901",
        "card_country": "US",
        "txn_country": "UK",
        "product_type": "jewelry",
        "p_email_domain": "gmail.com",
        "is_fraud": 1,
        "dispute_outcome": "approve",
        "dispute_scenario": "geo_mismatch_high_amt",
    },
]
```

### Merchant risk seed data

```python
MERCHANT_RISK_DATA = [
    {"merchant_name": "ElectroMart Inc.",        "risk_score": 0.21, "risk_category": "low",    "fraud_rate": 0.021, "total_txn_count": 4820, "typical_amt_min": 50,   "typical_amt_max": 2000},
    {"merchant_name": "StreamFlix Pro",           "risk_score": 0.08, "risk_category": "low",    "fraud_rate": 0.008, "total_txn_count": 28000, "typical_amt_min": 9,   "typical_amt_max": 120},
    {"merchant_name": "TechGadgets Online",       "risk_score": 0.34, "risk_category": "medium", "fraud_rate": 0.034, "total_txn_count": 2100, "typical_amt_min": 20,   "typical_amt_max": 800},
    {"merchant_name": "AirTravel Bookings",       "risk_score": 0.15, "risk_category": "low",    "fraud_rate": 0.015, "total_txn_count": 9400, "typical_amt_min": 80,   "typical_amt_max": 3000},
    {"merchant_name": "LuxuryWatch Co.",          "risk_score": 0.28, "risk_category": "medium", "fraud_rate": 0.028, "total_txn_count": 880,  "typical_amt_min": 200,  "typical_amt_max": 5000},
    {"merchant_name": "GroceryHub",               "risk_score": 0.03, "risk_category": "low",    "fraud_rate": 0.003, "total_txn_count": 91000, "typical_amt_min": 10,  "typical_amt_max": 300},
    {"merchant_name": "FashionForward",           "risk_score": 0.12, "risk_category": "low",    "fraud_rate": 0.012, "total_txn_count": 7200, "typical_amt_min": 30,   "typical_amt_max": 500},
    {"merchant_name": "HomeAppliances Direct",    "risk_score": 0.19, "risk_category": "low",    "fraud_rate": 0.019, "total_txn_count": 3100, "typical_amt_min": 100,  "typical_amt_max": 2500},
    {"merchant_name": "LuxuryGoods Direct",       "risk_score": 0.61, "risk_category": "high",   "fraud_rate": 0.061, "total_txn_count": 420,  "typical_amt_min": 500,  "typical_amt_max": 15000},
    {"merchant_name": "OnlineCasino888",          "risk_score": 0.88, "risk_category": "high",   "fraud_rate": 0.088, "total_txn_count": 6600, "typical_amt_min": 50,   "typical_amt_max": 5000},
    {"merchant_name": "CryptoExchange Pro",       "risk_score": 0.72, "risk_category": "high",   "fraud_rate": 0.072, "total_txn_count": 3300, "typical_amt_min": 200,  "typical_amt_max": 10000},
    {"merchant_name": "MedicalSupplies Inc.",     "risk_score": 0.09, "risk_category": "low",    "fraud_rate": 0.009, "total_txn_count": 1200, "typical_amt_min": 50,   "typical_amt_max": 8000},
    {"merchant_name": "GameStore Digital",        "risk_score": 0.45, "risk_category": "medium", "fraud_rate": 0.045, "total_txn_count": 18000, "typical_amt_min": 10,  "typical_amt_max": 600},
    {"merchant_name": "DiamondJewelers",          "risk_score": 0.38, "risk_category": "medium", "fraud_rate": 0.038, "total_txn_count": 640,  "typical_amt_min": 300,  "typical_amt_max": 8000},
    {"merchant_name": "UnknownMerchant XZ",       "risk_score": 0.95, "risk_category": "high",   "fraud_rate": 0.095, "total_txn_count": 12,   "typical_amt_min": 0,    "typical_amt_max": 9999},
    {"merchant_name": "PremiumAutos",             "risk_score": 0.22, "risk_category": "low",    "fraud_rate": 0.022, "total_txn_count": 310,  "typical_amt_min": 500,  "typical_amt_max": 50000},
    {"merchant_name": "CafeRoast",                "risk_score": 0.02, "risk_category": "low",    "fraud_rate": 0.002, "total_txn_count": 44000, "typical_amt_min": 3,   "typical_amt_max": 80},
    {"merchant_name": "AutoRepairPro",            "risk_score": 0.06, "risk_category": "low",    "fraud_rate": 0.006, "total_txn_count": 2200, "typical_amt_min": 50,   "typical_amt_max": 2000},
    {"merchant_name": "FurnitureWorld",           "risk_score": 0.07, "risk_category": "low",    "fraud_rate": 0.007, "total_txn_count": 1800, "typical_amt_min": 100,  "typical_amt_max": 5000},
    {"merchant_name": "ArtGallery Boutique",      "risk_score": 0.31, "risk_category": "medium", "fraud_rate": 0.031, "total_txn_count": 290,  "typical_amt_min": 200,  "typical_amt_max": 20000},
    {"merchant_name": "InternationalShipping Co.","risk_score": 0.42, "risk_category": "medium", "fraud_rate": 0.042, "total_txn_count": 5500, "typical_amt_min": 20,   "typical_amt_max": 10000},
]
```

### Seed script (`backend/data/seed_demo_data.py`)

```python
import sqlite3, os, json
from datetime import datetime, timedelta

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/disputes.db")

def seed():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS demo_disputes ( ... );  -- see schema above
        CREATE TABLE IF NOT EXISTS card_velocity ( ... );
        CREATE TABLE IF NOT EXISTS merchant_risk ( ... );
        CREATE INDEX IF NOT EXISTS idx_card_velocity ON card_velocity(card_id, transaction_date);
    """)

    # Insert demo disputes
    cur.executemany(
        "INSERT OR REPLACE INTO demo_disputes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [tuple(d.values()) for d in DEMO_DISPUTES]
    )

    # Insert merchant risk
    cur.executemany(
        "INSERT OR REPLACE INTO merchant_risk VALUES (?,?,?,?,?,?,?)",
        [tuple(m.values()) for m in MERCHANT_RISK_DATA]
    )

    # Populate card_velocity from demo_disputes (so velocity tool has history to query)
    cur.execute("""
        INSERT OR IGNORE INTO card_velocity
        SELECT card_id, transaction_date, transaction_amt, txn_country, merchant_name, is_fraud
        FROM demo_disputes
    """)

    conn.commit()
    conn.close()
    print(f"Seeded {len(DEMO_DISPUTES)} disputes and {len(MERCHANT_RISK_DATA)} merchants into {DB_PATH}")

if __name__ == "__main__":
    seed()
```

---

## Part 2 — IEEE-CIS dataset (batch eval only)

### Download

1. In the project directory, I have placed two CSVs `train_transaction.csv` and `train_identity.csv`
2. Place in `backend/data/`

### What it's used for

Only `run_eval.py` touches this data. It runs 200 sampled rows through a simplified pipeline (no live tool calls — uses pre-computed signals) and computes F1, precision, recall against the `isFraud` ground truth label.

### IEEE-CIS columns used for eval

| Column | Role in eval |
|---|---|
| `TransactionID` | Record identifier |
| `TransactionAmt` | Maps to dispute amount |
| `isFraud` | Ground truth label |
| `card1` | Card grouping for velocity proxy |
| `C1`–`C14` | Counting features → fraud signal inputs |
| `D1`–`D15` | Time deltas → velocity proxy |
| `V1`–`V50` | Top engineered features → model input |

### Eval ingest script (`backend/data/ingest_ieee.py`)

```python
import pandas as pd, sqlite3, os

CSV_PATH = os.getenv("IEEE_CSV_PATH", "./data/train_transaction.csv")
DB_PATH  = os.getenv("SQLITE_DB_PATH", "./data/disputes.db")

EVAL_COLS = [
    "TransactionID", "TransactionAmt", "isFraud",
    "card1", "ProductCD", "addr1",
    "C1","C2","C6","C11","C13","C14",
    "D1","D4","D10","D15",
    "V12","V13","V14","V17","V20","V29","V30",
    "V36","V37","V38","V44","V45","V46","V47",
]

def ingest():
    df = pd.read_csv(CSV_PATH, usecols=[c for c in EVAL_COLS if c in pd.read_csv(CSV_PATH, nrows=0).columns])
    df["TransactionID"] = df["TransactionID"].astype(str)

    # 80/20 split — train portion only stored (test reserved for eval)
    split = int(len(df) * 0.8)
    df_eval = df.iloc[split:].copy()
    df_eval.to_csv("./data/ieee_eval_set.csv", index=False)

    conn = sqlite3.connect(DB_PATH)
    df.iloc[:split].to_sql("ieee_transactions", conn, if_exists="replace", index=False, chunksize=10000)
    conn.close()
    print(f"IEEE-CIS: {split:,} train rows loaded, {len(df_eval):,} eval rows saved to ieee_eval_set.csv")

if __name__ == "__main__":
    ingest()
```

---

## Part 3 — Visa Rules RAG Pipeline

### Source document

In the project directory, I have put Visa Core Rules public PDF:
- Save as: `backend/data/visa_core_rules.pdf`

Relevant sections to prioritize during chunking:
- Section 11: Dispute Resolution
- Section 11.2: Chargeback time limits
- Appendix D: Reason codes (10.1–10.4, 12.x, 13.x)
- Any other important sections as per our need (read PDF index to understand)

### Chunking strategy

Semantic chunking — one complete rule per chunk. Target ~400 tokens, 50-token overlap.


### Ingest script (`backend/data/ingest_visa_rules.py`)

```python
import fitz  # PyMuPDF
import chromadb
from openai import OpenAI
import os, re

PDF_PATH   = os.getenv("VISA_RULES_PDF_PATH", "./data/visa_core_rules.pdf")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

client = OpenAI()
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection("visa_rules")

def extract_chunks(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        sections = re.split(r'(?=\d{1,2}\.\d{1,2}|\bAppendix\b)', text)
        for section in sections:
            if len(section.strip()) < 100:
                continue
            chunks.append({"text": section.strip(), "page": page_num + 1})
    return chunks

def embed_and_store(chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(input=texts[i:i+100], model="text-embedding-3-small")
        all_embeddings.extend([e.embedding for e in resp.data])
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=all_embeddings,
        documents=texts,
        metadatas=[{"page": c["page"], "source": "visa_core_rules"} for c in chunks]
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")

if __name__ == "__main__":
    chunks = extract_chunks(PDF_PATH)
    embed_and_store(chunks)
```

---

## Setup order

Run these in sequence before starting the backend:

```bash
python backend/data/seed_demo_data.py      # creates disputes.db with all demo records
python backend/data/ingest_ieee.py         # adds ieee_transactions table + eval CSV
python backend/data/ingest_visa_rules.py   # populates ChromaDB
```

## Verification

```python
import sqlite3, chromadb

# Check demo data
conn = sqlite3.connect("./data/disputes.db")
n = conn.execute("SELECT COUNT(*) FROM demo_disputes").fetchone()[0]
assert n == 25, f"Expected 25 disputes, got {n}"
scenarios = conn.execute("SELECT DISTINCT dispute_outcome FROM demo_disputes").fetchall()
print("Outcomes:", [s[0] for s in scenarios])  # approve, reject, escalate, policy_violation

# Check ChromaDB
chroma = chromadb.PersistentClient(path="./data/chroma")
col = chroma.get_collection("visa_rules")
print(f"ChromaDB chunks: {col.count()}")
assert col.count() > 0

print("All data sources OK")
```