"""
Tool: check_merchant_risk
Returns pre-computed risk profile for a merchant from merchant_risk table.
Falls back to a neutral 0.5 score for unknown merchants.
"""
import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/disputes.db")


def check_merchant_risk(merchant_name: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT merchant_name, risk_score, risk_category,
               fraud_rate, total_txn_count, typical_amt_min, typical_amt_max
        FROM merchant_risk
        WHERE merchant_name LIKE ?
        """,
        (f"%{merchant_name}%",),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return {
            "merchant_name":   merchant_name,
            "risk_score":      0.5,
            "risk_category":   "unknown",
            "note":            "Merchant not in database — treat as medium risk",
        }

    return {
        "merchant_name":   row[0],
        "risk_score":      row[1],
        "risk_category":   row[2],
        "fraud_rate":      row[3],
        "total_txn_count": row[4],
        "typical_amt_min": row[5],
        "typical_amt_max": row[6],
    }
