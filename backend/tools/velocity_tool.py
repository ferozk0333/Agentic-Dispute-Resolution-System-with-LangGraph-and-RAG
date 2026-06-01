"""
Tool: get_velocity_history
Returns recent card activity from card_velocity.

Because demo data uses ISO date strings (not Unix timestamps), 'hours' is
converted to days (ceiling) and applied as a date range lookback from the
card's most recent recorded transaction date.
"""
import math
import os
import sqlite3
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/disputes.db")


def get_velocity_history(card_id: str, hours: int = 24) -> dict:
    days = max(1, math.ceil(hours / 24))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Most recent transaction date for this card as the reference point
    ref_row = cur.execute(
        "SELECT MAX(transaction_date) FROM card_velocity WHERE card_id = ?",
        (card_id,),
    ).fetchone()

    if not ref_row or ref_row[0] is None:
        conn.close()
        return {"card_id": card_id, "error": "No velocity history found for this card"}

    ref_date = date.fromisoformat(ref_row[0])
    since_date = (ref_date - timedelta(days=days)).isoformat()

    rows = cur.execute(
        """
        SELECT transaction_date, transaction_amt, txn_country, merchant_name, is_fraud
        FROM card_velocity
        WHERE card_id = ? AND transaction_date >= ?
        ORDER BY transaction_date DESC
        """,
        (card_id, since_date),
    ).fetchall()
    conn.close()

    transactions = [
        {
            "date":          r[0],
            "amount":        r[1],
            "country":       r[2],
            "merchant":      r[3],
            "flagged_fraud": bool(r[4]),
        }
        for r in rows
    ]

    unique_countries = list({t["country"] for t in transactions})
    total_amount     = round(sum(t["amount"] for t in transactions), 2)

    return {
        "card_id":          card_id,
        "lookback_days":    days,
        "transaction_count": len(transactions),
        "total_amount":     total_amount,
        "unique_countries": unique_countries,
        "transactions":     transactions,
    }
