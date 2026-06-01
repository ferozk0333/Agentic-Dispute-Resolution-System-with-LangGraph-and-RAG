"""
Tool: query_transactions
Fetches a full dispute record from demo_disputes by transaction_id.
"""
import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/disputes.db")


def query_transactions(transaction_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM demo_disputes WHERE transaction_id = ?",
        (transaction_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"error": f"Transaction '{transaction_id}' not found in demo_disputes"}
    return dict(row)
