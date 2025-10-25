import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.getcwd(), "predictions.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        filename TEXT,
        predicted_class TEXT,
        confidence REAL,
        raw_probs TEXT
    )
    """)
    conn.commit()
    conn.close()


def log_prediction(filename: str, predicted_class: str, confidence: float, raw_probs: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (timestamp, filename, predicted_class, confidence, raw_probs) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), filename, predicted_class, confidence, json.dumps(raw_probs)),
    )
    conn.commit()
    conn.close()
