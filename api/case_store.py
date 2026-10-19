"""
FraudWatch Case Management & Analyst Feedback Persistence Store.
===============================================================

SQLite-backed persistent repository for transaction alerts, analyst case status,
dispositions, investigation notes, and label feedback history.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CASES_DB_PATH = os.path.join(DATA_DIR, "cases.db")
ALERTS_JSONL_PATH = os.path.join(PROJECT_ROOT, "reports", "kafka_fraud_alerts.jsonl")

CASE_STATUSES = ("OPEN", "INVESTIGATING", "CONFIRMED_FRAUD", "FALSE_POSITIVE")


def get_connection(db_path: str = CASES_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_case_db(db_path: str = CASES_DB_PATH) -> None:
    """Initialize case management and audit history tables."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            transaction_id TEXT PRIMARY KEY,
            timestamp REAL,
            step INTEGER,
            type TEXT,
            amount REAL,
            nameOrig TEXT,
            nameDest TEXT,
            orig_drain_ratio REAL,
            fraud_probability REAL,
            risk_level TEXT,
            decision_threshold REAL,
            case_status TEXT DEFAULT 'OPEN',
            analyst_disposition TEXT,
            analyst_notes TEXT,
            analyst_id TEXT,
            created_at REAL,
            updated_at REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS case_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            previous_status TEXT,
            new_status TEXT,
            disposition TEXT,
            notes TEXT,
            analyst_id TEXT,
            timestamp REAL,
            FOREIGN KEY (transaction_id) REFERENCES cases(transaction_id)
        )
    """)

    conn.commit()
    conn.close()


def insert_or_update_alert(alert: Dict[str, Any], db_path: str = CASES_DB_PATH) -> None:
    """Insert a new streaming alert if not already present."""
    init_case_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    tx_id = alert.get("transaction_id", f"tx_{int(time.time()*1000)}")
    now = time.time()

    cursor.execute("""
        INSERT OR IGNORE INTO cases (
            transaction_id, timestamp, step, type, amount,
            nameOrig, nameDest, orig_drain_ratio, fraud_probability,
            risk_level, decision_threshold, case_status,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
    """, (
        tx_id,
        float(alert.get("timestamp", now)),
        int(alert.get("step", 1)),
        str(alert.get("type", "TRANSFER")),
        float(alert.get("amount", 0.0)),
        str(alert.get("nameOrig", "")),
        str(alert.get("nameDest", "")),
        float(alert.get("orig_drain_ratio", 0.0)),
        float(alert.get("fraud_probability", 0.0)),
        str(alert.get("risk_level", "HIGH")),
        float(alert.get("decision_threshold", 0.70)),
        now,
        now,
    ))

    conn.commit()
    conn.close()


def sync_alerts_from_jsonl(jsonl_path: str = ALERTS_JSONL_PATH, db_path: str = CASES_DB_PATH) -> int:
    """Sync alerts from Kafka alerts JSONL into the SQLite cases database."""
    init_case_db(db_path)
    if not os.path.isfile(jsonl_path):
        return 0

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    alert = json.loads(line)
                    insert_or_update_alert(alert, db_path=db_path)
                    count += 1
                except Exception:
                    pass
    return count


def get_cases(
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    min_score: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 50,
    db_path: str = CASES_DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieve filtered cases from the database."""
    init_case_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM cases WHERE 1=1"
    params: List[Any] = []

    if status and status != "ALL":
        query += " AND case_status = ?"
        params.append(status)

    if risk_level and risk_level != "ALL":
        query += " AND risk_level = ?"
        params.append(risk_level)

    if min_score is not None:
        query += " AND fraud_probability >= ?"
        params.append(min_score)

    if search:
        query += " AND (transaction_id LIKE ? OR nameOrig LIKE ? OR nameDest LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


def get_case_by_id(transaction_id: str, db_path: str = CASES_DB_PATH) -> Optional[Dict[str, Any]]:
    """Get single case details including audit history."""
    init_case_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM cases WHERE transaction_id = ?", (transaction_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    case_data = dict(row)

    cursor.execute(
        "SELECT * FROM case_audit_log WHERE transaction_id = ? ORDER BY timestamp ASC",
        (transaction_id,)
    )
    audit_rows = cursor.fetchall()
    case_data["audit_history"] = [dict(r) for r in audit_rows]

    conn.close()
    return case_data


def update_case_review(
    transaction_id: str,
    new_status: str,
    disposition: str,
    notes: str,
    analyst_id: str = "analyst_1",
    db_path: str = CASES_DB_PATH
) -> bool:
    """Record analyst review, update case status, and append to audit log."""
    init_case_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT case_status FROM cases WHERE transaction_id = ?", (transaction_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False

    prev_status = row["case_status"]
    now = time.time()

    cursor.execute("""
        UPDATE cases
        SET case_status = ?, analyst_disposition = ?, analyst_notes = ?,
            analyst_id = ?, updated_at = ?
        WHERE transaction_id = ?
    """, (new_status, disposition, notes, analyst_id, now, transaction_id))

    cursor.execute("""
        INSERT INTO case_audit_log (
            transaction_id, previous_status, new_status, disposition, notes, analyst_id, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (transaction_id, prev_status, new_status, disposition, notes, analyst_id, now))

    conn.commit()
    conn.close()
    return True


def get_case_metrics(db_path: str = CASES_DB_PATH) -> Dict[str, int]:
    """Return counts of cases across all workflow stages."""
    init_case_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    metrics = {
        "total": 0,
        "open": 0,
        "investigating": 0,
        "confirmed_fraud": 0,
        "false_positive": 0,
    }

    cursor.execute("SELECT COUNT(*) FROM cases")
    metrics["total"] = cursor.fetchone()[0]

    cursor.execute("SELECT case_status, COUNT(*) FROM cases GROUP BY case_status")
    for row in cursor.fetchall():
        status = row[0]
        cnt = row[1]
        if status == "OPEN":
            metrics["open"] = cnt
        elif status == "INVESTIGATING":
            metrics["investigating"] = cnt
        elif status == "CONFIRMED_FRAUD":
            metrics["confirmed_fraud"] = cnt
        elif status == "FALSE_POSITIVE":
            metrics["false_positive"] = cnt

    conn.close()
    return metrics
