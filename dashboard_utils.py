"""
Utility functions for the FraudWatch Streamlit Dashboard.
Handles API communication, CSV parsing & validation, and Plotly chart generation.
"""

from __future__ import annotations

import io
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

DEFAULT_API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

# Risk palette
COLOR_MAP = {
    "LOW": "#10B981",       # Emerald Green
    "MEDIUM": "#F59E0B",    # Amber / Orange
    "HIGH": "#EF4444",      # Crimson Red
}


# ---------------------------------------------------------------------------
# API Client Helpers
# ---------------------------------------------------------------------------
def check_api_health(base_url: str = DEFAULT_API_BASE_URL) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Check whether the FastAPI service is reachable and healthy."""
    clean_url = base_url.rstrip("/")
    url = f"{clean_url}/health"
    try:
        resp = requests.get(url, timeout=3.0)
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, {}, f"API returned status {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        # Graceful fallback: check if switching 127.0.0.1 <-> localhost resolves Windows IPv4/IPv6 loopback differences
        alt_base = None
        if "127.0.0.1" in clean_url:
            alt_base = clean_url.replace("127.0.0.1", "localhost")
        elif "localhost" in clean_url:
            alt_base = clean_url.replace("localhost", "127.0.0.1")

        if alt_base:
            try:
                alt_resp = requests.get(f"{alt_base}/health", timeout=3.0)
                if alt_resp.status_code == 200:
                    return True, alt_resp.json(), None
            except Exception:
                pass

        return False, {}, f"Could not connect to API server at {base_url}. Ensure uvicorn is running on port 8000."
    except requests.exceptions.Timeout:
        return False, {}, f"Connection to {base_url} timed out (3.0s)."
    except Exception as e:
        return False, {}, f"Unexpected connection error: {str(e)}"


def score_single_transaction(
    tx_data: Dict[str, Any],
    base_url: str = DEFAULT_API_BASE_URL,
    block_threshold: Optional[float] = None,
    review_threshold: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Call POST /predict for a single transaction."""
    url = f"{base_url.rstrip('/')}/predict"
    params = {}
    if block_threshold is not None:
        params["block_threshold"] = block_threshold
    if review_threshold is not None:
        params["review_threshold"] = review_threshold

    try:
        resp = requests.post(url, json=tx_data, params=params, timeout=5.0)
        if resp.status_code == 200:
            return True, resp.json(), None
        
        # Parse error details if available
        try:
            err_json = resp.json()
            err_msg = err_json.get("message") or err_json.get("detail") or resp.text
            if "details" in err_json:
                detail_lines = [f"• {d.get('location')}: {d.get('message')}" for d in err_json["details"]]
                err_msg += "\n" + "\n".join(detail_lines)
        except Exception:
            err_msg = resp.text
        return False, {}, f"API Error (HTTP {resp.status_code}): {err_msg}"
    except Exception as e:
        return False, {}, f"Failed to score transaction: {str(e)}"


def score_batch_transactions(
    transactions: List[Dict[str, Any]],
    base_url: str = DEFAULT_API_BASE_URL,
    block_threshold: Optional[float] = None,
    review_threshold: Optional[float] = None,
    chunk_size: int = 500,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Send transactions in chunks <= chunk_size to POST /predict/batch."""
    url = f"{base_url.rstrip('/')}/predict/batch"
    total_tx = len(transactions)
    if total_tx == 0:
        return False, {}, "No transactions provided for batch scoring."

    aggregated_results: List[Dict[str, Any]] = []
    total_fraud = 0
    total_review = 0
    total_approved = 0
    start_time = time.perf_counter()

    chunks = [transactions[i:i + chunk_size] for i in range(0, total_tx, chunk_size)]
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks):
        if progress_callback:
            pct = (idx) / total_chunks
            progress_callback(pct, f"Scoring batch chunk {idx + 1} of {total_chunks} ({len(chunk)} items)...")

        payload: Dict[str, Any] = {"transactions": chunk}
        if block_threshold is not None:
            payload["block_threshold"] = block_threshold
        if review_threshold is not None:
            payload["review_threshold"] = review_threshold

        try:
            resp = requests.post(url, json=payload, timeout=30.0)
            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("detail") or err_json.get("message") or resp.text
                except Exception:
                    err_msg = resp.text
                return False, {}, f"Batch scoring failed at chunk {idx + 1} (HTTP {resp.status_code}): {err_msg}"

            data = resp.json()
            aggregated_results.extend(data.get("results", []))
            total_fraud += data.get("fraud_count", 0)
            total_review += data.get("review_count", 0)
            total_approved += data.get("approved_count", 0)
        except Exception as e:
            return False, {}, f"Batch request error at chunk {idx + 1}: {str(e)}"

    if progress_callback:
        progress_callback(1.0, f"Completed scoring {total_tx} transactions!")

    total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    summary = {
        "total_scored": total_tx,
        "fraud_count": total_fraud,
        "review_count": total_review,
        "approved_count": total_approved,
        "block_threshold": block_threshold if block_threshold is not None else 0.90,
        "review_threshold": review_threshold if review_threshold is not None else 0.50,
        "latency_ms": total_latency_ms,
        "results": aggregated_results,
    }
    return True, summary, None


# ---------------------------------------------------------------------------
# CSV Template & Validation
# ---------------------------------------------------------------------------
def generate_csv_template(mode: str = "historical") -> str:
    """Generate sample CSV data for download."""
    if mode == "historical":
        csv_data = (
            "transaction_id,step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,mode\n"
            "TX-HIST-001,1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,historical\n"
            "TX-HIST-002,741,TRANSFER,250000.0,C123456789,250000.0,0.0,C987654321,0.0,0.0,historical\n"
            "TX-HIST-003,12,CASH_OUT,15000.0,C999888777,15000.0,0.0,C111222333,5000.0,20000.0,historical\n"
        )
    else:
        csv_data = (
            "transaction_id,step,type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,mode\n"
            "TX-LIVE-001,50,TRANSFER,50000.0,C111111111,100000.0,C222222222,0.0,live_authorization\n"
            "TX-LIVE-002,10,PAYMENT,120.50,C555444333,1500.0,M987654321,0.0,live_authorization\n"
            "TX-LIVE-003,24,CASH_OUT,2500.0,C777666555,2500.0,C888999000,100.0,live_authorization\n"
        )
    return csv_data


def validate_and_parse_csv(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate uploaded DataFrame against API schema.
    Returns (valid_transactions_list, list_of_error_strings).
    """
    valid_txs: List[Dict[str, Any]] = []
    errors: List[str] = []

    # Standardize column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    required_base = {"type", "amount", "nameOrig", "oldbalanceOrg", "nameDest"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        errors.append(f"Missing mandatory CSV columns: {', '.join(sorted(missing_base))}")
        return [], errors

    for idx, row in df.iterrows():
        row_num = idx + 1
        row_errors = []

        # Validate step
        step_val = row.get("step", 1)
        try:
            step = int(step_val) if pd.notna(step_val) else 1
            if step < 1:
                row_errors.append(f"step must be >= 1 (got {step})")
        except Exception:
            row_errors.append(f"invalid step value '{step_val}'")
            step = 1

        # Validate type
        tx_type = str(row.get("type", "")).strip().upper()
        if tx_type not in VALID_TYPES:
            row_errors.append(f"invalid type '{tx_type}'. Must be one of: {', '.join(sorted(VALID_TYPES))}")

        # Validate amount
        try:
            amount = float(row.get("amount", 0.0))
            if amount <= 0:
                row_errors.append(f"amount must be > 0 (got {amount})")
        except Exception:
            row_errors.append(f"invalid numeric amount '{row.get('amount')}'")
            amount = 0.0

        # Validate accounts
        name_orig = str(row.get("nameOrig", "")).strip()
        if not name_orig:
            row_errors.append("nameOrig cannot be empty")

        name_dest = str(row.get("nameDest", "")).strip()
        if not name_dest:
            row_errors.append("nameDest cannot be empty")

        # Validate balances
        try:
            old_orig = float(row.get("oldbalanceOrg", 0.0))
            if old_orig < 0:
                row_errors.append(f"oldbalanceOrg must be >= 0 (got {old_orig})")
        except Exception:
            row_errors.append("invalid oldbalanceOrg")
            old_orig = 0.0

        try:
            old_dest = float(row.get("oldbalanceDest", 0.0)) if pd.notna(row.get("oldbalanceDest")) else 0.0
            if old_dest < 0:
                row_errors.append(f"oldbalanceDest must be >= 0 (got {old_dest})")
        except Exception:
            old_dest = 0.0

        # Mode determination
        raw_mode = str(row.get("mode", "")).strip().lower()
        if raw_mode in ("live_authorization", "live"):
            mode = "live_authorization"
        elif raw_mode in ("historical", "offline"):
            mode = "historical"
        else:
            # Auto-infer: if new balances are present and not null, historical; else live
            if pd.notna(row.get("newbalanceOrig")) and pd.notna(row.get("newbalanceDest")):
                mode = "historical"
            else:
                mode = "live_authorization"

        new_orig = None
        new_dest = None

        if mode == "historical":
            if pd.isna(row.get("newbalanceOrig")) or pd.isna(row.get("newbalanceDest")):
                row_errors.append("historical mode requires both 'newbalanceOrig' and 'newbalanceDest'")
            else:
                try:
                    new_orig = float(row.get("newbalanceOrig"))
                    new_dest = float(row.get("newbalanceDest"))
                except Exception:
                    row_errors.append("invalid newbalanceOrig or newbalanceDest numbers")
        else:
            # Live authorization mode - new balances are optional
            if pd.notna(row.get("newbalanceOrig")):
                try:
                    new_orig = float(row.get("newbalanceOrig"))
                except Exception:
                    pass
            if pd.notna(row.get("newbalanceDest")):
                try:
                    new_dest = float(row.get("newbalanceDest"))
                except Exception:
                    pass

        tx_id = str(row.get("transaction_id", f"TX-ROW-{row_num}")).strip()

        if row_errors:
            errors.append(f"Row {row_num} (ID: {tx_id}): {', '.join(row_errors)}")
        else:
            tx_obj = {
                "step": step,
                "type": tx_type,
                "amount": amount,
                "nameOrig": name_orig,
                "oldbalanceOrg": old_orig,
                "newbalanceOrig": new_orig,
                "nameDest": name_dest,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "mode": mode,
                "transaction_id": tx_id,
            }
            valid_txs.append(tx_obj)

    return valid_txs, errors


# ---------------------------------------------------------------------------
# Plotly Chart Generators
# ---------------------------------------------------------------------------
def create_risk_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Create a donut chart for risk level distribution."""
    if df.empty or "risk_level" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available", annotations=[{"text": "No transactions scored yet", "showarrow": False}])
        return fig

    counts = df["risk_level"].value_counts().reset_index()
    counts.columns = ["risk_level", "count"]

    fig = px.pie(
        counts,
        names="risk_level",
        values="count",
        hole=0.55,
        color="risk_level",
        color_discrete_map=COLOR_MAP,
        title="Transaction Risk Distribution",
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#1E293B", width=2)),
    )
    fig.update_layout(
        showlegend=True,
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_probability_histogram(df: pd.DataFrame) -> go.Figure:
    """Create a histogram of fraud probability scores."""
    if df.empty or "fraud_probability" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No probability data")
        return fig

    fig = px.histogram(
        df,
        x="fraud_probability",
        nbins=20,
        range_x=[0.0, 1.0],
        title="Fraud Probability Distribution",
        labels={"fraud_probability": "Predicted Fraud Probability"},
        color_discrete_sequence=["#3B82F6"],
    )
    fig.update_layout(
        xaxis=dict(tickformat=".1%", title="Fraud Probability Score"),
        yaxis=dict(title="Transaction Count"),
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_type_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart grouping transaction type by risk level."""
    if df.empty or "type" not in df.columns or "risk_level" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No type breakdown data")
        return fig

    type_counts = df.groupby(["type", "risk_level"]).size().reset_index(name="count")

    fig = px.bar(
        type_counts,
        x="type",
        y="count",
        color="risk_level",
        color_discrete_map=COLOR_MAP,
        title="Risk Tiers by Transaction Type",
        barmode="group",
        labels={"type": "Transaction Type", "count": "Volume", "risk_level": "Risk Tier"},
    )
    fig.update_layout(
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def create_gauge_chart(probability: float, risk_level: str) -> go.Figure:
    """Interactive gauge meter showing fraud probability."""
    color = COLOR_MAP.get(risk_level, "#3B82F6")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100.0,
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": f"Fraud Probability: {risk_level} RISK", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [50, 90], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [90, 100], "color": "rgba(239, 68, 68, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 4},
                    "thickness": 0.8,
                    "value": 90,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Case Management & Analyst Feedback Client Helpers
# ---------------------------------------------------------------------------
def fetch_cases(
    base_url: str = DEFAULT_API_BASE_URL,
    status: str = "ALL",
    risk_level: str = "ALL",
    min_score: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 50,
) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """Fetch persistent cases from the API case store."""
    url = f"{base_url.rstrip('/')}/platform/cases"
    params: Dict[str, Any] = {"status": status, "risk_level": risk_level, "limit": limit}
    if min_score is not None:
        params["min_score"] = min_score
    if search:
        params["search"] = search

    try:
        resp = requests.get(url, params=params, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("cases", []), None
        return False, [], f"API returned status {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, [], str(e)


def fetch_case_metrics(base_url: str = DEFAULT_API_BASE_URL) -> Dict[str, int]:
    """Fetch case counts by status from the API case store."""
    url = f"{base_url.rstrip('/')}/platform/cases/summary/metrics"
    try:
        resp = requests.get(url, timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"total": 0, "open": 0, "investigating": 0, "confirmed_fraud": 0, "false_positive": 0}


def fetch_case_detail(base_url: str, transaction_id: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Fetch full case details and audit history."""
    url = f"{base_url.rstrip('/')}/platform/cases/{transaction_id}"
    try:
        resp = requests.get(url, timeout=4.0)
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"Status {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, None, str(e)


def submit_case_review(
    base_url: str,
    transaction_id: str,
    status: str,
    disposition: str,
    notes: str,
    analyst_id: str = "analyst_1",
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Submit analyst review and update case disposition."""
    url = f"{base_url.rstrip('/')}/platform/cases/{transaction_id}/review"
    payload = {
        "status": status,
        "disposition": disposition,
        "notes": notes,
        "analyst_id": analyst_id,
    }
    try:
        resp = requests.post(url, json=payload, timeout=5.0)
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"Status {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, None, str(e)

