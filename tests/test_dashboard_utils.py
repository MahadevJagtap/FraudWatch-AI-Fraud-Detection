"""
Unit tests for dashboard_utils.py
"""

from __future__ import annotations

import pandas as pd
import pytest

import dashboard_utils as utils


def test_generate_csv_template():
    hist_tpl = utils.generate_csv_template("historical")
    assert "newbalanceOrig" in hist_tpl
    assert "newbalanceDest" in hist_tpl
    df_hist = pd.read_csv(pd.io.common.StringIO(hist_tpl))
    assert len(df_hist) >= 2

    live_tpl = utils.generate_csv_template("live_authorization")
    assert "newbalanceOrig" not in live_tpl
    df_live = pd.read_csv(pd.io.common.StringIO(live_tpl))
    assert len(df_live) >= 2


def test_validate_and_parse_csv_valid_historical():
    csv_text = (
        "transaction_id,step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,mode\n"
        "TX-01,1,PAYMENT,100.0,C123,1000.0,900.0,M456,0.0,0.0,historical\n"
    )
    df = pd.read_csv(pd.io.common.StringIO(csv_text))
    valid, errors = utils.validate_and_parse_csv(df)
    assert len(valid) == 1
    assert len(errors) == 0
    assert valid[0]["transaction_id"] == "TX-01"
    assert valid[0]["amount"] == 100.0


def test_validate_and_parse_csv_invalid_rows():
    csv_text = (
        "transaction_id,step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,mode\n"
        "TX-BAD-TYPE,1,INVALID_TYPE,100.0,C123,1000.0,900.0,M456,0.0,0.0,historical\n"
        "TX-BAD-AMT,1,PAYMENT,-50.0,C123,1000.0,950.0,M456,0.0,0.0,historical\n"
    )
    df = pd.read_csv(pd.io.common.StringIO(csv_text))
    valid, errors = utils.validate_and_parse_csv(df)
    assert len(valid) == 0
    assert len(errors) == 2


def test_chart_helpers():
    sample_df = pd.DataFrame([
        {"transaction_id": "TX-1", "type": "TRANSFER", "amount": 1000.0, "risk_level": "HIGH", "fraud_probability": 0.95},
        {"transaction_id": "TX-2", "type": "PAYMENT", "amount": 50.0, "risk_level": "LOW", "fraud_probability": 0.01},
        {"transaction_id": "TX-3", "type": "CASH_OUT", "amount": 500.0, "risk_level": "MEDIUM", "fraud_probability": 0.65},
    ])
    fig_donut = utils.create_risk_donut_chart(sample_df)
    assert fig_donut is not None

    fig_hist = utils.create_probability_histogram(sample_df)
    assert fig_hist is not None

    fig_type = utils.create_type_breakdown_chart(sample_df)
    assert fig_type is not None

    fig_gauge = utils.create_gauge_chart(0.95, "HIGH")
    assert fig_gauge is not None


def test_api_client_calls():
    # Test health check against live running API
    healthy, data, err = utils.check_api_health("http://127.0.0.1:8000")
    if healthy:
        assert data.get("status") == "healthy"

        # Test single transaction call
        tx_data = {
            "step": 1,
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "C123",
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 900.0,
            "nameDest": "M456",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "mode": "historical",
            "transaction_id": "TX-UNIT-01",
        }
        ok, res, err = utils.score_single_transaction(tx_data, "http://127.0.0.1:8000")
        assert ok is True
        assert res["transaction_id"] == "TX-UNIT-01"
        assert "fraud_probability" in res
