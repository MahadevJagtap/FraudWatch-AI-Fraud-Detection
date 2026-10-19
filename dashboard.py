"""
FraudWatch — Financial Fraud Detection Dashboard
Streamlit frontend connecting to the PaySim XGBoost FastAPI scoring microservice.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import dashboard_utils as utils

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudWatch — Financial Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #047857;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #1E3A8A;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "session_scored_transactions" not in st.session_state:
    st.session_state.session_scored_transactions = []

if "preset_data" not in st.session_state:
    st.session_state.preset_data = {}


def add_to_session(results: List[Dict[str, Any]]):
    """Accumulate scored transaction records in session state."""
    st.session_state.session_scored_transactions.extend(results)


# ---------------------------------------------------------------------------
# Sidebar: Settings & API Connection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/FastAPI.svg",
        width=50,
    )
    st.title("FraudWatch Control")
    st.caption("Real-Time Fraud Detection Platform")

    api_base_url = st.text_input(
        "API Base URL",
        value=utils.DEFAULT_API_BASE_URL,
        help="Address of the running FastAPI scoring microservice.",
    )

    if st.button("🔄 Refresh Connection", use_container_width=True):
        st.rerun()

    is_healthy, health_data, health_err = utils.check_api_health(api_base_url)

    if is_healthy:
        st.success("🟢 API Server Connected")
        st.caption(f"Model: {health_data.get('model_type', 'XGBoost')} ({health_data.get('feature_count')} features)")
    else:
        st.error("🔴 API Server Offline")
        st.caption(health_err)
        st.code("cd D:\\Fraud_detection\\PaySim-Fraud-Detection\n.\\.venv\\Scripts\\python.exe -m uvicorn api.app:app --port 8000", language="powershell")

    st.markdown("---")
    st.subheader("Decision Policy Thresholds")
    st.caption("Configurable risk boundaries (Demo Policy):")

    block_thresh = st.slider(
        "Block Threshold (HIGH Risk)",
        min_value=0.50,
        max_value=0.999,
        value=0.90,
        step=0.01,
        help="Transactions with fraud probability >= this value are recommended for automated BLOCK.",
    )

    review_thresh = st.slider(
        "Review Threshold (MEDIUM Risk)",
        min_value=0.10,
        max_value=0.80,
        value=0.50,
        step=0.01,
        help="Transactions with fraud probability between review and block are flagged for human REVIEW.",
    )

    st.markdown("---")
    if st.button("🗑️ Clear Dashboard Session Data"):
        st.session_state.session_scored_transactions = []
        st.session_state.preset_data = {}
        st.rerun()

    st.caption("Milestone 3 • PaySim Platform")

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ FraudWatch: Financial Fraud Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive Analyst Dashboard connected to the PaySim XGBoost Real-Time Scoring Microservice</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer-box">'
    '<strong>Demo Policy Notice:</strong> All risk tiers, probabilities, and recommendations are based on an offline demonstration policy '
    'derived from the PaySim synthetic benchmark. These statistics reflect in-memory session activity and do not represent a live bank ledger.'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_single, tab_batch, tab_analytics, tab_governance, tab_info = st.tabs([
    "📊 Overview & Monitor",
    "🔍 Single Transaction Scoring",
    "📁 Batch CSV Scoring",
    "📈 Risk Analytics",
    "📡 Streaming & Governance",
    "ℹ️ Model & System Info",
])

# ===========================================================================
# TAB 1: Overview & Live Monitor
# ===========================================================================
with tab_overview:
    st.subheader("Session Activity & System Status")

    session_records = st.session_state.session_scored_transactions
    total_scored = len(session_records)

    high_risk_count = sum(1 for r in session_records if r.get("risk_level") == "HIGH")
    medium_risk_count = sum(1 for r in session_records if r.get("risk_level") == "MEDIUM")
    low_risk_count = sum(1 for r in session_records if r.get("risk_level") == "LOW")

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric(
            "API Status",
            "ONLINE" if is_healthy else "OFFLINE",
            delta="Healthy" if is_healthy else "Check logs",
        )
    with m_col2:
        st.metric("Total Scored (Session)", f"{total_scored:,}")
    with m_col3:
        st.metric("HIGH Risk (Block)", f"{high_risk_count:,}", delta=f"{high_risk_count} alerts", delta_color="inverse")
    with m_col4:
        st.metric("MEDIUM Risk (Review)", f"{medium_risk_count:,}")
    with m_col5:
        st.metric("LOW Risk (Approved)", f"{low_risk_count:,}")

    st.markdown("---")

    if session_records:
        df_session = pd.DataFrame(session_records)
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.plotly_chart(utils.create_risk_donut_chart(df_session), use_container_width=True)
        with c_right:
            st.plotly_chart(utils.create_probability_histogram(df_session), use_container_width=True)

        st.subheader("Recent Scored Transactions (Latest 15)")
        recent_display = df_session[[
            col for col in ["transaction_id", "fraud_probability", "risk_level", "recommended_action", "input_mode"]
            if col in df_session.columns
        ]].tail(15).iloc[::-1]

        st.dataframe(
            recent_display.style.format({"fraud_probability": "{:.4%}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No transactions have been scored during this dashboard session yet. "
            "Navigate to **Single Transaction Scoring** or **Batch CSV Scoring** to test transactions."
        )

# ===========================================================================
# TAB 2: Single Transaction Analysis
# ===========================================================================
with tab_single:
    st.subheader("Real-Time Single Transaction Scoring")
    st.write("Submit transaction parameters to `POST /predict` and evaluate fraud probability in real time.")

    # Preset Quick Load Buttons
    st.write("##### Quick Presets (Click to populate form):")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("🚨 Load Typical Fraud (Account Drained)"):
            st.session_state.preset_data = {
                "tx_id": "PRESET-FRAUD-001",
                "step": 741,
                "type": "TRANSFER",
                "amount": 250000.0,
                "orig_id": "C123456789",
                "orig_old": 250000.0,
                "orig_new": 0.0,
                "dest_id": "C987654321",
                "dest_old": 0.0,
                "dest_new": 0.0,
                "mode": "historical",
            }
            st.rerun()
    with p_col2:
        if st.button("✅ Load Legitimate Payment (Merchant)"):
            st.session_state.preset_data = {
                "tx_id": "PRESET-LEGIT-001",
                "step": 1,
                "type": "PAYMENT",
                "amount": 9839.64,
                "orig_id": "C1231006815",
                "orig_old": 170136.0,
                "orig_new": 160296.36,
                "dest_id": "M1979787155",
                "dest_old": 0.0,
                "dest_new": 0.0,
                "mode": "historical",
            }
            st.rerun()
    with p_col3:
        if st.button("⚡ Load Live In-Flight Transfer"):
            st.session_state.preset_data = {
                "tx_id": "PRESET-LIVE-001",
                "step": 48,
                "type": "TRANSFER",
                "amount": 50000.0,
                "orig_id": "C555666777",
                "orig_old": 100000.0,
                "dest_id": "C888999000",
                "dest_old": 0.0,
                "mode": "live_authorization",
            }
            st.rerun()

    preset = st.session_state.preset_data

    with st.form("single_tx_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            tx_id = st.text_input("Transaction ID", value=preset.get("tx_id", "TX-MANUAL-001"))
            tx_step = st.number_input("Step (Simulation Hour 1..744)", min_value=1, max_value=744, value=preset.get("step", 1))
            tx_type = st.selectbox(
                "Transaction Type",
                options=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                index=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"].index(preset.get("type", "TRANSFER")),
            )
        with f_col2:
            tx_amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=float(preset.get("amount", 1000.0)), step=100.0)
            orig_id = st.text_input("Origin Account (nameOrig)", value=preset.get("orig_id", "C100000001"))
            orig_old = st.number_input("Origin Old Balance ($)", min_value=0.0, value=float(preset.get("orig_old", 1000.0)), step=100.0)
        with f_col3:
            mode_choice = st.radio(
                "Feature Ingestion Mode",
                options=["historical", "live_authorization"],
                index=0 if preset.get("mode", "historical") == "historical" else 1,
                help=(
                    "historical: Uses exact post-settlement balances.\n"
                    "live_authorization: Estimates post-settlement balances pre-authorization."
                ),
            )
            dest_id = st.text_input("Destination Account (nameDest)", value=preset.get("dest_id", "M200000002"))
            dest_old = st.number_input("Destination Old Balance ($)", min_value=0.0, value=float(preset.get("dest_old", 0.0)), step=100.0)

        # Historical post-settlement inputs
        if mode_choice == "historical":
            st.markdown("###### Post-Settlement Balance Information (Required in Historical Mode):")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                orig_new = st.number_input("Origin New Balance ($)", min_value=0.0, value=float(preset.get("orig_new", 0.0)), step=100.0)
            with b_col2:
                dest_new = st.number_input("Destination New Balance ($)", min_value=0.0, value=float(preset.get("dest_new", 0.0)), step=100.0)
        else:
            orig_new = None
            dest_new = None
            st.info("ℹ️ In `live_authorization` mode, new balances are estimated pre-settlement.")

        submit_btn = st.form_submit_button("⚡ Score Transaction with FastAPI", use_container_width=True)

    if submit_btn:
        if not is_healthy:
            st.error("Cannot score transaction: FastAPI service is offline. Please start it on port 8000.")
        else:
            tx_payload = {
                "step": int(tx_step),
                "type": tx_type,
                "amount": float(tx_amount),
                "nameOrig": orig_id.strip(),
                "oldbalanceOrg": float(orig_old),
                "newbalanceOrig": float(orig_new) if orig_new is not None else None,
                "nameDest": dest_id.strip(),
                "oldbalanceDest": float(dest_old),
                "newbalanceDest": float(dest_new) if dest_new is not None else None,
                "mode": mode_choice,
                "transaction_id": tx_id.strip(),
            }

            with st.spinner("Calling POST /predict..."):
                ok, pred, err = utils.score_single_transaction(
                    tx_payload,
                    base_url=api_base_url,
                    block_threshold=block_thresh,
                    review_threshold=review_thresh,
                )

            if ok:
                st.success("✅ Transaction scored successfully!")
                prob = pred["fraud_probability"]
                risk = pred["risk_level"]
                action = pred["recommended_action"]

                # Store with input fields for analytics
                full_record = {**tx_payload, **pred}
                add_to_session([full_record])

                # Visual Result Display
                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    st.plotly_chart(utils.create_gauge_chart(prob, risk), use_container_width=True)

                with res_col2:
                    st.markdown("### Decision Summary")
                    if risk == "HIGH":
                        badge_html = f'<span class="badge-high">HIGH RISK • {action}</span>'
                    elif risk == "MEDIUM":
                        badge_html = f'<span class="badge-medium">MEDIUM RISK • {action}</span>'
                    else:
                        badge_html = f'<span class="badge-low">LOW RISK • {action}</span>'

                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.write(f"**Transaction ID:** `{pred.get('transaction_id')}`")
                    st.write(f"**Fraud Probability:** `{prob:.4%}` (`{prob:.6f}`)")
                    st.write(f"**Recommended Action:** `{action}`")
                    st.write(f"**Thresholds:** Block $\\ge {pred['block_threshold']}$, Review $\\ge {pred['review_threshold']}$")

                if pred.get("audit_note"):
                    st.warning(f"⚠️ **Audit Notice:** {pred['audit_note']}")

                with st.expander("🔍 View Raw API Response JSON"):
                    st.json(pred)
            else:
                st.error(f"Scoring Failed:\n{err}")

# ===========================================================================
# TAB 3: Batch CSV Analysis
# ===========================================================================
with tab_batch:
    st.subheader("Batch Transaction Scoring")
    st.write("Upload a CSV file of transactions to score via `POST /predict/batch`.")

    # Templates Download
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.download_button(
            "📥 Download Historical CSV Template",
            data=utils.generate_csv_template("historical"),
            file_name="paysim_historical_template.csv",
            mime="text/csv",
            help="Template containing settled balance columns.",
        )
    with t_col2:
        st.download_button(
            "📥 Download Live Authorization CSV Template",
            data=utils.generate_csv_template("live_authorization"),
            file_name="paysim_live_authorization_template.csv",
            mime="text/csv",
            help="Template for in-flight transactions without settled balances.",
        )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV file to score", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Uploaded CSV contains **{len(df_upload):,} rows** and **{len(df_upload.columns)} columns**.")

            valid_txs, val_errors = utils.validate_and_parse_csv(df_upload)

            st.write(f"• **Valid Rows for Scoring:** {len(valid_txs):,}")
            if val_errors:
                st.warning(f"• **Invalid Rows Detected:** {len(val_errors):,} (see details below)")
                with st.expander("⚠️ View Row Validation Errors"):
                    for e in val_errors[:50]:
                        st.write(e)
                    if len(val_errors) > 50:
                        st.caption(f"...and {len(val_errors) - 50} more errors.")

            if valid_txs:
                if st.button(f"🚀 Score {len(valid_txs):,} Valid Transactions with FastAPI", type="primary"):
                    if not is_healthy:
                        st.error("FastAPI service is offline. Start the API before running batch scoring.")
                    else:
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()

                        def update_progress(pct: float, msg: str):
                            progress_bar.progress(pct)
                            status_text.text(msg)

                        ok, batch_res, b_err = utils.score_batch_transactions(
                            valid_txs,
                            base_url=api_base_url,
                            block_threshold=block_thresh,
                            review_threshold=review_thresh,
                            chunk_size=500,
                            progress_callback=update_progress,
                        )

                        if ok:
                            st.success(f"Batch scoring completed in {batch_res['latency_ms']:.2f} ms!")
                            progress_bar.empty()
                            status_text.empty()

                            # Combine inputs with results
                            results_list = batch_res["results"]
                            merged_records = []
                            for in_tx, res_tx in zip(valid_txs, results_list):
                                merged_records.append({**in_tx, **res_tx})

                            add_to_session(merged_records)

                            df_results = pd.DataFrame(merged_records)

                            # Summary KPI Row
                            k1, k2, k3, k4 = st.columns(4)
                            k1.metric("Batch Total Scored", f"{batch_res['total_scored']:,}")
                            k2.metric("Flagged Fraud (Block)", f"{batch_res['fraud_count']:,}", delta_color="inverse")
                            k3.metric("Flagged Review", f"{batch_res['review_count']:,}")
                            k4.metric("Approved", f"{batch_res['approved_count']:,}")

                            st.markdown("---")
                            st.subheader("Scored Results Table")

                            # Filters
                            flt_col1, flt_col2 = st.columns(2)
                            with flt_col1:
                                risk_filter = st.multiselect(
                                    "Filter by Risk Level",
                                    options=["HIGH", "MEDIUM", "LOW"],
                                    default=["HIGH", "MEDIUM", "LOW"],
                                )
                            with flt_col2:
                                mode_filter = st.multiselect(
                                    "Filter by Input Mode",
                                    options=list(df_results["input_mode"].unique()),
                                    default=list(df_results["input_mode"].unique()),
                                )

                            filtered_df = df_results[
                                (df_results["risk_level"].isin(risk_filter)) &
                                (df_results["input_mode"].isin(mode_filter))
                            ]

                            display_cols = [
                                "transaction_id", "type", "amount", "fraud_probability",
                                "risk_level", "recommended_action", "input_mode", "nameOrig", "nameDest"
                            ]
                            cols_to_show = [c for c in display_cols if c in filtered_df.columns]

                            st.dataframe(
                                filtered_df[cols_to_show].style.format({"fraud_probability": "{:.4%}", "amount": "${:,.2f}"}),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # CSV Export
                            csv_export = filtered_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download Scored Results CSV",
                                data=csv_export,
                                file_name="scored_fraud_results.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error(f"Batch Scoring Error:\n{b_err}")
            else:
                st.error("No valid transactions could be parsed from the uploaded CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# ===========================================================================
# TAB 4: Risk Analytics
# ===========================================================================
with tab_analytics:
    st.subheader("Aggregated Risk & Prediction Analytics")
    session_data = st.session_state.session_scored_transactions

    if session_data:
        df_analytics = pd.DataFrame(session_data)

        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.plotly_chart(utils.create_risk_donut_chart(df_analytics), use_container_width=True)
        with a_col2:
            st.plotly_chart(utils.create_probability_histogram(df_analytics), use_container_width=True)

        st.plotly_chart(utils.create_type_breakdown_chart(df_analytics), use_container_width=True)

        st.subheader("Risk Tier Breakdown Summary")
        summary_stats = df_analytics.groupby("risk_level").agg(
            Volume=("transaction_id", "count"),
            Avg_Probability=("fraud_probability", "mean"),
            Min_Probability=("fraud_probability", "min"),
            Max_Probability=("fraud_probability", "max"),
            Avg_Amount=("amount", "mean"),
            Total_Amount=("amount", "sum"),
        ).reset_index()

        st.dataframe(
            summary_stats.style.format({
                "Avg_Probability": "{:.4%}",
                "Min_Probability": "{:.4%}",
                "Max_Probability": "{:.4%}",
                "Avg_Amount": "${:,.2f}",
                "Total_Amount": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No transaction data available yet to build analytics. Score single transactions or upload a batch CSV first.")

# ===========================================================================
# TAB 5: Streaming & Governance (Kafka, Feast, Evidently AI, Multi-Model)
# ===========================================================================
with tab_governance:
    st.subheader("Streaming Alerts, Feature Store & Model Governance")

    gov_subtab1, gov_subtab2, gov_subtab3, gov_subtab4 = st.tabs([
        "🚨 Kafka Fraud Alerts & Analyst Cases",
        "⚖️ Evidently AI Drift & Promotion Gate",
        "🏆 Multi-Model Benchmark (XGBoost / FastAI / PyG)",
        "🏪 Feast Feature Store Online Query"
    ])

    # 1. Kafka Alerts & Case Management
    # 1. Kafka Alerts & Persistent Case Management
    with gov_subtab1:
        st.markdown("#### Analyst Case Management & Streaming Alerts Hub")
        st.caption("Live streaming fraud alerts from Kafka worker with persistent SQLite case workflow and analyst feedback persistence.")

        # Sync any newly emitted alerts from JSONL into cases.db
        try:
            from api.case_store import sync_alerts_from_jsonl
            sync_alerts_from_jsonl()
        except Exception:
            pass

        # Case Metrics Bar
        case_metrics = utils.fetch_case_metrics(api_url)
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Total Cases", f"{case_metrics.get('total', 0):,}")
        with k2:
            st.metric("Open Alerts", f"{case_metrics.get('open', 0):,}")
        with k3:
            st.metric("Investigating", f"{case_metrics.get('investigating', 0):,}")
        with k4:
            st.metric("Confirmed Fraud", f"{case_metrics.get('confirmed_fraud', 0):,}", delta="High Risk", delta_color="inverse")
        with k5:
            st.metric("False Positives", f"{case_metrics.get('false_positive', 0):,}")

        st.markdown("---")

        # Filters & Search
        st.markdown("##### Filter & Search Cases")
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 3, 1])
        with f_col1:
            status_filter = st.selectbox(
                "Case Status",
                ["ALL", "OPEN", "INVESTIGATING", "CONFIRMED_FRAUD", "FALSE_POSITIVE"],
                index=0,
                key="case_status_filter"
            )
        with f_col2:
            risk_filter = st.selectbox(
                "Risk Level",
                ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                index=0,
                key="case_risk_filter"
            )
        with f_col3:
            search_query = st.text_input(
                "Search by ID, Origin, or Destination",
                value="",
                placeholder="e.g. C123, M197, tx_...",
                key="case_search_query"
            )
        with f_col4:
            st.write("")
            st.write("")
            refresh_cases = st.button("🔄 Refresh", key="refresh_cases_btn")

        # Fetch filtered cases from SQLite case store
        ok_cases, cases_list, err_cases = utils.fetch_cases(
            base_url=api_url,
            status=status_filter,
            risk_level=risk_filter,
            search=search_query if search_query.strip() else None,
            limit=100
        )

        if ok_cases and cases_list:
            df_cases = pd.DataFrame(cases_list)
            st.markdown(f"**Showing {len(df_cases)} matching cases:**")

            display_cols = [c for c in [
                "transaction_id", "case_status", "risk_level", "fraud_probability",
                "amount", "type", "orig_drain_ratio", "nameOrig", "nameDest", "analyst_disposition"
            ] if c in df_cases.columns]

            st.dataframe(
                df_cases[display_cols].style.format({
                    "fraud_probability": "{:.2%}",
                    "amount": "${:,.2f}",
                    "orig_drain_ratio": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # Transaction Detail & Analyst Disposition Section
            st.markdown("---")
            st.markdown("##### Case Review & Decision Disposition")

            c_select, c_action = st.columns([3, 3])
            with c_select:
                selected_tx_id = st.selectbox(
                    "Select Case to Inspect & Review",
                    df_cases["transaction_id"].unique(),
                    key="selected_case_id"
                )

                # Fetch full case detail and audit history
                ok_det, case_det, _ = utils.fetch_case_detail(api_url, selected_tx_id)
                if ok_det and case_det:
                    st.markdown(f"**Transaction Details for `{selected_tx_id}`:**")
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.write(f"• **Type:** `{case_det.get('type')}`")
                        st.write(f"• **Amount:** `${case_det.get('amount', 0.0):,.2f}`")
                        st.write(f"• **Origin:** `{case_det.get('nameOrig')}`")
                        st.write(f"• **Destination:** `{case_det.get('nameDest')}`")
                    with info_col2:
                        st.write(f"• **Fraud Score:** `{case_det.get('fraud_probability', 0.0):.2%}`")
                        st.write(f"• **Risk Level:** `{case_det.get('risk_level')}`")
                        st.write(f"• **Drain Ratio:** `{case_det.get('orig_drain_ratio', 0.0):.4f}`")
                        st.write(f"• **Current Status:** `{case_det.get('case_status')}`")

                    # Audit History
                    audit_history = case_det.get("audit_history", [])
                    if audit_history:
                        st.caption(f"Audit Trail ({len(audit_history)} events):")
                        for event in audit_history:
                            st.caption(
                                f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.get('timestamp', time.time())))} | "
                                f"Analyst: `{event.get('analyst_id')}` | Status: `{event.get('previous_status')} -> {event.get('new_status')}` | "
                                f"Disposition: `{event.get('disposition')}` | Notes: {event.get('notes')}"
                            )

            with c_action:
                st.markdown("**Update Case Status & Submit Analyst Label:**")
                new_status = st.selectbox(
                    "New Case Status",
                    ["CONFIRMED_FRAUD", "FALSE_POSITIVE", "INVESTIGATING", "OPEN"],
                    key="review_new_status"
                )
                disposition = st.selectbox(
                    "Fraud Pattern / Disposition Category",
                    [
                        "ACCOUNT_TAKEOVER_FULL_DRAIN",
                        "MONEY_MULE_RAPID_TRANSFER",
                        "CARD_NOT_PRESENT_ANOMALY",
                        "LEGITIMATE_HIGH_VALUE_TRANSFER",
                        "AUTHORIZED_COMMERCIAL_PAYMENT",
                        "BENIGN_RECIPIENT_CLUSTER",
                        "PENDING_CUSTOMER_CONTACT",
                    ],
                    key="review_disposition"
                )
                analyst_notes = st.text_area(
                    "Investigation Notes & Feedback",
                    value="Account drain pattern verified against recipient mule cluster. Confirmed unauthorized withdrawal.",
                    height=85,
                    key="review_notes"
                )
                analyst_id = st.text_input("Analyst ID", value="analyst_1", key="review_analyst_id")

                if st.button("💾 Submit Analyst Disposition", type="primary", key="save_disposition_btn"):
                    ok_sub, res_sub, err_sub = utils.submit_case_review(
                        base_url=api_url,
                        transaction_id=selected_tx_id,
                        status=new_status,
                        disposition=disposition,
                        notes=analyst_notes,
                        analyst_id=analyst_id
                    )
                    if ok_sub:
                        st.success(f"Case `{selected_tx_id}` updated to `{new_status}` with disposition `{disposition}`!")
                        st.rerun()
                    else:
                        st.error(f"Failed to submit review: {err_sub}")

            # Export cases
            csv_cases = df_cases.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Filtered Cases to CSV",
                data=csv_cases,
                file_name="fraud_cases_export.csv",
                mime="text/csv",
                key="download_cases_btn"
            )
        else:
            st.info("No cases matching the selected filters found in the case store. Run `python streaming/consumer_worker.py` to ingest alerts.")

    # 2. Evidently AI Drift & Retraining Gate
    with gov_subtab2:
        st.markdown("#### Evidently AI Drift Audit & Controlled Retraining Gate")
        audit_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "model_promotion_audit.json")
        if os.path.isfile(audit_file):
            with open(audit_file, "r", encoding="utf-8") as f:
                audit = json.load(f)

            g1, g2, g3 = st.columns(3)
            with g1:
                st.metric("Champion Model AUPRC", f"{audit.get('champion_auprc', 0.0):.4f}")
            with g2:
                st.metric("Candidate Retrained AUPRC", f"{audit.get('candidate_auprc', 0.0):.4f}")
            with g3:
                st.metric("Governance Promotion Gate", audit.get("promotion_verdict", "UNKNOWN"), delta="Safe Promotion" if audit.get("is_model_promoted") else "Champion Retained")

            st.info(f"**Policy Invariant:** {audit.get('policy')}")
            st.write(f"**Audit Explanation:** {audit.get('explanation')}")
        else:
            st.info("No drift audit record found. Run `python src/10_drift_and_retrain_trigger.py`.")

    # 3. Multi-Model Benchmark Comparison
    with gov_subtab3:
        st.markdown("#### Multi-Model Performance Comparison Matrix")
        model_comparison = pd.DataFrame([
            {
                "Model Architecture": "Pre-Authorization XGBoost (Champion)",
                "Framework": "XGBoost 3.x",
                "Features": "16 Pre-Auth Features (Zero Leakage)",
                "Test AUPRC": 0.9920,
                "Test ROC-AUC": 0.9999,
                "Recall (Test)": "100.00%",
                "Precision (Test)": "90.81%",
                "Role": "Primary Real-Time Authorization Gatekeeper"
            },
            {
                "Model Architecture": "FastAI Tabular Deep Learning",
                "Framework": "FastAI / PyTorch",
                "Features": "Categorical Embeddings + MLP [200, 100]",
                "Test AUPRC": 0.9592,
                "Test ROC-AUC": 0.9993,
                "Recall (Test)": "85.31%",
                "Precision (Test)": "91.69%",
                "Role": "Tabular Deep Learning Entity-Embedding Benchmark"
            },
            {
                "Model Architecture": "PyTorch Geometric GraphSAGE",
                "Framework": "PyG / PyTorch",
                "Features": "Bipartite Account-Transaction Graph",
                "Test AUPRC": 0.0031,
                "Test ROC-AUC": 0.9211,
                "Recall (Test)": "88.10%",
                "Precision (Test)": "0.56%",
                "Role": "Structural Topology & Mule-Ring Detection Layer"
            },
            {
                "Model Architecture": "Baseline Model (Legacy)",
                "Framework": "XGBoost",
                "Features": "9 Features (Includes Post-Settlement Residuals)",
                "Test AUPRC": 0.8428,
                "Test ROC-AUC": 0.9984,
                "Recall (Test)": "98.71%",
                "Precision (Test)": "95.20%",
                "Role": "Historical Baseline Comparison Model"
            }
        ])
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)

    # 4. Feast Online Store Query Tool
    with gov_subtab4:
        st.markdown("#### Feast Feature Store Online Retrieval")
        st.write("Query the Feast SQLite online store for real-time account feature vectors at authorization time.")
        q_acc = st.text_input("Enter Account ID to Query", "C1231006815")
        if st.button("Fetch Features from Feast"):
            try:
                from feature_store.client import get_online_account_features
                f_data = get_online_account_features([q_acc])
                st.json(f_data)
            except Exception as e:
                st.error(f"Feast query error: {e}")

# ===========================================================================
# TAB 6: Model & System Info
# ===========================================================================
with tab_info:
    st.subheader("Underlying Machine Learning Architecture")

    if is_healthy:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Architecture", health_data.get("model_type", "XGBoost"))
        m2.metric("Features Count", f"{health_data.get('feature_count')}")
        m3.metric("Training AUPRC", f"{health_data.get('training_auprc', 0.0):.4f}")
        m4.metric("Training Recall", f"{health_data.get('training_recall', 0.0):.2%}")

        st.markdown("#### Expected Features (11 Columns in Exact Order):")
        st.write(health_data.get("expected_features", []))
    else:
        st.warning("FastAPI service is offline. Connect to view live model metadata.")

    st.markdown("---")
    st.markdown("### Technical Disclosure: Feature Derivation & Leakage Notice")
    st.write(
        """
        - **Historical Benchmark Mode (`mode="historical"`):**
          Uses the exact balance residuals derived from the PaySim dataset (`errorBalanceOrig` and `errorBalanceDest`).
          These achieve the benchmark AUPRC of **0.8428** on held-out test data.
        
        - **Live Authorization Mode (`mode="live_authorization"`):**
          In real-world payment authorization, post-settlement balances do not exist prior to payment clearance.
          The API provides a simulation mode that estimates pre-settlement residuals.
        
        - **Roadmap for Live Production Deployment:**
          For an uncompromised live gateway, the XGBoost model should be retrained exclusively on pre-authorization attributes
          (amount, historical velocity, time of day, merchant flags, transaction type) without relying on post-settlement balances.
        """
    )
