# FraudWatch — Model Comparison & Evaluation Audit Report

**Generated on:** 2026-10-18  
**Experiment:** `fraudwatch-preauth-models`  
**Tracking URI:** `sqlite:///mlflow.db`  

---

## 1. Executive Summary Table

| Metric / Dimension | Baseline XGBoost (Legacy) | Pre-Auth XGBoost (Champion) | FastAI Tabular Neural Net |
| :--- | :--- | :--- | :--- |
| **Leakage Status** | ⚠️ **LEAKAGE-PRONE** | ✅ **LEAKAGE-FREE** | ✅ **LEAKAGE-FREE** |
| **Feature Contract** | 11 features (post-settlement residuals) | 16 pre-authorization features | 16 pre-authorization features |
| **Split Strategy** | Random 80/20 Stratified Shuffle | Chronological Days 1–24 / 25–27 / 28–31 | Chronological Days 1–24 / 25–27 / 28–31 |
| **Threshold Tuning** | Default 0.50 | **Validation Set Only** (0.8247) | **Validation Set Only** (0.9763) |
| **Test Evaluation** | Random test slice | **Untouched Out-of-Time Test (Days 28–31)** | **Untouched Out-of-Time Test (Days 28–31)** |
| **Test Samples** | 1,272,524 (Random) | 81,110 (1,048 frauds) | 81,110 (1,048 frauds) |
| **Test AUPRC** | 0.8428 | **0.9917** | **0.9550** |
| **Test ROC-AUC** | 0.9969 | **0.9999** | **0.9993** |
| **Test Precision** | 13.92% | **92.09%** | **93.61%** |
| **Test Recall** | 91.97% | **97.71%** | **82.44%** |
| **Test F1 Score** | 0.2418 | **0.9481** | **0.8767** |
| **Confusion Matrix** | TP=1511, FP=9343, FN=132, TN=1261538 | TP=1024, FP=88, FN=24, TN=79974 | TP=864, FP=59, FN=184, TN=80003 |
| **Training Time** | ~110s | **26.23s** | **12.79s** |
| **CPU Latency / Row** | ~0.0005 ms | **0.0003 ms** | **0.0126 ms** |

---

## 2. In-Depth Comparative Analysis

### A. Pre-Auth XGBoost vs. FastAI Tabular Neural Net
Both models were trained and evaluated under the **exact same pre-authorization feature contract** and the **exact same chronological evaluation partitions**:
1. **Recall & Detection Rate:**
   - **Pre-Auth XGBoost:** Detected **1,024 of 1,048 frauds (97.71% recall)** with 24 false negatives.
   - **FastAI Tabular:** Detected **864 of 1,048 frauds (82.44% recall)** with 184 false negatives.
   - XGBoost's recursive decision trees split cleanly on sharp ledger ratios (`orig_drain_ratio`), which gives it superior sensitivity to account-takeover liquidations.
2. **Precision & False Positives:**
   - **FastAI Tabular:** Achieved **93.61% precision** with only 59 false positives across 80,062 legitimate transactions.
   - **Pre-Auth XGBoost:** Achieved **92.09% precision** with 88 false positives.
   - FastAI's continuous entity embeddings provide superior generalization on low-risk merchant payments, generating 33% fewer false alarms.
3. **Inference Latency:**
   - XGBoost executes in **0.0003 ms per row** (>3M rows/sec on CPU), making it ideal for sub-millisecond payment authorization gateways.
   - FastAI requires PyTorch tensor allocation and linear layer matrix multiplications, yielding **0.0126 ms per row** (~79k rows/sec on CPU).

### B. Legacy Baseline Disclosure
The legacy benchmark model relied on `errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`. Because `newbalanceOrig` reflects the ledger state *after* payment clearing, this feature is not legitimately available at authorization time. Furthermore, the legacy model was evaluated on a random train/test split where transactions from the same hours appeared in both sets. The new Pre-Auth XGBoost and FastAI models resolve both vulnerabilities.

---

## 3. PaySim Synthetic Data Limitations & Transfer Risks
- **Account-Drain Rule Invariant:** In PaySim, synthetic fraud agents almost exclusively execute total account liquidation (`amount == oldbalanceOrg`), making `orig_drain_ratio` the dominant feature (43.78% feature importance). In real banking systems, fraudsters frequently execute micro-authorizations or incremental transfers to evade AML rules.
- **Singleton Sender Topology:** 99.85% of PaySim senders appear only once, rendering sender-side transaction velocity near zero. Real-world systems require persistent multi-month account histories.
