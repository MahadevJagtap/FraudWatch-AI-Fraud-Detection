# PaySim Fraud Detection

Production-style ML pipeline for the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
synthetic mobile-money dataset. Detects fraudulent transactions on a
6.3M-row, highly imbalanced stream (fraud rate ≈ 0.13%).

## Pipeline

```
raw CSV  ->  01_data_cleaning.py  ->  paysim_cleaned.parquet
              ->  02_feature_eng.py    ->  X_features.parquet  +  y_target.parquet
              ->  03_train_model.py    ->  fraud_xgb_model.json  +  feature_importance.csv
```

Each stage is a standalone CLI script and can be re-run independently.

## Layout

```
paysim-fraud-detection/
├── data/
│   ├── raw/                 # raw CSV (gitignored)
│   └── processed/           # parquet artifacts
├── models/                  # trained XGBoost json (gitignored)
├── src/
│   ├── __init__.py
│   ├── 01_data_cleaning.py
│   ├── 02_feature_eng.py
│   └── 03_train_model.py
├── reports/                 # feature importance, future plots
├── .gitignore
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place the raw PaySim CSV in `data/raw/` (any `*.csv` filename is picked up
automatically; the first one alphabetically is used).

## Run

From the project root:

```bash
python src/01_data_cleaning.py    # ~25s, 6.3M rows -> ~110 MB parquet
python src/02_feature_eng.py      # ~3s,  11 engineered features
python src/03_train_model.py      # ~110s training on CPU; AUPRC ~ 0.84
```

## Pipeline Highlights


# 🛡️ Synthetic Financial Datasets For Fraud Detection (PaySim)

## 📌 Overview
This repository contains a production-grade Machine Learning pipeline designed to detect fraudulent mobile money transactions using the **PaySim** synthetic dataset. The project focuses heavily on rigorous data engineering, memory optimization, and preventing data leakage to ensure the model is deployable and reliable in a real-world financial system.

## ⚠️ Key Challenges & Engineering Solutions

### 1. Data Leakage Prevention (Critical)
In real-world systems, fraudulent transactions are canceled, making post-transaction balance columns highly unreliable for training. 
* **Solution:** Extracted behavioral indicators (`errorBalanceOrig` and `errorBalanceDest`) to capture account-draining behavior, then **strictly dropped** the original balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) before training.

### 2. Extreme Class Imbalance
Fraudulent transactions represent a tiny fraction of the 24 million records.
* **Solution:** Dynamically calculated the negative-to-positive class ratio and applied `scale_pos_weight` in the **XGBoost** classifier. Evaluated the model strictly using **AUPRC** (Area Under the Precision-Recall Curve) and **F1-Score**, entirely discarding standard Accuracy.

### 3. Memory Optimization & Scaling
Processing millions of financial logs can easily cause Out-Of-Memory (OOM) errors.
* **Solution:** Applied aggressive data downcasting (`float32`, `int8`, `category`). Exported all intermediate datasets as compressed `.parquet` files using `pyarrow` to ensure fast I/O and minimal memory footprint.

## 🧠 Feature Engineering Highlights
* **Merchant Accounts (`is_merchant`):** Accounts starting with 'M' do not track balance data. Created a vectorized binary feature to flag merchants, which became the model's most critical decision node.
* **Temporal Patterns:** Decomposed the `step` column (representing hours) into `hour_of_day` and `day_of_week` to capture the operating windows of fraudsters.

## 📊 Model Performance & Insights
An **XGBoost Classifier** was trained with stratified splitting. The Top 3 most important features identified by the model were:
1. `is_merchant` (**81.2%**) - *Fraudsters in this dataset do not use merchant accounts.*
2. `errorBalanceOrig` (**16.6%**) - *Captured the mathematical anomalies of accounts being emptied.*
3. `amount` (**0.9%**)

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy, PyArrow
* **Modeling:** Scikit-Learn, XGBoost
* **Storage Format:** Parquet / JSON (Model)

### 01 - Cleaning
- **Memory optimization**: int64 -> int8/16/32, float64 -> float32, `type`
  -> `category`. Typical reduction ≈ **84%** (~706 MB -> ~109 MB).
- **Leakage prevention**: drops `oldbalanceOrg`, `newbalanceOrig`,
  `oldbalanceDest`, `newbalanceDest`, `nameOrig`, `nameDest` after deriving
  the leakage-safe `errorBalanceOrig` / `errorBalanceDest` features. These
  residuals are the only balance-derived signals available at
  authorization time in a real system, so the model can never peek at the
  answer via post-settlement balances.
- **Merchant flag**: `is_merchant = nameDest.startswith("M")` (int8) -
  merchants have zero balances in PaySim by design.
- **Audit**: NaN per-column and full-row duplicate counts are printed
  before any destructive operation. Duplicates are dropped on the *raw*
  float64 data only - never after float32 rounding - to avoid inventing
  phantom duplicates.

### 02 - Feature Engineering
- **Temporal**: `hour_of_day = step % 24`, `day_of_week = (step // 24) % 7`
  (both int8).
- **One-hot**: `type` -> 5 int8 columns (`type_CASH_IN`, ...,
  `type_TRANSFER`); reindexed against the full PaySim taxonomy so the
  schema is stable across chunks.
- **Drops**: `step`, `isFlaggedFraud`, and the high-cardinality IDs (the
  latter already removed in step 01; a runtime assertion guards against
  regressions).
- **Output**: `X_features.parquet` (11 columns, ~75 MB) and
  `y_target.parquet` (1 column, 30 KB).

### 03 - Training
- **Model**: XGBClassifier (`tree_method="hist"`, `device="cuda"` with CPU
  fallback, `eval_metric="aucpr"`, 400 trees, max_depth=8).
- **Imbalance**: `scale_pos_weight = neg / pos` computed from the training
  fold only.
- **Split**: stratified 80/20 (`stratify=y`, `random_state=42`).
- **Primary metric**: **AUPRC (Average Precision)**. Accuracy is
  intentionally not reported because the class imbalance makes it
  meaningless (predicting `0` for everything yields 99.87% accuracy).
  Precision, Recall, and F1 for the positive class are reported alongside.
- **Feature importance**: top-5 gain-weighted importances are printed and
  saved to `reports/feature_importance.csv`. On this dataset,
  `is_merchant` alone accounts for ~81% of the model's gain, with
  `errorBalanceOrig` contributing another ~17%.

## Reproducible Metrics (last run on the full PaySim file)

| Metric             | Value   |
|--------------------|---------|
| Precision (fraud)  | 0.1412  |
| Recall (fraud)     | 0.9184  |
| F1-Score (fraud)   | 0.2448  |
| **AUPRC**          | **0.8436** |

Training time: ~104s on CPU. Validation AUPRC peaks around iteration ~150,
so production re-runs should consider `early_stopping_rounds=50` with
`n_estimators=1000` for marginal gains.

## Reproducibility

All randomness is pinned via `random_state=42` in both `train_test_split`
and `XGBClassifier`.

## License

Data: PaySim synthetic dataset (Kaggle). Code: MIT.
