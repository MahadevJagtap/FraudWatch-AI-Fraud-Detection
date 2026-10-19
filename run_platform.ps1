# ==============================================================================
# FraudWatch Platform Runner Script (PowerShell)
# ==============================================================================

param (
    [string]$Action = "test"
)

$VENV_PYTHON = ".\.venv\Scripts\python.exe"

switch ($Action.ToLower()) {
    "test" {
        Write-Host "Running comprehensive test suite (65 tests)..." -ForegroundColor Cyan
        & $VENV_PYTHON -m pytest tests/ -v
    }
    "acceptance" {
        Write-Host "Running 10-stage end-to-end acceptance verification..." -ForegroundColor Green
        & $VENV_PYTHON src/e2e_acceptance_test.py
    }
    "kafka-smoke" {
        Write-Host "Running Kafka broker probe & streaming smoke test..." -ForegroundColor Cyan
        & $VENV_PYTHON src/kafka_smoke_test.py
    }
    "api" {
        Write-Host "Starting FastAPI scoring microservice on http://127.0.0.1:8000..." -ForegroundColor Green
        & $VENV_PYTHON -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
    }
    "dashboard" {
        Write-Host "Starting Streamlit FraudWatch Dashboard on http://localhost:8501..." -ForegroundColor Green
        & $VENV_PYTHON -m streamlit run dashboard.py --server.port 8501
    }
    "mlflow" {
        Write-Host "Starting MLflow tracking UI on http://localhost:5000..." -ForegroundColor Cyan
        & $VENV_PYTHON -m mlflow ui --backend-store-uri sqlite:///mlflow.db
    }
    "pipeline" {
        Write-Host "Executing end-to-end training and feature store pipeline..." -ForegroundColor Yellow
        Write-Host "1. Feature Engineering..." -ForegroundColor Cyan
        & $VENV_PYTHON src/04_leakage_free_features.py
        Write-Host "2. XGBoost Training..." -ForegroundColor Cyan
        & $VENV_PYTHON src/05_train_leakage_free_model.py
        Write-Host "3. MLflow Logging..." -ForegroundColor Cyan
        & $VENV_PYTHON src/06_log_mlflow.py
        Write-Host "4. FastAI Tabular Deep Learning..." -ForegroundColor Cyan
        & $VENV_PYTHON src/07_train_fastai_tabular.py
        Write-Host "5. Feast Feature Store..." -ForegroundColor Cyan
        & $VENV_PYTHON src/08_setup_feast.py
        Write-Host "6. PyTorch Geometric GNN..." -ForegroundColor Cyan
        & $VENV_PYTHON src/09_train_pyg_gnn.py
        Write-Host "7. Evidently AI Drift & Governance..." -ForegroundColor Cyan
        & $VENV_PYTHON src/10_drift_and_retrain_trigger.py
        Write-Host "Pipeline execution complete!" -ForegroundColor Green
    }
    "stream" {
        Write-Host "Replaying 100 transactions to Kafka..." -ForegroundColor Cyan
        & $VENV_PYTHON streaming/producer.py --max-events 100 --delay-ms 20
    }
    "worker" {
        Write-Host "Starting Kafka scoring worker..." -ForegroundColor Cyan
        & $VENV_PYTHON streaming/consumer_worker.py
    }
    "frontend" {
        Write-Host "Starting modern FraudWatch HTML/CSS/JS frontend on http://localhost:3000..." -ForegroundColor Green
        & $VENV_PYTHON -m http.server 3000 --directory frontend
    }
    default {
        Write-Host "Usage: .\run_platform.ps1 -Action [frontend|test|acceptance|kafka-smoke|api|dashboard|mlflow|pipeline|stream|worker]" -ForegroundColor Yellow
    }
}
