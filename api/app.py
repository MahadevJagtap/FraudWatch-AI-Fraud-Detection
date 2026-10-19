"""
FastAPI application entrypoint for the PaySim Real-Time Fraud Scoring Service.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from api.config import settings
from api.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    HealthResponse,
    PredictionResult,
    TransactionInput,
)
from api.service import ModelLoadError, scoring_service

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: load the XGBoost model and metadata on startup."""
    logger.info("Initializing PaySim Fraud Scoring API...")
    try:
        scoring_service.load()
        logger.info("Fraud detection model loaded and ready to serve traffic.")
    except Exception as e:
        logger.error(f"FATAL: Model startup loading failed: {e}", exc_info=True)
    yield
    logger.info("Shutting down PaySim Fraud Scoring API.")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: Request timing & structured logging (no sensitive PII logged)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        f"HTTP {request.method} {request.url.path} -> Status {response.status_code} "
        f"[{duration_ms:.2f}ms]"
    )
    return response


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format validation errors clearly for API consumers."""
    error_details = []
    for err in exc.errors():
        loc = " -> ".join(str(item) for item in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        error_details.append({"location": loc, "message": msg, "type": err.get("type")})

    logger.warning(f"Validation error on {request.method} {request.url.path}: {error_details}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": "Validation Error",
            "message": "Invalid transaction input payload.",
            "details": error_details,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Bad Request", "message": str(exc)},
    )


@app.exception_handler(ModelLoadError)
async def model_error_handler(request: Request, exc: ModelLoadError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "Service Unavailable", "message": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/",
    summary="Service Index",
    tags=["General"],
)
async def root() -> Dict[str, Any]:
    """Return service metadata and links to interactive documentation."""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "status": "online",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "health": "GET /health",
            "score_single": "POST /predict",
            "score_batch": "POST /predict/batch",
        },
        "notice": (
            "PaySim fraud detection service. Supports 'historical' benchmark mode and "
            "'live_authorization' mode with configurable decision thresholds."
        ),
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Model Readiness Check",
    tags=["Monitoring"],
)
async def health():
    """Verify that the service is running and the XGBoost model is loaded."""
    if not scoring_service.is_loaded or scoring_service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or initialization failed.",
        )

    meta = scoring_service.metadata
    metrics = meta.get("metrics", {})

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_path=scoring_service.model_path,
        model_type=meta.get("model_type", "XGBClassifier"),
        expected_features=scoring_service.features,
        feature_count=len(scoring_service.features),
        training_auprc=metrics.get("auprc"),
        training_recall=metrics.get("recall_fraud"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Score a Single Transaction",
    tags=["Scoring"],
)
async def predict(
    transaction: TransactionInput,
    block_threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Override threshold for BLOCK action"
    ),
    review_threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Override threshold for REVIEW action"
    ),
) -> PredictionResult:
    """
    Score a single financial transaction for fraud risk.
    
    - **historical mode**: Requires full post-settlement balance records.
    - **live_authorization mode**: Accepts in-flight authorization requests without settled balances.
    """
    if not scoring_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring service is not ready. Model is not loaded.",
        )

    try:
        res = scoring_service.score_transaction(
            transaction,
            block_threshold=block_threshold,
            review_threshold=review_threshold,
        )
        if res.is_fraud or res.risk_level in ["HIGH", "MEDIUM"]:
            try:
                from api.case_store import insert_or_update_alert
                insert_or_update_alert({
                    "transaction_id": res.transaction_id or f"tx_{int(time.time()*1000)}",
                    "timestamp": time.time(),
                    "step": transaction.step,
                    "type": transaction.type.value if hasattr(transaction.type, "value") else str(transaction.type),
                    "amount": transaction.amount,
                    "nameOrig": transaction.nameOrig,
                    "nameDest": transaction.nameDest,
                    "orig_drain_ratio": round(transaction.amount / (transaction.oldbalanceOrg + 1.0), 4),
                    "fraud_probability": res.fraud_probability,
                    "risk_level": res.risk_level.value if hasattr(res.risk_level, "value") else str(res.risk_level),
                    "decision_threshold": res.block_threshold,
                })
            except Exception as exc:
                logger.warning(f"Could not persist alert case: {exc}")
        return res
    except Exception as e:
        logger.error(f"Inference error on single transaction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {e}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score a Batch of Transactions",
    tags=["Scoring"],
)
async def predict_batch(
    payload: BatchTransactionRequest,
) -> BatchPredictionResponse:
    """
    Score multiple transactions in a single vectorized batch request.
    Maximum batch size is capped by server settings (default 500 items).
    """
    if not scoring_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring service is not ready. Model is not loaded.",
        )

    if len(payload.transactions) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(payload.transactions)} exceeds maximum limit of {settings.max_batch_size}.",
        )

    try:
        return scoring_service.score_batch(
            payload.transactions,
            block_threshold=payload.block_threshold,
            review_threshold=payload.review_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error on batch request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {e}",
        )


@app.post(
    "/predict/preauth",
    response_model=PredictionResult,
    summary="Score a Pre-Authorization Transaction (Feast-Enriched)",
    tags=["Scoring"],
)
async def predict_preauth(
    transaction: TransactionInput,
    block_threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Override threshold for BLOCK action"
    ),
    review_threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Override threshold for REVIEW action"
    ),
) -> PredictionResult:
    """
    Score an in-flight pre-authorization transaction using the leakage-free champion XGBoost model (16 features),
    enriched with online account features from Feast SQLite online store.
    """
    if not scoring_service.preauth_is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pre-authorization scoring model is not loaded.",
        )

    try:
        res = scoring_service.score_preauth_transaction(
            transaction,
            block_threshold=block_threshold,
            review_threshold=review_threshold,
        )
        if res.is_fraud or res.risk_level in ["HIGH", "MEDIUM"]:
            try:
                from api.case_store import insert_or_update_alert
                insert_or_update_alert({
                    "transaction_id": res.transaction_id or f"tx_{int(time.time()*1000)}",
                    "timestamp": time.time(),
                    "step": transaction.step,
                    "type": transaction.type.value if hasattr(transaction.type, "value") else str(transaction.type),
                    "amount": transaction.amount,
                    "nameOrig": transaction.nameOrig,
                    "nameDest": transaction.nameDest,
                    "orig_drain_ratio": round(transaction.amount / (transaction.oldbalanceOrg + 1.0), 4),
                    "fraud_probability": res.fraud_probability,
                    "risk_level": res.risk_level.value if hasattr(res.risk_level, "value") else str(res.risk_level),
                    "decision_threshold": res.block_threshold,
                })
            except Exception as exc:
                logger.warning(f"Could not persist alert case: {exc}")
        return res
    except Exception as e:
        logger.error(f"Inference error on preauth transaction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preauth inference execution failed: {e}",
        )


@app.get(
    "/platform/feature-store/account/{account_id}",
    summary="Query Feast Online Feature Store for Account Profile",
    tags=["Feature Store"],
)
async def get_account_feature_store_profile(account_id: str):
    """Retrieve pre-computed online features for a given account directly from Feast."""
    t0 = time.perf_counter()
    try:
        from feature_store.client import get_online_account_features
        raw = get_online_account_features([account_id])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        orig_bal = raw.get("orig_old_balance", [None])[0]
        dest_bal = raw.get("dest_old_balance", [None])[0]
        drain = raw.get("orig_drain_ratio", [None])[0]
        is_m = raw.get("is_merchant", [None])[0]
        dest_cust = raw.get("dest_is_customer", [None])[0]

        found = (orig_bal is not None) or (dest_bal is not None)
        return {
            "account_id": account_id,
            "found": found,
            "status": "HIT" if found else "MISS",
            "features": {
                "orig_old_balance": orig_bal,
                "dest_old_balance": dest_bal,
                "orig_drain_ratio": drain,
                "is_merchant": is_m,
                "dest_is_customer": dest_cust,
            } if found else None,
            "latency_ms": round(latency_ms, 3),
            "store_provider": "sqlite",
            "feature_view": "account_features",
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.warning(f"Feast query error for account {account_id}: {exc}", exc_info=True)
        return {
            "account_id": account_id,
            "found": False,
            "status": "ERROR",
            "error": str(exc),
            "latency_ms": round(latency_ms, 3),
        }


@app.get(
    "/platform/status",
    summary="Platform Component Health & Technology Stack Status",
    tags=["Governance & Monitoring"],
)
async def platform_status():
    """Return status across all 8 required technologies in the platform."""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "status": "OPERATIONAL",
        "technologies": {
            "fastapi": {"status": "ONLINE"},
            "xgboost": {
                "benchmark_model_loaded": scoring_service.is_loaded,
                "pre_auth_model_exists": os.path.isfile(os.path.join(project_root, "models", "leakage_free_xgb_model.json")),
            },
            "fastai": {
                "model_exists": os.path.isfile(os.path.join(project_root, "models", "fastai_tabular_model.pkl")),
                "evaluation_report_exists": os.path.isfile(os.path.join(project_root, "reports", "fastai_evaluation_report.json")),
            },
            "pytorch_geometric": {
                "weights_exist": os.path.isfile(os.path.join(project_root, "models", "pyg_fraud_gnn_model.pt")),
                "evaluation_report_exists": os.path.isfile(os.path.join(project_root, "reports", "pyg_gnn_evaluation_report.json")),
            },
            "feast": {
                "online_store_exists": os.path.isfile(os.path.join(project_root, "feature_store", "data", "online_store.db")),
                "parquet_source_exists": os.path.isfile(os.path.join(project_root, "data", "processed", "account_features_feast.parquet")),
            },
            "mlflow": {
                "database_exists": os.path.isfile(os.path.join(project_root, "mlflow.db")),
            },
            "kafka_streaming": {
                "configured_bootstrap": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                "topics": ["fraudwatch.transactions.raw", "fraudwatch.alerts.fraud"],
            },
            "evidently_ai": {
                "html_report_exists": os.path.isfile(os.path.join(project_root, "reports", "evidently_drift_report.html")),
                "json_report_exists": os.path.isfile(os.path.join(project_root, "reports", "evidently_drift_report.json")),
                "promotion_audit_exists": os.path.isfile(os.path.join(project_root, "reports", "model_promotion_audit.json")),
            },
            "kubernetes": {
                "manifests_dir_exists": os.path.isdir(os.path.join(project_root, "k8s")),
            }
        }
    }


@app.get(
    "/platform/alerts",
    summary="Get Recent Kafka Fraud Alerts",
    tags=["Governance & Monitoring"],
)
async def get_recent_alerts(limit: int = 50):
    """Return recent streaming fraud alerts emitted by the Kafka scoring worker."""
    import os
    import json
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alerts_file = os.path.join(project_root, "reports", "kafka_fraud_alerts.jsonl")
    if not os.path.isfile(alerts_file):
        return {"alerts": [], "count": 0}

    alerts = []
    with open(alerts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    alerts.append(json.loads(line))
                except Exception:
                    pass
    return {"alerts": alerts[-limit:], "count": len(alerts)}


# ---------------------------------------------------------------------------
# Case Management & Analyst Feedback Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/platform/cases",
    summary="List Fraud Cases with Workflow Filtering",
    tags=["Case Management"],
)
async def list_cases(
    status: Optional[str] = Query("ALL", description="Case status filter (OPEN, INVESTIGATING, CONFIRMED_FRAUD, FALSE_POSITIVE)"),
    risk_level: Optional[str] = Query("ALL", description="Risk level filter (HIGH, MEDIUM, LOW)"),
    min_score: Optional[float] = Query(None, description="Minimum fraud score threshold"),
    search: Optional[str] = Query(None, description="Search term for tx_id, nameOrig, or nameDest"),
    limit: int = Query(50, ge=1, le=500),
):
    from api.case_store import get_cases
    cases = get_cases(
        status=status,
        risk_level=risk_level,
        min_score=min_score,
        search=search,
        limit=limit,
    )
    return {"cases": cases, "count": len(cases)}


@app.get(
    "/platform/cases/summary/metrics",
    summary="Get Case Management Pipeline Metrics",
    tags=["Case Management"],
)
async def get_case_pipeline_metrics():
    from api.case_store import get_case_metrics
    return get_case_metrics()


@app.get(
    "/platform/cases/{transaction_id}",
    summary="Get Single Fraud Case Detail & Audit History",
    tags=["Case Management"],
)
async def get_case_detail(transaction_id: str):
    from api.case_store import get_case_by_id
    case = get_case_by_id(transaction_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with transaction_id {transaction_id!r} not found.",
        )
    return case


@app.post(
    "/platform/cases/{transaction_id}/review",
    summary="Submit Analyst Feedback and Update Case Status",
    tags=["Case Management"],
)
async def submit_case_review(
    transaction_id: str,
    payload: Dict[str, Any],
):
    from api.case_store import update_case_review, CASE_STATUSES

    new_status = str(payload.get("status", "INVESTIGATING")).upper()
    disposition = str(payload.get("disposition", "PENDING_REVIEW"))
    notes = str(payload.get("notes", ""))
    analyst_id = str(payload.get("analyst_id", "analyst_1"))

    if new_status not in CASE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid status {new_status!r}. Must be one of {CASE_STATUSES}",
        )

    success = update_case_review(
        transaction_id=transaction_id,
        new_status=new_status,
        disposition=disposition,
        notes=notes,
        analyst_id=analyst_id,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with transaction_id {transaction_id!r} not found.",
        )

    return {
        "status": "success",
        "transaction_id": transaction_id,
        "new_status": new_status,
        "disposition": disposition,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Frontend Static Files Mount
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_frontend_dir = os.path.join(_project_root, "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="frontend")



