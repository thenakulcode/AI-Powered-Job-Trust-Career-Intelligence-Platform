"""FastAPI application entry point for fake job detection."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas import JobPredictionRequest, JobPredictionResponse
from backend.service import ModelLoadError, PredictionService

app = FastAPI(
    title="Fake Job Detection API",
    description="Predict whether a job post is likely fraudulent using the trained XGBoost model.",
    version="1.0.0",
)

# Allow the bundled frontend (and any other local client) to call the API.
# This does not change or remove any existing route.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_service = PredictionService()

# ---------------------------------------------------------------------------
# Frontend hosting (new — does not touch the existing API routes below).
# The dashboard lives in ../frontend/{templates,static} relative to this file.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the Sentinel.AI dashboard's single-page frontend."""

    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found.")
    return FileResponse(index_path)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Lightweight health endpoint."""

    return {"status": "ok"}


@app.post(
    "/predict-job",
    response_model=JobPredictionResponse,
    tags=["prediction"],
    summary="Predict whether a job post is fraudulent",
)
def predict_job(payload: JobPredictionRequest) -> JobPredictionResponse:
    """Predict fraud risk for the submitted job description."""

    try:
        result = prediction_service.predict(payload.job_description)
        return JobPredictionResponse(
            prediction=result.prediction,
            confidence=result.confidence,
            risk_score=result.risk_score,
            risk_factors=result.risk_factors,
        )
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive API guard.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze the job description.",
        ) from exc
