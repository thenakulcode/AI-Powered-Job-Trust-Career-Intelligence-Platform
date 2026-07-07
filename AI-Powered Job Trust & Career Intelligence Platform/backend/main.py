"""FastAPI application entry point for fake job detection."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from backend.schemas import JobPredictionRequest, JobPredictionResponse
from backend.service import ModelLoadError, PredictionService

app = FastAPI(
    title="Fake Job Detection API",
    description="Predict whether a job post is likely fraudulent using the trained XGBoost model.",
    version="1.0.0",
)

prediction_service = PredictionService()


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
