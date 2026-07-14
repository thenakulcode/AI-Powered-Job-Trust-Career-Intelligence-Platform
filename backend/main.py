"""FastAPI application entry point for fake job detection."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import File, FastAPI, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.extractor import router as extractor_router
from backend.schemas import JobPredictionRequest, JobPredictionResponse
from backend.service import ModelLoadError, PredictionService
from backend.services.ats_service import ATSService
from backend.services.resume_parser import extract_resume_text

logger = logging.getLogger(__name__)

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
ats_service = ATSService()
app.include_router(extractor_router)


@app.post("/resume-ats", summary="Analyze a resume against a job description")
def analyze_resume(job_description: str | None = Form(None), file: UploadFile | None = File(None)) -> dict[str, object]:
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="A job description is required.")

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="A resume file is required.")

    filename = (file.filename or "").lower()
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX resumes are supported.")

    logger.info(
        "Resume upload received: filename=%s content_type=%s",
        file.filename,
        file.content_type or "unknown",
    )

    try:
        file_bytes = file.file.read()
        logger.info("Resume bytes read: filename=%s size=%s", file.filename, len(file_bytes))
        resume_text = extract_resume_text(file.filename, file_bytes)
        logger.info("Resume text extracted successfully: filename=%s chars=%s", file.filename, len(resume_text))
    except ValueError as exc:
        logger.warning("Resume upload rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard.
        logger.exception("Failed to parse uploaded resume: %s", exc)
        raise HTTPException(status_code=422, detail="Unable to parse the uploaded resume.") from exc
    finally:
        if file is not None:
            file.file.close()

    logger.info("Running ATS assessment for: %s", file.filename)
    result = ats_service.assess_resume(resume_text, job_description)
    logger.info("ATS assessment completed for: %s", file.filename)
    return result

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
