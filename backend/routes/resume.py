"""Resume ATS analysis routes."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.services.ats_service import ATSService
from backend.services.resume_parser import extract_resume_text

router = APIRouter(tags=["resume"])
ats_service = ATSService()


@router.post(
    "/resume-ats",
    summary="Analyze a resume against a job description",
    response_model=dict,
)
def analyze_resume(job_description: str = Form(...), file: UploadFile = File(...)) -> dict[str, object]:
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="A job description is required.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="A resume file is required.")

    filename = (file.filename or "").lower()
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX resumes are supported.")

    try:
        file_bytes = file.file.read()
        resume_text = extract_resume_text(file.filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard.
        raise HTTPException(status_code=422, detail="Unable to parse the uploaded resume.") from exc
    finally:
        file.file.close()

    return ats_service.assess_resume(resume_text, job_description)
