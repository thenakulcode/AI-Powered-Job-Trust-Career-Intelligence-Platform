from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.services.scraper import ScrapingError, extract_job_post


router = APIRouter(tags=["extraction"])


class JobExtractionRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Public job posting URL to extract")
    timeout: int | None = Field(default=15, ge=5, le=45, description="Request timeout in seconds")


class JobExtractionResponse(BaseModel):
    success: bool
    title: str
    company: str
    location: str
    description: str


@router.post("/extract-job", response_model=JobExtractionResponse, summary="Extract job details from a public job posting URL")
def extract_job(payload: JobExtractionRequest) -> JobExtractionResponse:
    try:
        data = extract_job_post(payload.url, timeout=payload.timeout or 15)
    except ScrapingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive API guard.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract job details from the provided URL.",
        ) from exc

    return JobExtractionResponse(
        success=True,
        title=data.get("title", ""),
        company=data.get("company", ""),
        location=data.get("location", ""),
        description=data.get("description", ""),
    )
