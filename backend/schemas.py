"""Pydantic request and response schemas for the fake job detection API."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, HttpUrl


class JobPredictionRequest(BaseModel):
    """Request payload for scoring a job description."""

    job_description: str = Field(
        ...,
        min_length=1,
        description="Job description text to analyze for fake-job signals.",
        examples=["We are hiring immediately. Send your bank details to receive payment."],
    )


class JobPredictionResponse(BaseModel):
    """Response payload returned by the prediction endpoint."""

    prediction: Literal["Fraudulent Job Post", "Legitimate Job Post"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_factors: List[str]


class JobExtractionRequest(BaseModel):
    """Request payload for extracting a job description from a posting URL."""

    url: HttpUrl = Field(
        ...,
        description="Job posting URL to extract a description from.",
        examples=["https://www.linkedin.com/jobs/view/1234567890"],
    )


class JobExtractionResponse(BaseModel):
    """Response payload returned by the URL extraction endpoint.

    `success` is False (with `error` populated) for expected failure modes —
    blocked/rate-limited sites, 404s, CAPTCHAs, or pages with no identifiable
    job content — so the frontend can show a friendly message instead of
    treating every miss as a hard error.
    """

    success: bool
    description: str | None = None
    title: str | None = None
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None
    experience: str | None = None
    salary: str | None = None
    error: str | None = None