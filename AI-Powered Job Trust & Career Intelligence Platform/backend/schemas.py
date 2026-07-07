"""Pydantic request and response schemas for the fake job detection API."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


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
