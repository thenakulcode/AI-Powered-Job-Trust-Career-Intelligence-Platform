"""Model loading and inference service for fake job detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from backend.config import FRAUD_THRESHOLD, MODEL_PATH, VECTORIZER_PATH
from backend.preprocessing import clean_text, extract_risk_factors


@dataclass(slots=True)
class PredictionResult:
    """Container for model inference output."""

    prediction: str
    confidence: float
    risk_score: float
    risk_factors: list[str]


class ModelLoadError(RuntimeError):
    """Raised when the saved artifacts cannot be loaded."""


class PredictionService:
    """Encapsulates artifact loading and scoring logic."""

    def __init__(self, model_path: Path = MODEL_PATH, vectorizer_path: Path = VECTORIZER_PATH) -> None:
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None

    def load(self) -> None:
        """Load the saved XGBoost model and TF-IDF vectorizer."""

        if not self.model_path.exists():
            raise ModelLoadError(f"Saved model not found at {self.model_path}")
        if not self.vectorizer_path.exists():
            raise ModelLoadError(f"Saved vectorizer not found at {self.vectorizer_path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict(self, job_description: str) -> PredictionResult:
        """Score a job description using the trained pipeline artifacts."""

        if self.model is None or self.vectorizer is None:
            self.load()

        cleaned_text = clean_text(job_description)
        features = self.vectorizer.transform([cleaned_text])

        prediction_value = int(self.model.predict(features)[0])
        prediction_probabilities = self.model.predict_proba(features)[0]
        fraudulent_probability = float(prediction_probabilities[1])
        confidence = float(max(prediction_probabilities))
        risk_score = round(fraudulent_probability * 100, 2)

        prediction_label = (
            "Fraudulent Job Post"
            if fraudulent_probability >= FRAUD_THRESHOLD
            else "Legitimate Job Post"
        )
        risk_factors = extract_risk_factors(job_description, cleaned_text, fraudulent_probability)

        return PredictionResult(
            prediction=prediction_label,
            confidence=round(confidence, 4),
            risk_score=risk_score,
            risk_factors=risk_factors,
        )
