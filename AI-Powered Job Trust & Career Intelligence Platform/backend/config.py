"""Application configuration for the fake job detection backend."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_fake_job_detector.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
FRAUD_THRESHOLD = 0.35
