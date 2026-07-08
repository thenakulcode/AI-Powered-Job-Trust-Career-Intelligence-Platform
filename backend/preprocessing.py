"""Text preprocessing utilities shared by the backend service."""

from __future__ import annotations

import html
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RISK_KEYWORDS = {
    "urgent": ["urgent", "immediately", "asap", "quickly"],
    "money": ["wire", "bitcoin", "cash", "fee", "payment", "deposit", "transfer"],
    "vague": ["no experience", "easy money", "work from home", "guaranteed", "part time"],
    "contact": ["gmail", "yahoo", "hotmail", "telegram", "whatsapp", "skype"],
    "pressure": ["limited time", "act now", "fast hiring", "openings", "multiple positions"],
    "personal_info": ["ssn", "passport", "bank account", "id card", "credit card"],
}

_STOP_WORDS: set[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora used by the preprocessing pipeline."""

    resources = ["stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def get_stop_words() -> set[str]:
    """Return the cached English stop-word set."""

    global _STOP_WORDS
    if _STOP_WORDS is None:
        ensure_nltk_resources()
        _STOP_WORDS = set(stopwords.words("english"))
    return _STOP_WORDS


def get_lemmatizer() -> WordNetLemmatizer:
    """Return the cached WordNet lemmatizer."""

    global _LEMMATIZER
    if _LEMMATIZER is None:
        ensure_nltk_resources()
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def clean_text(text: str) -> str:
    """Apply the same normalization used during training."""

    if not text:
        return ""

    stop_words = get_stop_words()
    lemmatizer = get_lemmatizer()

    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [lemmatizer.lemmatize(token) for token in text.split() if token not in stop_words]
    return " ".join(tokens)


def extract_risk_factors(raw_text: str, cleaned_text: str, probability: float) -> list[str]:
    """Generate simple explainable signals for the API response."""

    factors: list[str] = []
    raw_lower = raw_text.lower()

    for label, keywords in RISK_KEYWORDS.items():
        matches = [keyword for keyword in keywords if keyword in raw_lower]
        if matches:
            factors.append(f"{label.replace('_', ' ').title()} language detected: {', '.join(matches[:3])}")

    token_counts = Counter(cleaned_text.split())
    if any(token_counts[word] > 2 for word in ["urgent", "quick", "easy", "guaranteed"]):
        factors.append("Repeated urgency or unrealistic promise language")

    if len(cleaned_text) < 120:
        factors.append("Very short job description, which reduces trust signals")

    if probability >= 0.85:
        factors.append("Model confidence is very high")
    elif probability >= 0.65:
        factors.append("Model confidence is moderately high")
    else:
        factors.append("Model confidence is moderate")

    return factors or ["No strong red-flag language detected"]
