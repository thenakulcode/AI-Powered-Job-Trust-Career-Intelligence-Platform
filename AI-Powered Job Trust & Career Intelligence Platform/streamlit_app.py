"""Streamlit app for fake job detection.

The app loads the saved XGBoost model and TF-IDF vectorizer produced by
`fake_job_detection.py`, accepts a pasted job description, and returns a
human-friendly risk assessment with clear explanations.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

import joblib
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_fake_job_detector.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
FRAUD_THRESHOLD = 0.35

RISK_KEYWORDS = {
    "urgent": ["urgent", "immediately", "asap", "quickly"],
    "money": ["wire", "bitcoin", "cash", "fee", "payment", "deposit", "transfer"],
    "vague": ["no experience", "easy money", "work from home", "guaranteed", "part time"],
    "contact": ["gmail", "yahoo", "hotmail", "telegram", "whatsapp", "skype"],
    "pressure": ["limited time", "act now", "fast hiring", "openings", "multiple positions"],
    "personal_info": ["ssn", "passport", "bank account", "id card", "credit card"],
}

RISK_LEVELS = {
    "High": "High risk of being fraudulent",
    "Medium": "Medium risk - needs review",
    "Low": "Low risk - likely legitimate",
}

_STOP_WORDS: set[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🛡️",
    layout="wide",
)


def ensure_nltk_resources() -> None:
    """Download the minimal NLTK resources required for text cleaning."""

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


@st.cache_resource
def load_artifacts():
    """Load the saved model and vectorizer once per session."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_PATH}. Run fake_job_detection.py first."
        )
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Saved vectorizer not found at {VECTORIZER_PATH}. Run fake_job_detection.py first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def clean_text(text: str) -> str:
    """Match the preprocessing used during training."""

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


def get_risk_factors(raw_text: str, cleaned_text: str, probability: float) -> list[str]:
    """Generate explainable reasons for the risk score."""

    factors: list[str] = []
    raw_lower = raw_text.lower()

    for label, keywords in RISK_KEYWORDS.items():
        matches = [keyword for keyword in keywords if keyword in raw_lower]
        if matches:
            factors.append(f"{label.replace('_', ' ').title()} language detected: {', '.join(matches[:3])}")

    cleaned_tokens = cleaned_text.split()
    token_counts = Counter(cleaned_tokens)

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


def risk_level_from_probability(probability: float) -> str:
    """Map fraudulent probability to a simple risk band."""

    if probability >= 0.75:
        return "High"
    if probability >= FRAUD_THRESHOLD:
        return "Medium"
    return "Low"


def build_css() -> str:
    """Return custom CSS for a polished modern layout."""

    return """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #0b1120 0%, #111827 45%, #f8fafc 45%, #f8fafc 100%);
            color: #0f172a;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        .hero {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.92));
            color: white;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 2rem 2rem 1.5rem 2rem;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.22);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            font-size: 2.3rem;
            margin-bottom: 0.35rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
            margin-bottom: 0;
        }
        .metric-card, .panel-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148,163,184,0.24);
            border-radius: 20px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            color: #0f172a;
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }
        .metric-subtext {
            color: #475569;
            margin-top: 0.35rem;
            font-size: 0.92rem;
        }
        .risk-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.55rem 0.9rem;
            font-weight: 600;
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }
        .risk-high { background: #fef2f2; color: #b91c1c; }
        .risk-medium { background: #fffbeb; color: #b45309; }
        .risk-low { background: #ecfdf5; color: #047857; }
        .section-title {
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
            color: #0f172a;
        }
        .explain-list {
            margin: 0;
            padding-left: 1.15rem;
            color: #0f172a !important;
        }
        .explain-list li {
            margin-bottom: 0.6rem;
            color: #0f172a !important;
            line-height: 1.45;
        }
        .explain-list li::marker {
            color: #64748b;
        }
        .explain-item {
            display: block;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            color: #0f172a !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
        }
        .result-summary {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(51, 65, 85, 0.96));
            color: #f8fafc;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-top: 0.75rem;
        }
        .result-summary strong {
            color: #ffffff;
        }
        .result-summary .muted {
            color: rgba(248, 250, 252, 0.78);
        }
        .footer-note {
            color: #64748b;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 1.25rem;
        }
        div[data-testid="stTextArea"] textarea {
            border-radius: 16px !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            padding: 1rem !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.7rem 1.3rem;
            font-weight: 700;
            box-shadow: 0 10px 25px rgba(15,23,42,0.2);
        }
        .stButton > button:hover {
            border: none;
            transform: translateY(-1px);
        }
        .stAlert {
            border-radius: 16px;
        }
    </style>
    """


def render_metric_card(label: str, value: str, subtext: str) -> str:
    """Render a reusable metric card."""

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """


def main() -> None:
    st.markdown(build_css(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>AI-Powered Fake Job Detection</h1>
            <p>Paste a job description to assess whether it looks legitimate or potentially fraudulent using the trained XGBoost model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, vectorizer = load_artifacts()

    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Job Description Input</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "Paste the job description here",
            height=340,
            placeholder="Paste the full job post, including title, responsibilities, and company details...",
            label_visibility="collapsed",
        )
        analyze = st.button("Analyze Job Post", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
        st.write(
            "The app cleans the pasted text with the same preprocessing used during training, converts it to TF-IDF features, and scores it with the saved XGBoost model."
        )
        st.info("Confidence is the model's estimated probability for the predicted class.")
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not job_description.strip():
            st.warning("Please paste a job description before analyzing.")
            return

        cleaned_text = clean_text(job_description)
        transformed = vectorizer.transform([cleaned_text])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed)[0]
            confidence = float(np.max(probabilities))
            fraudulent_probability = float(probabilities[1])
        else:
            decision = model.decision_function(transformed)
            confidence = float(1.0 / (1.0 + np.exp(-decision[0])))
            fraudulent_probability = confidence

        prediction = 1 if fraudulent_probability >= FRAUD_THRESHOLD else 0
        risk_level = risk_level_from_probability(fraudulent_probability)
        factors = get_risk_factors(job_description, cleaned_text, fraudulent_probability)

        prediction_label = "Fraudulent Job Post" if prediction == 1 else "Legitimate Job Post"
        prediction_color = "#b91c1c" if prediction == 1 else "#047857"

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.markdown(
                render_metric_card("Prediction", prediction_label, "Model classification of the pasted job description"),
                unsafe_allow_html=True,
            )
        with metric_col2:
            st.markdown(
                render_metric_card("Confidence Score", f"{confidence * 100:.2f}%", "Highest class probability from the model"),
                unsafe_allow_html=True,
            )
        with metric_col3:
            risk_class = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="risk-pill {risk_class}">{RISK_LEVELS[risk_level]}</div>
                    <div class="metric-subtext">Based on the model's fraudulent probability: {fraudulent_probability * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="result-summary">
                <div><strong>Summary</strong></div>
                <div class="muted">Model confidence: {confidence * 100:.2f}%</div>
                <div class="muted">Fraud probability: {fraudulent_probability * 100:.2f}%</div>
                <div class="muted">Risk level: {RISK_LEVELS[risk_level]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        details_left, details_right = st.columns([0.62, 0.38], gap="large")

        with details_left:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Risk Factors</div>', unsafe_allow_html=True)
            st.markdown(
                "<ul class='explain-list'>"
                + "".join(f"<li><span class='explain-item'>{factor}</span></li>" for factor in factors)
                + "</ul>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with details_right:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Model Notes</div>', unsafe_allow_html=True)
            st.write("Saved model:", MODEL_PATH.name)
            st.write("Saved vectorizer:", VECTORIZER_PATH.name)
            st.write("Preprocessing:")
            st.code("lowercase -> remove urls/html -> remove non-letters -> stop words -> lemmatize", language="text")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='footer-note'>Prediction score is derived from the trained XGBoost model output. <span style='color:{prediction_color}; font-weight:700;'>{prediction_label}</span></div>",
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            "<div class='footer-note'>Enter a job post and click Analyze Job Post to generate a trust assessment.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()