"""Fake Job Detection pipeline for the AI-Powered Job Trust & Career Intelligence Platform.

This script performs the full machine-learning workflow end to end:
1. Load the Kaggle fake job posting dataset.
2. Run exploratory data analysis.
3. Handle missing values.
4. Clean and normalize the text.
5. Remove stop words and lemmatize.
6. Vectorize the text with TF-IDF.
7. Split into train and test sets.
8. Train Logistic Regression, Random Forest, and XGBoost models.
9. Compare the models with common classification metrics.
10. Display and save a confusion matrix for the best model.
11. Persist the trained model and fitted vectorizer with joblib.

The code is written to be readable and executable as a standalone script.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - handled at runtime if missing.
    raise ImportError(
        "xgboost is required for this script. Install it with `pip install xgboost`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "fake_job_postings.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_PATH = OUTPUT_DIR / "best_fake_job_detector.joblib"
VECTORIZER_PATH = OUTPUT_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = OUTPUT_DIR / "roc_curve.png"
PRECISION_RECALL_CURVE_PATH = OUTPUT_DIR / "precision_recall_curve.png"
THRESHOLD_RECALL_PATH = OUTPUT_DIR / "threshold_vs_recall.png"
THRESHOLD_PRECISION_PATH = OUTPUT_DIR / "threshold_vs_precision.png"

THRESHOLD_VALUES = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

RANDOM_STATE = 42

TrainedModel = LogisticRegression | RandomForestClassifier | XGBClassifier


@dataclass(slots=True)
class PredictionResult:
    """Container for reusable real-world prediction output."""

    prediction: str
    confidence: float
    probability: float

_STOP_WORDS: set[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


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


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the CSV dataset into a pandas DataFrame."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place the Kaggle CSV in the workspace root or pass --data-path."
        )
    return pd.read_csv(data_path)


def load_artifacts() -> Tuple[TrainedModel, TfidfVectorizer]:
    """Load the persisted model and vectorizer from disk."""

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


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using text-friendly and numeric-friendly defaults."""

    cleaned = df.copy()
    for column in cleaned.columns:
        if column == "fraudulent":
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            cleaned[column] = cleaned[column].fillna("")
    return cleaned


def normalize_text(text: str) -> str:
    """Lowercase, remove noise, drop stop words, and lemmatize tokens."""

    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    stop_words = get_stop_words()
    lemmatizer = get_lemmatizer()

    text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [lemmatizer.lemmatize(token) for token in text.split() if token not in stop_words]
    return " ".join(tokens)


def build_text_corpus(df: pd.DataFrame) -> pd.Series:
    """Combine the relevant text fields into a single corpus column."""

    available_columns = [column for column in TEXT_COLUMNS if column in df.columns]
    if not available_columns:
        raise ValueError("None of the expected text columns were found in the dataset.")

    combined = df[available_columns].astype(str).agg(" ".join, axis=1)
    return combined.apply(normalize_text)


def calculate_class_imbalance_ratio(y_train: pd.Series) -> Tuple[int, int, float]:
    """Return the negative count, positive count, and imbalance ratio."""

    class_counts = y_train.value_counts().sort_index()
    negative_count = int(class_counts.get(0, 0))
    positive_count = int(class_counts.get(1, 0))

    if positive_count == 0:
        raise ValueError("The training target contains no fraudulent examples.")

    imbalance_ratio = negative_count / positive_count
    return negative_count, positive_count, imbalance_ratio


def apply_smote_to_training_data(
    x_train,
    y_train: pd.Series,
) -> Tuple[Any, pd.Series]:
    """Apply SMOTE only to the training split."""

    _, positive_count, _ = calculate_class_imbalance_ratio(y_train)
    if positive_count < 2:
        raise ValueError("SMOTE requires at least two fraudulent samples in the training split.")

    k_neighbors = min(5, positive_count - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    x_resampled, y_resampled = cast(Tuple[Any, Any], smote.fit_resample(x_train, y_train))
    return x_resampled, pd.Series(y_resampled)


def build_models(scale_pos_weight: float) -> Dict[str, TrainedModel]:
    """Build the model suite with the requested XGBoost class weight."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def get_positive_class_probabilities(model: TrainedModel, features) -> np.ndarray:
    """Return fraud probabilities for binary models."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    decision_function = getattr(model, "decision_function", None)
    if callable(decision_function):
        scores = np.asarray(decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("The model does not expose probability or decision scores.")


def evaluate_threshold_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: Tuple[float, ...] = THRESHOLD_VALUES,
) -> pd.DataFrame:
    """Evaluate the selected thresholds against the held-out test set."""

    rows: list[dict[str, float]] = []
    print("\n=== Threshold Sweep ===")
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        row = {
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1": f1_score(y_true, predictions, zero_division=0),
        }
        rows.append(row)
        print(
            f"Threshold {threshold:.2f} -> Accuracy={row['accuracy']:.4f} | "
            f"Precision={row['precision']:.4f} | Recall={row['recall']:.4f} | F1={row['f1']:.4f}"
        )

    return pd.DataFrame(rows)


def recommend_threshold(threshold_frame: pd.DataFrame) -> Tuple[float, pd.Series]:
    """Pick the threshold with the strongest F1, then recall, on the test set."""

    ranked = threshold_frame.sort_values(by=["f1", "recall", "precision"], ascending=False)
    best_row = ranked.iloc[0]
    return float(best_row["threshold"]), best_row


def plot_threshold_tradeoffs(threshold_frame: pd.DataFrame) -> None:
    """Save threshold-versus-precision and threshold-versus-recall plots."""

    plt.figure(figsize=(7, 5))
    plt.plot(threshold_frame["threshold"], threshold_frame["recall"], marker="o", color="#0f766e")
    plt.title("Threshold vs Recall")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Recall")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(THRESHOLD_RECALL_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(threshold_frame["threshold"], threshold_frame["precision"], marker="o", color="#b45309")
    plt.title("Threshold vs Precision")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Precision")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(THRESHOLD_PRECISION_PATH, dpi=150)
    plt.close()


def plot_precision_recall_curve(y_true: pd.Series, probabilities: np.ndarray) -> None:
    """Save the precision-recall curve for the selected model."""

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, linewidth=2, color="#7c3aed", label=f"PR AUC = {pr_auc:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PRECISION_RECALL_CURVE_PATH, dpi=150)
    plt.close()


def print_metric_explanations() -> None:
    """Explain the evaluation metrics used in the training report."""

    print("\n=== Metric Guide ===")
    print("Precision: Of the posts predicted as fraudulent, how many were truly fraudulent.")
    print("Recall: Of all fraudulent posts, how many the model successfully found.")
    print("F1-score: Harmonic mean of precision and recall; useful when classes are imbalanced.")
    print("Support: Number of true samples for each class in the evaluated dataset.")
    print(
        "ROC-AUC: Measures how well the model ranks fraudulent posts ahead of genuine ones across thresholds."
    )
    print(
        "Cross-validation: Repeats training on multiple folds to estimate whether performance is stable or noisy."
    )
    print(
        "Overfitting check: Compares training and testing scores to detect memorization versus generalization."
    )


def run_cross_validation(
    model: TrainedModel,
    x_train,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """Run 5-fold cross validation and return fold-level and summary metrics."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    fold_metrics = {
        "accuracy": cv_results["test_accuracy"].tolist(),
        "precision": cv_results["test_precision"].tolist(),
        "recall": cv_results["test_recall"].tolist(),
        "f1": cv_results["test_f1"].tolist(),
    }

    print("5-Fold Cross Validation Scores:")
    for metric_name, scores in fold_metrics.items():
        formatted_scores = ", ".join(f"{score:.4f}" for score in scores)
        print(f"  {metric_name.title()}: {formatted_scores}")

    summary = {
        "mean_accuracy": float(np.mean(cv_results["test_accuracy"])),
        "mean_precision": float(np.mean(cv_results["test_precision"])),
        "mean_recall": float(np.mean(cv_results["test_recall"])),
        "mean_f1": float(np.mean(cv_results["test_f1"])),
        "std_accuracy": float(np.std(cv_results["test_accuracy"])),
        "std_precision": float(np.std(cv_results["test_precision"])),
        "std_recall": float(np.std(cv_results["test_recall"])),
        "std_f1": float(np.std(cv_results["test_f1"])),
    }

    print(
        "Cross-validation means: "
        f"Accuracy={summary['mean_accuracy']:.4f}, "
        f"Precision={summary['mean_precision']:.4f}, "
        f"Recall={summary['mean_recall']:.4f}, "
        f"F1={summary['mean_f1']:.4f}"
    )

    stability_note = "stable" if summary["std_f1"] <= 0.05 else "somewhat variable"
    print(
        f"Cross-validation stability: {stability_note} (F1 std={summary['std_f1']:.4f})."
    )

    return {"fold_metrics": fold_metrics, **summary}


def assess_fit(train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> Tuple[str, str]:
    """Classify the model as underfit, well-fit, or overfit using score gaps."""

    accuracy_gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    f1_gap = train_metrics["f1"] - test_metrics["f1"]

    if test_metrics["accuracy"] < 0.65 and test_metrics["f1"] < 0.65:
        return (
            "Underfitting",
            "Training and testing scores are both weak, so the model is not learning enough signal.",
        )

    if accuracy_gap > 0.10 or f1_gap > 0.10:
        return (
            "Overfitting",
            "Training scores are materially higher than testing scores, which suggests memorization.",
        )

    return (
        "Good Fit",
        "Training and testing scores are close enough to suggest reasonable generalization.",
    )


def print_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Print and return the classification report for the test split."""

    print(f"\nClassification Report - {model_name}")
    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate Job Post", "Fraudulent Job Post"],
        zero_division=0,
    )
    print(report_text)
    report_dict = cast(
        Dict[str, Any],
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate Job Post", "Fraudulent Job Post"],
            zero_division=0,
            output_dict=True,
        ),
    )
    return report_dict


def plot_roc_curves(
    roc_data: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Save a combined ROC curve for all trained models."""

    plt.figure(figsize=(7, 6))
    for model_name, data in roc_data.items():
        plt.plot(data["fpr"], data["tpr"], linewidth=2, label=f"{model_name} (AUC={data['auc']:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.title("ROC Curves for Fake Job Detection Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close()


def predict_job(
    job_description: str,
    model: TrainedModel | None = None,
    vectorizer: TfidfVectorizer | None = None,
    threshold: float = 0.5,
) -> PredictionResult:
    """Predict fraud risk for a single job description.

    If no model artifacts are supplied, the saved best model and vectorizer are loaded.
    """

    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned_text = normalize_text(job_description)
    features = vectorizer.transform([cleaned_text])
    probability = float(get_positive_class_probabilities(model, features)[0])
    confidence = float(max(model.predict_proba(features)[0]))
    predicted_label = "Fraudulent Job Post" if probability >= threshold else "Legitimate Job Post"

    return PredictionResult(
        prediction=predicted_label,
        confidence=round(confidence, 4),
        probability=round(probability, 4),
    )


def run_real_world_prediction_tests(
    model: TrainedModel,
    vectorizer: TfidfVectorizer,
    threshold: float,
) -> None:
    """Evaluate the trained model on manually written job descriptions."""

    examples = [
        (
            "Genuine Software Engineer job",
            "We are hiring a Software Engineer to build scalable APIs in Python and TypeScript. "
            "You will collaborate with product managers, participate in code reviews, write automated tests, "
            "and contribute to system design discussions. Competitive salary, health benefits, and hybrid work are included.",
        ),
        (
            "Genuine Data Analyst job",
            "Join our analytics team as a Data Analyst. You will work with SQL, Excel, Tableau, and Python to analyze business trends, "
            "prepare dashboards, and support decision-making with clear reporting. Prior experience in data visualization and statistics is preferred.",
        ),
        (
            "Genuine Marketing job",
            "We need a Marketing Specialist to manage campaigns, measure performance, create content calendars, and coordinate with design and sales teams. "
            "Experience with SEO, social media, and email marketing is a plus. Full-time role with paid leave and performance bonuses.",
        ),
        (
            "Fake Work From Home job",
            "Earn guaranteed income from home with no experience required. Work only 2 hours per day and make huge weekly payments. "
            "Limited openings, apply immediately, and contact us on Telegram for fast approval. No interview needed.",
        ),
        (
            "Fake Data Entry job",
            "Simple data entry job from home. No interview, no experience, and instant hiring. You will earn high salary every week by typing documents. "
            "Send your details now to get started right away.",
        ),
        (
            "Fake Crypto job",
            "Looking for remote crypto promoters to help us move funds and earn massive returns fast. Guaranteed profits, urgent hiring, and payment in Bitcoin. "
            "Anyone can join today with no background check.",
        ),
        (
            "Fake Registration Fee scam",
            "We selected you for an exclusive job opportunity, but there is a small registration fee required before onboarding. "
            "Pay the deposit today to secure your spot and receive the offer letter immediately.",
        ),
        (
            "Fake Telegram hiring scam",
            "Immediate hiring available for thousands of openings. Please message our recruiter on Telegram for private instructions and fast processing. "
            "No resume required and multiple positions are open now.",
        ),
        (
            "Fake High Salary scam",
            "Join now and earn $8,000 per week for easy part-time work. No skills needed, guaranteed salary, fast approval, and remote work only. "
            "Act now because the offer expires today.",
        ),
        (
            "Fake No Interview scam",
            "We are hiring immediately with no interview required. This is a guaranteed placement, and all candidates are accepted. "
            "Submit your basic information now for instant onboarding and quick payment.",
        ),
    ]

    print("\n=== Real-World Prediction Tests ===")
    print(f"Decision threshold used for these examples: {threshold:.2f}")
    for example_name, description in examples:
        prediction = predict_job(description, model=model, vectorizer=vectorizer, threshold=threshold)
        print(
            f"{example_name}: {prediction.prediction} | "
            f"Confidence={prediction.confidence:.4f} | Probability={prediction.probability:.4f}"
        )


def run_eda(df: pd.DataFrame) -> None:
    """Print and save lightweight EDA outputs for the dataset."""

    OUTPUT_DIR.mkdir(exist_ok=True)
    EDA_DIR.mkdir(exist_ok=True)

    print("\n=== Exploratory Data Analysis ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False).to_string())
    print("\nTarget distribution:")
    print(df["fraudulent"].value_counts(dropna=False).sort_index().to_string())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="fraudulent", data=df, hue="fraudulent", palette="Set2", legend=False)
    plt.title("Fake vs Genuine Job Posts")
    plt.xlabel("Fraudulent Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "target_distribution.png", dpi=150)
    plt.close()

    missing_counts = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_counts.index[:10], y=missing_counts.values[:10], hue=missing_counts.index[:10], palette="viridis", legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Missing-Value Columns")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "missing_values_top10.png", dpi=150)
    plt.close()


def prepare_features(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Create the cleaned text corpus and binary target."""

    if "fraudulent" not in df.columns:
        raise ValueError("The dataset must contain a 'fraudulent' target column.")

    text_corpus = build_text_corpus(df)
    target = df["fraudulent"].astype(int)
    return text_corpus, target


def train_and_evaluate_models(
    x_train,
    x_test,
    y_train,
    y_test,
) -> Tuple[Dict[str, Dict[str, object]], str, TrainedModel, np.ndarray]:
    """Train the requested models and return metrics, best estimator, and its probabilities."""

    metrics: Dict[str, Dict[str, object]] = {}
    best_model_name = ""
    best_model: TrainedModel | None = None
    best_f1 = -1.0
    best_test_probabilities: np.ndarray | None = None
    roc_data: Dict[str, Dict[str, np.ndarray]] = {}

    original_negative_count, original_positive_count, original_ratio = calculate_class_imbalance_ratio(y_train)
    print_metric_explanations()
    print(
        f"Class imbalance ratio (negative/positive): {original_negative_count}/{original_positive_count} = {original_ratio:.4f}"
    )

    for scenario_name, use_smote in (("Before SMOTE", False), ("After SMOTE", True)):
        print(f"\n=== {scenario_name} ===")
        if use_smote:
            x_train_used, y_train_used = apply_smote_to_training_data(x_train, y_train)
            smote_negative_count, smote_positive_count, smote_ratio = calculate_class_imbalance_ratio(y_train_used)
            print(
                f"SMOTE-balanced training set: negative={smote_negative_count}, positive={smote_positive_count}, ratio={smote_ratio:.4f}"
            )
            run_cv = False
        else:
            x_train_used: Any = x_train
            y_train_used = y_train
            smote_ratio = original_ratio
            run_cv = True

        models = build_models(scale_pos_weight=smote_ratio)

        for model_name, model in models.items():
            print(f"\n--- Training {model_name} [{scenario_name}] ---")
            model.fit(x_train_used, y_train_used)
            train_predictions = model.predict(x_train_used)
            test_predictions = model.predict(x_test)
            train_probabilities = get_positive_class_probabilities(model, x_train_used)
            test_probabilities = get_positive_class_probabilities(model, x_test)

            train_metrics = {
                "accuracy": accuracy_score(y_train_used, train_predictions),
                "precision": precision_score(y_train_used, train_predictions, zero_division=0),
                "recall": recall_score(y_train_used, train_predictions, zero_division=0),
                "f1": f1_score(y_train_used, train_predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_train_used, train_probabilities),
            }
            test_metrics = {
                "accuracy": accuracy_score(y_test, test_predictions),
                "precision": precision_score(y_test, test_predictions, zero_division=0),
                "recall": recall_score(y_test, test_predictions, zero_division=0),
                "f1": f1_score(y_test, test_predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, test_probabilities),
            }

            report_dict = print_classification_report(y_test, test_predictions, model_name)
            cv_summary = run_cross_validation(model, x_train_used, y_train_used) if run_cv else None
            fit_status, fit_explanation = assess_fit(train_metrics, test_metrics)

            print(
                f"Overfitting check: {fit_status}. {fit_explanation} "
                f"(Train F1={train_metrics['f1']:.4f}, Test F1={test_metrics['f1']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.4f}, Test Acc={test_metrics['accuracy']:.4f})"
            )

            fpr, tpr, _ = roc_curve(y_test, test_probabilities)
            roc_data[f"{scenario_name} - {model_name}"] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": test_metrics["roc_auc"],
            }

            print(
                f"Training Metrics: Accuracy={train_metrics['accuracy']:.4f} | "
                f"Precision={train_metrics['precision']:.4f} | Recall={train_metrics['recall']:.4f} | "
                f"F1={train_metrics['f1']:.4f} | ROC-AUC={train_metrics['roc_auc']:.4f}"
            )
            print(
                f"Testing Metrics:  Accuracy={test_metrics['accuracy']:.4f} | "
                f"Precision={test_metrics['precision']:.4f} | Recall={test_metrics['recall']:.4f} | "
                f"F1={test_metrics['f1']:.4f} | ROC-AUC={test_metrics['roc_auc']:.4f}"
            )

            model_metrics: Dict[str, object] = {
                "scenario": scenario_name,
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "roc_auc": test_metrics["roc_auc"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_roc_auc": train_metrics["roc_auc"],
                "scale_pos_weight": smote_ratio,
                "classification_report": report_dict,
                "fit_status": fit_status,
                "fit_explanation": fit_explanation,
            }

            if cv_summary is not None:
                model_metrics.update(
                    {
                        "cv_fold_accuracy": cv_summary["fold_metrics"]["accuracy"],
                        "cv_fold_precision": cv_summary["fold_metrics"]["precision"],
                        "cv_fold_recall": cv_summary["fold_metrics"]["recall"],
                        "cv_fold_f1": cv_summary["fold_metrics"]["f1"],
                        "cv_mean_accuracy": cv_summary["mean_accuracy"],
                        "cv_mean_precision": cv_summary["mean_precision"],
                        "cv_mean_recall": cv_summary["mean_recall"],
                        "cv_mean_f1": cv_summary["mean_f1"],
                        "cv_std_accuracy": cv_summary["std_accuracy"],
                        "cv_std_precision": cv_summary["std_precision"],
                        "cv_std_recall": cv_summary["std_recall"],
                        "cv_std_f1": cv_summary["std_f1"],
                    }
                )
            else:
                model_metrics.update(
                    {
                        "cv_fold_accuracy": None,
                        "cv_fold_precision": None,
                        "cv_fold_recall": None,
                        "cv_fold_f1": None,
                        "cv_mean_accuracy": None,
                        "cv_mean_precision": None,
                        "cv_mean_recall": None,
                        "cv_mean_f1": None,
                        "cv_std_accuracy": None,
                        "cv_std_precision": None,
                        "cv_std_recall": None,
                        "cv_std_f1": None,
                    }
                )

            metrics[f"{scenario_name} - {model_name}"] = model_metrics

            if test_metrics["f1"] > best_f1:
                best_f1 = test_metrics["f1"]
                best_model_name = f"{scenario_name} - {model_name}"
                best_model = model
                best_test_probabilities = test_probabilities

    if best_model is None or best_test_probabilities is None:
        raise RuntimeError("No model was trained successfully.")

    plot_roc_curves(roc_data)

    return metrics, best_model_name, best_model, best_test_probabilities


def plot_confusion_matrix(y_test, y_pred, model_name: str) -> None:
    """Save a confusion matrix figure for the selected model."""

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()


def save_artifacts(model: TrainedModel, vectorizer: TfidfVectorizer, metrics: Dict[str, Dict[str, object]]) -> None:
    """Persist the best model, vectorizer, and metrics to disk."""

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main(data_path: Path = DEFAULT_DATA_PATH) -> None:
    """Run the full fake-job detection workflow."""

    print("Loading dataset...")
    df = load_dataset(data_path)

    print("Handling missing values...")
    df = handle_missing_values(df)

    run_eda(df)

    print("Preparing cleaned text features...")
    features, target = prepare_features(df)

    print("Splitting train and test sets...")
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    print("Training and evaluating models...")
    metrics, best_model_name, best_model, best_test_probabilities = train_and_evaluate_models(
        x_train,
        x_test,
        y_train,
        y_test,
    )

    best_predictions = best_model.predict(x_test)
    plot_confusion_matrix(y_test, best_predictions, best_model_name)

    threshold_frame = evaluate_threshold_metrics(y_test, best_test_probabilities)
    plot_precision_recall_curve(y_test, best_test_probabilities)
    plot_threshold_tradeoffs(threshold_frame)
    recommended_threshold, recommended_row = recommend_threshold(threshold_frame)

    print("\n=== Model Comparison ===")
    metrics_frame = pd.DataFrame(metrics).T.sort_values(by=["recall", "f1", "accuracy"], ascending=False)
    print(metrics_frame.to_string(float_format=lambda value: f"{value:.4f}"))

    before_smote_frame = metrics_frame[metrics_frame["scenario"] == "Before SMOTE"]
    after_smote_frame = metrics_frame[metrics_frame["scenario"] == "After SMOTE"]

    print("\n=== Before SMOTE Summary ===")
    print(before_smote_frame[["accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(float_format=lambda value: f"{value:.4f}"))

    print("\n=== After SMOTE Summary ===")
    print(after_smote_frame[["accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(float_format=lambda value: f"{value:.4f}"))

    print(
        f"\nRecommended threshold: {recommended_threshold:.2f} "
        f"(Accuracy={recommended_row['accuracy']:.4f}, Precision={recommended_row['precision']:.4f}, "
        f"Recall={recommended_row['recall']:.4f}, F1={recommended_row['f1']:.4f})"
    )

    print(f"\nBest model selected: {best_model_name}")
    save_artifacts(best_model, vectorizer, metrics)
    run_real_world_prediction_tests(best_model, vectorizer, recommended_threshold)

    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved TF-IDF vectorizer to: {VECTORIZER_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")
    print(f"Saved ROC curve to: {ROC_CURVE_PATH}")
    print(f"Saved precision-recall curve to: {PRECISION_RECALL_CURVE_PATH}")
    print(f"Saved threshold-vs-recall plot to: {THRESHOLD_RECALL_PATH}")
    print(f"Saved threshold-vs-precision plot to: {THRESHOLD_PRECISION_PATH}")
    print(f"Saved EDA plots to: {EDA_DIR}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake job detection training pipeline")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the Kaggle fake job postings CSV file.",
    )
    args = parser.parse_args()
    main(args.data_path)