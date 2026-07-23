"""Domain-independent skill and keyphrase extraction.

Replaces the old hardcoded SKILL_TERMS catalog. Works for any profession
(software, medicine, law, teaching, forensic science, marketing, etc.) by
combining:

  1. Statistical keyphrase extraction (RAKE-style: candidate noun-ish chunks
     ranked by word co-occurrence degree/frequency). No model download and
     no network access required, so it always works.
  2. A lightweight noun-phrase / capitalized-term / acronym-pattern extractor
     that catches domain terms statistical extraction misses ("DNA Analysis",
     "PhD Supervision", "B.Tech", "NLP").
  3. An optional semantic layer (sentence-transformers) that, if the
     dependency is installed and a model can be loaded, is used to cluster
     near-duplicate phrasings ("Artificial Intelligence" / "AI") and to do
     semantic (not just lexical) matching between resume and job phrases.
     If unavailable, the system degrades gracefully to synonym-table +
     token-overlap matching. Nothing hard-fails if the optional dependency
     is missing.

Everything here is stateless / pure functions plus one cached singleton for
the optional embedding model, so it is safe to reuse across requests without
re-loading anything expensive per request.
"""

from __future__ import annotations

import functools
import logging
import re
from collections import Counter
from typing import Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords / noise control (domain-agnostic, not a skill catalog)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "for", "with", "as", "at", "by", "from", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "we", "you", "they", "he", "she", "our", "your", "their", "will",
    "shall", "should", "would", "can", "could", "may", "might", "must",
    "have", "has", "had", "do", "does", "did", "not", "no", "yes", "so",
    "such", "than", "into", "onto", "about", "over", "under", "per", "via",
    "etc", "including", "include", "includes", "job", "role", "position",
    "candidate", "candidates", "applicant", "we're", "responsibilities",
    "requirements", "qualifications", "preferred", "required", "must-have",
    "years", "year", "experience", "strong", "excellent", "good", "ability",
    "skills", "skill", "knowledge", "understanding", "looking", "seeking",
    "work", "working", "team", "teams", "company", "organization", "please",
    "apply", "join", "opportunity", "description", "summary", "responsible",
    "hiring", "hire", "seeking", "minimum", "preferred", "plus", "duties",
    "various", "other", "general", "overall", "role", "roles",
    "you", "your", "our", "us", "them", "someone", "person", "individual",
    "detail", "details", "conducting", "developing", "creating", "using",
    "provide", "providing", "ensure", "ensuring", "maintain", "maintaining",
    "developer", "developers", "engineer", "engineers", "requiring",
    "specialist", "manager", "associate", "analyst", "consultant",
}

# Trailing/leading words that make an otherwise fine candidate phrase read as
# generic filler rather than a real skill/requirement (e.g. "Related Field",
# "Degree Required"). These are filtered at the phrase edges only, so
# legitimate multi-word terms that happen to *contain* these words elsewhere
# (e.g. "Industry Collaboration") are not affected.
_GENERIC_PHRASE_EDGES = {
    "related", "field", "degree", "required", "requires", "requirement",
    "including", "background", "preferred", "needed", "plus", "minimum",
    "developer", "engineer", "role", "position", "specialist", "manager",
    "associate", "analyst", "consultant", "requiring",
}

_SPLIT_RE = re.compile(r"[\n\r\u2022•·▪●\-–—]+")
_SENT_RE = re.compile(r"(?<=[.!?;:])\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+/#.]*(?:'[A-Za-z]+)?")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")
_CAP_PHRASE_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:\s+(?:of|for|and|in|the|&)?\s*[A-Z][a-zA-Z0-9]*){0,4})\b"
)
_DEGREE_RE = re.compile(
    r"\b(b\.?\s?tech|b\.?\s?sc|b\.?\s?a|b\.?\s?e|m\.?\s?tech|m\.?\s?sc|m\.?\s?a|m\.?\s?b\.?\s?a|"
    r"ph\.?\s?d|m\.?\s?d|j\.?\s?d|bachelor(?:'s)?|master(?:'s)?|doctorate|diploma|associate degree)\b",
    re.I,
)

# Generic synonym / abbreviation table. This is NOT a skill catalog — it is a
# normalization layer so that different surface forms of the *same* concept
# collapse to one canonical key, regardless of domain. New domains add
# entries here over time, but the *extractor* itself never needs new code to
# support a new profession.
CANONICAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "artificial intelligence": ("artificial intelligence", "ai"),
    "machine learning": ("machine learning", "ml"),
    "natural language processing": ("natural language processing", "nlp"),
    "deep learning": ("deep learning", "dl"),
    "crime scene investigation": ("crime scene investigation", "csi"),
    "bachelor of technology": ("bachelor of technology", "b.tech", "btech"),
    "bachelor of science": ("bachelor of science", "b.sc", "bsc"),
    "master of business administration": ("master of business administration", "mba"),
    "master of technology": ("master of technology", "m.tech", "mtech"),
    "doctor of philosophy": ("doctor of philosophy", "phd", "ph.d"),
    "doctor of medicine": ("doctor of medicine", "md", "m.d"),
    "search engine optimization": ("search engine optimization", "seo"),
    "continuous integration continuous deployment": ("ci/cd", "cicd", "continuous integration"),
    "javascript": ("javascript", "js"),
    "typescript": ("typescript", "ts"),
    "user experience": ("user experience", "ux"),
    "user interface": ("user interface", "ui"),
    "human resources": ("human resources", "hr"),
    "search engine marketing": ("search engine marketing", "sem"),
    "electronic health record": ("electronic health record", "ehr", "emr"),
    "deoxyribonucleic acid": ("dna analysis", "dna"),
    "return on investment": ("return on investment", "roi"),
    "key performance indicator": ("key performance indicator", "kpi"),
    "chief financial officer": ("chief financial officer", "cfo"),
    "generally accepted accounting principles": ("gaap",),
    "certified public accountant": ("certified public accountant", "cpa"),
    "registered nurse": ("registered nurse", "rn"),
    "national academic accreditation council": ("naac",),
}

_SURFACE_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in CANONICAL_SYNONYMS.items():
    for _alias in _aliases:
        _SURFACE_TO_CANONICAL[_alias.lower()] = _canonical


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_phrase(value: str | None) -> str:
    """Domain-agnostic replacement for the old normalize_skill_name().

    Lowercases, strips punctuation noise, collapses whitespace, and folds
    known synonym/abbreviation variants onto one canonical string. Falls
    back to a cleaned version of the original phrase for anything not in
    the synonym table (which is the common case — this function does not
    require a phrase to be "known" to normalize it).
    """
    if not value:
        return ""
    cleaned = normalize_text(value).lower()
    cleaned = re.sub(r"[^a-z0-9.+/#&\s-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
    if not cleaned:
        return ""
    if cleaned in _SURFACE_TO_CANONICAL:
        return _SURFACE_TO_CANONICAL[cleaned]
    stripped = cleaned.replace(".", "").replace(" ", "")
    for alias, canonical in _SURFACE_TO_CANONICAL.items():
        if alias.replace(".", "").replace(" ", "") == stripped:
            return canonical
    return cleaned


def display_phrase(value: str | None) -> str:
    phrase = normalize_phrase(value)
    if not phrase:
        return ""

    DISPLAY_NAMES = {
        "react": "React",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "java": "Java",
        "spring": "Spring",
        "spring boot": "Spring Boot",
        "boot": "Boot",
        "angular": "Angular",
        "git": "Git",

        "html": "HTML",
        "css": "CSS",
        "sql": "SQL",
        "api": "API",
        "rest": "REST",
        "rest api": "REST API",
        "aws": "AWS",
        "gcp": "GCP",
        "azure": "Azure",
        "ai": "AI",
        "ml": "ML",
        "nlp": "NLP",
        "kpi": "KPI",
        "roi": "ROI",
        "gaap": "GAAP",
        "naac": "NAAC",
        "rn": "RN",
        "dna analysis": "DNA Analysis",
    }

    if phrase in DISPLAY_NAMES:
        return DISPLAY_NAMES[phrase]

    return phrase.title()

# ---------------------------------------------------------------------------
# Candidate phrase extraction (statistical + pattern based, no ML required)
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    for block in _SPLIT_RE.split(text):
        chunks.extend(_SENT_RE.split(block))
    return [c.strip() for c in chunks if c.strip()]


def _rake_candidates(text: str) -> list[str]:
    """RAKE-style candidate generation: split on stopwords, keep runs of
    content words as candidate phrases. Domain-independent by construction —
    it only depends on stopwords, not on any skill vocabulary.

    Short skill-list-style inputs ("Python React SQL", "Skills: Java,
    Spring Boot, SQL") have no stopwords between tokens, so without extra
    handling they'd collapse into one run-on phrase. To avoid that we also
    split on commas/slashes/pipes (explicit list separators), and break up
    any surviving candidate that looks like a bare run of capitalized/short
    tech tokens with no lowercase glue words — that pattern indicates a
    skills list, not a natural-language phrase.
    """
    candidates: list[str] = []
    for sentence in _sentences(text):
        for segment in re.split(r"[,/|]+", sentence):
            words = _WORD_RE.findall(segment)
            current: list[str] = []

            def flush():
                if current:
                    candidates.append(" ".join(current))
                    current.clear()

            for word in words:
                if word.lower() in _STOPWORDS or len(word) <= 1:
                    flush()
                    continue
                current.append(word)
            flush()

    expanded: list[str] = []
    for phrase in candidates:
        words = phrase.split()
        if len(words) >= 2 and all(_looks_like_standalone_token(w) for w in words):
            expanded.extend(words)
        else:
            expanded.append(phrase)

    return [c for c in expanded if c and len(c.split()) <= 5]


def _looks_like_standalone_token(word: str) -> bool:
    """True for tokens that read as an individual skill/technology name
    rather than a connector inside a longer phrase — e.g. 'Python', 'SQL',
    'React', 'C++' — but not lowercase words like 'management', 'field'."""
    if word[0].isupper() or word.isupper():
        return True
    return bool(re.match(r"^[a-z0-9.+#/-]+$", word)) and len(word) <= 3


def _score_candidates(candidates: list[str]) -> dict[str, float]:
    """RAKE degree/frequency scoring: word score = (co-occurrence degree /
    frequency), phrase score = sum of member word scores."""
    freq: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for phrase in candidates:
        words = [w.lower() for w in phrase.split()]
        deg = len(words) - 1
        for w in words:
            freq[w] += 1
            degree[w] += deg
    word_score = {w: (degree[w] + freq[w]) / freq[w] for w in freq}
    phrase_score: dict[str, float] = {}
    for phrase in candidates:
        words = [w.lower() for w in phrase.split()]
        score = sum(word_score.get(w, 0.0) for w in words)
        key = phrase.strip()
        phrase_score[key] = max(phrase_score.get(key, 0.0), score)
    return phrase_score


def _acronym_and_capitalized_candidates(text: str) -> list[str]:
    """Catches domain terms RAKE tends to miss: acronyms (NLP, DNA, PhD) and
    multi-word capitalized phrases (Crime Scene Investigation, Evidence
    Collection) — this is a lightweight stand-in for NER / noun-phrase
    chunking when spaCy is not installed."""
    found: list[str] = []
    for match in _ACRONYM_RE.finditer(text):
        found.append(match.group(1))
    for match in _CAP_PHRASE_RE.finditer(text):
        phrase = match.group(1).strip()
        if len(phrase.split()) >= 2 and phrase.split()[0].lower() not in _STOPWORDS:
            found.append(phrase)
    for match in _DEGREE_RE.finditer(text):
        found.append(match.group(1))
    return found


def extract_key_phrases(text: str, top_n: int = 40) -> list[str]:
    """Domain-independent replacement for a hardcoded skill list.

    Extracts the most salient phrases from arbitrary text (a job description
    or resume section) regardless of profession, using RAKE-style statistical
    scoring plus pattern-based acronym/capitalized-phrase capture.
    """
    text = normalize_text(text)
    if not text:
        return []

    rake_candidates = _rake_candidates(text)
    scored = _score_candidates(rake_candidates)
    pattern_candidates = _acronym_and_capitalized_candidates(text)
    expanded_pattern_candidates: list[str] = []
    for phrase in pattern_candidates:
        words = phrase.split()
        if len(words) >= 2 and all(_looks_like_standalone_token(w) for w in words):
            expanded_pattern_candidates.extend(words)
        else:
            expanded_pattern_candidates.append(phrase)

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    ordered_phrases = [normalize_phrase(p) for p, _ in ranked]
    ordered_phrases += [normalize_phrase(p) for p in expanded_pattern_candidates]
    seen: set[str] = set()
    result: list[str] = []
    for phrase in ordered_phrases:
        phrase = _trim_generic_edges(phrase)
        if not phrase or phrase in seen:
            continue
        if phrase in _STOPWORDS:
            continue
        if len(phrase) < 2:
            continue
        seen.add(phrase)
        result.append(phrase)
        if len(result) >= top_n:
            break
    return result


def _trim_generic_edges(phrase: str) -> str:
    """Strips generic filler words from the start/end of a candidate phrase
    (e.g. "related field" -> "" , "degree required" -> "") without
    penalizing legitimate multi-word terms that merely contain one of these
    words in the middle (e.g. "industry collaboration" is untouched)."""
    words = phrase.split()
    while words and words[0] in _GENERIC_PHRASE_EDGES:
        words = words[1:]
    while words and words[-1] in _GENERIC_PHRASE_EDGES:
        words = words[:-1]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Optional semantic layer (lazy-loaded, cached, gracefully degrades)
# ---------------------------------------------------------------------------

import os as _os

_SEMANTIC_ENABLED = _os.environ.get("ATS_ENABLE_SEMANTIC_MATCHING", "").lower() in {"1", "true", "yes"}


@functools.lru_cache(maxsize=1)
def _get_embedding_model():
    """Lazily load a sentence-transformers model exactly once per process,
    only if explicitly enabled via ATS_ENABLE_SEMANTIC_MATCHING=1.

    Semantic matching is opt-in rather than attempted by default: a
    cold-start model load can take several seconds and may hit the network
    if the model is not already cached locally, which would blow the ATS's
    2-3 second response budget on a process's first request. Operators who
    want semantic matching and have pre-warmed/pre-downloaded the model can
    opt in explicitly; everyone else gets the fast, dependency-free lexical
    fallback with no cold-start surprises.

    Returns None if disabled, not installed, or the model can't be loaded.
    Callers must treat None as fall back to lexical matching.
    """
    if not _SEMANTIC_ENABLED:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.info("Semantic model unavailable, using lexical fallback: %s", exc)
        return None


def semantic_similarity_matrix(left: list[str], right: list[str]):
    """Returns a similarity matrix if the embedding model is available,
    else None — callers should use lexical fallback matching instead."""
    model = _get_embedding_model()
    if model is None or not left or not right:
        return None
    try:
        import numpy as np  # type: ignore

        left_vecs = model.encode(left, normalize_embeddings=True)
        right_vecs = model.encode(right, normalize_embeddings=True)
        return np.matmul(left_vecs, right_vecs.T)
    except Exception as exc:  # pragma: no cover
        logger.info("Semantic similarity computation failed, using lexical fallback: %s", exc)
        return None


def _token_set(phrase: str) -> set[str]:
    return {t for t in re.split(r"[\s/+_-]+", phrase.lower()) if t and t not in _STOPWORDS}


def lexical_similarity(a: str, b: str) -> float:
    """Jaccard token overlap as the always-available fallback similarity."""
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 1.0 if a == b else 0.0
    if ta == tb:
        return 1.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    base = intersection / union if union else 0.0
    if a.startswith(b) or b.startswith(a):
        base = max(base, 0.6)
    return base


def match_phrases(
    resume_phrases: Iterable[str],
    job_phrases: Iterable[str],
    threshold: float = 0.6,
) -> tuple[list[str], list[str], list[str]]:
    """Match job-required phrases against resume phrases using semantic
    similarity when available, else synonym-normalized lexical overlap.

    Returns (matched_job_phrases, missing_job_phrases, extra_resume_phrases).
    """
    resume_list = [normalize_phrase(p) for p in resume_phrases if normalize_phrase(p)]
    job_list = [normalize_phrase(p) for p in job_phrases if normalize_phrase(p)]
    resume_set = set(resume_list)

    matched: list[str] = []
    missing: list[str] = []

    matrix = semantic_similarity_matrix(job_list, resume_list) if job_list and resume_list else None

    for i, job_phrase in enumerate(job_list):
        if job_phrase in resume_set:
            matched.append(job_phrase)
            continue
        found = False
        if matrix is not None:
            row = matrix[i]
            if len(row) and float(row.max()) >= threshold:
                found = True
        if not found:
            for resume_phrase in resume_list:
                if lexical_similarity(job_phrase, resume_phrase) >= threshold:
                    found = True
                    break
        if found:
            matched.append(job_phrase)
        else:
            missing.append(job_phrase)

    job_set = set(job_list)
    extra = [p for p in resume_list if p not in job_set and p not in set(matched)]
    return matched, missing, extra