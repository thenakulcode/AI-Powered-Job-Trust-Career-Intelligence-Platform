"""Domain-independent ATS matching engine.

Orchestrates the modular pipeline:
  SkillExtractor (skill_extraction.py) -> DomainDetector -> EducationMatcher
  -> ExperienceMatcher -> ProjectMatcher -> RecommendationEngine
  -> InterviewGenerator -> JobRecommender

The public method signature (`assess_resume`) and the JSON response shape
are unchanged from the previous ats_service.py, so the existing frontend and
resume.py route keep working without modification. Only the intelligence
underneath changed: nothing here assumes a software engineering role.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from typing import Any

from backend.services.domain_detector import DomainMatch, detect_domain, humanize_domain
from backend.services.education_matcher import EducationMatcher
from backend.services.experience_matcher import ExperienceMatcher
from backend.services.interview_generator import InterviewGenerator
from backend.services.job_recommender import JobRecommender
from backend.services.project_matcher import ProjectMatcher
from backend.services.recommendation_engine_v2 import DomainRecommendationEngine
from backend.services.resume_parser import parse_resume_sections
from backend.services.scraper import extract_job_post
from backend.services.skill_extraction import (
    display_phrase,
    extract_key_phrases,
    match_phrases,
    normalize_phrase,
)

logger = logging.getLogger(__name__)


class ATSService:
    """Generate a deterministic, domain-independent ATS-style match score
    from resume and job data. Public API unchanged from the previous
    implementation for backward compatibility."""

    def __init__(self) -> None:
        self._education_matcher = EducationMatcher()
        self._experience_matcher = ExperienceMatcher()
        self._project_matcher = ProjectMatcher()
        self._recommendation_engine = DomainRecommendationEngine()
        self._interview_generator = InterviewGenerator()
        self._job_recommender = JobRecommender()

    def assess_resume(self, resume_text: str, job_description: str) -> dict[str, Any]:
        logger.info("------------------------------------------------")
        logger.info("ATS stage: starting resume assessment (domain-independent pipeline)")
        if not resume_text or not resume_text.strip():
            return self._build_error("Unable to extract readable resume text.")

        parsed = parse_resume_sections(resume_text)
        if not self._is_readable_resume(parsed, resume_text):
            return self._build_error("This document does not contain a readable resume profile.")

        cleaned_job_text = self._prepare_job_description(job_description)

        # --- Domain detection (job description drives it; resume text is a
        # secondary signal if the JD is thin) ---------------------------------
        domain_match = detect_domain(cleaned_job_text)
        if domain_match.domain == "general":
            resume_domain = detect_domain(resume_text)
            if resume_domain.domain != "general":
                domain_match = DomainMatch(domain=resume_domain.domain, confidence=resume_domain.confidence * 0.5)

        # --- Dynamic skill/phrase extraction (replaces hardcoded catalog) ----
        job_phrases = extract_key_phrases(job_description, top_n=25)
        resume_phrase_source = " ".join(
            [
                resume_text or "",
                " ".join(parsed.get("skills", []) or []),
                " ".join(parsed.get("experience", []) or []),
                " ".join(parsed.get("projects", []) or []),
                " ".join(parsed.get("certifications", []) or []),
                " ".join(parsed.get("achievements", []) or []),
            ]
        )
        resume_phrases = extract_key_phrases(resume_phrase_source, top_n=40)

        matched_skills, missing_skills, extra_skills = match_phrases(resume_phrases, job_phrases, threshold=0.6)

        job_keywords = job_phrases
        resume_keywords = resume_phrases
        matched_keywords, missing_keywords, _ = match_phrases(resume_keywords, job_keywords, threshold=0.6)

        score_breakdown = self._build_score_breakdown(
            parsed, cleaned_job_text, domain_match, matched_skills, job_phrases, matched_keywords, job_keywords
        )
        ats_score = int(round(score_breakdown["ats_score"]))
        match_score = int(round(score_breakdown["match_score"]))
        status = "Suitable" if ats_score >= 60 else "Not Suitable"
        reason = self._build_reason(matched_skills, missing_skills, missing_keywords, parsed, domain_match)

        recommendation_payload = self._recommendation_engine.generate(
            missing_skills,
            domain_match,
            {
                "experience_level": parsed.get("total_experience", 0),
                "projects": parsed.get("projects", []),
                "education": parsed.get("education", []),
            },
        )
        recommendations = recommendation_payload["improvementSuggestions"]
        learning_roadmap = recommendation_payload["recommendedProjects"]
        recommended_jobs = self._job_recommender.recommend(domain_match, matched_skills, missing_skills)
        interview_preparation = self._interview_generator.generate(domain_match, matched_skills, missing_skills)
        warning = self._build_warning(ats_score, status, reason)

        logger.info("Detected Domain: %s (confidence=%.2f)", domain_match.domain, domain_match.confidence)
        logger.info("Resume Skills: %s", [display_phrase(s) for s in resume_phrases[:20]])
        logger.info("Job Skills: %s", [display_phrase(s) for s in job_phrases])
        logger.info("Matched Skills: %s", [display_phrase(s) for s in matched_skills])
        logger.info("Missing Skills: %s", [display_phrase(s) for s in missing_skills])
        logger.info("ATS Breakdown: %s", score_breakdown)
        logger.info("Final Score: %s", ats_score)
        logger.info("------------------------------------------------")

        return {
            "success": True,
            "atsScore": ats_score,
            "matchScore": match_score,
            "status": status,
            "matchedSkills": [display_phrase(s) for s in matched_skills],
            "missingSkills": [display_phrase(s) for s in missing_skills],
            "matchedKeywords": [display_phrase(k) for k in matched_keywords],
            "missingKeywords": [display_phrase(k) for k in missing_keywords],
            "extraSkills": [display_phrase(s) for s in extra_skills[:8]],
            "strengths": self._build_strengths(matched_skills, matched_keywords, domain_match),
            "weaknesses": self._build_weaknesses(missing_skills, missing_keywords, parsed, domain_match),
            "recommendations": recommendations,
            "learningRoadmap": recommendation_payload.get("learningRoadmap", learning_roadmap),
            "recommendedJobs": recommended_jobs,
            "warning": warning,
            "reason": reason,
            "improvementSuggestions": recommendation_payload["improvementSuggestions"],
            "recommendedCourses": recommendation_payload["recommendedCourses"],
            "recommendedProjects": recommendation_payload["recommendedProjects"],
            "interviewPreparation": interview_preparation,
            "resumeSummary": {
                "name": parsed.get("name", ""),
                "email": parsed.get("email", ""),
                "phone": parsed.get("phone", ""),
                "totalExperience": parsed.get("total_experience", 0),
                "skills": parsed.get("skills", []),
                "education": parsed.get("education", []),
                "projects": parsed.get("projects", []),
                "experience": parsed.get("experience", []),
                "certifications": parsed.get("certifications", []),
                "achievements": parsed.get("achievements", []),
            },
            "scoreBreakdown": {
                "skills": round(score_breakdown["skills_score"], 0),
                "keywords": round(score_breakdown["keywords_score"], 0),
                "experience": round(score_breakdown["experience_score"], 0),
                "projects": round(score_breakdown["projects_score"], 0),
                "education": round(score_breakdown["education_score"], 0),
                "certifications": round(score_breakdown["certifications_score"], 0),
                "contact": round(score_breakdown["contact_score"], 0),
            },
            "summary": self._build_summary(matched_skills, missing_skills, ats_score, domain_match),
            "recommendationsList": recommendations,
            "learningRoadmapList": recommendation_payload.get("learningRoadmap", learning_roadmap),
            "recommendedJobsList": recommended_jobs,
            # Additive field: detected domain, safe for old frontends to ignore.
            "detectedDomain": humanize_domain(domain_match.domain),
        }

    # ------------------------------------------------------------------
    # Error / validation
    # ------------------------------------------------------------------

    def _build_error(self, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "message": message,
            "atsScore": 0,
            "matchScore": 0,
            "status": "Not Suitable",
            "matchedSkills": [],
            "missingSkills": [],
            "matchedKeywords": [],
            "missingKeywords": [],
            "extraSkills": [],
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "learningRoadmap": [],
            "recommendedJobs": [],
            "warning": message,
            "improvementSuggestions": [],
            "recommendedCourses": [],
            "recommendedProjects": [],
            "interviewPreparation": [],
            "reason": message,
            "resumeSummary": {},
            "scoreBreakdown": {
                "skills": 0,
                "keywords": 0,
                "experience": 0,
                "projects": 0,
                "education": 0,
                "certifications": 0,
                "contact": 0,
            },
            "summary": message,
            "detectedDomain": "",
        }

    def _is_readable_resume(self, parsed: dict[str, Any], resume_text: str) -> bool:
        if not resume_text or not str(resume_text).strip():
            return False
        if len(re.sub(r"\s+", "", resume_text)) < 60:
            return False
        has_contact = bool(parsed.get("email") or parsed.get("phone"))
        has_education = bool(parsed.get("education"))
        has_experience = bool(parsed.get("experience"))
        has_any_content = has_education or has_experience or bool(parsed.get("skills")) or bool(parsed.get("projects"))
        return has_contact and has_any_content

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        parsed: dict[str, Any],
        job_description: str,
        domain_match: DomainMatch,
        matched_skills: list[str],
        job_skills: list[str],
        matched_keywords: list[str],
        job_keywords: list[str],
    ) -> dict[str, float]:
        skills_score = self._safe_percent(len(matched_skills), max(len(job_skills), 1))
        keywords_score = self._safe_percent(len(matched_keywords), max(len(job_keywords), 1))

        experience_score, _ = self._experience_matcher.score(
            job_description, parsed.get("experience", []) or [], parsed.get("total_experience", 0) or 0
        )
        projects_score, _ = self._project_matcher.score(job_description, parsed, domain_match)
        education_score, _ = self._education_matcher.score(job_description, parsed.get("education", []) or [])
        certifications_score = self._calculate_certifications_score(parsed, job_description)
        contact_score = self._calculate_contact_score(parsed)

        ats_score = (
            skills_score * 0.35
            + keywords_score * 0.20
            + experience_score * 0.15
            + projects_score * 0.10
            + education_score * 0.10
            + certifications_score * 0.05
            + contact_score * 0.05
        )
        match_score = ats_score
        if not matched_skills and job_skills:
            ats_score = min(ats_score, 15.0)
            match_score = min(match_score, 15.0)

        return {
            "skills_score": skills_score,
            "keywords_score": keywords_score,
            "experience_score": experience_score,
            "projects_score": projects_score,
            "education_score": education_score,
            "certifications_score": certifications_score,
            "contact_score": contact_score,
            "ats_score": ats_score,
            "match_score": match_score,
        }

    def _calculate_certifications_score(self, parsed: dict[str, Any], job_description: str) -> float:
        certifications = parsed.get("certifications") or []
        if not certifications:
            return 0.0
        job_phrases = extract_key_phrases(job_description, top_n=20)
        cert_phrases = extract_key_phrases(" ".join(certifications), top_n=20)
        if not job_phrases:
            return 40.0
        matched, _, _ = match_phrases(cert_phrases, job_phrases, threshold=0.55)
        return 100.0 if matched else 40.0

    def _calculate_contact_score(self, parsed: dict[str, Any]) -> float:
        count = sum(1 for value in [parsed.get("email"), parsed.get("phone")] if value)
        return 100.0 if count == 2 else 60.0 if count == 1 else 20.0

    def _safe_percent(self, part: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round((part / total) * 100, 2)

    # ------------------------------------------------------------------
    # Narrative text builders
    # ------------------------------------------------------------------

    def _build_strengths(self, matched_skills: list[str], matched_keywords: list[str], domain_match: DomainMatch) -> list[str]:
        strengths: list[str] = []
        if matched_skills:
            top = ", ".join(display_phrase(s) for s in matched_skills[:3])
            strengths.append(f"Strong overlap with {top} for this {humanize_domain(domain_match.domain)} role.")
        if matched_keywords:
            strengths.append("The resume text overlaps with several job keywords.")
        return strengths or ["The resume contains enough evidence to be evaluated."]

    def _build_weaknesses(self, missing_skills: list[str], missing_keywords: list[str], parsed: dict[str, Any], domain_match: DomainMatch) -> list[str]:
        weaknesses: list[str] = []
        if missing_skills:
            missing_list = ", ".join(display_phrase(s) for s in missing_skills[:4])
            weaknesses.append(f"Missing areas for this {humanize_domain(domain_match.domain)} role: {missing_list}.")
        if missing_keywords:
            weaknesses.append("Missing keyword coverage from the job description.")
        if not (parsed.get("projects") or parsed.get("achievements") or parsed.get("experience")):
            weaknesses.append("The resume does not include strong supporting evidence (projects, achievements, or experience).")
        return weaknesses or ["No obvious weaknesses detected from the extracted content."]

    def _build_reason(
        self,
        matched_skills: list[str],
        missing_skills: list[str],
        missing_keywords: list[str],
        parsed: dict[str, Any],
        domain_match: DomainMatch,
    ) -> str:
        if not matched_skills and missing_skills:
            missing_list = ", ".join(display_phrase(s) for s in missing_skills[:6])
            return f"The resume does not show the areas required for this {humanize_domain(domain_match.domain)} role ({missing_list})."
        if missing_keywords:
            missing_list = ", ".join(display_phrase(s) for s in missing_skills[:4])
            return f"The resume is missing several job keywords and required areas ({missing_list})."
        if not parsed.get("projects") and not parsed.get("achievements"):
            return "The resume lacks supporting evidence that backs up the role's requirements."
        return "The resume has a meaningful overlap with the role requirements."

    def _build_warning(self, ats_score: int, status: str, reason: str) -> str:
        if status == "Not Suitable":
            return f"Not Suitable: {reason}"
        return ""

    def _build_summary(self, matched_skills: list[str], missing_skills: list[str], ats_score: int, domain_match: DomainMatch) -> str:
        domain_label = humanize_domain(domain_match.domain)
        if ats_score >= 80:
            return f"Strong fit for this {domain_label} role with {len(matched_skills)} matched areas and an ATS score of {ats_score}."
        if ats_score >= 60:
            missing_list = ", ".join(display_phrase(s) for s in missing_skills[:3])
            return f"Good fit for this {domain_label} role with {len(matched_skills)} matched areas; add evidence for {missing_list}."
        missing_list = ", ".join(display_phrase(s) for s in missing_skills[:3])
        return f"This profile does not yet align with the {domain_label} role; prioritize {missing_list}."

    # ------------------------------------------------------------------
    # Job description preparation (unchanged behavior)
    # ------------------------------------------------------------------

    def _prepare_job_description(self, value: str) -> str:
        if not value:
            return ""
        text = str(value).strip()
        if self._looks_like_url(text):
            try:
                data = extract_job_post(text, timeout=10)
                parts = [data.get("description", ""), data.get("responsibilities", ""), data.get("requirements", ""), data.get("skills", "")]
                return self._clean_text(" ".join([part for part in parts if part]))
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("ATS stage: failed to scrape URL %s: %s", text, exc)
        return self._clean_text(text)

    def _clean_text(self, value: str) -> str:
        text = unescape((value or "").strip())
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"(?i)([?&])(?:utm_[^&\s]+|gclid|fbclid|sid|px|seo_srp|ref|cid|mc_[^&\s]+)=[^&\s]+", " ", text)
        text = re.sub(r"\b(?:www|https|http|com|org|net|co|in|jobs|career|naukri|linkedin|indeed|glassdoor|lever|greenhouse)\b", " ", text, flags=re.I)
        text = re.sub(r"[^a-z0-9\s+.#/-]", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    def _looks_like_url(self, value: str) -> bool:
        return bool(re.match(r"^https?://", (value or "").strip(), re.I))