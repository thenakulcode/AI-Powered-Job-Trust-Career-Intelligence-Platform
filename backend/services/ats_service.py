"""Deterministic ATS matching engine for resumes and job descriptions."""

from __future__ import annotations

import logging
import re
from html import unescape
from typing import Any

from backend.services.recommendation_engine import RecommendationEngine
from backend.services.resume_parser import SKILL_TERMS, normalize_skill_name, parse_resume_sections
from backend.services.scraper import extract_job_post

logger = logging.getLogger(__name__)


class ATSService:
    """Generate a deterministic ATS-style match score from resume and job data."""

    def __init__(self) -> None:
        self._skill_catalog = {skill.lower(): skill for skill in SKILL_TERMS}
        self._skill_catalog.update(
            {
                "html": "HTML",
                "css": "CSS",
                "javascript": "JavaScript",
                "react": "React",
                "sql": "SQL",
                "python": "Python",
                "java": "Java",
                "spring boot": "Spring Boot",
                "rest api": "REST API",
                "node.js": "Node.js",
                "angular": "Angular",
                "git": "Git",
                "docker": "Docker",
                "aws": "AWS",
                "azure": "Azure",
                "microservices": "Microservices",
                "postgres": "PostgreSQL",
                "mysql": "MySQL",
                "mongodb": "MongoDB",
                "fastapi": "FastAPI",
                "django": "Django",
                "flask": "Flask",
                "kubernetes": "Kubernetes",
                "linux": "Linux",
                "terraform": "Terraform",
                "agile": "Agile",
                "scrum": "Scrum",
                "devops": "DevOps",
                "machine learning": "Machine Learning",
            }
        )
        self._role_skill_map = {
            "react": ["React", "JavaScript", "HTML", "CSS"],
            "angular": ["Angular", "TypeScript", "HTML", "CSS"],
            "java": ["Java", "Spring Boot", "REST API", "SQL"],
            "python": ["Python", "SQL", "FastAPI", "Django", "Flask"],
            "node": ["Node.js", "JavaScript", "REST API"],
            "node.js": ["Node.js", "JavaScript", "REST API"],
            "spring boot": ["Spring Boot", "Java", "REST API"],
            "docker": ["Docker", "DevOps", "Linux"],
            "aws": ["AWS", "Cloud", "DevOps"],
            "azure": ["Azure", "Cloud", "DevOps"],
        }
        self._recommendation_engine = RecommendationEngine()

    def assess_resume(self, resume_text: str, job_description: str) -> dict[str, Any]:
        logger.info("------------------------------------------------")
        logger.info("ATS stage: starting resume assessment")
        if not resume_text or not resume_text.strip():
            return self._build_error("Unable to extract readable resume text.")

        parsed = parse_resume_sections(resume_text)
        if not self._is_readable_resume(parsed, resume_text):
            return self._build_error("This document does not contain a readable resume profile.")

        cleaned_job_text = self._prepare_job_description(job_description)
        resume_skills = self._extract_resume_skills(parsed, resume_text)
        job_skills = self._extract_job_skills(cleaned_job_text)
        job_keywords = self._extract_keywords(cleaned_job_text)
        resume_keywords = self._extract_keywords(resume_text)

        matched_skills = self._intersect(resume_skills, job_skills)
        missing_skills = self._difference(job_skills, resume_skills)
        extra_skills = self._difference(resume_skills, job_skills)
        matched_keywords = self._intersect(resume_keywords, job_keywords)
        missing_keywords = self._difference(job_keywords, resume_keywords)

        score_breakdown = self._build_score_breakdown(parsed, cleaned_job_text, resume_skills, job_skills, resume_keywords, job_keywords)
        ats_score = int(round(score_breakdown["ats_score"]))
        match_score = int(round(score_breakdown["match_score"]))
        status = "Suitable" if ats_score >= 60 else "Not Suitable"
        reason = self._build_reason(matched_skills, missing_skills, missing_keywords, parsed, cleaned_job_text)
        recommendation_payload = self._recommendation_engine.generate(missing_skills, cleaned_job_text, {
            "experience_level": parsed.get("total_experience", 0),
            "projects": parsed.get("projects", []),
            "education": parsed.get("education", []),
        })
        recommendations = recommendation_payload["improvementSuggestions"]
        learning_roadmap = recommendation_payload["recommendedProjects"]
        recommended_jobs = self._build_recommended_jobs(matched_skills, missing_skills)
        warning = self._build_warning(ats_score, status, reason)

        logger.info("Resume Skills: %s", [self._display_name(skill) for skill in resume_skills])
        logger.info("Job Skills: %s", [self._display_name(skill) for skill in job_skills])
        logger.info("Matched Skills: %s", [self._display_name(skill) for skill in matched_skills])
        logger.info("Missing Skills: %s", [self._display_name(skill) for skill in missing_skills])
        logger.info("Extra Skills: %s", [self._display_name(skill) for skill in extra_skills])
        logger.info("ATS Breakdown: %s", score_breakdown)
        logger.info("Final Score: %s", ats_score)
        logger.info("------------------------------------------------")

        return {
            "success": True,
            "atsScore": ats_score,
            "matchScore": match_score,
            "status": status,
            "matchedSkills": [self._display_name(skill) for skill in matched_skills],
            "missingSkills": [self._display_name(skill) for skill in missing_skills],
            "matchedKeywords": [self._display_name(keyword) for keyword in matched_keywords],
            "missingKeywords": [self._display_name(keyword) for keyword in missing_keywords],
            "extraSkills": [self._display_name(skill) for skill in extra_skills[:8]],
            "strengths": self._build_strengths(matched_skills, matched_keywords),
            "weaknesses": self._build_weaknesses(missing_skills, missing_keywords, parsed),
            "recommendations": recommendations,
            "learningRoadmap": learning_roadmap,
            "recommendedJobs": recommended_jobs,
            "warning": warning,
            "reason": reason,
            "improvementSuggestions": recommendation_payload["improvementSuggestions"],
            "recommendedCourses": recommendation_payload["recommendedCourses"],
            "recommendedProjects": recommendation_payload["recommendedProjects"],
            "interviewPreparation": recommendation_payload["interviewPreparation"],
            "learningRoadmap": recommendation_payload.get("learningRoadmap", learning_roadmap),
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
            "summary": self._build_summary(matched_skills, missing_skills, ats_score),
            "recommendationsList": recommendations,
            "learningRoadmapList": learning_roadmap,
            "recommendedJobsList": recommended_jobs,
        }

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
            "learningRoadmap": [],
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

    def _extract_resume_skills(self, parsed: dict[str, Any], resume_text: str) -> list[str]:
        candidates: list[str] = []
        for skill in parsed.get("skills", []) or []:
            candidates.extend(self._split_skill_candidates(skill))
        for section in [parsed.get("experience", []), parsed.get("projects", []), parsed.get("certifications", []), parsed.get("achievements", [])]:
            for item in section or []:
                candidates.extend(self._split_skill_candidates(item))
        text = " ".join([resume_text or "", *(parsed.get("education", []) or []), *(parsed.get("experience", []) or []), *(parsed.get("projects", []) or [])])
        for skill in self._skill_catalog:
            if self._contains_skill(text.lower(), skill):
                candidates.append(skill)
        normalized = []
        for candidate in candidates:
            normalized_skill = normalize_skill_name(candidate)
            if normalized_skill and normalized_skill.lower() in self._skill_catalog:
                normalized.append(normalized_skill)
        return self._unique_items(normalized)

    def _extract_job_skills(self, job_description: str) -> list[str]:
        cleaned = self._prepare_job_description(job_description)
        candidates: list[str] = []
        for skill in self._skill_catalog:
            if self._contains_skill(cleaned.lower(), skill):
                candidates.append(skill)
        if self._looks_like_role_description(cleaned):
            for role, inferred_skills in self._role_skill_map.items():
                if self._contains_skill(cleaned.lower(), role):
                    candidates.extend(inferred_skills)
        normalized = [normalize_skill_name(candidate) for candidate in candidates]
        normalized = [skill for skill in normalized if skill]
        return self._unique_items(normalized)[:10]

    def _extract_keywords(self, value: str) -> list[str]:
        cleaned = self._clean_text(value)
        candidates: list[str] = []
        for skill in self._skill_catalog:
            if self._contains_skill(cleaned.lower(), skill):
                candidates.append(skill)
        return self._unique_items([normalize_skill_name(candidate) for candidate in candidates])

    def _build_score_breakdown(
        self,
        parsed: dict[str, Any],
        job_description: str,
        resume_skills: list[str],
        job_skills: list[str],
        resume_keywords: list[str],
        job_keywords: list[str],
    ) -> dict[str, float]:
        matched_skills = self._intersect(resume_skills, job_skills)
        matched_keywords = self._intersect(resume_keywords, job_keywords)
        skills_score = self._safe_percent(len(matched_skills), max(len(job_skills), 1))
        keywords_score = self._safe_percent(len(matched_keywords), max(len(job_keywords), 1))
        experience_score = self._calculate_experience_score(parsed, job_description)
        projects_score = self._calculate_project_score(parsed, job_description, resume_skills)
        education_score = self._calculate_education_score(parsed, job_description)
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
        match_score = (
            skills_score * 0.35
            + keywords_score * 0.20
            + experience_score * 0.15
            + projects_score * 0.10
            + education_score * 0.10
            + certifications_score * 0.05
            + contact_score * 0.05
        )
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

    def _calculate_experience_score(self, parsed: dict[str, Any], job_description: str) -> float:
        resume_years = parsed.get("total_experience") or 0
        target_years = self._extract_years(job_description)
        experience_text = " ".join(parsed.get("experience", []) or [])
        lowered = (job_description + " " + experience_text).lower()
        has_related_terms = any(term in lowered for term in ["react", "angular", "java", "python", "sql", "api", "developer", "backend", "frontend", "engineer"])
        if target_years:
            if resume_years >= target_years:
                return 100.0
            return max(20.0, 50.0 + (resume_years / max(target_years, 1)) * 40.0)
        if resume_years >= 3:
            return 85.0 if has_related_terms else 45.0
        if resume_years >= 1:
            return 70.0 if has_related_terms else 30.0
        return 20.0

    def _calculate_project_score(self, parsed: dict[str, Any], job_description: str, resume_skills: list[str]) -> float:
        if not parsed.get("projects"):
            return 20.0
        lowered = (job_description + " " + " ".join(parsed.get("projects", []) or [])).lower()
        if any(skill.lower() in lowered for skill in resume_skills):
            return 100.0
        if any(term in lowered for term in ["react", "angular", "java", "python", "sql", "api", "backend", "frontend"]):
            return 80.0
        return 60.0

    def _calculate_education_score(self, parsed: dict[str, Any], job_description: str) -> float:
        if not parsed.get("education"):
            return 20.0
        lowered = (job_description + " " + " ".join(parsed.get("education", []) or [])).lower()
        if any(term in lowered for term in ["computer science", "information technology", "software", "engineering", "technology", "developer", "developer"]):
            return 90.0
        return 60.0

    def _calculate_certifications_score(self, parsed: dict[str, Any], job_description: str) -> float:
        if not parsed.get("certifications"):
            return 0.0
        lowered = (job_description + " " + " ".join(parsed.get("certifications", []) or [])).lower()
        if any(term in lowered for term in ["aws", "azure", "scrum", "cloud", "python", "java", "react"]):
            return 100.0
        return 40.0

    def _calculate_contact_score(self, parsed: dict[str, Any]) -> float:
        count = sum(1 for value in [parsed.get("email"), parsed.get("phone")] if value)
        return 100.0 if count == 2 else 60.0 if count == 1 else 20.0

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

    def _contains_skill(self, text: str, skill: str) -> bool:
        pattern = re.escape(skill).replace(r"\ ", r"(?:\\s|\\-|\\/)+")
        return bool(re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text))

    def _looks_like_role_description(self, text: str) -> bool:
        return bool(re.search(r"\b(developer|engineer|role|position|frontend|backend|full stack|fullstack|web|frontend|ui)\b", text))

    def _split_skill_candidates(self, value: str) -> list[str]:
        if not value:
            return []
        parts = re.split(r"[,;|/]+", str(value))
        return [part.strip() for part in parts if part.strip()]

    def _build_strengths(self, matched_skills: list[str], matched_keywords: list[str]) -> list[str]:
        strengths: list[str] = []
        if matched_skills:
            strengths.append(f"Strong overlap with {', '.join(self._display_name(skill) for skill in matched_skills[:3])}.")
        if matched_keywords:
            strengths.append("The resume text overlaps with several job keywords.")
        return strengths or ["The resume contains enough evidence to be evaluated."]

    def _build_weaknesses(self, missing_skills: list[str], missing_keywords: list[str], parsed: dict[str, Any]) -> list[str]:
        weaknesses: list[str] = []
        if missing_skills:
            weaknesses.append(f"Missing technologies: {', '.join(self._display_name(skill) for skill in missing_skills[:4])}.")
        if missing_keywords:
            weaknesses.append("Missing keyword coverage from the job description.")
        if not parsed.get("projects"):
            weaknesses.append("The resume does not include a strong project section.")
        return weaknesses or ["No obvious weaknesses detected from the extracted content."]

    def _build_recommendations(self, missing_skills: list[str]) -> list[str]:
        recommendations: list[str] = []
        for skill in missing_skills[:6]:
            key = normalize_skill_name(skill).lower()
            recommendations.append(self._advice_map.get(key, f"Practice {self._display_name(skill)} in a small project"))
        return self._unique_items(recommendations)

    def _build_learning_roadmap(self, missing_skills: list[str], parsed: dict[str, Any]) -> list[str]:
        roadmap: list[str] = []
        if not parsed.get("projects"):
            roadmap.append("Add one portfolio project that demonstrates your strongest technical work.")
        for skill in missing_skills[:4]:
            roadmap.append(f"Practice {self._display_name(skill)} in a small hands-on project.")
        return self._unique_items(roadmap)

    def _build_recommended_jobs(self, matched_skills: list[str], missing_skills: list[str]) -> list[str]:
        role_map = {
            "python": ["Python Developer", "Backend Developer"],
            "react": ["Frontend Developer", "React Developer"],
            "java": ["Java Developer", "Backend Developer"],
            "spring boot": ["Backend Developer", "Java Developer"],
            "sql": ["Data Engineer", "Backend Developer"],
            "aws": ["Cloud Engineer", "DevOps Engineer"],
            "azure": ["Cloud Engineer", "DevOps Engineer"],
            "docker": ["DevOps Engineer", "Platform Engineer"],
            "machine learning": ["ML Engineer", "AI Engineer"],
        }
        roles: list[str] = []
        for skill in [*matched_skills, *missing_skills]:
            roles.extend(role_map.get(normalize_skill_name(skill).lower(), []))
        return self._unique_items(roles)[:4] or ["Software Engineer", "Backend Developer"]

    def _build_reason(self, matched_skills: list[str], missing_skills: list[str], missing_keywords: list[str], parsed: dict[str, Any], job_description: str) -> str:
        if not matched_skills and missing_skills:
            return f"The resume does not show the required technologies ({', '.join(self._display_name(skill) for skill in missing_skills[:6])}) and does not demonstrate the experience requested by the role."
        if missing_keywords:
            return f"The resume is missing several job keywords and the required technologies ({', '.join(self._display_name(skill) for skill in missing_skills[:4])})."
        if not parsed.get("projects"):
            return "The resume lacks project evidence that supports the role." 
        return "The resume has a meaningful overlap with the role requirements."

    def _build_warning(self, ats_score: int, status: str, reason: str) -> str:
        if status == "Not Suitable" and ats_score < 40:
            return f"Not Suitable: {reason}"
        if status == "Not Suitable":
            return f"Not Suitable: {reason}"
        return ""

    def _build_summary(self, matched_skills: list[str], missing_skills: list[str], ats_score: int) -> str:
        if ats_score >= 80:
            return f"Strong fit with {len(matched_skills)} matched skills and an ATS score of {ats_score}."
        if ats_score >= 60:
            return f"Good fit with {len(matched_skills)} matched skills; add evidence for {', '.join(self._display_name(skill) for skill in missing_skills[:3])}."
        return f"This profile does not yet align with the role; prioritize {', '.join(self._display_name(skill) for skill in missing_skills[:3])}."

    def _display_name(self, value: str) -> str:
        normalized = normalize_skill_name(value)
        display_map = {
            "node.js": "Node.js",
            "rest api": "REST API",
            "ci/cd": "CI/CD",
            "power bi": "Power BI",
            "machine learning": "Machine Learning",
            "spring boot": "Spring Boot",
            "c++": "C++",
            "c#": "C#",
            "sql": "SQL",
            "html": "HTML",
            "css": "CSS",
            "javascript": "JavaScript",
            "python": "Python",
            "java": "Java",
            "react": "React",
            "angular": "Angular",
            "git": "Git",
            "docker": "Docker",
            "aws": "AWS",
            "azure": "Azure",
            "postgres": "PostgreSQL",
            "mysql": "MySQL",
            "mongodb": "MongoDB",
            "fastapi": "FastAPI",
            "django": "Django",
            "flask": "Flask",
            "kubernetes": "Kubernetes",
            "terraform": "Terraform",
            "devops": "DevOps",
            "agile": "Agile",
            "scrum": "Scrum",
            "microservices": "Microservices",
        }
        return display_map.get(normalized.lower(), normalized.title())

    def _safe_percent(self, part: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round((part / total) * 100, 2)

    def _extract_years(self, value: str) -> int | None:
        matches = re.findall(r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?|yr)", (value or "").lower())
        if not matches:
            return None
        return int(matches[0])

    def _intersect(self, left: list[str], right: list[str]) -> list[str]:
        left_set = {item.lower() for item in left}
        return [item for item in right if item.lower() in left_set]

    def _difference(self, left: list[str], right: list[str]) -> list[str]:
        right_set = {item.lower() for item in right}
        return [item for item in left if item.lower() not in right_set]

    def _unique_items(self, items: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = str(item).strip()
            if not value:
                continue
            key = normalize_skill_name(value).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(normalize_skill_name(value))
        return unique
