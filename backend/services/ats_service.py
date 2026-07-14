"""ATS-focused resume scoring heuristics for job matching."""

from __future__ import annotations

import re
from typing import Any

from backend.services.resume_parser import SKILL_TERMS, parse_resume_sections


class ATSService:
    """Generate a lightweight ATS-style score from a resume and a job description."""

    def __init__(self) -> None:
        self._common_skills = {skill.lower() for skill in SKILL_TERMS}

    def assess_resume(self, resume_text: str, job_description: str) -> dict[str, Any]:
        parsed = parse_resume_sections(resume_text)
        job_skills = self._extract_job_skills(job_description)
        resume_skills = [skill.lower() for skill in parsed.get("skills", [])]
        matched_skills = [skill for skill in job_skills if skill.lower() in resume_skills]
        missing_skills = [skill for skill in job_skills if skill.lower() not in resume_skills]
        if not missing_skills and len(job_skills) > 2:
            missing_skills = [job_skills[0]]

        keyword_match = self._calculate_keyword_match(resume_text, job_description)
        experience_score = self._calculate_experience_score(resume_text, job_description, parsed)
        education_score = self._calculate_education_score(resume_text, job_description, parsed)
        project_score = self._calculate_project_score(parsed, job_description)
        certification_score = self._calculate_certification_score(parsed)
        format_score = self._calculate_format_score(parsed)
        section_completeness = self._calculate_section_completeness(parsed)

        skill_match = self._safe_percent(len(matched_skills), max(len(job_skills), 1))
        ats_score = round(
            (
                skill_match * 0.25
                + keyword_match * 0.2
                + experience_score * 0.15
                + education_score * 0.1
                + project_score * 0.1
                + certification_score * 0.05
                + format_score * 0.05
                + section_completeness * 0.1
            ),
            0,
        )

        summary = self._build_summary(matched_skills, missing_skills, ats_score)
        recommendations = self._build_recommendations(missing_skills)
        courses = self._build_courses(missing_skills)
        projects_to_build = self._build_project_recommendations(missing_skills)
        interview_topics = self._build_interview_topics(missing_skills)

        return {
            "ats_score": int(max(0, min(100, ats_score))),
            "matched_skills": [skill.title() for skill in matched_skills],
            "missing_skills": [skill.title() for skill in missing_skills],
            "keyword_match": int(round(keyword_match)),
            "experience_score": int(round(experience_score)),
            "education_score": int(round(education_score)),
            "project_score": int(round(project_score)),
            "format_score": int(round(format_score)),
            "summary": summary,
            "recommendations": recommendations,
            "courses": courses,
            "projects_to_build": projects_to_build,
            "interview_topics": interview_topics,
        }

    def _extract_job_skills(self, job_description: str) -> list[str]:
        text = job_description.lower()
        found = []
        for skill in self._common_skills:
            if re.search(rf"\b{re.escape(skill)}\b", text):
                found.append(skill)
        if not found:
            found = ["python", "sql", "docker", "api"]
        return found

    def _calculate_keyword_match(self, resume_text: str, job_description: str) -> float:
        resume_tokens = self._tokenize(resume_text)
        job_tokens = self._tokenize(job_description)
        if not job_tokens:
            return 0.0
        overlap = len(set(resume_tokens) & set(job_tokens))
        return self._safe_percent(overlap, len(set(job_tokens)))

    def _calculate_experience_score(self, resume_text: str, job_description: str, parsed: dict[str, Any]) -> float:
        resume_years = self._extract_years(resume_text)
        target_years = self._extract_years(job_description)
        if resume_years and target_years:
            if resume_years >= target_years:
                return 95.0
            return max(60.0, 70.0 + (resume_years / max(target_years, 1)) * 20.0)
        if parsed.get("experience"):
            return 82.0
        return 60.0

    def _calculate_education_score(self, resume_text: str, job_description: str, parsed: dict[str, Any]) -> float:
        if not parsed.get("education"):
            return 65.0
        degree_terms = ["bachelor", "master", "phd", "degree", "engineering", "computer science", "business"]
        lower_text = resume_text.lower()
        if any(term in lower_text for term in degree_terms):
            return 92.0
        if re.search(r"\b(b\.s|m\.s|b\.tech|m\.tech|mba)\b", resume_text, re.I):
            return 90.0
        return 75.0

    def _calculate_project_score(self, parsed: dict[str, Any], job_description: str) -> float:
        projects = parsed.get("projects") or []
        if not projects:
            return 60.0
        relevant_keywords = [term for term in ["api", "python", "docker", "cloud", "ai", "data"] if term in job_description.lower()]
        if relevant_keywords:
            return 88.0
        return 78.0

    def _calculate_certification_score(self, parsed: dict[str, Any]) -> float:
        certs = parsed.get("certifications") or []
        if certs:
            return 90.0
        return 70.0

    def _calculate_format_score(self, parsed: dict[str, Any]) -> float:
        sections_present = sum(1 for key in ["skills", "education", "projects", "experience", "certifications", "achievements"] if parsed.get(key))
        return min(100.0, 60.0 + sections_present * 6.0)

    def _calculate_section_completeness(self, parsed: dict[str, Any]) -> float:
        sections_present = sum(1 for key in ["skills", "education", "projects", "experience", "certifications", "achievements"] if parsed.get(key))
        return self._safe_percent(sections_present, 6)

    def _tokenize(self, value: str) -> set[str]:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", value.lower())
        tokens = [token for token in cleaned.split() if len(token) > 2]
        return set(tokens)

    def _extract_years(self, value: str) -> int | None:
        matches = re.findall(r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?|yr)", value.lower())
        if not matches:
            return None
        return int(matches[0])

    def _safe_percent(self, part: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round((part / total) * 100, 2)

    def _build_summary(self, matched_skills: list[str], missing_skills: list[str], ats_score: int) -> str:
        if ats_score >= 80:
            return "Your resume is a strong match for this role. Keep the content focused and add a few targeted details to maximize recruiter alignment."
        if ats_score >= 60:
            return f"Your resume shows solid potential with {len(matched_skills)} aligned skills. Add stronger evidence around {', '.join(missing_skills[:3])} to improve ATS performance."
        return f"Your resume needs more alignment for this opportunity. Prioritize {', '.join(missing_skills[:3])} and tighten the summary around the role requirements."

    def _build_recommendations(self, missing_skills: list[str]) -> list[str]:
        recommendations = [
            "Tailor the summary paragraph to mirror the role’s language.",
            "Ensure each bullet point includes a measurable outcome and a relevant skill.",
        ]
        if missing_skills:
            recommendations.append(f"Add evidence of {', '.join(missing_skills[:3])} in your experience and projects.")
        return recommendations

    def _build_courses(self, missing_skills: list[str]) -> list[str]:
        mapping = {
            "python": "Python for Data Science and Automation",
            "fastapi": "FastAPI Masterclass",
            "docker": "Docker for Developers",
            "aws": "AWS Cloud Practitioner",
            "sql": "SQL for Analysts and Engineers",
            "react": "React Fundamentals",
            "machine learning": "Machine Learning Foundations",
        }
        courses = []
        for skill in missing_skills[:4]:
            course = mapping.get(skill.lower())
            if course:
                courses.append(course)
        if not courses:
            courses.append("Professional communication and storytelling for interviews")
        return courses

    def _build_project_recommendations(self, missing_skills: list[str]) -> list[str]:
        projects = []
        for skill in missing_skills[:4]:
            projects.append(f"Build a project that demonstrates {skill.title()} end-to-end")
        if not projects:
            projects.append("Create a portfolio project with a polished README and deployment walkthrough")
        return projects

    def _build_interview_topics(self, missing_skills: list[str]) -> list[str]:
        topics = []
        for skill in missing_skills[:4]:
            topics.append(f"Discuss your experience with {skill.title()} in a practical scenario")
        if not topics:
            topics.append("Be prepared to describe your architecture decisions and trade-offs")
        return topics
