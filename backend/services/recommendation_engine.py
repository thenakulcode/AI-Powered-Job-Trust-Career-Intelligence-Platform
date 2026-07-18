"""Context-aware recommendation generation for ATS matching."""

from __future__ import annotations

import re
from typing import Any

from backend.services.resume_parser import normalize_skill_name


class RecommendationEngine:
    """Generate job-specific improvement suggestions, courses, projects, and interview prep."""

    def __init__(self) -> None:
        self._skill_templates: dict[str, dict[str, list[str]]] = {
            "java": {
                "suggestions": [
                    "Learn Core Java OOP, Collections, Streams and Multithreading.",
                    "Solve 50 Java DSA problems and build one console-based core Java project.",
                ],
                "courses": ["Oracle Java Tutorials", "Java Programming Masterclass - Udemy"],
                "projects": ["Library Management System", "Banking Application", "Student Management System"],
                "interview": ["Java OOP", "JVM", "Collections", "Streams", "Exception Handling", "Multithreading"],
            },
            "spring boot": {
                "suggestions": [
                    "Learn Spring Boot REST APIs, Spring Security and JPA + Hibernate.",
                    "Build one Employee Management System or Inventory Management API.",
                ],
                "courses": ["Spring Official Documentation", "Amigoscode Spring Boot"],
                "projects": ["Employee Management System", "Inventory Management System", "E-Commerce Backend"],
                "interview": ["Spring Boot", "REST API", "Hibernate", "JPA", "Dependency Injection", "Spring Security"],
            },
            "rest api": {
                "suggestions": [
                    "Design REST endpoints with clean resource modeling and authentication.",
                    "Practice pagination, validation and error handling for APIs.",
                ],
                "courses": ["REST API Design - MDN", "REST API Tutorial"],
                "projects": ["Task Management API", "Order Management API", "User Service API"],
                "interview": ["REST API", "Authentication", "Pagination", "Idempotency", "Versioning"],
            },
            "angular": {
                "suggestions": [
                    "Learn Angular Components, Routing, Services and RxJS.",
                    "Convert an existing React UI into an Angular dashboard.",
                ],
                "courses": ["Angular Official Documentation", "Angular Complete Guide"],
                "projects": ["Admin Dashboard", "Task Management App", "Inventory Dashboard"],
                "interview": ["Angular Components", "Routing", "RxJS", "Dependency Injection", "Lifecycle Hooks"],
            },
            "react": {
                "suggestions": [
                    "Build a React dashboard with hooks, props and state management.",
                    "Practice component composition and modern React patterns.",
                ],
                "courses": ["React Official Documentation", "The Road to React"],
                "projects": ["Expense Tracker", "Job Portal", "Chat Application"],
                "interview": ["React Hooks", "Props vs State", "Component Lifecycle", "State Management"],
            },
            "node.js": {
                "suggestions": [
                    "Build a Node.js service with Express and REST endpoints.",
                    "Practice middleware, routing and async error handling.",
                ],
                "courses": ["Node.js Official Documentation", "Node.js Tutorial"],
                "projects": ["Task Manager", "Chat Application", "Job Portal"],
                "interview": ["Node.js", "Express", "Middleware", "Async Programming", "Event Loop"],
            },
            "docker": {
                "suggestions": [
                    "Learn Docker basics and containerize one backend project.",
                    "Practice Dockerfiles, images and container networking.",
                ],
                "courses": ["Docker Official Documentation", "Docker for Beginners"],
                "projects": ["Containerized Spring Boot App", "Containerized Task Service"],
                "interview": ["Docker", "Dockerfile", "Container Networking", "Images vs Containers"],
            },
            "aws": {
                "suggestions": [
                    "Deploy one project on AWS EC2 or AWS App Runner.",
                    "Practice IAM, S3 and environment configuration for deployment.",
                ],
                "courses": ["AWS Skill Builder", "AWS Official Training"],
                "projects": ["Hosted REST API on AWS", "Static Site Deployment on S3"],
                "interview": ["AWS EC2", "IAM", "S3", "Deployment", "Cloud Security"],
            },
            "git": {
                "suggestions": [
                    "Practice branching, merging, rebasing and pull requests.",
                    "Use GitHub workflows in one collaborative project.",
                ],
                "courses": ["GitHub Skills", "Pro Git"],
                "projects": ["Collaborative Feature Branch Project", "Open Source Contribution Workflow"],
                "interview": ["Git Branching", "Merge Conflicts", "Pull Requests", "Rebasing"],
            },
            "maven": {
                "suggestions": [
                    "Learn Maven lifecycle, dependency management and plugins.",
                    "Package one Spring Boot project with Maven.",
                ],
                "courses": ["Apache Maven Documentation", "Maven Crash Course"],
                "projects": ["Spring Boot Project with Maven Build", "Multi-module Backend Project"],
                "interview": ["Maven Lifecycle", "pom.xml", "Dependency Management", "Plugins"],
            },
            "microservices": {
                "suggestions": [
                    "Break one project into multiple services with clear APIs.",
                    "Practice service communication, logging and failure handling.",
                ],
                "courses": ["Microservices Architecture Guide", "Building Microservices"],
                "projects": ["Order Service + Payment Service", "User Service + Inventory Service"],
                "interview": ["Microservices", "API Gateway", "Service Communication", "Failure Handling"],
            },
            "sql": {
                "suggestions": [
                    "Practice SQL joins, indexes and transaction design.",
                    "Build one normalized database-backed application.",
                ],
                "courses": ["SQLBolt", "W3Schools SQL Tutorial"],
                "projects": ["Inventory Database", "Sales Analytics Dashboard", "Library Database"],
                "interview": ["SQL Joins", "Normalization", "Indexes", "Transactions"],
            },
            "python": {
                "suggestions": [
                    "Build one Python backend project and practice clean module structure.",
                    "Solve script-based automation and API problems.",
                ],
                "courses": ["Python Official Documentation", "Python for Everybody"],
                "projects": ["Resume Analyzer", "Spam Detection", "House Price Prediction"],
                "interview": ["Python Basics", "Functions", "Modules", "Error Handling"],
            },
            "machine learning": {
                "suggestions": [
                    "Train one small machine learning model and evaluate its accuracy.",
                    "Practice feature engineering and model selection.",
                ],
                "courses": ["Machine Learning by Andrew Ng", "Hands-On Machine Learning"],
                "projects": ["House Price Prediction", "Spam Detection", "Resume Analyzer"],
                "interview": ["Bias-Variance", "Model Evaluation", "Feature Engineering", "Overfitting"],
            },
        }

    def generate(self, missing_skills: list[str], job_description: str, resume_context: dict[str, Any] | None = None) -> dict[str, list[str]]:
        missing = self._normalize_skills(missing_skills)
        job_text = self._normalize_text(job_description or "")
        context = self._infer_context(job_text, missing)

        ranked_missing = self._rank_missing_skills(missing, job_text, context)
        improvement_suggestions: list[str] = []
        recommended_courses: list[str] = []
        recommended_projects: list[str] = []
        interview_preparation: list[str] = []

        for skill in ranked_missing:
            if not self._should_include_skill(skill, job_text, context):
                continue
            template = self._skill_templates.get(skill.lower(), self._fallback_template(skill, context))
            improvement_suggestions.extend(template.get("suggestions", []))
            recommended_courses.extend(template.get("courses", []))
            recommended_projects.extend(template.get("projects", []))
            interview_preparation.extend(template.get("interview", []))

        return {
            "improvementSuggestions": self._dedupe(improvement_suggestions)[:8],
            "recommendedCourses": self._dedupe(recommended_courses)[:8],
            "recommendedProjects": self._dedupe(recommended_projects)[:8],
            "interviewPreparation": self._dedupe(interview_preparation)[:10],
            "learningRoadmap": self._build_learning_roadmap(ranked_missing, context),
        }

    def _infer_context(self, job_text: str, missing_skills: list[str]) -> dict[str, Any]:
        lowered = job_text.lower()
        return {
            "role": "java" if re.search(r"\b(java|spring|backend|full stack)\b", lowered) else "react" if re.search(r"\b(react|node|mern|frontend)\b", lowered) else "python" if re.search(r"\b(python|machine learning|data science|ai)\b", lowered) else "general",
            "keywords": [token for token in missing_skills if token.lower() in self._skill_templates],
        }

    def _rank_missing_skills(self, missing_skills: list[str], job_text: str, context: dict[str, Any]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[a-z0-9.+/#]+", job_text.lower()):
            normalized = normalize_skill_name(token)
            if not normalized or normalized.lower() not in self._skill_templates:
                continue
            if normalized.lower() in missing_skills and normalized.lower() not in seen:
                ordered.append(normalized)
                seen.add(normalized.lower())
        for skill in missing_skills:
            normalized = normalize_skill_name(skill)
            if normalized and normalized.lower() not in seen:
                ordered.append(normalized)
                seen.add(normalized.lower())
        return ordered

    def _should_include_skill(self, skill: str, job_text: str, context: dict[str, Any]) -> bool:
        normalized = normalize_skill_name(skill).lower()
        if not normalized:
            return False
        if normalized in {"fastapi", "django", "flask"}:
            return bool(re.search(r"\b(python|fastapi|django|flask)\b", job_text))
        if context.get("role") == "java" and normalized in {"fastapi", "django", "flask", "pandas", "numpy"}:
            return False
        if context.get("role") == "react" and normalized in {"spring boot", "java", "maven", "microservices"}:
            return False
        if context.get("role") == "python" and normalized in {"angular", "spring boot", "maven"}:
            return False
        return normalized in self._skill_templates

    def _build_learning_roadmap(self, missing_skills: list[str], context: dict[str, Any]) -> list[str]:
        roadmap: list[str] = []
        if not missing_skills:
            return ["Week 1: Review core fundamentals and build one small project."]
        for index, skill in enumerate(missing_skills[:6], start=1):
            label = normalize_skill_name(skill)
            if label.lower() == "java":
                roadmap.append(f"Week {index}: Core Java OOP, Collections and Streams")
            elif label.lower() == "spring boot":
                roadmap.append(f"Week {index}: Spring Boot REST APIs and Security")
            elif label.lower() == "angular":
                roadmap.append(f"Week {index}: Angular Components, Routing and RxJS")
            elif label.lower() == "react":
                roadmap.append(f"Week {index}: React hooks, state and component design")
            elif label.lower() == "docker":
                roadmap.append(f"Week {index}: Docker basics and containerization")
            elif label.lower() == "aws":
                roadmap.append(f"Week {index}: AWS deployment and cloud basics")
            elif label.lower() == "git":
                roadmap.append(f"Week {index}: Git workflows and collaboration")
            elif label.lower() == "sql":
                roadmap.append(f"Week {index}: SQL joins, indexes and schema design")
            else:
                roadmap.append(f"Week {index}: Build depth around {label}")
        roadmap.append("Final Week: Deploy one end-to-end project for your portfolio")
        return roadmap

    def _fallback_template(self, skill: str, context: dict[str, Any]) -> dict[str, list[str]]:
        return {
            "suggestions": [f"Study {skill} through hands-on practice and build one project around the topic."],
            "courses": [f"{skill.title()} learning resources"],
            "projects": [f"Prototype project using {skill}"],
            "interview": [f"Explain the core concepts behind {skill}"],
        }

    def _dedupe(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._normalize_text(value)
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            unique.append(normalized)
        return unique

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _normalize_skills(self, skills: list[str]) -> list[str]:
        normalized: list[str] = []
        for skill in skills:
            value = normalize_skill_name(skill)
            if value:
                normalized.append(value)
        return normalized
