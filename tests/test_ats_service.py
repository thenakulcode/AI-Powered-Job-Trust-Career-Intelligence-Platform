from backend.services.ats_service import ATSService


def test_resume_assessment_returns_expected_sections() -> None:
    service = ATSService()
    result = service.assess_resume(
        resume_text=(
            "John Doe\n"
            "Skills: Python, FastAPI, SQL, Docker\n"
            "Education: B.S. Computer Science\n"
            "Experience: 3 years building backend APIs and data pipelines\n"
            "Projects: Built a FastAPI service with Docker deployment\n"
            "Certifications: AWS Cloud Practitioner\n"
            "Achievements: Improved API latency by 35%\n"
        ),
        job_description=(
            "Senior Python Engineer. Requirements: Python, FastAPI, SQL, Docker, "
            "backend APIs, cloud deployment. 3 years experience. Bachelor degree."
        ),
    )

    assert result["ats_score"] >= 0
    assert result["ats_score"] <= 100
    assert result["matched_skills"]
    assert result["missing_skills"]
    assert result["summary"]
    assert result["recommendations"]
    assert result["courses"]
    assert result["projects_to_build"]
    assert result["interview_topics"]
