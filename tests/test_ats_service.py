from backend.services.ats_service import ATSService


def test_resume_skill_match_and_score_are_high_for_good_fit() -> None:
    service = ATSService()
    resume_text = "John Doe\nEmail: john@example.com\nPhone: +1 555 123 4567\nSkills: Python, React, SQL\nEducation: B.S. Computer Science\nExperience: 3 years of backend development\nProjects: Built a Python analytics platform"
    job_description = "Python React SQL"

    result = service.assess_resume(resume_text, job_description)

    assert result["matchedSkills"] == ["Python", "React", "SQL"]
    assert result["missingSkills"] == []
    assert result["atsScore"] >= 90


def test_resume_is_marked_not_suitable_for_unrelated_role() -> None:
    service = ATSService()
    resume_text = "Jane Smith\nEmail: jane@example.com\nPhone: +1 555 987 6543\nEducation: Mechanical Engineering\nExperience: 5 years in manufacturing\nProjects: Built assembly-line automation"
    job_description = "Java Developer role requiring Java Spring Boot Angular REST API SQL Git Microservices"

    result = service.assess_resume(resume_text, job_description)

    assert result["matchedSkills"] == []
    assert len(result["missingSkills"]) > 5
    assert result["atsScore"] < 20
    assert result["status"] == "Not Suitable"


def test_resume_frontend_stack_matches_react_role() -> None:
    service = ATSService()
    resume_text = "Alice Brown\nEmail: alice@example.com\nPhone: +1 555 111 2222\nSkills: HTML, CSS, JavaScript, React\nEducation: B.S. Information Technology\nExperience: 2 years of frontend work\nProjects: Built a React dashboard"
    job_description = "React Developer role"

    result = service.assess_resume(resume_text, job_description)

    assert set(["HTML", "CSS", "JavaScript", "React"]).issubset(set(result["matchedSkills"]))
    assert result["atsScore"] > 80


def test_recommendations_are_context_aware_for_java_full_stack_role() -> None:
    service = ATSService()
    resume_text = "John Doe\nEmail: john@example.com\nPhone: +1 555 123 4567\nSkills: HTML, CSS, JavaScript, React, SQL\nEducation: B.S. Computer Science\nExperience: 2 years of frontend work\nProjects: Built a React dashboard"
    job_description = "Java Full Stack Developer role requiring Java Spring Boot Angular REST API Maven Git Microservices"

    result = service.assess_resume(resume_text, job_description)

    flattened = " ".join(result["recommendations"] + result["learningRoadmap"] + result["recommendedJobs"]).lower()
    assert "java" in flattened or "spring" in flattened
    assert "fastapi" not in flattened
    assert "django" not in flattened
    assert "flask" not in flattened
    assert result["improvementSuggestions"]
    assert result["recommendedCourses"]
    assert result["recommendedProjects"]
    assert result["interviewPreparation"]
