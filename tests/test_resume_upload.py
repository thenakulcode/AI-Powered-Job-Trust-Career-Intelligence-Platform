from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_resume_ats_returns_400_when_file_is_missing() -> None:
    response = client.post(
        "/resume-ats",
        data={"job_description": "We are hiring a Senior Python Engineer"},
    )

    assert response.status_code == 400
    assert "resume file" in response.text.lower()
