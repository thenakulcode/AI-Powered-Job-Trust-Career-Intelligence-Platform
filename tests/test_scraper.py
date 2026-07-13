from backend.services.scraper import extract_job_data_from_html


def test_extract_job_data_from_html_collects_core_fields() -> None:
    html = """
    <html>
      <head>
        <title>Senior Python Engineer</title>
        <meta name="description" content="Build internal tools for an innovative company." />
      </head>
      <body>
        <h1>Senior Python Engineer</h1>
        <h2>Acme Labs</h2>
        <div>Remote · New York, NY</div>
        <section>
          <h3>Responsibilities</h3>
          <ul>
            <li>Build backend APIs</li>
            <li>Ship quality features</li>
          </ul>
        </section>
        <section>
          <h3>Requirements</h3>
          <ul>
            <li>Python experience</li>
            <li>FastAPI knowledge</li>
          </ul>
        </section>
        <section>
          <h3>Benefits</h3>
          <ul>
            <li>Health insurance</li>
          </ul>
        </section>
      </body>
    </html>
    """

    data = extract_job_data_from_html(html, "https://example.com/jobs/python")

    assert data["title"] == "Senior Python Engineer"
    assert data["company"] == "Acme Labs"
    assert data["location"] == "New York, NY"
    assert "Build backend APIs" in data["responsibilities"]
    assert "FastAPI knowledge" in data["requirements"]
    assert "Health insurance" in data["benefits"]
    assert data["description"].startswith("Build internal tools")
