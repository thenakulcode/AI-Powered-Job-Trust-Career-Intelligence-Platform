/**
 * SENTINEL.AI — Fake Job Detection Dashboard
 * Vanilla JS controller: tabs, validation, API integration, result rendering,
 * local scan history, statistics, theme toggle, toasts.
 *
 * Backend contract (FastAPI):
 *   GET  /health        -> { status: "ok" }
 *   POST /predict-job    body: { job_description: string }
 *                         -> { prediction: "Fraudulent Job Post" | "Legitimate Job Post",
 *                               confidence: 0..1, risk_score: 0..100, risk_factors: string[] }
 */
(function () {
  "use strict";

  const API_PREDICT_URL = "/predict-job";
  const API_EXTRACT_URL = "/extract-job";
  const API_RESUME_ATS_URL = "/resume-ats";
  const API_HEALTH_URL = "/health";
  const HISTORY_KEY = "sentinel_scan_history";
  const THEME_KEY = "sentinel_theme";
  const MAX_HISTORY = 10;

  const HERO_SUBTITLE_TEXT =
    "Analyze job descriptions, URLs, and recruitment messages using AI to detect potentially fraudulent job postings.";

  /* Keyword map for the fixed threat-indicator chips shown in the UI. */
  const THREAT_INDICATOR_DEFS = [
    { id: "salary", label: "High Salary", iconClass: "fa-solid fa-sack-dollar", keywords: ["high salary", "$", "per week", "per day", "unlimited earning", "earn up to"] },
    { id: "experience", label: "No Experience", iconClass: "fa-solid fa-user-slash", keywords: ["no experience", "no experience required", "no experience needed"] },
    { id: "urgent", label: "Urgent Hiring", iconClass: "fa-solid fa-bolt", keywords: ["urgent", "immediately", "asap", "hiring fast", "fast hiring"] },
    { id: "telegram", label: "Telegram Contact", iconClass: "fa-brands fa-telegram", keywords: ["telegram"] },
    { id: "whatsapp", label: "WhatsApp Only", iconClass: "fa-brands fa-whatsapp", keywords: ["whatsapp"] },
    { id: "payment", label: "Payment Required", iconClass: "fa-solid fa-money-check-dollar", keywords: ["fee", "deposit", "wire", "bitcoin", "payment required", "registration fee", "processing fee"] },
    { id: "wfh", label: "Work From Home Scam", iconClass: "fa-solid fa-house-laptop", keywords: ["work from home", "guaranteed income", "easy money"] },
  ];

  /* ---------------------------------------------------------------------
   * DOM references
   * ------------------------------------------------------------------- */
  const el = {
    themeToggle: document.getElementById("themeToggle"),
    themeIcon: document.getElementById("themeIcon"),
    apiStatusDot: document.getElementById("apiStatusDot"),
    apiStatusText: document.getElementById("apiStatusText"),
    modelBackendStatusDot: document.getElementById("modelBackendStatusDot"),
    modelBackendStatusText: document.getElementById("modelBackendStatusText"),
    heroSubtitle: document.getElementById("heroSubtitle"),
    spotlightText: document.getElementById("spotlightText"),

    tabWebsite: document.getElementById("tabWebsite"),
    tabEmail: document.getElementById("tabEmail"),
    urlFieldGroup: document.getElementById("urlFieldGroup"),
    jobUrl: document.getElementById("jobUrl"),
    urlError: document.getElementById("urlError"),

    jobDescription: document.getElementById("jobDescription"),
    charCounter: document.getElementById("charCounter"),
    descError: document.getElementById("descError"),

    scanForm: document.getElementById("scanForm"),
    scanBtn: document.getElementById("scanBtn"),
    scanBtnLabel: document.getElementById("scanBtnLabel"),
    clearBtn: document.getElementById("clearBtn"),
    extractBtn: document.getElementById("extractBtn"),
    extractionSummary: document.getElementById("extractionSummary"),

    scanLoading: document.getElementById("scanLoading"),
    scanLoadingStep: document.getElementById("scanLoadingStep"),
    scanLoadingBarFill: document.getElementById("scanLoadingBarFill"),

    resultCard: document.getElementById("resultCard"),
    verdictBadge: document.getElementById("verdictBadge"),
    verdictIcon: document.getElementById("verdictIcon"),
    verdictText: document.getElementById("verdictText"),
    threatLevelPill: document.getElementById("threatLevelPill"),
    threatLevelPillText: document.getElementById("threatLevelPillText"),
    reportId: document.getElementById("reportId"),
    reportTimestamp: document.getElementById("reportTimestamp"),
    recommendationBlock: document.getElementById("recommendationBlock"),
    recommendationIcon: document.getElementById("recommendationIcon"),
    recommendationText: document.getElementById("recommendationText"),
    aiSummaryText: document.getElementById("aiSummaryText"),
    resumeAtsBlock: document.getElementById("resumeAtsBlock"),
    resumeUploadBlock: document.getElementById("resumeUploadBlock"),
    resumeFileInput: document.getElementById("resumeFileInput"),
    resumeUploadLabel: document.getElementById("resumeUploadLabel"),
    resumeFileName: document.getElementById("resumeFileName"),
    resumeAnalyzeBtn: document.getElementById("resumeAnalyzeBtn"),
    resumeUploadHint: document.getElementById("resumeUploadHint"),
    atsScoreValue: document.getElementById("atsScoreValue"),
    atsGaugeFill: document.getElementById("atsGaugeFill"),
    atsSkillMatchValue: document.getElementById("atsSkillMatchValue"),
    atsKeywordMatchValue: document.getElementById("atsKeywordMatchValue"),
    atsExperienceValue: document.getElementById("atsExperienceValue"),
    atsEducationValue: document.getElementById("atsEducationValue"),
    atsProjectValue: document.getElementById("atsProjectValue"),
    atsFormatValue: document.getElementById("atsFormatValue"),
    atsSummaryText: document.getElementById("atsSummaryText"),
    matchedSkillsList: document.getElementById("matchedSkillsList"),
    missingSkillsList: document.getElementById("missingSkillsList"),
    recommendationsList: document.getElementById("recommendationsList"),
    coursesList: document.getElementById("coursesList"),
    projectsList: document.getElementById("projectsList"),
    interviewList: document.getElementById("interviewList"),
    copyResultBtn: document.getElementById("copyResultBtn"),
    exportPdfBtn: document.getElementById("exportPdfBtn"),

    gaugeFill: document.getElementById("gaugeFill"),
    gaugeValue: document.getElementById("gaugeValue"),
    confidenceValue: document.getElementById("confidenceValue"),
    confidenceFill: document.getElementById("confidenceFill"),
    probabilityValue: document.getElementById("probabilityValue"),
    threatLevelValue: document.getElementById("threatLevelValue"),
    scanTimeValue: document.getElementById("scanTimeValue"),

    indicatorGrid: document.getElementById("indicatorGrid"),
    riskFactorsList: document.getElementById("riskFactorsList"),

    historyList: document.getElementById("historyList"),
    historyEmpty: document.getElementById("historyEmpty"),
    clearHistoryBtn: document.getElementById("clearHistoryBtn"),

    statTotal: document.getElementById("statTotal"),
    statFake: document.getElementById("statFake"),
    statReal: document.getElementById("statReal"),
    statAvgConfidence: document.getElementById("statAvgConfidence"),
    statTotalTrend: document.getElementById("statTotalTrend"),
    statFakeTrend: document.getElementById("statFakeTrend"),
    statRealTrend: document.getElementById("statRealTrend"),

    modelVersion: document.getElementById("modelVersion"),
    modelThreshold: document.getElementById("modelThreshold"),

    historySearchInput: document.getElementById("historySearchInput"),
    historyFilterSelect: document.getElementById("historyFilterSelect"),
    historyPagination: document.getElementById("historyPagination"),
    historyPaginationLabel: document.getElementById("historyPaginationLabel"),
    historyPrevBtn: document.getElementById("historyPrevBtn"),
    historyNextBtn: document.getElementById("historyNextBtn"),

    detectionRatioChartCanvas: document.getElementById("detectionRatioChart"),
    dailyScansChartCanvas: document.getElementById("dailyScansChart"),
    confidenceScoresChartCanvas: document.getElementById("confidenceScoresChart"),

    toastContainer: document.getElementById("toastContainer"),
  };

  const GAUGE_CIRCUMFERENCE = 251; // matches the arc path length used in the SVG
  let selectedResumeFile = null;

  /* ---------------------------------------------------------------------
   * Button ripple micro-interaction
   * ------------------------------------------------------------------- */
  function initRipples() {
    const targets = document.querySelectorAll(".btn-gradient, .btn-ghost, .btn");
    targets.forEach((btn) => {
      btn.addEventListener("click", (event) => {
        if (btn.disabled) return;
        const rect = btn.getBoundingClientRect();
        const ripple = document.createElement("span");
        const size = Math.max(rect.width, rect.height);
        ripple.className = "ripple";
        ripple.style.width = ripple.style.height = `${size}px`;
        ripple.style.left = `${(event.clientX ?? rect.left + rect.width / 2) - rect.left - size / 2}px`;
        ripple.style.top = `${(event.clientY ?? rect.top + rect.height / 2) - rect.top - size / 2}px`;
        btn.appendChild(ripple);
        ripple.addEventListener("animationend", () => ripple.remove(), { once: true });
      });
    });
  }

  function updateResumeUploadUi() {
    const hasFile = Boolean(selectedResumeFile && selectedResumeFile.name);
    const hasDescription = Boolean(el.jobDescription.value.trim());

    el.resumeAnalyzeBtn.disabled = !(hasFile && hasDescription);
    el.resumeFileName.textContent = hasFile ? `Selected: ${selectedResumeFile.name}` : "No file selected";
    el.resumeUploadLabel.textContent = hasFile ? "Resume ready to upload" : "Upload PDF or DOCX resume";

    if (!hasFile) {
      el.resumeUploadHint.textContent = "Upload a PDF or DOCX resume to evaluate its ATS alignment.";
      return;
    }

    el.resumeUploadHint.textContent = `Ready to analyze ${selectedResumeFile.name}.`;
  }

  /* ---------------------------------------------------------------------
   * Theme
   * ------------------------------------------------------------------- */
  function initTheme() {
    const saved = localStorage.getItem(THEME_KEY) || "dark";
    applyTheme(saved);
  }

  function applyTheme(theme) {
    document.body.setAttribute("data-theme", theme);
    el.themeIcon.className = theme === "dark" ? "fa-solid fa-moon" : "fa-solid fa-sun";
    localStorage.setItem(THEME_KEY, theme);
  }

  el.themeToggle.addEventListener("click", () => {
    const current = document.body.getAttribute("data-theme");
    applyTheme(current === "dark" ? "light" : "dark");
    window.setTimeout(refreshCharts, 50);
  });

  el.resumeFileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
    selectedResumeFile = file;
    console.log("Resume file selected:", file);

    if (!file) {
      updateResumeUploadUi();
      return;
    }

    const allowed = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
    if (!allowed.includes(file.type) && !file.name.toLowerCase().endsWith(".pdf") && !file.name.toLowerCase().endsWith(".docx")) {
      showToast("Please choose a PDF or DOCX file.", "warning");
      selectedResumeFile = null;
      updateResumeUploadUi();
      return;
    }

    updateResumeUploadUi();
    showToast(`Selected ${file.name}.`, "success", 2000);
  });

  el.jobDescription.addEventListener("input", () => {
    updateResumeUploadUi();
  });

  /* ---------------------------------------------------------------------
   * Backend health check
   * ------------------------------------------------------------------- */
  async function checkApiStatus() {
    try {
      const res = await fetch(API_HEALTH_URL, { method: "GET" });
      if (!res.ok) throw new Error("Backend responded with an error status.");
      el.apiStatusDot.className = "status-pill__dot status-pill__dot--online";
      el.apiStatusText.textContent = "Backend online";
      el.modelBackendStatusDot.className = "status-tag__dot status-tag__dot--online";
      el.modelBackendStatusText.textContent = "Online";
    } catch (err) {
      el.apiStatusDot.className = "status-pill__dot status-pill__dot--offline";
      el.apiStatusText.textContent = "Backend unavailable";
      el.modelBackendStatusDot.className = "status-tag__dot status-tag__dot--offline";
      el.modelBackendStatusText.textContent = "Unavailable";
    }
  }

  /* ---------------------------------------------------------------------
   * Hero subtitle typing animation
   * ------------------------------------------------------------------- */
  function typeHeroSubtitle() {
    const cursor = document.createElement("span");
    cursor.className = "typed-cursor";
    cursor.textContent = "\u00A0";
    el.heroSubtitle.textContent = "";
    el.heroSubtitle.appendChild(cursor);

    let i = 0;
    const speed = 16;

    function step() {
      if (i <= HERO_SUBTITLE_TEXT.length) {
        el.heroSubtitle.textContent = HERO_SUBTITLE_TEXT.slice(0, i);
        el.heroSubtitle.appendChild(cursor);
        i += 1;
        window.setTimeout(step, speed);
      } else {
        window.setTimeout(() => cursor.remove(), 900);
      }
    }
    step();
  }

  /* ---------------------------------------------------------------------
   * Tabs — Website / Link vs Email / Text
   * ------------------------------------------------------------------- */
  function setActiveTab(mode) {
    const isWebsite = mode === "website";
    el.tabWebsite.classList.toggle("tab--active", isWebsite);
    el.tabWebsite.setAttribute("aria-selected", String(isWebsite));
    el.tabEmail.classList.toggle("tab--active", !isWebsite);
    el.tabEmail.setAttribute("aria-selected", String(!isWebsite));
    el.urlFieldGroup.hidden = !isWebsite;
    if (!isWebsite) {
      clearFieldError(el.jobUrl, el.urlError);
    }
  }

  el.tabWebsite.addEventListener("click", () => setActiveTab("website"));
  el.tabEmail.addEventListener("click", () => setActiveTab("email"));

  /* ---------------------------------------------------------------------
   * Character counter
   * ------------------------------------------------------------------- */
  el.jobDescription.addEventListener("input", () => {
    const len = el.jobDescription.value.length;
    el.charCounter.textContent = `${len.toLocaleString()} character${len === 1 ? "" : "s"}`;
    if (len > 0) clearFieldError(el.jobDescription, el.descError);
  });

  /* ---------------------------------------------------------------------
   * Field error helpers
   * ------------------------------------------------------------------- */
  function showFieldError(input, errorEl, message) {
    input.classList.add("is-invalid");
    errorEl.textContent = message;
    errorEl.classList.add("is-visible");
  }

  function clearFieldError(input, errorEl) {
    input.classList.remove("is-invalid");
    errorEl.textContent = "";
    errorEl.classList.remove("is-visible");
  }

  function isLikelyValidUrl(value) {
    try {
      const url = new URL(value);
      return url.protocol === "http:" || url.protocol === "https:";
    } catch {
      return false;
    }
  }

  /* ---------------------------------------------------------------------
   * Toasts
   * ------------------------------------------------------------------- */
  const TOAST_ICONS = {
    success: "fa-solid fa-circle-check",
    error: "fa-solid fa-circle-exclamation",
    warning: "fa-solid fa-triangle-exclamation",
    info: "fa-solid fa-circle-info",
  };

  function showToast(message, type = "info", duration = 4200) {
    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `<i class="${TOAST_ICONS[type] || TOAST_ICONS.info}"></i><span>${escapeHtml(message)}</span>`;
    el.toastContainer.appendChild(toast);

    window.setTimeout(() => {
      toast.classList.add("is-leaving");
      toast.addEventListener("animationend", () => toast.remove(), { once: true });
    }, duration);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  /* ---------------------------------------------------------------------
   * Extraction flow
   * ------------------------------------------------------------------- */
  async function extractJobFromUrl(url) {
    if (!url) {
      showToast("Enter a job posting URL to fetch the details.", "warning");
      return null;
    }

    if (!isLikelyValidUrl(url)) {
      showFieldError(el.jobUrl, el.urlError, "That doesn't look like a valid URL (include https://).");
      showToast("The job URL looks invalid. Double-check it and try again.", "warning");
      return null;
    }

    clearFieldError(el.jobUrl, el.urlError);
    setExtractionLoading(true);

    try {
      const response = await fetch(API_EXTRACT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, timeout: 20 }),
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = payload && payload.detail ? payload.detail : "Unable to extract job details from this website.";
        throw new Error(detail);
      }

      if (!payload || !payload.success) {
        throw new Error("Unable to extract job description from this website.");
      }

      el.jobDescription.value = payload.description || "";
      el.charCounter.textContent = `${el.jobDescription.value.length.toLocaleString()} characters`;
      renderExtractionSummary(payload);
      showToast(`Fetched ${payload.title || "job details"} from the posting.`, "success");
      return payload;
    } catch (err) {
      console.error("Extraction failed:", err);
      renderExtractionSummary({ error: err.message || "Unable to extract job description from this website." });
      showToast(err.message || "Unable to extract job description from this website.", "error", 6000);
      return null;
    } finally {
      setExtractionLoading(false);
    }
  }

  function setExtractionLoading(isLoading) {
    el.extractBtn.disabled = isLoading;
    el.extractBtn.classList.toggle("is-loading", isLoading);
    const label = el.extractBtn.querySelector("i");
    if (label) {
      label.className = isLoading ? "fa-solid fa-spinner fa-spin" : "fa-solid fa-download";
    }
  }

  function renderExtractionSummary(payload) {
    if (!payload || payload.error) {
      el.extractionSummary.hidden = false;
      el.extractionSummary.className = "extraction-summary extraction-summary--error";
      el.extractionSummary.innerHTML = `<strong>Extraction failed.</strong> ${escapeHtml(payload && payload.error ? payload.error : "Unable to extract job description from this website.")}`;
      return;
    }

    const title = payload.title || "Untitled role";
    const company = payload.company || "Unknown company";
    el.extractionSummary.hidden = false;
    el.extractionSummary.className = "extraction-summary";
    el.extractionSummary.innerHTML = `<strong>${escapeHtml(title)}</strong> at <span>${escapeHtml(company)}</span>`;
  }

  el.extractBtn.addEventListener("click", async () => {
    const url = el.jobUrl.value.trim();
    const payload = await extractJobFromUrl(url);
    if (payload) {
      await runScan({ description: el.jobDescription.value.trim(), url });
    }
  });

  /* ---------------------------------------------------------------------
   * Form submission — validation + API call
   * ------------------------------------------------------------------- */
  el.scanForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const isWebsiteMode = el.tabWebsite.classList.contains("tab--active");
    const description = el.jobDescription.value.trim();
    const url = el.jobUrl.value.trim();

    let hasError = false;

    if (!description) {
      showFieldError(el.jobDescription, el.descError, "Paste a job description before scanning.");
      hasError = true;
    } else {
      clearFieldError(el.jobDescription, el.descError);
    }

    if (isWebsiteMode && url && !isLikelyValidUrl(url)) {
      showFieldError(el.jobUrl, el.urlError, "That doesn't look like a valid URL (include https://).");
      showToast("The job URL looks invalid. Double-check it or leave it blank.", "warning");
      hasError = true;
    } else {
      clearFieldError(el.jobUrl, el.urlError);
    }

    if (hasError) return;

    await runScan({ description, url: isWebsiteMode ? url : "" });
  });

  async function runScan({ description, url }) {
    if (!description) {
      showToast("Paste or fetch a job description before scanning.", "warning");
      return;
    }

    setLoading(true);

    const startedAt = performance.now();

    try {
      const response = await fetch(API_PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_description: description }),
      });

      if (!response.ok) {
        let detail = "The scan could not be completed.";
        try {
          const errBody = await response.json();
          if (errBody && errBody.detail) detail = errBody.detail;
        } catch {
          /* ignore parse errors on error body */
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const elapsedMs = Math.round(performance.now() - startedAt);

      renderResult(data, { description, url, elapsedMs });
      saveToHistory(data, { description, url });
      refreshHistoryUI();
      refreshStats();
      refreshCharts();

      showToast(
        data.prediction === "Fraudulent Job Post"
          ? "Scan complete — this posting shows fraud signals."
          : "Scan complete — this posting looks legitimate.",
        data.prediction === "Fraudulent Job Post" ? "warning" : "success"
      );
    } catch (err) {
      console.error("Prediction request failed:", err);
      showToast(
        "Couldn't reach the detection service. Confirm the backend is running and try again.",
        "error",
        6000
      );
      el.apiStatusDot.className = "status-pill__dot status-pill__dot--offline";
      el.apiStatusText.textContent = "Backend unavailable";
      el.modelBackendStatusDot.className = "status-tag__dot status-tag__dot--offline";
      el.modelBackendStatusText.textContent = "Unavailable";
    } finally {
      setLoading(false);
    }
  }

  const LOADING_STEPS = [
    "Cleaning input text",
    "Extracting TF-IDF features",
    "Running XGBoost inference",
    "Calculating confidence score",
  ];
  let loadingStepTimer = null;

  function setLoading(isLoading) {
    el.scanBtn.disabled = isLoading;
    el.scanBtn.classList.toggle("is-loading", isLoading);
    el.scanBtnLabel.textContent = isLoading ? "Scanning..." : "Scan for Threats";

    // Disable inputs while a scan is in flight so nothing changes mid-request.
    el.jobDescription.disabled = isLoading;
    el.jobUrl.disabled = isLoading;
    el.tabWebsite.disabled = isLoading;
    el.tabEmail.disabled = isLoading;
    el.clearBtn.disabled = isLoading;

    if (isLoading) {
      // Hide any previous result while a fresh scan runs, and show the skeleton.
      el.resultCard.hidden = true;
      el.scanLoading.hidden = false;
      el.scanLoading.scrollIntoView({ behavior: "smooth", block: "nearest" });

      let stepIndex = 0;
      el.scanLoadingStep.textContent = LOADING_STEPS[0];
      el.scanLoadingBarFill.style.width = "0%";
      window.clearInterval(loadingStepTimer);
      loadingStepTimer = window.setInterval(() => {
        stepIndex = Math.min(stepIndex + 1, LOADING_STEPS.length - 1);
        el.scanLoadingStep.textContent = LOADING_STEPS[stepIndex];
      }, 550);
    } else {
      window.clearInterval(loadingStepTimer);
      el.scanLoading.hidden = true;
    }
  }

  /* ---------------------------------------------------------------------
   * Result rendering
   * ------------------------------------------------------------------- */
  function renderResult(data, context) {
    const isFraud = data.prediction === "Fraudulent Job Post";
    const riskScore = clamp(data.risk_score ?? 0, 0, 100);
    const confidencePct = clamp((data.confidence ?? 0) * 100, 0, 100);

    el.resumeAtsBlock.hidden = true;
    el.resumeUploadBlock.hidden = isFraud;
    el.resumeUploadHint.textContent = isFraud
      ? "Resume analysis is only available when the job appears legitimate."
      : "Upload a PDF or DOCX resume to evaluate its ATS alignment.";

    el.resultCard.hidden = false;
    // Restart the reveal animation each time so the report re-animates in.
    el.resultCard.style.animation = "none";
    void el.resultCard.offsetWidth;
    el.resultCard.style.animation = "";
    el.resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Verdict badge
    el.verdictBadge.className = `verdict ${isFraud ? "verdict--danger" : "verdict--safe"}`;
    el.verdictIcon.className = isFraud ? "fa-solid fa-triangle-exclamation" : "fa-solid fa-circle-check";
    el.verdictText.textContent = isFraud ? "Fake Job" : "Real Job";

    // Threat level pill (color-coded low/medium/high)
    const levelKey = threatLevelLabel(riskScore).toLowerCase();
    el.threatLevelPill.className = `threat-pill threat-pill--${levelKey}`;
    el.threatLevelPillText.textContent = `${threatLevelLabel(riskScore)} Threat`;

    // Report meta
    el.reportId.textContent = Math.random().toString(36).slice(2, 8).toUpperCase();
    el.reportTimestamp.textContent = new Date().toLocaleString();

    // Gauge (risk score)
    const gaugeColor = riskScore >= 65 ? "var(--danger)" : riskScore >= 35 ? "var(--warn)" : "var(--safe)";
    el.gaugeFill.style.stroke = gaugeColor;
    const offset = GAUGE_CIRCUMFERENCE - (riskScore / 100) * GAUGE_CIRCUMFERENCE;
    requestAnimationFrame(() => {
      el.gaugeFill.style.strokeDashoffset = String(offset);
    });
    el.gaugeValue.textContent = `${riskScore.toFixed(0)}%`;

    // Confidence bar
    el.confidenceValue.textContent = `${confidencePct.toFixed(1)}%`;
    requestAnimationFrame(() => {
      el.confidenceFill.style.width = `${confidencePct}%`;
    });

    // Metric tiles
    el.probabilityValue.textContent = `${riskScore.toFixed(1)}%`;
    const level = threatLevelLabel(riskScore);
    el.threatLevelValue.textContent = level;
    el.threatLevelValue.className =
      `metric-tile__value metric-tile__value--${level.toLowerCase()}`;
    el.scanTimeValue.textContent = `${context.elapsedMs} ms`;

    // Threat indicator chips
    renderThreatIndicators(context.description);

    // Model risk factors (from backend)
    renderRiskFactors(data.risk_factors || []);

    // Recommendation + AI summary (derived from the model's actual output)
    try {
      const recommendation = buildRecommendation(isFraud, riskScore);
      el.recommendationBlock.className = `recommendation recommendation--${recommendation.tone}`;
      el.recommendationIcon.className = recommendation.icon;
      el.recommendationText.textContent = recommendation.text;
      el.aiSummaryText.textContent = buildAiSummary(data, { isFraud, riskScore, confidencePct });

      // Stash for copy/export
      el.resultCard.dataset.lastResult = JSON.stringify({
        prediction: data.prediction,
        riskScore,
        confidencePct,
        threatLevel: threatLevelLabel(riskScore),
        recommendation: recommendation.text,
        summary: el.aiSummaryText.textContent,
        scannedAt: new Date().toISOString(),
      });
    } catch (err) {
      // Don't let a malformed response silently blank these two sections —
      // show something and log the real cause to the console.
      console.error("Failed to build recommendation/summary:", err, data);
      el.recommendationBlock.className = "recommendation";
      el.recommendationIcon.className = "fa-solid fa-circle-info";
      el.recommendationText.textContent = "Recommendation unavailable — check the console for details.";
      el.aiSummaryText.textContent = "Summary unavailable — check the console for details.";
    }

    if (!isFraud) {
      el.resumeUploadBlock.hidden = false;
      el.resumeAtsBlock.hidden = true;
      el.resumeUploadHint.textContent = "Upload a PDF or DOCX resume to evaluate its ATS alignment.";
    }
  }

  /* ---------------------------------------------------------------------
   * Recommendation + AI summary copy (derived from real model output only)
   * ------------------------------------------------------------------- */
  function buildRecommendation(isFraud, riskScore) {
    if (isFraud || riskScore >= 65) {
      return {
        tone: "danger",
        icon: "fa-solid fa-ban",
        text: "Do not apply or share personal information. This posting shows strong fraud signals — report it to the platform and avoid any contact requesting payment or ID documents.",
      };
    }
    if (riskScore >= 35) {
      return {
        tone: "medium",
        icon: "fa-solid fa-triangle-exclamation",
        text: "Proceed with caution. Verify the company's legitimacy independently — check its official site, LinkedIn presence, and recruiter identity before sharing sensitive details.",
      };
    }
    return {
      tone: "safe",
      icon: "fa-solid fa-circle-check",
      text: "No major fraud signals detected. Standard due diligence is still recommended before sharing sensitive personal or financial information.",
    };
  }

  function buildAiSummary(data, { isFraud, riskScore, confidencePct }) {
    const factors = Array.isArray(data.risk_factors) ? data.risk_factors : [];
    const level = threatLevelLabel(riskScore).toLowerCase();
    const verdictPhrase = isFraud ? "flagged this posting as likely fraudulent" : "classified this posting as likely legitimate";
    const confidencePhrase = `with ${confidencePct.toFixed(1)}% model confidence`;
    const factorPhrase = factors.length
      ? ` Key signals contributing to this assessment include ${factors.slice(0, 3).join(", ")}.`
      : " No strong red-flag language was detected in the text.";
    return `The model ${verdictPhrase} ${confidencePhrase}, placing it in the ${level} threat tier (risk score ${riskScore.toFixed(0)}/100).${factorPhrase}`;
  }

  function threatLevelLabel(riskScore) {
    if (riskScore >= 65) return "High";
    if (riskScore >= 35) return "Medium";
    return "Low";
  }

  function renderThreatIndicators(rawText) {
    const lower = (rawText || "").toLowerCase();
    el.indicatorGrid.innerHTML = "";

    THREAT_INDICATOR_DEFS.forEach((def) => {
      const active = def.keywords.some((kw) => lower.includes(kw));
      const chip = document.createElement("div");
      chip.className = `indicator-chip ${active ? "indicator-chip--active" : ""}`;
      chip.innerHTML = `<i class="${def.iconClass}"></i><span>${def.label}</span>`;
      el.indicatorGrid.appendChild(chip);
    });
  }

  function renderRiskFactors(factors) {
    el.riskFactorsList.innerHTML = "";
    if (!factors.length) {
      const li = document.createElement("li");
      li.innerHTML = `<i class="fa-solid fa-circle-info"></i><span>No strong red-flag language detected.</span>`;
      el.riskFactorsList.appendChild(li);
      return;
    }
    factors.forEach((factor) => {
      const li = document.createElement("li");
      li.innerHTML = `<i class="fa-solid fa-caret-right"></i><span>${escapeHtml(factor)}</span>`;
      el.riskFactorsList.appendChild(li);
    });
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function renderAtsResults(payload) {
    const score = clamp(Number(payload.atsScore ?? payload.ats_score ?? 0), 0, 100);
    const radius = 48;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;

    el.atsScoreValue.textContent = score;
    el.atsGaugeFill.style.strokeDasharray = `${circumference}`;
    el.atsGaugeFill.style.strokeDashoffset = `${offset}`;
    el.atsGaugeFill.style.stroke = score >= 80 ? "var(--safe)" : score >= 60 ? "var(--warn)" : "var(--danger)";

    const breakdown = payload.scoreBreakdown || {};
    el.atsSkillMatchValue.textContent = `${Math.round(Number(breakdown.skills ?? payload.skill_match ?? payload.matchScore ?? 0))}%`;
    el.atsKeywordMatchValue.textContent = `${Math.round(Number(breakdown.keywords ?? payload.keyword_match ?? 0))}%`;
    el.atsExperienceValue.textContent = `${Math.round(Number(breakdown.experience ?? payload.experience_score ?? 0))}%`;
    el.atsEducationValue.textContent = `${Math.round(Number(breakdown.education ?? payload.education_score ?? 0))}%`;
    el.atsProjectValue.textContent = `${Math.round(Number(breakdown.projects ?? payload.project_score ?? 0))}%`;
    el.atsFormatValue.textContent = `${Math.round(Number(breakdown.formatting ?? payload.format_score ?? 0))}%`;

    const summaryParts = [
      payload.summary || payload.message || "Resume summary unavailable.",
      payload.status ? `Status: ${payload.status}` : "",
      payload.matchScore != null ? `Match Score: ${payload.matchScore}%` : "",
      payload.warning ? `Warning: ${payload.warning}` : "",
    ].filter(Boolean);
    el.atsSummaryText.textContent = summaryParts.join(" \n");

    renderChipList(el.matchedSkillsList, payload.matchedSkills || payload.matched_skills || [], "chip-item", false);
    renderChipList(el.missingSkillsList, payload.missingSkills || payload.missing_skills || [], "chip-item chip-item--muted", true);
    renderList(el.recommendationsList, payload.recommendations || payload.recommendationsList || []);
    renderList(el.coursesList, payload.learningRoadmap || payload.courses || []);
    renderList(el.projectsList, payload.recommendedJobs || payload.projects_to_build || []);
    renderList(el.interviewList, payload.recommendations || payload.interview_topics || []);

    el.resumeAtsBlock.hidden = false;
  }

  function renderChipList(container, items, baseClass, isMuted) {
    container.innerHTML = "";
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.innerHTML = '<i class="fa-solid fa-circle-info"></i><p>No items detected.</p>';
      container.appendChild(empty);
      return;
    }
    items.forEach((item) => {
      const chip = document.createElement("span");
      chip.className = baseClass;
      chip.textContent = item;
      container.appendChild(chip);
    });
  }

  function renderList(container, items) {
    container.innerHTML = "";
    if (!items.length) {
      const li = document.createElement("li");
      li.textContent = "No suggestions available.";
      container.appendChild(li);
      return;
    }
    items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      container.appendChild(li);
    });
  }

  async function analyzeResume() {
    const file = selectedResumeFile || (el.resumeFileInput.files && el.resumeFileInput.files[0]);
    const description = el.jobDescription.value.trim();
    console.log("Resume analysis started", { file, description });

    if (!file) {
      showToast("Choose a PDF or DOCX resume before analyzing.", "warning");
      return;
    }
    if (!description) {
      showToast("Add a job description or fetch one before analyzing the resume.", "warning");
      return;
    }

    const formData = new FormData();
    formData.append("job_description", description);
    formData.append("file", file, file.name);

    console.log("Resume upload payload", {
      job_description: description,
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type,
    });

    try {
      const response = await fetch(API_RESUME_ATS_URL, {
        method: "POST",
        body: formData,
      });

      console.log("Resume ATS response status:", response.status);
      const responseText = await response.text();
      console.log("Resume ATS response body:", responseText);

      let payload = {};
      try {
        payload = responseText ? JSON.parse(responseText) : {};
      } catch (err) {
        console.error("Failed to parse ATS response JSON:", err);
      }

      if (!response.ok) {
        throw new Error(payload && payload.detail ? payload.detail : "Resume analysis failed.");
      }

      renderAtsResults(payload);
      showToast("Resume ATS analysis complete.", "success");
    } catch (err) {
      console.error("Resume analysis failed:", err);
      showToast(err.message || "Resume analysis failed.", "error", 6000);
    }
  }

  el.resumeAnalyzeBtn.addEventListener("click", analyzeResume);

  /* ---------------------------------------------------------------------
   * Copy / Export
   * ------------------------------------------------------------------- */
  el.copyResultBtn.addEventListener("click", async () => {
    const raw = el.resultCard.dataset.lastResult;
    if (!raw) return;
    const result = JSON.parse(raw);
    const summary = [
      `Sentinel.AI Scan Result`,
      `Verdict: ${result.prediction}`,
      `Fraud Risk: ${result.riskScore.toFixed(0)}%`,
      `Confidence: ${result.confidencePct.toFixed(1)}%`,
      `Threat Level: ${result.threatLevel}`,
      `Recommendation: ${result.recommendation}`,
      `Summary: ${result.summary}`,
      `Scanned: ${new Date(result.scannedAt).toLocaleString()}`,
    ].join("\n");

    try {
      await navigator.clipboard.writeText(summary);
      showToast("Result copied to clipboard.", "success");
    } catch {
      showToast("Couldn't copy automatically — select and copy the result manually.", "warning");
    }
  });

  el.exportPdfBtn.addEventListener("click", () => {
    if (el.resultCard.hidden) return;
    showToast("Opening print dialog — choose \u201cSave as PDF\u201d.", "info");
    window.setTimeout(() => window.print(), 200);
  });

  /* ---------------------------------------------------------------------
   * Clear form
   * ------------------------------------------------------------------- */
  el.clearBtn.addEventListener("click", () => {
    el.jobDescription.value = "";
    el.jobUrl.value = "";
    el.charCounter.textContent = "0 characters";
    clearFieldError(el.jobDescription, el.descError);
    clearFieldError(el.jobUrl, el.urlError);
    el.extractionSummary.hidden = true;
    el.extractionSummary.innerHTML = "";
    el.resultCard.hidden = true;
    el.scanLoading.hidden = true;
    showToast("Form cleared.", "info", 2200);
  });

  /* ---------------------------------------------------------------------
   * Local scan history + statistics (LocalStorage)
   * ------------------------------------------------------------------- */
  function getHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function saveToHistory(data, context) {
    const history = getHistory();
    const entry = {
      prediction: data.prediction,
      confidence: data.confidence,
      riskScore: data.risk_score,
      snippet: (context.description || "").slice(0, 80),
      timestamp: new Date().toISOString(),
    };
    history.unshift(entry);
    const trimmed = history.slice(0, MAX_HISTORY);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  }

  const HISTORY_PAGE_SIZE = 5;
  let historyPage = 1;

  function getFilteredHistory() {
    const query = (el.historySearchInput.value || "").trim().toLowerCase();
    const filter = el.historyFilterSelect.value;

    return getHistory().filter((entry) => {
      const isFraud = entry.prediction === "Fraudulent Job Post";
      if (filter === "fake" && !isFraud) return false;
      if (filter === "real" && isFraud) return false;
      if (query && !(entry.snippet || "").toLowerCase().includes(query)) return false;
      return true;
    });
  }

  function refreshHistoryUI() {
    const filtered = getFilteredHistory();
    const totalPages = Math.max(1, Math.ceil(filtered.length / HISTORY_PAGE_SIZE));
    historyPage = Math.min(historyPage, totalPages);

    el.historyList.innerHTML = "";

    if (!filtered.length) {
      const empty = el.historyEmpty.cloneNode(true);
      if (getHistory().length && !filtered.length) {
        empty.querySelector("p").textContent = "No scans match your search or filter.";
      } else {
        empty.querySelector("p").textContent = "No scans yet. Run your first analysis to see it here.";
      }
      el.historyList.appendChild(empty);
      el.historyPagination.hidden = true;
      return;
    }

    const start = (historyPage - 1) * HISTORY_PAGE_SIZE;
    const pageItems = filtered.slice(start, start + HISTORY_PAGE_SIZE);

    pageItems.forEach((entry) => {
      const isFraud = entry.prediction === "Fraudulent Job Post";
      const item = document.createElement("div");
      item.className = "history-item";
      item.innerHTML = `
        <div class="history-item__left">
          <span class="history-item__icon ${isFraud ? "history-item__icon--danger" : "history-item__icon--safe"}">
            <i class="fa-solid ${isFraud ? "fa-triangle-exclamation" : "fa-circle-check"}"></i>
          </span>
          <span class="history-item__text">
            <span class="history-item__label">${isFraud ? "Fake Job" : "Real Job"}</span>
            <span class="history-item__snippet">${escapeHtml(entry.snippet || "No description")}</span>
          </span>
        </div>
        <div class="history-item__right">
          <span class="history-item__confidence">${((entry.confidence || 0) * 100).toFixed(0)}%</span>
          <span class="history-item__time">${formatRelativeTime(entry.timestamp)}</span>
          <span class="history-item__badge ${isFraud ? "history-item__badge--danger" : "history-item__badge--safe"}">${isFraud ? "Fake" : "Real"}</span>
        </div>
      `;
      el.historyList.appendChild(item);
    });

    if (filtered.length > HISTORY_PAGE_SIZE) {
      el.historyPagination.hidden = false;
      el.historyPaginationLabel.textContent = `Page ${historyPage} of ${totalPages}`;
      el.historyPrevBtn.disabled = historyPage <= 1;
      el.historyNextBtn.disabled = historyPage >= totalPages;
    } else {
      el.historyPagination.hidden = true;
    }
  }

  el.historySearchInput.addEventListener("input", () => {
    historyPage = 1;
    refreshHistoryUI();
  });
  el.historyFilterSelect.addEventListener("change", () => {
    historyPage = 1;
    refreshHistoryUI();
  });
  el.historyPrevBtn.addEventListener("click", () => {
    historyPage = Math.max(1, historyPage - 1);
    refreshHistoryUI();
  });
  el.historyNextBtn.addEventListener("click", () => {
    historyPage += 1;
    refreshHistoryUI();
  });

  function formatRelativeTime(isoString) {
    const date = new Date(isoString);
    const diffMs = Date.now() - date.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return "just now";
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    return date.toLocaleDateString();
  }

  el.clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    historyPage = 1;
    refreshHistoryUI();
    refreshStats();
    refreshCharts();
    showToast("Scan history cleared.", "info", 2200);
  });

  function refreshStats() {
    const history = getHistory();
    const total = history.length;
    const fake = history.filter((h) => h.prediction === "Fraudulent Job Post").length;
    const real = total - fake;
    const avgConfidence = total
      ? history.reduce((sum, h) => sum + (h.confidence || 0), 0) / total
      : null;

    animateStatValue(el.statTotal, total);
    animateStatValue(el.statFake, fake);
    animateStatValue(el.statReal, real);
    el.statAvgConfidence.textContent = avgConfidence === null ? "\u2014" : `${(avgConfidence * 100).toFixed(1)}%`;

    setTrend(el.statFakeTrend, fake, real, true);
    setTrend(el.statRealTrend, real, fake, false);
    el.statTotalTrend.textContent = total ? `${total} logged` : "\u2014";
    el.statTotalTrend.className = "stat-card__trend stat-card__trend--flat";
  }

  function setTrend(node, value, otherValue, isRisk) {
    if (!value && !otherValue) {
      node.textContent = "\u2014";
      node.className = "stat-card__trend stat-card__trend--flat";
      return;
    }
    const share = value / Math.max(1, value + otherValue);
    const pct = Math.round(share * 100);
    const goodDirection = isRisk ? share <= 0.5 : share >= 0.5;
    node.innerHTML = `<i class="fa-solid ${goodDirection ? "fa-arrow-down" : "fa-arrow-up"}"></i> ${pct}%`;
    node.className = `stat-card__trend ${goodDirection ? "stat-card__trend--up" : "stat-card__trend--down"}`;
  }

  /* Count-up animation for the big stat numbers. */
  function animateStatValue(node, target) {
    const current = parseInt(node.textContent, 10) || 0;
    if (current === target) {
      node.textContent = String(target);
      return;
    }
    const duration = 500;
    const startTime = performance.now();
    const from = current;

    function tick(now) {
      const progress = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = Math.round(from + (target - from) * eased);
      node.textContent = String(value);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ---------------------------------------------------------------------
   * Analytics charts (Chart.js) — derived entirely from local history
   * ------------------------------------------------------------------- */
  let detectionRatioChart = null;
  let dailyScansChart = null;
  let confidenceScoresChart = null;

  function chartColors() {
    const styles = getComputedStyle(document.body);
    return {
      primary: styles.getPropertyValue("--primary").trim() || "#6d5df6",
      danger: styles.getPropertyValue("--danger").trim() || "#ef4444",
      success: styles.getPropertyValue("--success").trim() || "#22c55e",
      text: styles.getPropertyValue("--text-secondary").trim() || "#a1a1aa",
      grid: styles.getPropertyValue("--border").trim() || "rgba(250,250,250,0.08)",
    };
  }

  function refreshCharts() {
    if (typeof Chart === "undefined") return; // CDN blocked/offline — charts simply don't render
    const history = getHistory();
    const colors = chartColors();

    Chart.defaults.color = colors.text;
    Chart.defaults.font.family = "Inter, sans-serif";

    // ---- Detection ratio (pie) ----
    const fakeCount = history.filter((h) => h.prediction === "Fraudulent Job Post").length;
    const realCount = history.length - fakeCount;
    if (detectionRatioChart) detectionRatioChart.destroy();
    detectionRatioChart = new Chart(el.detectionRatioChartCanvas, {
      type: "doughnut",
      data: {
        labels: ["Fake", "Real"],
        datasets: [{
          data: history.length ? [fakeCount, realCount] : [1],
          backgroundColor: history.length ? [colors.danger, colors.success] : [colors.grid],
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: { legend: { position: "bottom", labels: { boxWidth: 10, padding: 14 } } },
      },
    });

    // ---- Daily scans (line, last 7 days) ----
    const days = [];
    for (let i = 6; i >= 0; i -= 1) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      days.push(d);
    }
    const dayCounts = days.map((day) =>
      history.filter((h) => {
        const t = new Date(h.timestamp);
        return t.toDateString() === day.toDateString();
      }).length
    );
    if (dailyScansChart) dailyScansChart.destroy();
    dailyScansChart = new Chart(el.dailyScansChartCanvas, {
      type: "line",
      data: {
        labels: days.map((d) => d.toLocaleDateString(undefined, { weekday: "short" })),
        datasets: [{
          data: dayCounts,
          borderColor: colors.primary,
          backgroundColor: "rgba(109, 93, 246, 0.18)",
          fill: true,
          tension: 0.35,
          pointRadius: 3,
          pointBackgroundColor: colors.primary,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: colors.grid } },
        },
      },
    });

    // ---- Confidence scores (bar, most recent scans) ----
    const recent = history.slice(0, 8).reverse();
    if (confidenceScoresChart) confidenceScoresChart.destroy();
    confidenceScoresChart = new Chart(el.confidenceScoresChartCanvas, {
      type: "bar",
      data: {
        labels: recent.map((_, i) => `#${history.length - recent.length + i + 1}`),
        datasets: [{
          data: recent.map((h) => Math.round((h.confidence || 0) * 100)),
          backgroundColor: recent.map((h) =>
            h.prediction === "Fraudulent Job Post" ? colors.danger : colors.primary
          ),
          borderRadius: 6,
          maxBarThickness: 28,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, max: 100, grid: { color: colors.grid } },
        },
      },
    });
  }

  /* ---------------------------------------------------------------------
   * Spotlight brand text — cursor-tracking gradient reveal, with a slow
   * ambient drift while idle so it's never static.
   * ------------------------------------------------------------------- */
  function initSpotlightText() {
    const textEl = el.spotlightText;
    if (!textEl) return;

    // Track pointer movement across the whole banner (generous hit area),
    // not just the tight bounding box of the letters themselves — otherwise
    // moving the cursor near the text but not directly over a glyph does nothing.
    const zone = textEl.closest(".spotlight-banner") || textEl;

    let ambientRaf = null;
    let hovering = false;

    function setSpot(xPct, yPct) {
      textEl.style.setProperty("--spot-x", `${xPct}%`);
      textEl.style.setProperty("--spot-y", `${yPct}%`);
    }

    function ambientLoop(time) {
      if (hovering) return;
      const t = time / 3800;
      const x = 50 + Math.sin(t) * 32;
      const y = 45 + Math.cos(t * 0.7) * 30;
      setSpot(x, y);
      ambientRaf = requestAnimationFrame(ambientLoop);
    }

    function updateFromEvent(event) {
      hovering = true;
      if (ambientRaf) {
        cancelAnimationFrame(ambientRaf);
        ambientRaf = null;
      }
      // Position is expressed relative to the TEXT element's box (not the
      // zone), so the gradient still lines up with the letters even though
      // we're listening on a larger surrounding area.
      const rect = textEl.getBoundingClientRect();
      const xPct = clamp(((event.clientX - rect.left) / rect.width) * 100, -40, 140);
      const yPct = clamp(((event.clientY - rect.top) / rect.height) * 100, -60, 160);
      setSpot(xPct, yPct);
    }

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    zone.addEventListener("pointermove", updateFromEvent);
    zone.addEventListener("pointerenter", updateFromEvent);

    zone.addEventListener("pointerleave", () => {
      hovering = false;
      if (!ambientRaf) ambientRaf = requestAnimationFrame(ambientLoop);
    });

    ambientRaf = requestAnimationFrame(ambientLoop);
  }

  /* ---------------------------------------------------------------------
   * Init
   * ------------------------------------------------------------------- */
  function init() {
    initTheme();
    checkApiStatus();
    typeHeroSubtitle();
    setActiveTab("website");
    refreshHistoryUI();
    refreshStats();
    initRipples();
    initSpotlightText();
    window.setTimeout(refreshCharts, 0); // wait a tick for Chart.js (deferred) to be ready
  }

  document.addEventListener("DOMContentLoaded", init);
})();