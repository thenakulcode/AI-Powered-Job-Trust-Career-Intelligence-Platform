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

    toastContainer: document.getElementById("toastContainer"),
  };

  const GAUGE_CIRCUMFERENCE = 251; // matches the arc path length used in the SVG

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

  function refreshHistoryUI() {
    const history = getHistory();
    el.historyList.innerHTML = "";

    if (!history.length) {
      el.historyList.appendChild(el.historyEmpty);
      return;
    }

    history.forEach((entry) => {
      const isFraud = entry.prediction === "Fraudulent Job Post";
      const item = document.createElement("div");
      item.className = "history-item";
      item.innerHTML = `
        <div class="history-item__left">
          <span class="history-item__dot ${isFraud ? "history-item__dot--danger" : "history-item__dot--safe"}"></span>
          <span class="history-item__text">
            <span class="history-item__label">${isFraud ? "Fake Job" : "Real Job"}</span>
            <span class="history-item__snippet">${escapeHtml(entry.snippet || "No description")}</span>
          </span>
        </div>
        <div class="history-item__right">
          <span class="history-item__confidence">${((entry.confidence || 0) * 100).toFixed(0)}%</span>
          <span class="history-item__time">${formatRelativeTime(entry.timestamp)}</span>
        </div>
      `;
      el.historyList.appendChild(item);
    });
  }

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
    refreshHistoryUI();
    refreshStats();
    showToast("Scan history cleared.", "info", 2200);
  });

  function refreshStats() {
    const history = getHistory();
    const total = history.length;
    const fake = history.filter((h) => h.prediction === "Fraudulent Job Post").length;
    const real = total - fake;

    animateStatValue(el.statTotal, total);
    animateStatValue(el.statFake, fake);
    animateStatValue(el.statReal, real);
  }

  function animateStatValue(node, target) {
    const current = parseInt(node.textContent, 10) || 0;
    if (current === target) return;
    node.textContent = String(target);
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
  }

  document.addEventListener("DOMContentLoaded", init);
})();