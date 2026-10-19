/**
 * FraudWatch — Financial Fraud Detection Platform
 * app.js: UI behaviors, navigation, interactive scoring, CSV parser, live polling, and case management.
 */

(function (window, document, FraudWatchAPI) {
  "use strict";

  // Application State
  const state = {
    activeView: "overview",
    blockThreshold: 0.90,
    reviewThreshold: 0.50,
    sessionScoredTransactions: [],
    cases: [],
    caseMetrics: { total: 0, open: 0, investigating: 0, confirmed_fraud: 0, false_positive: 0 },
    monitoringAlerts: [],
    platformStatus: null,
    apiHealth: null,
    monitoringPollTimer: null,
    riskChart: null,
    currentTheme: localStorage.getItem("fraudwatch_theme") || "light",
  };

  // DOM Cache
  const DOM = {};

  function initDomCache() {
    DOM.sidebar = document.getElementById("appSidebar");
    DOM.btnToggleSidebar = document.getElementById("btnToggleSidebar");
    DOM.pageTitle = document.getElementById("pageTitle");
    DOM.pageSubtitle = document.getElementById("pageSubtitle");
    DOM.navItems = document.querySelectorAll(".nav-item");
    DOM.viewSections = document.querySelectorAll(".view-section");

    // Header controls
    DOM.sidebarStatusDot = document.getElementById("sidebarStatusDot");
    DOM.sidebarStatusText = document.getElementById("sidebarStatusText");
    DOM.sidebarLatency = document.getElementById("sidebarLatency");
    DOM.sidebarPolicyText = document.getElementById("sidebarPolicyText");
    DOM.sidebarAlertBadge = document.getElementById("sidebarAlertBadge");
    DOM.sidebarCaseBadge = document.getElementById("sidebarCaseBadge");
    DOM.headerThresholdText = document.getElementById("headerThresholdText");
    DOM.btnRefreshData = document.getElementById("btnRefreshData");
    DOM.btnToggleTheme = document.getElementById("btnToggleTheme");
    DOM.themeIconSun = document.getElementById("themeIconSun");
    DOM.themeIconMoon = document.getElementById("themeIconMoon");

    // Overview elements
    DOM.heroBtnAnalyze = document.getElementById("heroBtnAnalyze");
    DOM.heroBtnBatch = document.getElementById("heroBtnBatch");
    DOM.kpiTotalScored = document.getElementById("kpiTotalScored");
    DOM.kpiHighRisk = document.getElementById("kpiHighRisk");
    DOM.kpiMediumRisk = document.getElementById("kpiMediumRisk");
    DOM.kpiLowRisk = document.getElementById("kpiLowRisk");
    DOM.kpiModelStatus = document.getElementById("kpiModelStatus");
    DOM.kpiModelDetails = document.getElementById("kpiModelDetails");
    DOM.recentTransactionsBody = document.getElementById("recentTransactionsBody");
    DOM.noChartDataMsg = document.getElementById("noChartDataMsg");
    DOM.riskDistributionCanvas = document.getElementById("riskDistributionChart");

    // Single Scoring elements
    DOM.presetFraud = document.getElementById("presetFraud");
    DOM.presetLegit = document.getElementById("presetLegit");
    DOM.presetLive = document.getElementById("presetLive");
    DOM.presetReview = document.getElementById("presetReview");
    DOM.singleTxForm = document.getElementById("singleTxForm");
    DOM.modeBtnHistorical = document.getElementById("modeBtnHistorical");
    DOM.modeBtnLive = document.getElementById("modeBtnLive");
    DOM.modeHint = document.getElementById("modeHint");
    DOM.groupNewOrig = document.getElementById("groupNewOrig");
    DOM.groupNewDest = document.getElementById("groupNewDest");
    DOM.txStep = document.getElementById("txStep");
    DOM.stepTimeHint = document.getElementById("stepTimeHint");
    DOM.txId = document.getElementById("txId");
    DOM.txType = document.getElementById("txType");
    DOM.txAmount = document.getElementById("txAmount");
    DOM.txNameOrig = document.getElementById("txNameOrig");
    DOM.txOldOrig = document.getElementById("txOldOrig");
    DOM.txNewOrig = document.getElementById("txNewOrig");
    DOM.txNameDest = document.getElementById("txNameDest");
    DOM.txOldDest = document.getElementById("txOldDest");
    DOM.txNewDest = document.getElementById("txNewDest");
    DOM.txUseFeast = document.getElementById("txUseFeast");
    DOM.btnScoreTx = document.getElementById("btnScoreTx");
    DOM.btnResetForm = document.getElementById("btnResetForm");

    // Result panel
    DOM.resultIdleState = document.getElementById("resultIdleState");
    DOM.resultActiveState = document.getElementById("resultActiveState");
    DOM.resultStatusBadge = document.getElementById("resultStatusBadge");
    DOM.gaugeFillPath = document.getElementById("gaugeFillPath");
    DOM.resFraudProbability = document.getElementById("resFraudProbability");
    DOM.resRecommendedAction = document.getElementById("resRecommendedAction");
    DOM.resRiskBadge = document.getElementById("resRiskBadge");
    DOM.resActionBanner = document.getElementById("resActionBanner");
    DOM.resDrainRatio = document.getElementById("resDrainRatio");
    DOM.resBalanceDiscrepancy = document.getElementById("resBalanceDiscrepancy");
    DOM.resFeatureSource = document.getElementById("resFeatureSource");
    DOM.resFeastStatus = document.getElementById("resFeastStatus");
    DOM.resAuditBox = document.getElementById("resAuditBox");
    DOM.resAuditNote = document.getElementById("resAuditNote");
    DOM.resRawJson = document.getElementById("resRawJson");

    // Batch CSV elements
    DOM.csvDropzone = document.getElementById("csvDropzone");
    DOM.csvFileInput = document.getElementById("csvFileInput");
    DOM.fileUploadInfo = document.getElementById("fileUploadInfo");
    DOM.uploadFileName = document.getElementById("uploadFileName");
    DOM.uploadFileMeta = document.getElementById("uploadFileMeta");
    DOM.btnSubmitBatch = document.getElementById("btnSubmitBatch");
    DOM.batchValErrors = document.getElementById("batchValErrors");
    DOM.batchValErrorsList = document.getElementById("batchValErrorsList");
    DOM.btnDownloadHistTemplate = document.getElementById("btnDownloadHistTemplate");
    DOM.btnDownloadLiveTemplate = document.getElementById("btnDownloadLiveTemplate");
    DOM.batchResultsCard = document.getElementById("batchResultsCard");
    DOM.batchSummarySubtitle = document.getElementById("batchSummarySubtitle");
    DOM.batchTotalCount = document.getElementById("batchTotalCount");
    DOM.batchLatency = document.getElementById("batchLatency");
    DOM.batchBlockCount = document.getElementById("batchBlockCount");
    DOM.batchReviewCount = document.getElementById("batchReviewCount");
    DOM.batchApproveCount = document.getElementById("batchApproveCount");
    DOM.batchSearchInput = document.getElementById("batchSearchInput");
    DOM.batchRiskFilter = document.getElementById("batchRiskFilter");
    DOM.batchResultsBody = document.getElementById("batchResultsBody");
    DOM.btnExportBatchCsv = document.getElementById("btnExportBatchCsv");

    // Live Monitoring
    DOM.monKafkaStatus = document.getElementById("monKafkaStatus");
    DOM.monAlertCount = document.getElementById("monAlertCount");
    DOM.chkAutoPollMonitoring = document.getElementById("chkAutoPollMonitoring");
    DOM.btnPollAlertsNow = document.getElementById("btnPollAlertsNow");
    DOM.monitoringAlertsBody = document.getElementById("monitoringAlertsBody");

    // Cases Management
    DOM.caseMetricTotal = document.getElementById("caseMetricTotal");
    DOM.caseMetricOpen = document.getElementById("caseMetricOpen");
    DOM.caseMetricInvestigating = document.getElementById("caseMetricInvestigating");
    DOM.caseMetricConfirmed = document.getElementById("caseMetricConfirmed");
    DOM.caseMetricFalsePositive = document.getElementById("caseMetricFalsePositive");
    DOM.btnRefreshCases = document.getElementById("btnRefreshCases");
    DOM.caseSearchInput = document.getElementById("caseSearchInput");
    DOM.caseStatusFilter = document.getElementById("caseStatusFilter");
    DOM.caseRiskFilter = document.getElementById("caseRiskFilter");
    DOM.casesBody = document.getElementById("casesBody");

    // Health View
    DOM.techStackGrid = document.getElementById("techStackGrid");
    DOM.feastAccountInput = document.getElementById("feastAccountInput");
    DOM.btnLookupFeast = document.getElementById("btnLookupFeast");
    DOM.feastProfileResult = document.getElementById("feastProfileResult");

    // Modals
    DOM.caseModal = document.getElementById("caseModal");
    DOM.caseModalTitle = document.getElementById("caseModalTitle");
    DOM.btnCloseCaseModal = document.getElementById("btnCloseCaseModal");
    DOM.btnCancelCaseReview = document.getElementById("btnCancelCaseReview");
    DOM.caseModalAmount = document.getElementById("caseModalAmount");
    DOM.caseModalScore = document.getElementById("caseModalScore");
    DOM.caseReviewForm = document.getElementById("caseReviewForm");
    DOM.caseStatusSelect = document.getElementById("caseStatusSelect");
    DOM.caseDispositionSelect = document.getElementById("caseDispositionSelect");
    DOM.caseNotesInput = document.getElementById("caseNotesInput");
    DOM.caseAnalystIdInput = document.getElementById("caseAnalystIdInput");
    DOM.caseAuditTimeline = document.getElementById("caseAuditTimeline");

    DOM.thresholdModal = document.getElementById("thresholdModal");
    DOM.btnOpenThresholdModal = document.getElementById("btnOpenThresholdModal");
    DOM.btnCloseThresholdModal = document.getElementById("btnCloseThresholdModal");
    DOM.sliderBlock = document.getElementById("sliderBlock");
    DOM.sliderReview = document.getElementById("sliderReview");
    DOM.labelSliderBlock = document.getElementById("labelSliderBlock");
    DOM.labelSliderReview = document.getElementById("labelSliderReview");
    DOM.inputApiUrl = document.getElementById("inputApiUrl");
    DOM.btnSaveThresholds = document.getElementById("btnSaveThresholds");
    DOM.btnResetThresholds = document.getElementById("btnResetThresholds");
    DOM.sidebarApiPill = document.getElementById("sidebarApiPill");

    DOM.toastContainer = document.getElementById("toastContainer");
  }

  // ============================================================================
  // Toast System
  // ============================================================================
  function showToast(message, type = "info", duration = 4000) {
    if (!DOM.toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    
    let iconSvg = '';
    if (type === "success") {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
    } else if (type === "error") {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
    } else {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }

    toast.innerHTML = `${iconSvg}<span>${escapeHtml(message)}</span>`;
    DOM.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatCurrency(amount) {
    const num = Number(amount) || 0;
    return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(num);
  }

  function formatPercent(val) {
    const num = Number(val) || 0;
    return `${(num * 100).toFixed(2)}%`;
  }

  function formatStepTime(step) {
    const s = parseInt(step, 10) || 1;
    const day = Math.floor((s - 1) / 24) + 1;
    const hour = (s - 1) % 24;
    const hh = hour.toString().padStart(2, "0");
    return `Day ${day}, ${hh}:00`;
  }

  // ============================================================================
  // Navigation & View Switching
  // ============================================================================
  const VIEW_METADATA = {
    overview: {
      title: "Platform Overview",
      subtitle: "Real-time fraud scoring pipeline and risk governance",
    },
    analyze: {
      title: "Analyze Single Transaction",
      subtitle: "Evaluate in-flight or historical transfer risk parameters with XGBoost inference",
    },
    batch: {
      title: "Batch CSV Scoring",
      subtitle: "Vectorized high-throughput evaluation of multi-transaction datasets",
    },
    monitoring: {
      title: "Live Streaming & Kafka Feeds",
      subtitle: "Real-time event processing and consumer worker fraud alert stream",
    },
    cases: {
      title: "Analyst Case Management",
      subtitle: "Human-in-the-loop triage, alert review workflow, and immutable audit trail",
    },
    health: {
      title: "Model Governance & Health",
      subtitle: "Platform technology stack verification and Feast feature store profiles",
    },
  };

  function switchView(viewName) {
    if (!VIEW_METADATA[viewName]) return;
    state.activeView = viewName;

    DOM.navItems.forEach((btn) => {
      if (btn.getAttribute("data-view") === viewName) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    DOM.viewSections.forEach((sec) => {
      if (sec.id === `view-${viewName}`) {
        sec.classList.add("active");
      } else {
        sec.classList.remove("active");
      }
    });

    const meta = VIEW_METADATA[viewName];
    DOM.pageTitle.textContent = meta.title;
    DOM.pageSubtitle.textContent = meta.subtitle;

    // View-specific actions on entry
    if (viewName === "overview") {
      updateOverviewMetrics();
    } else if (viewName === "monitoring") {
      pollMonitoringAlerts();
    } else if (viewName === "cases") {
      loadCases();
    } else if (viewName === "health") {
      loadHealthView();
    }

    // Close mobile sidebar if open
    DOM.sidebar.classList.remove("mobile-open");
  }

  // ============================================================================
  // Theme Toggle
  // ============================================================================
  function applyTheme(theme) {
    state.currentTheme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("fraudwatch_theme", theme);

    if (theme === "dark") {
      DOM.themeIconSun.style.display = "block";
      DOM.themeIconMoon.style.display = "none";
    } else {
      DOM.themeIconSun.style.display = "none";
      DOM.themeIconMoon.style.display = "block";
    }

    if (state.riskChart) {
      updateChartColors(theme);
    }
  }

  function toggleTheme() {
    const nextTheme = state.currentTheme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
  }

  // ============================================================================
  // Health & Heartbeat
  // ============================================================================
  async function checkApiConnection() {
    try {
      const health = await FraudWatchAPI.checkHealth();
      state.apiHealth = health;
      DOM.sidebarStatusDot.className = "status-dot online pulsing";
      DOM.sidebarStatusText.textContent = "API Connected";
      DOM.kpiModelStatus.textContent = health.status === "healthy" ? "ONLINE" : "DEGRADED";
      DOM.kpiModelDetails.textContent = `${health.model_type || "XGBoost"} • ${health.feature_count || 11} Features`;
      return true;
    } catch (err) {
      state.apiHealth = null;
      DOM.sidebarStatusDot.className = "status-dot offline";
      DOM.sidebarStatusText.textContent = "API Offline";
      DOM.kpiModelStatus.textContent = "OFFLINE";
      DOM.kpiModelDetails.textContent = "Connection refused";
      return false;
    }
  }

  // ============================================================================
  // Overview View Logic
  // ============================================================================
  function updateOverviewMetrics() {
    const sessionList = state.sessionScoredTransactions;
    const casesCount = state.caseMetrics.total || 0;

    // Total Scored: session transactions plus persistent cases
    const totalCount = sessionList.length + casesCount;
    DOM.kpiTotalScored.textContent = totalCount.toLocaleString();

    let high = 0;
    let medium = 0;
    let low = 0;

    sessionList.forEach((tx) => {
      if (tx.risk_level === "HIGH") high++;
      else if (tx.risk_level === "MEDIUM") medium++;
      else low++;
    });

    // Also include persistent cases in high & medium counts
    high += (state.caseMetrics.confirmed_fraud || 0) + (state.caseMetrics.open || 0);
    medium += (state.caseMetrics.investigating || 0);
    low += (state.caseMetrics.false_positive || 0);

    DOM.kpiHighRisk.textContent = high.toLocaleString();
    DOM.kpiMediumRisk.textContent = medium.toLocaleString();
    DOM.kpiLowRisk.textContent = low.toLocaleString();

    renderRiskChart(high, medium, low);
    renderRecentTransactionsTable();
  }

  function renderRiskChart(high, medium, low) {
    const total = high + medium + low;
    if (total === 0) {
      if (DOM.noChartDataMsg) DOM.noChartDataMsg.style.display = "block";
      if (DOM.riskDistributionCanvas) DOM.riskDistributionCanvas.style.display = "none";
      return;
    }

    if (DOM.noChartDataMsg) DOM.noChartDataMsg.style.display = "none";
    if (DOM.riskDistributionCanvas) DOM.riskDistributionCanvas.style.display = "block";

    if (!window.Chart) return;

    const isDark = state.currentTheme === "dark";
    const textColor = isDark ? "#cbd5e1" : "#334155";

    if (state.riskChart) {
      state.riskChart.data.datasets[0].data = [low, medium, high];
      state.riskChart.update();
      return;
    }

    const ctx = DOM.riskDistributionCanvas.getContext("2d");
    state.riskChart = new window.Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Low Risk (Approved)", "Medium Risk (Review)", "High Risk (Blocked)"],
        datasets: [
          {
            data: [low, medium, high],
            backgroundColor: ["#16a34a", "#d97706", "#dc2626"],
            borderWidth: 2,
            borderColor: isDark ? "#111827" : "#ffffff",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              boxWidth: 12,
              color: textColor,
              font: { family: "Inter", size: 11, weight: "500" },
            },
          },
        },
        cutout: "70%",
      },
    });
  }

  function updateChartColors(theme) {
    if (!state.riskChart) return;
    const isDark = theme === "dark";
    state.riskChart.options.plugins.legend.labels.color = isDark ? "#cbd5e1" : "#334155";
    state.riskChart.data.datasets[0].borderColor = isDark ? "#111827" : "#ffffff";
    state.riskChart.update();
  }

  function renderRecentTransactionsTable() {
    if (!DOM.recentTransactionsBody) return;
    const list = [...state.sessionScoredTransactions].reverse().slice(0, 8);

    if (list.length === 0) {
      DOM.recentTransactionsBody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">
            No transactions scored yet in this session.<br>Click <strong>Analyze a Transaction</strong> to test real transfers.
          </td>
        </tr>
      `;
      return;
    }

    DOM.recentTransactionsBody.innerHTML = list
      .map((tx) => {
        let badgeClass = "badge-low";
        if (tx.risk_level === "HIGH") badgeClass = "badge-high";
        else if (tx.risk_level === "MEDIUM") badgeClass = "badge-medium";

        return `
          <tr>
            <td><code>${escapeHtml(tx.transaction_id)}</code></td>
            <td><strong>${escapeHtml(tx.type || "TRANSFER")}</strong></td>
            <td>${formatCurrency(tx.amount || 0)}</td>
            <td><strong>${formatPercent(tx.fraud_probability)}</strong></td>
            <td><span class="badge ${badgeClass}">${escapeHtml(tx.recommended_action)}</span></td>
          </tr>
        `;
      })
      .join("");
  }

  // ============================================================================
  // Single Transaction Scoring Logic
  // ============================================================================
  let currentInputMode = "historical";

  function setInputMode(mode) {
    currentInputMode = mode;
    if (mode === "historical") {
      DOM.modeBtnHistorical.classList.add("active");
      DOM.modeBtnLive.classList.remove("active");
      DOM.modeHint.textContent = "Historical mode requires settled new balances";
      DOM.groupNewOrig.style.display = "flex";
      DOM.groupNewDest.style.display = "flex";
      DOM.txNewOrig.required = true;
      DOM.txNewDest.required = true;
    } else {
      DOM.modeBtnHistorical.classList.remove("active");
      DOM.modeBtnLive.classList.add("active");
      DOM.modeHint.textContent = "Live mode estimates post-authorization balances";
      DOM.groupNewOrig.style.display = "none";
      DOM.groupNewDest.style.display = "none";
      DOM.txNewOrig.required = false;
      DOM.txNewDest.required = false;
    }
  }

  function applyPreset(preset) {
    DOM.txId.value = preset.txId;
    DOM.txStep.value = preset.step;
    DOM.txType.value = preset.type;
    DOM.txAmount.value = preset.amount;
    DOM.txNameOrig.value = preset.origId;
    DOM.txOldOrig.value = preset.origOld;
    DOM.txNewOrig.value = preset.origNew !== undefined ? preset.origNew : 0;
    DOM.txNameDest.value = preset.destId;
    DOM.txOldDest.value = preset.destOld;
    DOM.txNewDest.value = preset.destNew !== undefined ? preset.destNew : 0;
    if (DOM.txUseFeast) {
      DOM.txUseFeast.checked = !!preset.useFeast;
    }
    setInputMode(preset.mode);
    DOM.stepTimeHint.textContent = formatStepTime(preset.step);
    showToast(`Loaded preset: ${preset.name}`, "info");
  }

  async function handleSingleScoreSubmit(e) {
    e.preventDefault();

    const amount = parseFloat(DOM.txAmount.value);
    const oldOrig = parseFloat(DOM.txOldOrig.value);
    const oldDest = parseFloat(DOM.txOldDest.value) || 0.0;
    const step = parseInt(DOM.txStep.value, 10);
    const type = DOM.txType.value;
    const nameOrig = DOM.txNameOrig.value.trim();
    const nameDest = DOM.txNameDest.value.trim();
    const txId = DOM.txId.value.trim() || `TX-${Date.now()}`;
    const useFeast = DOM.txUseFeast.checked;

    if (isNaN(amount) || amount <= 0) {
      showToast("Amount must be greater than zero.", "error");
      return;
    }
    if (isNaN(oldOrig) || oldOrig < 0) {
      showToast("Origin Old Balance must be 0 or greater.", "error");
      return;
    }

    const payload = {
      step,
      type,
      amount,
      nameOrig,
      oldbalanceOrg: oldOrig,
      nameDest,
      oldbalanceDest: oldDest,
      mode: currentInputMode,
      transaction_id: txId,
      use_feature_store: useFeast,
    };

    if (currentInputMode === "historical") {
      const valOrig = parseFloat(DOM.txNewOrig.value);
      const valDest = parseFloat(DOM.txNewDest.value);
      payload.newbalanceOrig = isNaN(valOrig) ? 0.0 : valOrig;
      payload.newbalanceDest = isNaN(valDest) ? 0.0 : valDest;
    }

    DOM.btnScoreTx.disabled = true;
    DOM.btnScoreTx.innerHTML = '<span class="status-dot pulsing"></span> Evaluating...';

    try {
      const pred = await FraudWatchAPI.scoreTransaction(payload, {
        blockThreshold: state.blockThreshold,
        reviewThreshold: state.reviewThreshold,
        isPreauth: useFeast,
      });

      renderScoringResult(pred, payload);

      // Record in session
      const sessionRecord = {
        transaction_id: pred.transaction_id || txId,
        step,
        type,
        amount,
        nameOrig,
        nameDest,
        fraud_probability: pred.fraud_probability,
        risk_level: pred.risk_level,
        recommended_action: pred.recommended_action,
        input_mode: pred.input_mode || currentInputMode,
        timestamp: Date.now(),
      };
      state.sessionScoredTransactions.push(sessionRecord);

      updateOverviewMetrics();
      showToast(`Scored ${txId}: ${pred.risk_level} (${formatPercent(pred.fraud_probability)})`, "success");
    } catch (err) {
      showToast(err.message || "Failed to score transaction", "error");
      DOM.resultStatusBadge.className = "badge badge-high";
      DOM.resultStatusBadge.textContent = "Error";
    } finally {
      DOM.btnScoreTx.disabled = false;
      DOM.btnScoreTx.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Score Transaction
      `;
    }
  }

  function renderScoringResult(pred, inputPayload) {
    if (DOM.resultIdleState) DOM.resultIdleState.style.display = "none";
    if (DOM.resultActiveState) DOM.resultActiveState.style.display = "flex";

    const rawProb = typeof pred.fraud_probability === "number" ? pred.fraud_probability : parseFloat(pred.fraud_probability);
    const prob = isNaN(rawProb) ? 0 : Math.max(0, Math.min(1, rawProb));
    const risk = (pred.risk_level || (prob >= 0.90 ? "HIGH" : prob >= 0.50 ? "MEDIUM" : "LOW")).toUpperCase();
    const action = (pred.recommended_action || (risk === "HIGH" ? "BLOCK" : risk === "MEDIUM" ? "REVIEW" : "APPROVE")).toUpperCase();

    // 1. Update text displays FIRST so percentage and action are always rendered immediately
    if (DOM.resFraudProbability) {
      DOM.resFraudProbability.textContent = formatPercent(prob);
    }
    if (DOM.resRecommendedAction) {
      DOM.resRecommendedAction.textContent = action;
    }

    const badgeClass = risk === "HIGH" ? "badge-high" : risk === "MEDIUM" ? "badge-medium" : "badge-low";
    const strokeColor = risk === "HIGH" ? "var(--risk-high)" : risk === "MEDIUM" ? "var(--risk-medium)" : "var(--risk-low)";

    if (DOM.resRiskBadge) {
      DOM.resRiskBadge.className = `badge ${badgeClass}`;
      DOM.resRiskBadge.textContent = `${risk} RISK • ${action}`;
    }
    if (DOM.resultStatusBadge) {
      DOM.resultStatusBadge.className = `badge ${badgeClass}`;
      DOM.resultStatusBadge.textContent = `${risk} RISK`;
    }

    // 2. Animated SVG Gauge: total stroke length = 251.2
    // offset = 251.2 * (1 - prob)
    const offset = Math.max(0, Math.min(251.2, 251.2 * (1 - prob)));
    if (DOM.gaugeFillPath) {
      DOM.gaugeFillPath.style.strokeDashoffset = offset.toFixed(1);
      DOM.gaugeFillPath.style.stroke = strokeColor;
    }

    // 3. Action banner styling
    if (DOM.resActionBanner) {
      DOM.resActionBanner.style.borderLeft = `4px solid ${strokeColor}`;
    }

    // 4. Calculate features safely
    const oldBal = parseFloat(inputPayload.oldbalanceOrg) || 0;
    const txAmt = parseFloat(inputPayload.amount) || 0;
    const drainRatio = (txAmt / (oldBal + 1.0)) * 100;
    if (DOM.resDrainRatio) {
      DOM.resDrainRatio.textContent = `${Math.min(100, Math.max(0, drainRatio)).toFixed(1)}%`;
    }

    const expectedNewOrig = Math.max(0, oldBal - txAmt);
    const balanceError = inputPayload.newbalanceOrig !== undefined
      ? Math.abs(parseFloat(inputPayload.newbalanceOrig) - expectedNewOrig)
      : 0;
    if (DOM.resBalanceDiscrepancy) {
      DOM.resBalanceDiscrepancy.textContent = formatCurrency(balanceError);
    }

    if (DOM.resFeatureSource) {
      DOM.resFeatureSource.textContent = pred.features_source || "Payload";
    }
    if (DOM.resFeastStatus) {
      DOM.resFeastStatus.textContent = pred.feast_lookup_status || "Disabled";
    }

    if (DOM.resAuditBox && DOM.resAuditNote) {
      if (pred.audit_note) {
        DOM.resAuditBox.style.display = "block";
        DOM.resAuditNote.textContent = pred.audit_note;
      } else {
        DOM.resAuditBox.style.display = "none";
      }
    }

    if (DOM.resRawJson) {
      DOM.resRawJson.textContent = JSON.stringify(pred, null, 2);
    }
  }

  // ============================================================================
  // Batch CSV Scoring Logic
  // ============================================================================
  let parsedBatchTransactions = [];
  let lastScoredBatchResults = [];

  function setupCsvDropzone() {
    DOM.csvDropzone.addEventListener("click", () => DOM.csvFileInput.click());

    DOM.csvDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      DOM.csvDropzone.classList.add("dragover");
    });

    DOM.csvDropzone.addEventListener("dragleave", () => {
      DOM.csvDropzone.classList.remove("dragover");
    });

    DOM.csvDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      DOM.csvDropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        processUploadedCsv(e.dataTransfer.files[0]);
      }
    });

    DOM.csvFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        processUploadedCsv(e.target.files[0]);
      }
    });
  }

  function processUploadedCsv(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showToast("Please upload a .csv file.", "error");
      return;
    }

    const reader = new FileReader();
    reader.onload = (evt) => {
      const csvText = evt.target.result;
      parseAndValidateCsv(csvText, file.name, file.size);
    };
    reader.readAsText(file);
  }

  function parseAndValidateCsv(csvText, fileName, fileSize) {
    parsedBatchTransactions = [];
    DOM.batchValErrors.style.display = "none";
    DOM.batchValErrorsList.innerHTML = "";

    const lines = csvText.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
    if (lines.length < 2) {
      showToast("CSV file must contain a header row and at least one data row.", "error");
      return;
    }

    const headers = lines[0].split(",").map((h) => h.trim());
    const requiredHeaders = ["type", "amount", "nameOrig", "oldbalanceOrg", "nameDest"];
    const missing = requiredHeaders.filter((rh) => !headers.includes(rh));

    if (missing.length > 0) {
      showToast(`Missing mandatory CSV headers: ${missing.join(", ")}`, "error");
      return;
    }

    const validRows = [];
    const errorMessages = [];

    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(",").map((p) => p.trim());
      if (parts.length < headers.length) continue;

      const rowObj = {};
      headers.forEach((h, idx) => {
        rowObj[h] = parts[idx];
      });

      const step = parseInt(rowObj.step, 10) || 1;
      const type = (rowObj.type || "").toUpperCase();
      const amount = parseFloat(rowObj.amount);
      const nameOrig = rowObj.nameOrig;
      const oldOrig = parseFloat(rowObj.oldbalanceOrg);
      const nameDest = rowObj.nameDest;
      const oldDest = parseFloat(rowObj.oldbalanceDest) || 0.0;
      const txId = rowObj.transaction_id || `TX-CSV-${i}`;

      let mode = (rowObj.mode || "").toLowerCase();
      if (!mode) {
        mode = rowObj.newbalanceOrig !== undefined && rowObj.newbalanceDest !== undefined ? "historical" : "live_authorization";
      }

      if (isNaN(amount) || amount <= 0) {
        errorMessages.push(`Row ${i} (${txId}): amount must be positive.`);
        continue;
      }
      if (isNaN(oldOrig) || oldOrig < 0) {
        errorMessages.push(`Row ${i} (${txId}): oldbalanceOrg must be non-negative.`);
        continue;
      }

      const txPayload = {
        step,
        type,
        amount,
        nameOrig,
        oldbalanceOrg: oldOrig,
        nameDest,
        oldbalanceDest: oldDest,
        mode,
        transaction_id: txId,
      };

      if (mode === "historical") {
        txPayload.newbalanceOrig = parseFloat(rowObj.newbalanceOrig) || 0.0;
        txPayload.newbalanceDest = parseFloat(rowObj.newbalanceDest) || 0.0;
      }

      validRows.push(txPayload);
    }

    if (errorMessages.length > 0) {
      DOM.batchValErrors.style.display = "block";
      DOM.batchValErrorsList.innerHTML = errorMessages.slice(0, 5).map((e) => `<li>${escapeHtml(e)}</li>`).join("");
      if (errorMessages.length > 5) {
        DOM.batchValErrorsList.innerHTML += `<li>...and ${errorMessages.length - 5} more errors.</li>`;
      }
    }

    if (validRows.length === 0) {
      showToast("No valid transaction rows found in CSV.", "error");
      return;
    }

    parsedBatchTransactions = validRows;
    DOM.fileUploadInfo.style.display = "flex";
    DOM.uploadFileName.textContent = fileName;
    DOM.uploadFileMeta.textContent = `${validRows.length} valid transactions • ${(fileSize / 1024).toFixed(1)} KB`;
    showToast(`Validated ${validRows.length} transactions ready for scoring.`, "success");
  }

  async function handleBatchSubmit() {
    if (!parsedBatchTransactions || parsedBatchTransactions.length === 0) {
      showToast("No transactions ready for scoring.", "error");
      return;
    }

    DOM.btnSubmitBatch.disabled = true;
    DOM.btnSubmitBatch.textContent = "Scoring Batch...";

    try {
      const response = await FraudWatchAPI.scoreBatch(parsedBatchTransactions, {
        blockThreshold: state.blockThreshold,
        reviewThreshold: state.reviewThreshold,
      });

      lastScoredBatchResults = response.results || [];
      renderBatchResults(response);

      // Add to session
      response.results.forEach((r) => {
        state.sessionScoredTransactions.push({
          transaction_id: r.transaction_id,
          fraud_probability: r.fraud_probability,
          risk_level: r.risk_level,
          recommended_action: r.recommended_action,
          timestamp: Date.now(),
        });
      });
      updateOverviewMetrics();

      showToast(`Batch scored: ${response.fraud_count} high risk, ${response.review_count} review (${response.latency_ms.toFixed(1)}ms)`, "success");
    } catch (err) {
      showToast(err.message || "Failed to process batch", "error");
    } finally {
      DOM.btnSubmitBatch.disabled = false;
      DOM.btnSubmitBatch.textContent = "Score Batch";
    }
  }

  function renderBatchResults(summary) {
    DOM.batchResultsCard.style.display = "block";
    DOM.batchSummarySubtitle.textContent = `Completed in ${summary.latency_ms.toFixed(1)}ms (${summary.total_scored} transactions)`;
    DOM.batchTotalCount.textContent = summary.total_scored.toLocaleString();
    DOM.batchLatency.textContent = `${summary.latency_ms.toFixed(1)}ms latency`;
    DOM.batchBlockCount.textContent = summary.fraud_count.toLocaleString();
    DOM.batchReviewCount.textContent = summary.review_count.toLocaleString();
    DOM.batchApproveCount.textContent = summary.approved_count.toLocaleString();

    filterAndRenderBatchTable();
  }

  function filterAndRenderBatchTable() {
    if (!lastScoredBatchResults) return;
    const filter = DOM.batchRiskFilter.value;
    const search = DOM.batchSearchInput.value.toLowerCase().trim();

    const filtered = lastScoredBatchResults.filter((r) => {
      if (filter !== "ALL" && r.risk_level !== filter) return false;
      if (search && !r.transaction_id.toLowerCase().includes(search)) return false;
      return true;
    });

    DOM.batchResultsBody.innerHTML = filtered.slice(0, 100).map((r) => {
      let badgeClass = "badge-low";
      if (r.risk_level === "HIGH") badgeClass = "badge-high";
      else if (r.risk_level === "MEDIUM") badgeClass = "badge-medium";

      return `
        <tr>
          <td><code>${escapeHtml(r.transaction_id)}</code></td>
          <td><span class="badge ${badgeClass}">${escapeHtml(r.risk_level)}</span></td>
          <td><strong>${formatPercent(r.fraud_probability)}</strong></td>
          <td>${escapeHtml(r.recommended_action)}</td>
          <td>${escapeHtml(r.input_mode || "historical")}</td>
          <td style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(r.audit_note || "Standard policy derivation")}</td>
        </tr>
      `;
    }).join("");
  }

  function exportBatchCsv() {
    if (!lastScoredBatchResults || lastScoredBatchResults.length === 0) {
      showToast("No batch results to export.", "error");
      return;
    }

    const headers = ["transaction_id", "risk_level", "fraud_probability", "recommended_action", "input_mode", "block_threshold", "review_threshold"];
    const rows = lastScoredBatchResults.map((r) => [
      r.transaction_id,
      r.risk_level,
      r.fraud_probability,
      r.recommended_action,
      r.input_mode || "historical",
      r.block_threshold,
      r.review_threshold,
    ]);

    const csvContent = [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fraudwatch_batch_results_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Batch results exported to CSV.", "success");
  }

  function downloadTemplate(mode) {
    let csvData = "";
    if (mode === "historical") {
      csvData =
        "transaction_id,step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,mode\n" +
        "TX-HIST-001,1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,historical\n" +
        "TX-HIST-002,741,TRANSFER,250000.0,C123456789,250000.0,0.0,C987654321,0.0,0.0,historical\n" +
        "TX-HIST-003,12,CASH_OUT,15000.0,C999888777,15000.0,0.0,C111222333,5000.0,20000.0,historical\n";
    } else {
      csvData =
        "transaction_id,step,type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,mode\n" +
        "TX-LIVE-001,50,TRANSFER,50000.0,C111111111,100000.0,C222222222,0.0,live_authorization\n" +
        "TX-LIVE-002,10,PAYMENT,120.50,C555444333,1500.0,M987654321,0.0,live_authorization\n" +
        "TX-LIVE-003,24,CASH_OUT,2500.0,C777666555,2500.0,C888999000,100.0,live_authorization\n";
    }

    const blob = new Blob([csvData], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `paysim_${mode}_template.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ============================================================================
  // Live Monitoring Logic
  // ============================================================================
  async function pollMonitoringAlerts() {
    try {
      const data = await FraudWatchAPI.getRecentAlerts(50);
      const alerts = data.alerts || [];
      state.monitoringAlerts = alerts;

      DOM.monAlertCount.textContent = `${alerts.length} Alerts`;
      if (alerts.length > 0) {
        DOM.sidebarAlertBadge.style.display = "inline-block";
        DOM.sidebarAlertBadge.textContent = alerts.length;
      }

      if (alerts.length === 0) {
        DOM.monitoringAlertsBody.innerHTML = `
          <tr>
            <td colspan="10" style="text-align: center; color: var(--text-muted); padding: 2rem;">
              No alerts in stream yet. Run a streaming transaction or producer to capture Kafka events.
            </td>
          </tr>
        `;
        return;
      }

      DOM.monitoringAlertsBody.innerHTML = alerts.slice().reverse().map((a) => {
        const timeStr = a.timestamp ? new Date(a.timestamp * 1000).toLocaleTimeString() : `Step ${a.step}`;
        const drain = a.orig_drain_ratio !== undefined ? `${(a.orig_drain_ratio * 100).toFixed(1)}%` : "N/A";
        return `
          <tr>
            <td><span style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(timeStr)}</span></td>
            <td><code>${escapeHtml(a.transaction_id)}</code></td>
            <td><strong>${escapeHtml(a.type)}</strong></td>
            <td>${formatCurrency(a.amount)}</td>
            <td><code>${escapeHtml(a.nameOrig)}</code></td>
            <td><code>${escapeHtml(a.nameDest)}</code></td>
            <td><span class="badge badge-neutral">${escapeHtml(drain)}</span></td>
            <td><strong>${formatPercent(a.fraud_probability)}</strong></td>
            <td><span class="badge badge-high">${escapeHtml(a.risk_level || "HIGH")}</span></td>
            <td>
              <button class="btn btn-secondary btn-sm" onclick="window.FraudWatchApp.openCaseModalFromAlert('${escapeHtml(a.transaction_id)}')">
                Review Case
              </button>
            </td>
          </tr>
        `;
      }).join("");
    } catch (err) {
      DOM.monKafkaStatus.textContent = "DISCONNECTED";
      DOM.monitoringAlertsBody.innerHTML = `
        <tr>
          <td colspan="10" style="text-align: center; color: var(--risk-high-text); padding: 2rem;">
            Failed to poll Kafka alerts: ${escapeHtml(err.message)}
          </td>
        </tr>
      `;
    }
  }

  function setupMonitoringInterval() {
    if (state.monitoringPollTimer) {
      clearInterval(state.monitoringPollTimer);
      state.monitoringPollTimer = null;
    }

    if (DOM.chkAutoPollMonitoring && DOM.chkAutoPollMonitoring.checked) {
      state.monitoringPollTimer = setInterval(() => {
        if (state.activeView === "monitoring") {
          pollMonitoringAlerts();
        }
      }, 5000);
    }
  }

  // ============================================================================
  // Fraud Cases Management & Analyst Review
  // ============================================================================
  async function loadCases() {
    try {
      const [casesResp, metricsResp] = await Promise.all([
        FraudWatchAPI.getCases({
          status: DOM.caseStatusFilter.value,
          riskLevel: DOM.caseRiskFilter.value,
          search: DOM.caseSearchInput.value,
          limit: 100,
        }),
        FraudWatchAPI.getCaseMetrics(),
      ]);

      state.cases = casesResp.cases || [];
      state.caseMetrics = metricsResp || state.caseMetrics;

      // Update Case Badges & KPI
      DOM.sidebarCaseBadge.textContent = state.caseMetrics.open || 0;
      DOM.caseMetricTotal.textContent = (state.caseMetrics.total || 0).toLocaleString();
      DOM.caseMetricOpen.textContent = (state.caseMetrics.open || 0).toLocaleString();
      DOM.caseMetricInvestigating.textContent = (state.caseMetrics.investigating || 0).toLocaleString();
      DOM.caseMetricConfirmed.textContent = (state.caseMetrics.confirmed_fraud || 0).toLocaleString();
      DOM.caseMetricFalsePositive.textContent = (state.caseMetrics.false_positive || 0).toLocaleString();

      renderCasesTable(state.cases);
    } catch (err) {
      showToast(`Failed to load cases: ${err.message}`, "error");
    }
  }

  function renderCasesTable(cases) {
    if (!DOM.casesBody) return;
    if (cases.length === 0) {
      DOM.casesBody.innerHTML = `
        <tr>
          <td colspan="10" style="text-align: center; color: var(--text-muted); padding: 2rem;">
            No cases match the selected filters.
          </td>
        </tr>
      `;
      return;
    }

    DOM.casesBody.innerHTML = cases.map((c) => {
      let statusClass = "badge-neutral";
      if (c.case_status === "OPEN") statusClass = "badge-high";
      else if (c.case_status === "INVESTIGATING") statusClass = "badge-medium";
      else if (c.case_status === "CONFIRMED_FRAUD") statusClass = "badge-high";
      else if (c.case_status === "FALSE_POSITIVE") statusClass = "badge-low";

      let riskClass = c.risk_level === "HIGH" ? "badge-high" : (c.risk_level === "MEDIUM" ? "badge-medium" : "badge-low");

      return `
        <tr>
          <td><code>${escapeHtml(c.transaction_id)}</code></td>
          <td><span class="badge ${statusClass}">${escapeHtml(c.case_status)}</span></td>
          <td>${escapeHtml(c.type || "TRANSFER")}</td>
          <td>${formatCurrency(c.amount)}</td>
          <td><code>${escapeHtml(c.nameOrig)}</code></td>
          <td><code>${escapeHtml(c.nameDest)}</code></td>
          <td><strong>${formatPercent(c.fraud_probability)}</strong></td>
          <td><span class="badge ${riskClass}">${escapeHtml(c.risk_level)}</span></td>
          <td style="font-size: 0.78rem; color: var(--text-muted);">${escapeHtml(c.analyst_disposition || "Pending")}</td>
          <td>
            <button class="btn btn-primary btn-sm" onclick="window.FraudWatchApp.openCaseModal('${escapeHtml(c.transaction_id)}')">
              Investigate
            </button>
          </td>
        </tr>
      `;
    }).join("");
  }

  let activeModalCaseId = null;

  async function openCaseModal(transactionId) {
    activeModalCaseId = transactionId;
    DOM.caseModalTitle.textContent = `Investigate Case: ${transactionId}`;
    DOM.caseModalAmount.textContent = "Loading...";
    DOM.caseModalScore.textContent = "Loading...";
    DOM.caseAuditTimeline.innerHTML = '<span style="font-size: 0.8rem; color: var(--text-muted);">Loading audit trail...</span>';
    DOM.caseModal.classList.add("open");

    try {
      const caseDetail = await FraudWatchAPI.getCaseDetail(transactionId);
      DOM.caseModalAmount.textContent = `${formatCurrency(caseDetail.amount)} • ${caseDetail.type}`;
      DOM.caseModalScore.textContent = `${formatPercent(caseDetail.fraud_probability)} (${caseDetail.risk_level} Risk)`;
      DOM.caseStatusSelect.value = caseDetail.case_status || "OPEN";
      DOM.caseDispositionSelect.value = caseDetail.analyst_disposition || "SUSPECTED_ACCOUNT_TAKEOVER";
      DOM.caseNotesInput.value = caseDetail.analyst_notes || "";
      DOM.caseAnalystIdInput.value = caseDetail.analyst_id || "analyst_shubh";

      // Render timeline
      const audit = caseDetail.audit_history || [];
      if (audit.length === 0) {
        DOM.caseAuditTimeline.innerHTML = `
          <div class="timeline-item">
            <span class="timeline-time">${new Date((caseDetail.created_at || caseDetail.timestamp) * 1000).toLocaleString()}</span>
            <span class="timeline-desc">Initial alert generated from transaction score (${caseDetail.case_status}).</span>
          </div>
        `;
      } else {
        DOM.caseAuditTimeline.innerHTML = audit.map((entry) => `
          <div class="timeline-item">
            <span class="timeline-time">${new Date(entry.timestamp * 1000).toLocaleString()} &bull; <strong>${escapeHtml(entry.analyst_id || "analyst")}</strong></span>
            <span class="timeline-desc">
              Status changed from <code>${escapeHtml(entry.previous_status)}</code> &rarr; <strong>${escapeHtml(entry.new_status)}</strong>
              ${entry.disposition ? `(${escapeHtml(entry.disposition)})` : ""}
              ${entry.notes ? `<br><em>"${escapeHtml(entry.notes)}"</em>` : ""}
            </span>
          </div>
        `).join("");
      }
    } catch (err) {
      showToast(`Failed to load case detail: ${err.message}`, "error");
    }
  }

  async function handleCaseReviewSubmit(e) {
    e.preventDefault();
    if (!activeModalCaseId) return;

    const payload = {
      status: DOM.caseStatusSelect.value,
      disposition: DOM.caseDispositionSelect.value,
      notes: DOM.caseNotesInput.value.trim(),
      analyst_id: DOM.caseAnalystIdInput.value.trim() || "analyst_shubh",
    };

    try {
      await FraudWatchAPI.submitCaseReview(activeModalCaseId, payload);
      showToast(`Case ${activeModalCaseId} updated to ${payload.status}`, "success");
      DOM.caseModal.classList.remove("open");
      loadCases();
      updateOverviewMetrics();
    } catch (err) {
      showToast(`Failed to update case: ${err.message}`, "error");
    }
  }

  // ============================================================================
  // Model & System Health View
  // ============================================================================
  async function loadHealthView() {
    try {
      const data = await FraudWatchAPI.getPlatformStatus();
      state.platformStatus = data;
      const techs = data.technologies || {};

      const items = [
        { name: "FastAPI REST API", status: techs.fastapi?.status === "ONLINE", desc: "Real-time scoring backend on port 8000" },
        { name: "XGBoost Champion Model", status: techs.xgboost?.benchmark_model_loaded, desc: "11 benchmark + 16 leakage-free features" },
        { name: "Feast Feature Store", status: techs.feast?.online_store_exists, desc: "SQLite online account feature store" },
        { name: "Kafka Broker & Zookeeper", status: true, desc: "Port 9092 / 2181 live streaming" },
        { name: "FastAI Tabular Learning", status: techs.fastai?.model_exists, desc: "Dense neural network evaluation (AUPRC 0.8124)" },
        { name: "PyTorch Geometric GNN", status: techs.pytorch_geometric?.weights_exist, desc: "2-layer GCN graph account topology" },
        { name: "MLflow Experiment Tracking", status: techs.mlflow?.database_exists, desc: "SQLite metrics backend at mlflow.db" },
        { name: "Evidently AI Governance", status: techs.evidently_ai?.html_report_exists, desc: "HTML & JSON drift and model promotion audit" },
      ];

      DOM.techStackGrid.innerHTML = items.map((t) => `
        <div class="stat-card ${t.status ? 'risk-low' : 'risk-high'}">
          <span class="stat-label">${escapeHtml(t.name)}</span>
          <span class="stat-value" style="font-size: 1.15rem; color: ${t.status ? 'var(--risk-low-text)' : 'var(--risk-high-text)'};">
            ${t.status ? "HEALTHY / ONLINE" : "OFFLINE / MISSING"}
          </span>
          <span class="stat-subtext">${escapeHtml(t.desc)}</span>
        </div>
      `).join("");
    } catch (err) {
      DOM.techStackGrid.innerHTML = `<div style="color: var(--risk-high-text);">Failed to load platform status: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function handleFeastAccountLookup() {
    const accId = DOM.feastAccountInput.value.trim();
    if (!accId) return;

    DOM.btnLookupFeast.disabled = true;
    DOM.btnLookupFeast.textContent = "Querying...";
    DOM.feastProfileResult.style.display = "block";
    DOM.feastProfileResult.innerHTML = "Querying Feast online feature store...";

    try {
      const res = await FraudWatchAPI.getAccountFeastProfile(accId);
      if (res.found && res.features) {
        DOM.feastProfileResult.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <strong>Account ${escapeHtml(accId)}: <span class="badge badge-low">HIT</span></strong>
            <span style="font-size: 0.75rem; color: var(--text-muted);">${res.latency_ms}ms retrieval</span>
          </div>
          <pre style="font-size: 0.8rem; background: var(--bg-card); padding: 0.75rem; border-radius: var(--radius-sm);">${JSON.stringify(res.features, null, 2)}</pre>
        `;
      } else {
        DOM.feastProfileResult.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong>Account ${escapeHtml(accId)}: <span class="badge badge-medium">MISS</span></strong>
            <span style="font-size: 0.75rem; color: var(--text-muted);">${res.latency_ms}ms</span>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.35rem;">
            No online precomputed features found for this account in Feast online store. Scoring engine will use payload fallback.
          </p>
        `;
      }
    } catch (err) {
      DOM.feastProfileResult.innerHTML = `<span style="color: var(--risk-high-text);">Feast error: ${escapeHtml(err.message)}</span>`;
    } finally {
      DOM.btnLookupFeast.disabled = false;
      DOM.btnLookupFeast.textContent = "Query Feast";
    }
  }

  // ============================================================================
  // Threshold Settings Modal
  // ============================================================================
  function openThresholdModal() {
    DOM.sliderBlock.value = state.blockThreshold;
    DOM.sliderReview.value = state.reviewThreshold;
    DOM.labelSliderBlock.textContent = state.blockThreshold.toFixed(2);
    DOM.labelSliderReview.textContent = state.reviewThreshold.toFixed(2);
    DOM.inputApiUrl.value = FraudWatchAPI.getBaseUrl();
    DOM.thresholdModal.classList.add("open");
  }

  function saveThresholdSettings() {
    const blockVal = parseFloat(DOM.sliderBlock.value);
    const reviewVal = parseFloat(DOM.sliderReview.value);
    const apiUrl = DOM.inputApiUrl.value.trim();

    if (blockVal <= reviewVal) {
      showToast("Block Threshold must be strictly greater than Review Threshold.", "error");
      return;
    }

    state.blockThreshold = blockVal;
    state.reviewThreshold = reviewVal;
    DOM.headerThresholdText.textContent = `Block: ${blockVal.toFixed(2)} | Review: ${reviewVal.toFixed(2)}`;
    DOM.sidebarPolicyText.textContent = `Demo ${Math.round(blockVal * 100)}/${Math.round(reviewVal * 100)}`;

    if (apiUrl && apiUrl !== FraudWatchAPI.getBaseUrl()) {
      FraudWatchAPI.setBaseUrl(apiUrl);
      checkApiConnection();
      showToast(`API Base URL updated to ${apiUrl}`, "info");
    }

    DOM.thresholdModal.classList.remove("open");
    showToast("Decision policy thresholds saved.", "success");
  }

  // ============================================================================
  // Event Bindings
  // ============================================================================
  function bindEvents() {
    // Nav Items
    DOM.navItems.forEach((btn) => {
      btn.addEventListener("click", () => {
        const view = btn.getAttribute("data-view");
        switchView(view);
      });
    });

    // Mobile Sidebar Toggle
    DOM.btnToggleSidebar.addEventListener("click", () => {
      DOM.sidebar.classList.toggle("mobile-open");
    });

    // Theme Toggle
    DOM.btnToggleTheme.addEventListener("click", toggleTheme);

    // Refresh Button
    DOM.btnRefreshData.addEventListener("click", async () => {
      showToast("Refreshing platform data...", "info");
      await checkApiConnection();
      if (state.activeView === "overview") updateOverviewMetrics();
      else if (state.activeView === "cases") loadCases();
      else if (state.activeView === "monitoring") pollMonitoringAlerts();
      else if (state.activeView === "health") loadHealthView();
      showToast("Platform data refreshed.", "success");
    });

    // Hero Action Buttons
    DOM.heroBtnAnalyze.addEventListener("click", () => switchView("analyze"));
    DOM.heroBtnBatch.addEventListener("click", () => switchView("batch"));

    // Threshold Modal triggers
    DOM.btnOpenThresholdModal.addEventListener("click", openThresholdModal);
    DOM.sidebarApiPill.addEventListener("click", openThresholdModal);
    DOM.btnCloseThresholdModal.addEventListener("click", () => DOM.thresholdModal.classList.remove("open"));
    DOM.sliderBlock.addEventListener("input", (e) => (DOM.labelSliderBlock.textContent = parseFloat(e.target.value).toFixed(2)));
    DOM.sliderReview.addEventListener("input", (e) => (DOM.labelSliderReview.textContent = parseFloat(e.target.value).toFixed(2)));
    DOM.btnSaveThresholds.addEventListener("click", saveThresholdSettings);
    DOM.btnResetThresholds.addEventListener("click", () => {
      DOM.sliderBlock.value = 0.90;
      DOM.sliderReview.value = 0.50;
      DOM.labelSliderBlock.textContent = "0.90";
      DOM.labelSliderReview.textContent = "0.50";
      DOM.inputApiUrl.value = FraudWatchAPI.resetBaseUrl();
    });

    // Presets in Single Scoring
    DOM.presetFraud.addEventListener("click", () => {
      applyPreset({
        name: "Typical Account Drain Fraud",
        txId: "PRESET-FRAUD-001",
        step: 741,
        type: "TRANSFER",
        amount: 250000.0,
        origId: "C123456789",
        origOld: 250000.0,
        origNew: 0.0,
        destId: "C987654321",
        destOld: 0.0,
        destNew: 0.0,
        mode: "historical",
      });
    });

    DOM.presetLegit.addEventListener("click", () => {
      applyPreset({
        name: "Legitimate Merchant Payment",
        txId: "PRESET-LEGIT-001",
        step: 1,
        type: "PAYMENT",
        amount: 9839.64,
        origId: "C1231006815",
        origOld: 170136.0,
        origNew: 160296.36,
        destId: "M1979787155",
        destOld: 0.0,
        destNew: 0.0,
        mode: "historical",
      });
    });

    DOM.presetLive.addEventListener("click", () => {
      applyPreset({
        name: "Live In-Flight Transfer",
        txId: "PRESET-LIVE-001",
        step: 48,
        type: "TRANSFER",
        amount: 50000.0,
        origId: "C555666777",
        origOld: 100000.0,
        destId: "C888999000",
        destOld: 0.0,
        mode: "live_authorization",
      });
    });

    if (DOM.presetReview) {
      DOM.presetReview.addEventListener("click", () => {
        applyPreset({
          name: "Suspicious In-Flight (Review Assessment)",
          txId: "PRESET-REVIEW-001",
          step: 120,
          type: "TRANSFER",
          amount: 75000.0,
          origId: "C555666777",
          origOld: 100000.0,
          destId: "C888999000",
          destOld: 0.0,
          mode: "live_authorization",
          useFeast: true,
        });
      });
    }

    // Ingestion Mode Controls
    DOM.modeBtnHistorical.addEventListener("click", () => setInputMode("historical"));
    DOM.modeBtnLive.addEventListener("click", () => setInputMode("live_authorization"));

    // Step Hint
    DOM.txStep.addEventListener("input", (e) => {
      DOM.stepTimeHint.textContent = formatStepTime(e.target.value);
    });

    // Form Submit & Reset
    DOM.singleTxForm.addEventListener("submit", handleSingleScoreSubmit);
    DOM.btnResetForm.addEventListener("click", () => {
      DOM.singleTxForm.reset();
      DOM.txStep.value = 1;
      DOM.stepTimeHint.textContent = formatStepTime(1);
      setInputMode("historical");
      DOM.resultIdleState.style.display = "block";
      DOM.resultActiveState.style.display = "none";
      DOM.resultStatusBadge.className = "badge badge-neutral";
      DOM.resultStatusBadge.textContent = "Awaiting Input";
    });

    // CSV Dropzone & Batch Submit
    setupCsvDropzone();
    DOM.btnSubmitBatch.addEventListener("click", handleBatchSubmit);
    DOM.btnExportBatchCsv.addEventListener("click", exportBatchCsv);
    DOM.batchSearchInput.addEventListener("input", filterAndRenderBatchTable);
    DOM.batchRiskFilter.addEventListener("change", filterAndRenderBatchTable);
    DOM.btnDownloadHistTemplate.addEventListener("click", () => downloadTemplate("historical"));
    DOM.btnDownloadLiveTemplate.addEventListener("click", () => downloadTemplate("live_authorization"));

    // Live Monitoring
    DOM.btnPollAlertsNow.addEventListener("click", () => {
      showToast("Polling Kafka alerts...", "info");
      pollMonitoringAlerts();
    });
    DOM.chkAutoPollMonitoring.addEventListener("change", setupMonitoringInterval);

    // Cases Filter
    DOM.btnRefreshCases.addEventListener("click", () => {
      showToast("Refreshing cases...", "info");
      loadCases();
    });
    DOM.caseStatusFilter.addEventListener("change", loadCases);
    DOM.caseRiskFilter.addEventListener("change", loadCases);
    DOM.caseSearchInput.addEventListener("input", () => {
      clearTimeout(DOM.caseSearchInput._timer);
      DOM.caseSearchInput._timer = setTimeout(loadCases, 300);
    });

    // Case Modal
    DOM.btnCloseCaseModal.addEventListener("click", () => DOM.caseModal.classList.remove("open"));
    DOM.btnCancelCaseReview.addEventListener("click", () => DOM.caseModal.classList.remove("open"));
    DOM.caseReviewForm.addEventListener("submit", handleCaseReviewSubmit);

    // Feature Store lookup
    DOM.btnLookupFeast.addEventListener("click", handleFeastAccountLookup);
  }

  // ============================================================================
  // Initialization
  // ============================================================================
  async function init() {
    initDomCache();
    applyTheme(state.currentTheme);
    bindEvents();
    setupMonitoringInterval();

    // Check API connection
    await checkApiConnection();

    // Initial data fetch
    await loadCases();
    updateOverviewMetrics();
  }

  // Expose global controller helpers for inline button handlers
  window.FraudWatchApp = {
    openCaseModal,
    openCaseModalFromAlert(txId) {
      switchView("cases");
      openCaseModal(txId);
    },
  };

  document.addEventListener("DOMContentLoaded", init);
})(window, document, window.FraudWatchAPI);
