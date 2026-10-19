/**
 * FraudWatch — Financial Fraud Detection Platform
 * api.js: Centralized API client module with timeout handling and error formatting.
 */

(function (window) {
  "use strict";

  const STORAGE_KEY = "fraudwatch_api_url";
  const DEFAULT_FALLBACK_URL = "http://127.0.0.1:8000";

  // Auto-detect base URL: If served from FastAPI (e.g. port 8000), use origin; otherwise default to 127.0.0.1:8000
  function detectDefaultBaseUrl() {
    if (window.location.port === "8000" && window.location.protocol.startsWith("http")) {
      return window.location.origin;
    }
    return DEFAULT_FALLBACK_URL;
  }

  let currentBaseUrl = localStorage.getItem(STORAGE_KEY) || detectDefaultBaseUrl();

  const FraudWatchAPI = {
    getBaseUrl() {
      return currentBaseUrl;
    },

    setBaseUrl(url) {
      if (!url || typeof url !== "string") return;
      currentBaseUrl = url.trim().replace(/\/+$/, "");
      localStorage.setItem(STORAGE_KEY, currentBaseUrl);
    },

    resetBaseUrl() {
      currentBaseUrl = detectDefaultBaseUrl();
      localStorage.removeItem(STORAGE_KEY);
      return currentBaseUrl;
    },

    /**
     * Core fetch wrapper with timeout, header management, and uniform error extraction.
     */
    async request(path, options = {}) {
      const {
        method = "GET",
        body = null,
        params = null,
        timeoutMs = 8000,
        headers = {},
      } = options;

      const cleanPath = path.startsWith("/") ? path : `/${path}`;
      let url = `${currentBaseUrl}${cleanPath}`;

      if (params && typeof params === "object") {
        const query = new URLSearchParams();
        for (const [key, val] of Object.entries(params)) {
          if (val !== null && val !== undefined && val !== "") {
            query.append(key, String(val));
          }
        }
        const qs = query.toString();
        if (qs) {
          url += (url.includes("?") ? "&" : "?") + qs;
        }
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      const reqHeaders = {
        Accept: "application/json",
        ...headers,
      };

      if (body && !reqHeaders["Content-Type"]) {
        reqHeaders["Content-Type"] = "application/json";
      }

      const reqConfig = {
        method,
        headers: reqHeaders,
        signal: controller.signal,
      };

      if (body) {
        reqConfig.body = typeof body === "string" ? body : JSON.stringify(body);
      }

      const t0 = performance.now();
      try {
        const response = await fetch(url, reqConfig);
        clearTimeout(timer);
        const latencyMs = Math.round(performance.now() - t0);

        let data = null;
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          data = await response.json();
        } else {
          data = await response.text();
        }

        if (!response.ok) {
          let errorMessage = `HTTP ${response.status} ${response.statusText}`;
          if (data && typeof data === "object") {
            if (data.message) {
              errorMessage = data.message;
            } else if (data.detail) {
              errorMessage = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
            }
            if (Array.isArray(data.details)) {
              const fieldErrors = data.details.map((d) => `${d.location}: ${d.message}`).join("; ");
              errorMessage += ` (${fieldErrors})`;
            }
          } else if (typeof data === "string" && data.length < 200) {
            errorMessage = data;
          }

          const error = new Error(errorMessage);
          error.status = response.status;
          error.data = data;
          error.latencyMs = latencyMs;
          throw error;
        }

        return {
          ok: true,
          status: response.status,
          latencyMs,
          data,
        };
      } catch (err) {
        clearTimeout(timer);
        const latencyMs = Math.round(performance.now() - t0);

        if (err.name === "AbortError") {
          const timeoutErr = new Error(`Request timed out after ${timeoutMs}ms.`);
          timeoutErr.status = 408;
          timeoutErr.latencyMs = latencyMs;
          throw timeoutErr;
        }

        if (!err.status) {
          // Network failure / Connection refused / CORS failure
          const netErr = new Error(
            `Unable to connect to FraudWatch API at ${currentBaseUrl}. Check if the FastAPI service is running.`
          );
          netErr.status = 0;
          netErr.original = err;
          netErr.latencyMs = latencyMs;
          throw netErr;
        }

        throw err;
      }
    },

    /**
     * Check API server readiness and loaded model details.
     */
    async checkHealth() {
      const res = await this.request("/health", { timeoutMs: 4000 });
      return res.data;
    },

    /**
     * Query full 8-technology platform component statuses.
     */
    async getPlatformStatus() {
      const res = await this.request("/platform/status", { timeoutMs: 5000 });
      return res.data;
    },

    /**
     * Score a single transaction.
     * Uses /predict by default, or /predict/preauth if feast feature store enrichment is requested.
     */
    async scoreTransaction(payload, options = {}) {
      const { blockThreshold, reviewThreshold, isPreauth = false } = options;
      const params = {};
      if (blockThreshold !== undefined && blockThreshold !== null) {
        params.block_threshold = blockThreshold;
      }
      if (reviewThreshold !== undefined && reviewThreshold !== null) {
        params.review_threshold = reviewThreshold;
      }

      const endpoint = isPreauth ? "/predict/preauth" : "/predict";
      const res = await this.request(endpoint, {
        method: "POST",
        body: payload,
        params,
        timeoutMs: 10000,
      });
      return res.data;
    },

    /**
     * Score a batch of transactions via vectorized POST /predict/batch.
     */
    async scoreBatch(transactions, options = {}) {
      const { blockThreshold, reviewThreshold } = options;
      const payload = {
        transactions,
      };
      if (blockThreshold !== undefined && blockThreshold !== null) {
        payload.block_threshold = blockThreshold;
      }
      if (reviewThreshold !== undefined && reviewThreshold !== null) {
        payload.review_threshold = reviewThreshold;
      }

      const res = await this.request("/predict/batch", {
        method: "POST",
        body: payload,
        timeoutMs: 30000,
      });
      return res.data;
    },

    /**
     * Retrieve recent streaming alerts from Kafka pipeline log.
     */
    async getRecentAlerts(limit = 50) {
      const res = await this.request("/platform/alerts", {
        params: { limit },
        timeoutMs: 5000,
      });
      return res.data;
    },

    /**
     * List cases from persistent SQLite case store with filtering.
     */
    async getCases(filter = {}) {
      const {
        status = "ALL",
        riskLevel = "ALL",
        minScore = null,
        search = null,
        limit = 50,
      } = filter;

      const params = {
        status: status || "ALL",
        risk_level: riskLevel || "ALL",
        limit,
      };
      if (minScore !== null && minScore !== undefined) {
        params.min_score = minScore;
      }
      if (search && search.trim()) {
        params.search = search.trim();
      }

      const res = await this.request("/platform/cases", {
        params,
        timeoutMs: 6000,
      });
      return res.data;
    },

    /**
     * Retrieve aggregate counts for case management workflow pipeline.
     */
    async getCaseMetrics() {
      const res = await this.request("/platform/cases/summary/metrics", { timeoutMs: 4000 });
      return res.data;
    },

    /**
     * Get single fraud case details including chronological audit history.
     */
    async getCaseDetail(transactionId) {
      const res = await this.request(`/platform/cases/${encodeURIComponent(transactionId)}`, {
        timeoutMs: 4000,
      });
      return res.data;
    },

    /**
     * Submit analyst feedback and update case status.
     */
    async submitCaseReview(transactionId, reviewPayload) {
      const res = await this.request(`/platform/cases/${encodeURIComponent(transactionId)}/review`, {
        method: "POST",
        body: reviewPayload,
        timeoutMs: 6000,
      });
      return res.data;
    },

    /**
     * Query Feast online feature store profile for an account ID.
     */
    async getAccountFeastProfile(accountId) {
      const res = await this.request(`/platform/feature-store/account/${encodeURIComponent(accountId)}`, {
        timeoutMs: 5000,
      });
      return res.data;
    },
  };

  window.FraudWatchAPI = FraudWatchAPI;
})(window);
