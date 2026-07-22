// Typed-by-convention endpoint wrappers used by the worker and legacy UI pipeline.
import { requestBackend } from "./api.js";

const ENDPOINTS = Object.freeze({
  health: "/health",
  extract: "/extract",
  extractRendered: "/extract-rendered",
  analyzeBias: "/analyze-bias",
  research: "/research",
  articleJobs: "/article-jobs",
  podcastJobs: "/podcast-jobs"
});

const TIMEOUTS = Object.freeze({
  health: 20_000,
  extract: 45_000,
  extractRendered: 75_000,
  analyzeBias: 120_000,
  research: 180_000,
  articleRequest: 45_000,
  podcastRequest: 45_000
});

function postJson(endpoint, payload, timeoutMs, signal) {
  return requestBackend(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal
  }, timeoutMs);
}

export function warmBackend() {
  return requestBackend(ENDPOINTS.health, { method: "GET" }, TIMEOUTS.health);
}

export function extractArticleFromUrl(url, signal) {
  return postJson(ENDPOINTS.extract, { url: url.trim() }, TIMEOUTS.extract, signal);
}

export function extractRenderedArticleFromUrl(url, signal) {
  return postJson(
    ENDPOINTS.extractRendered,
    { url: url.trim() },
    TIMEOUTS.extractRendered,
    signal
  );
}

export function analyzeBiasText(text, title = "Article Analysis", signal) {
  return postJson(
    ENDPOINTS.analyzeBias,
    { text, title },
    TIMEOUTS.analyzeBias,
    signal
  );
}

export function researchText(
  text,
  title = "Article Analysis",
  factOpinion = null,
  biasResult = null,
  signal
) {
  return postJson(
    ENDPOINTS.research,
    {
      text,
      title,
      ...(factOpinion ? { fact_opinion: factOpinion } : {}),
      ...(biasResult ? { bias_result: biasResult } : {})
    },
    TIMEOUTS.research,
    signal
  );
}

export function createArticleJob(pageUrl, clientRequestId, text = "", title = "Article Analysis") {
  return postJson(
    ENDPOINTS.articleJobs,
    {
      page_url: String(pageUrl ?? "").trim(),
      client_request_id: String(clientRequestId ?? "").trim(),
      text: String(text ?? ""),
      title: String(title ?? "Article Analysis").trim() || "Article Analysis"
    },
    TIMEOUTS.articleRequest
  );
}

export function getArticleJob(jobId) {
  return requestBackend(`${ENDPOINTS.articleJobs}/${encodeURIComponent(jobId)}`, {
    method: "GET"
  }, TIMEOUTS.articleRequest);
}

export function createPodcastJob(pageUrl, hints = {}) {
  return postJson(
    ENDPOINTS.podcastJobs,
    {
      page_url: String(pageUrl ?? "").trim(),
      hints: {
        feed_urls: Array.isArray(hints.feed_urls) ? hints.feed_urls.slice(0, 5) : [],
        transcript_urls: Array.isArray(hints.transcript_urls) ? hints.transcript_urls.slice(0, 5) : [],
        audio_urls: Array.isArray(hints.audio_urls) ? hints.audio_urls.slice(0, 5) : []
      }
    },
    TIMEOUTS.podcastRequest
  );
}

export function getPodcastJob(jobId) {
  return requestBackend(`${ENDPOINTS.podcastJobs}/${encodeURIComponent(jobId)}`, {
    method: "GET"
  }, TIMEOUTS.podcastRequest);
}

export function getPodcastSegments(jobId, cursor = "", limit = 100) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cursor !== "" && cursor !== null && cursor !== undefined) {
    params.set("cursor", String(cursor));
  }
  return requestBackend(
    `${ENDPOINTS.podcastJobs}/${encodeURIComponent(jobId)}/segments?${params.toString()}`,
    { method: "GET" },
    TIMEOUTS.podcastRequest
  );
}
