// Typed-by-convention endpoint wrappers used by the worker and legacy UI pipeline.
import { requestBackend } from "./api.js";

const ENDPOINTS = Object.freeze({
  health: "/health",
  extract: "/extract",
  extractRendered: "/extract-rendered",
  analyzeBias: "/analyze-bias",
  research: "/research",
  podcastJobs: "/podcast-jobs"
});

function postJson(endpoint, payload) {
  return requestBackend(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function warmBackend() {
  return requestBackend(ENDPOINTS.health, { method: "GET" });
}

export function extractArticleFromUrl(url) {
  return postJson(ENDPOINTS.extract, { url: url.trim() });
}

export function extractRenderedArticleFromUrl(url) {
  return postJson(ENDPOINTS.extractRendered, { url: url.trim() });
}

export function analyzeBiasText(text, title = "Article Analysis") {
  return postJson(ENDPOINTS.analyzeBias, { text, title });
}

export function researchText(
  text,
  title = "Article Analysis",
  factOpinion = null,
  biasResult = null
) {
  return postJson(ENDPOINTS.research, {
    text,
    title,
    ...(factOpinion ? { fact_opinion: factOpinion } : {}),
    ...(biasResult ? { bias_result: biasResult } : {})
  });
}

export function createPodcastJob(pageUrl, hints = {}) {
  return postJson(ENDPOINTS.podcastJobs, {
    page_url: String(pageUrl ?? "").trim(),
    hints: {
      feed_urls: Array.isArray(hints.feed_urls) ? hints.feed_urls.slice(0, 5) : [],
      transcript_urls: Array.isArray(hints.transcript_urls) ? hints.transcript_urls.slice(0, 5) : [],
      audio_urls: Array.isArray(hints.audio_urls) ? hints.audio_urls.slice(0, 5) : []
    }
  });
}

export function getPodcastJob(jobId) {
  return requestBackend(`${ENDPOINTS.podcastJobs}/${encodeURIComponent(jobId)}`, {
    method: "GET"
  });
}

export function getPodcastSegments(jobId, cursor = "", limit = 100) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cursor !== "" && cursor !== null && cursor !== undefined) {
    params.set("cursor", String(cursor));
  }
  return requestBackend(
    `${ENDPOINTS.podcastJobs}/${encodeURIComponent(jobId)}/segments?${params.toString()}`,
    { method: "GET" }
  );
}
