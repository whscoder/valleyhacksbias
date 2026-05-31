import { requestBackend } from "./api.js";

const ENDPOINTS = Object.freeze({
  health: "/health",
  extract: "/extract",
  extractRendered: "/extract-rendered",
  analyzeBias: "/analyze-bias",
  research: "/research"
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

export function researchText(text, title = "Article Analysis") {
  return postJson(ENDPOINTS.research, { text, title });
}
