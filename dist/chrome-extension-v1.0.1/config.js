function normalizeBaseUrl(value) {
  return String(value ?? "").trim().replace(/\/+$/, "");
}

const configuredHostedUrl = "https://bias-article-detector.onrender.com";
const runtimeOverride = globalThis.FACTGPT_BACKEND_URL;
const fallbackLocalUrl = "http://127.0.0.1:8000";

// Set `configuredHostedUrl` after Render or Cloud Run gives you the public API URL.
export const BACKEND_BASE_URLS = Array.from(
  new Set(
    [runtimeOverride, configuredHostedUrl, fallbackLocalUrl]
      .map(normalizeBaseUrl)
      .filter(Boolean)
  )
);
