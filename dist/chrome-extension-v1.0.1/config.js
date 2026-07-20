// Central backend-location policy shared by every production extension request.
function normalizeBaseUrl(value) {
  return String(value ?? "").trim().replace(/\/+$/, "");
}

const configuredHostedUrl = "https://bias-article-detector.onrender.com";
const runtimeOverride = globalThis.FACTGPT_BACKEND_URL;

export const BACKEND_BASE_URLS = Array.from(
  new Set(
    [runtimeOverride, configuredHostedUrl]
      .map(normalizeBaseUrl)
      .filter(Boolean)
  )
);
