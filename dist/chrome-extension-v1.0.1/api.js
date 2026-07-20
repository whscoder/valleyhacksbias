// Low-level HTTP adapter: tries configured backends and normalizes error bodies.
import { BACKEND_BASE_URLS } from "./config.js";

function buildBackendUrl(baseUrl, endpoint) {
  const normalizedBase = String(baseUrl ?? "").replace(/\/+$/, "");
  const normalizedEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${normalizedBase}${normalizedEndpoint}`;
}

async function readResponseBody(response) {
  try {
    return { json: await response.clone().json(), text: "" };
  } catch {
    try {
      return { json: {}, text: await response.text() };
    } catch {
      return { json: {}, text: "" };
    }
  }
}

export function formatBackendErrorDetail(value, fallback) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const message = String(value.message ?? fallback ?? "Request failed.").trim();
    const diagnostics = [
      value.code ? `Code: ${String(value.code).trim()}` : "",
      value.reference ? `Reference: ${String(value.reference).trim()}` : ""
    ].filter(Boolean);
    return [message, ...diagnostics].join(" ");
  }
  return String(value || fallback || "Request failed.").trim();
}

export async function requestBackend(endpoint, init = {}) {
  let lastError = null;

  // Fail over only when a base URL is unreachable. An HTTP response proves that
  // the backend was reached, so surface its error instead of retrying elsewhere.
  for (const baseUrl of BACKEND_BASE_URLS) {
    let response;
    try {
      response = await fetch(buildBackendUrl(baseUrl, endpoint), init);
    } catch (error) {
      lastError = error;
      continue;
    }

    const { json, text } = await readResponseBody(response);
    if (!response.ok) {
      const detail = json.detail || json.error || json.message || text;
      throw new Error(formatBackendErrorDetail(
        detail,
        `Request failed with HTTP ${response.status}.`
      ));
    }
    return json;
  }

  throw lastError ?? new Error("No reachable backend URL is configured.");
}
