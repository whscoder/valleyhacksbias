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

export async function requestBackend(endpoint, init = {}) {
  let lastError = null;

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
      throw new Error(String(detail || `Request failed with HTTP ${response.status}.`).trim());
    }
    return json;
  }

  throw lastError ?? new Error("No reachable backend URL is configured.");
}
