import { BACKEND_BASE_URLS } from "./config.js";

export function buildBackendUrl(baseUrl, endpoint) {
  const normalizedBase = String(baseUrl ?? "").replace(/\/+$/, "");
  const normalizedEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${normalizedBase}${normalizedEndpoint}`;
}

export async function fetchBackend(endpoint, init) {
  let lastError = null;

  for (const baseUrl of BACKEND_BASE_URLS) {
    try {
      return await fetch(buildBackendUrl(baseUrl, endpoint), init);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError ?? new Error("No reachable backend URL is configured.");
}
