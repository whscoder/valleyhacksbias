import { fetchBackend } from "./api.js";
import {
  highlightPhrasesInTab,
  normalizeHighlights,
  normalizeText,
  renderResult,
  renderSources,
  setOutputMessage
} from "./parseScript.js";

export let capturedUrl = "";

const POLL_MS = 800;
const STALE_RUNNING_MS = 5 * 60 * 1000;

// Reads the currently focused tab so the popup can scope saved results to the page.
async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab found.");
  return tab;
}

function sendRuntimeMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      const lastError = chrome.runtime.lastError;
      if (lastError) {
        reject(new Error(lastError.message));
        return;
      }
      resolve(response);
    });
  });
}

async function getAnalysisKey(url) {
  const response = await sendRuntimeMessage({ type: "FACTGPT_ANALYSIS_KEY", url });
  return response?.key || "";
}

async function getSavedAnalysis(key) {
  if (!key) {
    return null;
  }
  const data = await chrome.storage.local.get(key);
  return data[key] || null;
}

function setSectionVisible(section, visible) {
  if (section) {
    section.style.display = visible ? "block" : "none";
  }
}

function isActiveRun(state) {
  // MV3 workers can be stopped by Chrome. Treat very old "running" states as
  // stale so the button is not disabled forever after an interrupted job.
  return (
    state?.status === "running" &&
    Date.now() - Number(state.updatedAt || state.startedAt || 0) < STALE_RUNNING_MS
  );
}

function renderAnalysisState(state, tabId) {
  // The popup is a view over persisted state. It can close at any time, then
  // rebuild the UI from chrome.storage.local when the user opens it again.
  const out = document.getElementById("out");
  const outputArea = document.getElementById("parsed-output_bin");
  const resultsSection = document.getElementById("results-area_extension");
  const sourcesSection = document.getElementById("sources-area_extension");
  const sourcesOutput = document.getElementById("sources-output_bin");
  const thinkingArea = document.getElementById("thinking-area");
  const thinkingText = document.getElementById("thinking-text");

  if (!out || !outputArea || !resultsSection || !sourcesSection || !sourcesOutput) {
    return;
  }

  const status = state?.status || "";
  const stage = normalizeText(state?.stage, "");
  const result = state?.result || state?.partialResult || null;

  if (status === "running" && isActiveRun(state)) {
    out.textContent = stage || "Analysis running...";
    setSectionVisible(thinkingArea, true);
    if (thinkingText) {
      thinkingText.textContent = stage || "AI is reading the page...";
    }
  } else {
    setSectionVisible(thinkingArea, false);
  }

  if (status === "error") {
    setSectionVisible(resultsSection, true);
    setSectionVisible(sourcesSection, false);
    setOutputMessage(outputArea, normalizeText(state.error, "Analysis failed."), true);
    out.textContent = normalizeText(state.error, "Analysis failed.");
    return;
  }

  if (!result?.ai_result) {
    setSectionVisible(resultsSection, false);
    setSectionVisible(sourcesSection, false);
    if (status === "running") {
      out.textContent = "Previous analysis stopped before finishing. Run it again.";
    } else if (!status) {
      out.textContent = "";
    }
    return;
  }

  const highlights = normalizeHighlights(result.ai_result.highlights);
  setSectionVisible(resultsSection, true);
  renderResult(outputArea, result.ai_result, {
    onHighlightClick: (phrase) => highlightPhrasesInTab(tabId, highlights, phrase)
  });
  renderSources(sourcesOutput, sourcesSection, result.ai_research || {});

  if (status === "partial") {
    out.textContent = state.researchError
      ? `Bias done. ${state.researchError}`
      : "Bias analysis complete. Sources unavailable.";
    return;
  }

  out.textContent = status === "complete" ? "Analysis complete." : stage;
}

async function refreshHealth() {
  // Measure health latency from the popup because the backend can only report
  // process uptime after it has already woken up.
  const healthOutput = document.getElementById("health-output");
  if (!healthOutput) {
    return;
  }

  const startedAt = performance.now();
  try {
    const response = await fetchBackend("/health", { method: "GET" });
    const elapsedMs = Math.round(performance.now() - startedAt);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(normalizeText(data.detail, "Backend health check failed."));
    }

    const uptime = Number(data.uptime_seconds);
    const uptimeText = Number.isFinite(uptime) ? `${uptime.toFixed(1)}s uptime` : "uptime unknown";
    const recentText = data.recent_process_start ? "fresh start" : "warm";
    healthOutput.textContent = `Backend ${recentText} • ${elapsedMs}ms response • ${uptimeText}`;
  } catch (error) {
    healthOutput.textContent = `Backend unavailable: ${normalizeText(error.message, "health check failed")}`;
  }
}

async function startAnalysis(tab, key) {
  // Start the durable worker job and return control to the popup immediately.
  // Future progress arrives through storage updates watched below.
  const out = document.getElementById("out");
  const analyzeButton = document.getElementById("analyze");
  const url = String(tab.url ?? "");

  if (!url.startsWith("http")) {
    if (out) out.textContent = "Active tab URL is not valid for analysis.";
    return;
  }

  capturedUrl = url;
  if (analyzeButton) analyzeButton.disabled = true;
  if (out) out.textContent = "Starting analysis...";

  await sendRuntimeMessage({
    type: "FACTGPT_START_ANALYSIS",
    url,
    tabId: Number.isInteger(tab.id) ? tab.id : null
  });

  const saved = await getSavedAnalysis(key);
  renderAnalysisState(saved || { status: "running", stage: "Starting analysis..." }, tab.id);
}

function watchAnalysisState(key, tabId) {
  // Storage events are the main live update path while the popup is open.
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "local" || !changes[key]) {
      return;
    }
    renderAnalysisState(changes[key].newValue, tabId);
  });

  // Polling covers missed storage events and handles popup reopen cases.
  const pollId = setInterval(async () => {
    const saved = await getSavedAnalysis(key);
    renderAnalysisState(saved, tabId);
    const status = saved?.status || "";
    const analyzeButton = document.getElementById("analyze");
    if (analyzeButton) {
      analyzeButton.disabled = isActiveRun(saved);
    }
    if (status && status !== "running") {
      clearInterval(pollId);
    }
  }, POLL_MS);
}

async function initPopup() {
  const analyzeButton = document.getElementById("analyze");
  const out = document.getElementById("out");

  if (!analyzeButton || !out) {
    console.error("Popup UI failed to initialize: missing required elements.");
    return;
  }

  try {
    const tab = await getActiveTab();
    const url = String(tab.url ?? "");
    const key = await getAnalysisKey(url);
    const saved = await getSavedAnalysis(key);

    // Hydrate from the last saved run before wiring the button, so returning
    // to the same page immediately shows cached progress or completed results.
    renderAnalysisState(saved, tab.id);
    watchAnalysisState(key, tab.id);

    analyzeButton.disabled = isActiveRun(saved);
    analyzeButton.addEventListener("click", () => startAnalysis(tab, key));
  } catch (error) {
    out.textContent = `Error: ${error.message}`;
  }

  refreshHealth();
}

initPopup();
