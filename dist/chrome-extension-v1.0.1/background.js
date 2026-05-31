import {
  analyzeBiasText,
  extractArticleFromUrl,
  extractRenderedArticleFromUrl,
  researchText
} from "./backendClient.js";

const ANALYSIS_PREFIX = "factgpt:analysis:";
const LATEST_KEY = "factgpt:latestAnalysisKey";
const MIN_TEXT_CHARS = 200;

// In-memory dedupe for the current service-worker lifetime. The durable copy
// of progress/results lives in chrome.storage.local below.
const runningJobs = new Map();

function normalizeText(value, fallback = "") {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text || fallback;
}

function buildAnalysisKey(url) {
  // Results are keyed by normalized page URL so reopening the popup on the
  // same article can restore the latest saved analysis.
  try {
    const parsed = new URL(String(url ?? ""));
    parsed.hash = "";
    return `${ANALYSIS_PREFIX}${parsed.toString()}`;
  } catch {
    return `${ANALYSIS_PREFIX}${String(url ?? "").trim()}`;
  }
}

function normalizeUrlForCompare(url) {
  try {
    const parsed = new URL(String(url ?? ""));
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return String(url ?? "").trim();
  }
}

async function tabStillOnUrl(tabId, originalUrl) {
  // If the user navigated away, do not inject into the new page and accidentally
  // analyze text from the wrong article.
  if (!Number.isInteger(tabId)) {
    return false;
  }

  try {
    const tab = await chrome.tabs.get(tabId);
    return normalizeUrlForCompare(tab.url) === normalizeUrlForCompare(originalUrl);
  } catch {
    return false;
  }
}

function shouldFallbackToDom(errorMessage) {
  const text = String(errorMessage ?? "").toLowerCase();
  return (
    text.includes("failed to extract") ||
    text.includes("fetch failed") ||
    text.includes("network error fetching url") ||
    text.includes("bot protection") ||
    text.includes("blocked by bot protection") ||
    text.includes("not enough readable text") ||
    text.includes("timed out") ||
    text.includes("protocol error") ||
    text.includes("http2") ||
    text.includes("playwright extraction failed")
  );
}

async function saveAnalysisState(key, patch) {
  // Every meaningful stage is persisted so popup.js can be destroyed/reopened
  // without losing the visible progress or final backend response.
  const existing = (await chrome.storage.local.get(key))[key] || {};
  const next = {
    ...existing,
    ...patch,
    key,
    updatedAt: Date.now()
  };

  await chrome.storage.local.set({
    [key]: next,
    [LATEST_KEY]: key
  });
  return next;
}

async function extractVisibleTextFromTab(tabId) {
  if (!Number.isInteger(tabId)) {
    return "";
  }

  const [injection] = await chrome.scripting.executeScript({
    target: { tabId },
    func: () => {
      const clone = document.body ? document.body.cloneNode(true) : null;
      if (!clone) {
        return "";
      }

      const removeSelectors = [
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "nav",
        "footer",
        "header"
      ];

      for (const selector of removeSelectors) {
        clone.querySelectorAll(selector).forEach((node) => node.remove());
      }

      const preferredRoot = clone.querySelector("article, main, [role='main']");
      const text = (preferredRoot || clone).innerText || "";
      return text.replace(/\s+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
    }
  });

  return String(injection?.result ?? "").trim();
}

async function extractArticleText(url, tabId, key) {
  // Prefer the cheapest backend extraction first. The DOM and rendered paths are
  // fallbacks for pages that block direct fetches or render content with JS.
  try {
    await saveAnalysisState(key, { stage: "Extracting readable text..." });
    const data = await extractArticleFromUrl(url);
    const text = normalizeText(data.text, "");
    if (text.length >= MIN_TEXT_CHARS) {
      return text;
    }
    throw new Error("Not enough readable text found on this page.");
  } catch (error) {
    if (!shouldFallbackToDom(error.message)) {
      throw error;
    }
  }

  if (await tabStillOnUrl(tabId, url)) {
    await saveAnalysisState(key, { stage: "Reading visible page text..." });
    let domText = "";
    try {
      domText = await extractVisibleTextFromTab(tabId);
    } catch {
      // Some browser pages cannot be scripted; Playwright remains the last-resort path.
      domText = "";
    }
    if (domText.length >= MIN_TEXT_CHARS) {
      return domText;
    }
  }

  await saveAnalysisState(key, { stage: "Trying browser-rendered extraction..." });
  const rendered = await extractRenderedArticleFromUrl(url);
  const renderedText = normalizeText(rendered.text, "");
  if (renderedText.length < MIN_TEXT_CHARS) {
    throw new Error("Not enough readable text found on this page.");
  }
  return renderedText;
}

async function runAnalysisJob({ key, url, tabId }) {
  // The background worker owns long-running backend work. The popup only starts
  // the job and later reads this saved state, so closing the popup does not
  // immediately kill the analysis flow.
  await saveAnalysisState(key, {
    status: "running",
    stage: "Starting analysis...",
    url,
    tabId,
    startedAt: Date.now(),
    completedAt: null,
    error: "",
    researchError: "",
    partialResult: null,
    result: null
  });

  try {
    const articleText = await extractArticleText(url, tabId, key);

    await saveAnalysisState(key, { stage: "Running quick bias scan..." });
    const biasResult = await analyzeBiasText(articleText);
    const aiBias = biasResult.ai_result || {};

    // Save the bias result immediately; research can fail or take longer, but
    // the user should still get the core bias scan when they reopen the popup.
    await saveAnalysisState(key, {
      status: "running",
      stage: "Bias complete. Gathering sources...",
      partialResult: { ai_result: aiBias, ai_research: {} }
    });

    try {
      const researchResult = await researchText(articleText);

      await saveAnalysisState(key, {
        status: "complete",
        stage: "Analysis complete.",
        completedAt: Date.now(),
        result: {
          ai_result: aiBias,
          ai_research: researchResult.ai_research || {}
        },
        partialResult: null
      });
    } catch (error) {
      await saveAnalysisState(key, {
        status: "partial",
        stage: "Bias complete. Sources unavailable right now.",
        completedAt: Date.now(),
        result: { ai_result: aiBias, ai_research: {} },
        partialResult: null,
        researchError: normalizeText(error.message, "Research failed.")
      });
    }
  } catch (error) {
    await saveAnalysisState(key, {
      status: "error",
      stage: "Analysis failed.",
      completedAt: Date.now(),
      error: normalizeText(error.message, "Analysis failed.")
    });
  } finally {
    runningJobs.delete(key);
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  // Popup messages are the entrypoint into this worker. Keep responses immediate;
  // the async job reports progress through chrome.storage.local.
  if (message?.type === "FACTGPT_ANALYSIS_KEY") {
    sendResponse({ key: buildAnalysisKey(message.url) });
    return false;
  }

  if (message?.type === "FACTGPT_START_ANALYSIS") {
    const key = buildAnalysisKey(message.url);
    if (!runningJobs.has(key)) {
      const job = runAnalysisJob({ key, url: message.url, tabId: message.tabId });
      runningJobs.set(key, job);
    }
    sendResponse({ ok: true, key });
    return false;
  }

  return false;
});
