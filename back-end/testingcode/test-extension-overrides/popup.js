import { parseText } from "./parseScript.js";
export let capturedUrl = "";
// Test hook: Playwright can pass a target page URL directly into the popup.
const testParams = new URLSearchParams(window.location.search);
const testUrlOverride = String(testParams.get("testUrl") ?? "").trim();
const autoRun = testParams.get("autoRun") === "1";
const biasOnlyMode = testParams.get("biasOnly") === "1";

// Test hook: expose a simple machine-readable state so Playwright knows
// whether the popup is idle, running, complete, partial, or failed.
function setAutomationState(status, details = {}) {
  document.body.dataset.testStatus = status;
  window.__FACTGPT_TEST_STATE__ = { status, ...details };
}

// Test hook: when a test URL is provided, skip reading the active Chrome tab
// and analyze the supplied URL instead.
function getTargetConfig() {
  if (testUrlOverride.startsWith("http")) {
    return { url: testUrlOverride, tabId: null };
  }
  return null;
}

function buildAttempt(stage, result) {
  return {
    stage,
    ok: result?.ok === true,
    partial: result?.partial === true,
    error: String(result?.error ?? ""),
    metadata: result?.metadata || {}
  };
}

// Reads the currently focused tab so the popup knows what page to analyze.
async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab found.");
  return tab;
}

// Detects extraction-style errors where reading page DOM text is a good next step.
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

// Reuses the same extraction-error signal for the final Playwright fallback.
function shouldFallbackToPlaywright(errorMessage) {
  // Playwright is the last resort, so only use it for extraction-related failures.
  return shouldFallbackToDom(errorMessage);
}

// Pulls visible text directly from the active page when backend URL fetching fails.
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

const analyzeButton = document.getElementById("analyze");
const out = document.getElementById("out");

if (!analyzeButton || !out) {
  console.error("Popup UI failed to initialize: missing required elements.");
} else {
  // Shared runner used by both the real button click and test auto-run mode.
  async function runAnalysis() {
    setAutomationState("running");
    out.textContent = "Reading active tab URL...";

    try {
      const attempts = [];
      const testTarget = getTargetConfig();
      let url = "";
      let tabId = null;

      if (testTarget) {
        // Test hook: Playwright supplies the page URL through the popup query string.
        url = testTarget.url;
      } else {
        const tab = await getActiveTab();
        url = String(tab.url ?? "");
        tabId = Number.isInteger(tab.id) ? tab.id : null;
      }

      if (!url.startsWith("http")) {
        out.textContent = "Active tab URL is not valid for analysis.";
        setAutomationState("error", { message: out.textContent });
        return;
      }

      capturedUrl = url;
      // Fallback order: fast backend extraction -> current tab DOM text -> Playwright (last resort).
      let parseResult = await parseText(capturedUrl, (statusMessage) => {
        out.textContent = statusMessage;
      }, { skipResearch: biasOnlyMode });
      attempts.push(buildAttempt("direct", parseResult));

      if (!parseResult.ok && tabId !== null && shouldFallbackToDom(parseResult.error)) {
        out.textContent = "Site blocks direct fetch. Reading visible page text...";
        try {
          const tabText = await extractVisibleTextFromTab(tabId);
          if (tabText.length >= 200) {
            // Send extracted page text directly for analysis (no backend URL fetch needed).
            parseResult = await parseText(tabText, (statusMessage) => {
              out.textContent = statusMessage;
            }, { skipResearch: biasOnlyMode });
            attempts.push(buildAttempt("dom-fallback", parseResult));
          }
        } catch {
          // Keep original extraction error if DOM fallback cannot run.
        }
      }

      if (!parseResult.ok && shouldFallbackToPlaywright(parseResult.error)) {
        out.textContent = "Trying browser-rendered extraction (last resort)...";
        // Final fallback: ask backend to use Playwright rendering for hard sites.
        parseResult = await parseText(capturedUrl, (statusMessage) => {
          out.textContent = statusMessage;
        }, { extractEndpoint: "/extract-rendered", skipResearch: biasOnlyMode });
        attempts.push(buildAttempt("rendered-fallback", parseResult));
      }

      if (!parseResult.ok) {
        out.textContent = parseResult.error || "Analysis failed.";
        setAutomationState("error", {
          message: out.textContent,
          finalUrl: capturedUrl,
          attempts,
          metadata: parseResult.metadata || {}
        });
        return;
      }

      if (parseResult.partial) {
        out.textContent = parseResult.researchError
          ? `Bias done. ${parseResult.researchError}`
          : "Bias analysis complete. Sources unavailable.";
        setAutomationState("partial", {
          message: out.textContent,
          result: parseResult.result || {},
          finalUrl: capturedUrl,
          attempts,
          metadata: parseResult.metadata || {}
        });
        return;
      }

      out.textContent = parseResult.completionMessage || "Analysis complete.";
      setAutomationState("complete", {
        message: out.textContent,
        result: parseResult.result || {},
        finalUrl: capturedUrl,
        attempts,
        metadata: parseResult.metadata || {}
      });
    } catch (err) {
      out.textContent = `Error: ${err.message}`;
      setAutomationState("error", {
        message: out.textContent,
        metadata: { errorStage: "popup-exception" }
      });
    }
  }

  // Main popup action: capture page URL, then run analysis with ordered fallbacks.
  analyzeButton.addEventListener("click", runAnalysis);

  if (autoRun) {
    // Test hook: lets Playwright open the popup page and start analysis immediately.
    runAnalysis();
  } else {
    setAutomationState("idle");
  }
}
