import { parseText } from "./parseScript.js";

export let capturedUrl = "";

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
  async function runAnalysis() {
    out.textContent = "Reading active tab URL...";

    try {
      const tab = await getActiveTab();
      const url = String(tab.url ?? "");
      const tabId = Number.isInteger(tab.id) ? tab.id : null;

      if (!url.startsWith("http")) {
        out.textContent = "Active tab URL is not valid for analysis.";
        return;
      }

      capturedUrl = url;

      // Fallback order: fast backend extraction -> current tab DOM text -> Playwright.
      let parseResult = await parseText(capturedUrl, (statusMessage) => {
        out.textContent = statusMessage;
      }, { tabId });

      if (!parseResult.ok && tabId !== null && shouldFallbackToDom(parseResult.error)) {
        out.textContent = "Site blocks direct fetch. Reading visible page text...";
        try {
          const tabText = await extractVisibleTextFromTab(tabId);
          if (tabText.length >= 200) {
            parseResult = await parseText(tabText, (statusMessage) => {
              out.textContent = statusMessage;
            }, { tabId });
          }
        } catch {
          // Keep the original extraction error if the DOM fallback cannot run.
        }
      }

      if (!parseResult.ok && shouldFallbackToPlaywright(parseResult.error)) {
        out.textContent = "Trying browser-rendered extraction (last resort)...";
        parseResult = await parseText(capturedUrl, (statusMessage) => {
          out.textContent = statusMessage;
        }, { extractEndpoint: "/extract-rendered", tabId });
      }

      if (!parseResult.ok) {
        out.textContent = parseResult.error || "Analysis failed.";
        return;
      }

      if (parseResult.partial) {
        out.textContent = parseResult.researchError
          ? `Bias done. ${parseResult.researchError}`
          : "Bias analysis complete. Sources unavailable.";
        return;
      }

      out.textContent = parseResult.completionMessage || "Analysis complete.";
    } catch (err) {
      out.textContent = `Error: ${err.message}`;
    }
  }

  analyzeButton.addEventListener("click", runAnalysis);
}
