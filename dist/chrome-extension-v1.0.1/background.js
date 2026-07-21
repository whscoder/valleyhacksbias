// MV3 service worker: owns extraction/analysis jobs and persists their lifecycle.
import {
  analyzeBiasText,
  createPodcastJob,
  extractArticleFromUrl,
  extractRenderedArticleFromUrl,
  getPodcastJob,
  getPodcastSegments,
  researchText
} from "./backendClient.js";
import {
  buildPodcastAnalysisKey,
  normalizePodcastResult,
  podcastStageText
} from "./podcast.js";

const LEGACY_ANALYSIS_PREFIXES = ["factgpt:analysis:"];
const ANALYSIS_PREFIX = "factgpt:v2:analysis:";
const LATEST_KEY = "factgpt:v2:latestAnalysisKey";
const LEGACY_LATEST_KEYS = ["factgpt:latestAnalysisKey"];
const MIN_TEXT_CHARS = 200;
const PODCAST_POLL_MS = 1500;

// In-memory dedupe for the current service-worker lifetime. The durable copy
// of progress/results lives in chrome.storage.local below.
const runningJobs = new Map();
const runningPodcastJobs = new Map();

async function purgeLegacyAnalysisCache() {
  const allData = await chrome.storage.local.get(null);
  const keysToRemove = Object.keys(allData).filter((key) => (
    LEGACY_LATEST_KEYS.includes(key) ||
    LEGACY_ANALYSIS_PREFIXES.some((prefix) => key.startsWith(prefix))
  ));

  if (keysToRemove.length) {
    await chrome.storage.local.remove(keysToRemove);
  }
}

async function recoverInterruptedArticleJobs() {
  // Article work lives in this service worker, unlike backend-owned podcast
  // jobs. If the worker starts with a persisted running article state, there is
  // no in-memory promise left to finish it, so make retry available immediately.
  const allData = await chrome.storage.local.get(null);
  const recovered = {};
  for (const [key, value] of Object.entries(allData)) {
    if (!key.startsWith(ANALYSIS_PREFIX) || !value || typeof value !== "object") {
      continue;
    }
    if (!["running", "queued"].includes(String(value.status || "").toLowerCase())) {
      continue;
    }
    recovered[key] = {
      ...value,
      status: "error",
      stage: "Previous analysis was interrupted.",
      completedAt: Date.now(),
      updatedAt: Date.now(),
      error: "The previous analysis stopped before finishing. Run it again."
    };
  }
  if (Object.keys(recovered).length) {
    await chrome.storage.local.set(recovered);
  }
}

const startupReady = Promise.all([
  purgeLegacyAnalysisCache(),
  recoverInterruptedArticleJobs()
]).catch((error) => {
  console.debug("Analysis cache startup cleanup skipped:", normalizeText(error.message, "cleanup failed"));
});

function normalizeText(value, fallback = "") {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text || fallback;
}

function normalizeArticleText(value, fallback = "") {
  const text = String(value ?? "")
    .replace(/\r\n?/g, "\n")
    .replace(/[^\S\n]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
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

async function savePodcastState(key, patch) {
  const existing = (await chrome.storage.local.get(key))[key] || {};
  const next = {
    ...existing,
    ...patch,
    mode: "podcast",
    key,
    updatedAt: Date.now()
  };
  await chrome.storage.local.set({ [key]: next });
  return next;
}

async function discoverPodcastHints(tabId) {
  if (!Number.isInteger(tabId)) {
    return { feed_urls: [], transcript_urls: [], audio_urls: [] };
  }

  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        const buckets = {
          feed_urls: new Set(),
          transcript_urls: new Set(),
          audio_urls: new Set()
        };
        const add = (bucket, rawUrl) => {
          if (!rawUrl) return;
          try {
            const url = new URL(String(rawUrl), document.baseURI);
            if (url.protocol === "http:" || url.protocol === "https:") {
              buckets[bucket].add(url.toString());
            }
          } catch {
            // Ignore malformed publisher metadata; the backend validates every URL again.
          }
        };

        document.querySelectorAll("link[rel~='alternate'], a[href]").forEach((node) => {
          const href = node.getAttribute("href");
          const type = String(node.getAttribute("type") || "").toLowerCase();
          const rel = String(node.getAttribute("rel") || "").toLowerCase();
          const label = String(node.textContent || node.getAttribute("aria-label") || "").toLowerCase();
          if (type.includes("rss") || type.includes("atom") || /(^|\/)feed(?:\.|\/|$)|\.rss(?:$|\?)/i.test(href || "")) {
            add("feed_urls", href);
          }
          if (
            rel.includes("transcript") ||
            type.includes("text/vtt") ||
            type.includes("subrip") ||
            /transcript|show notes/.test(label) ||
            /transcript|\.vtt(?:$|\?)|\.srt(?:$|\?)/i.test(href || "")
          ) {
            add("transcript_urls", href);
          }
          if (
            type.startsWith("audio/") ||
            rel.includes("enclosure") && type.includes("audio") ||
            /\.(?:mp3|m4a|aac|wav|ogg|opus)(?:$|\?)/i.test(href || "")
          ) {
            add("audio_urls", href);
          }
        });

        document.querySelectorAll("audio[src], audio source[src], video source[type^='audio/']").forEach((node) => {
          add("audio_urls", node.getAttribute("src"));
        });
        document.querySelectorAll(
          "meta[property='og:audio'], meta[property='og:audio:url'], meta[name='twitter:player:stream']"
        ).forEach((node) => add("audio_urls", node.getAttribute("content")));

        const visitJsonLd = (value) => {
          if (!value || typeof value !== "object") return;
          if (Array.isArray(value)) {
            value.forEach(visitJsonLd);
            return;
          }
          const type = Array.isArray(value["@type"]) ? value["@type"].join(" ") : value["@type"];
          if (/AudioObject|PodcastEpisode/i.test(String(type || ""))) {
            add("audio_urls", value.contentUrl || value.url);
            if (value.associatedMedia) visitJsonLd(value.associatedMedia);
          }
          if (value.transcript && /^https?:/i.test(String(value.transcript))) {
            add("transcript_urls", value.transcript);
          }
          if (value["@graph"]) visitJsonLd(value["@graph"]);
        };
        document.querySelectorAll("script[type='application/ld+json']").forEach((node) => {
          try {
            visitJsonLd(JSON.parse(node.textContent || "null"));
          } catch {
            // Invalid structured data is common and should not block podcast discovery.
          }
        });

        return Object.fromEntries(
          Object.entries(buckets).map(([key, urls]) => [key, Array.from(urls).slice(0, 5)])
        );
      }
    });
    return injection?.result || { feed_urls: [], transcript_urls: [], audio_urls: [] };
  } catch {
    return { feed_urls: [], transcript_urls: [], audio_urls: [] };
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function pollPodcastJob({ key, jobId }) {
  while (true) {
    const response = await getPodcastJob(jobId);
    const status = String(response?.status || "running").toLowerCase();
    const stage = podcastStageText(response?.stage, "Podcast analysis running...");
    const progress = Number(response?.progress);

    if (status === "complete") {
      await savePodcastState(key, {
        status: "complete",
        stage,
        progress: 100,
        completedAt: Date.now(),
        retryable: false,
        result: normalizePodcastResult(response)
      });
      return;
    }
    if (status === "failed") {
      const backendError = response?.error;
      const message = typeof backendError === "object"
        ? backendError.message
        : backendError;
      await savePodcastState(key, {
        status: "error",
        stage: "Podcast analysis failed.",
        completedAt: Date.now(),
        retryable: response?.retryable !== false,
        error: normalizeText(message, "Podcast analysis failed. Run it again.")
      });
      return;
    }

    await savePodcastState(key, {
      status: status === "queued" ? "queued" : "running",
      stage,
      progress: Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : null
    });
    await delay(PODCAST_POLL_MS);
  }
}

async function runPodcastJob({ key, url, tabId, existingJobId = "" }) {
  try {
    let jobId = existingJobId;
    if (!jobId) {
      await savePodcastState(key, {
        status: "running",
        stage: "Finding podcast sources on this page...",
        url,
        tabId,
        startedAt: Date.now(),
        completedAt: null,
        progress: 0,
        error: "",
        retryable: false,
        result: null
      });
      const hints = await discoverPodcastHints(tabId);
      const created = await createPodcastJob(url, hints);
      jobId = normalizeText(created?.job_id, "");
      if (!jobId) {
        throw new Error("The podcast service did not return a job ID.");
      }
      await savePodcastState(key, {
        jobId,
        status: String(created?.status || "queued").toLowerCase(),
        stage: podcastStageText(created?.stage || "queued"),
        createdAt: created?.created_at || new Date().toISOString(),
        reused: Boolean(created?.reused)
      });
    }
    await pollPodcastJob({ key, jobId });
  } catch (error) {
    const current = (await chrome.storage.local.get(key))[key] || {};
    await savePodcastState(key, {
      status: "error",
      stage: "Podcast analysis interrupted.",
      completedAt: Date.now(),
      retryable: true,
      error: normalizeText(
        error.message,
        current.jobId
          ? "The podcast job could not be resumed. Run it again."
          : "Podcast analysis could not start."
      )
    });
  } finally {
    runningPodcastJobs.delete(key);
  }
}

function ensurePodcastJob({ key, url, tabId, jobId = "" }) {
  if (!runningPodcastJobs.has(key)) {
    const job = runPodcastJob({ key, url, tabId, existingJobId: jobId });
    runningPodcastJobs.set(key, job);
  }
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

async function extractArticleText(url, tabId, key, signal) {
  // Prefer the cheapest backend extraction first. The DOM and rendered paths are
  // fallbacks for pages that block direct fetches or render content with JS.
  try {
    await saveAnalysisState(key, { stage: "Extracting readable text..." });
    const data = await extractArticleFromUrl(url, signal);
    const text = normalizeArticleText(data.text, "");
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
  const rendered = await extractRenderedArticleFromUrl(url, signal);
  const renderedText = normalizeArticleText(rendered.text, "");
  if (renderedText.length < MIN_TEXT_CHARS) {
    throw new Error("Not enough readable text found on this page.");
  }
  return renderedText;
}

async function runAnalysisJob({ key, url, tabId, signal }) {
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
    const articleText = await extractArticleText(url, tabId, key, signal);

    await saveAnalysisState(key, { stage: "Classifying passages and analyzing bias..." });
    const biasResult = await analyzeBiasText(articleText, "Article Analysis", signal);
    const aiBias = biasResult.ai_result || {};
    const factOpinion = biasResult.fact_opinion || null;

    // Save the bias result immediately; research can fail or take longer, but
    // the user should still get the core bias scan when they reopen the popup.
    await saveAnalysisState(key, {
      status: "running",
      stage: "Bias complete. Gathering sources...",
      partialResult: {
        ai_result: aiBias,
        ai_research: {},
        fact_opinion: factOpinion
      }
    });

    try {
      const researchResult = await researchText(
        articleText,
        "Article Analysis",
        factOpinion,
        aiBias,
        signal
      );
      const finalFactOpinion = researchResult.fact_opinion || factOpinion;

      await saveAnalysisState(key, {
        status: "complete",
        stage: "Analysis complete.",
        completedAt: Date.now(),
        result: {
          ai_result: aiBias,
          ai_research: researchResult.ai_research || {},
          fact_opinion: finalFactOpinion
        },
        partialResult: null
      });
    } catch (error) {
      if (signal?.aborted) {
        throw error;
      }
      await saveAnalysisState(key, {
        status: "partial",
        stage: "Bias complete. Sources unavailable right now.",
        completedAt: Date.now(),
        result: {
          ai_result: aiBias,
          ai_research: {},
          fact_opinion: factOpinion
        },
        partialResult: null,
        researchError: normalizeText(error.message, "Research failed.")
      });
    }
  } catch (error) {
    if (signal?.aborted) {
      return;
    }
    await saveAnalysisState(key, {
      status: "error",
      stage: "Analysis failed.",
      completedAt: Date.now(),
      error: normalizeText(error.message, "Analysis failed.")
    });
  }
}

function startAnalysisJob({ key, url, tabId }) {
  // A manual Analyze click is also a recovery action. Abort any request that
  // survived in this worker after its persisted state became stale, then start
  // a fresh run. The identity guard prevents the old promise from deleting the
  // replacement job when its abort finishes.
  runningJobs.get(key)?.controller.abort();
  const controller = new AbortController();
  let promise;
  promise = runAnalysisJob({
    key,
    url,
    tabId,
    signal: controller.signal
  }).finally(() => {
    if (runningJobs.get(key)?.promise === promise) {
      runningJobs.delete(key);
    }
  });
  runningJobs.set(key, { controller, promise });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  // Popup messages are the entrypoint into this worker. Keep responses immediate;
  // the async job reports progress through chrome.storage.local.
  if (message?.type === "FACTGPT_ANALYSIS_KEY") {
    sendResponse({ key: buildAnalysisKey(message.url) });
    return false;
  }

  if (message?.type === "FACTGPT_PODCAST_ANALYSIS_KEY") {
    sendResponse({ key: buildPodcastAnalysisKey(message.url) });
    return false;
  }

  if (message?.type === "FACTGPT_START_ANALYSIS") {
    const key = buildAnalysisKey(message.url);
    startupReady.then(() => {
      startAnalysisJob({ key, url: message.url, tabId: message.tabId });
    });
    sendResponse({ ok: true, key });
    return false;
  }


  if (message?.type === "FACTGPT_START_PODCAST_ANALYSIS") {
    const key = buildPodcastAnalysisKey(message.url);
    ensurePodcastJob({ key, url: message.url, tabId: message.tabId });
    sendResponse({ ok: true, key });
    return false;
  }

  if (message?.type === "FACTGPT_RESUME_PODCAST_ANALYSIS") {
    const key = buildPodcastAnalysisKey(message.url);
    ensurePodcastJob({
      key,
      url: message.url,
      tabId: message.tabId,
      jobId: normalizeText(message.jobId, "")
    });
    sendResponse({ ok: true, key });
    return false;
  }

  if (message?.type === "FACTGPT_GET_PODCAST_SEGMENTS") {
    getPodcastSegments(message.jobId, message.cursor, message.limit)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((error) => sendResponse({
        ok: false,
        error: normalizeText(error.message, "Transcript could not be loaded.")
      }));
    return true;
  }

  return false;
});
