// MV3 service worker: creates backend jobs, persists IDs, then becomes idle.
import {
  createArticleJob,
  createPodcastJob,
  getPodcastJob,
  getPodcastSegments,
} from "./backendClient.js";
import {
  buildArticleAnalysisKey,
  normalizeArticleUrl
} from "./article.js";
import {
  buildPodcastAnalysisKey,
  normalizePodcastResult,
  podcastStageText
} from "./podcast.js";

const LEGACY_ANALYSIS_PREFIXES = ["factgpt:analysis:"];
const LATEST_KEY = "factgpt:v2:latestAnalysisKey";
const LEGACY_LATEST_KEYS = ["factgpt:latestAnalysisKey"];
const MIN_TEXT_CHARS = 200;
const MAX_TEXT_CHARS = 50_000;
const PODCAST_POLL_MS = 1500;

// In-memory dedupe for the current service-worker lifetime. The durable copy
// of progress/results lives in chrome.storage.local below.
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

const startupReady = purgeLegacyAnalysisCache().catch((error) => {
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

async function startArticleJob({ key, url, tabId, runId }) {
  await startupReady;
  await saveAnalysisState(key, {
    mode: "article",
    runId,
    jobId: "",
    status: "starting",
    stage: "Sending article to the analysis service...",
    progress: 0,
    url: normalizeArticleUrl(url),
    tabId,
    startedAt: Date.now(),
    completedAt: null,
    error: "",
    retryable: true,
    result: null
  });

  let visibleText = "";
  try {
    visibleText = normalizeArticleText(await extractVisibleTextFromTab(tabId), "");
  } catch {
    // Restricted tabs cannot be scripted. The backend performs URL extraction.
  }

  const created = await createArticleJob(
    url,
    runId,
    visibleText.length >= MIN_TEXT_CHARS ? visibleText.slice(0, MAX_TEXT_CHARS) : ""
  );
  const jobId = normalizeText(created?.job_id, "");
  if (!jobId) throw new Error("The analysis service did not return a job ID.");

  const current = (await chrome.storage.local.get(key))[key] || {};
  if (current.runId !== runId) return current;
  return saveAnalysisState(key, {
    jobId,
    status: String(created?.status || "queued").toLowerCase(),
    stage: normalizeText(created?.stage, "Article analysis queued."),
    progress: 0,
    createdAt: created?.created_at || new Date().toISOString(),
    reused: Boolean(created?.reused)
  });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  // Popup messages are the entrypoint into this worker. Keep responses immediate;
  // the async job reports progress through chrome.storage.local.
  if (message?.type === "FACTGPT_ANALYSIS_KEY") {
    sendResponse({ key: buildArticleAnalysisKey(message.url) });
    return false;
  }

  if (message?.type === "FACTGPT_PODCAST_ANALYSIS_KEY") {
    sendResponse({ key: buildPodcastAnalysisKey(message.url) });
    return false;
  }

  if (message?.type === "FACTGPT_START_ANALYSIS") {
    const key = buildArticleAnalysisKey(message.url);
    const runId = normalizeText(message.runId, crypto.randomUUID());
    startArticleJob({ key, url: message.url, tabId: message.tabId, runId })
      .then((state) => sendResponse({ ok: true, key, state }))
      .catch(async (error) => {
        const current = (await chrome.storage.local.get(key))[key] || {};
        if (current.runId === runId) {
          await saveAnalysisState(key, {
            status: "error",
            stage: "Analysis could not start.",
            completedAt: Date.now(),
            retryable: true,
            error: normalizeText(error.message, "Analysis could not start.")
          });
        }
        sendResponse({ ok: false, key, error: normalizeText(error.message, "Analysis could not start.") });
      });
    return true;
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
