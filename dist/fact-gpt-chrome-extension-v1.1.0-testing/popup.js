// Popup controller: starts worker jobs and renders their durable storage state.
import { getArticleJob, warmBackend } from "./backendClient.js";
import {
  articleStageText,
  isResumableArticleState,
  normalizeArticleResult
} from "./article.js";
import {
  highlightPhrasesInTab,
  normalizeHighlights,
  normalizeText,
  renderFactOpinion,
  renderResult,
  renderSources,
  setOutputMessage
} from "./parseScript.js";
import {
  formatPodcastTimestamp,
  isResumablePodcastState,
  normalizePodcastSegmentPage,
  podcastStageText
} from "./podcast.js";

export let capturedUrl = "";

const POLL_MS = 800;
const ARTICLE_POLL_MS = 2500;
const SELECTED_MODE_KEY = "factgpt:v2:selectedMode";
const MODES = Object.freeze({ article: "article", podcast: "podcast" });

let activeMode = MODES.article;
let activeTab = null;
let modeKeys = { article: "", podcast: "" };
let transcriptState = { jobId: "", cursor: "", loading: false, hasMore: false };
let articlePollInFlight = false;
let articleLastPolledAt = 0;

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

async function getAnalysisKey(url, mode = MODES.article) {
  const response = await sendRuntimeMessage({
    type: mode === MODES.podcast
      ? "FACTGPT_PODCAST_ANALYSIS_KEY"
      : "FACTGPT_ANALYSIS_KEY",
    url
  });
  return response?.key || "";
}

async function getSavedAnalysis(key) {
  if (!key) return null;
  const data = await chrome.storage.local.get(key);
  return data[key] || null;
}

function setSectionVisible(section, visible) {
  if (section) section.style.display = visible ? "block" : "none";
}

function isActiveRun(state) {
  const status = String(state?.status || "").toLowerCase();
  return status === "starting" || status === "running" || status === "queued";
}

function isPodcastActiveRun(state) {
  const status = String(state?.status || "").toLowerCase();
  return status === "running" || status === "queued";
}

function resultElements() {
  return {
    out: document.getElementById("out"),
    outputArea: document.getElementById("parsed-output_bin"),
    resultsSection: document.getElementById("results-area_extension"),
    factOpinionSection: document.getElementById("fact-opinion-area_extension"),
    factOpinionOutput: document.getElementById("fact-opinion-output_bin"),
    sourcesSection: document.getElementById("sources-area_extension"),
    sourcesOutput: document.getElementById("sources-output_bin"),
    thinkingArea: document.getElementById("thinking-area"),
    thinkingText: document.getElementById("thinking-text"),
    transcriptSection: document.getElementById("podcast-transcript-area")
  };
}

function renderAnalysisState(state, tabId) {
  const ui = resultElements();
  if (!ui.out || !ui.outputArea || !ui.resultsSection || !ui.factOpinionSection ||
      !ui.factOpinionOutput || !ui.sourcesSection || !ui.sourcesOutput) return;

  setSectionVisible(ui.transcriptSection, false);
  const status = state?.status || "";
  const stage = normalizeText(state?.stage, "");
  const result = state?.result || state?.partialResult || null;

  if (status === "running" && isActiveRun(state)) {
    ui.out.textContent = stage || "Analysis running...";
    setSectionVisible(ui.thinkingArea, true);
    if (ui.thinkingText) ui.thinkingText.textContent = stage || "AI is reading the page...";
  } else {
    setSectionVisible(ui.thinkingArea, false);
  }

  if (status === "error") {
    setSectionVisible(ui.resultsSection, true);
    setSectionVisible(ui.factOpinionSection, false);
    setSectionVisible(ui.sourcesSection, false);
    setOutputMessage(ui.outputArea, normalizeText(state.error, "Analysis failed."), true);
    ui.out.textContent = normalizeText(state.error, "Analysis failed.");
    return;
  }

  if (!result?.ai_result) {
    setSectionVisible(ui.resultsSection, false);
    setSectionVisible(ui.factOpinionSection, false);
    setSectionVisible(ui.sourcesSection, false);
    if (status === "running") {
      ui.out.textContent = "Previous analysis stopped before finishing. Run it again.";
    } else if (!status) {
      ui.out.textContent = "";
    }
    return;
  }

  const highlights = normalizeHighlights(result.ai_result.highlights);
  setSectionVisible(ui.resultsSection, true);
  renderResult(ui.outputArea, result.ai_result, {
    factOpinion: result.fact_opinion,
    onHighlightClick: (phrase) => highlightPhrasesInTab(tabId, highlights, phrase)
  });
  renderFactOpinion(ui.factOpinionOutput, ui.factOpinionSection, result.fact_opinion);
  renderSources(ui.sourcesOutput, ui.sourcesSection, result.ai_research || {}, result.fact_opinion);

  if (status === "partial") {
    ui.out.textContent = state.researchError
      ? `Bias done. ${state.researchError}`
      : "Bias analysis complete. Sources unavailable.";
    return;
  }
  ui.out.textContent = status === "complete" ? "Analysis complete." : stage;
}

function resetTranscript(jobId = "") {
  transcriptState = { jobId, cursor: "", loading: false, hasMore: false };
  const list = document.getElementById("podcast-segments");
  const details = document.getElementById("podcast-transcript-details");
  const status = document.getElementById("podcast-seek-status");
  const loadMore = document.getElementById("podcast-load-more");
  if (list) list.replaceChildren();
  if (details) details.open = false;
  if (status) status.textContent = "";
  if (loadMore) loadMore.style.display = "none";
}

function podcastMetaText(podcast) {
  const title = normalizeText(podcast?.title, "Podcast episode");
  const source = normalizeText(podcast?.transcript_source || podcast?.source, "");
  const duration = formatPodcastTimestamp(podcast?.duration_seconds ?? podcast?.duration);
  return [title, duration ? `Duration ${duration}` : "", source ? `Source: ${source}` : ""]
    .filter(Boolean)
    .join(" · ");
}

function renderPodcastState(state) {
  const ui = resultElements();
  if (!ui.out || !ui.outputArea || !ui.resultsSection || !ui.factOpinionSection ||
      !ui.factOpinionOutput || !ui.sourcesSection || !ui.sourcesOutput) return;

  const status = String(state?.status || "").toLowerCase();
  const stage = podcastStageText(state?.stage, "Podcast analysis running...");
  const result = state?.result || null;
  const active = isPodcastActiveRun(state);
  setSectionVisible(ui.thinkingArea, active);
  if (ui.thinkingText && active) ui.thinkingText.textContent = stage;

  if (status === "error") {
    setSectionVisible(ui.resultsSection, true);
    setSectionVisible(ui.factOpinionSection, false);
    setSectionVisible(ui.sourcesSection, false);
    setSectionVisible(ui.transcriptSection, false);
    const suffix = state?.retryable ? " Run the podcast analysis again to retry." : "";
    const message = `${normalizeText(state?.error, "Podcast analysis failed.")}${suffix}`;
    setOutputMessage(ui.outputArea, message, true);
    ui.out.textContent = message;
    return;
  }

  if (!result?.ai_result || status !== "complete") {
    setSectionVisible(ui.resultsSection, false);
    setSectionVisible(ui.factOpinionSection, false);
    setSectionVisible(ui.sourcesSection, false);
    setSectionVisible(ui.transcriptSection, false);
    const progress = Number(state?.progress);
    ui.out.textContent = status
      ? `${stage}${Number.isFinite(progress) ? ` (${Math.round(progress)}%)` : ""}`
      : "";
    return;
  }

  setSectionVisible(ui.resultsSection, true);
  const highlightLocations = Array.isArray(result.podcast?.highlight_locations)
    ? result.podcast.highlight_locations
    : [];
  renderResult(ui.outputArea, result.ai_result, {
    factOpinion: result.fact_opinion,
    highlightActionLabel: "click to seek its timestamp",
    includeOpinionBias: true,
    onHighlightClick: async (_phrase, index) => {
      const location = highlightLocations[index];
      const timestamp = Number(location?.start_seconds);
      const feedback = document.getElementById("podcast-seek-status");
      if (!Number.isFinite(timestamp)) {
        if (feedback) feedback.textContent = "This transcript highlight has no publisher timestamp.";
        return;
      }
      const moved = await seekPodcastMedia(activeTab?.id, timestamp);
      const label = formatPodcastTimestamp(timestamp);
      if (feedback) {
        feedback.textContent = moved
          ? `Moved the page player to ${label}.`
          : `${label} — use this timestamp in the page's podcast player.`;
      }
    }
  });
  renderFactOpinion(ui.factOpinionOutput, ui.factOpinionSection, result.fact_opinion);
  renderSources(ui.sourcesOutput, ui.sourcesSection, result.ai_research || {}, result.fact_opinion);
  setSectionVisible(ui.transcriptSection, true);
  const meta = document.getElementById("podcast-meta");
  if (meta) meta.textContent = podcastMetaText(result.podcast || {});
  if (transcriptState.jobId !== state.jobId) resetTranscript(state.jobId || "");
  ui.out.textContent = "Podcast analysis complete.";
}

async function seekPodcastMedia(tabId, seconds) {
  if (!Number.isInteger(tabId) || !Number.isFinite(Number(seconds))) return false;
  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId },
      args: [Number(seconds)],
      func: (targetSeconds) => {
        const media = Array.from(document.querySelectorAll("audio, video"))
          .find((node) => Number.isFinite(node.duration) && node.duration > 0) ||
          document.querySelector("audio, video");
        if (!media) return false;
        try {
          media.currentTime = Math.max(0, Math.min(targetSeconds, media.duration || targetSeconds));
          media.scrollIntoView({ behavior: "smooth", block: "center" });
          return true;
        } catch {
          return false;
        }
      }
    });
    return Boolean(injection?.result);
  } catch {
    return false;
  }
}

function segmentClassification(segment) {
  const classification = segment.classification;
  const value = (
    classification && typeof classification === "object"
      ? classification.final_prediction?.label ?? classification.resolved_label ?? classification.label ?? classification.classification
      : classification
  ) ?? segment.label ?? segment.fact_opinion?.label;
  return normalizeText(value, "");
}

function appendPodcastSegments(segments) {
  const list = document.getElementById("podcast-segments");
  if (!list) return;
  for (const segment of segments) {
    const item = document.createElement("li");
    item.className = "podcast-segment";
    const heading = document.createElement("div");
    heading.className = "podcast-segment-heading";
    const speaker = document.createElement("span");
    speaker.textContent = normalizeText(segment.speaker, "Unknown speaker");
    heading.appendChild(speaker);
    const timestamp = formatPodcastTimestamp(segment.start_seconds ?? segment.start);
    if (timestamp) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "podcast-timestamp";
      button.textContent = timestamp;
      button.title = "Seek the episode player to this timestamp";
      button.addEventListener("click", async () => {
        const feedback = document.getElementById("podcast-seek-status");
        const moved = await seekPodcastMedia(activeTab?.id, Number(segment.start_seconds ?? segment.start));
        if (feedback) {
          feedback.textContent = moved
            ? `Moved the page player to ${timestamp}.`
            : `${timestamp} — this page's player is not directly accessible. Use this timestamp in the player.`;
        }
      });
      heading.appendChild(button);
    }
    const text = document.createElement("p");
    text.className = "podcast-segment-text";
    text.textContent = normalizeText(segment.text, "Transcript text unavailable.");
    item.append(heading, text);
    const label = segmentClassification(segment);
    if (label) {
      const badge = document.createElement("span");
      badge.className = "podcast-segment-label";
      badge.textContent = label;
      item.appendChild(badge);
    }
    list.appendChild(item);
  }
}

async function loadTranscriptPage() {
  if (!transcriptState.jobId || transcriptState.loading) return;
  transcriptState.loading = true;
  const loadMore = document.getElementById("podcast-load-more");
  if (loadMore) {
    loadMore.disabled = true;
    loadMore.textContent = "Loading transcript...";
  }
  try {
    const response = await sendRuntimeMessage({
      type: "FACTGPT_GET_PODCAST_SEGMENTS",
      jobId: transcriptState.jobId,
      cursor: transcriptState.cursor,
      limit: 100
    });
    if (!response?.ok) throw new Error(response?.error || "Transcript could not be loaded.");
    const page = normalizePodcastSegmentPage(response.result);
    appendPodcastSegments(page.segments);
    transcriptState.cursor = page.nextCursor;
    transcriptState.hasMore = page.hasMore;
    if (loadMore) loadMore.style.display = page.hasMore ? "block" : "none";
  } catch (error) {
    const feedback = document.getElementById("podcast-seek-status");
    if (feedback) feedback.textContent = normalizeText(error.message, "Transcript could not be loaded.");
  } finally {
    transcriptState.loading = false;
    if (loadMore) {
      loadMore.disabled = false;
      loadMore.textContent = "Load more transcript";
    }
  }
}

async function refreshHealth() {
  try {
    await warmBackend();
  } catch (error) {
    console.debug("Service warm-up skipped:", normalizeText(error.message, "health check failed"));
  }
}

async function startCurrentMode() {
  const out = document.getElementById("out");
  const button = document.getElementById("analyze");
  const url = String(activeTab?.url ?? "");
  if (!url.startsWith("http")) {
    if (out) out.textContent = "Active tab URL is not valid for analysis.";
    return;
  }
  capturedUrl = url;
  if (button) button.disabled = true;
  if (out) out.textContent = activeMode === MODES.podcast
    ? "Starting podcast analysis..."
    : "Starting analysis...";

  const response = await sendRuntimeMessage({
    type: activeMode === MODES.podcast
      ? "FACTGPT_START_PODCAST_ANALYSIS"
      : "FACTGPT_START_ANALYSIS",
    url,
    tabId: Number.isInteger(activeTab?.id) ? activeTab.id : null,
    ...(activeMode === MODES.article ? { runId: crypto.randomUUID() } : {})
  });
  if (!response?.ok) throw new Error(response?.error || "Analysis could not start.");
  const saved = await getSavedAnalysis(modeKeys[activeMode]);
  renderCurrentState(saved || {
    status: "running",
    stage: activeMode === MODES.podcast ? "queued" : "Starting analysis..."
  });
}

function renderCurrentState(state) {
  const button = document.getElementById("analyze");
  if (button) {
    button.disabled = activeMode === MODES.podcast
      ? isPodcastActiveRun(state)
      : isActiveRun(state);
  }
  if (activeMode === MODES.podcast) renderPodcastState(state);
  else renderAnalysisState(state, activeTab?.id);
}

async function resumePodcastIfNeeded(state) {
  if (!isResumablePodcastState(state)) return;
  try {
    await sendRuntimeMessage({
      type: "FACTGPT_RESUME_PODCAST_ANALYSIS",
      url: String(activeTab?.url ?? ""),
      tabId: Number.isInteger(activeTab?.id) ? activeTab.id : null,
      jobId: state.jobId
    });
  } catch (error) {
    console.debug("Podcast resume skipped:", normalizeText(error.message, "resume failed"));
  }
}

async function refreshArticleIfNeeded(state, force = false) {
  if (!isResumableArticleState(state) || articlePollInFlight) return;
  const now = Date.now();
  if (!force && now - articleLastPolledAt < ARTICLE_POLL_MS) return;
  articlePollInFlight = true;
  articleLastPolledAt = now;
  const key = modeKeys.article;
  const expectedRunId = state.runId;
  const expectedJobId = state.jobId;

  try {
    const response = await getArticleJob(expectedJobId);
    const current = await getSavedAnalysis(key);
    if (current?.runId !== expectedRunId || current?.jobId !== expectedJobId) return;

    const status = String(response?.status || "running").toLowerCase();
    const patch = {
      status: status === "failed" ? "error" : status,
      stage: articleStageText(response?.stage),
      progress: Number.isFinite(Number(response?.progress)) ? Number(response.progress) : null,
      backendUpdatedAt: response?.updated_at || null,
      lastCheckedAt: Date.now()
    };
    if (status === "complete") {
      patch.completedAt = Date.now();
      patch.progress = 100;
      patch.retryable = false;
      patch.result = normalizeArticleResult(response);
      patch.error = "";
    } else if (status === "failed") {
      const backendError = response?.error;
      const message = normalizeText(
        typeof backendError === "object" ? backendError.message : backendError,
        "Article analysis failed. Run it again."
      );
      patch.completedAt = Date.now();
      patch.retryable = true;
      const partial = normalizeArticleResult(response);
      if (Object.keys(partial.ai_result).length) {
        patch.status = "partial";
        patch.stage = "Bias complete. Sources unavailable right now.";
        patch.result = partial;
        patch.researchError = message;
        patch.error = "";
      } else {
        patch.error = message;
      }
    }
    await chrome.storage.local.set({
      [key]: { ...current, ...patch, key, updatedAt: Date.now() }
    });
  } catch (error) {
    const current = await getSavedAnalysis(key);
    if (current?.runId !== expectedRunId || current?.jobId !== expectedJobId) return;
    const message = normalizeText(error.message, "Article job could not be checked.");
    if (/not found|backend may have restarted|expired|HTTP 404|HTTP 410/i.test(message)) {
      await chrome.storage.local.set({
        [key]: {
          ...current,
          status: "error",
          stage: "Article job is no longer available.",
          completedAt: Date.now(),
          updatedAt: Date.now(),
          retryable: true,
          error: `${message} Run the analysis again.`
        }
      });
    } else {
      // A temporary polling failure does not mean the backend-owned job stopped.
      console.debug("Article status check skipped:", message);
    }
  } finally {
    articlePollInFlight = false;
  }
}

async function switchMode(mode) {
  activeMode = mode === MODES.podcast ? MODES.podcast : MODES.article;
  await chrome.storage.local.set({ [SELECTED_MODE_KEY]: activeMode });
  for (const candidate of Object.values(MODES)) {
    const option = document.getElementById(`mode-${candidate}`);
    const selected = candidate === activeMode;
    option?.classList.toggle("mode-option-active", selected);
    option?.setAttribute("aria-pressed", String(selected));
  }
  const button = document.getElementById("analyze");
  if (button) {
    button.textContent = activeMode === MODES.podcast
      ? "Analyze Current Podcast"
      : "Analyze Current Page";
  }
  resetTranscript();
  const saved = await getSavedAnalysis(modeKeys[activeMode]);
  renderCurrentState(saved);
  if (activeMode === MODES.podcast) await resumePodcastIfNeeded(saved);
  else await refreshArticleIfNeeded(saved, true);
}

function watchStoredStates() {
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "local") return;
    const key = modeKeys[activeMode];
    if (key && changes[key]) renderCurrentState(changes[key].newValue);
  });

  setInterval(async () => {
    const saved = await getSavedAnalysis(modeKeys[activeMode]);
    renderCurrentState(saved);
    if (activeMode === MODES.podcast) await resumePodcastIfNeeded(saved);
    else await refreshArticleIfNeeded(saved);
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
    activeTab = await getActiveTab();
    const url = String(activeTab.url ?? "");
    [modeKeys.article, modeKeys.podcast] = await Promise.all([
      getAnalysisKey(url, MODES.article),
      getAnalysisKey(url, MODES.podcast)
    ]);
    const settings = await chrome.storage.local.get(SELECTED_MODE_KEY);
    const savedMode = settings[SELECTED_MODE_KEY] === MODES.podcast
      ? MODES.podcast
      : MODES.article;

    document.getElementById("mode-article")?.addEventListener("click", () => {
      switchMode(MODES.article).catch((error) => { out.textContent = `Error: ${error.message}`; });
    });
    document.getElementById("mode-podcast")?.addEventListener("click", () => {
      switchMode(MODES.podcast).catch((error) => { out.textContent = `Error: ${error.message}`; });
    });
    analyzeButton.addEventListener("click", () => {
      startCurrentMode().catch((error) => {
        analyzeButton.disabled = false;
        out.textContent = `Error: ${normalizeText(error.message, "Analysis could not start.")}`;
      });
    });
    document.getElementById("podcast-transcript-details")?.addEventListener("toggle", (event) => {
      if (event.currentTarget.open && !document.getElementById("podcast-segments")?.children.length) {
        loadTranscriptPage();
      }
    });
    document.getElementById("podcast-load-more")?.addEventListener("click", loadTranscriptPage);
    watchStoredStates();
    await switchMode(savedMode);
  } catch (error) {
    out.textContent = `Error: ${error.message}`;
  }
  refreshHealth();
}

initPopup();
