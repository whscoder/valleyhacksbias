// Pure podcast response helpers shared by the service worker, popup, and Node tests.
export const PODCAST_ANALYSIS_PREFIX = "factgpt:v2:podcast:";

export function normalizePodcastPageUrl(value) {
  try {
    const url = new URL(String(value ?? ""));
    url.hash = "";
    return url.toString();
  } catch {
    return String(value ?? "").trim();
  }
}

export function buildPodcastAnalysisKey(pageUrl) {
  return `${PODCAST_ANALYSIS_PREFIX}${normalizePodcastPageUrl(pageUrl)}`;
}

export function normalizePodcastResult(payload) {
  const source = payload && typeof payload === "object" ? payload : {};
  const nested = source.result && typeof source.result === "object" ? source.result : source;
  const rawPodcast = nested.podcast && typeof nested.podcast === "object"
    ? nested.podcast
    : {};
  // The complete transcript is fetched page-by-page and must never be copied
  // into extension storage as part of the compact job result.
  const podcast = { ...rawPodcast };
  delete podcast.text;
  delete podcast.segments;
  delete podcast.transcript;
  return {
    podcast,
    ai_result: nested.ai_result && typeof nested.ai_result === "object" ? nested.ai_result : {},
    ai_research: nested.ai_research && typeof nested.ai_research === "object"
      ? nested.ai_research
      : {},
    fact_opinion: nested.fact_opinion && typeof nested.fact_opinion === "object"
      ? nested.fact_opinion
      : null
  };
}

export function normalizePodcastSegmentPage(payload) {
  const source = payload && typeof payload === "object" ? payload : {};
  const segments = Array.isArray(source.segments)
    ? source.segments.filter((segment) => segment && typeof segment === "object")
    : [];
  const nextCursor = source.next_cursor ?? source.nextCursor ?? null;
  return {
    segments,
    nextCursor,
    hasMore: typeof source.has_more === "boolean"
      ? source.has_more
      : (typeof source.hasMore === "boolean" ? source.hasMore : nextCursor !== null)
  };
}

export function formatPodcastTimestamp(value) {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "";
  }
  const whole = Math.floor(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const remainder = whole % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

export function podcastStageText(stage, fallback = "Podcast analysis running...") {
  const normalized = String(stage ?? "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  const labels = {
    queued: "Podcast queued...",
    discovering: "Finding the episode transcript or audio...",
    fetching_transcript: "Downloading the publisher transcript...",
    downloading_audio: "Downloading episode audio...",
    transcoding: "Preparing episode audio...",
    transcribing: "Transcribing speakers and timestamps...",
    classifying: "Classifying facts and opinions...",
    analyzing_bias: "Analyzing episode bias...",
    researching: "Checking the episode's factual claims...",
    finalizing: "Finalizing podcast results...",
    complete: "Podcast analysis complete."
  };
  return labels[normalized] || String(stage ?? "").trim() || fallback;
}

export function isResumablePodcastState(state) {
  const status = String(state?.status ?? "").toLowerCase();
  return Boolean(state?.jobId) && (status === "queued" || status === "running");
}
