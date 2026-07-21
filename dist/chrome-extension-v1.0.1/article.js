export const ARTICLE_ANALYSIS_PREFIX = "factgpt:v2:analysis:";

export function normalizeArticleUrl(url) {
  try {
    const parsed = new URL(String(url ?? ""));
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return String(url ?? "").trim();
  }
}

export function buildArticleAnalysisKey(url) {
  return `${ARTICLE_ANALYSIS_PREFIX}${normalizeArticleUrl(url)}`;
}

export function articleStageText(stage, fallback = "Article analysis running...") {
  const text = String(stage ?? "").replace(/\s+/g, " ").trim();
  return text || fallback;
}

export function isResumableArticleState(state) {
  const status = String(state?.status || "").toLowerCase();
  return Boolean(state?.jobId) && (status === "queued" || status === "running");
}

export function normalizeArticleResult(response) {
  const result = response?.result && typeof response.result === "object"
    ? response.result
    : {};
  return {
    ai_result: result.ai_result || {},
    ai_research: result.ai_research || {},
    fact_opinion: result.fact_opinion || null
  };
}
