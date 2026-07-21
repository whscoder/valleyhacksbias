import assert from "node:assert/strict";
import {
  buildArticleAnalysisKey,
  isResumableArticleState,
  normalizeArticleResult,
  normalizeArticleUrl
} from "./article.js";

assert.equal(
  normalizeArticleUrl("https://example.com/story#comments"),
  "https://example.com/story"
);
assert.equal(
  buildArticleAnalysisKey("https://example.com/story#comments"),
  "factgpt:v2:analysis:https://example.com/story"
);
assert.equal(isResumableArticleState({ status: "running", jobId: "job-1" }), true);
assert.equal(isResumableArticleState({ status: "starting", jobId: "" }), false);
assert.equal(isResumableArticleState({ status: "complete", jobId: "job-1" }), false);
assert.deepEqual(normalizeArticleResult({ result: { ai_result: { score: 1 } } }), {
  ai_result: { score: 1 },
  ai_research: {},
  fact_opinion: null
});

console.log("Article helper tests passed.");
