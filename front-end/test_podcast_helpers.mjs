import assert from "node:assert/strict";
import {
  buildPodcastAnalysisKey,
  formatPodcastTimestamp,
  isResumablePodcastState,
  normalizePodcastPageUrl,
  normalizePodcastResult,
  normalizePodcastSegmentPage,
  podcastStageText
} from "./podcast.js";

assert.equal(
  normalizePodcastPageUrl("https://example.com/show/episode#player"),
  "https://example.com/show/episode"
);
assert.equal(
  buildPodcastAnalysisKey("https://example.com/show/episode#transcript"),
  "factgpt:v2:podcast:https://example.com/show/episode"
);

assert.deepEqual(normalizePodcastResult({
  status: "complete",
  result: {
    podcast: { title: "Episode 1", text: "large transcript", segments: [{ id: "s1" }] },
    ai_result: { bias_score: 4 },
    ai_research: { claims: [] },
    fact_opinion: { counts: { fact: 2 } }
  }
}), {
  podcast: { title: "Episode 1" },
  ai_result: { bias_score: 4 },
  ai_research: { claims: [] },
  fact_opinion: { counts: { fact: 2 } }
});

assert.deepEqual(normalizePodcastSegmentPage({
  segments: [{ id: "s1", speaker: "Host", text: "Hello" }, null],
  next_cursor: "100",
  has_more: true
}), {
  segments: [{ id: "s1", speaker: "Host", text: "Hello" }],
  nextCursor: "100",
  hasMore: true
});

assert.equal(formatPodcastTimestamp(null), "");
assert.equal(formatPodcastTimestamp(-1), "");
assert.equal(formatPodcastTimestamp(65.9), "1:05");
assert.equal(formatPodcastTimestamp(3661), "1:01:01");
assert.equal(podcastStageText("fetching-transcript"), "Downloading the publisher transcript...");
assert.equal(podcastStageText("custom stage"), "custom stage");
assert.equal(isResumablePodcastState({ jobId: "job-1", status: "running" }), true);
assert.equal(isResumablePodcastState({ jobId: "job-1", status: "complete" }), false);
assert.equal(isResumablePodcastState({ status: "queued" }), false);

console.log("Podcast helper tests passed.");
