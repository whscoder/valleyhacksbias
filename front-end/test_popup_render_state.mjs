import assert from "node:assert/strict";

import {
  captureOpenDecisionDetailIds,
  restoreOpenDecisionDetailIds,
  resultRenderIdentity,
  resultRenderSignature
} from "./popupRenderState.js";

const base = {
  runId: "run-1",
  jobId: "job-1",
  status: "complete",
  stage: "complete",
  updatedAt: 1,
  result: {
    ai_result: { summary: "Bias result", highlights: [] },
    fact_opinion: { items: [{ id: "fact-1" }] },
    ai_research: { claims: [] }
  }
};

const initial = resultRenderSignature("article", base);
assert.equal(initial, resultRenderSignature("article", {
  ...base, status: "running", stage: "Researching", progress: 45, updatedAt: 2
}), "poll-only status changes must not rebuild result DOM");
assert.notEqual(initial, resultRenderSignature("article", {
  ...base, result: { ...base.result, ai_research: { claims: [{ claim: "Updated" }] } }
}), "research payload updates must rerender results");
assert.notEqual(initial, resultRenderSignature("podcast", base), "mode changes must reset result UI");
assert.notEqual(resultRenderIdentity("article", base), resultRenderIdentity("article", {
  ...base, runId: "run-2"
}), "new runs must not inherit open details");
assert.equal(resultRenderSignature("article", { status: "error" }), "", "errors clear result state");

function detail(id, open = false) {
  return { open, parentElement: { dataset: { itemId: id } } };
}
const first = detail("fact-1", true);
const second = detail("fact-2", false);
const root = {
  querySelectorAll(selector) {
    if (selector.endsWith("[open]")) return [first];
    return [first, second];
  }
};
const saved = captureOpenDecisionDetailIds(root);
assert.deepEqual(saved, ["fact-1"], "open decision details are captured before a real rerender");
first.open = false;
restoreOpenDecisionDetailIds(root, saved);
assert.equal(first.open, true, "open decision details are restored after the same run rerenders");
assert.equal(second.open, false, "closed details stay closed");

console.log("Popup render-state interaction tests passed.");
