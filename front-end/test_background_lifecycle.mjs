import assert from "node:assert/strict";

const runningKey = "factgpt:v2:analysis:https://example.com/running";
const storage = {
  [runningKey]: {
    status: "running",
    jobId: "existing-backend-job",
    runId: "existing-run",
    stage: "Classifying article passages..."
  }
};
let messageListener = null;
let intervalCount = 0;
const originalSetInterval = globalThis.setInterval;
globalThis.setInterval = (...args) => {
  intervalCount += 1;
  return originalSetInterval(...args);
};

globalThis.fetch = async (_url, init) => {
  const body = JSON.parse(init.body);
  assert.equal(body.client_request_id, "run-123");
  assert.match(body.text, /visible article text/i);
  return new Response(JSON.stringify({
    job_id: "backend-job-123",
    status: "queued",
    stage: "Article analysis queued.",
    created_at: "2026-07-21T00:00:00Z",
    reused: false
  }), { status: 202, headers: { "Content-Type": "application/json" } });
};

globalThis.chrome = {
  runtime: {
    onMessage: {
      addListener(listener) {
        messageListener = listener;
      }
    }
  },
  scripting: {
    async executeScript() {
      return [{ result: `Visible article text ${"content ".repeat(40)}` }];
    }
  },
  storage: {
    local: {
      async get(key) {
        if (key === null) return { ...storage };
        if (typeof key === "string") return { [key]: storage[key] };
        return { ...storage };
      },
      async set(values) {
        Object.assign(storage, values);
      },
      async remove(keys) {
        for (const key of Array.isArray(keys) ? keys : [keys]) delete storage[key];
      }
    }
  }
};

await import(`./background.js?test=${Date.now()}`);
await new Promise((resolve) => setTimeout(resolve, 0));

assert.equal(storage[runningKey].status, "running");
assert.equal(storage[runningKey].jobId, "existing-backend-job");
assert.equal(intervalCount, 0, "the service worker must not start a polling timer");
assert.equal(typeof messageListener, "function");

const response = await new Promise((resolve, reject) => {
  const keepChannelOpen = messageListener({
    type: "FACTGPT_START_ANALYSIS",
    url: "https://example.com/article",
    tabId: 7,
    runId: "run-123"
  }, {}, resolve);
  assert.equal(keepChannelOpen, true);
  setTimeout(() => reject(new Error("Background response timed out.")), 1000);
});

assert.equal(response.ok, true);
assert.equal(response.state.jobId, "backend-job-123");
assert.equal(response.state.status, "queued");
assert.equal(intervalCount, 0, "completed request must leave no worker polling timer");

globalThis.setInterval = originalSetInterval;
console.log("Background lifecycle tests passed.");
