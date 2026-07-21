import assert from "node:assert/strict";

const runningKey = "factgpt:v2:analysis:https://example.com/running";
const completeKey = "factgpt:v2:analysis:https://example.com/complete";
const storage = {
  [runningKey]: {
    status: "running",
    stage: "Classifying passages and analyzing bias...",
    startedAt: Date.now()
  },
  [completeKey]: {
    status: "complete",
    stage: "Analysis complete."
  }
};

globalThis.chrome = {
  runtime: {
    onMessage: { addListener() {} }
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

await import("./background.js");
for (let attempt = 0; attempt < 20 && storage[runningKey].status === "running"; attempt += 1) {
  await new Promise((resolve) => setTimeout(resolve, 1));
}

assert.equal(storage[runningKey].status, "error");
assert.equal(storage[runningKey].stage, "Previous analysis was interrupted.");
assert.match(storage[runningKey].error, /run it again/i);
assert.equal(storage[completeKey].status, "complete");

console.log("Background lifecycle tests passed.");
