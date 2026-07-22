// Small, DOM-light helpers for keeping popup result rendering stable across polls.

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.keys(value).sort().reduce((result, key) => {
      result[key] = stableValue(value[key]);
      return result;
    }, {});
  }
  return value;
}

export function resultRenderIdentity(mode, state) {
  return JSON.stringify({
    mode: String(mode || ""),
    runId: String(state?.runId || ""),
    jobId: String(state?.jobId || "")
  });
}

// Deliberately excludes progress, stage, timestamps, and status. Those belong in
// the lightweight status refresh; only a changed result should replace result DOM.
export function resultRenderSignature(mode, state) {
  const result = state?.result || state?.partialResult || null;
  if (!result?.ai_result) return "";
  return JSON.stringify(stableValue({
    identity: JSON.parse(resultRenderIdentity(mode, state)),
    result
  }));
}

export function captureOpenDecisionDetailIds(root) {
  if (!root?.querySelectorAll) return [];
  return Array.from(root.querySelectorAll(".fact-opinion-item[data-item-id] > details[open]"))
    .map((details) => String(details.parentElement?.dataset?.itemId || ""))
    .filter(Boolean);
}

export function restoreOpenDecisionDetailIds(root, ids) {
  if (!root?.querySelectorAll || !Array.isArray(ids) || !ids.length) return;
  const openIds = new Set(ids);
  for (const details of root.querySelectorAll(".fact-opinion-item[data-item-id] > details")) {
    if (openIds.has(String(details.parentElement?.dataset?.itemId || ""))) {
      details.open = true;
    }
  }
}
