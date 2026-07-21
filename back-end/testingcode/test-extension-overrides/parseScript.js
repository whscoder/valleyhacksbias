// Test-only analysis pipeline records extraction metadata for Playwright reports.
import { BACKEND_BASE_URLS } from "./config.js";
import {
  renderFactOpinion as renderProductionFactOpinion,
  renderResult as renderProductionResult,
  renderSources as renderProductionSources
} from "./productionParseScript.js";

const THINKING_DOT_MS = 180;

function buildBackendUrl(baseUrl, endpoint) {
  const normalizedBase = String(baseUrl ?? "").replace(/\/+$/, "");
  const normalizedEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${normalizedBase}${normalizedEndpoint}`;
}

async function fetchBackend(endpoint, init) {
  let lastError = null;

  for (const baseUrl of BACKEND_BASE_URLS) {
    try {
      const response = await fetch(buildBackendUrl(baseUrl, endpoint), init);
      return response;
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError ?? new Error("No reachable backend URL is configured.");
}

// Converts arrays/values into a clean display string with a fallback.
function normalizeText(value, fallback = "N/A") {
  if (Array.isArray(value)) {
    const joined = value
      .map((item) => String(item ?? "").trim())
      .filter(Boolean)
      .join(", ");
    return joined || fallback;
  }

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

// Splits bullet-like model text into individual lines for UI rendering/tooltips.
function parseBulletLikeText(value) {
  const text = String(value ?? "").trim();
  if (!text) {
    return [];
  }

  return text
    .split(/\n+/)
    .map((line) => line.replace(/^[\s\-*•]+/, "").trim())
    .filter(Boolean);
}

// Deduplicates bias highlight phrases while preserving original order.
function normalizeHighlights(highlights) {
  if (!Array.isArray(highlights)) {
    return [];
  }

  const seen = new Set();
  const normalized = [];
  for (const item of highlights) {
    const phrase = String(item ?? "").trim();
    if (!phrase) {
      continue;
    }

    const lower = phrase.toLowerCase();
    if (seen.has(lower)) {
      continue;
    }

    seen.add(lower);
    normalized.push(phrase);
  }
  return normalized;
}

// Normalizes text into searchable words for simple phrase/reason matching.
function toWords(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 2);
}

// Picks the best explanation bullet for a highlighted phrase using word overlap.
function pickReasonForHighlight(phrase, reasonCandidates) {
  const phraseWords = new Set(toWords(phrase));
  if (!phraseWords.size || reasonCandidates.length === 0) {
    return "This phrasing may steer the reader emotionally instead of neutrally.";
  }

  let bestReason = "";
  let bestScore = -1;
  for (const candidate of reasonCandidates) {
    const words = toWords(candidate);
    if (!words.length) {
      continue;
    }

    let overlap = 0;
    for (const word of words) {
      if (phraseWords.has(word)) {
        overlap += 1;
      }
    }

    if (overlap > bestScore) {
      bestScore = overlap;
      bestReason = candidate;
    }
  }

  if (!bestReason) {
    return "This phrasing may steer the reader emotionally instead of neutrally.";
  }

  return bestReason;
}

// Controls the animated "thinking" status shown while the backend is processing.
function createThinkingController(thinkingArea, thinkingText) {
  if (!thinkingArea || !thinkingText) {
    return {
      setPhase() {},
      stop() {}
    };
  }

  let phase = "AI is reading the screen";
  let dots = 0;

  const render = () => {
    thinkingText.textContent = `${phase}${".".repeat(dots)}`;
  };

  thinkingArea.style.display = "block";
  render();
  const intervalId = setInterval(() => {
    dots = (dots + 1) % 4;
    render();
  }, THINKING_DOT_MS);

  return {
    setPhase(nextPhase) {
      phase = nextPhase;
      dots = 0;
      render();
    },
    stop() {
      clearInterval(intervalId);
      thinkingArea.style.display = "none";
    }
  };
}

// Resets the result area and shows a single status/error line.
function setOutputMessage(outputArea, message, isError = false) {
  outputArea.innerHTML = "";
  const line = document.createElement("p");
  line.className = "copy-block";
  if (isError) {
    line.classList.add("error-text");
  }
  line.textContent = message;
  outputArea.appendChild(line);
}

// Renders highlight chips and attaches tooltip reasons for each phrase.
function renderBiasHighlights(highlights, explanation) {
  const wrap = document.createElement("div");
  wrap.className = "bias-chip-grid";

  if (!highlights.length) {
    const noHighlights = document.createElement("p");
    noHighlights.className = "copy-block";
    noHighlights.textContent = "No strong bias keywords were returned.";
    wrap.appendChild(noHighlights);
    return wrap;
  }

  const reasonCandidates = parseBulletLikeText(explanation);

  for (const phrase of highlights) {
    const reason = pickReasonForHighlight(phrase, reasonCandidates);

    const chip = document.createElement("span");
    chip.className = "bias-chip";
    chip.tabIndex = 0;
    chip.textContent = phrase;

    const tooltip = document.createElement("span");
    tooltip.className = "chip-tooltip";
    tooltip.textContent = normalizeText(reason);

    chip.appendChild(tooltip);
    wrap.appendChild(chip);
  }

  return wrap;
}

// Builds a consistent "Label: value" paragraph for the results card.
function createLabeledCopy(label, value) {
  const block = document.createElement("p");
  block.className = "copy-block";

  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  block.appendChild(strong);
  block.appendChild(document.createTextNode(normalizeText(value)));
  return block;
}

// Renders the bias-analysis card returned by the backend.
function renderResult(outputArea, ai) {
  outputArea.innerHTML = "";

  const card = document.createElement("div");
  card.className = "result-card";

  const metric = document.createElement("p");
  metric.className = "metric-line";
  const metricLabel = document.createElement("strong");
  metricLabel.textContent = "Bias Score";
  const scorePill = document.createElement("span");
  scorePill.className = "score-pill";
  scorePill.textContent = `${normalizeText(ai.bias_score, "N/A")} / 10`;
  metric.append(metricLabel, scorePill);

  const explanation = createLabeledCopy("Explanation", ai.explanation);

  const highlightsHeader = document.createElement("p");
  highlightsHeader.className = "copy-block";
  const highlightsLabel = document.createElement("strong");
  highlightsLabel.textContent = "Bias Keywords";
  highlightsHeader.append(highlightsLabel, document.createTextNode(": hover for reasoning"));

  const highlights = normalizeHighlights(ai.highlights);
  const highlightsWrap = renderBiasHighlights(highlights, ai.explanation);

  const missing = createLabeledCopy("Missing Perspectives", ai.missing_perspectives);

  card.append(metric, explanation, highlightsHeader, highlightsWrap, missing);
  outputArea.appendChild(card);
}

function factOpinionPresentation(item) {
  const finalPrediction = item?.final_prediction || {};
  const excerpts = Array.isArray(finalPrediction.opinion_excerpts)
    ? finalPrediction.opinion_excerpts.filter((excerpt) => String(excerpt ?? "").length > 0)
    : [];
  if (finalPrediction.status !== "resolved") {
    return { badge: "Unresolved — not analyzed", kind: "unresolved", excerpts: [] };
  }
  if (finalPrediction.label === "opinion") {
    return { badge: "Opinion — ignored during analysis", kind: "opinion", excerpts: [] };
  }
  if (finalPrediction.label === "fact" && excerpts.length) {
    return { badge: "Fact + opinion wording", kind: "mixed", excerpts };
  }
  if (finalPrediction.label === "fact") {
    return { badge: "Fact", kind: "fact", excerpts: [] };
  }
  return { badge: "Unresolved — not analyzed", kind: "unresolved", excerpts: [] };
}

function appendHighlightedOpinionText(container, textValue, excerpts) {
  const text = String(textValue ?? "");
  const matches = [];
  for (const excerptValue of Array.from(new Set(excerpts))) {
    const excerpt = String(excerptValue ?? "");
    let fromIndex = 0;
    while (excerpt && fromIndex < text.length) {
      const start = text.indexOf(excerpt, fromIndex);
      if (start === -1) break;
      matches.push({ start, end: start + excerpt.length });
      fromIndex = start + excerpt.length;
    }
  }
  matches.sort((left, right) => left.start - right.start || right.end - left.end);
  let cursor = 0;
  for (const match of matches) {
    if (match.start < cursor) continue;
    if (match.start > cursor) {
      container.appendChild(document.createTextNode(text.slice(cursor, match.start)));
    }
    const mark = document.createElement("mark");
    mark.className = "opinion-excerpt";
    mark.textContent = text.slice(match.start, match.end);
    container.appendChild(mark);
    cursor = match.end;
  }
  if (cursor < text.length || !matches.length) {
    container.appendChild(document.createTextNode(text.slice(cursor)));
  }
}

function renderFactOpinion(outputArea, section, factOpinion) {
  outputArea.innerHTML = "";
  const items = Array.isArray(factOpinion?.items) ? factOpinion.items : [];
  if (!factOpinion || typeof factOpinion !== "object" || !items.length) {
    section.style.display = "none";
    return;
  }

  section.style.display = "block";
  const counts = factOpinion.counts || {};
  const countGrid = document.createElement("div");
  countGrid.className = "fact-opinion-counts";
  for (const [label, key] of [
    ["Fact", "fact"],
    ["Opinion", "opinion"],
    ["Unresolved", "unresolved"],
    ["OpenAI reviewed", "openai_reviewed"]
  ]) {
    const count = document.createElement("div");
    count.className = "fact-opinion-count";
    const strong = document.createElement("strong");
    strong.textContent = String(Number.isInteger(counts[key]) ? counts[key] : 0);
    count.append(strong, document.createTextNode(label));
    countGrid.appendChild(count);
  }

  const list = document.createElement("ol");
  list.className = "fact-opinion-list";
  for (const item of items) {
    const presentation = factOpinionPresentation(item);
    const row = document.createElement("li");
    row.className = `fact-opinion-item fact-opinion-item-${presentation.kind}`;
    const badge = document.createElement("span");
    badge.className = `fact-opinion-badge fact-opinion-badge-${presentation.kind}`;
    badge.textContent = presentation.badge;
    const text = document.createElement("p");
    text.className = "fact-opinion-text";
    appendHighlightedOpinionText(text, item?.text, presentation.excerpts);
    row.append(badge, text);
    const explanationValue = String(item?.final_prediction?.explanation ?? "").trim();
    if (explanationValue) {
      const explanation = document.createElement("p");
      explanation.className = "fact-opinion-explanation";
      explanation.textContent = explanationValue;
      row.appendChild(explanation);
    }
    list.appendChild(row);
  }
  outputArea.append(countGrid, list);
}

// Accepts only http/https URLs for external source links.
function validUrl(urlValue) {
  try {
    const parsed = new URL(String(urlValue ?? ""));
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return "";
    }
    return parsed.toString();
  } catch {
    return "";
  }
}

// Chooses a readable source title, falling back to hostname.
function safeSourceTitle(url, title) {
  const cleanTitle = normalizeText(title, "").trim();
  if (cleanTitle) {
    return cleanTitle;
  }
  try {
    return new URL(url).hostname;
  } catch {
    return "Source";
  }
}

// Collects and deduplicates citation links from current and legacy research formats.
function collectSourceLinks(aiResearch) {
  const collected = [];
  const seen = new Set();

  const pushSource = (candidate, contextText) => {
    if (!candidate || typeof candidate !== "object") {
      return;
    }

    const sourceUrl = validUrl(candidate.url);
    if (!sourceUrl || seen.has(sourceUrl)) {
      return;
    }

    seen.add(sourceUrl);
    collected.push({
      title: safeSourceTitle(sourceUrl, candidate.title),
      url: sourceUrl,
      hint: normalizeText(contextText, "External source")
    });
  };

  if (Array.isArray(aiResearch?.claims)) {
    for (const claimEntry of aiResearch.claims) {
      if (!claimEntry || typeof claimEntry !== "object") {
        continue;
      }

      const claimContext = claimEntry.claim || claimEntry.evidence_summary || "Claim reference";
      if (Array.isArray(claimEntry.sources)) {
        for (const source of claimEntry.sources) {
          pushSource(source, claimContext);
        }
      }
    }
  }

  if (Array.isArray(aiResearch?.sources)) {
    for (const source of aiResearch.sources) {
      pushSource(source, aiResearch.evidence_summary || "Research source");
    }
  }

  return collected;
}

// Opens a cited source in a new browser tab (extension-safe when available).
function openExternalSource(url) {
  if (typeof chrome !== "undefined" && chrome?.tabs?.create) {
    chrome.tabs.create({ url });
    return;
  }

  window.open(url, "_blank", "noopener,noreferrer");
}

// Renders the source list section only when research links are available.
function renderSources(sourcesOutput, sourcesSection, aiResearch) {
  sourcesOutput.innerHTML = "";
  const links = collectSourceLinks(aiResearch);

  if (!links.length) {
    sourcesSection.style.display = "none";
    return;
  }

  sourcesSection.style.display = "block";

  const list = document.createElement("ul");
  list.className = "source-list";

  for (const link of links) {
    const item = document.createElement("li");
    item.className = "source-item";

    const anchor = document.createElement("a");
    anchor.href = link.url;
    anchor.className = "source-link";
    anchor.textContent = link.title;
    anchor.addEventListener("click", (event) => {
      event.preventDefault();
      openExternalSource(link.url);
    });

    const hint = document.createElement("p");
    hint.className = "source-hint";
    hint.textContent = link.hint;

    item.append(anchor, hint);
    list.appendChild(item);
  }

  sourcesOutput.appendChild(list);
}

function countResearchSources(aiResearch) {
  let total = 0;

  if (Array.isArray(aiResearch?.claims)) {
    for (const claim of aiResearch.claims) {
      if (Array.isArray(claim?.sources)) {
        total += claim.sources.length;
      }
    }
  }

  if (Array.isArray(aiResearch?.sources)) {
    total += aiResearch.sources.length;
  }

  return total;
}

// Main frontend pipeline: extract text (if needed), run bias scan, then fetch sources.
export async function parseText(url, updateStatus = () => {}, options = {}) {
  // Allow the popup to choose lightweight extraction first and Playwright only as a last resort.
  const extractEndpoint = typeof options.extractEndpoint === "string" && options.extractEndpoint
    ? options.extractEndpoint
    : "/extract";
  const skipResearch = options.skipResearch === true;
  const outputArea = document.getElementById("parsed-output_bin");
  const resultsSection = document.getElementById("results-area_extension");
  const factOpinionSection = document.getElementById("fact-opinion-area_extension");
  const factOpinionOutput = document.getElementById("fact-opinion-output_bin");
  const sourcesSection = document.getElementById("sources-area_extension");
  const sourcesOutput = document.getElementById("sources-output_bin");
  const thinkingArea = document.getElementById("thinking-area");
  const thinkingText = document.getElementById("thinking-text");

  if (
    !outputArea ||
    !resultsSection ||
    !factOpinionSection ||
    !factOpinionOutput ||
    !sourcesSection ||
    !sourcesOutput
  ) {
    return { ok: false, error: "Extension UI is missing required containers." };
  }

  if (!url) {
    return { ok: false, error: "URL not captured. Please run Analyze again." };
  }

  const setStatus = (message) => {
    updateStatus(message);
  };

  const fail = (message, statusMessage = message, extra = {}) => {
    setOutputMessage(outputArea, message, true);
    setStatus(statusMessage);
    return { ok: false, error: message, ...extra };
  };

  resultsSection.style.display = "block";
  factOpinionSection.style.display = "none";
  factOpinionOutput.innerHTML = "";
  sourcesSection.style.display = "none";
  sourcesOutput.innerHTML = "";
  setOutputMessage(outputArea, "Preparing scan...");
  setStatus("Preparing scan...");

  const thinking = createThinkingController(thinkingArea, thinkingText);
  let finalRawText = String(url);
  const metadata = {
    extractEndpoint,
    extractMethod: "",
    extractedTextChars: 0,
    extractedTextPreview: "",
    inputKind: url.startsWith("http") ? "url" : "text",
    skipResearch,
    biasScore: null,
    highlightCount: 0,
    explanationPresent: false,
    missingPerspectivesPresent: false,
    researchClaimCount: 0,
    researchSourceCount: 0
  };

  try {
    if (url.startsWith("http")) {
      // URL input path: ask the backend to extract readable article text first.
      thinking.setPhase("Scanning page structure");
      setStatus("Extracting readable text...");

      let extractData = {};
      const extractResponse = await fetchBackend(extractEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim() })
      });

      try {
        extractData = await extractResponse.json();
      } catch {
        extractData = {};
      }

      if (!extractResponse.ok) {
        return fail(
          normalizeText(extractData.detail, "Failed to extract text"),
          "Extraction failed.",
          {
            metadata: {
              ...metadata,
              errorStage: "extract",
              extractStatus: extractResponse.status
            }
          }
        );
      }

      finalRawText = normalizeArticleText(extractData.text, "");
      metadata.extractMethod = normalizeText(extractData.method, "unknown");
      metadata.extractedTextChars = finalRawText.length;
      metadata.extractedTextPreview = finalRawText.slice(0, 200);
      if (finalRawText.length < 200) {
        return fail(
          "Not enough readable text found on this page.",
          "Page text too short for analysis.",
          {
            metadata: {
              ...metadata,
              errorStage: "extract",
              extractStatus: extractResponse.status
            }
          }
        );
      }
    } else {
      metadata.extractMethod = "provided-text";
      metadata.extractedTextChars = finalRawText.length;
      metadata.extractedTextPreview = finalRawText.slice(0, 200);
    }
    // Raw text path: popup may pass visible tab text directly after DOM fallback.

    thinking.setPhase("Evaluating language bias");
    setStatus("Classifying passages and analyzing bias...");
    const biasResponse = await fetchBackend("/analyze-bias", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: finalRawText,
        title: "Article Analysis"
      })
    });

    let biasResult = {};
    try {
      biasResult = await biasResponse.json();
    } catch {
      biasResult = {};
    }

    if (!biasResponse.ok) {
      return fail(
        normalizeText(biasResult.detail, "Bias analysis failed."),
        "Bias analysis failed.",
        {
          metadata: {
            ...metadata,
            errorStage: "bias",
            biasStatus: biasResponse.status
          }
        }
      );
    }

    // Show the bias result immediately even if source-research fails later.
    const aiBiasResult = biasResult.ai_result || {};
    const factOpinionResult = biasResult.fact_opinion || null;
    metadata.biasScore = Number.isInteger(aiBiasResult.bias_score) ? aiBiasResult.bias_score : null;
    metadata.highlightCount = Array.isArray(aiBiasResult.highlights) ? aiBiasResult.highlights.length : 0;
    metadata.explanationPresent = Boolean(String(aiBiasResult.explanation ?? "").trim());
    metadata.missingPerspectivesPresent = Boolean(String(aiBiasResult.missing_perspectives ?? "").trim());
    metadata.factOpinionCounts = factOpinionResult?.counts || {};
    renderProductionResult(outputArea, aiBiasResult, {
      factOpinion: factOpinionResult
    });
    renderProductionFactOpinion(factOpinionOutput, factOpinionSection, factOpinionResult);

    if (skipResearch) {
      renderProductionSources(sourcesOutput, sourcesSection, {}, factOpinionResult);
      setStatus("Bias-only test complete.");
      return {
        ok: true,
        result: {
          ai_result: aiBiasResult,
          ai_research: {},
          fact_opinion: factOpinionResult
        },
        completionMessage: "Bias-only test complete.",
        metadata
      };
    }

    thinking.setPhase("Finding sources");
    setStatus("Bias complete. Gathering sources...");
    const researchResponse = await fetchBackend("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: finalRawText,
        title: "Article Analysis",
        ...(factOpinionResult ? { fact_opinion: factOpinionResult } : {}),
        ...(aiBiasResult ? { bias_result: aiBiasResult } : {})
      })
    });

    let researchResult = {};
    try {
      researchResult = await researchResponse.json();
    } catch {
      researchResult = {};
    }

    if (!researchResponse.ok) {
      // Research is optional for UX; return a partial success so bias results still display.
      renderProductionSources(sourcesOutput, sourcesSection, {}, factOpinionResult);
      setStatus("Bias complete. Sources unavailable right now.");
      return {
        ok: true,
        partial: true,
        result: {
          ai_result: aiBiasResult,
          ai_research: {},
          fact_opinion: factOpinionResult
        },
        researchError: normalizeText(researchResult.detail, "Research failed."),
        metadata: {
          ...metadata,
          researchStatus: researchResponse.status,
          researchSkipped: false
        }
      };
    }

    const aiResearchResult = researchResult.ai_research || {};
    const finalFactOpinion = researchResult.fact_opinion || factOpinionResult;
    renderProductionFactOpinion(factOpinionOutput, factOpinionSection, finalFactOpinion);
    renderProductionSources(
      sourcesOutput,
      sourcesSection,
      aiResearchResult,
      finalFactOpinion
    );
    setStatus("Analysis complete.");
    return {
      ok: true,
      result: {
        ai_result: aiBiasResult,
        ai_research: aiResearchResult,
        fact_opinion: finalFactOpinion
      },
      metadata: {
        ...metadata,
        researchStatus: researchResponse.status,
        researchSkipped: false,
        researchClaimCount: Array.isArray(aiResearchResult.claims) ? aiResearchResult.claims.length : 0,
        researchSourceCount: countResearchSources(aiResearchResult)
      }
    };
  } catch {
    // Network/backend startup issue (e.g., local FastAPI server not running).
    return fail(
      "Error connecting to AI. Is your terminal running?",
      "Backend connection error.",
      {
        metadata: {
          ...metadata,
          errorStage: "network"
        }
      }
    );
  } finally {
    // Always stop the animated loading state, including on failures.
    thinking.stop();
  }
}
