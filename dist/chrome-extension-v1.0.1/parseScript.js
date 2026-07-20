// Popup presentation helpers plus the older direct-from-popup analysis pipeline.
import {
  analyzeBiasText,
  extractArticleFromUrl,
  extractRenderedArticleFromUrl,
  researchText
} from "./backendClient.js";

const THINKING_DOT_MS = 180;

// Converts arrays/values into a clean display string with a fallback.
export function normalizeText(value, fallback = "N/A") {
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

function shortenText(value, maxChars = 150) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (!text || text.length <= maxChars) {
    return text;
  }

  const sentences = text
    .match(/[^.!?]+[.!?]?/g)
    ?.map((sentence) => sentence.trim())
    .filter(Boolean) || [];

  if (sentences.length) {
    let combined = "";
    for (const sentence of sentences) {
      const candidate = combined ? `${combined} ${sentence}` : sentence;
      if (candidate.length > maxChars) {
        break;
      }
      combined = candidate;
    }

    if (combined) {
      return combined;
    }

    if (sentences[0].length <= maxChars + 24) {
      return sentences[0];
    }
  }

  const clipped = text.slice(0, maxChars);
  const lastSpace = clipped.lastIndexOf(" ");
  return `${(lastSpace > 50 ? clipped.slice(0, lastSpace) : clipped).trim()}...`;
}

function simplifyBulletText(value, maxItems = 2, maxChars = 140) {
  const bullets = parseBulletLikeText(value);
  if (!bullets.length) {
    return shortenText(value, maxChars);
  }

  return bullets
    .slice(0, maxItems)
    .map((bullet) => shortenText(bullet, maxChars))
    .join(" • ");
}

// Deduplicates bias highlight phrases while preserving original order.
export function normalizeHighlights(highlights) {
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

function normalizeHighlightReasonMap(highlightReasons) {
  const reasonMap = new Map();
  if (!Array.isArray(highlightReasons)) {
    return reasonMap;
  }

  for (const entry of highlightReasons) {
    if (!entry || typeof entry !== "object") {
      continue;
    }

    const phrase = String(entry.phrase ?? "").trim();
    const reason = normalizeText(entry.reason, "");
    if (!phrase || !reason) {
      continue;
    }

    reasonMap.set(phrase.toLowerCase(), reason);
  }

  return reasonMap;
}

function canScriptActivePage(tabId) {
  return (
    Number.isInteger(tabId) &&
    typeof chrome !== "undefined" &&
    Boolean(chrome?.scripting?.executeScript)
  );
}

// Injects page-side markup for bias phrases and optionally scrolls to one phrase.
export async function highlightPhrasesInTab(tabId, phrases, selectedPhrase = "") {
  const normalizedPhrases = normalizeHighlights(phrases);
  if (!canScriptActivePage(tabId) || !normalizedPhrases.length) {
    return { ok: false, matchedCount: 0 };
  }

  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId },
      args: [normalizedPhrases, String(selectedPhrase ?? "")],
      func: (rawPhrases, targetPhrase) => {
        const HIGHLIGHT_CLASS = "factgpt-bias-highlight";
        const ACTIVE_CLASS = "factgpt-bias-highlight-active";
        const STYLE_ID = "factgpt-bias-highlight-style";

        const normalize = (value) => String(value ?? "").trim().toLowerCase();
        // Longest-first ordering makes overlapping phrases deterministic.
        const phrases = Array.from(new Set(
          rawPhrases
            .map((phrase) => String(phrase ?? "").trim())
            .filter(Boolean)
        ))
          .sort((a, b) => b.length - a.length)
          .map((phrase) => ({ text: phrase, lower: phrase.toLowerCase() }));

        // Undo the previous run before inserting a fresh set of highlight spans.
        document.querySelectorAll(`span.${HIGHLIGHT_CLASS}`).forEach((node) => {
          node.replaceWith(document.createTextNode(node.textContent || ""));
        });
        document.body?.normalize();

        if (!phrases.length || !document.body) {
          return { ok: false, matchedCount: 0 };
        }

        let style = document.getElementById(STYLE_ID);
        if (!style) {
          style = document.createElement("style");
          style.id = STYLE_ID;
          style.textContent = `
            .${HIGHLIGHT_CLASS} {
              background: #fff176 !important;
              color: inherit !important;
              box-shadow: 0 0 0 2px rgba(255, 193, 7, 0.42) !important;
              border-radius: 3px !important;
              padding: 0 2px !important;
            }
            .${ACTIVE_CLASS} {
              background: #ffca28 !important;
              box-shadow: 0 0 0 3px rgba(245, 124, 0, 0.6) !important;
            }
          `;
          document.head.appendChild(style);
        }

        const ignoredParents = new Set([
          "SCRIPT",
          "STYLE",
          "NOSCRIPT",
          "TEXTAREA",
          "INPUT",
          "SELECT",
          "OPTION",
          "BUTTON"
        ]);

        const walker = document.createTreeWalker(
          document.body,
          NodeFilter.SHOW_TEXT,
          {
            acceptNode(node) {
              const parent = node.parentElement;
              if (!parent || ignoredParents.has(parent.tagName)) {
                return NodeFilter.FILTER_REJECT;
              }

              if (parent.closest(`.${HIGHLIGHT_CLASS}`)) {
                return NodeFilter.FILTER_REJECT;
              }

              return node.nodeValue && node.nodeValue.trim()
                ? NodeFilter.FILTER_ACCEPT
                : NodeFilter.FILTER_REJECT;
            }
          }
        );

        // Snapshot nodes before replacing them so TreeWalker is not invalidated by mutation.
        const textNodes = [];
        while (walker.nextNode()) {
          textNodes.push(walker.currentNode);
        }

        const selected = normalize(targetPhrase);
        const matched = [];

        for (const textNode of textNodes) {
          const text = textNode.nodeValue || "";
          const lowerText = text.toLowerCase();
          const matches = [];
          let cursor = 0;

          while (cursor < text.length) {
            let best = null;

            // Take the earliest next match; prefer the longer phrase at equal offsets.
            for (const phrase of phrases) {
              const index = lowerText.indexOf(phrase.lower, cursor);
              if (index === -1) {
                continue;
              }

              if (
                !best ||
                index < best.index ||
                (index === best.index && phrase.text.length > best.phrase.text.length)
              ) {
                best = { index, phrase };
              }
            }

            if (!best) {
              break;
            }

            matches.push({
              start: best.index,
              end: best.index + best.phrase.text.length,
              phrase: best.phrase.text
            });
            cursor = best.index + Math.max(best.phrase.text.length, 1);
          }

          if (!matches.length) {
            continue;
          }

          // Rebuild this text node as untouched text interleaved with safe spans.
          const fragment = document.createDocumentFragment();
          let lastIndex = 0;
          for (const match of matches) {
            if (match.start > lastIndex) {
              fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.start)));
            }

            const span = document.createElement("span");
            span.className = HIGHLIGHT_CLASS;
            if (selected && normalize(match.phrase) === selected) {
              span.classList.add(ACTIVE_CLASS);
            }
            span.dataset.factgptPhrase = match.phrase;
            span.textContent = text.slice(match.start, match.end);
            fragment.appendChild(span);
            matched.push(span);
            lastIndex = match.end;
          }

          if (lastIndex < text.length) {
            fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
          }

          textNode.replaceWith(fragment);
        }

        const activeMatch = selected
          ? matched.find((node) => normalize(node.dataset.factgptPhrase) === selected)
          : null;

        if (activeMatch) {
          activeMatch.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
        }

        // window.find can still locate a selected phrase split across multiple DOM nodes.
        let browserFoundSelection = false;
        if (!activeMatch && selected) {
          try {
            window.getSelection()?.removeAllRanges();
            browserFoundSelection = window.find(
              targetPhrase,
              false,
              false,
              true,
              false,
              true,
              false
            );
          } catch {
            browserFoundSelection = false;
          }
        }

        return {
          ok: true,
          matchedCount: matched.length,
          scrolled: Boolean(activeMatch) || browserFoundSelection
        };
      }
    });

    return injection?.result || { ok: false, matchedCount: 0 };
  } catch {
    return { ok: false, matchedCount: 0 };
  }
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
export function setOutputMessage(outputArea, message, isError = false) {
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
function renderBiasHighlights(highlights, explanation, highlightReasons = [], options = {}) {
  const wrap = document.createElement("div");
  wrap.className = "bias-chip-grid";

  if (!highlights.length) {
    const noHighlights = document.createElement("p");
    noHighlights.className = "copy-block";
    noHighlights.textContent = "No strong bias keywords were returned.";
    wrap.appendChild(noHighlights);
    return wrap;
  }

  const aiReasonByPhrase = normalizeHighlightReasonMap(highlightReasons);

  for (const [index, phrase] of highlights.entries()) {
    const reason = aiReasonByPhrase.get(String(phrase).trim().toLowerCase()) ||
      "No AI-generated reason is available for this cached result. Run the analysis again to generate specific keyword reasoning.";

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "bias-chip";
    chip.appendChild(document.createTextNode(phrase));
    const action = normalizeText(
      options.highlightActionLabel,
      "click to find on page"
    );
    chip.title = `${action}: ${phrase}`;
    chip.setAttribute("aria-label", `${action}: ${phrase}`);
    chip.setAttribute("aria-expanded", "false");
    chip.addEventListener("click", async () => {
      wrap.querySelectorAll(".bias-chip").forEach((button) => {
        button.classList.remove("bias-chip-active");
        button.setAttribute("aria-expanded", "false");
      });
      chip.classList.add("bias-chip-active");
      chip.setAttribute("aria-expanded", "true");
      await options.onHighlightClick?.(phrase, index);
    });

    const tooltip = document.createElement("span");
    tooltip.className = "chip-tooltip";
    tooltip.id = `bias-reason-${index}`;
    tooltip.setAttribute("role", "note");
    tooltip.textContent = normalizeText(reason);
    chip.setAttribute("aria-describedby", tooltip.id);

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

function createSimplifiedCopy(label, value, maxItems = 2, maxChars = 140) {
  return createLabeledCopy(label, simplifyBulletText(value, maxItems, maxChars));
}

function createCompleteBulletCopy(label, value) {
  const bullets = parseBulletLikeText(value);
  if (!bullets.length) {
    return createLabeledCopy(label, value);
  }
  return createLabeledCopy(label, bullets.join(" • "));
}

// Renders the bias-analysis card returned by the backend.
export function renderResult(outputArea, ai, options = {}) {
  outputArea.innerHTML = "";

  const card = document.createElement("div");
  card.className = "result-card";

  const factOpinionItems = Array.isArray(options.factOpinion?.items)
    ? options.factOpinion.items
    : [];
  const hasResolvedFact = factOpinionItems.some((item) => {
    const finalPrediction = item?.final_prediction || {};
    return (
      finalPrediction.status === "resolved" &&
      (finalPrediction.label === "fact" || finalPrediction.label === "mixed")
    );
  });
  const backendSaysNoFacts = String(ai?.summary ?? "")
    .toLowerCase()
    .includes("no resolved factual");
  const isNotScored = !options.includeOpinionBias && (
    (factOpinionItems.length > 0 && !hasResolvedFact) || backendSaysNoFacts
  );

  if (isNotScored) {
    card.classList.add("result-card-not-scored");
    const notice = document.createElement("p");
    notice.className = "not-scored-notice";
    const opinionCount = factOpinionItems.filter((item) => (
      item?.final_prediction?.status === "resolved" &&
      item?.final_prediction?.label === "opinion"
    )).length;
    notice.textContent = opinionCount === factOpinionItems.length && opinionCount > 0
      ? "Opinion-only content — no factual bias score was calculated."
      : "No resolved factual content — no bias score was calculated.";
    card.appendChild(notice);
  }

  const metric = document.createElement("p");
  metric.className = "metric-line";
  const metricLabel = document.createElement("strong");
  metricLabel.textContent = "Bias Score";
  const scorePill = document.createElement("span");
  scorePill.className = "score-pill";
  scorePill.textContent = isNotScored
    ? "Not scored"
    : `${normalizeText(ai.bias_score, "N/A")} / 10`;
  if (isNotScored) {
    scorePill.classList.add("score-pill-not-scored");
  }
  metric.append(metricLabel, scorePill);

  const summary = createLabeledCopy("Quick Summary", ai.summary);
  const explanation = createCompleteBulletCopy("Explanation", ai.explanation);

  const highlightsHeader = document.createElement("p");
  highlightsHeader.className = "copy-block";
  const highlightsLabel = document.createElement("strong");
  highlightsLabel.textContent = "Bias Keywords";
  const highlightActionLabel = normalizeText(
    options.highlightActionLabel,
    "click to find on page"
  );
  highlightsHeader.append(
    highlightsLabel,
    document.createTextNode(`: ${highlightActionLabel}`)
  );

  const highlights = normalizeHighlights(ai.highlights);
  const highlightsWrap = renderBiasHighlights(highlights, ai.explanation, ai.highlight_reasons, options);

  const missing = createCompleteBulletCopy("Missing Perspectives", ai.missing_perspectives);

  card.append(metric, summary, explanation, highlightsHeader, highlightsWrap, missing);
  outputArea.appendChild(card);
}

export function factOpinionPresentation(item) {
  const finalPrediction = item?.final_prediction || {};
  const excerpts = Array.isArray(finalPrediction.opinion_excerpts)
    ? finalPrediction.opinion_excerpts.filter((excerpt) => String(excerpt ?? "").length > 0)
    : [];

  if (finalPrediction.status !== "resolved") {
    return {
      badge: "Unresolved — not analyzed",
      kind: "unresolved",
      excerpts: []
    };
  }
  if (finalPrediction.label === "opinion") {
    return {
      badge: "Opinion — ignored during analysis",
      kind: "opinion",
      excerpts: []
    };
  }
  if (finalPrediction.label === "mixed" || (finalPrediction.label === "fact" && excerpts.length)) {
    return { badge: "Fact + opinion wording", kind: "mixed", excerpts };
  }
  if (finalPrediction.label === "fact") {
    return { badge: "Fact", kind: "fact", excerpts: [] };
  }
  return {
    badge: "Unresolved — not analyzed",
    kind: "unresolved",
    excerpts: []
  };
}

// Appends exact mixed-opinion spans using DOM nodes only, never HTML strings.
function appendHighlightedOpinionText(container, textValue, excerpts) {
  const text = String(textValue ?? "");
  const matches = [];

  for (const excerptValue of Array.from(new Set(excerpts))) {
    const excerpt = String(excerptValue ?? "");
    let fromIndex = 0;
    while (excerpt && fromIndex < text.length) {
      const start = text.indexOf(excerpt, fromIndex);
      if (start === -1) {
        break;
      }
      matches.push({ start, end: start + excerpt.length });
      fromIndex = start + excerpt.length;
    }
  }

  matches.sort((left, right) => left.start - right.start || right.end - left.end);
  let cursor = 0;
  for (const match of matches) {
    if (match.start < cursor) {
      continue;
    }
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

export function displayedFactOpinionCounts(factOpinion, items) {
  const derived = { fact: 0, opinion: 0, mixed: 0, unresolved: 0, openai_reviewed: 0 };
  for (const item of items) {
    const presentation = factOpinionPresentation(item);
    if (presentation.kind === "opinion") {
      derived.opinion += 1;
    } else if (presentation.kind === "fact") {
      derived.fact += 1;
    } else if (presentation.kind === "mixed") {
      derived.mixed += 1;
    } else {
      derived.unresolved += 1;
    }
    if (item?.final_prediction?.source === "openai") {
      derived.openai_reviewed += 1;
    }
  }

  const supplied = factOpinion?.counts || {};
  const suppliedCountKeys = factOpinion?.items_truncated
    ? ["fact", "mixed", "opinion", "unresolved", "openai_reviewed"]
    : ["opinion", "unresolved", "openai_reviewed"];
  for (const key of suppliedCountKeys) {
    if (Number.isInteger(supplied[key]) && supplied[key] >= 0) {
      derived[key] = supplied[key];
    }
  }
  return derived;
}

// Renders the canonical hybrid classifier result between Summary and Sources.
export function renderFactOpinion(outputArea, section, factOpinion) {
  if (!outputArea || !section) {
    return;
  }
  outputArea.innerHTML = "";
  const items = Array.isArray(factOpinion?.items) ? factOpinion.items : [];
  if (!factOpinion || typeof factOpinion !== "object" || !items.length) {
    section.style.display = "none";
    return;
  }

  section.style.display = "block";
  const counts = displayedFactOpinionCounts(factOpinion, items);
  const countGrid = document.createElement("div");
  countGrid.className = "fact-opinion-counts";
  const countLabels = [
    ["Fact only", counts.fact],
    ["Mixed", counts.mixed],
    ["Opinion", counts.opinion],
    ["Unresolved", counts.unresolved],
    ["OpenAI reviewed", counts.openai_reviewed]
  ];
  for (const [label, value] of countLabels) {
    const count = document.createElement("div");
    count.className = "fact-opinion-count";
    const strong = document.createElement("strong");
    strong.textContent = String(value);
    count.append(strong, document.createTextNode(label));
    countGrid.appendChild(count);
  }

  const list = document.createElement("ol");
  list.className = "fact-opinion-list";
  for (const item of items) {
    const presentation = factOpinionPresentation(item);
    const row = document.createElement("li");
    row.className = `fact-opinion-item fact-opinion-item-${presentation.kind}`;
    row.dataset.kind = presentation.kind;
    row.dataset.itemId = String(item?.id ?? "");

    const details = document.createElement("details");
    details.className = "fact-opinion-details";
    const summary = document.createElement("summary");
    summary.className = "fact-opinion-summary";

    const badge = document.createElement("span");
    badge.className = `fact-opinion-badge fact-opinion-badge-${presentation.kind}`;
    badge.textContent = presentation.badge;

    const text = document.createElement("span");
    text.className = "fact-opinion-text";
    appendHighlightedOpinionText(text, item?.text, presentation.excerpts);
    const detailHint = document.createElement("span");
    detailHint.className = "fact-opinion-detail-hint";
    detailHint.textContent = "View decision details";
    summary.append(badge, text, detailHint);

    const detailBody = document.createElement("div");
    detailBody.className = "fact-opinion-detail-body";
    const detailList = document.createElement("dl");
    detailList.className = "fact-opinion-metadata";
    const addDetail = (label, value) => {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = normalizeText(value, "Not available");
      detailList.append(term, description);
    };

    const localPrediction = item?.local_prediction || {};
    const confidence = Number(localPrediction.confidence);
    const threshold = Number(factOpinion?.confidence_threshold);
    const decisionSource = item?.final_prediction?.source === "openai"
      ? "OpenAI review"
      : item?.final_prediction?.source === "local"
        ? "Local classifier"
        : "Unresolved after review";
    addDetail(
      "Local confidence",
      Number.isFinite(confidence) ? `${(confidence * 100).toFixed(1)}%` : "Not available"
    );
    addDetail(
      "Local threshold",
      Number.isFinite(threshold) ? `${(threshold * 100).toFixed(1)}%` : "Not available"
    );
    addDetail("Final decision source", decisionSource);

    const reviewReasons = Array.isArray(localPrediction.review_reasons)
      ? localPrediction.review_reasons.map((reason) => {
        const [kind, cue] = String(reason).split(":", 2);
        if (kind === "low_confidence") {
          return "Local confidence was below the acceptance threshold";
        }
        if (kind === "factual_exclusion_risk") {
          return "A local opinion decision required review before excluding possible facts";
        }
        if (kind === "possible_mixed") {
          return `Possible mixed fact/opinion wording (${String(cue || "language cue").replaceAll("_", " ")})`;
        }
        return String(reason).replaceAll("_", " ");
      })
      : [];
    if (reviewReasons.length) {
      addDetail("Review trigger", reviewReasons.join("; "));
    }

    const explanationValue = String(item?.final_prediction?.explanation ?? "").trim();
    addDetail(
      "Basis",
      explanationValue || (
        item?.final_prediction?.source === "local"
          ? "The local classifier confidence met the acceptance threshold."
          : "No decision explanation was returned."
      )
    );

    if (presentation.excerpts.length) {
      addDetail("Opinion factor", presentation.excerpts.join(" | "));
    } else if (presentation.kind === "opinion") {
      addDetail("Opinion factor", "The full passage was classified as opinion and excluded from factual research.");
    }

    detailBody.appendChild(detailList);
    details.append(summary, detailBody);
    row.appendChild(details);
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

function nonNegativeInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) && number >= 0 ? number : null;
}

function normalizedClaimSources(sources) {
  const normalized = [];
  const seen = new Set();
  if (!Array.isArray(sources)) {
    return normalized;
  }
  for (const candidate of sources) {
    if (!candidate || typeof candidate !== "object") {
      continue;
    }
    const url = validUrl(candidate.url);
    if (!url || seen.has(url)) {
      continue;
    }
    seen.add(url);
    normalized.push({
      title: safeSourceTitle(url, candidate.title),
      url,
      sourceType: normalizeText(candidate.source_type, "source").replaceAll("_", " "),
      relevanceSummary: normalizeText(candidate.relevance_summary, "")
    });
  }
  return normalized;
}

// Converts current and saved legacy research payloads into one deterministic UI model.
export function normalizeResearchPresentation(aiResearch, factOpinion = null) {
  if (!aiResearch || typeof aiResearch !== "object" || !Object.keys(aiResearch).length) {
    return { available: false, claims: [], coverage: {} };
  }

  const claims = [];
  if (Array.isArray(aiResearch.claims)) {
    for (const entry of aiResearch.claims) {
      if (!entry || typeof entry !== "object") {
        continue;
      }
      claims.push({
        claim: normalizeText(entry.claim, "Claim details not provided"),
        verdict: normalizeText(entry.verdict, "unclear").toLowerCase(),
        evidenceSummary: normalizeText(entry.evidence_summary, "No evidence summary was returned."),
        sources: normalizedClaimSources(entry.sources)
      });
    }
  }

  // Older cached responses placed citations at the top level.
  if (!claims.length && Array.isArray(aiResearch.sources)) {
    claims.push({
      claim: normalizeText(
        Array.isArray(aiResearch.claims) ? aiResearch.claims.join("; ") : aiResearch.claim,
        "Legacy research result"
      ),
      verdict: normalizeText(aiResearch.verdict, "unclear").toLowerCase(),
      evidenceSummary: normalizeText(aiResearch.evidence_summary, "No evidence summary was returned."),
      sources: normalizedClaimSources(aiResearch.sources)
    });
  }

  const rawCoverage = aiResearch.coverage && typeof aiResearch.coverage === "object"
    ? aiResearch.coverage
    : aiResearch;
  const coverageStatus = normalizeText(rawCoverage.status, "not reported").toLowerCase();
  const candidateCount = nonNegativeInteger(rawCoverage.candidate_claim_count);
  const checkedCount = nonNegativeInteger(rawCoverage.checked_claim_count) ?? claims.length;
  const uncheckedCount = nonNegativeInteger(rawCoverage.unchecked_claim_count) ?? (
    candidateCount === null ? null : Math.max(0, candidateCount - checkedCount)
  );
  const factualPassageCount = Array.isArray(factOpinion?.items)
    ? factOpinion.items.filter((item) => {
      const kind = factOpinionPresentation(item).kind;
      return kind === "fact" || kind === "mixed";
    }).length
    : null;
  const hasExplicitCoverage = (
    aiResearch.coverage && typeof aiResearch.coverage === "object"
  ) || [
    aiResearch.candidate_claim_count,
    aiResearch.checked_claim_count,
    aiResearch.unchecked_claim_count,
    aiResearch.coverage_note
  ].some((value) => value !== undefined);
  const fallbackCoverageNote = hasExplicitCoverage
    ? ""
    : factualPassageCount === null
      ? `Checked ${checkedCount} selected factual claim${checkedCount === 1 ? "" : "s"}. This saved result did not report how many other claims were left unchecked.`
      : `Checked ${checkedCount} selected claim${checkedCount === 1 ? "" : "s"} from ${factualPassageCount} classified factual passage${factualPassageCount === 1 ? "" : "s"}. This saved result did not prove that every claim was checked.`;
  const notes = normalizeText(aiResearch.notes, "");
  const noFactualResearch = coverageStatus === "none" || (
    !claims.length && notes.toLowerCase().includes("no resolved factual")
  );

  return {
    available: true,
    claims,
    overallReliability: noFactualResearch
      ? "not applicable"
      : normalizeText(aiResearch.overall_reliability, "not reported").toLowerCase(),
    notes,
    coverage: {
      status: coverageStatus,
      candidateCount,
      checkedCount,
      uncheckedCount,
      inputCharacters: nonNegativeInteger(rawCoverage.input_characters),
      totalFactualCharacters: nonNegativeInteger(rawCoverage.total_factual_characters),
      inputTruncated: rawCoverage.input_truncated === true,
      note: normalizeText(
        rawCoverage.scope_note ?? aiResearch.coverage_note,
        fallbackCoverageNote
      )
    }
  };
}

// Opens a cited source in a new browser tab (extension-safe when available).
function openExternalSource(url) {
  if (typeof chrome !== "undefined" && chrome?.tabs?.create) {
    chrome.tabs.create({ url, active: false });
    return;
  }

  window.open(url, "_blank", "noopener,noreferrer");
}

function appendSourceLink(container, source) {
  const item = document.createElement("li");
  item.className = "source-item";

  const anchor = document.createElement("a");
  anchor.href = source.url;
  anchor.className = "source-link";
  anchor.textContent = source.title;
  anchor.title = "Open source in a background tab";
  anchor.addEventListener("click", (event) => {
    event.preventDefault();
    openExternalSource(source.url);
  });
  item.appendChild(anchor);
  const sourceContext = [source.sourceType, source.relevanceSummary]
    .filter(Boolean)
    .join(" — ");
  if (sourceContext) {
    const hint = document.createElement("p");
    hint.className = "source-hint";
    hint.textContent = sourceContext;
    item.appendChild(hint);
  }
  container.appendChild(item);
}

// Renders verdicts, evidence, per-claim citations, reliability, and coverage.
export function renderSources(sourcesOutput, sourcesSection, aiResearch, factOpinion = null) {
  sourcesOutput.innerHTML = "";
  const research = normalizeResearchPresentation(aiResearch, factOpinion);

  if (!research.available) {
    sourcesSection.style.display = "none";
    return;
  }

  sourcesSection.style.display = "block";

  const overview = document.createElement("div");
  overview.className = "research-overview";
  const reliability = document.createElement("p");
  reliability.className = "research-reliability";
  const reliabilityLabel = document.createElement("strong");
  reliabilityLabel.textContent = "Overall reliability: ";
  const reliabilityPill = document.createElement("span");
  const reliabilityKind = ["high", "medium", "low"].includes(research.overallReliability)
    ? research.overallReliability
    : "unknown";
  reliabilityPill.className = `reliability-pill reliability-pill-${reliabilityKind}`;
  reliabilityPill.textContent = research.overallReliability;
  reliability.append(reliabilityLabel, reliabilityPill);

  const coverage = document.createElement("div");
  coverage.className = "research-coverage";
  const coverageHeading = document.createElement("strong");
  coverageHeading.textContent = "Research coverage";
  coverage.appendChild(coverageHeading);
  const coverageCounts = document.createElement("p");
  coverageCounts.className = "research-coverage-counts";
  const coverageParts = [`Checked: ${research.coverage.checkedCount}`];
  if (research.coverage.candidateCount !== null) {
    coverageParts.unshift(`Candidates: ${research.coverage.candidateCount}`);
  }
  if (research.coverage.uncheckedCount !== null) {
    coverageParts.push(`Unchecked: ${research.coverage.uncheckedCount}`);
  }
  if (
    research.coverage.inputCharacters !== null &&
    research.coverage.totalFactualCharacters !== null
  ) {
    coverageParts.push(
      `Research input: ${research.coverage.inputCharacters}/${research.coverage.totalFactualCharacters} factual characters`
    );
  }
  coverageCounts.textContent = coverageParts.join(" • ");
  coverage.appendChild(coverageCounts);

  if (research.coverage.inputTruncated) {
    const truncation = document.createElement("p");
    truncation.className = "research-coverage-warning";
    truncation.textContent = "Coverage is partial because the factual input was truncated before research.";
    coverage.appendChild(truncation);
  }
  if (research.coverage.note) {
    const scopeNote = document.createElement("p");
    scopeNote.className = "research-coverage-note";
    scopeNote.textContent = research.coverage.note;
    coverage.appendChild(scopeNote);
  }
  overview.append(reliability, coverage);
  sourcesOutput.appendChild(overview);

  if (research.claims.length) {
    const claimList = document.createElement("ol");
    claimList.className = "research-claim-list";
    for (const [index, claim] of research.claims.entries()) {
      const item = document.createElement("li");
      item.className = "research-claim";
      const heading = document.createElement("div");
      heading.className = "research-claim-heading";
      const title = document.createElement("strong");
      title.textContent = `Claim ${index + 1}`;
      const verdictKind = ["supported", "contradicted", "unclear"].includes(claim.verdict)
        ? claim.verdict
        : "unclear";
      item.dataset.verdict = verdictKind;
      const verdict = document.createElement("span");
      verdict.className = `verdict-pill verdict-pill-${verdictKind}`;
      verdict.textContent = claim.verdict;
      heading.append(title, verdict);

      const claimText = document.createElement("p");
      claimText.className = "research-claim-text";
      claimText.textContent = claim.claim;
      const evidence = createLabeledCopy("Evidence", claim.evidenceSummary);
      evidence.classList.add("research-evidence");
      item.append(heading, claimText, evidence);

      if (claim.sources.length) {
        const sourceHeading = document.createElement("strong");
        sourceHeading.className = "research-source-heading";
        sourceHeading.textContent = "Sources used for this verdict";
        const sourceList = document.createElement("ul");
        sourceList.className = "source-list";
        for (const source of claim.sources) {
          appendSourceLink(sourceList, source);
        }
        item.append(sourceHeading, sourceList);
      } else {
        const noCitation = document.createElement("p");
        noCitation.className = "research-no-citation";
        noCitation.textContent = "No valid source citation was returned for this verdict.";
        item.appendChild(noCitation);
      }
      claimList.appendChild(item);
    }
    sourcesOutput.appendChild(claimList);
  } else {
    const noClaims = document.createElement("p");
    noClaims.className = "research-empty";
    noClaims.textContent = research.coverage.status === "none"
      ? "No resolved factual claims were available to research."
      : "No claim-level verification results were returned.";
    sourcesOutput.appendChild(noClaims);
  }

  if (research.notes) {
    const notes = createLabeledCopy("Research notes", research.notes);
    notes.classList.add("research-notes");
    sourcesOutput.appendChild(notes);
  }
}

// Main frontend pipeline: extract text (if needed), run bias scan, then fetch sources.
export async function parseText(url, updateStatus = () => {}, options = {}) {
  // Allow the popup to choose lightweight extraction first and Playwright only as a last resort.
  const useRenderedExtraction = options.extractionMode === "rendered";
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

  const fail = (message, statusMessage = message) => {
    setOutputMessage(outputArea, message, true);
    setStatus(statusMessage);
    return { ok: false, error: message };
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

  try {
    if (url.startsWith("http")) {
      // URL input path: ask the backend to extract readable article text first.
      thinking.setPhase("Scanning page structure");
      setStatus("Extracting readable text...");

      const extractData = useRenderedExtraction
        ? await extractRenderedArticleFromUrl(url)
        : await extractArticleFromUrl(url);

      finalRawText = normalizeArticleText(extractData.text, "");
      if (finalRawText.length < 200) {
        return fail("Not enough readable text found on this page.", "Page text too short for analysis.");
      }
    }
    // Raw text path: popup may pass visible tab text directly after DOM fallback.

    thinking.setPhase("Evaluating language bias");
    setStatus("Running quick bias scan...");
    const biasResult = await analyzeBiasText(finalRawText);

    // Show the bias result immediately even if source-research fails later.
    const aiBiasResult = biasResult.ai_result || {};
    const factOpinionResult = biasResult.fact_opinion || null;
    const biasHighlights = normalizeHighlights(aiBiasResult.highlights);
    const tabId = Number.isInteger(options.tabId) ? options.tabId : null;
    renderResult(outputArea, aiBiasResult, {
      factOpinion: factOpinionResult,
      onHighlightClick: (phrase) => highlightPhrasesInTab(tabId, biasHighlights, phrase)
    });
    renderFactOpinion(factOpinionOutput, factOpinionSection, factOpinionResult);
    highlightPhrasesInTab(tabId, biasHighlights);

    thinking.setPhase("Finding sources");
    setStatus("Bias complete. Gathering sources...");
    try {
      const researchResult = await researchText(
        finalRawText,
        "Article Analysis",
        factOpinionResult,
        aiBiasResult
      );
      const aiResearchResult = researchResult.ai_research || {};
      const finalFactOpinion = researchResult.fact_opinion || factOpinionResult;
      renderFactOpinion(factOpinionOutput, factOpinionSection, finalFactOpinion);
      renderSources(sourcesOutput, sourcesSection, aiResearchResult, finalFactOpinion);
      setStatus("Analysis complete.");
      return {
        ok: true,
        result: {
          ai_result: aiBiasResult,
          ai_research: aiResearchResult,
          fact_opinion: finalFactOpinion
        }
      };
    } catch (error) {
      // Research is optional for UX; return a partial success so bias results still display.
      renderSources(sourcesOutput, sourcesSection, {}, factOpinionResult);
      setStatus("Bias complete. Sources unavailable right now.");
      return {
        ok: true,
        partial: true,
        result: {
          ai_result: aiBiasResult,
          ai_research: {},
          fact_opinion: factOpinionResult
        },
        researchError: normalizeText(error.message, "Research failed.")
      };
    }
  } catch {
    // Network/backend startup issue (e.g., local FastAPI server not running).
    return fail("The analysis service is unavailable right now. Please try again shortly.", "Service unavailable.");
  } finally {
    // Always stop the animated loading state, including on failures.
    thinking.stop();
  }
}
