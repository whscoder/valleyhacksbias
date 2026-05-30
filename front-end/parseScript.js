import { fetchBackend } from "./api.js";

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
        const phrases = Array.from(new Set(
          rawPhrases
            .map((phrase) => String(phrase ?? "").trim())
            .filter(Boolean)
        ))
          .sort((a, b) => b.length - a.length)
          .map((phrase) => ({ text: phrase, lower: phrase.toLowerCase() }));

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
function renderBiasHighlights(highlights, explanation, options = {}) {
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

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "bias-chip";
    chip.textContent = phrase;
    chip.title = "Find this phrase on the page";
    chip.addEventListener("click", async () => {
      wrap.querySelectorAll(".bias-chip").forEach((button) => {
        button.classList.remove("bias-chip-active");
      });
      chip.classList.add("bias-chip-active");
      await options.onHighlightClick?.(phrase);
    });

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

function createSimplifiedCopy(label, value, maxItems = 2, maxChars = 140) {
  return createLabeledCopy(label, simplifyBulletText(value, maxItems, maxChars));
}

// Renders the bias-analysis card returned by the backend.
export function renderResult(outputArea, ai, options = {}) {
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

  const summary = createLabeledCopy("Quick Summary", ai.summary);
  const explanation = createSimplifiedCopy("Explanation", ai.explanation, 2, 135);

  const highlightsHeader = document.createElement("p");
  highlightsHeader.className = "copy-block";
  const highlightsLabel = document.createElement("strong");
  highlightsLabel.textContent = "Bias Keywords";
  highlightsHeader.append(highlightsLabel, document.createTextNode(": click to find on page"));

  const highlights = normalizeHighlights(ai.highlights);
  const highlightsWrap = renderBiasHighlights(highlights, ai.explanation, options);

  const missing = createSimplifiedCopy("Missing Perspectives", ai.missing_perspectives, 2, 130);

  card.append(metric, summary, explanation, highlightsHeader, highlightsWrap, missing);
  outputArea.appendChild(card);
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
    chrome.tabs.create({ url, active: false });
    return;
  }

  window.open(url, "_blank", "noopener,noreferrer");
}

// Renders the source list section only when research links are available.
export function renderSources(sourcesOutput, sourcesSection, aiResearch) {
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
    anchor.title = "Open source in a background tab";
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

// Main frontend pipeline: extract text (if needed), run bias scan, then fetch sources.
export async function parseText(url, updateStatus = () => {}, options = {}) {
  // Allow the popup to choose lightweight extraction first and Playwright only as a last resort.
  const extractEndpoint = typeof options.extractEndpoint === "string" && options.extractEndpoint
    ? options.extractEndpoint
    : "/extract";
  const outputArea = document.getElementById("parsed-output_bin");
  const resultsSection = document.getElementById("results-area_extension");
  const sourcesSection = document.getElementById("sources-area_extension");
  const sourcesOutput = document.getElementById("sources-output_bin");
  const thinkingArea = document.getElementById("thinking-area");
  const thinkingText = document.getElementById("thinking-text");

  if (!outputArea || !resultsSection || !sourcesSection || !sourcesOutput) {
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
        return fail(normalizeText(extractData.detail, "Failed to extract text"), "Extraction failed.");
      }

      finalRawText = normalizeText(extractData.text, "");
      if (finalRawText.length < 200) {
        return fail("Not enough readable text found on this page.", "Page text too short for analysis.");
      }
    }
    // Raw text path: popup may pass visible tab text directly after DOM fallback.

    thinking.setPhase("Evaluating language bias");
    setStatus("Running quick bias scan...");
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
      return fail(normalizeText(biasResult.detail, "Bias analysis failed."), "Bias analysis failed.");
    }

    // Show the bias result immediately even if source-research fails later.
    const aiBiasResult = biasResult.ai_result || {};
    const biasHighlights = normalizeHighlights(aiBiasResult.highlights);
    const tabId = Number.isInteger(options.tabId) ? options.tabId : null;
    renderResult(outputArea, aiBiasResult, {
      onHighlightClick: (phrase) => highlightPhrasesInTab(tabId, biasHighlights, phrase)
    });
    highlightPhrasesInTab(tabId, biasHighlights);

    thinking.setPhase("Finding sources");
    setStatus("Bias complete. Gathering sources...");
    const researchResponse = await fetchBackend("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: finalRawText,
        title: "Article Analysis"
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
      renderSources(sourcesOutput, sourcesSection, {});
      setStatus("Bias complete. Sources unavailable right now.");
      return {
        ok: true,
        partial: true,
        result: { ai_result: aiBiasResult, ai_research: {} },
        researchError: normalizeText(researchResult.detail, "Research failed.")
      };
    }

    const aiResearchResult = researchResult.ai_research || {};
    renderSources(sourcesOutput, sourcesSection, aiResearchResult);
    setStatus("Analysis complete.");
    return {
      ok: true,
      result: {
        ai_result: aiBiasResult,
        ai_research: aiResearchResult
      }
    };
  } catch {
    // Network/backend startup issue (e.g., local FastAPI server not running).
    return fail("Error connecting to AI. Is your terminal running?", "Backend connection error.");
  } finally {
    // Always stop the animated loading state, including on failures.
    thinking.stop();
  }
}
