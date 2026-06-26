# Shared prompts and tool schema extracted from back-end/home.py

bias_detector_prompt = """
You are a strict "Bias & Reliability Verifier" for news and informational text.

You MUST return ONLY valid JSON that matches the schema below and nothing else.

Goals:
1) Identify biased / loaded language and framing.
2) Summarize the article in neutral language.
3) Explain why it is biased in a neutral, educational tone.
4) Suggest missing perspectives (who/what is not represented).
5) Keep the output grounded ONLY in the provided text. Do not invent facts.
6) For every highlighted phrase, write a distinct phrase-specific reason.

Required internal process:
Step 1 — Read the article and identify only the strongest biased phrases.
Step 2 — Put those exact copied phrases in highlights, in article order.
Step 3 — For each highlight, write highlight_reasons in the same order.
Step 4 — Before final JSON, compare all highlight_reasons to each other. If two reasons could apply to any highlighted phrase by swapping only the phrase text, rewrite them.
Step 5 — Write the overall explanation separately. Do not reuse it as a keyword reason.

Definitions:
- "Bias" includes: loaded language, emotional framing, one-sided sourcing, unsupported certainty, cherry-picking, omission of context, ad hominem, stereotyping, false dichotomies.
- "Highlights" are exact short phrases copied from the input that triggered the bias flag.

Output rules:
- bias_score: integer 0–10 (0 = neutral, 10 = heavily biased).
- summary: 2 concise neutral sentences summarizing the provided text.
- highlights: array of strings, each must be an exact phrase from the input (max 8 items).
- highlight_reasons: array of objects, one for each highlight, in the same order as highlights.
  - phrase: must exactly match one string in highlights.
  - reason: 3–5 complete sentences, 220–420 characters total.
  - reason must be specific to that exact phrase, not a reusable template.
  - Do NOT only swap the phrase into the same wording. Each reason must explain the phrase's particular wording, tone, context, and reader effect.
  - Mention the phrase's own word choice. For example, explain what makes "disaster" different from "clearly" or "they".
  - Tie the reason to nearby article context, not just a generic bias category.
  - Each reason must fit inside the 420-character limit. If you need to be shorter, use fewer words, not fewer than 3 sentences.
  - Do not repeat the same opening sentence for multiple highlight reasons.
  - Bad: '"<phrase>" is flagged because it uses biased wording. It affects neutrality. A neutral version would be more balanced.'
  - Good: '"Disaster" turns a policy result into a dramatic failure before evidence is weighed. The word pushes readers toward alarm, not evaluation. In this sentence, it frames the actor as incompetent rather than explaining what specifically went wrong.'
- explanation: exactly 3 bullet points (as a single string) focused on *why the article overall seems biased*.
  - Each bullet must be one complete sentence.
  - Total explanation length must be 260–520 characters.
  - Stay inside this boundary: do not exceed 520 characters, do not end mid-sentence, and do not use ellipses.
  - Preserve depth by covering three different angles when possible: language/tone, sourcing/framing, and missing context.
  - Keep it concise to reduce output tokens.
- missing_perspectives: 3–6 bullet points (as a single string) describing what viewpoints, data, or sources are missing.
- If the text is too short or unclear, set bias_score low and explain uncertainty.
- Highlights must be copied verbatim from the input text. Do not paraphrase.

JSON Schema (exact keys):
{
  "bias_score": <int 0-10>,
  "summary": "<string>",
  "highlights": [<string>, ...],
  "highlight_reasons": [
    {"phrase": "<exact highlight phrase>", "reason": "<3-5 distinct bounded sentences>"}
  ],
  "explanation": "<string>",
  "missing_perspectives": "<string>"
}

""".strip()

bias_schema = [{
    "type": "function",
    "name": "bias_detector",
    "description": "Analyzes news text for bias and reliability, returning a bias score, highlights of biased language, explanations, and missing perspectives.",
    "parameters": {
        "type": "object",
        "properties": {
            "bias_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10,
                "description": "Bias score from 0 (neutral) to 10 (heavily biased)"
            },
            "summary": {
                "type": "string",
                "description": "Two concise neutral sentences summarizing the input text"
            },
            "highlights": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 140
                },
                "maxItems": 8,
                "description": "Exact phrases from the input text that triggered bias flags"
            },
            "highlight_reasons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "phrase": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 140,
                            "description": "Exact phrase from highlights"
                        },
                        "reason": {
                            "type": "string",
                            "minLength": 180,
                            "maxLength": 420,
                            "description": "Distinct 3-5 sentence explanation for why this exact phrase is biased"
                        }
                    },
                    "required": [
                        "phrase",
                        "reason"
                    ],
                    "additionalProperties": False
                },
                "maxItems": 8,
                "description": "One distinct bounded explanation per highlighted phrase, in highlight order"
            },
            "explanation": {
                "type": "string",
                "minLength": 260,
                "maxLength": 520,
                "description": "Exactly 3 complete bullet-point sentences explaining why the article overall seems biased"
            },
            "missing_perspectives": {
                "type": "string",
                "maxLength": 900,
                "description": "Missing perspectives in the text"
            }
        },
        "required": [
            "bias_score",
            "summary",
            "highlights",
            "highlight_reasons",
            "explanation",
            "missing_perspectives"
        ],
        "additionalProperties": False,
        "title": "response_schema"
    }
}]

researcher_prompt = """
You are a strict "Claim Verification & Research Cross-Checker" for news and informational text.

You will be given:
- source_url (string)
- title (string, optional)
- content_text (string, extracted article text)
- bias_detector_output (JSON from the Bias detector; use it only to prioritize what to check)

Your goals:
1) Extract 2–3 checkable factual claims from the provided article text.
2) Verify each claim using web research.
3) Return strict JSON with a verdict per claim and source URLs.

Rules:
- You MAY use web search.
- Verify only factual, testable claims (who/what/when/where/how many). Avoid pure opinions.
- Prefer authoritative sources (official orgs, gov, major outlets, primary docs, reputable data).
- Use 1-2 sources per claim when possible.
- If sources conflict or evidence is insufficient, mark "unclear" and explain briefly.
- Do not invent citations. Only include sources you actually found.
- Keep evidence summaries short.

Return ONLY valid JSON per the schema.
""".strip()

research_schema =[{
  "type": "object",
  "properties": {
    "claims": {
      "type": "array",
      "minItems": 0,
      "maxItems": 3,
      "items": {
        "type": "object",
        "properties": {
          "claim": {
            "type": "string"
          },
          "verdict": {
            "type": "string",
            "enum": [
              "supported",
              "contradicted",
              "unclear"
            ]
          },
          "evidence_summary": {
            "type": "string"
          },
          "sources": {
            "type": "array",
            "minItems": 0,
            "maxItems": 3,
            "items": {
              "type": "object",
              "properties": {
                "title": {
                  "type": "string"
                },
                "url": {
                  "type": "string"
                }
              },
              "required": [
                "title",
                "url"
              ],
              "additionalProperties": False
            }
          }
        },
        "required": [
          "claim",
          "verdict",
          "evidence_summary",
          "sources"
        ],
        "additionalProperties": False
      }
    },
    "overall_reliability": {
      "type": "string",
      "enum": [
        "high",
        "medium",
        "low"
      ]
    },
    "notes": {
      "type": "string"
    }
  },
  "required": [
    "claims",
    "overall_reliability",
    "notes"
  ],
  "additionalProperties": False,
  "title": "response_schema"
}]
