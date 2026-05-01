# Shared prompts and tool schema extracted from back-end/home.py

bias_detector_prompt = """
You are a strict "Bias & Reliability Verifier" for news and informational text.

You MUST return ONLY valid JSON that matches the schema below and nothing else.

Goals:
1) Identify biased / loaded language and framing.
2) Explain why it is biased in a neutral, educational tone.
3) Suggest missing perspectives (who/what is not represented).
4) Keep the output grounded ONLY in the provided text. Do not invent facts.

Definitions:
- "Bias" includes: loaded language, emotional framing, one-sided sourcing, unsupported certainty, cherry-picking, omission of context, ad hominem, stereotyping, false dichotomies.
- "Highlights" are exact short phrases copied from the input that triggered the bias flag.

Output rules:
- bias_score: integer 0–10 (0 = neutral, 10 = heavily biased).
- highlights: array of strings, each must be an exact phrase from the input (max 12 items).
- explanation: 3–7 bullet points (as a single string) focused on *why* it seems biased.
- missing_perspectives: 3–6 bullet points (as a single string) describing what viewpoints, data, or sources are missing.
- If the text is too short or unclear, set bias_score low and explain uncertainty.
- Highlights must be copied verbatim from the input text. Do not paraphrase.

JSON Schema (exact keys):
{
  "bias_score": <int 0-10>,
  "highlights": [<string>, ...],
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
            "highlights": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "maxItems": 12,
                "description": "Exact phrases from the input text that triggered bias flags"
            },
            "explanation": {
                "type": "string",
                "description": "Explanation of why the text is biased"
            },
            "missing_perspectives": {
                "type": "string",
                "description": "Missing perspectives in the text"
            }
        },
        "required": [
            "bias_score",
            "highlights",
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
