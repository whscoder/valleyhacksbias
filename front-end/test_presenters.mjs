import assert from "node:assert/strict";

import {
  displayedFactOpinionCounts,
  factOpinionPresentation,
  normalizeResearchPresentation
} from "./parseScript.js";
import { formatBackendErrorDetail } from "./api.js";

const mixedItem = {
  id: "mixed-1",
  final_prediction: {
    status: "resolved",
    label: "mixed",
    source: "openai",
    opinion_excerpts: ["reckless"]
  }
};

assert.deepEqual(factOpinionPresentation(mixedItem), {
  badge: "Fact + opinion wording",
  kind: "mixed",
  excerpts: ["reckless"]
});

const items = [
  mixedItem,
  {
    final_prediction: {
      status: "resolved",
      label: "fact",
      source: "local",
      opinion_excerpts: []
    }
  },
  {
    final_prediction: {
      status: "resolved",
      label: "opinion",
      source: "openai",
      opinion_excerpts: []
    }
  },
  {
    final_prediction: {
      status: "unresolved",
      label: null,
      source: "unresolved",
      opinion_excerpts: []
    }
  }
];

assert.deepEqual(displayedFactOpinionCounts({}, items), {
  fact: 1,
  opinion: 1,
  mixed: 1,
  unresolved: 1,
  openai_reviewed: 2
});

const research = normalizeResearchPresentation({
  claims: [
    {
      claim: "The program cost four million dollars.",
      verdict: "supported",
      evidence_summary: "The official budget lists the same appropriation.",
      sources: [
        {
          title: "Official budget",
          url: "https://example.gov/budget",
          source_type: "official",
          relevance_summary: "The appropriation table directly lists the program cost."
        },
        {
          title: "Unsafe link",
          url: "javascript:alert(1)",
          source_type: "other",
          relevance_summary: "This must be rejected by URL normalization."
        }
      ]
    }
  ],
  overall_reliability: "high",
  notes: "Only the returned claim was checked.",
  coverage: {
    status: "partial",
    candidate_claim_count: 3,
    checked_claim_count: 1,
    unchecked_claim_count: 2,
    input_characters: 6000,
    total_factual_characters: 8000,
    input_truncated: true,
    scope_note: "One selected claim was checked; two candidate passages remain unchecked."
  }
});

assert.equal(research.claims.length, 1);
assert.equal(research.claims[0].verdict, "supported");
assert.equal(research.claims[0].sources.length, 1);
assert.equal(research.claims[0].sources[0].sourceType, "official");
assert.deepEqual(research.coverage, {
  status: "partial",
  candidateCount: 3,
  checkedCount: 1,
  uncheckedCount: 2,
  inputCharacters: 6000,
  totalFactualCharacters: 8000,
  inputTruncated: true,
  note: "One selected claim was checked; two candidate passages remain unchecked."
});

assert.equal(
  formatBackendErrorDetail({
    message: "Research verification failed.",
    code: "research_no_web_search",
    reference: "abc123"
  }),
  "Research verification failed. Code: research_no_web_search Reference: abc123"
);

console.log("Frontend presenter contract tests passed.");
