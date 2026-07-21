"""Focused tests for source-backed research validation and orchestration."""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import home
from test_fact_opinion_route import make_result


def research_claim(
    *,
    verdict: str = "supported",
    url: str = "https://example.gov/report",
    source_type: str = "official",
) -> dict:
    return {
        "claim": "The published rate was four percent.",
        "verdict": verdict,
        "evidence_summary": "The official report lists the published rate as four percent.",
        "sources": [
            {
                "title": "Official rate report",
                "url": url,
                "source_type": source_type,
                "relevance_summary": (
                    "The report directly publishes the rate discussed in the claim."
                ),
            }
        ],
    }


def research_output(*claims: dict) -> dict:
    return {
        "claims": list(claims),
        "overall_reliability": "high",
        "notes": "Only the returned high-priority factual claims were checked.",
    }


def web_response(parsed: dict, *urls: str, annotation_urls: tuple[str, ...] = ()) -> dict:
    output = [
        {
            "type": "web_search_call",
            "status": "completed",
            "action": {
                "type": "search",
                "sources": [{"type": "url", "url": url} for url in urls],
            },
        }
    ]
    if annotation_urls:
        output.append(
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(parsed),
                        "annotations": [
                            {"type": "url_citation", "url": url}
                            for url in annotation_urls
                        ],
                    }
                ],
            }
        )
    return {
        "output_text": json.dumps(parsed),
        "output": output,
    }


class StrictResearchValidationTests(unittest.TestCase):
    def test_decisive_verdict_requires_a_source(self):
        raw = research_output(research_claim())
        raw["claims"][0]["sources"] = []

        with self.assertRaises(HTTPException) as raised:
            home.validate_ai_research(raw, candidate_claim_count=1)

        self.assertEqual(raised.exception.status_code, 502)

    def test_source_requires_http_or_https(self):
        raw = research_output(research_claim(url="ftp://example.gov/report"))

        with self.assertRaises(HTTPException):
            home.validate_ai_research(raw, candidate_claim_count=1)

    def test_decisive_verdict_rejects_only_other_quality_sources(self):
        raw = research_output(research_claim(source_type="other"))

        with self.assertRaises(HTTPException):
            home.validate_ai_research(raw, candidate_claim_count=1)

    def test_coverage_discloses_checked_unchecked_and_truncation(self):
        result = home.validate_ai_research(
            research_output(research_claim()),
            candidate_claim_count=4,
            total_factual_characters=home.MAX_RESEARCH_INPUT_CHARS + 200,
        )

        self.assertEqual(result.coverage.status, "partial")
        self.assertEqual(result.coverage.checked_claim_count, 1)
        self.assertEqual(result.coverage.unchecked_claim_count, 3)
        self.assertTrue(result.coverage.input_truncated)
        self.assertIn("does not imply", result.coverage.scope_note)

    def test_no_facts_are_explicitly_not_assessed(self):
        result = home.no_factual_research_result()

        self.assertEqual(result.overall_reliability, "not_assessed")
        self.assertEqual(result.coverage.status, "none")
        self.assertEqual(result.coverage.checked_claim_count, 0)

    def test_no_facts_disclose_an_unclassified_article_tail(self):
        result = home.no_factual_research_result(article_input_truncated=True)

        self.assertTrue(result.coverage.input_truncated)
        self.assertIn("outside that window", result.coverage.scope_note)


class ResearchToolOrchestrationTests(unittest.TestCase):
    def test_research_forces_web_search_and_passes_bias_output(self):
        parsed = research_output(research_claim())
        response = web_response(parsed, "https://example.gov/report")
        bias = home.no_factual_bias_result()

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ) as run:
            result = asyncio.run(
                home.researcher_ai(
                    'A source said "The rate was four percent."',
                    title="Rates",
                    bias_result=bias,
                    candidate_claim_count=1,
                )
            )

        self.assertEqual(result, parsed)
        call = run.await_args.kwargs
        self.assertEqual(call["tools"], [{"type": "web_search"}])
        self.assertEqual(call["tool_choice"], {"type": "web_search"})
        self.assertEqual(call["include"], ["web_search_call.action.sources"])
        self.assertEqual(call["model"], "gpt-5.5")
        self.assertEqual(
            call["payload"]["bias_detector_output"], bias.model_dump(mode="json")
        )
        self.assertEqual(
            call["payload"]["quoted_spans"][0]["text"],
            "The rate was four percent.",
        )

    def test_research_rejects_response_without_completed_web_search(self):
        response = {
            "output_text": json.dumps(research_output(research_claim())),
            "output": [{"type": "message", "content": []}],
        }

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result["error_code"], "research_no_web_search")
        self.assertNotIn("completed", result["error"])

    def test_research_removes_unverified_citation_and_downgrades_claim(self):
        parsed = research_output(research_claim(url="https://invented.example/report"))
        response = web_response(parsed, "https://example.gov/report")

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result["claims"][0]["verdict"], "unclear")
        self.assertEqual(result["claims"][0]["sources"], [])
        self.assertEqual(result["overall_reliability"], "low")
        self.assertIn("Unverified citation links were removed", result["notes"])

    def test_research_accepts_url_citation_annotations_without_action_sources(self):
        parsed = research_output(research_claim(url="https://EXAMPLE.gov/report/"))
        response = web_response(
            parsed, annotation_urls=("https://example.gov/report",)
        )

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result, parsed)

    def test_research_accepts_completed_action_url(self):
        for action_type in ("open_page", "find_in_page"):
            with self.subTest(action_type=action_type):
                parsed = research_output(research_claim())
                response = web_response(parsed, "https://search.example/result")
                response["output"].insert(
                    1,
                    {
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {
                            "type": action_type,
                            "url": "https://example.gov/report",
                        },
                    },
                )

                with patch.object(
                    home, "run_model_json", new=AsyncMock(return_value=response)
                ):
                    result = asyncio.run(
                        home.researcher_ai("A factual statement.")
                    )

                self.assertEqual(result, parsed)

    def test_research_accepts_equivalent_url_with_tracking_parameters(self):
        parsed = research_output(
            research_claim(url="https://example.gov/report?section=2")
        )
        response = web_response(
            parsed,
            "https://example.gov/report?utm_source=search&section=2&gclid=abc",
        )

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result, parsed)

    def test_research_does_not_trust_incomplete_action_url(self):
        parsed = research_output(research_claim())
        response = web_response(parsed, "https://search.example/result")
        response["output"].insert(
            1,
            {
                "type": "web_search_call",
                "status": "incomplete",
                "action": {
                    "type": "find_in_page",
                    "url": "https://example.gov/report",
                },
            },
        )

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result["claims"][0]["verdict"], "unclear")
        self.assertEqual(result["claims"][0]["sources"], [])

    def test_research_completed_search_without_citations_returns_safe_unclear_claim(self):
        parsed = research_output(research_claim())
        response = web_response(parsed)

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result["claims"][0]["verdict"], "unclear")
        self.assertEqual(result["claims"][0]["sources"], [])
        self.assertEqual(result["overall_reliability"], "low")

    def test_research_retries_token_truncation_with_remaining_global_budget(self):
        parsed = research_output(research_claim())
        incomplete = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
        complete = web_response(parsed, "https://example.gov/report")

        with patch.object(
            home,
            "run_model_json",
            new=AsyncMock(side_effect=[incomplete, complete]),
        ) as run:
            result = asyncio.run(
                home.researcher_ai(
                    "A factual statement.", timeout_seconds=60.0
                )
            )

        self.assertEqual(result, parsed)
        self.assertEqual(run.await_count, 2)
        self.assertEqual(
            [call.kwargs["max_tokens"] for call in run.await_args_list],
            [1800, 2600],
        )
        self.assertGreater(run.await_args_list[0].kwargs["timeout"], 30.0)
        self.assertGreater(run.await_args_list[1].kwargs["timeout"], 30.0)

    def test_analyze_passes_validated_bias_to_research_sequentially(self):
        text = ("The rate was four percent. " * 10).strip()
        classification = make_result(
            text, [("fact", []) for _ in home.segment_article(text)]
        )
        bias_output = home.no_factual_bias_result().model_dump(mode="json")
        raw_research = research_output(research_claim())
        request = home.AnalyzeRequest(text=text, title="Rates")

        with (
            patch.object(
                home,
                "classify_article_fact_opinion",
                new=AsyncMock(return_value=classification),
            ),
            patch.object(
                home, "analyze_bias", new=AsyncMock(return_value=bias_output)
            ) as bias,
            patch.object(
                home, "researcher_ai", new=AsyncMock(return_value=raw_research)
            ) as research,
        ):
            response = asyncio.run(home.analyze(request))

        bias.assert_awaited_once()
        supplied_bias = research.await_args.kwargs["bias_result"]
        self.assertIsInstance(supplied_bias, home.AIresultBias)
        self.assertEqual(supplied_bias.model_dump(mode="json"), bias_output)
        self.assertEqual(response["ai_research"].coverage.status, "partial")

    def test_article_job_completes_when_research_is_unavailable(self):
        text = ("The published rate was four percent. " * 8).strip()
        classification = make_result(
            text, [("fact", []) for _ in home.segment_article(text)]
        )
        job_id = "research-partial-test"
        home.article_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "result": None,
            "error": None,
        }
        request = home.ArticleJobRequest(
            page_url="https://example.gov/article", text=text, title="Rates"
        )

        try:
            with (
                patch.object(
                    home,
                    "classify_article_fact_opinion",
                    new=AsyncMock(return_value=classification),
                ),
                patch.object(
                    home,
                    "analyze_bias",
                    new=AsyncMock(
                        return_value=home.no_factual_bias_result().model_dump(mode="json")
                    ),
                ),
                patch.object(
                    home,
                    "researcher_ai",
                    new=AsyncMock(
                        return_value={
                            "error": "Research verification failed. Please try again.",
                            "error_code": "research_unverified_citation",
                            "error_id": "safe-reference",
                        }
                    ),
                ),
            ):
                asyncio.run(home.run_article_job(job_id, request))

            job = home.article_jobs[job_id]
            self.assertEqual(job["status"], "complete")
            self.assertIsNone(job["error"])
            self.assertIsNotNone(job["result"]["ai_result"])
            self.assertEqual(job["result"]["status"], "analyzed")
            research = job["result"]["ai_research"]
            self.assertEqual(research["claims"], [])
            self.assertEqual(research["overall_reliability"], "not_assessed")
            self.assertEqual(research["coverage"]["checked_claim_count"], 0)
            self.assertGreater(research["coverage"]["candidate_claim_count"], 0)
            self.assertIn("verification was unavailable", research["notes"])
        finally:
            home.article_jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
