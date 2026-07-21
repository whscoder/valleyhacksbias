"""Tests for the hybrid local/OpenAI fact-opinion pipeline."""

import asyncio
import math
import os
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi import HTTPException

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import home


class FakeClassifier:
    classes_ = np.asarray(["fact", "opinion"])
    confidence_threshold_ = 0.80

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_log_proba(self, texts):
        return np.log(self.probabilities[: len(texts)])


def make_result(text: str, decisions: list[tuple[str | None, list[str]]]):
    segments = home.segment_article(text)
    items = []
    for segment, (label, excerpts) in zip(segments, decisions):
        resolved = label is not None
        local_label = "opinion" if label == "opinion" else "fact"
        items.append(
            {
                "id": segment.id,
                "text": segment.text,
                "start_offset": segment.start_offset,
                "end_offset": segment.end_offset,
                "local_prediction": {
                    "label": local_label,
                    "confidence": 0.9 if resolved else 0.55,
                    "log_probability": math.log(0.9 if resolved else 0.55),
                    "accepted": resolved,
                    "review_reasons": [],
                },
                "final_prediction": {
                    "status": "resolved" if resolved else "unresolved",
                    "label": label,
                    "source": "local" if resolved else "unresolved",
                    "explanation": None,
                    "opinion_excerpts": excerpts,
                },
            }
        )
    counts = home._fact_opinion_counts(items)
    return home.FactOpinionResult(
        status="partial" if counts["unresolved"] else "classified",
        confidence_threshold=0.79,
        counts=counts,
        items=items,
    )


class FactOpinionPipelineTests(unittest.TestCase):
    def setUp(self):
        home.fact_opinion_cache.clear()
        home.fact_opinion_cache_order.clear()

    def test_returns_local_and_openai_decisions_in_one_schema(self):
        request = home.FactOpinionRequest(
            items=[
                {"id": "first", "text": "The rate was four percent."},
                {"id": "second", "text": "The plan is awful, but it costs $4 million."},
            ]
        )
        classifier = FakeClassifier([[0.90, 0.10], [0.21, 0.79]])
        api_decision = {
            1: {
                "status": "resolved",
                "label": "mixed",
                "source": "openai",
                "explanation": "A checkable cost claim includes subjective wording.",
                "opinion_excerpts": ["awful"],
            }
        }

        with (
            patch.object(home, "load_fact_opinion_classifier", return_value=classifier),
            patch.object(
                home, "_classify_openai_batch", new=AsyncMock(return_value=api_decision)
            ) as api_call,
        ):
            result = asyncio.run(home.classify_fact_opinion(request))

        self.assertEqual(result.status, "classified")
        self.assertEqual(result.counts.fact, 1)
        self.assertEqual(result.counts.mixed, 1)
        self.assertEqual(result.counts.openai_reviewed, 1)
        self.assertEqual(result.items[0].final_prediction.source, "local")
        self.assertEqual(result.items[1].final_prediction.source, "openai")
        self.assertEqual(result.items[1].final_prediction.opinion_excerpts, ["awful"])
        api_call.assert_awaited_once()

    def test_confident_items_never_call_openai(self):
        request = home.FactOpinionRequest(
            items=[{"text": "The rate was four percent."}]
        )
        classifier = FakeClassifier([[0.91, 0.09]])
        api_call = AsyncMock()

        with (
            patch.object(home, "load_fact_opinion_classifier", return_value=classifier),
            patch.object(home, "_classify_openai_batch", new=api_call),
        ):
            result = asyncio.run(home.classify_fact_opinion(request))

        self.assertEqual(result.items[0].final_prediction.source, "local")
        api_call.assert_not_awaited()

    def test_confident_possible_mixed_fact_is_reviewed(self):
        request = home.FactOpinionRequest(
            items=[{"text": "The reckless policy costs taxpayers four million dollars."}]
        )
        classifier = FakeClassifier([[0.99, 0.01]])
        decision = {
            0: {
                "status": "resolved",
                "label": "mixed",
                "source": "openai",
                "explanation": "The cost is checkable, while reckless is evaluative.",
                "opinion_excerpts": ["reckless"],
            }
        }

        with (
            patch.object(home, "load_fact_opinion_classifier", return_value=classifier),
            patch.object(
                home, "_classify_openai_batch", new=AsyncMock(return_value=decision)
            ) as api_call,
        ):
            result = asyncio.run(home.classify_fact_opinion(request))

        api_call.assert_awaited_once()
        self.assertEqual(result.items[0].final_prediction.label, "mixed")
        self.assertEqual(result.counts.mixed, 1)
        self.assertIn(
            "possible_mixed:evaluative_language",
            result.items[0].local_prediction.review_reasons,
        )

    def test_confident_local_opinion_requires_exclusion_review(self):
        request = home.FactOpinionRequest(
            items=[{"text": "Next Tuesday is Election Day."}]
        )
        classifier = FakeClassifier([[0.01, 0.99]])
        decision = {
            0: {
                "status": "resolved",
                "label": "fact",
                "source": "openai",
                "explanation": "The date of an election is externally verifiable.",
                "opinion_excerpts": [],
            }
        }

        with (
            patch.object(home, "load_fact_opinion_classifier", return_value=classifier),
            patch.object(
                home, "_classify_openai_batch", new=AsyncMock(return_value=decision)
            ) as api_call,
        ):
            result = asyncio.run(home.classify_fact_opinion(request))

        api_call.assert_awaited_once()
        self.assertEqual(result.items[0].final_prediction.label, "fact")
        self.assertIn(
            "factual_exclusion_risk",
            result.items[0].local_prediction.review_reasons,
        )

    def test_review_gate_covers_subjective_cues_and_clean_fact_control(self):
        risky = [
            "The reckless policy costs four million dollars.",
            "Fortunately, the rate fell to four percent.",
            "The official is clearly evil.",
            "I think the bridge opened in 2024.",
            "The project might cost four million dollars.",
        ]
        for text in risky:
            with self.subTest(text=text):
                self.assertTrue(home.local_review_reasons(text, "fact", True))
        self.assertEqual(
            home.local_review_reasons(
                "The audited report lists a four percent rate.", "fact", True
            ),
            [],
        )

    def test_openai_failure_for_confident_risk_item_is_unresolved(self):
        request = home.FactOpinionRequest(
            items=[{"text": "The awful policy costs four million dollars."}]
        )
        classifier = FakeClassifier([[0.99, 0.01]])
        with (
            patch.object(home, "load_fact_opinion_classifier", return_value=classifier),
            patch.object(
                home, "_classify_openai_batch", new=AsyncMock(side_effect=TimeoutError)
            ),
        ):
            result = asyncio.run(home.classify_fact_opinion(request))

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.items[0].final_prediction.status, "unresolved")

    def test_openai_failure_marks_item_unresolved(self):
        request = home.FactOpinionRequest(items=[{"text": "That may be the case."}])
        classifier = FakeClassifier([[0.51, 0.49]])

        with (
            patch.object(home, "load_fact_opinion_classifier", return_value=classifier),
            patch.object(
                home, "_classify_openai_batch", new=AsyncMock(side_effect=TimeoutError)
            ),
        ):
            result = asyncio.run(home.classify_fact_opinion(request))

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.counts.unresolved, 1)
        self.assertIsNone(result.items[0].final_prediction.label)

    def test_returns_service_unavailable_when_local_model_cannot_load(self):
        request = home.FactOpinionRequest(items=[{"text": "A valid item."}])

        with patch.object(
            home,
            "load_fact_opinion_classifier",
            side_effect=FileNotFoundError,
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(home.classify_fact_opinion(request))

        self.assertEqual(raised.exception.status_code, 503)

    def test_rejects_blank_and_oversized_batches(self):
        with self.assertRaises(ValueError):
            home.FactOpinionRequest(items=[{"text": "   "}])
        with self.assertRaises(ValueError):
            home.FactOpinionRequest(
                items=[{"text": "x" * 5_000}, {"text": "y" * 5_000}, {"text": "z" * 3_000}]
            )

    def test_segmenter_preserves_exact_offsets_newlines_and_quotes(self):
        text = 'Heading\n\nShe said “This is terrible.”\nThe rate was four percent.'
        segments = home.segment_article(text)

        self.assertLessEqual(len(segments), home.MAX_CLASSIFICATION_ITEMS)
        for segment in segments:
            self.assertEqual(
                text[segment.start_offset : segment.end_offset], segment.text
            )
            self.assertLessEqual(len(segment.text), home.MAX_ARTICLE_SEGMENT_CHARS)
        self.assertIn('“This is terrible.”', "\n".join(item.text for item in segments))

    def test_segmenter_splits_overlong_text(self):
        text = ("word " * 900).strip()
        segments = home.segment_article(text)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(item.text) <= 2_000 for item in segments))
        self.assertTrue(all(text[item.start_offset:item.end_offset] == item.text for item in segments))

    def test_openai_batches_respect_item_and_character_limits(self):
        items = [
            home.FactOpinionItem(id=index, text="x" * 5_000)
            for index in range(2)
        ]
        batches = home._openai_classification_batches([0, 1], items)

        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertLessEqual(len(batch), home.MAX_OPENAI_CLASSIFICATION_ITEMS)
            self.assertLessEqual(
                sum(sum(len(value) for value in api_item.values()) for _, api_item in batch),
                home.MAX_OPENAI_CLASSIFICATION_CHARS,
            )

    def test_semantic_validation_rejects_duplicate_missing_and_bad_excerpts(self):
        batch = [
            (0, {"id": "item-0000", "text": "The policy is awful but costs $4."}),
            (1, {"id": "item-0001", "text": "I love it."}),
        ]
        parsed = {
            "items": [
                {
                    "id": "item-0000",
                    "label": "fact",
                    "explanation": "Mixed.",
                    "opinion_excerpts": ["not present"],
                },
                {
                    "id": "item-0000",
                    "label": "fact",
                    "explanation": "Duplicate.",
                    "opinion_excerpts": [],
                },
            ]
        }
        self.assertEqual(home._validated_api_decisions(batch, parsed), {})

    def test_semantic_validation_enforces_first_class_mixed_invariants(self):
        batch = [(0, {"id": "item-0000", "text": "The awful plan costs $4."})]
        for label, excerpts in [
            ("mixed", []),
            ("fact", ["awful"]),
            ("opinion", ["awful"]),
        ]:
            with self.subTest(label=label, excerpts=excerpts):
                parsed = {
                    "items": [
                        {
                            "id": "item-0000",
                            "label": label,
                            "explanation": "A bounded semantic explanation.",
                            "opinion_excerpts": excerpts,
                        }
                    ]
                }
                self.assertEqual(home._validated_api_decisions(batch, parsed), {})

        valid = {
            "items": [
                {
                    "id": "item-0000",
                    "label": "mixed",
                    "explanation": "The cost is factual and awful is evaluative.",
                    "opinion_excerpts": ["awful"],
                }
            ]
        }
        self.assertEqual(home._validated_api_decisions(batch, valid)[0]["label"], "mixed")

    def test_incomplete_openai_output_retries_once(self):
        batch = [(0, {"id": "item-0000", "text": "It may happen."})]
        incomplete = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
        complete = {
            "output_text": (
                '{"items":[{"id":"item-0000","label":"opinion",'
                '"explanation":"This is a prediction.","opinion_excerpts":[]}]}'
            )
        }
        run = AsyncMock(side_effect=[incomplete, complete])

        with patch.object(home, "run_model_json", new=run):
            result = asyncio.run(home._classify_openai_batch(batch, "Title"))

        self.assertEqual(result[0]["label"], "opinion")
        self.assertEqual(run.await_count, 2)
        for call in run.await_args_list:
            self.assertEqual(call.kwargs["model"], home.FACT_OPINION_API_MODEL)
            self.assertEqual(
                call.kwargs["reasoning"],
                {"effort": home.FACT_OPINION_REASONING_EFFORT},
            )
            self.assertFalse(call.kwargs["store"])

    def test_refusal_or_malformed_output_does_not_create_a_decision(self):
        batch = [(0, {"id": "item-0000", "text": "It may happen."})]
        refusal = {
            "status": "completed",
            "output": [{"content": [{"type": "refusal", "refusal": "Cannot comply"}]}],
        }
        with patch.object(home, "run_model_json", new=AsyncMock(return_value=refusal)):
            with self.assertRaises(ValueError):
                asyncio.run(home._classify_openai_batch(batch, "Title"))

    def test_factual_text_excludes_opinion_unresolved_and_mixed_excerpt(self):
        text = "The policy is awful but costs $4.\nI love it.\nThis may be true."
        result = make_result(
            text,
            [
                ("mixed", ["awful"]),
                ("opinion", []),
                (None, []),
            ],
        )

        factual = home.build_factual_text(result)

        self.assertIn("costs $4", factual)
        self.assertNotIn("awful", factual)
        self.assertNotIn("I love it", factual)
        self.assertNotIn("may be true", factual)

    def test_factual_content_preserves_quote_attribution_after_opinion_drop(self):
        text = (
            "The source said “I hate this proposal. "
            "The measure costs four million dollars.”"
        )
        result = make_result(text, [("opinion", []), ("fact", [])])

        factual, quoted_spans = home.build_factual_content(result, text)

        self.assertEqual(factual, "The measure costs four million dollars.”")
        self.assertEqual(len(quoted_spans), 1)
        self.assertEqual(
            quoted_spans[0]["text"], "The measure costs four million dollars."
        )
        self.assertEqual(quoted_spans[0]["attribution"], "external_speaker_or_author")
        self.assertEqual(
            factual[
                quoted_spans[0]["start_offset"] : quoted_spans[0]["end_offset"]
            ],
            quoted_spans[0]["text"],
        )

    def test_bias_keeps_mixed_wording_while_research_removes_it(self):
        text = "The reckless policy costs four million dollars."
        result = make_result(text, [("mixed", ["reckless"])])

        research_text, _ = home.build_factual_content(result, text)
        bias_text, _ = home.build_bias_content(result, text)

        self.assertNotIn("reckless", research_text)
        self.assertIn("costs four million dollars", research_text)
        self.assertIn("reckless", bias_text)

    def test_bias_content_excludes_quote_crossing_classified_segments(self):
        text = (
            'The official said “The proposal is\na disaster. It will fail.” '
            'The proposal costs four million dollars.'
        )
        result = make_result(
            text, [("fact", []) for _ in home.segment_article(text)]
        )

        bias_text, bias_quotes = home.build_bias_content(result, text)

        self.assertNotIn("The proposal is\na disaster.", bias_text)
        self.assertNotIn("It will fail.", bias_text)
        self.assertIn("The proposal costs four million dollars.", bias_text)
        self.assertEqual(bias_quotes, [])

    def test_research_reuses_matching_classification(self):
        text = ("The rate was four percent. " * 10).strip()
        result = make_result(text, [("fact", []) for _ in home.segment_article(text)])
        research_output = {
            "claims": [],
            "overall_reliability": "medium",
            "notes": "No external checks in this unit test.",
        }
        request = home.ResearchRequest(
            text=text,
            title="Rates",
            fact_opinion=result.model_dump(),
        )
        home.cache_article_classification(text, "Rates", result)

        with (
            patch.object(
                home, "classify_article_fact_opinion", new=AsyncMock()
            ) as classify,
            patch.object(
                home, "researcher_ai", new=AsyncMock(return_value=research_output)
            ) as research,
        ):
            response = asyncio.run(home.receive_research(request))

        classify.assert_not_awaited()
        research.assert_awaited_once()
        self.assertEqual(response["fact_opinion"], result)

    def test_research_does_not_trust_a_forged_consistent_classification(self):
        text = ("The rate was four percent. " * 10).strip()
        supplied = make_result(
            text, [("fact", []) for _ in home.segment_article(text)]
        )
        corrected = make_result(
            text, [("opinion", []) for _ in home.segment_article(text)]
        )
        request = home.ResearchRequest(
            text=text,
            title="Rates",
            fact_opinion=supplied.model_dump(),
        )

        with (
            patch.object(
                home,
                "classify_article_fact_opinion",
                new=AsyncMock(return_value=corrected),
            ) as classify,
            patch.object(home, "researcher_ai", new=AsyncMock()) as research,
        ):
            response = asyncio.run(home.receive_research(request))

        classify.assert_awaited_once_with(text, "Rates")
        research.assert_not_awaited()
        self.assertEqual(response["fact_opinion"], corrected)

    def test_signed_classification_survives_process_cache_loss(self):
        text = ("The rate was four percent. " * 10).strip()
        result = make_result(
            text, [("fact", []) for _ in home.segment_article(text)]
        )
        signed = home.sign_article_classification(text, "Rates", result)

        home.fact_opinion_cache.clear()
        home.fact_opinion_cache_order.clear()
        with patch.object(
            home, "classify_article_fact_opinion", new=AsyncMock()
        ) as classify:
            reused = asyncio.run(
                home.ensure_article_classification(text, "Rates", signed)
            )

        classify.assert_not_awaited()
        self.assertEqual(reused, signed)
        self.assertEqual(
            home.cached_article_classification(text, "Rates"), signed
        )

    def test_signed_classification_rejects_article_or_payload_tampering(self):
        text = ("The rate was four percent. " * 10).strip()
        result = make_result(
            text, [("fact", []) for _ in home.segment_article(text)]
        )
        signed = home.sign_article_classification(text, "Rates", result)
        tampered = signed.model_copy(deep=True)
        tampered.items[0].text = "A substituted claim."

        self.assertFalse(
            home.article_classification_is_authentic(text, "Other title", signed)
        )
        self.assertFalse(
            home.article_classification_is_authentic(text, "Rates", tampered)
        )

    def test_bias_validation_drops_non_source_highlights_and_keeps_reasons_aligned(self):
        raw = home.no_factual_bias_result().model_dump(mode="json")
        raw["bias_score"] = 5
        raw["highlights"] = ["loaded phrase", "paraphrased phrase"]
        raw["highlight_reasons"] = [
            {"phrase": "loaded phrase", "reason": "x" * 180},
            {"phrase": "paraphrased phrase", "reason": "y" * 180},
        ]

        with patch.object(home.logger, "warning") as warning:
            accepted = home.validate_ai_bias(
                raw, "Text with a loaded phrase inside."
            )

        self.assertEqual(accepted.highlights, ["loaded phrase"])
        self.assertEqual(
            [reason.phrase for reason in accepted.highlight_reasons],
            ["loaded phrase"],
        )
        warning.assert_called_once_with(
            "Dropped non-verbatim bias highlights; dropped_count=%s", 1
        )

        raw["bias_score"] = 11
        with self.assertRaises(HTTPException):
            home.validate_ai_bias(raw, "Text with a loaded phrase inside.")

    def test_safe_model_error_detail_includes_code_and_reference(self):
        detail = home.model_error_detail(
            {
                "error": "Research verification failed.",
                "error_code": "research_no_web_search",
                "error_id": "abc123",
            },
            "Fallback",
        )

        self.assertEqual(
            detail,
            {
                "message": "Research verification failed.",
                "code": "research_no_web_search",
                "reference": "abc123",
            },
        )

    def test_research_recomputes_inconsistent_classification(self):
        text = ("The rate was four percent. " * 10).strip()
        supplied = make_result(text, [("fact", []) for _ in home.segment_article(text)])
        supplied.counts.fact = 0
        corrected = make_result(text, [("fact", []) for _ in home.segment_article(text)])
        research_output = {
            "claims": [],
            "overall_reliability": "medium",
            "notes": "No external checks in this unit test.",
        }
        request = home.ResearchRequest(
            text=text,
            fact_opinion=supplied.model_dump(),
        )

        with (
            patch.object(
                home,
                "classify_article_fact_opinion",
                new=AsyncMock(return_value=corrected),
            ) as classify,
            patch.object(
                home, "researcher_ai", new=AsyncMock(return_value=research_output)
            ),
        ):
            response = asyncio.run(home.receive_research(request))

        classify.assert_awaited_once()
        self.assertEqual(response["fact_opinion"], corrected)

    def test_analyze_classifies_once_then_shares_factual_text(self):
        text = ("The rate was four percent. " * 10).strip()
        result = make_result(text, [("fact", []) for _ in home.segment_article(text)])
        bias_output = home.no_factual_bias_result().model_dump()
        research_output = {
            "claims": [],
            "overall_reliability": "medium",
            "notes": "No external checks in this unit test.",
        }
        request = home.AnalyzeRequest(text=text, title="Rates")

        with (
            patch.object(
                home,
                "classify_article_fact_opinion",
                new=AsyncMock(return_value=result),
            ) as classify,
            patch.object(
                home, "analyze_bias", new=AsyncMock(return_value=bias_output)
            ) as bias,
            patch.object(
                home, "researcher_ai", new=AsyncMock(return_value=research_output)
            ) as research,
        ):
            response = asyncio.run(home.analyze(request))

        classify.assert_awaited_once_with(text, "Rates")
        bias.assert_awaited_once()
        research.assert_awaited_once()
        self.assertEqual(response["fact_opinion"], result)

    def test_no_facts_skips_bias_and_research_models(self):
        text = ("I love this proposal. " * 12).strip()
        result = make_result(text, [("opinion", []) for _ in home.segment_article(text)])
        request = home.AnalyzeRequest(text=text)

        with (
            patch.object(
                home,
                "classify_article_fact_opinion",
                new=AsyncMock(return_value=result),
            ),
            patch.object(home, "analyze_bias", new=AsyncMock()) as bias,
            patch.object(home, "researcher_ai", new=AsyncMock()) as research,
        ):
            response = asyncio.run(home.analyze(request))

        bias.assert_not_awaited()
        research.assert_not_awaited()
        self.assertEqual(response["ai_result"].bias_score, 0)
        self.assertEqual(response["ai_research"].claims, [])


if __name__ == "__main__":
    unittest.main()
