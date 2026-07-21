"""Deterministic tests for quote location, pairing, and model metadata."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import home


class QuoteParsingTests(unittest.TestCase):
    def test_finds_straight_quote_locations_with_line_and_column(self):
        text = 'Lead\nShe said "quoted words" afterward.'

        self.assertEqual(
            home.find_quote_locations(text),
            [
                {"quote": '"', "role": "ambiguous", "offset": 14, "line": 2, "column": 10},
                {"quote": '"', "role": "ambiguous", "offset": 27, "line": 2, "column": 23},
            ],
        )

    def test_pairs_multiline_curly_quote_as_external_text(self):
        text = "Lead\nShe said \u201cThis is\nvery bad.\u201d Afterward."

        spans = home.extract_quoted_phrases(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["text"], "This is\nvery bad.")
        self.assertEqual(spans[0]["start_line"], 2)
        self.assertEqual(spans[0]["end_line"], 3)
        self.assertEqual(spans[0]["opening_column"], 10)
        self.assertEqual(spans[0]["closing_column"], 10)
        self.assertEqual(spans[0]["attribution"], "external_speaker_or_author")

    def test_pairs_multiple_quote_styles_in_article_order(self):
        text = 'A said "first." B said \u00absecond.\u00bb'

        spans = home.extract_quoted_phrases(text)

        self.assertEqual([span["text"] for span in spans], ["first.", "second."])

    def test_ignores_unmatched_quote(self):
        self.assertEqual(home.extract_quoted_phrases('An unmatched "quote.'), [])

    def test_ignores_empty_quote_pair(self):
        self.assertEqual(home.extract_quoted_phrases('No phrase here "".'), [])

    def test_ignores_escaped_quotes(self):
        text = r'The source wrote \"not a quote\" but said "external words."'

        spans = home.extract_quoted_phrases(text)

        self.assertEqual([span["text"] for span in spans], ["external words."])

    def test_span_offsets_slice_the_exact_phrase(self):
        text = 'First "same" and then "same" again.'

        for span in home.extract_quoted_phrases(text):
            self.assertEqual(text[span["start_offset"]:span["end_offset"]], span["text"])

    def test_crlf_text_reports_quote_marks_on_the_correct_line(self):
        text = 'First\r\nSecond "quote"\r\nThird'

        locations = home.find_quote_locations(text)

        self.assertEqual([location["line"] for location in locations], [2, 2])
        self.assertEqual([location["column"] for location in locations], [8, 14])

    def test_analyze_bias_sends_external_quote_spans_to_model(self):
        text = 'The source called it "a total disaster."'
        model_response = {"output_text": '{"parsed": true}'}

        with patch.object(home, "run_model_json", new=AsyncMock(return_value=model_response)) as run:
            result = asyncio.run(home.analyze_bias(text))

        self.assertEqual(result, {"parsed": True})
        sent_payload = run.await_args.kwargs["payload"]
        self.assertEqual(sent_payload["article_text"], text)
        self.assertEqual(sent_payload["quoted_spans"][0]["text"], "a total disaster.")
        self.assertEqual(
            sent_payload["quoted_spans"][0]["attribution"],
            "external_speaker_or_author",
        )

    def test_analyze_bias_timeout_returns_retryable_error(self):
        async def never_finishes(**_kwargs):
            await asyncio.sleep(1)

        with (
            patch.object(home, "OPENAI_BIAS_TIMEOUT_SECONDS", 0.01),
            patch.object(home, "run_model_json", side_effect=never_finishes),
        ):
            result = asyncio.run(home.analyze_bias("A factual passage long enough to analyze."))

        self.assertEqual(result["error_code"], "bias_timeout")
        self.assertIn("timed out", result["error"].lower())

    def test_research_sends_truncated_external_quote_spans_to_model(self):
        text = 'The source said "external words."'
        model_response = {"output_text": '{"parsed": true}'}

        with patch.object(home, "run_model_json", new=AsyncMock(return_value=model_response)) as run:
            result = asyncio.run(home.researcher_ai(text))

        self.assertEqual(result, {"parsed": True})
        sent_payload = run.await_args.kwargs["payload"]
        self.assertEqual(sent_payload["content_text"], text)
        self.assertEqual(sent_payload["quoted_spans"][0]["text"], "external words.")

    def test_research_timeout_returns_retryable_error(self):
        async def never_finishes(**_kwargs):
            await asyncio.sleep(1)

        with (
            patch.object(home, "OPENAI_RESEARCH_TIMEOUT_SECONDS", 0.01),
            patch.object(home, "run_model_json", side_effect=never_finishes),
        ):
            result = asyncio.run(home.researcher_ai("A factual passage long enough to research."))

        self.assertEqual(result["error_code"], "research_timeout")
        self.assertIn("timed out", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
