"""Deterministic contract tests for podcast discovery and analysis."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import home
import podcast


def canonical_transcript(
    raw_segments: list[dict],
    *,
    title: str = "Episode 7",
    source: str = "page_transcript",
) -> podcast.PodcastTranscript:
    return podcast.canonicalize_transcript(
        raw_segments,
        title=title,
        page_url="https://publisher.example/episodes/7",
        source=source,
    )


def classification_result(
    entries: list[tuple[str, int, int, str, list[str]]],
) -> home.FactOpinionResult:
    items = []
    counts = {
        "fact": 0,
        "opinion": 0,
        "mixed": 0,
        "unresolved": 0,
        "openai_reviewed": 0,
    }
    for index, (text, start, end, label, excerpts) in enumerate(entries, start=1):
        counts[label] += 1
        items.append(
            {
                "id": f"segment-{index:04d}",
                "text": text,
                "start_offset": start,
                "end_offset": end,
                "local_prediction": {
                    "label": "opinion" if label == "opinion" else "fact",
                    "confidence": 0.95,
                    "log_probability": -0.05,
                    "accepted": True,
                    "review_reasons": [],
                },
                "final_prediction": {
                    "status": "resolved",
                    "label": label,
                    "source": "local",
                    "explanation": None,
                    "opinion_excerpts": excerpts,
                },
            }
        )
    return home.FactOpinionResult(
        status="classified",
        confidence_threshold=0.8,
        counts=counts,
        items=items,
    )


class RssDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def test_matches_canonical_episode_and_reads_podcast_namespace_sources(self):
        feed = """<?xml version="1.0"?>
        <rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
          <channel>
            <language>en-US</language>
            <item>
              <title>Episode 7</title>
              <link>https://publisher.example/episodes/7/</link>
              <guid>episode-seven</guid>
              <podcast:transcript
                url="https://cdn.example/episode-7.vtt"
                type="text/vtt"
                language="en"
              />
              <enclosure
                url="https://cdn.example/episode-7.mp3"
                type="audio/mpeg"
              />
            </item>
          </channel>
        </rss>"""

        episode = podcast.select_rss_episode(
            feed,
            page_url="https://publisher.example/episodes/7",
            page_title="Episode 7",
        )

        self.assertIsNotNone(episode)
        self.assertEqual(episode.title, "Episode 7")
        self.assertEqual(episode.language, "en-US")
        self.assertEqual(
            episode.transcript_urls,
            (("https://cdn.example/episode-7.vtt", "text/vtt", "en"),),
        )
        self.assertEqual(episode.audio_url, "https://cdn.example/episode-7.mp3")

    def test_rejects_ambiguous_title_only_matches(self):
        feed = """<rss><channel>
          <item><title>Daily Briefing</title><link>https://example.test/one</link></item>
          <item><title>Daily Briefing</title><link>https://example.test/two</link></item>
        </channel></rss>"""

        self.assertIsNone(
            podcast.select_rss_episode(
                feed,
                page_url="https://publisher.example/current",
                page_title="Daily Briefing",
            )
        )

    def test_publication_date_resolves_duplicate_title_evidence(self):
        feed = """<rss><channel>
          <item><title>Daily Briefing</title><pubDate>Fri, 17 Jul 2026 08:00:00 GMT</pubDate>
            <link>https://example.test/one</link></item>
          <item><title>Daily Briefing</title><pubDate>Sat, 18 Jul 2026 08:00:00 GMT</pubDate>
            <link>https://example.test/two</link></item>
        </channel></rss>"""

        episode = podcast.select_rss_episode(
            feed,
            page_url="https://publisher.example/current",
            page_title="Daily Briefing",
            page_date="2026-07-18T08:00:00Z",
        )

        self.assertIsNotNone(episode)
        self.assertEqual(episode.published_date, "2026-07-18")

    def test_structured_page_metadata_exposes_publication_date(self):
        page = """<html><head><title>Episode</title>
          <link rel="canonical" href="/canonical-episode">
          <meta property="article:published_time" content="2026-07-18T08:00:00Z">
        </head><body></body></html>"""

        info = podcast.inspect_podcast_page(
            page, "https://publisher.example/episode"
        )

        self.assertEqual(info.published_date, "2026-07-18")
        self.assertEqual(
            info.canonical_url,
            "https://publisher.example/canonical-episode",
        )

    async def test_discovery_prefers_matching_rss_transcript_over_page_sources(self):
        page_url = "https://publisher.example/episodes/7"
        feed_url = "https://publisher.example/feed.xml"
        page_html = f"""<html><head>
          <title>Episode 7</title>
          <link rel="alternate" type="application/rss+xml" href="{feed_url}">
          <link rel="transcript" type="text/plain" href="/episode-7.txt">
        </head><body>
          <audio src="https://cdn.example/episode-7.mp3"></audio>
          <section id="transcript">{"page transcript " * 50}</section>
        </body></html>"""
        feed = """<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
          <channel><item>
            <title>Episode 7</title>
            <link>https://publisher.example/episodes/7</link>
            <podcast:transcript url="https://cdn.example/episode-7.vtt" type="text/vtt" />
            <enclosure url="https://cdn.example/episode-7.mp3" type="audio/mpeg" />
          </item></channel>
        </rss>"""
        rss_transcript = canonical_transcript(
            [{"speaker": "Host", "text": "Publisher supplied transcript."}],
            source="rss_transcript",
        )

        async def fake_fetch(url, **_kwargs):
            if str(url) == page_url:
                return page_html.encode(), "text/html", page_url
            if str(url) == feed_url:
                return feed.encode(), "application/rss+xml", feed_url
            self.fail(f"Unexpected fetch: {url}")

        with (
            patch.object(home, "fetch_public_bytes", side_effect=fake_fetch),
            patch.object(
                home,
                "_publisher_transcript_from_url",
                new=AsyncMock(return_value=rss_transcript),
            ) as publisher,
            patch.object(home, "probe_duration", new=AsyncMock()) as probe,
        ):
            result = await home.discover_podcast_transcript(
                podcast.PodcastJobRequest(page_url=page_url),
                workdir=Path("/unused"),
            )

        self.assertEqual(result.source, "rss_transcript")
        self.assertEqual(result.text, "Publisher supplied transcript.")
        publisher.assert_awaited_once()
        self.assertEqual(
            publisher.await_args.args[0], "https://cdn.example/episode-7.vtt"
        )
        probe.assert_not_awaited()


class TranscriptNormalizationTests(unittest.TestCase):
    def assert_exact_offsets(self, transcript: podcast.PodcastTranscript):
        for segment in transcript.segments:
            self.assertEqual(
                transcript.text[segment.start_offset : segment.end_offset],
                segment.text,
            )

    def test_vtt_preserves_published_speakers_timestamps_and_offsets(self):
        raw = """WEBVTT

00:00:01.250 --> 00:00:03.500
<v Dr. Rivera>The rate was four percent.

00:00:04.000 --> 00:00:06.000
HOST: That policy is outrageous.
"""

        transcript = canonical_transcript(
            podcast.parse_publisher_transcript(raw, "text/vtt"),
            source="rss_transcript",
        )

        self.assertEqual(transcript.text, "The rate was four percent.\nThat policy is outrageous.")
        self.assertEqual([item.speaker for item in transcript.segments], ["Dr. Rivera", "HOST"])
        self.assertEqual(transcript.segments[0].start_seconds, 1.25)
        self.assertEqual(transcript.segments[1].end_seconds, 6.0)
        self.assert_exact_offsets(transcript)

    def test_srt_and_plain_transcripts_use_safe_deterministic_speakers(self):
        srt = """1
00:00:00,000 --> 00:00:02,000
Opening statement.

2
00:00:02,100 --> 00:00:04,000
Guest: A response.
"""
        plain = "Moderator: First line.\n: Second line."

        srt_segments = podcast.parse_publisher_transcript(srt, "application/x-subrip")
        plain_segments = podcast.parse_publisher_transcript(plain, "text/plain")

        self.assertEqual([item["speaker"] for item in srt_segments], ["Speaker A", "Guest"])
        self.assertEqual([item["speaker"] for item in plain_segments], ["Moderator", "Speaker A"])
        transcript = canonical_transcript(srt_segments + plain_segments)
        self.assert_exact_offsets(transcript)

    def test_json_accepts_zero_timestamp_and_never_infers_missing_speaker(self):
        oversized_name = "N" * 140
        raw = (
            '{"segments": ['
            '{"speaker": "", "start": 0, "end": 1.5, "text": "Opening."},'
            f'{{"speaker": "{oversized_name}", "startTime": "00:01.500", '
            '"endTime": "00:03.000", "text": "Reply."}'
            "]}"
        )

        transcript = canonical_transcript(
            podcast.parse_publisher_transcript(raw, "application/json")
        )

        self.assertEqual(transcript.segments[0].speaker, "Speaker A")
        self.assertEqual(transcript.segments[0].start_seconds, 0.0)
        self.assertEqual(transcript.segments[0].end_seconds, 1.5)
        self.assertEqual(len(transcript.segments[1].speaker), 100)
        self.assertEqual(transcript.segments[1].start_seconds, 1.5)
        self.assert_exact_offsets(transcript)


class DownloadSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_redirect_target_is_revalidated_before_second_request(self):
        requested = []
        validated = []

        def handler(request):
            requested.append(str(request.url))
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/private.mp3"},
                request=request,
            )

        async def validate(url):
            validated.append(url)
            if "127.0.0.1" in url:
                raise ValueError("private address")

        real_async_client = httpx.AsyncClient

        def client_factory(**_kwargs):
            return real_async_client(transport=httpx.MockTransport(handler))

        with patch.object(podcast.httpx, "AsyncClient", side_effect=client_factory):
            with self.assertRaisesRegex(ValueError, "private address"):
                await podcast.fetch_public_bytes(
                    "https://publisher.example/audio",
                    validate_url=validate,
                    max_bytes=100,
                )

        self.assertEqual(requested, ["https://publisher.example/audio"])
        self.assertEqual(
            validated,
            ["https://publisher.example/audio", "http://127.0.0.1/private.mp3"],
        )

    async def test_streamed_download_enforces_byte_cap(self):
        def handler(request):
            return httpx.Response(200, content=b"0123456789", request=request)

        async def validate(_url):
            return None

        real_async_client = httpx.AsyncClient

        def client_factory(**_kwargs):
            return real_async_client(transport=httpx.MockTransport(handler))

        with patch.object(podcast.httpx, "AsyncClient", side_effect=client_factory):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                await podcast.fetch_public_bytes(
                    "https://cdn.example/audio.mp3",
                    validate_url=validate,
                    max_bytes=5,
                )

    async def test_protected_player_without_public_sources_fails_closed(self):
        async def fake_fetch(url, **_kwargs):
            return (
                b"<html><head><title>Protected show</title></head>"
                b"<body><iframe src='https://player.example/embed'></iframe></body></html>",
                "text/html",
                str(url),
            )

        with patch.object(home, "fetch_public_bytes", side_effect=fake_fetch):
            with self.assertRaisesRegex(ValueError, "No publisher transcript"):
                await home.discover_podcast_transcript(
                    podcast.PodcastJobRequest(
                        page_url="https://publisher.example/protected"
                    ),
                    workdir=Path("/unused"),
                )

    async def test_audio_duration_cap_stops_before_transcoding(self):
        async def fake_fetch(url, **kwargs):
            if kwargs.get("destination"):
                kwargs["destination"].write_bytes(b"audio")
                return None, "audio/mpeg", str(url)
            return (
                b"<html><head><title>Long show</title></head>"
                b"<body><audio src='https://cdn.example/long.mp3'></audio></body></html>",
                "text/html",
                str(url),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(home, "fetch_public_bytes", side_effect=fake_fetch),
                patch.object(
                    home,
                    "probe_duration",
                    new=AsyncMock(return_value=home.MAX_PODCAST_DURATION_SECONDS + 1),
                ),
                patch.object(home, "transcode_audio_chunks", new=AsyncMock()) as transcode,
            ):
                with self.assertRaisesRegex(ValueError, "configured duration limit"):
                    await home.discover_podcast_transcript(
                        podcast.PodcastJobRequest(
                            page_url="https://publisher.example/long"
                        ),
                        workdir=Path(temp_dir),
                    )
            transcode.assert_not_awaited()


class AudioChunkTests(unittest.IsolatedAsyncioTestCase):
    def test_prefers_nearby_silence_and_always_covers_full_duration(self):
        self.assertEqual(
            podcast.choose_chunk_boundaries(
                5600,
                [2500, 2690, 2810, 5395],
                target_seconds=2700,
                search_seconds=120,
            ),
            [0.0, 2690, 5395, 5600],
        )

    def test_deduplicates_only_equal_overlapping_boundary_cues(self):
        segments = [
            {"speaker": "Speaker A", "start_seconds": 0, "end_seconds": 4, "text": "Same cue"},
            {"speaker": "Speaker A", "start_seconds": 3.5, "end_seconds": 6, "text": " same cue "},
            {"speaker": "Speaker A", "start_seconds": 8, "end_seconds": 9, "text": "Same cue"},
        ]

        result = podcast.deduplicate_adjacent_segments(segments)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["end_seconds"], 6.0)
        self.assertEqual(result[1]["start_seconds"], 8)

    async def test_transcription_offsets_chunks_and_reuses_known_speaker_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "first.mp3"
            second_path = root / "second.mp3"
            first_path.write_bytes(b"mock audio one")
            second_path.write_bytes(b"mock audio two")
            create = AsyncMock(
                side_effect=[
                    {
                        "segments": [
                            {"speaker": "provider-a", "start": 0, "end": 3, "text": "Opening."}
                        ]
                    },
                    {
                        "segments": [
                            {"speaker": "Speaker A", "start": 1, "end": 4, "text": "Continued."}
                        ]
                    },
                ]
            )
            client = SimpleNamespace(
                audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
            )
            chunks = [
                podcast.AudioChunk(first_path, 0, 10),
                podcast.AudioChunk(second_path, 10, 20),
            ]

            with patch.object(
                podcast,
                "_reference_clip",
                new=AsyncMock(return_value="data:audio/mpeg;base64,bW9jaw=="),
            ) as reference:
                result = await podcast.transcribe_audio_chunks(
                    client, chunks, reference_dir=root
                )

        self.assertEqual([item["speaker"] for item in result], ["Speaker A", "Speaker A"])
        self.assertEqual(result[1]["start_seconds"], 11.0)
        self.assertEqual(result[1]["end_seconds"], 14.0)
        reference.assert_awaited_once()
        second_request = create.await_args_list[1].kwargs
        self.assertEqual(second_request["known_speaker_names"], ["Speaker A"])
        self.assertEqual(
            second_request["known_speaker_references"],
            ["data:audio/mpeg;base64,bW9jaw=="],
        )
        self.assertEqual(second_request["model"], "gpt-4o-transcribe-diarize")
        self.assertEqual(second_request["response_format"], "diarized_json")
        self.assertEqual(second_request["chunking_strategy"], "auto")

    async def test_transcription_propagates_provider_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "chunk.mp3"
            audio_path.write_bytes(b"mock audio")
            create = AsyncMock(side_effect=RuntimeError("provider unavailable"))
            client = SimpleNamespace(
                audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
            )

            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await podcast.transcribe_audio_chunks(
                    client,
                    [podcast.AudioChunk(audio_path, 0, 5)],
                    reference_dir=Path(temp_dir),
                )


class PodcastAnalysisRoutingTests(unittest.TestCase):
    def test_opinions_reach_bias_but_only_factual_language_reaches_research(self):
        transcript = canonical_transcript(
            [
                {"speaker": "Host", "text": "The rate was four percent."},
                {"speaker": "Guest", "text": "This plan is awful."},
                {
                    "speaker": "Host",
                    "text": "The budget is ten million and reckless.",
                },
            ]
        )
        first, second, third = transcript.segments
        classification = classification_result(
            [
                (first.text, first.start_offset, first.end_offset, "fact", []),
                (second.text, second.start_offset, second.end_offset, "opinion", []),
                (third.text, third.start_offset, third.end_offset, "mixed", ["reckless"]),
            ]
        )

        bias_text, bias_speakers = home._build_podcast_routed_content(
            classification,
            transcript.text,
            transcript.segments,
            include_opinions=True,
        )
        research_text, research_speakers = home._build_podcast_routed_content(
            classification,
            transcript.text,
            transcript.segments,
            include_opinions=False,
        )

        self.assertIn("This plan is awful.", bias_text)
        self.assertIn("reckless", bias_text)
        self.assertNotIn("This plan is awful.", research_text)
        self.assertIn("The budget is ten million", research_text)
        self.assertNotIn("reckless", research_text)
        self.assertEqual([item["speaker"] for item in bias_speakers], ["Host", "Guest", "Host"])
        self.assertEqual([item["speaker"] for item in research_speakers], ["Host", "Host"])
        for span in bias_speakers:
            self.assertTrue(bias_text[span["start_offset"] : span["end_offset"]].strip())
        for span in research_speakers:
            self.assertTrue(
                research_text[span["start_offset"] : span["end_offset"]].strip()
            )

    def test_windows_cover_every_speaker_segment_and_split_oversized_turns(self):
        transcript = canonical_transcript(
            [
                {"speaker": "A", "text": "a" * 9},
                {"speaker": "B", "text": "b" * 25},
                {"speaker": "C", "text": "c" * 8},
            ]
        )

        with patch.object(home, "MAX_ANALYSIS_INPUT_CHARS", 10):
            windows = home._podcast_windows(transcript)

        self.assertTrue(windows)
        self.assertTrue(all(len(window["text"]) <= 10 for window in windows))
        covered = "".join(window["text"] for window in windows)
        self.assertEqual(covered, "a" * 9 + "b" * 25 + "c" * 8)
        self.assertEqual(
            [segment.text for window in windows for segment in window["segments"]],
            ["a" * 9, "b" * 10, "b" * 10, "b" * 5, "c" * 8],
        )

    def test_repeated_highlights_map_to_distinct_exact_speaker_turns(self):
        phrase = "The rate was four percent."
        routed = f"{phrase}\n{phrase}"
        second_start = len(phrase) + 1
        spans = [
            {
                "segment_id": "first",
                "speaker": "Host",
                "start_seconds": 2.0,
                "end_seconds": 4.0,
                "start_offset": 0,
                "end_offset": len(phrase),
                "source_start_offset": 100,
                "source_end_offset": 100 + len(phrase),
            },
            {
                "segment_id": "second",
                "speaker": "Guest",
                "start_seconds": 12.0,
                "end_seconds": 14.0,
                "start_offset": second_start,
                "end_offset": second_start + len(phrase),
                "source_start_offset": 500,
                "source_end_offset": 500 + len(phrase),
            },
        ]
        used = set()

        first = home._find_highlight_location(phrase, routed, spans, used)
        second = home._find_highlight_location(phrase, routed, spans, used)

        self.assertEqual(first["segment_id"], "first")
        self.assertEqual(first["start_offset"], 100)
        self.assertEqual(second["segment_id"], "second")
        self.assertEqual(second["speaker"], "Guest")
        self.assertEqual(second["start_offset"], 500)


class PodcastEpisodeAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_window_is_classified_and_research_runs_once(self):
        transcript = canonical_transcript(
            [
                {"speaker": "Host", "text": "A" * 30},
                {"speaker": "Guest", "text": "B" * 30},
                {"speaker": "Host", "text": "C" * 30},
            ]
        )
        classifier_inputs = []
        bias_scores = iter((2, 4, 6))

        async def classify(text, _title):
            classifier_inputs.append(text)
            return classification_result(
                [(text, 0, len(text), "fact", [])]
            )

        async def bias(_text, _title, _quotes, **_kwargs):
            return {
                "bias_score": next(bias_scores),
                "summary": "This window presents one test claim. It is used only for deterministic aggregation.",
                "highlights": [],
                "highlight_reasons": [],
                "explanation": "- The window uses a deliberately simple fixture so routing can be measured without a paid model call.\n- Each classifier result remains tied to its exact speaker-aligned input offsets for this test.\n- The aggregate therefore exercises coverage mechanics rather than making a real editorial judgment.",
                "missing_perspectives": "- A production episode would need supporting primary evidence and fuller context around each claim.\n- Additional speakers could supply competing interpretations and correct omissions in the discussion.\n- Episode metadata and source documents would help readers evaluate the factual framing independently.",
            }

        research = AsyncMock(
            return_value={
                "claims": [],
                "overall_reliability": "not_assessed",
                "notes": "No fixture claims were checked against the live web in this deterministic test.",
            }
        )
        synthesis = SimpleNamespace(
            output_text=(
                '{"summary":"The fixture covers all transcript windows. Its aggregate is deterministic for testing.",'
                '"selected_highlight_ids":[],'
                '"explanation":"- Every speaker-aligned window contributes to the final weighted score in this deterministic fixture.\\n- Classification is invoked separately for each bounded input before the episode result is synthesized.\\n- The test isolates routing and coverage behavior without interpreting the repeated placeholder speech.",'
                '"missing_perspectives":"- Real podcast analysis should include documents that support the speakers factual assertions and provide missing context.\\n- Additional participants may offer viewpoints absent from the episode and challenge selective framing.\\n- Publisher metadata and independent reporting would help listeners evaluate the discussion more fully."}'
            )
        )

        with (
            patch.object(home, "MAX_ANALYSIS_INPUT_CHARS", 40),
            patch.object(home, "classify_article_fact_opinion", side_effect=classify) as classifier,
            patch.object(home, "analyze_bias", side_effect=bias),
            patch.object(home, "researcher_ai", new=research),
            patch.object(home, "run_model_json", new=AsyncMock(return_value=synthesis)),
        ):
            result = await home.analyze_podcast_transcript(transcript)

        self.assertEqual(len(classifier_inputs), 3)
        self.assertEqual(classifier.await_count, 3)
        research.assert_awaited_once()
        self.assertEqual(result["podcast"]["window_count"], 3)
        self.assertEqual(result["podcast"]["windows_analyzed"], 3)
        self.assertEqual(result["ai_result"]["bias_score"], 4)
        self.assertEqual(result["ai_research"]["coverage"]["candidate_claim_count"], 3)
        self.assertEqual(result["fact_opinion"]["counts"]["fact"], 3)


class PodcastJobApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.auth_patch = patch.object(home, "REQUIRE_API_TOKEN", False)
        self.origin_patch = patch.object(home, "REQUIRE_ALLOWED_ORIGIN", False)
        self.auth_patch.start()
        self.origin_patch.start()
        for task in home.podcast_job_tasks.values():
            task.cancel()
        home.podcast_job_tasks.clear()
        home.podcast_jobs.clear()
        home.podcast_jobs_by_url.clear()
        home.request_timestamps_by_client.clear()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=home.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        for task in list(home.podcast_job_tasks.values()):
            task.cancel()
        if home.podcast_job_tasks:
            await asyncio.gather(*home.podcast_job_tasks.values(), return_exceptions=True)
        home.podcast_job_tasks.clear()
        home.podcast_jobs.clear()
        home.podcast_jobs_by_url.clear()
        home.request_timestamps_by_client.clear()
        self.origin_patch.stop()
        self.auth_patch.stop()

    async def test_create_job_shape_and_normalized_url_deduplication(self):
        self.assertTrue(home.is_protected_path("/podcast-jobs"))
        self.assertTrue(home.is_protected_path("/podcast-jobs/example"))
        self.assertTrue(home.is_protected_path("/podcast-jobs/example/segments"))
        run = AsyncMock()
        payload = {
            "page_url": "https://publisher.example/episodes/7/",
            "hints": {"feed_urls": [], "transcript_urls": [], "audio_urls": []},
        }

        with patch.object(home, "run_podcast_job", new=run):
            created = await self.client.post("/podcast-jobs", json=payload)
            reused = await self.client.post(
                "/podcast-jobs",
                json={**payload, "page_url": "https://publisher.example/episodes/7"},
            )
            await asyncio.sleep(0)

        self.assertEqual(created.status_code, 202)
        self.assertEqual(
            set(created.json()),
            {"job_id", "status", "stage", "created_at", "reused"},
        )
        self.assertEqual(created.json()["status"], "queued")
        self.assertFalse(created.json()["reused"])
        self.assertEqual(reused.status_code, 202)
        self.assertEqual(reused.json()["job_id"], created.json()["job_id"])
        self.assertTrue(reused.json()["reused"])
        run.assert_awaited_once()

    async def test_missing_job_explains_restart_retry_state(self):
        response = await self.client.get("/podcast-jobs/missing-after-restart")

        self.assertEqual(response.status_code, 404)
        self.assertIn("backend may have restarted", response.json()["detail"])
        self.assertIn("start the analysis again", response.json()["detail"])

    def test_completed_results_expire_after_configured_ttl(self):
        job_id = "expired-job"
        url_key = "https://publisher.example/expired"
        home.podcast_jobs[job_id] = {
            "status": "complete",
            "updated_at": 1000.0,
            "url_key": url_key,
        }
        home.podcast_jobs_by_url[url_key] = job_id

        with patch.object(
            home.time,
            "time",
            return_value=1000.0 + home.PODCAST_JOB_TTL_SECONDS + 1,
        ):
            home._cleanup_podcast_jobs()

        self.assertNotIn(job_id, home.podcast_jobs)
        self.assertNotIn(url_key, home.podcast_jobs_by_url)

    async def test_failed_job_removes_temporary_files_and_hides_provider_error(self):
        job_id = "job-provider-failure"
        now = 1000.0
        home.podcast_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Podcast analysis queued.",
            "progress": 0,
            "created_at": now,
            "updated_at": now,
            "url_key": "https://publisher.example/failure",
        }
        observed_workdirs = []

        async def fail_discovery(_request, *, workdir, stage=None):
            observed_workdirs.append(workdir)
            (workdir / "partial-audio").write_bytes(b"temporary")
            raise RuntimeError("provider secret detail")

        with (
            patch.object(home, "discover_podcast_transcript", side_effect=fail_discovery),
            patch.object(home.logger, "exception"),
        ):
            await home.run_podcast_job(
                job_id,
                podcast.PodcastJobRequest(
                    page_url="https://publisher.example/failure"
                ),
            )

        self.assertEqual(home.podcast_jobs[job_id]["status"], "failed")
        self.assertEqual(
            home.podcast_jobs[job_id]["error"]["message"],
            "Podcast analysis failed.",
        )
        self.assertNotIn(
            "provider secret detail", home.podcast_jobs[job_id]["error"]["message"]
        )
        self.assertEqual(len(observed_workdirs), 1)
        self.assertFalse(observed_workdirs[0].exists())

    async def test_complete_job_and_segment_pagination_shapes(self):
        now = 1000.0
        job_id = "job-ready"
        segments = [
            {
                "id": f"podcast-segment-{index:05d}",
                "speaker": "Speaker A",
                "start_seconds": float(index),
                "end_seconds": float(index + 1),
                "text": f"Segment {index}",
                "start_offset": index * 10,
                "end_offset": index * 10 + 9,
                "classification": None,
            }
            for index in range(3)
        ]
        result = {
            "podcast": {"title": "Episode 7", "window_count": 1},
            "ai_result": {"bias_score": 3},
            "ai_research": {"claims": []},
            "fact_opinion": {"status": "classified"},
        }
        home.podcast_jobs[job_id] = {
            "job_id": job_id,
            "status": "complete",
            "stage": "Podcast analysis complete.",
            "progress": 100,
            "created_at": now,
            "updated_at": now,
            "completed_at": now,
            "error": None,
            "result": result,
            "segments": segments,
            "url_key": "https://publisher.example/episodes/7",
        }

        with patch.object(home.time, "time", return_value=now):
            job_response = await self.client.get(f"/podcast-jobs/{job_id}")
            segment_response = await self.client.get(
                f"/podcast-jobs/{job_id}/segments", params={"cursor": 1, "limit": 1}
            )

        self.assertEqual(job_response.status_code, 200)
        self.assertEqual(
            set(job_response.json()),
            {
                "job_id",
                "status",
                "stage",
                "progress",
                "created_at",
                "updated_at",
                "completed_at",
                "error",
                "result",
            },
        )
        self.assertEqual(set(job_response.json()["result"]), {"podcast", "ai_result", "ai_research", "fact_opinion"})
        self.assertEqual(segment_response.status_code, 200)
        self.assertEqual(
            set(segment_response.json()),
            {"job_id", "segments", "cursor", "next_cursor", "total"},
        )
        self.assertEqual(segment_response.json()["segments"], [segments[1]])
        self.assertEqual(segment_response.json()["next_cursor"], 2)
        self.assertEqual(segment_response.json()["total"], 3)


if __name__ == "__main__":
    unittest.main()
