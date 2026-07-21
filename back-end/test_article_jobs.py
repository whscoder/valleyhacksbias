"""Deterministic contracts for backend-owned article analysis jobs."""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import home
from test_fact_opinion_route import make_result


def article_text() -> str:
    return ("The published rate was four percent. " * 12).strip()


def result_for(text: str, label: str) -> home.FactOpinionResult:
    return make_result(text, [(label, []) for _ in home.segment_article(text)])


class ArticleJobApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.auth_patch = patch.object(home, "REQUIRE_API_TOKEN", False)
        self.origin_patch = patch.object(home, "REQUIRE_ALLOWED_ORIGIN", False)
        self.auth_patch.start()
        self.origin_patch.start()
        for task in home.article_job_tasks.values():
            task.cancel()
        home.article_job_tasks.clear()
        home.article_jobs.clear()
        home.article_jobs_by_content.clear()
        home.request_timestamps_by_client.clear()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=home.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        for task in list(home.article_job_tasks.values()):
            task.cancel()
        if home.article_job_tasks:
            await asyncio.gather(
                *home.article_job_tasks.values(), return_exceptions=True
            )
        home.article_job_tasks.clear()
        home.article_jobs.clear()
        home.article_jobs_by_content.clear()
        home.request_timestamps_by_client.clear()
        self.origin_patch.stop()
        self.auth_patch.stop()

    async def test_create_returns_immediately_and_reuses_normalized_content(self):
        self.assertTrue(home.is_protected_path("/article-jobs"))
        self.assertTrue(home.is_protected_path("/article-jobs/example"))
        run = AsyncMock()
        payload = {
            "page_url": "https://publisher.example/story",
            "title": "  Daily   Report ",
            "text": article_text().replace(". ", ".\r\n"),
        }

        with patch.object(home, "run_article_job", new=run):
            created = await self.client.post("/article-jobs", json=payload)
            reused = await self.client.post(
                "/article-jobs",
                json={
                    **payload,
                    "page_url": "https://another.example/same-copy",
                    "title": "daily report",
                    "text": payload["text"].replace("\r\n", "\n"),
                },
            )
            await asyncio.sleep(0)

        self.assertEqual(created.status_code, 202)
        self.assertEqual(
            set(created.json()),
            {"job_id", "status", "stage", "created_at", "reused"},
        )
        self.assertFalse(created.json()["reused"])
        self.assertEqual(reused.status_code, 202)
        self.assertEqual(reused.json()["job_id"], created.json()["job_id"])
        self.assertTrue(reused.json()["reused"])
        run.assert_awaited_once()

    async def test_client_request_id_retries_same_job_but_new_run_is_distinct(self):
        run = AsyncMock()
        payload = {
            "page_url": "https://publisher.example/story",
            "title": "Daily Report",
            "text": article_text(),
            "client_request_id": "run-one",
        }

        with patch.object(home, "run_article_job", new=run):
            first = await self.client.post("/article-jobs", json=payload)
            retry = await self.client.post("/article-jobs", json=payload)
            home.article_jobs[first.json()["job_id"]]["status"] = "complete"
            second = await self.client.post(
                "/article-jobs",
                json={**payload, "client_request_id": "run-two"},
            )
            await asyncio.sleep(0)

        self.assertEqual(retry.json()["job_id"], first.json()["job_id"])
        self.assertTrue(retry.json()["reused"])
        self.assertNotEqual(second.json()["job_id"], first.json()["job_id"])
        self.assertFalse(second.json()["reused"])
        self.assertEqual(run.await_count, 2)

    async def test_complete_job_has_full_analysis_shape(self):
        text = article_text()
        classification = result_for(text, "opinion")
        payload = home.ArticleJobRequest(
            page_url="https://publisher.example/story", text=text, title="Rates"
        )
        job_id = "article-complete"
        now = 1000.0
        home.article_jobs[job_id] = {
            "status": "queued",
            "stage": "Article analysis queued.",
            "progress": 0,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "error": None,
            "result": None,
            "content_key": "complete-key",
        }

        with patch.object(
            home, "classify_article_fact_opinion", new=AsyncMock(return_value=classification)
        ):
            await home.run_article_job(job_id, payload)

        response = await self.client.get(f"/article-jobs/{job_id}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            set(body),
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
        self.assertEqual(body["status"], "complete")
        self.assertEqual(body["progress"], 100)
        self.assertEqual(
            set(body["result"]),
            {"status", "ai_result", "ai_research", "fact_opinion"},
        )
        self.assertEqual(body["result"]["status"], "analyzed")

    async def test_client_cannot_queue_two_different_active_articles(self):
        first_id = "already-running"
        home.article_jobs[first_id] = {
            "client_id": "127.0.0.1",
            "status": "running",
            "updated_at": 1000.0,
            "content_key": "different-content",
        }

        response = await self.client.post(
            "/article-jobs",
            json={
                "page_url": "https://publisher.example/new-story",
                "text": article_text(),
                "title": "New story",
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("already running", response.json()["detail"])

    async def test_short_supplied_text_uses_direct_then_rendered_extraction(self):
        payload = home.ArticleJobRequest(
            page_url="https://publisher.example/story",
            text="Loading...",
        )
        rendered = f"<html><body><article>{article_text()}</article></body></html>"

        with (
            patch.object(
                home,
                "validate_public_url",
                new=AsyncMock(),
            ) as validate,
            patch.object(
                home,
                "extract_text_with_httpx",
                new=AsyncMock(return_value=("", "blocked")),
            ) as direct,
            patch.object(
                home,
                "fetch_html_with_playwright",
                new=AsyncMock(return_value=rendered),
            ) as rendered_fetch,
        ):
            extracted = await home._article_job_text(payload)

        self.assertGreaterEqual(len(extracted), home.MIN_EXTRACT_CHARS)
        validate.assert_awaited_once_with("https://publisher.example/story")
        direct.assert_awaited_once()
        rendered_fetch.assert_awaited_once()

    async def test_research_failure_preserves_bias_and_sanitizes_error(self):
        text = article_text()
        classification = result_for(text, "fact")
        bias = home.no_factual_bias_result().model_dump(mode="json")
        payload = home.ArticleJobRequest(
            page_url="https://publisher.example/story", text=text, title="Rates"
        )
        job_id = "article-research-failed"
        now = 1000.0
        home.article_jobs[job_id] = {
            "status": "queued",
            "stage": "Article analysis queued.",
            "progress": 0,
            "created_at": now,
            "updated_at": now,
            "content_key": "failure-key",
        }

        with (
            patch.object(
                home,
                "classify_article_fact_opinion",
                new=AsyncMock(return_value=classification),
            ),
            patch.object(home, "analyze_bias", new=AsyncMock(return_value=bias)),
            patch.object(
                home,
                "researcher_ai",
                new=AsyncMock(
                    return_value={
                        "error": "Research failed. Please try again.",
                        "error_code": "research_model_failure",
                        "error_id": "safe-reference",
                        "provider_secret": "must-not-leak",
                    }
                ),
            ),
            patch.object(home.logger, "exception"),
        ):
            await home.run_article_job(job_id, payload)

        response = await self.client.get(f"/article-jobs/{job_id}")
        body = response.json()
        self.assertEqual(body["status"], "failed")
        self.assertIsNotNone(body["result"]["ai_result"])
        self.assertIsNone(body["result"]["ai_research"])
        self.assertEqual(body["error"]["code"], "research_model_failure")
        self.assertEqual(body["error"]["reference"], "safe-reference")
        self.assertNotIn("provider_secret", str(body["error"]))

    async def test_missing_job_explains_backend_restart(self):
        response = await self.client.get("/article-jobs/missing-after-restart")

        self.assertEqual(response.status_code, 404)
        self.assertIn("backend may have restarted", response.json()["detail"])
        self.assertIn("start the analysis again", response.json()["detail"])

    def test_terminal_jobs_expire_after_ttl(self):
        job_id = "expired-article"
        content_key = "expired-content"
        home.article_jobs[job_id] = {
            "status": "failed",
            "updated_at": 1000.0,
            "content_key": content_key,
        }
        home.article_jobs_by_content[content_key] = job_id

        with patch.object(
            home.time,
            "time",
            return_value=1000.0 + home.ARTICLE_JOB_TTL_SECONDS + 1,
        ):
            home._cleanup_article_jobs()

        self.assertNotIn(job_id, home.article_jobs)
        self.assertNotIn(content_key, home.article_jobs_by_content)

    async def test_unexpected_provider_error_is_not_returned(self):
        text = article_text()
        payload = home.ArticleJobRequest(
            page_url="https://publisher.example/story", text=text
        )
        job_id = "article-provider-failed"
        home.article_jobs[job_id] = {
            "status": "queued",
            "stage": "Article analysis queued.",
            "progress": 0,
            "created_at": 1000.0,
            "updated_at": 1000.0,
            "content_key": "provider-failure",
        }

        with (
            patch.object(
                home,
                "classify_article_fact_opinion",
                new=AsyncMock(side_effect=RuntimeError("provider secret detail")),
            ),
            patch.object(home.logger, "exception"),
        ):
            await home.run_article_job(job_id, payload)

        error = home.article_jobs[job_id]["error"]
        self.assertEqual(error["code"], "article_classification_failed")
        self.assertNotIn("provider secret detail", error["message"])


if __name__ == "__main__":
    unittest.main()
