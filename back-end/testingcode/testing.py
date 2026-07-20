"""Run the extension end to end against a randomized Common Crawl URL batch."""

import json
import os
import random
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from crawlparse import DEFAULT_WARC_PATHS_URL, extract_urls

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_EXTENSION_DIR = REPO_ROOT / "front-end"
TEST_EXTENSION_OVERRIDES_DIR = Path(__file__).resolve().parent / "test-extension-overrides"
BACKEND_HEALTHCHECK_URL = "http://127.0.0.1:8000/"
PLAYWRIGHT_TIMEOUT_MS = 120000
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Test settings:
# - `HEADLESS=True` runs Chromium without a visible window.
# - `BIAS_ONLY_TEST=True` skips the more expensive research step.
# - `CRAWL_URL_POOL_SIZE` controls how many crawled article URLs to collect first.
# - `TEST_RUN_COUNT` controls how many URLs from that pool get tested this run.
HEADLESS = True
BIAS_ONLY_TEST = os.getenv("FACTGPT_E2E_BIAS_ONLY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CRAWL_URL_POOL_SIZE = 12
TEST_RUN_COUNT = 12
CRAWL_INSECURE_TLS = True


def build_labeled_url_pool() -> list[dict]:
    # First grab a pool of candidate URLs from Common Crawl.
    result = extract_urls(
        warc_paths_url=DEFAULT_WARC_PATHS_URL,
        warc_index=0,
        limit=CRAWL_URL_POOL_SIZE,
        insecure=CRAWL_INSECURE_TLS,
    )
    # Store the crawled URLs as labeled dictionaries so each URL keeps a stable
    # identity (`label`) plus useful metadata (`domain`, `source_index`).
    labeled_urls = []
    for index, url in enumerate(result["urls"], start=1):
        parsed_url = urllib.parse.urlparse(url)
        labeled_urls.append(
            {
                "label": f"url_{index:03d}",
                "source_index": index - 1,
                "url": url,
                "domain": parsed_url.netloc,
            }
        )
    return labeled_urls


def choose_test_batch(labeled_url_pool: list[dict]) -> list[dict]:
    # `remaining` starts as a copy of the full labeled URL pool.
    # We remove items from it as we randomly choose them so we do not test
    # the same URL twice in one run.
    remaining = list(labeled_url_pool)
    # `selected` is the final randomized batch that this run will actually test.
    selected = []
    # `target_count` is how many URLs we want to test this run.
    # Usually this is `TEST_RUN_COUNT`, but if the crawl returned fewer URLs
    # than requested we cap it at the size of `remaining`.
    target_count = min(TEST_RUN_COUNT, len(remaining))

    while remaining and len(selected) < target_count:
        # Pick one random labeled URL from the ones we have not used yet.
        chosen = random.choice(remaining)
        # Store that chosen URL in the randomized test batch.
        selected.append(chosen)
        # Remove it from `remaining` so it cannot be picked again this run.
        remaining.remove(chosen)

    return selected


def ensure_backend_running():
    try:
        with urllib.request.urlopen(BACKEND_HEALTHCHECK_URL, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("Backend health check returned a non-200 status.")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Backend is not reachable at http://127.0.0.1:8000. "
            "Start back-end/home.py before running this test."
        ) from exc


def create_test_extension_bundle(bundle_dir: Path) -> Path:
    # Build a temporary extension directory for Playwright by copying the clean
    # production extension and then layering the saved test-only override files on top.
    shutil.copytree(PRODUCTION_EXTENSION_DIR, bundle_dir, dirs_exist_ok=True)
    # The test pipeline keeps richer extraction telemetry, but delegates all result
    # rendering to this untouched production module so UI contract checks exercise
    # the same expandable mixed/research presentation that ships to users.
    shutil.copy2(
        PRODUCTION_EXTENSION_DIR / "parseScript.js",
        bundle_dir / "productionParseScript.js",
    )
    for override_path in TEST_EXTENSION_OVERRIDES_DIR.iterdir():
        if override_path.is_file():
            shutil.copy2(override_path, bundle_dir / override_path.name)
    return bundle_dir


def find_extension_id(context) -> str:
    service_worker = None

    if context.service_workers:
        workers = context.service_workers
        service_worker = workers[0] if workers else None

    if service_worker is None:
        service_worker = context.wait_for_event("serviceworker", timeout=15000)

    service_worker_url = service_worker.url
    if not service_worker_url.startswith("chrome-extension://"):
        raise RuntimeError("Loaded service worker did not expose a Chrome extension URL.")

    return service_worker_url.split("/")[2]


def build_popup_url(extension_id: str, target_url: str) -> str:
    query = urllib.parse.urlencode(
        {
            "testUrl": target_url,
            "autoRun": "1",
            "biasOnly": "1" if BIAS_ONLY_TEST else "0",
        }
    )
    return f"chrome-extension://{extension_id}/extension.html?{query}"


def expected_mixed_count(fact_opinion: dict) -> int:
    """Count explicit and legacy implicit mixed decisions in an API result."""
    total = 0
    for item in fact_opinion.get("items") or []:
        final = item.get("final_prediction") or {}
        if final.get("status") != "resolved":
            continue
        if final.get("label") == "mixed" or (
            final.get("label") == "fact" and final.get("opinion_excerpts")
        ):
            total += 1
    return total


def validate_popup_contract(state: dict, ui: dict) -> list[str]:
    """Return user-visible contract failures captured from the rendered popup."""
    failures: list[str] = []
    result = state.get("result") or {}
    fact_opinion = result.get("fact_opinion") or {}
    fact_items = fact_opinion.get("items") or []
    research = result.get("ai_research") or {}
    claims = research.get("claims") or []

    if fact_items:
        if not ui.get("fact_opinion_visible"):
            failures.append("Fact/opinion section was not visible.")
        if ui.get("fact_opinion_item_count") != len(fact_items):
            failures.append("Rendered fact/opinion item count did not match the API result.")
        if ui.get("fact_opinion_details_count") != len(fact_items):
            failures.append("Every fact/opinion item must provide expandable decision details.")
        if ui.get("mixed_item_count") != expected_mixed_count(fact_opinion):
            failures.append("Rendered mixed-item count did not match the API result.")
        expected_review_triggers = sum(
            bool((item.get("local_prediction") or {}).get("review_reasons"))
            for item in fact_items
        )
        if ui.get("review_trigger_count") != expected_review_triggers:
            failures.append("Rendered review-trigger count did not match the API result.")

    if not BIAS_ONLY_TEST and claims:
        if not ui.get("sources_visible"):
            failures.append("Research section was not visible.")
        if ui.get("research_claim_count") != len(claims):
            failures.append("Rendered research claim count did not match the API result.")
        if ui.get("research_verdict_count") != len(claims):
            failures.append("Every researched claim must show a verdict.")
        if ui.get("research_evidence_count") != len(claims):
            failures.append("Every researched claim must show its evidence summary.")
        expected_source_links = sum(len(claim.get("sources") or []) for claim in claims)
        if ui.get("source_link_count") != expected_source_links:
            failures.append("Rendered source-link count did not match the API result.")

    if not BIAS_ONLY_TEST and research.get("coverage") and not ui.get("coverage_visible"):
        failures.append("Research coverage disclosure was not visible.")

    return failures


def build_result_record(
    url_entry: dict,
    state: dict,
    status_text: str,
    ui: dict,
) -> dict:
    metadata = state.get("metadata") or {}
    attempts = state.get("attempts") or []
    ai_result = (state.get("result") or {}).get("ai_result") or {}
    highlights = ai_result.get("highlights")
    result = state.get("result") or {}
    fact_opinion = result.get("fact_opinion") or {}
    research = result.get("ai_research") or {}
    contract_failures = validate_popup_contract(state, ui)
    final_stage = attempts[-1]["stage"] if attempts else ""

    return {
        "label": url_entry["label"],
        "source_index": url_entry["source_index"],
        "url": url_entry["url"],
        "domain": url_entry["domain"],
        "status": state.get("status", "error"),
        "message": status_text,
        "bias_score": ai_result.get("bias_score"),
        "highlight_count": len(highlights) if isinstance(highlights, list) else 0,
        "fact_opinion_counts": fact_opinion.get("counts") or {},
        "fact_opinion_item_count": len(fact_opinion.get("items") or []),
        "research_claim_count": len(research.get("claims") or []),
        "research_coverage": research.get("coverage") or {},
        "ui_contract_passed": not contract_failures,
        "ui_contract_failures": contract_failures,
        "ui": ui,
        "extract_method": metadata.get("extractMethod") or "",
        "extract_endpoint": metadata.get("extractEndpoint") or "",
        "extracted_text_chars": metadata.get("extractedTextChars"),
        "extracted_text_preview": metadata.get("extractedTextPreview") or "",
        "error_stage": metadata.get("errorStage") or "",
        "final_stage": final_stage,
        "attempts": attempts,
        "metadata": metadata,
    }


def run_extension_test(context, extension_id: str, url_entry: dict) -> dict:
    article_page = context.new_page()
    popup_page = context.new_page()

    try:
        article_page.goto(url_entry["url"], wait_until="domcontentloaded", timeout=30000)
        popup_page.goto(
            build_popup_url(extension_id, url_entry["url"]),
            wait_until="load",
        )
        popup_page.wait_for_function(
            """
            () => {
              const status = document.body?.dataset?.testStatus;
              return status === "complete" || status === "partial" || status === "error";
            }
            """,
            timeout=PLAYWRIGHT_TIMEOUT_MS,
        )

        status_text = popup_page.locator("#out").inner_text()
        state = popup_page.evaluate("() => window.__FACTGPT_TEST_STATE__ || {}")
        ui = popup_page.evaluate(
            """
            () => {
              const visible = (selector) => {
                const node = document.querySelector(selector);
                return Boolean(node && getComputedStyle(node).display !== "none");
              };
              return {
                fact_opinion_visible: visible("#fact-opinion-area_extension"),
                fact_opinion_item_count: document.querySelectorAll(".fact-opinion-item").length,
                fact_opinion_details_count: document.querySelectorAll(".fact-opinion-details").length,
                mixed_item_count: document.querySelectorAll(".fact-opinion-item-mixed").length,
                review_trigger_count: [...document.querySelectorAll(".fact-opinion-metadata dt")]
                  .filter((node) => node.textContent.trim() === "Review trigger").length,
                sources_visible: visible("#sources-area_extension"),
                research_claim_count: document.querySelectorAll(".research-claim").length,
                research_verdict_count: document.querySelectorAll(".research-claim .verdict-pill").length,
                research_evidence_count: document.querySelectorAll(".research-claim .research-evidence").length,
                source_link_count: document.querySelectorAll(".research-claim .source-link").length,
                coverage_visible: Boolean(document.querySelector(".research-coverage"))
              };
            }
            """
        )
        return build_result_record(url_entry, state, status_text, ui)
    finally:
        popup_page.close()
        article_page.close()


def write_report(
    labeled_url_pool: list[dict],
    selected_test_batch: list[dict],
    test_results: list[dict],
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = REPORTS_DIR / f"playwright-report-{timestamp}.json"
    report_payload = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "headless": HEADLESS,
            "bias_only_test": BIAS_ONLY_TEST,
            "crawl_url_pool_size": CRAWL_URL_POOL_SIZE,
            "test_run_count": TEST_RUN_COUNT,
            "crawl_insecure_tls": CRAWL_INSECURE_TLS,
        },
        "url_pool": labeled_url_pool,
        "selected_test_batch": selected_test_batch,
        "results": test_results,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return report_path


def print_summary(test_results: list[dict]) -> None:
    status_counts = Counter(result["status"] for result in test_results)
    extract_method_counts = Counter(
        result["extract_method"] or "unknown" for result in test_results
    )
    final_stage_counts = Counter(
        result["final_stage"] or "unknown" for result in test_results
    )
    error_stage_counts = Counter(
        result["error_stage"] or "unknown"
        for result in test_results
        if result["status"] == "error"
    )
    error_counts = Counter(
        result["message"] for result in test_results if result["status"] == "error"
    )
    contract_failure_counts = Counter(
        failure
        for result in test_results
        for failure in result.get("ui_contract_failures", [])
    )

    print("\nSummary")
    print(f"- total: {len(test_results)}")
    for status, count in sorted(status_counts.items()):
        print(f"- {status}: {count}")

    print("\nExtraction Methods")
    for method, count in sorted(extract_method_counts.items()):
        print(f"- {method}: {count}")

    print("\nFinal Stages")
    for stage, count in sorted(final_stage_counts.items()):
        print(f"- {stage}: {count}")

    if error_stage_counts:
        print("\nError Stages")
        for stage, count in sorted(error_stage_counts.items()):
            print(f"- {stage}: {count}")

    if error_counts:
        print("\nError Reasons")
        for message, count in error_counts.most_common():
            print(f"- {count}x {message}")

    print("\nUI Contract")
    print(
        "- passed:",
        sum(1 for result in test_results if result.get("ui_contract_passed")),
    )
    print("- failed:", sum(1 for result in test_results if not result.get("ui_contract_passed")))
    for failure, count in contract_failure_counts.most_common():
        print(f"- {count}x {failure}")


def main():
    ensure_backend_running()
    # `labeled_url_pool` stores the full crawled pool of URLs plus labels.
    labeled_url_pool = build_labeled_url_pool()
    # `selected_test_batch` stores the smaller randomized subset we will test now.
    selected_test_batch = choose_test_batch(labeled_url_pool)
    user_data_dir = Path(tempfile.mkdtemp(prefix="factgpt-playwright-"))
    extension_bundle_dir = Path(tempfile.mkdtemp(prefix="factgpt-extension-"))
    # `test_results` stores the result dictionary for each completed test.
    # This is the main list of outputs produced by the run.
    test_results = []

    try:
        create_test_extension_bundle(extension_bundle_dir)
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                str(user_data_dir),
                channel="chromium",
                headless=HEADLESS,
                args=[
                    f"--disable-extensions-except={extension_bundle_dir}",
                    f"--load-extension={extension_bundle_dir}",
                ],
            )

            try:
                print(
                    f"Config: headless={HEADLESS}, bias_only={BIAS_ONLY_TEST}, "
                    f"url_pool_size={len(labeled_url_pool)}, test_run_count={len(selected_test_batch)}"
                )
                extension_id = find_extension_id(context)
                print(f"Loaded extension: {extension_id}")

                for url_entry in selected_test_batch:
                    print(f"Testing: {url_entry['label']} | {url_entry['url']}")
                    try:
                        result = run_extension_test(context, extension_id, url_entry)
                        # Save each finished test result into the results list.
                        test_results.append(result)
                        print(
                            "Result: "
                            f"{result['label']} | "
                            f"{result['status']} | domain={result['domain']} | "
                            f"method={result['extract_method'] or 'unknown'} | "
                            f"stage={result['final_stage'] or 'unknown'} | "
                            f"bias_score={result['bias_score']} | "
                            f"message={result['message']}"
                        )
                    except PlaywrightTimeoutError:
                        result = {
                            "label": url_entry["label"],
                            "source_index": url_entry["source_index"],
                            "url": url_entry["url"],
                            "domain": url_entry["domain"],
                            "status": "error",
                            "message": "Timed out waiting for the extension to finish.",
                            "bias_score": None,
                            "highlight_count": 0,
                            "extract_method": "",
                            "extract_endpoint": "",
                            "extracted_text_chars": None,
                            "extracted_text_preview": "",
                            "error_stage": "timeout",
                            "final_stage": "",
                            "attempts": [],
                            "metadata": {"errorStage": "timeout"},
                        }
                        # Even timeout failures are saved into the same results list.
                        test_results.append(result)
                        print(
                            "Result: "
                            f"{result['label']} | {result['status']} | "
                            f"domain={result['domain']} | "
                            f"method=unknown | stage=timeout | "
                            f"bias_score=None | message={result['message']}"
                        )
            finally:
                context.close()
    finally:
        shutil.rmtree(user_data_dir, ignore_errors=True)
        shutil.rmtree(extension_bundle_dir, ignore_errors=True)

    # Save all run data:
    # - `labeled_url_pool`: everything we crawled
    # - `selected_test_batch`: the randomized URLs we chose to test
    # - `test_results`: the actual outcome for each tested URL
    report_path = write_report(labeled_url_pool, selected_test_batch, test_results)
    print_summary(test_results)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
