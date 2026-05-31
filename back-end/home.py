import asyncio
from collections import defaultdict, deque
import ipaddress
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import httpx
import uvicorn
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from openai import AsyncOpenAI
from pydantic import BaseModel, HttpUrl

from ai_prompts import bias_detector_prompt, bias_schema, research_schema, researcher_prompt

try:
    from playwright.async_api import async_playwright
except Exception:
    # Playwright is optional; extraction fallback will report a clear error if missing.
    async_playwright = None


# Centralized runtime settings keep thresholds easy to tune.
MODEL_NAME = "gpt-4o-mini"
MIN_EXTRACT_CHARS = 200
MAX_RESEARCH_INPUT_CHARS = 6000
MAX_ANALYSIS_INPUT_CHARS = 12000
HTTP_CONNECT_TIMEOUT_SECONDS = 10
HTTP_READ_TIMEOUT_SECONDS = 15
BROWSER_TIMEOUT_MS = 30000
DEFAULT_ALLOWED_ORIGIN_REGEX = r"^chrome-extension://[a-z]{32}$|^http://(127\.0\.0\.1|localhost)(:\d+)?$"
BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain"}
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_REQUESTS = int(os.getenv("FACTGPT_RATE_LIMIT_PER_MINUTE", "45"))
SERVICE_NAME = "factgpt-backend"
COLD_START_WINDOW_SECONDS = 120
# Health probes may run on every popup open, so they stay cheap and unmetered.
HEALTHCHECK_PATHS = {"/", "/health"}
STARTED_AT_UNIX = time.time()
STARTED_AT_MONOTONIC = time.monotonic()


# FastAPI app used by the extension popup/frontend.
app = FastAPI()
request_timestamps_by_client: dict[str, deque[float]] = defaultdict(deque)

# Load local API key used by backend model calls.
load_dotenv(Path(__file__).resolve().parent / "data" / "apikey.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in environment or back-end/data/apikey.env.")

# Shared async OpenAI client for all endpoints.
client = AsyncOpenAI(api_key=api_key)

# Allow the browser extension UI to call this local backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("FACTGPT_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ],
    allow_origin_regex=os.getenv("FACTGPT_ALLOWED_ORIGIN_REGEX", DEFAULT_ALLOWED_ORIGIN_REGEX),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def rate_limit_requests(request: Request, call_next):
    """Simple per-client guard for public AI endpoints."""
    if request.method == "OPTIONS" or request.url.path in HEALTHCHECK_PATHS:
        return await call_next(request)

    forwarded_for = request.headers.get("x-forwarded-for", "")
    client_id = forwarded_for.split(",", 1)[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    timestamps = request_timestamps_by_client[client_id]

    while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait and try again."},
        )

    timestamps.append(now)
    return await call_next(request)


class AnalyzeRequest(BaseModel):
    """Input payload for bias/research endpoints."""
    text: str
    title: str = "Article Analysis"


class AIresultBias(BaseModel):
    """Normalized bias-analysis response returned to the frontend."""
    bias_score: int
    summary: str
    highlights: list[str]
    explanation: str
    missing_perspectives: str


class ResearchSource(BaseModel):
    """Single citation/source entry for claim verification."""
    title: str
    url: str


class ResearchClaim(BaseModel):
    """One fact-checkable claim plus verdict and supporting sources."""
    claim: str
    verdict: str
    evidence_summary: str
    sources: list[ResearchSource]


class AIresultResearch(BaseModel):
    """Normalized research/cross-check response."""
    claims: list[ResearchClaim]
    overall_reliability: str
    notes: str


class LegacyAIresultResearch(BaseModel):
    """Fallback schema for older/looser research outputs from the model."""
    claims: list[str]
    evidence_summary: str
    sources: list[dict]
    verdict: str


class URLRequest(BaseModel):
    """Input payload for URL extraction endpoint."""
    url: HttpUrl


def build_health_response() -> dict[str, Any]:
    """Return liveness and cold-start metrics without touching external services."""
    uptime_seconds = time.monotonic() - STARTED_AT_MONOTONIC
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "server_time_unix": round(time.time(), 3),
        "process_started_at_unix": round(STARTED_AT_UNIX, 3),
        "uptime_seconds": round(uptime_seconds, 3),
        # True wake latency is measured by the client request duration.
        # This flag tells you whether the request hit a recently started process.
        "recent_process_start": uptime_seconds < COLD_START_WINDOW_SECONDS,
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_analysis_text(text: str) -> str:
    """Trim large requests before sending content to model endpoints."""
    normalized = str(text or "").strip()
    if len(normalized) > MAX_ANALYSIS_INPUT_CHARS:
        return normalized[:MAX_ANALYSIS_INPUT_CHARS]
    return normalized


def is_public_ip(address: str) -> bool:
    """Return true only for public internet IP addresses."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.is_global


async def validate_public_url(url: str) -> None:
    """Block local/private network fetch targets before extraction."""
    parsed = httpx.URL(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http and https URLs are supported.")

    hostname = (parsed.host or "").rstrip(".").lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="URL hostname is missing.")
    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(".local"):
        raise HTTPException(status_code=400, detail="Local network URLs are not supported.")

    if not is_public_ip(hostname):
        try:
            addresses = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise HTTPException(status_code=400, detail="URL hostname could not be resolved.") from exc

        resolved_ips = {entry[4][0] for entry in addresses}
        if not resolved_ips or any(not is_public_ip(address) for address in resolved_ips):
            raise HTTPException(status_code=400, detail="Private or internal network URLs are not supported.")


def _load_json_object(candidate: Any) -> dict | None:
    """Best-effort parse for model JSON output (raw dict, string, or fenced JSON)."""
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None

    text = candidate.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_model_json(response: Any) -> dict:
    """Extract the first JSON object from several OpenAI response formats."""
    for candidate in (
        _get(response, "output_parsed"),
        _get(response, "output_text"),
    ):
        parsed = _load_json_object(candidate)
        if parsed:
            return parsed

    for item in _get(response, "output", []) or []:
        parsed = _load_json_object(_get(item, "parsed"))
        if parsed:
            return parsed
        for part in _get(item, "content", []) or []:
            for candidate in (_get(part, "parsed"), _get(part, "text"), _get(part, "output_text")):
                parsed = _load_json_object(candidate)
                if parsed:
                    return parsed

    if _get(response, "status") == "incomplete":
        reason = _get(_get(response, "incomplete_details"), "reason", "unknown")
        raise ValueError(f"Model response incomplete before JSON output (reason: {reason}).")
    raise ValueError("No JSON content found in model response.")


async def run_model_json(
    *,
    prompt: str,
    payload: dict,
    schema_name: str,
    schema: dict,
    max_tokens: int,
    tools: list[dict] | None = None,
    temperature: float | None = None,
) -> Any:
    """Send a prompt + JSON schema request and return the raw model response."""
    request_args = {
        "model": MODEL_NAME,
        "instructions": prompt,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(payload),
                    }
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": max_tokens,
    }

    if tools:
        request_args["tools"] = tools
    if temperature is not None:
        request_args["temperature"] = temperature

    return await client.responses.create(**request_args)


async def analyze_bias(text: str) -> dict:
    """Run the bias prompt and return parsed JSON (or {error})."""
    payload = {
        "task": "bias_analysis",
        "article_text": text,
        "return": "valid JSON only",
    }
    try:
        response = await run_model_json(
            prompt=bias_detector_prompt,
            payload=payload,
            schema_name="bias_result",
            schema=bias_schema[0]["parameters"],
            max_tokens=1000,
            temperature=0.2,
        )
        return parse_model_json(response)
    except Exception as exc:
        return {"error": str(exc)}


async def researcher_ai(text: str) -> dict:
    """Run the research prompt with retry on token-limit truncation."""
    condensed_text = text.strip()[:MAX_RESEARCH_INPUT_CHARS]
    payload = {
        "source_url": "",
        "title": "",
        "content_text": condensed_text,
        "bias_detector_output": {},
        "return": "valid JSON only",
    }

    try:
        last_error: Exception | None = None
        for max_tokens in (1800, 2600):
            # Retry once with more output tokens if the model truncates.
            response = await run_model_json(
                prompt=researcher_prompt,
                payload=payload,
                schema_name="research_schema",
                schema=research_schema[0],
                max_tokens=max_tokens,
                tools=[{"type": "web_search_preview"}],
            )
            try:
                return parse_model_json(response)
            except ValueError as exc:
                last_error = exc
                reason = _get(_get(response, "incomplete_details"), "reason")
                if _get(response, "status") == "incomplete" and reason == "max_output_tokens":
                    continue
                raise

        if last_error:
            raise last_error
        raise ValueError("Research model returned no JSON output.")
    except Exception as exc:
        return {"error": str(exc)}


def extract_readable_text(html: str) -> str:
    """Strip noisy tags and return readable article-like text from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    for selector in ["article", "main", "div.article-body", "div#content"]:
        node = soup.select_one(selector)
        if node:
            return node.get_text("\n", strip=True)

    # Fallback when article-specific containers are missing.
    return soup.body.get_text("\n", strip=True) if soup.body else ""


def looks_like_bot_block(html: str) -> bool:
    """Heuristic check for common bot-protection/challenge pages."""
    lowered = html.lower()
    markers = [
        "enable javascript",
        "access denied",
        "captcha",
        "cf-chl",
        "cloudflare",
        "verify you are human",
        "request blocked",
    ]
    return any(marker in lowered for marker in markers)


async def extract_text_with_httpx(url: str) -> tuple[str, str]:
    """Try direct HTTP fetch first; return (text, error_reason)."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            # Split timeouts so slow servers fail fast and we can try Playwright.
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT_SECONDS,
                read=HTTP_READ_TIMEOUT_SECONDS,
                write=10.0,
                pool=10.0,
            ),
            headers={"User-Agent": "Mozilla/5.0"},
        ) as http:
            response = await http.get(url)
    except httpx.ReadTimeout:
        return "", "Direct fetch timed out while reading the page."
    except httpx.ConnectTimeout:
        return "", "Direct fetch timed out while connecting to the site."
    except httpx.HTTPError as exc:
        return "", f"Network error fetching URL: {exc}"

    html = response.text
    text = extract_readable_text(html)
    # Convert fetch outcomes into one reason string so the caller can report it cleanly.
    if response.status_code >= 400:
        return "", f"Fetch failed with status {response.status_code}."
    if looks_like_bot_block(html):
        return "", "Direct fetch looked blocked by bot protection."
    if len(text) < MIN_EXTRACT_CHARS:
        return "", f"Direct fetch returned too little readable text ({len(text)} chars)."
    return text, ""


async def fetch_html_with_playwright(url: str) -> str:
    """Render a page in headless Chromium for JS-heavy or bot-protected sites."""
    if async_playwright is None:
        raise RuntimeError("Playwright is not installed on the backend.")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT_MS)
            # Brief wait gives client-side rendered article content time to appear.
            await page.wait_for_timeout(1200)
            return await page.content()
        finally:
            await context.close()
            await browser.close()


def validate_ai_bias(ai: dict) -> AIresultBias:
    """Validate and normalize bias JSON before sending to frontend."""
    try:
        return AIresultBias(**ai)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="AI bias response malformed.") from exc


def validate_ai_research(ai: dict) -> AIresultResearch:
    """Validate research JSON, with fallback conversion from older schema."""
    try:
        return AIresultResearch(**ai)
    except Exception:
        pass

    try:
        legacy = LegacyAIresultResearch(**ai)
        # Normalize older source shape into the current strict schema.
        sources: list[dict[str, str]] = []
        for source in legacy.sources:
            if not isinstance(source, dict):
                continue
            url = str(source.get("url", "")).strip()
            if not url:
                continue
            title = str(source.get("title") or source.get("name") or "Source").strip() or "Source"
            sources.append({"title": title, "url": url})

        claims = [str(claim).strip() for claim in legacy.claims if str(claim).strip()]
        if not claims:
            claims = ["Claim details not provided"]

        # Coarse reliability mapping derived from the legacy overall verdict wording.
        reliability = "medium"
        verdict_lower = legacy.verdict.lower()
        if "support" in verdict_lower:
            reliability = "high"
        elif "contradict" in verdict_lower:
            reliability = "low"

        normalized = {
            "claims": [
                {
                    "claim": claim,
                    "verdict": legacy.verdict,
                    "evidence_summary": legacy.evidence_summary,
                    "sources": sources,
                }
                for claim in claims
            ],
            "overall_reliability": reliability,
            "notes": legacy.evidence_summary,
        }
        return AIresultResearch(**normalized)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="AI research response malformed.") from exc


@app.get("/")
async def root():
    """Backwards-compatible root probe for hosts and manual checks."""
    return build_health_response()


@app.head("/")
async def root_head():
    """Allow uptime monitors that use HEAD instead of GET."""
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    """Cheap wake/liveness endpoint for Render and extension warm-up pings."""
    return build_health_response()


@app.head("/health")
async def health_check_head():
    """Allow uptime monitors that use HEAD instead of GET."""
    return Response(status_code=200)


@app.post("/extract")
async def extract_text(req: URLRequest):
    """Fast path: extract readable article text from a URL using httpx + BeautifulSoup only."""
    url = str(req.url)
    await validate_public_url(url)

    text, reason = await extract_text_with_httpx(url)
    if text:
        return {"status": "extracted", "method": "httpx", "text": text}

    # Return early so the frontend can try lighter fallbacks (tab DOM text) before Playwright.
    raise HTTPException(status_code=502, detail=reason or "Direct fetch failed.")


@app.post("/extract-rendered")
async def extract_text_rendered(req: URLRequest):
    """Last-resort extraction using Playwright for JS-heavy or protected pages."""
    url = str(req.url)
    await validate_public_url(url)
    try:
        rendered_html = await fetch_html_with_playwright(url)
        rendered_text = extract_readable_text(rendered_html)
        if len(rendered_text) < MIN_EXTRACT_CHARS:
            raise ValueError("Rendered page still produced too little readable text.")
        return {"status": "extracted", "method": "playwright", "text": rendered_text}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Playwright extraction failed: {exc}") from exc


@app.post("/analyze")
async def analyze(article: AnalyzeRequest):
    """Run bias + research in parallel and return the combined response."""
    text = normalize_analysis_text(article.text)
    if len(text) < MIN_EXTRACT_CHARS:
        raise HTTPException(status_code=400, detail="Not enough text to analyze.")

    bias_raw, research_raw = await asyncio.gather(
        analyze_bias(text),
        researcher_ai(text),
    )

    if "error" in bias_raw:
        raise HTTPException(status_code=502, detail=bias_raw["error"])
    if "error" in research_raw:
        raise HTTPException(status_code=502, detail=research_raw["error"])

    return {
        "status": "analyzed",
        "ai_result": validate_ai_bias(bias_raw),
        "ai_research": validate_ai_research(research_raw),
    }


@app.post("/analyze-bias")
async def receive_bias(article: AnalyzeRequest):
    """Run only the bias-analysis step."""
    text = normalize_analysis_text(article.text)
    if len(text) < MIN_EXTRACT_CHARS:
        raise HTTPException(status_code=400, detail="Not enough text to analyze.")

    bias_raw = await analyze_bias(text)
    if "error" in bias_raw:
        raise HTTPException(status_code=502, detail=bias_raw["error"])
    return {"status": "bias_analyzed", "ai_result": validate_ai_bias(bias_raw)}


@app.post("/research")
async def receive_research(article: AnalyzeRequest):
    """Run only the research/cross-check step."""
    text = normalize_analysis_text(article.text)
    if len(text) < MIN_EXTRACT_CHARS:
        raise HTTPException(status_code=400, detail="Not enough text to research.")

    research_raw = await researcher_ai(text)
    if "error" in research_raw:
        raise HTTPException(status_code=502, detail=research_raw["error"])
    return {"status": "researched", "ai_research": validate_ai_research(research_raw)}


if __name__ == "__main__":
    # Local development entrypoint.
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
