"""Fact GPT's FastAPI application, extraction pipeline, and OpenAI orchestration."""

import asyncio
from collections import defaultdict, deque
from functools import lru_cache
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import pickle
import re
import secrets
import socket
import tempfile
import time
from typing import Any, Callable, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
import uvicorn
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from ai_prompts import (
    bias_detector_prompt,
    bias_schema,
    fact_opinion_prompt,
    fact_opinion_schema,
    podcast_bias_synthesis_prompt,
    podcast_bias_synthesis_schema,
    research_schema,
    researcher_prompt,
)
from podcast import (
    TRANSCRIPT_MIME_TYPES,
    PodcastJobRequest,
    PodcastSegment,
    PodcastTranscript,
    canonicalize_transcript,
    fetch_public_bytes,
    inspect_podcast_page,
    parse_publisher_transcript,
    probe_duration,
    select_rss_episode,
    transcode_audio_chunks,
    transcribe_audio_chunks,
)

try:
    from playwright.async_api import async_playwright
except Exception:
    # Playwright is optional; extraction fallback will report a clear error if missing.
    async_playwright = None


def parse_env_bool(value: str | None) -> bool:
    """Parse common truthy environment variable values."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


# Supported article quotation marks. Straight double quotes are ambiguous until
# they are paired; typographic marks have explicit opening/closing roles.
QUOTE_MARKS = {
    '"': "ambiguous",
    "\u201c": "opening",
    "\u201d": "closing",
    "\u00ab": "opening",
    "\u00bb": "closing",
}
QUOTE_PAIRS = {
    '"': '"',
    "\u201c": "\u201d",
    "\u00ab": "\u00bb",
}


def find_quote_locations(text: str) -> list[dict[str, Any]]:
    """Locate supported quote delimiters with offsets and 1-based positions."""
    locations: list[dict[str, Any]] = []
    line = 1
    column = 1
    article_text = str(text or "")

    for offset, character in enumerate(article_text):
        role = QUOTE_MARKS.get(character)
        preceding_backslashes = 0
        backslash_offset = offset - 1
        while backslash_offset >= 0 and article_text[backslash_offset] == "\\":
            preceding_backslashes += 1
            backslash_offset -= 1

        if role and preceding_backslashes % 2 == 0:
            locations.append(
                {
                    "quote": character,
                    "role": role,
                    "offset": offset,
                    "line": line,
                    "column": column,
                }
            )

        if character == "\n":
            line += 1
            column = 1
        else:
            column += 1

    return locations


def extract_quoted_phrases(
    text: str,
    quote_locations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Pair quote locations and return exact externally authored text spans."""
    article_text = str(text or "")
    locations = quote_locations if quote_locations is not None else find_quote_locations(article_text)
    spans: list[dict[str, Any]] = []
    open_quotes: list[dict[str, Any]] = []

    for location in locations:
        quote = location["quote"]
        role = location["role"]

        if role == "opening":
            open_quotes.append(location)
            continue

        if role == "ambiguous":
            if open_quotes and open_quotes[-1]["quote"] == quote:
                opening = open_quotes.pop()
            else:
                open_quotes.append(location)
                continue
        else:
            opening_index = next(
                (
                    index
                    for index in range(len(open_quotes) - 1, -1, -1)
                    if QUOTE_PAIRS.get(open_quotes[index]["quote"]) == quote
                ),
                None,
            )
            if opening_index is None:
                continue
            opening = open_quotes[opening_index]
            # A directional closer also discards any malformed nested opener so
            # it cannot pair with an unrelated quote later in the article.
            del open_quotes[opening_index:]

        content_start = opening["offset"] + len(opening["quote"])
        content_end = location["offset"]
        phrase = article_text[content_start:content_end]
        if not phrase.strip():
            continue

        spans.append(
            {
                "text": phrase,
                "opening_quote": opening["quote"],
                "closing_quote": quote,
                "start_offset": content_start,
                "end_offset": content_end,
                "start_line": opening["line"],
                "end_line": location["line"],
                "opening_column": opening["column"],
                "closing_column": location["column"],
                "attribution": "external_speaker_or_author",
            }
        )

    return sorted(spans, key=lambda span: span["start_offset"])


# Centralized runtime settings keep thresholds easy to tune.
MODEL_NAME = "gpt-4o-mini"
FACT_OPINION_API_MODEL = "gpt-5.6-sol"
RESEARCH_API_MODEL = "gpt-5.5"
MIN_EXTRACT_CHARS = 200
MAX_RESEARCH_INPUT_CHARS = 6000
MAX_ANALYSIS_INPUT_CHARS = 12000
MAX_REQUEST_TEXT_CHARS = int(os.getenv("FACTGPT_MAX_REQUEST_TEXT_CHARS", "50000"))
MAX_REQUEST_BODY_BYTES = int(os.getenv("FACTGPT_MAX_REQUEST_BODY_BYTES", "200000"))
MAX_FETCH_BYTES = int(os.getenv("FACTGPT_MAX_FETCH_BYTES", "1000000"))
MAX_EXTRACTED_TEXT_CHARS = int(os.getenv("FACTGPT_MAX_EXTRACTED_TEXT_CHARS", str(MAX_ANALYSIS_INPUT_CHARS)))
MAX_REDIRECTS = int(os.getenv("FACTGPT_MAX_REDIRECTS", "5"))
HTTP_CONNECT_TIMEOUT_SECONDS = 10
HTTP_READ_TIMEOUT_SECONDS = 15
BROWSER_TIMEOUT_MS = 30000
DEFAULT_ALLOWED_ORIGIN_REGEX = r"^http://(127\.0\.0\.1|localhost)(:\d+)?$"
BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain"}
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_REQUESTS = int(os.getenv("FACTGPT_RATE_LIMIT_PER_MINUTE", "45"))
PROTECTED_PATHS = {
    "/analyze",
    "/analyze-bias",
    "/article-jobs",
    "/classify-fact-opinion",
    "/research",
    "/extract",
    "/extract-rendered",
    "/podcast-jobs",
}
SERVICE_NAME = "factgpt-backend"
COLD_START_WINDOW_SECONDS = 120
PUBLIC_API_TOKEN = os.getenv("FACTGPT_PUBLIC_API_TOKEN", "").strip()
REQUIRE_API_TOKEN = parse_env_bool(os.getenv("FACTGPT_REQUIRE_API_TOKEN")) or bool(PUBLIC_API_TOKEN)
REQUIRE_ALLOWED_ORIGIN = parse_env_bool(os.getenv("FACTGPT_REQUIRE_ALLOWED_ORIGIN"))
TRUST_X_FORWARDED_FOR = parse_env_bool(os.getenv("FACTGPT_TRUST_X_FORWARDED_FOR"))
TRUSTED_PROXY_IPS = {
    ip.strip()
    for ip in os.getenv("FACTGPT_TRUSTED_PROXY_IPS", "").split(",")
    if ip.strip()
}
ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv("FACTGPT_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}
ALLOWED_ORIGIN_REGEX = os.getenv("FACTGPT_ALLOWED_ORIGIN_REGEX", DEFAULT_ALLOWED_ORIGIN_REGEX)
ALLOWED_ORIGIN_PATTERN = re.compile(ALLOWED_ORIGIN_REGEX) if ALLOWED_ORIGIN_REGEX else None
# Health probes may run on every popup open, so they stay cheap and unmetered.
HEALTHCHECK_PATHS = {"/", "/health"}
STARTED_AT_UNIX = time.time()
STARTED_AT_MONOTONIC = time.monotonic()
FACT_OPINION_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fact_opinion"
    / "processed"
    / "fact_opinion_classifier.pkl"
)
MAX_CLASSIFICATION_ITEMS = 100
MAX_CLASSIFICATION_TEXT_CHARS = 5_000
MAX_ARTICLE_SEGMENT_CHARS = 2_000
MAX_CLASSIFICATION_TOTAL_CHARS = 12_000
MAX_OPENAI_CLASSIFICATION_ITEMS = 25
MAX_OPENAI_CLASSIFICATION_CHARS = 6_000
OPENAI_CLASSIFICATION_TIMEOUT_SECONDS = 45
OPENAI_CLASSIFICATION_TOKEN_LIMITS = (4_000, 6_000)
OPENAI_BIAS_TIMEOUT_SECONDS = float(
    os.getenv("FACTGPT_OPENAI_BIAS_TIMEOUT_SECONDS", "60")
)
OPENAI_RESEARCH_TIMEOUT_SECONDS = float(
    os.getenv("FACTGPT_OPENAI_RESEARCH_TIMEOUT_SECONDS", "120")
)
MAX_PODCAST_AUDIO_BYTES = int(
    os.getenv("FACTGPT_MAX_PODCAST_AUDIO_BYTES", "200000000")
)
MAX_PODCAST_DURATION_SECONDS = int(
    os.getenv("FACTGPT_MAX_PODCAST_DURATION_SECONDS", str(3 * 60 * 60))
)
MAX_PODCAST_TRANSCRIPT_BYTES = int(
    os.getenv("FACTGPT_MAX_PODCAST_TRANSCRIPT_BYTES", "5000000")
)
MAX_PODCAST_PAGE_BYTES = int(
    os.getenv("FACTGPT_MAX_PODCAST_PAGE_BYTES", "2000000")
)
PODCAST_JOB_TTL_SECONDS = int(
    os.getenv("FACTGPT_PODCAST_JOB_TTL_SECONDS", str(24 * 60 * 60))
)
ARTICLE_JOB_TTL_SECONDS = int(
    os.getenv("FACTGPT_ARTICLE_JOB_TTL_SECONDS", str(24 * 60 * 60))
)
PODCAST_COMPACT_ITEMS = 100

# The local artifact is binary, so it cannot represent mixed passages. These
# deliberately high-precision cues send likely mixed facts for semantic review.
SUBJECTIVITY_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "evaluative_language",
        re.compile(
            r"\b(?:awful|best|brilliant|disgraceful|evil|excellent|horrible|"
            r"idiotic|outrageous|reckless|shameful|terrible|wonderful|worst)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "loaded_framing",
        re.compile(
            r"\b(?:clearly|obviously|fortunately|unfortunately|undoubtedly|"
            r"disaster|catastrophe|heroic)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "personal_stance",
        re.compile(
            r"\b(?:I\s+(?:think|believe|feel)|in\s+my\s+opinion|we\s+(?:think|believe|feel))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "normative_or_predictive",
        re.compile(
            r"\b(?:should|shouldn't|ought|must|mustn't|may|might|probably|"
            r"perhaps|likely|unlikely)\b",
            re.IGNORECASE,
        ),
    ),
)


# FastAPI app used by the extension popup/frontend.
app = FastAPI()
logger = logging.getLogger(__name__)
request_timestamps_by_client: dict[str, deque[float]] = defaultdict(deque)
fact_opinion_cache: dict[str, Any] = {}
fact_opinion_cache_order: deque[str] = deque()
FACT_OPINION_CACHE_SIZE = 128
podcast_jobs: dict[str, dict[str, Any]] = {}
podcast_jobs_by_url: dict[str, str] = {}
podcast_job_tasks: dict[str, asyncio.Task] = {}
article_jobs: dict[str, dict[str, Any]] = {}
article_jobs_by_content: dict[str, str] = {}
article_job_tasks: dict[str, asyncio.Task] = {}

# Load local API key used by backend model calls.
load_dotenv(Path(__file__).resolve().parent / "data" / "apikey.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in environment or back-end/data/apikey.env.")
if REQUIRE_API_TOKEN and not PUBLIC_API_TOKEN:
    raise RuntimeError("FACTGPT_REQUIRE_API_TOKEN is true, but FACTGPT_PUBLIC_API_TOKEN is not set.")

# A dedicated secret is preferred. Falling back to a domain-separated digest of
# the backend-only OpenAI key keeps signed classification handoffs stable across
# workers/restarts without exposing either secret to the extension.
classification_signing_secret = os.getenv("FACTGPT_CLASSIFICATION_SIGNING_KEY")
CLASSIFICATION_SIGNING_KEY = (
    classification_signing_secret.encode("utf-8")
    if classification_signing_secret
    else hashlib.sha256(f"factgpt-classification:{api_key}".encode("utf-8")).digest()
)

# Shared async OpenAI client for all endpoints.
client = AsyncOpenAI(api_key=api_key)

# Allow the browser extension UI to call this local backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-FactGPT-API-Key"],
)


def origin_is_allowed(origin: str) -> bool:
    """Return whether a browser origin is allowed to use protected routes."""
    if not origin:
        return False
    if origin in ALLOWED_ORIGINS:
        return True
    return bool(ALLOWED_ORIGIN_PATTERN and ALLOWED_ORIGIN_PATTERN.fullmatch(origin))


def request_api_token(request: Request) -> str:
    """Read a caller token without assuming a specific client transport."""
    bearer_prefix = "Bearer "
    authorization = request.headers.get("authorization", "")
    if authorization.startswith(bearer_prefix):
        return authorization[len(bearer_prefix):].strip()
    return request.headers.get("x-factgpt-api-key", "").strip()


def request_has_valid_api_token(request: Request) -> bool:
    """Validate the optional public API token in constant time."""
    if not REQUIRE_API_TOKEN:
        return True
    return hmac.compare_digest(request_api_token(request), PUBLIC_API_TOKEN)


def is_protected_path(path: str) -> bool:
    """Return whether a request path may trigger expensive backend work."""
    return (
        path in PROTECTED_PATHS
        or path.startswith("/podcast-jobs/")
        or path.startswith("/article-jobs/")
    )


def client_identifier(request: Request) -> str:
    """Choose a rate-limit key without trusting spoofable proxy headers by default."""
    peer_host = request.client.host if request.client else ""
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for and (TRUST_X_FORWARDED_FOR or peer_host in TRUSTED_PROXY_IPS):
        return forwarded_for.split(",", 1)[0].strip() or peer_host or "unknown"
    return peer_host or "unknown"


@app.middleware("http")
async def guard_public_requests(request: Request, call_next):
    """Apply cheap public-edge checks before model/browser work starts."""
    if request.method == "OPTIONS" or request.url.path in HEALTHCHECK_PATHS:
        return await call_next(request)

    if is_protected_path(request.url.path):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body is too large."},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})

        if REQUIRE_ALLOWED_ORIGIN and not origin_is_allowed(request.headers.get("origin", "")):
            return JSONResponse(status_code=403, content={"detail": "Origin is not allowed."})

        if not request_has_valid_api_token(request):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid API token."})

    client_id = client_identifier(request)
    now = time.monotonic()
    # This sliding window is process-local; multiple server workers do not share it.
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
    text: str = Field(..., max_length=MAX_REQUEST_TEXT_CHARS)
    title: str = Field(default="Article Analysis", max_length=200)


class ArticleJobRequest(BaseModel):
    """Article input queued for backend-owned extraction and analysis."""

    page_url: HttpUrl = Field(..., max_length=2048)
    client_request_id: str = Field(default="", max_length=128)
    text: str = Field(default="", max_length=MAX_REQUEST_TEXT_CHARS)
    title: str = Field(default="Article Analysis", max_length=200)

    @model_validator(mode="after")
    def normalize_fields(self):
        self.text = str(self.text or "").strip()
        self.title = str(self.title or "").strip() or "Article Analysis"
        self.client_request_id = str(self.client_request_id or "").strip()
        return self


class FactOpinionItem(BaseModel):
    """One text item that should retain its identity through classification."""

    id: str | int | None = None
    text: str = Field(..., min_length=1, max_length=MAX_CLASSIFICATION_TEXT_CHARS)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def normalize_text(self):
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("Classification text cannot be blank.")
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("Classification offsets must be supplied together.")
        if self.start_offset is not None and self.end_offset <= self.start_offset:
            raise ValueError("Classification end_offset must follow start_offset.")
        return self


class FactOpinionRequest(BaseModel):
    """Batch of independent text items for the hybrid classifier."""

    items: list[FactOpinionItem] = Field(
        ..., min_length=1, max_length=MAX_CLASSIFICATION_ITEMS
    )
    title: str = Field(default="Article Analysis", max_length=200)

    @model_validator(mode="after")
    def validate_total_text(self):
        if sum(len(item.text) for item in self.items) > MAX_CLASSIFICATION_TOTAL_CHARS:
            raise ValueError(
                f"Classification text exceeds {MAX_CLASSIFICATION_TOTAL_CHARS} characters."
            )
        return self


class LocalFactOpinionPrediction(BaseModel):
    """Auditable output from the saved sklearn classifier."""

    label: Literal["fact", "opinion"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    log_probability: float
    accepted: bool
    review_reasons: list[str] = Field(default_factory=list, max_length=8)


class FinalFactOpinionPrediction(BaseModel):
    """Final downstream decision after optional OpenAI review."""

    status: Literal["resolved", "unresolved"]
    label: Literal["fact", "opinion", "mixed"] | None
    source: Literal["local", "openai", "unresolved"]
    explanation: str | None = Field(default=None, max_length=240)
    opinion_excerpts: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_resolution(self):
        if self.status == "resolved" and self.label is None:
            raise ValueError("Resolved classifications require a label.")
        if self.status == "unresolved" and self.label is not None:
            raise ValueError("Unresolved classifications cannot have a label.")
        if self.status == "unresolved" and self.opinion_excerpts:
            raise ValueError("Unresolved classifications cannot have opinion excerpts.")
        if self.label == "mixed" and not self.opinion_excerpts:
            raise ValueError("Mixed classifications require opinion excerpts.")
        if self.label in {"fact", "opinion"} and self.opinion_excerpts:
            raise ValueError("Pure fact/opinion classifications cannot have opinion excerpts.")
        return self


class FactOpinionResultItem(BaseModel):
    """Canonical transferable classification for one exact text segment."""

    id: str | int
    text: str
    start_offset: int | None = None
    end_offset: int | None = None
    local_prediction: LocalFactOpinionPrediction
    final_prediction: FinalFactOpinionPrediction


class FactOpinionCounts(BaseModel):
    fact: int = Field(..., ge=0)
    opinion: int = Field(..., ge=0)
    mixed: int = Field(default=0, ge=0)
    unresolved: int = Field(..., ge=0)
    openai_reviewed: int = Field(..., ge=0)


class FactOpinionResult(BaseModel):
    """Complete local-plus-OpenAI result shared across backend stages."""

    status: Literal["classified", "partial"]
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    counts: FactOpinionCounts
    verification_token: str | None = Field(default=None, min_length=64, max_length=64)
    items: list[FactOpinionResultItem] = Field(
        ..., min_length=1, max_length=MAX_CLASSIFICATION_ITEMS
    )


@lru_cache(maxsize=1)
def load_fact_opinion_classifier():
    """Load the trusted local sklearn artifact once per backend process."""
    with FACT_OPINION_MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    if not hasattr(model, "predict_log_proba"):
        raise ValueError("Fact-opinion model does not provide log probabilities.")
    if not hasattr(model, "confidence_threshold_"):
        raise ValueError("Fact-opinion model is missing its confidence threshold.")
    if set(map(str, model.classes_)) != {"fact", "opinion"}:
        raise ValueError("Fact-opinion model has unexpected classes.")
    return model


def local_review_reasons(text: str, label: str, accepted: bool) -> list[str]:
    """Return deterministic reasons a binary local decision needs semantic review."""
    reasons: list[str] = []
    if not accepted:
        reasons.append("low_confidence")
    # Excluding a factual statement is consequential, so every local opinion
    # receives confirmation before it is withheld from bias and research.
    if label == "opinion":
        reasons.append("factual_exclusion_risk")
    elif label == "fact":
        for reason, pattern in SUBJECTIVITY_CUE_PATTERNS:
            if pattern.search(text):
                reasons.append(f"possible_mixed:{reason}")
    return reasons


def classify_fact_opinion_items(items: list[FactOpinionItem]) -> dict[str, Any]:
    """Run the saved local classifier and build the canonical result shell."""
    model = load_fact_opinion_classifier()
    threshold = float(model.confidence_threshold_)
    log_threshold = math.log(threshold)
    texts = [item.text for item in items]
    log_probabilities = model.predict_log_proba(texts)
    classes = [str(label) for label in model.classes_]
    results: list[dict[str, Any]] = []

    for position, (item, row) in enumerate(zip(items, log_probabilities)):
        class_index = int(row.argmax())
        log_probability = float(row[class_index])
        confidence = math.exp(log_probability)
        accepted = log_probability >= log_threshold
        local_label = classes[class_index]
        review_reasons = local_review_reasons(item.text, local_label, accepted)
        resolved_locally = accepted and not review_reasons
        results.append(
            {
                "id": item.id if item.id is not None else position,
                "text": item.text,
                "start_offset": item.start_offset,
                "end_offset": item.end_offset,
                "local_prediction": {
                    "label": local_label,
                    "confidence": round(confidence, 6),
                    "log_probability": round(log_probability, 6),
                    "accepted": accepted,
                    "review_reasons": review_reasons,
                },
                "final_prediction": {
                    "status": "resolved" if resolved_locally else "unresolved",
                    "label": local_label if resolved_locally else None,
                    "source": "local" if resolved_locally else "unresolved",
                    "explanation": (
                        "The binary local classifier resolved this as a clean factual statement."
                        if resolved_locally
                        else None
                    ),
                    "opinion_excerpts": [],
                },
            }
        )

    return {
        "status": "partial" if any(
            item["local_prediction"]["review_reasons"] for item in results
        ) else "classified",
        "confidence_threshold": threshold,
        "counts": _fact_opinion_counts(results),
        "items": results,
    }


SENTENCE_BOUNDARY_RE = re.compile(
    r"(?P<punct>[.!?]+)(?P<closers>[\"\u201d\u2019\u00bb)\]]*)(?=\s+|$)"
)
COMMON_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
}


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _is_abbreviation_boundary(text: str, match: re.Match[str]) -> bool:
    if match.group("punct") != ".":
        return False
    prefix = text[: match.start("punct")]
    token_match = re.search(r"([A-Za-z](?:[A-Za-z.]*)?)$", prefix)
    if not token_match:
        return False
    token = token_match.group(1)
    return (
        token.casefold() in COMMON_ABBREVIATIONS
        or len(token) == 1
        or (token.count(".") >= 1 and len(token) <= 6)
    )


def _split_long_span(text: str, start: int, end: int) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > MAX_ARTICLE_SEGMENT_CHARS:
        limit = cursor + MAX_ARTICLE_SEGMENT_CHARS
        split_at = text.rfind(" ", cursor + 1, limit + 1)
        newline_at = text.rfind("\n", cursor + 1, limit + 1)
        split_at = max(split_at, newline_at)
        if split_at <= cursor:
            split_at = limit
        trimmed = _trimmed_span(text, cursor, split_at)
        if trimmed:
            pieces.append(trimmed)
        cursor = split_at
    trimmed = _trimmed_span(text, cursor, end)
    if trimmed:
        pieces.append(trimmed)
    return pieces


def segment_article(text: str) -> list[FactOpinionItem]:
    """Split an article into exact, offset-preserving classifier segments."""
    article_text = str(text or "")[:MAX_CLASSIFICATION_TOTAL_CHARS]
    boundaries: set[int] = {len(article_text)}
    for match in SENTENCE_BOUNDARY_RE.finditer(article_text):
        if not _is_abbreviation_boundary(article_text, match):
            boundaries.add(match.end())
    for match in re.finditer(r"\n\s*\n", article_text):
        boundaries.add(match.start())

    spans: list[tuple[int, int]] = []
    start = 0
    for end in sorted(boundaries):
        if end <= start:
            continue
        trimmed = _trimmed_span(article_text, start, end)
        if trimmed:
            spans.extend(_split_long_span(article_text, *trimmed))
        start = end

    # Attach heading-like fragments to their neighbor so tiny units are not
    # sent through a sentence-trained classifier by themselves.
    coalesced: list[tuple[int, int]] = []
    index = 0
    while index < len(spans):
        current = spans[index]
        if current[1] - current[0] < 20 and index + 1 < len(spans):
            candidate = _trimmed_span(article_text, current[0], spans[index + 1][1])
            if candidate and candidate[1] - candidate[0] <= MAX_ARTICLE_SEGMENT_CHARS:
                coalesced.append(candidate)
                index += 2
                continue
        if current[1] - current[0] < 20 and coalesced:
            candidate = _trimmed_span(article_text, coalesced[-1][0], current[1])
            if candidate and candidate[1] - candidate[0] <= MAX_ARTICLE_SEGMENT_CHARS:
                coalesced[-1] = candidate
                index += 1
                continue
        coalesced.append(current)
        index += 1

    while len(coalesced) > MAX_CLASSIFICATION_ITEMS:
        merge_candidates = [
            (coalesced[i + 1][1] - coalesced[i][0], i)
            for i in range(len(coalesced) - 1)
            if coalesced[i + 1][1] - coalesced[i][0] <= MAX_ARTICLE_SEGMENT_CHARS
        ]
        if not merge_candidates:
            raise ValueError("Article cannot be represented within classification limits.")
        _, merge_index = min(merge_candidates)
        coalesced[merge_index : merge_index + 2] = [
            (coalesced[merge_index][0], coalesced[merge_index + 1][1])
        ]

    return [
        FactOpinionItem(
            id=f"segment-{position + 1:04d}",
            text=article_text[start:end],
            start_offset=start,
            end_offset=end,
        )
        for position, (start, end) in enumerate(coalesced)
    ]


def _fact_opinion_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "fact": 0,
        "opinion": 0,
        "mixed": 0,
        "unresolved": 0,
        "openai_reviewed": 0,
    }
    for item in items:
        final = item["final_prediction"]
        if final["status"] == "unresolved":
            counts["unresolved"] += 1
            continue
        counts[str(final["label"])] += 1
        if final["source"] == "openai":
            counts["openai_reviewed"] += 1
    return counts


def _openai_classification_batches(
    ambiguous_indexes: list[int],
    items: list[FactOpinionItem],
    include_neighbor_context: bool = False,
) -> list[list[tuple[int, dict[str, str]]]]:
    batches: list[list[tuple[int, dict[str, str]]]] = []
    current: list[tuple[int, dict[str, str]]] = []
    current_chars = 0

    for index in ambiguous_indexes:
        api_id = f"item-{index:04d}"
        context_budget = max(
            0,
            MAX_OPENAI_CLASSIFICATION_CHARS - len(api_id) - len(items[index].text),
        )
        before_budget = min(500, context_budget // 2)
        after_budget = min(500, context_budget - before_budget)
        before = (
            items[index - 1].text[-before_budget:]
            if include_neighbor_context and index and before_budget
            else ""
        )
        after = (
            items[index + 1].text[:after_budget]
            if include_neighbor_context and index + 1 < len(items) and after_budget
            else ""
        )
        api_item = {
            "id": api_id,
            "text": items[index].text,
            "context_before": before,
            "context_after": after,
        }
        item_chars = sum(len(value) for value in api_item.values())
        if current and (
            len(current) >= MAX_OPENAI_CLASSIFICATION_ITEMS
            or current_chars + item_chars > MAX_OPENAI_CLASSIFICATION_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append((index, api_item))
        current_chars += item_chars

    if current:
        batches.append(current)
    return batches


def _valid_opinion_excerpts(text: str, excerpts: Any) -> bool:
    if not isinstance(excerpts, list) or len(excerpts) > 3:
        return False
    occupied: list[tuple[int, int]] = []
    for excerpt in excerpts:
        if not isinstance(excerpt, str) or not excerpt.strip() or len(excerpt) > 500:
            return False
        search_from = 0
        selected: tuple[int, int] | None = None
        while True:
            start = text.find(excerpt, search_from)
            if start < 0:
                break
            candidate = (start, start + len(excerpt))
            if all(candidate[1] <= old[0] or candidate[0] >= old[1] for old in occupied):
                selected = candidate
                break
            search_from = start + 1
        if selected is None:
            return False
        occupied.append(selected)
    return True


def _validated_api_decisions(
    batch: list[tuple[int, dict[str, str]]], parsed: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    expected = {api_item["id"]: (index, api_item["text"]) for index, api_item in batch}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_items = parsed.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != len(expected):
        return {}
    for raw in raw_items:
        if isinstance(raw, dict) and isinstance(raw.get("id"), str):
            grouped[raw["id"]].append(raw)
    if set(grouped) != set(expected):
        return {}

    decisions: dict[int, dict[str, Any]] = {}
    for api_id, (index, text) in expected.items():
        matches = grouped.get(api_id, [])
        if len(matches) != 1:
            continue
        raw = matches[0]
        label = raw.get("label")
        explanation = raw.get("explanation")
        excerpts = raw.get("opinion_excerpts")
        if label not in {"fact", "opinion", "mixed"}:
            continue
        if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 240:
            continue
        if label in {"fact", "opinion"} and excerpts != []:
            continue
        if not _valid_opinion_excerpts(text, excerpts):
            continue
        if label == "mixed" and not excerpts:
            continue
        decisions[index] = {
            "status": "resolved",
            "label": label,
            "source": "openai",
            "explanation": explanation.strip(),
            "opinion_excerpts": excerpts,
        }
    return decisions


async def _classify_openai_batch(
    batch: list[tuple[int, dict[str, str]]], title: str
) -> dict[int, dict[str, Any]]:
    payload = {
        "title": title,
        "items": [api_item for _, api_item in batch],
    }
    last_response: Any = None
    deadline = (
        asyncio.get_running_loop().time()
        + OPENAI_CLASSIFICATION_TIMEOUT_SECONDS
    )
    for max_tokens in OPENAI_CLASSIFICATION_TOKEN_LIMITS:
        remaining_seconds = deadline - asyncio.get_running_loop().time()
        if remaining_seconds <= 0:
            raise TimeoutError("OpenAI classification exceeded its time limit.")
        response = await asyncio.wait_for(
            run_model_json(
                prompt=fact_opinion_prompt,
                payload=payload,
                schema_name="fact_opinion_result",
                schema=fact_opinion_schema,
                max_tokens=max_tokens,
                model=FACT_OPINION_API_MODEL,
                reasoning={"effort": "medium"},
                store=False,
                timeout=remaining_seconds,
            ),
            timeout=remaining_seconds,
        )
        last_response = response
        try:
            return _validated_api_decisions(batch, parse_model_json(response))
        except ValueError:
            reason = _get(_get(response, "incomplete_details"), "reason")
            if _get(response, "status") == "incomplete" and reason == "max_output_tokens":
                continue
            raise
    if last_response is not None:
        return _validated_api_decisions(batch, parse_model_json(last_response))
    return {}


async def resolve_fact_opinion_items(
    items: list[FactOpinionItem],
    title: str = "Article Analysis",
    include_neighbor_context: bool = False,
) -> FactOpinionResult:
    """Resolve exclusion-sensitive and possible-mixed decisions with bounded OpenAI calls."""
    local_result = await asyncio.to_thread(classify_fact_opinion_items, items)
    review_indexes = [
        index
        for index, item in enumerate(local_result["items"])
        if item["local_prediction"]["review_reasons"]
    ]

    for batch in _openai_classification_batches(
        review_indexes, items, include_neighbor_context=include_neighbor_context
    ):
        try:
            decisions = await _classify_openai_batch(batch, title)
        except Exception as exc:
            error_id = hashlib.sha256(
                f"{time.time_ns()}:{type(exc).__name__}".encode("utf-8")
            ).hexdigest()[:12]
            logger.warning(
                "Fact-opinion review failed; error_id=%s error_type=%s",
                error_id,
                type(exc).__name__,
            )
            decisions = {}
        else:
            error_id = ""
        for index, _ in batch:
            local_result["items"][index]["final_prediction"] = decisions.get(
                index,
                {
                    "status": "unresolved",
                    "label": None,
                    "source": "unresolved",
                    "explanation": (
                        "Semantic review was unavailable or invalid."
                        + (f" Reference: {error_id}." if error_id else "")
                    ),
                    "opinion_excerpts": [],
                },
            )

    local_result["counts"] = _fact_opinion_counts(local_result["items"])
    local_result["status"] = (
        "partial" if local_result["counts"]["unresolved"] else "classified"
    )
    return FactOpinionResult(**local_result)


async def classify_article_fact_opinion(
    text: str, title: str = "Article Analysis"
) -> FactOpinionResult:
    segments = segment_article(text)
    if not segments:
        raise ValueError("Article contains no classifiable text.")
    result = await resolve_fact_opinion_items(
        segments, title, include_neighbor_context=True
    )
    result = sign_article_classification(text, title, result)
    cache_article_classification(text, title, result)
    return result


def _fact_opinion_cache_key(text: str, title: str) -> str:
    """Identify the exact article/title pair without retaining duplicate raw text."""
    payload = f"{title}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _classification_signature(
    text: str, title: str, result: FactOpinionResult
) -> str:
    """Sign an exact article/result pair for safe transfer between workers."""
    unsigned_result = result.model_dump(
        mode="json", exclude={"verification_token"}
    )
    payload = json.dumps(
        {
            "article_key": _fact_opinion_cache_key(text, title),
            "result": unsigned_result,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(CLASSIFICATION_SIGNING_KEY, payload, hashlib.sha256).hexdigest()


def sign_article_classification(
    text: str, title: str, result: FactOpinionResult
) -> FactOpinionResult:
    """Return a transferable result authenticated by a server-only key."""
    return result.model_copy(
        deep=True,
        update={"verification_token": _classification_signature(text, title, result)},
    )


def article_classification_is_authentic(
    text: str, title: str, result: FactOpinionResult
) -> bool:
    """Verify both the signature and the exact offset-preserving segmentation."""
    token = result.verification_token
    if not token or not hmac.compare_digest(
        token, _classification_signature(text, title, result)
    ):
        return False

    expected_segments = segment_article(text)
    if len(expected_segments) != len(result.items):
        return False
    return all(
        actual.id == expected.id
        and actual.text == expected.text
        and actual.start_offset == expected.start_offset
        and actual.end_offset == expected.end_offset
        for actual, expected in zip(result.items, expected_segments)
    )


def cache_article_classification(
    text: str, title: str, result: FactOpinionResult
) -> None:
    """Keep a small process-local set of trusted, server-generated results."""
    key = _fact_opinion_cache_key(text, title)
    if key not in fact_opinion_cache:
        fact_opinion_cache_order.append(key)
    fact_opinion_cache[key] = result.model_copy(deep=True)
    while len(fact_opinion_cache_order) > FACT_OPINION_CACHE_SIZE:
        expired = fact_opinion_cache_order.popleft()
        fact_opinion_cache.pop(expired, None)


def cached_article_classification(
    text: str, title: str
) -> FactOpinionResult | None:
    """Return a copy of a trusted prior result for this exact request."""
    cached = fact_opinion_cache.get(_fact_opinion_cache_key(text, title))
    return cached.model_copy(deep=True) if cached is not None else None


def _mask_opinion_excerpts(text: str, excerpts: list[str]) -> str:
    """Replace opinion characters with spaces while preserving offsets/newlines."""
    characters = list(text)
    occupied: list[tuple[int, int]] = []
    for excerpt in excerpts:
        search_from = 0
        selected: tuple[int, int] | None = None
        while True:
            start = text.find(excerpt, search_from)
            if start < 0:
                break
            candidate = (start, start + len(excerpt))
            if all(candidate[1] <= old[0] or candidate[0] >= old[1] for old in occupied):
                selected = candidate
                break
            search_from = start + 1
        if selected is None:
            continue
        occupied.append(selected)
        for index in range(*selected):
            if characters[index] != "\n":
                characters[index] = " "
    return "".join(characters)


def _remove_opinion_excerpts(text: str, excerpts: list[str]) -> str:
    cleaned = _mask_opinion_excerpts(text, excerpts)
    return re.sub(r"[^\S\n]+", " ", cleaned).strip()


def _line_and_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    last_newline = text.rfind("\n", 0, offset)
    return line, offset - last_newline


def build_factual_content(
    result: FactOpinionResult,
    article_text: str,
    max_chars: int = MAX_ANALYSIS_INPUT_CHARS,
    *,
    retain_mixed_opinion: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Build routed model input while retaining source quote attribution."""
    original_quotes = extract_quoted_phrases(article_text)
    parts: list[str] = []
    derived_quotes: list[dict[str, Any]] = []
    output_length = 0

    for item in result.items:
        final = item.final_prediction
        if final.status != "resolved" or final.label not in {"fact", "mixed"}:
            continue
        if item.start_offset is None or item.end_offset is None:
            continue

        masked = (
            item.text
            if retain_mixed_opinion and final.label == "mixed"
            else _mask_opinion_excerpts(item.text, final.opinion_excerpts)
        )
        if not masked.strip():
            continue
        separator = "\n" if parts else ""
        available = max_chars - output_length - len(separator)
        if available <= 0:
            break
        emitted = masked[:available]
        emitted_start = output_length + len(separator)
        parts.append(separator + emitted)

        source_limit = item.start_offset + len(emitted)
        for quote in original_quotes:
            overlap_start = max(item.start_offset, int(quote["start_offset"]))
            overlap_end = min(source_limit, int(quote["end_offset"]))
            if overlap_start >= overlap_end:
                continue
            relative_start = overlap_start - item.start_offset
            relative_end = overlap_end - item.start_offset
            quoted_text = emitted[relative_start:relative_end]
            left_trim = len(quoted_text) - len(quoted_text.lstrip())
            right_trim = len(quoted_text.rstrip())
            if right_trim <= left_trim:
                continue
            start_offset = emitted_start + relative_start + left_trim
            end_offset = emitted_start + relative_start + right_trim
            derived_quotes.append(
                {
                    "text": quoted_text[left_trim:right_trim],
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "source_start_offset": overlap_start + left_trim,
                    "source_end_offset": overlap_start + right_trim,
                    "attribution": "external_speaker_or_author",
                }
            )
        output_length += len(separator) + len(emitted)
        if len(emitted) < len(masked):
            break

    factual_text = "".join(parts)
    for quote in derived_quotes:
        start_line, start_column = _line_and_column(
            factual_text, int(quote["start_offset"])
        )
        end_line, end_column = _line_and_column(
            factual_text, int(quote["end_offset"])
        )
        quote.update(
            {
                "start_line": start_line,
                "end_line": end_line,
                "start_column": start_column,
                "end_column": end_column,
            }
        )
    return factual_text, derived_quotes


def build_bias_content(
    result: FactOpinionResult,
    article_text: str,
    max_chars: int = MAX_ANALYSIS_INPUT_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """Return article-authored factual/mixed text, excluding external quotations."""
    bias_text, quoted_spans = build_factual_content(
        result,
        article_text,
        max_chars,
        retain_mixed_opinion=True,
    )
    return _remove_external_quotes(bias_text, quoted_spans), []


def _remove_external_quotes(
    text: str, quoted_spans: list[dict[str, Any]]
) -> str:
    """Remove exact externally attributed quote ranges while preserving article flow."""
    ranges: list[tuple[int, int]] = []
    for span in quoted_spans:
        try:
            start = max(0, int(span["start_offset"]))
            end = min(len(text), int(span["end_offset"]))
        except (KeyError, TypeError, ValueError):
            continue
        if start >= end:
            continue
        # Quote spans describe their content, so also consume adjoining delimiters.
        if start > 0 and text[start - 1] in QUOTE_MARKS:
            start -= 1
        if end < len(text) and text[end] in QUOTE_MARKS:
            end += 1
        ranges.append((start, end))

    if not ranges:
        return text.strip()

    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    # Do not collapse line breaks: they remain useful for later highlight offsets.
    without_quotes = "".join(pieces)
    without_quotes = re.sub(r"[ \t]{2,}", " ", without_quotes)
    without_quotes = re.sub(r"[ \t]*\n[ \t]*", "\n", without_quotes)
    without_quotes = re.sub(r"\s+([,.;:!?])", r"\1", without_quotes)
    return without_quotes.strip()


def build_factual_text(
    result: FactOpinionResult, max_chars: int = MAX_ANALYSIS_INPUT_CHARS
) -> str:
    """Return only resolved factual content, with mixed opinion wording removed."""
    factual_segments: list[str] = []
    used_chars = 0
    for item in result.items:
        final = item.final_prediction
        if final.status != "resolved" or final.label not in {"fact", "mixed"}:
            continue
        cleaned = _remove_opinion_excerpts(item.text, final.opinion_excerpts)
        if not cleaned:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        factual_segments.append(cleaned[:remaining])
        used_chars += min(len(cleaned), remaining) + 1
    return "\n".join(factual_segments).strip()[:max_chars]


class BiasHighlightReason(BaseModel):
    """Phrase-specific reason for a single bias highlight."""
    phrase: str = Field(..., min_length=1, max_length=140)
    reason: str = Field(..., min_length=180, max_length=420)


class AIresultBias(BaseModel):
    """Normalized bias-analysis response returned to the frontend."""
    bias_score: int = Field(..., ge=0, le=10)
    summary: str = Field(..., max_length=700)
    highlights: list[str] = Field(default_factory=list, max_length=8)
    highlight_reasons: list[BiasHighlightReason] = Field(default_factory=list, max_length=8)
    explanation: str = Field(..., min_length=260, max_length=520)
    missing_perspectives: str = Field(..., min_length=240, max_length=520)

    @model_validator(mode="after")
    def validate_highlight_reasons(self):
        highlights = [highlight.strip() for highlight in self.highlights]
        reason_phrases = [item.phrase.strip() for item in self.highlight_reasons]
        if reason_phrases != highlights:
            raise ValueError("highlight_reasons must match highlights exactly and in order.")
        return self


class ResearchSource(BaseModel):
    """Single citation/source entry for claim verification."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=240)
    url: HttpUrl = Field(..., max_length=2048)
    source_type: Literal[
        "primary", "official", "reputable_secondary", "other"
    ]
    relevance_summary: str = Field(..., min_length=20, max_length=400)


class ResearchClaim(BaseModel):
    """One fact-checkable claim plus verdict and supporting sources."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(..., min_length=1, max_length=700)
    verdict: Literal["supported", "contradicted", "unclear"]
    evidence_summary: str = Field(..., min_length=20, max_length=700)
    sources: list[ResearchSource] = Field(..., max_length=3)

    @model_validator(mode="after")
    def validate_decisive_verdict_sources(self):
        if self.verdict in {"supported", "contradicted"} and not self.sources:
            raise ValueError("Supported and contradicted claims require sources.")
        if self.verdict in {"supported", "contradicted"} and not any(
            source.source_type != "other" for source in self.sources
        ):
            raise ValueError(
                "A decisive verdict requires at least one authoritative or reputable source."
            )
        return self


class ResearchCoverage(BaseModel):
    """Server-derived disclosure of the bounded claim-verification scope."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "partial", "none"]
    candidate_claim_count: int = Field(..., ge=0)
    checked_claim_count: int = Field(..., ge=0)
    unchecked_claim_count: int = Field(..., ge=0)
    input_characters: int = Field(..., ge=0)
    total_factual_characters: int = Field(..., ge=0)
    input_truncated: bool
    scope_note: str = Field(..., min_length=20, max_length=700)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.checked_claim_count + self.unchecked_claim_count != self.candidate_claim_count:
            raise ValueError("Research coverage counts are inconsistent.")
        if self.input_characters > self.total_factual_characters:
            raise ValueError("Research input cannot exceed the available factual text.")
        if self.status == "none" and self.candidate_claim_count != 0:
            raise ValueError("Only zero-candidate research can use status 'none'.")
        if self.status == "complete" and (
            self.unchecked_claim_count or self.input_truncated
        ):
            raise ValueError("Complete research cannot omit claims or truncate input.")
        return self


class AIresultResearch(BaseModel):
    """Normalized research/cross-check response."""

    model_config = ConfigDict(extra="forbid")

    claims: list[ResearchClaim] = Field(..., max_length=3)
    overall_reliability: Literal["high", "medium", "low", "not_assessed"]
    notes: str = Field(..., min_length=10, max_length=1000)
    coverage: ResearchCoverage

    @model_validator(mode="after")
    def validate_research_result(self):
        if self.coverage.checked_claim_count != len(self.claims):
            raise ValueError("checked_claim_count must match the returned claims.")
        if not self.claims and self.overall_reliability != "not_assessed":
            raise ValueError("No-claim research must use reliability 'not_assessed'.")
        if self.claims and self.overall_reliability == "not_assessed":
            raise ValueError("Researched claims require an assessed reliability.")
        return self


class ResearchRequest(AnalyzeRequest):
    """Research input may reuse prior classification and bias analysis."""

    fact_opinion: FactOpinionResult | None = None
    bias_result: AIresultBias | None = None


class URLRequest(BaseModel):
    """Input payload for URL extraction endpoint."""
    url: HttpUrl = Field(..., max_length=2048)


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


def cap_extracted_text(text: str) -> str:
    """Limit text returned to clients to what downstream analysis can use."""
    clean_text = str(text or "").strip()
    if len(clean_text) > MAX_EXTRACTED_TEXT_CHARS:
        return clean_text[:MAX_EXTRACTED_TEXT_CHARS]
    return clean_text


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
    tool_choice: str | dict[str, Any] | None = None,
    include: list[str] | None = None,
    temperature: float | None = None,
    model: str | None = None,
    reasoning: dict[str, Any] | None = None,
    store: bool | None = None,
    timeout: float | None = None,
) -> Any:
    """Send a prompt + JSON schema request and return the raw model response."""
    request_args = {
        "model": model or MODEL_NAME,
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
    if tool_choice is not None:
        request_args["tool_choice"] = tool_choice
    if include:
        request_args["include"] = include
    if temperature is not None:
        request_args["temperature"] = temperature
    if reasoning is not None:
        request_args["reasoning"] = reasoning
    if store is not None:
        request_args["store"] = store
    if timeout is not None:
        request_args["timeout"] = timeout

    return await client.responses.create(**request_args)


async def analyze_bias(
    text: str,
    title: str = "Article Analysis",
    quoted_spans: list[dict[str, Any]] | None = None,
    *,
    speaker_spans: list[dict[str, Any]] | None = None,
    source_kind: Literal["article", "podcast"] = "article",
) -> dict:
    """Run the bias prompt and return parsed JSON (or {error})."""
    external_quotes = (
        quoted_spans if quoted_spans is not None else extract_quoted_phrases(text)
    )
    bias_text = (
        _remove_external_quotes(text, external_quotes)
        if source_kind == "article"
        else text
    )
    payload = {
        "task": "bias_analysis",
        "title": title,
        "article_text": bias_text,
        "source_kind": source_kind,
        # Quotes are useful provenance for research, but are intentionally not
        # available to article bias scoring.
        "quoted_spans": [] if source_kind == "article" else external_quotes,
        "speaker_spans": speaker_spans or [],
        "return": "valid JSON only",
    }
    try:
        response = await asyncio.wait_for(
            run_model_json(
                prompt=bias_detector_prompt,
                payload=payload,
                schema_name="bias_result",
                schema=bias_schema[0]["parameters"],
                max_tokens=1800,
                temperature=0.2,
                timeout=OPENAI_BIAS_TIMEOUT_SECONDS,
            ),
            timeout=OPENAI_BIAS_TIMEOUT_SECONDS,
        )
        return parse_model_json(response)
    except Exception as exc:
        error_id = hashlib.sha256(
            f"{time.time_ns()}:{type(exc).__name__}".encode("utf-8")
        ).hexdigest()[:12]
        timed_out = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
        error_code = "bias_timeout" if timed_out else "bias_model_failure"
        logger.error(
            "Bias request failed; error_id=%s error_code=%s error_type=%s",
            error_id,
            error_code,
            type(exc).__name__,
        )
        return {
            "error": (
                "Bias analysis timed out. Please try again."
                if timed_out
                else "Bias analysis failed. Please try again."
            ),
            "error_code": error_code,
            "error_id": error_id,
        }


class MissingWebSearchError(ValueError):
    """Raised when research output is not grounded in its required web call."""


class ResearchCitationProvenanceError(ValueError):
    """Raised when a citation was not returned by the response's web search."""


def _quoted_spans_within_prefix(
    quoted_spans: list[dict[str, Any]] | None,
    prefix_text: str,
) -> list[dict[str, Any]]:
    """Clip supplied quote metadata to the exact research input prefix."""
    if quoted_spans is None:
        return extract_quoted_phrases(prefix_text)

    clipped: list[dict[str, Any]] = []
    for raw_span in quoted_spans:
        try:
            start = max(0, int(raw_span["start_offset"]))
            end = min(len(prefix_text), int(raw_span["end_offset"]))
        except (KeyError, TypeError, ValueError):
            continue
        if start >= end:
            continue
        span = dict(raw_span)
        span["start_offset"] = start
        span["end_offset"] = end
        span["text"] = prefix_text[start:end]
        start_line, start_column = _line_and_column(prefix_text, start)
        end_line, end_column = _line_and_column(prefix_text, end)
        span.update(
            {
                "start_line": start_line,
                "end_line": end_line,
                "start_column": start_column,
                "end_column": end_column,
                "attribution": "external_speaker_or_author",
            }
        )
        clipped.append(span)
    return sorted(clipped, key=lambda span: span["start_offset"])


def _normalize_provenance_url(value: Any) -> str | None:
    """Canonicalize an HTTP(S) URL for web-result provenance comparison."""
    try:
        parsed = urlsplit(str(value).strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    hostname = parsed.hostname.lower()
    port = parsed.port
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), hostname, path, parsed.query, ""))


def _web_search_provenance_diagnostics(response: Any) -> dict[str, int]:
    """Return safe, payload-free counts for web-search provenance diagnostics."""
    completed_searches = 0
    action_sources = 0
    annotations = 0
    for item in _get(response, "output", []) or []:
        if _get(item, "type") == "web_search_call" and _get(item, "status") == "completed":
            completed_searches += 1
            action_sources += len(_get(_get(item, "action", {}), "sources", []) or [])
        if _get(item, "type") == "message":
            for content in _get(item, "content", []) or []:
                annotations += sum(
                    _get(annotation, "type") == "url_citation"
                    for annotation in (_get(content, "annotations", []) or [])
                )
    return {
        "completed_searches": completed_searches,
        "action_sources": action_sources,
        "url_citation_annotations": annotations,
    }


def _has_completed_web_search(response: Any) -> bool:
    """Return whether this response completed the required web-search tool call."""
    return _web_search_provenance_diagnostics(response)["completed_searches"] > 0


def _completed_web_search_urls(response: Any) -> set[str]:
    """Collect provenance URLs from completed web searches and their citations."""
    source_urls: set[str] = set()
    for item in _get(response, "output", []) or []:
        if _get(item, "type") != "web_search_call":
            continue
        if _get(item, "status") != "completed":
            continue
        action = _get(item, "action", {})
        for source in _get(action, "sources", []) or []:
            source_url = _normalize_provenance_url(_get(source, "url"))
            if source_url:
                source_urls.add(source_url)

    if not _has_completed_web_search(response):
        return set()

    annotation_urls: set[str] = set()
    for item in _get(response, "output", []) or []:
        if _get(item, "type") != "message":
            continue
        for content in _get(item, "content", []) or []:
            for annotation in _get(content, "annotations", []) or []:
                if _get(annotation, "type") != "url_citation":
                    continue
                citation_url = _normalize_provenance_url(_get(annotation, "url"))
                if citation_url:
                    annotation_urls.add(citation_url)
    return source_urls | annotation_urls


def _validate_research_url_provenance(
    parsed: dict[str, Any], searched_urls: set[str]
) -> None:
    """Reject citations that were not present in this response's web results."""
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        raise ValueError("Research response claims are malformed.")
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("sources"), list):
            raise ValueError("Research response sources are malformed.")
        for source in claim["sources"]:
            if not isinstance(source, dict):
                raise ValueError("Research response source is malformed.")
            normalized = _normalize_provenance_url(source.get("url"))
            if not normalized or normalized not in searched_urls:
                raise ResearchCitationProvenanceError(
                    "A cited source was absent from the completed web search."
                )


async def researcher_ai(
    text: str,
    title: str = "Article Analysis",
    quoted_spans: list[dict[str, Any]] | None = None,
    bias_result: AIresultBias | dict[str, Any] | None = None,
    candidate_claim_count: int | None = None,
    article_input_truncated: bool = False,
    speaker_spans: list[dict[str, Any]] | None = None,
    source_kind: Literal["article", "podcast"] = "article",
) -> dict:
    """Run source-backed research with retry on token-limit truncation."""
    condensed_text = text.strip()[:MAX_RESEARCH_INPUT_CHARS]
    if isinstance(bias_result, BaseModel):
        serialized_bias = bias_result.model_dump(mode="json")
    elif isinstance(bias_result, dict):
        serialized_bias = bias_result
    else:
        serialized_bias = {
            "status": "unavailable",
            "note": "No prior bias analysis was supplied for prioritization.",
        }
    payload = {
        "source_url": "",
        "source_kind": source_kind,
        "title": title,
        "content_text": condensed_text,
        "quoted_spans": _quoted_spans_within_prefix(
            quoted_spans, condensed_text
        ),
        "speaker_spans": _quoted_spans_within_prefix(
            speaker_spans, condensed_text
        ) if speaker_spans else [],
        "bias_detector_output": serialized_bias,
        "research_scope": {
            "candidate_claim_count": candidate_claim_count,
            "maximum_claims_to_check": 3,
            "input_characters": len(condensed_text),
            "total_factual_characters": len(text.strip()),
            "input_truncated": (
                len(text.strip()) > len(condensed_text) or article_input_truncated
            ),
        },
        "return": "valid JSON only",
    }

    try:
        last_error: Exception | None = None
        deadline = (
            asyncio.get_running_loop().time()
            + OPENAI_RESEARCH_TIMEOUT_SECONDS
        )
        for max_tokens in (1800, 2600):
            # Retry once with more output tokens if the model truncates.
            remaining_seconds = deadline - asyncio.get_running_loop().time()
            if remaining_seconds <= 0:
                raise TimeoutError("Research exceeded its time limit.")
            response = await asyncio.wait_for(
                run_model_json(
                    prompt=researcher_prompt,
                    payload=payload,
                    schema_name="research_schema",
                    schema=research_schema[0],
                    max_tokens=max_tokens,
                    tools=[{"type": "web_search"}],
                    tool_choice={"type": "web_search"},
                    include=["web_search_call.action.sources"],
                    model=RESEARCH_API_MODEL,
                    timeout=remaining_seconds,
                ),
                timeout=remaining_seconds,
            )
            try:
                parsed = parse_model_json(response)
                # Real Responses SDK objects always expose output. Keeping the
                # guard conditional permits small unit-test response doubles.
                if _get(response, "output") is not None:
                    diagnostics = _web_search_provenance_diagnostics(response)
                    if not _has_completed_web_search(response):
                        logger.warning(
                            "Research provenance rejected; category=no_completed_search "
                            "completed_searches=%d action_sources=%d url_citation_annotations=%d",
                            diagnostics["completed_searches"],
                            diagnostics["action_sources"],
                            diagnostics["url_citation_annotations"],
                        )
                        raise MissingWebSearchError(
                            "Research response did not complete a web search."
                        )
                    searched_urls = _completed_web_search_urls(response)
                    try:
                        _validate_research_url_provenance(parsed, searched_urls)
                    except ResearchCitationProvenanceError:
                        logger.warning(
                            "Research provenance rejected; category=unverified_citation "
                            "completed_searches=%d action_sources=%d url_citation_annotations=%d canonical_urls=%d",
                            diagnostics["completed_searches"],
                            diagnostics["action_sources"],
                            diagnostics["url_citation_annotations"],
                            len(searched_urls),
                        )
                        raise
                return parsed
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
        error_id = hashlib.sha256(
            f"{time.time_ns()}:{type(exc).__name__}".encode("utf-8")
        ).hexdigest()[:12]
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            error_code = "research_timeout"
        elif isinstance(exc, MissingWebSearchError):
            error_code = "research_no_web_search"
        elif isinstance(exc, ResearchCitationProvenanceError):
            error_code = "research_unverified_citation"
        elif isinstance(exc, (ValueError, TypeError)):
            error_code = "research_invalid_response"
        else:
            error_code = "research_model_failure"
        logger.error(
            "Research request failed; error_id=%s error_code=%s error_type=%s",
            error_id,
            error_code,
            type(exc).__name__,
        )
        return {
            "error": (
                "Research timed out. Please try again."
                if error_code == "research_timeout"
                else "Research verification failed. Please try again."
            ),
            "error_code": error_code,
            "error_id": error_id,
        }


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
            follow_redirects=False,
            # Split timeouts so slow servers fail fast and we can try Playwright.
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT_SECONDS,
                read=HTTP_READ_TIMEOUT_SECONDS,
                write=10.0,
                pool=10.0,
            ),
            headers={"User-Agent": "Mozilla/5.0"},
        ) as http:
            response, html = await fetch_html_with_httpx_redirects(http, url)
    except httpx.ReadTimeout:
        return "", "Direct fetch timed out while reading the page."
    except httpx.ConnectTimeout:
        return "", "Direct fetch timed out while connecting to the site."
    except HTTPException as exc:
        return "", str(exc.detail)
    except ValueError as exc:
        return "", str(exc)
    except httpx.HTTPError as exc:
        return "", f"Network error fetching URL: {exc}"

    text = extract_readable_text(html)
    # Convert fetch outcomes into one reason string so the caller can report it cleanly.
    if response.status_code >= 400:
        return "", f"Fetch failed with status {response.status_code}."
    if looks_like_bot_block(html):
        return "", "Direct fetch looked blocked by bot protection."
    if len(text) < MIN_EXTRACT_CHARS:
        return "", f"Direct fetch returned too little readable text ({len(text)} chars)."
    return cap_extracted_text(text), ""


async def read_limited_response_bytes(response: httpx.Response) -> bytes:
    """Read a response body with a hard byte limit."""
    chunks: list[bytes] = []
    total_bytes = 0
    async for chunk in response.aiter_bytes():
        total_bytes += len(chunk)
        if total_bytes > MAX_FETCH_BYTES:
            raise ValueError("Fetched page is too large.")
        chunks.append(chunk)
    return b"".join(chunks)


async def fetch_html_with_httpx_redirects(
    http: httpx.AsyncClient,
    url: str,
) -> tuple[httpx.Response, str]:
    """Fetch HTML while validating every redirect target before following it."""
    current_url = str(url)
    for _ in range(MAX_REDIRECTS + 1):
        await validate_public_url(current_url)
        async with http.stream("GET", current_url) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("Redirect response did not include a Location header.")
                current_url = urljoin(str(response.url), location)
                continue

            body = await read_limited_response_bytes(response)
            encoding = response.encoding or "utf-8"
            html = body.decode(encoding, errors="replace")
            return response, html

    raise ValueError("Too many redirects while fetching URL.")


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
        await context.route("**/*", route_public_network_requests)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT_MS)
            # Brief wait gives client-side rendered article content time to appear.
            await page.wait_for_timeout(1200)
            html = await page.content()
            if len(html.encode("utf-8")) > MAX_FETCH_BYTES:
                raise RuntimeError("Rendered page is too large.")
            return html
        finally:
            await context.close()
            await browser.close()


async def route_public_network_requests(route, request) -> None:
    """Block Playwright document/subresource requests to private networks."""
    try:
        await validate_public_url(request.url)
    except Exception:
        await route.abort()
        return
    await route.continue_()


def validate_ai_bias(ai: dict, source_text: str | None = None) -> AIresultBias:
    """Validate and normalize bias JSON before sending to frontend."""
    try:
        result = AIresultBias(**ai)
        if source_text is not None and any(
            phrase not in source_text for phrase in result.highlights
        ):
            raise ValueError("Bias highlights must be exact substrings of the analyzed text.")
        return result
    except Exception as exc:
        raise HTTPException(status_code=502, detail="AI bias response malformed.") from exc


def model_error_detail(payload: dict, fallback: str) -> dict[str, str]:
    """Expose safe diagnostics while keeping provider exception text server-side."""
    detail = {"message": str(payload.get("error") or fallback)}
    if payload.get("error_code"):
        detail["code"] = str(payload["error_code"])
    if payload.get("error_id"):
        detail["reference"] = str(payload["error_id"])
    return detail


def _research_candidate_count(result: FactOpinionResult) -> int:
    """Count resolved factual segments that contain researchable text."""
    return sum(
        1
        for item in result.items
        if item.final_prediction.status == "resolved"
        and item.final_prediction.label in {"fact", "mixed"}
        and _remove_opinion_excerpts(
            item.text, item.final_prediction.opinion_excerpts
        )
    )


def _research_coverage(
    *,
    candidate_claim_count: int,
    checked_claim_count: int,
    total_factual_characters: int,
    article_input_truncated: bool,
) -> ResearchCoverage:
    """Build conservative, server-controlled disclosure for sampled research."""
    if checked_claim_count > candidate_claim_count:
        raise ValueError("Research returned more claims than candidate factual segments.")
    input_characters = min(total_factual_characters, MAX_RESEARCH_INPUT_CHARS)
    input_truncated = (
        total_factual_characters > input_characters or article_input_truncated
    )
    unchecked = candidate_claim_count - checked_claim_count
    if candidate_claim_count == 0:
        status: Literal["complete", "partial", "none"] = "none"
        if article_input_truncated:
            scope_note = (
                "No resolved factual segments were available in the classified analysis "
                "window, so no external verification was performed. Additional article "
                "content fell outside that window and was not assessed."
            )
        else:
            scope_note = (
                "No resolved factual segments were available, so no external claim "
                "verification was performed."
            )
    elif unchecked or input_truncated:
        status = "partial"
        scope_note = (
            f"Checked {checked_claim_count} of {candidate_claim_count} resolved factual "
            "segments. Counts apply to the classified analysis window; the result does "
            "not imply that every factual assertion in the full article was verified."
        )
    else:
        status = "complete"
        scope_note = (
            f"Checked one claim from each of {candidate_claim_count} resolved factual "
            "segments in the classified analysis window. This is segment-level coverage, "
            "not a guarantee that every embedded assertion was verified."
        )
    return ResearchCoverage(
        status=status,
        candidate_claim_count=candidate_claim_count,
        checked_claim_count=checked_claim_count,
        unchecked_claim_count=unchecked,
        input_characters=input_characters,
        total_factual_characters=total_factual_characters,
        input_truncated=input_truncated,
        scope_note=scope_note,
    )


def validate_ai_research(
    ai: dict,
    *,
    candidate_claim_count: int | None = None,
    total_factual_characters: int = 0,
    article_input_truncated: bool = False,
) -> AIresultResearch:
    """Strictly validate research JSON and attach server-derived coverage."""
    try:
        payload = dict(ai)
        raw_claims = payload.get("claims")
        if not isinstance(raw_claims, list):
            raise ValueError("Research claims must be a list.")
        checked_claim_count = len(raw_claims)
        if checked_claim_count == 0:
            payload["overall_reliability"] = "not_assessed"
        candidates = (
            checked_claim_count
            if candidate_claim_count is None
            else candidate_claim_count
        )
        payload["coverage"] = _research_coverage(
            candidate_claim_count=candidates,
            checked_claim_count=checked_claim_count,
            total_factual_characters=total_factual_characters,
            article_input_truncated=article_input_truncated,
        ).model_dump(mode="json")
        return AIresultResearch(**payload)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="AI research response malformed."
        ) from exc


def no_factual_bias_result() -> AIresultBias:
    """Return a valid non-score when every segment was excluded."""
    return AIresultBias(
        bias_score=0,
        summary=(
            "No resolved factual statements were available for bias analysis. "
            "Opinion and unresolved statements were intentionally excluded."
        ),
        highlights=[],
        highlight_reasons=[],
        explanation=(
            "- The pipeline found no resolved factual language that could be evaluated without relying on opinion.\n"
            "- Opinion statements were labeled for the reader but excluded under the selected analysis policy.\n"
            "- Unresolved statements were also withheld so a weak classifier guess could not affect the bias result."
        ),
        missing_perspectives=(
            "- Add concrete, externally verifiable claims before drawing conclusions about factual framing.\n"
            "- Include attributable evidence, dates, quantities, or primary-source statements that can be checked.\n"
            "- Clarify any unresolved passages so future analysis can separate factual assertions from personal judgment."
        ),
    )


def no_factual_research_result(
    article_input_truncated: bool = False,
) -> AIresultResearch:
    """Return a valid research response when there are no resolved facts."""
    return AIresultResearch(
        claims=[],
        overall_reliability="not_assessed",
        notes=(
            "No resolved factual claims were available for research. Opinion and "
            "unresolved statements were intentionally excluded."
        ),
        coverage=_research_coverage(
            candidate_claim_count=0,
            checked_claim_count=0,
            total_factual_characters=0,
            article_input_truncated=article_input_truncated,
        ),
    )


async def ensure_article_classification(
    text: str,
    title: str,
    supplied: FactOpinionResult | None = None,
) -> FactOpinionResult:
    """Reuse a cached or authenticated server result; otherwise reclassify."""
    cached = cached_article_classification(text, title)
    if cached is not None:
        return cached

    if supplied is not None and article_classification_is_authentic(
        text, title, supplied
    ):
        cache_article_classification(text, title, supplied)
        return supplied.model_copy(deep=True)

    return await classify_article_fact_opinion(text, title)


class ArticleJobFailure(Exception):
    """Expected article-job failure with a client-safe message."""

    def __init__(
        self,
        message: str,
        code: str = "article_job_failed",
        reference: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.reference = reference


def _article_content_key(request: ArticleJobRequest, client_id: str = "") -> str:
    """Make retries idempotent without suppressing intentional re-analysis."""
    if request.client_request_id:
        identity = f"request\0{client_id}\0{request.client_request_id}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    # Backward compatibility for callers deployed before client_request_id.
    # New extension runs always send an ID, so a fresh click creates a fresh job.
    title = " ".join(request.title.split()).casefold()
    supplied_text = (
        str(request.text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if supplied_text:
        identity = f"text\0{client_id}\0{title}\0{supplied_text}"
    else:
        identity = f"url\0{client_id}\0{title}\0{_normalized_podcast_url(str(request.page_url))}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _cleanup_article_jobs() -> None:
    """Expire terminal in-process article jobs without touching active tasks."""
    now = time.time()
    expired = [
        job_id
        for job_id, job in article_jobs.items()
        if job.get("status") in {"complete", "failed"}
        and now - float(job.get("updated_at", now)) > ARTICLE_JOB_TTL_SECONDS
    ]
    for job_id in expired:
        job = article_jobs.pop(job_id, None)
        if job:
            content_key = str(job.get("content_key", ""))
            if article_jobs_by_content.get(content_key) == job_id:
                article_jobs_by_content.pop(content_key, None)
        article_job_tasks.pop(job_id, None)


def _update_article_job(job_id: str, **patch: Any) -> None:
    job = article_jobs.get(job_id)
    if job is None:
        return
    job.update(patch)
    job["updated_at"] = time.time()


async def _article_job_text(request: ArticleJobRequest) -> str:
    """Use supplied page text or perform both extraction fallbacks in the job."""
    supplied = normalize_analysis_text(request.text)
    if len(supplied) >= MIN_EXTRACT_CHARS:
        return supplied

    page_url = str(request.page_url)
    await validate_public_url(page_url)
    extracted, direct_reason = await extract_text_with_httpx(page_url)
    if extracted:
        return normalize_analysis_text(extracted)
    try:
        rendered_html = await fetch_html_with_playwright(page_url)
        rendered_text = cap_extracted_text(extract_readable_text(rendered_html))
    except Exception as exc:
        logger.info(
            "Article rendered extraction failed; error_type=%s", type(exc).__name__
        )
        rendered_text = ""
    if len(rendered_text) < MIN_EXTRACT_CHARS:
        logger.info("Article extraction exhausted; direct_reason=%s", direct_reason)
        raise ArticleJobFailure(
            "The article text could not be extracted. Open the article fully and try again.",
            "article_extraction_failed",
        )
    return normalize_analysis_text(rendered_text)


async def run_article_job(job_id: str, request: ArticleJobRequest) -> None:
    """Own article extraction, classification, bias, and research server-side."""
    partial_result: dict[str, Any] | None = None

    def stage(message: str, progress: int, **patch: Any) -> None:
        _update_article_job(
            job_id,
            status="running",
            stage=message,
            progress=max(0, min(99, int(progress))),
            **patch,
        )

    try:
        stage("Extracting article text...", 5)
        text = await _article_job_text(request)
        if len(text) < MIN_EXTRACT_CHARS:
            raise ArticleJobFailure(
                "Not enough article text was available to analyze.",
                "article_text_too_short",
            )

        stage("Classifying article passages...", 20)
        try:
            fact_opinion = await classify_article_fact_opinion(text, request.title)
        except Exception as exc:
            raise ArticleJobFailure(
                "The fact-opinion classifier is unavailable. Please try again.",
                "article_classification_failed",
            ) from exc

        partial_result = {
            "status": "partial",
            "ai_result": None,
            "ai_research": None,
            "fact_opinion": fact_opinion.model_dump(mode="json"),
        }
        stage("Analyzing article bias...", 45, result=partial_result)
        factual_text, factual_quotes = build_factual_content(fact_opinion, text)
        bias_text, bias_quotes = build_bias_content(fact_opinion, text)
        candidate_claim_count = _research_candidate_count(fact_opinion)
        article_input_truncated = len(str(request.text or "").strip()) > len(text)
        if bias_text:
            bias_raw = await analyze_bias(bias_text, request.title, bias_quotes)
            if "error" in bias_raw:
                detail = model_error_detail(bias_raw, "Bias analysis failed.")
                raise ArticleJobFailure(
                    detail["message"],
                    detail.get("code", "article_bias_failed"),
                    detail.get("reference"),
                )
            bias_result = validate_ai_bias(bias_raw, bias_text)
        else:
            bias_result = no_factual_bias_result()

        partial_result["ai_result"] = bias_result.model_dump(mode="json")
        stage("Researching factual claims...", 70, result=partial_result)
        if factual_text:
            research_raw = await researcher_ai(
                factual_text,
                request.title,
                factual_quotes,
                bias_result=bias_result,
                candidate_claim_count=candidate_claim_count,
                article_input_truncated=article_input_truncated,
            )
            if "error" in research_raw:
                detail = model_error_detail(research_raw, "Research verification failed.")
                raise ArticleJobFailure(
                    detail["message"],
                    detail.get("code", "article_research_failed"),
                    detail.get("reference"),
                )
            research_result = validate_ai_research(
                research_raw,
                candidate_claim_count=candidate_claim_count,
                total_factual_characters=len(factual_text),
                article_input_truncated=article_input_truncated,
            )
        else:
            research_result = no_factual_research_result(article_input_truncated)

        result = {
            **partial_result,
            "status": "analyzed",
            "ai_research": research_result.model_dump(mode="json"),
        }
        _update_article_job(
            job_id,
            status="complete",
            stage="Article analysis complete.",
            progress=100,
            result=result,
            error=None,
            completed_at=time.time(),
        )
    except Exception as exc:
        error_id = hashlib.sha256(
            f"{time.time_ns()}:{type(exc).__name__}".encode("utf-8")
        ).hexdigest()[:12]
        logger.exception(
            "Article job failed; job_id=%s error_id=%s error_type=%s",
            job_id,
            error_id,
            type(exc).__name__,
        )
        if isinstance(exc, ArticleJobFailure):
            safe_message = exc.message
            error_code = exc.code
            reference = exc.reference or error_id
        else:
            safe_message = "Article analysis failed. Please try again."
            error_code = "article_job_failed"
            reference = error_id
        _update_article_job(
            job_id,
            status="failed",
            stage="Article analysis failed.",
            progress=100,
            result=partial_result,
            error={
                "message": safe_message[:300],
                "code": error_code,
                "reference": reference,
            },
            completed_at=time.time(),
        )
    finally:
        article_job_tasks.pop(job_id, None)


def _normalized_podcast_url(url: str) -> str:
    parsed = urlsplit(str(url).strip())
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.lower(), (parsed.netloc or "").lower(), path, parsed.query, "")
    )


def _cleanup_podcast_jobs() -> None:
    """Expire completed in-process jobs without touching active tasks."""
    now = time.time()
    expired = [
        job_id
        for job_id, job in podcast_jobs.items()
        if job.get("status") in {"complete", "failed"}
        and now - float(job.get("updated_at", now)) > PODCAST_JOB_TTL_SECONDS
    ]
    for job_id in expired:
        job = podcast_jobs.pop(job_id, None)
        if job:
            podcast_jobs_by_url.pop(str(job.get("url_key", "")), None)
        podcast_job_tasks.pop(job_id, None)


def _update_podcast_job(job_id: str, **patch: Any) -> None:
    job = podcast_jobs.get(job_id)
    if job is None:
        return
    job.update(patch)
    job["updated_at"] = time.time()


def _decode_podcast_text(body: bytes | None, content_type: str) -> str:
    raw = body or b""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


async def _publisher_transcript_from_url(
    url: str,
    *,
    mime_type: str | None,
    title: str,
    page_url: str,
    source: Literal["rss_transcript", "page_transcript"],
    language: str | None = None,
) -> PodcastTranscript | None:
    try:
        body, response_type, _ = await fetch_public_bytes(
            url,
            validate_url=validate_public_url,
            max_bytes=MAX_PODCAST_TRANSCRIPT_BYTES,
        )
        selected_type = (
            str(mime_type or "").split(";", 1)[0].lower()
            or response_type
            or "text/plain"
        )
        if selected_type not in TRANSCRIPT_MIME_TYPES:
            return None
        segments = parse_publisher_transcript(
            _decode_podcast_text(body, selected_type), selected_type
        )
        if not segments:
            return None
        return canonicalize_transcript(
            segments,
            title=title,
            page_url=page_url,
            source=source,
            language=language,
        )
    except Exception as exc:
        logger.info(
            "Publisher transcript candidate was unusable; source=%s error_type=%s",
            source,
            type(exc).__name__,
        )
        return None


async def discover_podcast_transcript(
    request: PodcastJobRequest,
    *,
    workdir: Path,
    stage: Callable[[str, int], None] | None = None,
) -> PodcastTranscript:
    """Prefer publisher transcripts, otherwise transcribe one public audio source."""
    page_url = str(request.page_url)
    if stage:
        stage("Finding publisher transcript...", 8)
    page_body, page_type, resolved_page_url = await fetch_public_bytes(
        page_url,
        validate_url=validate_public_url,
        max_bytes=MAX_PODCAST_PAGE_BYTES,
    )
    page_text = _decode_podcast_text(page_body, page_type)
    page_info = inspect_podcast_page(page_text, resolved_page_url, request.hints)

    rss_audio_urls: list[str] = []
    feed_candidates = list(page_info.feed_urls)
    if "xml" in page_type or page_text.lstrip().startswith("<rss"):
        feed_candidates.insert(0, resolved_page_url)
    for feed_url in feed_candidates:
        try:
            feed_body, feed_type, _ = await fetch_public_bytes(
                feed_url,
                validate_url=validate_public_url,
                max_bytes=MAX_PODCAST_TRANSCRIPT_BYTES,
            )
            episode = select_rss_episode(
                _decode_podcast_text(feed_body, feed_type),
                page_url=page_info.canonical_url,
                page_title=page_info.title,
                page_date=page_info.published_date,
            )
        except Exception as exc:
            logger.info("Podcast feed candidate failed; error_type=%s", type(exc).__name__)
            continue
        if episode is None:
            continue
        for transcript_url, mime_type, language in episode.transcript_urls:
            transcript = await _publisher_transcript_from_url(
                transcript_url,
                mime_type=mime_type,
                title=episode.title,
                page_url=resolved_page_url,
                source="rss_transcript",
                language=language or episode.language,
            )
            if transcript is not None:
                return transcript
        if episode.audio_url:
            rss_audio_urls.append(episode.audio_url)

    if page_info.embedded_transcript:
        segments = parse_publisher_transcript(page_info.embedded_transcript, "text/plain")
        if segments:
            return canonicalize_transcript(
                segments,
                title=page_info.title,
                page_url=resolved_page_url,
                source="page_transcript",
            )
    for transcript_url in page_info.transcript_urls:
        transcript = await _publisher_transcript_from_url(
            transcript_url,
            mime_type=None,
            title=page_info.title,
            page_url=resolved_page_url,
            source="page_transcript",
        )
        if transcript is not None:
            return transcript

    audio_candidates = list(dict.fromkeys(rss_audio_urls + list(page_info.audio_urls)))
    if not audio_candidates:
        raise ValueError(
            "No publisher transcript or direct public podcast audio was found on this page."
        )
    if stage:
        stage("Downloading public podcast audio...", 18)
    source_path = workdir / "podcast-source"
    last_error: Exception | None = None
    selected_audio_url = ""
    for audio_url in audio_candidates:
        try:
            await fetch_public_bytes(
                audio_url,
                validate_url=validate_public_url,
                max_bytes=MAX_PODCAST_AUDIO_BYTES,
                timeout_seconds=120.0,
                destination=source_path,
            )
            selected_audio_url = audio_url
            break
        except Exception as exc:
            last_error = exc
            source_path.unlink(missing_ok=True)
    if not selected_audio_url:
        raise ValueError("The discovered podcast audio could not be downloaded.") from last_error

    duration = await probe_duration(source_path)
    if duration > MAX_PODCAST_DURATION_SECONDS:
        raise ValueError("Podcast duration exceeds the configured duration limit.")
    if stage:
        stage("Preparing audio chunks...", 25)
    chunks = await transcode_audio_chunks(
        source_path,
        workdir,
        duration_seconds=duration,
    )
    if stage:
        stage("Transcribing speakers...", 32)
    raw_segments = await transcribe_audio_chunks(
        client,
        chunks,
        reference_dir=workdir,
    )
    return canonicalize_transcript(
        raw_segments,
        title=page_info.title,
        page_url=resolved_page_url,
        source="openai_audio",
        duration_seconds=duration,
    )


def _podcast_windows(transcript: PodcastTranscript) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: list[PodcastSegment] = []
    current_start = 0
    for segment in transcript.segments:
        if not current:
            current_start = segment.start_offset
        proposed_length = segment.end_offset - current_start
        if current and proposed_length > MAX_ANALYSIS_INPUT_CHARS:
            end = current[-1].end_offset
            windows.append(
                {
                    "start": current_start,
                    "end": end,
                    "text": transcript.text[current_start:end],
                    "segments": current,
                }
            )
            current = []
            current_start = segment.start_offset
        if len(segment.text) > MAX_ANALYSIS_INPUT_CHARS:
            for offset in range(0, len(segment.text), MAX_ANALYSIS_INPUT_CHARS):
                piece = segment.text[offset : offset + MAX_ANALYSIS_INPUT_CHARS]
                start = segment.start_offset + offset
                synthetic = segment.model_copy(
                    update={
                        "id": f"{segment.id}-part-{offset // MAX_ANALYSIS_INPUT_CHARS + 1}",
                        "text": piece,
                        "start_offset": start,
                        "end_offset": start + len(piece),
                    }
                )
                windows.append(
                    {
                        "start": start,
                        "end": start + len(piece),
                        "text": piece,
                        "segments": [synthetic],
                    }
                )
            current = []
            continue
        current.append(segment)
    if current:
        end = current[-1].end_offset
        windows.append(
            {
                "start": current_start,
                "end": end,
                "text": transcript.text[current_start:end],
                "segments": current,
            }
        )
    return windows


def _build_podcast_routed_content(
    result: FactOpinionResult,
    source_text: str,
    source_segments: list[PodcastSegment],
    *,
    include_opinions: bool,
    max_chars: int = MAX_ANALYSIS_INPUT_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """Route classified speech and remap speaker/timestamp spans exactly."""
    parts: list[str] = []
    speaker_spans: list[dict[str, Any]] = []
    used = 0
    for item in result.items:
        final = item.final_prediction
        allowed = {"fact", "mixed", "opinion"} if include_opinions else {"fact", "mixed"}
        if final.status != "resolved" or final.label not in allowed:
            continue
        if item.start_offset is None or item.end_offset is None:
            continue
        emitted_source = (
            item.text
            if include_opinions
            else _mask_opinion_excerpts(item.text, final.opinion_excerpts)
        )
        if not emitted_source.strip():
            continue
        separator = "\n" if parts else ""
        available = max_chars - used - len(separator)
        if available <= 0:
            break
        emitted = emitted_source[:available]
        output_start = used + len(separator)
        parts.append(separator + emitted)
        source_limit = item.start_offset + len(emitted)
        for segment in source_segments:
            local_start = segment.start_offset - source_segments[0].start_offset
            local_end = segment.end_offset - source_segments[0].start_offset
            overlap_start = max(item.start_offset, local_start)
            overlap_end = min(source_limit, local_end)
            if overlap_start >= overlap_end:
                continue
            relative_start = overlap_start - item.start_offset
            relative_end = overlap_end - item.start_offset
            fragment = emitted[relative_start:relative_end]
            left = len(fragment) - len(fragment.lstrip())
            right = len(fragment.rstrip())
            if right <= left:
                continue
            span_start = output_start + relative_start + left
            span_end = output_start + relative_start + right
            speaker_spans.append(
                {
                    "segment_id": segment.id,
                    "speaker": segment.speaker,
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "start_offset": span_start,
                    "end_offset": span_end,
                    "source_start_offset": segment.start_offset + (overlap_start - local_start) + left,
                    "source_end_offset": segment.start_offset + (overlap_start - local_start) + right,
                }
            )
        used += len(separator) + len(emitted)
        if len(emitted) < len(emitted_source):
            break
    return "".join(parts), speaker_spans


def _attach_segment_classifications(
    transcript: PodcastTranscript,
    window_start: int,
    result: FactOpinionResult,
) -> None:
    for item in result.items:
        if item.start_offset is None or item.end_offset is None:
            continue
        global_start = window_start + item.start_offset
        global_end = window_start + item.end_offset
        classification = {
            "item_id": item.id,
            "local_prediction": item.local_prediction.model_dump(mode="json"),
            "final_prediction": item.final_prediction.model_dump(mode="json"),
        }
        for segment in transcript.segments:
            overlap = min(global_end, segment.end_offset) - max(global_start, segment.start_offset)
            if overlap > 0:
                segment.classification = classification


def _find_highlight_location(
    phrase: str,
    text: str,
    speaker_spans: list[dict[str, Any]],
    used_offsets: set[int],
) -> dict[str, Any] | None:
    search_from = 0
    while True:
        start = text.find(phrase, search_from)
        if start < 0:
            return None
        search_from = start + 1
        if start in used_offsets:
            continue
        used_offsets.add(start)
        end = start + len(phrase)
        span = next(
            (
                candidate
                for candidate in speaker_spans
                if start >= int(candidate["start_offset"])
                and end <= int(candidate["end_offset"])
            ),
            None,
        )
        if span is None:
            return None
        local_delta = max(0, start - int(span["start_offset"]))
        source_start = int(span["source_start_offset"]) + local_delta
        return {
            "phrase": phrase,
            "segment_id": span["segment_id"],
            "speaker": span["speaker"],
            "start_seconds": span.get("start_seconds"),
            "end_seconds": span.get("end_seconds"),
            "start_offset": source_start,
            "end_offset": source_start + len(phrase),
        }


async def _aggregate_podcast_bias(
    title: str,
    window_results: list[dict[str, Any]],
) -> tuple[AIresultBias, list[dict[str, Any]]]:
    total_weight = sum(max(1, int(item["weight"])) for item in window_results)
    score = round(
        sum(item["bias"].bias_score * max(1, int(item["weight"])) for item in window_results)
        / max(1, total_weight)
    )
    candidates: list[dict[str, Any]] = []
    for window_index, item in enumerate(window_results):
        reasons = {reason.phrase: reason.reason for reason in item["bias"].highlight_reasons}
        used_offsets: set[int] = set()
        for highlight_index, phrase in enumerate(item["bias"].highlights):
            location = _find_highlight_location(
                phrase,
                item["bias_text"],
                item["speaker_spans"],
                used_offsets,
            )
            if location is None:
                continue
            candidates.append(
                {
                    "id": f"window-{window_index + 1}-highlight-{highlight_index + 1}",
                    "phrase": phrase,
                    "reason": reasons.get(phrase, ""),
                    "location": location,
                    "window_bias_score": item["bias"].bias_score,
                }
            )
    ordered_ids = [candidate["id"] for candidate in candidates[:8]]
    if len(window_results) == 1:
        synthesis = {
            "summary": window_results[0]["bias"].summary,
            "selected_highlight_ids": ordered_ids,
            "explanation": window_results[0]["bias"].explanation,
            "missing_perspectives": window_results[0]["bias"].missing_perspectives,
        }
    else:
        payload = {
            "title": title,
            "windows": [
                {
                    "window": index + 1,
                    "summary": item["bias"].summary,
                    "bias_score": item["bias"].bias_score,
                }
                for index, item in enumerate(window_results)
            ],
            "highlight_candidates": [
                {
                    "id": candidate["id"],
                    "phrase": candidate["phrase"],
                    "speaker": candidate["location"]["speaker"],
                    "reason": candidate["reason"],
                }
                for candidate in candidates
            ],
        }
        try:
            response = await run_model_json(
                prompt=podcast_bias_synthesis_prompt,
                payload=payload,
                schema_name="podcast_bias_synthesis",
                schema=podcast_bias_synthesis_schema,
                max_tokens=1800,
                temperature=0.2,
            )
            synthesis = parse_model_json(response)
            requested = synthesis.get("selected_highlight_ids")
            if not isinstance(requested, list) or any(value not in {c["id"] for c in candidates} for value in requested):
                raise ValueError("Podcast synthesis selected an unknown highlight.")
            ordered_ids = list(dict.fromkeys(requested))[:8]
        except Exception as exc:
            logger.warning("Podcast bias synthesis fell back; error_type=%s", type(exc).__name__)
            strongest = max(window_results, key=lambda item: item["bias"].bias_score)["bias"]
            synthesis = {
                "summary": strongest.summary,
                "selected_highlight_ids": ordered_ids,
                "explanation": strongest.explanation,
                "missing_perspectives": strongest.missing_perspectives,
            }
    by_id = {candidate["id"]: candidate for candidate in candidates}
    selected = [by_id[value] for value in ordered_ids if value in by_id]
    result = AIresultBias(
        bias_score=score,
        summary=synthesis["summary"],
        highlights=[item["phrase"] for item in selected],
        highlight_reasons=[
            {"phrase": item["phrase"], "reason": item["reason"]} for item in selected
        ],
        explanation=synthesis["explanation"],
        missing_perspectives=synthesis["missing_perspectives"],
    )
    return result, [item["location"] for item in selected]


def _aggregate_fact_opinion(window_results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"fact": 0, "opinion": 0, "mixed": 0, "unresolved": 0, "openai_reviewed": 0}
    items: list[dict[str, Any]] = []
    threshold = 0.0
    for window_index, item in enumerate(window_results):
        result: FactOpinionResult = item["classification"]
        threshold = result.confidence_threshold
        for key in counts:
            counts[key] += int(getattr(result.counts, key))
        for raw in result.items:
            serialized = raw.model_dump(mode="json")
            serialized["id"] = f"window-{window_index + 1}:{serialized['id']}"
            if serialized.get("start_offset") is not None:
                serialized["start_offset"] += int(item["window_start"])
                serialized["end_offset"] += int(item["window_start"])
            items.append(serialized)
    return {
        "status": "partial" if counts["unresolved"] else "classified",
        "confidence_threshold": threshold,
        "counts": counts,
        "items": items[:PODCAST_COMPACT_ITEMS],
        "items_truncated": len(items) > PODCAST_COMPACT_ITEMS,
        "total_items": len(items),
    }


def _distributed_research_input(window_results: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], int, int]:
    available = [item for item in window_results if item["factual_text"]]
    if not available:
        return "", [], 0, 0
    total_chars = sum(len(item["factual_text"]) for item in available)
    candidate_count = sum(int(item["candidate_count"]) for item in available)
    per_window = max(1, MAX_RESEARCH_INPUT_CHARS // len(available))
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    used = 0
    for item in available:
        remaining = MAX_RESEARCH_INPUT_CHARS - used - (1 if parts else 0)
        if remaining <= 0:
            break
        take = min(len(item["factual_text"]), per_window, remaining)
        separator = "\n" if parts else ""
        output_start = used + len(separator)
        parts.append(separator + item["factual_text"][:take])
        for span in item["factual_speaker_spans"]:
            start = int(span["start_offset"])
            end = min(take, int(span["end_offset"]))
            if start >= end:
                continue
            clipped = dict(span)
            clipped["start_offset"] = output_start + start
            clipped["end_offset"] = output_start + end
            spans.append(clipped)
        used += len(separator) + take
    return "".join(parts), spans, candidate_count, total_chars


async def analyze_podcast_transcript(
    transcript: PodcastTranscript,
    *,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    windows = _podcast_windows(transcript)
    if not windows:
        raise ValueError("Podcast transcript contains no analyzable windows.")
    window_results: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        if progress:
            progress(
                f"Classifying transcript window {index + 1} of {len(windows)}...",
                45 + round(25 * index / len(windows)),
            )
        classification = await classify_article_fact_opinion(
            window["text"], f"{transcript.title} — transcript part {index + 1}"
        )
        _attach_segment_classifications(transcript, int(window["start"]), classification)
        bias_text, speaker_spans = _build_podcast_routed_content(
            classification,
            window["text"],
            window["segments"],
            include_opinions=True,
        )
        factual_text, factual_speaker_spans = _build_podcast_routed_content(
            classification,
            window["text"],
            window["segments"],
            include_opinions=False,
        )
        if bias_text:
            raw_bias = await analyze_bias(
                bias_text,
                transcript.title,
                extract_quoted_phrases(bias_text),
                speaker_spans=speaker_spans,
                source_kind="podcast",
            )
            if "error" in raw_bias:
                raise ValueError("Podcast bias analysis failed.")
            bias = validate_ai_bias(raw_bias, bias_text)
        else:
            bias = no_factual_bias_result()
        window_results.append(
            {
                "window_start": window["start"],
                # Weight the aggregate by the complete transcript window, not
                # only the subset the classifier resolved for model routing.
                "weight": len(window["text"]),
                "classification": classification,
                "bias": bias,
                "bias_text": bias_text,
                "speaker_spans": speaker_spans,
                "factual_text": factual_text,
                "factual_speaker_spans": factual_speaker_spans,
                "candidate_count": _research_candidate_count(classification),
            }
        )
    if progress:
        progress("Combining episode-wide bias results...", 74)
    bias_result, highlight_locations = await _aggregate_podcast_bias(
        transcript.title, window_results
    )
    research_text, research_speakers, candidate_count, total_fact_chars = _distributed_research_input(window_results)
    if progress:
        progress("Researching episode-wide factual claims...", 82)
    if research_text:
        raw_research = await researcher_ai(
            research_text,
            transcript.title,
            extract_quoted_phrases(research_text),
            bias_result=bias_result,
            candidate_claim_count=candidate_count,
            article_input_truncated=total_fact_chars > len(research_text),
            speaker_spans=research_speakers,
            source_kind="podcast",
        )
        if "error" in raw_research:
            raise ValueError("Podcast research verification failed.")
        research_result = validate_ai_research(
            raw_research,
            candidate_claim_count=candidate_count,
            total_factual_characters=total_fact_chars,
            article_input_truncated=total_fact_chars > len(research_text),
        )
    else:
        research_result = no_factual_research_result(False)
    return {
        "podcast": {
            "title": transcript.title,
            "page_url": transcript.page_url,
            "transcript_source": transcript.source,
            "language": transcript.language,
            "duration_seconds": transcript.duration_seconds,
            "segment_count": len(transcript.segments),
            "transcript_characters": len(transcript.text),
            "window_count": len(windows),
            "windows_analyzed": len(window_results),
            "highlight_locations": highlight_locations,
        },
        "ai_result": bias_result.model_dump(mode="json"),
        "ai_research": research_result.model_dump(mode="json"),
        "fact_opinion": _aggregate_fact_opinion(window_results),
    }


async def run_podcast_job(job_id: str, request: PodcastJobRequest) -> None:
    """Own one complete discover/transcribe/analyze job beyond popup lifetime."""
    def stage(message: str, progress: int) -> None:
        _update_podcast_job(
            job_id,
            status="running",
            stage=message,
            progress=max(0, min(99, int(progress))),
        )

    try:
        with tempfile.TemporaryDirectory(prefix="factgpt-podcast-") as temp_dir:
            transcript = await discover_podcast_transcript(
                request,
                workdir=Path(temp_dir),
                stage=stage,
            )
            result = await analyze_podcast_transcript(transcript, progress=stage)
            _update_podcast_job(
                job_id,
                status="complete",
                stage="Podcast analysis complete.",
                progress=100,
                result=result,
                segments=[segment.model_dump(mode="json") for segment in transcript.segments],
                error=None,
                completed_at=time.time(),
            )
    except Exception as exc:
        error_id = hashlib.sha256(
            f"{time.time_ns()}:{type(exc).__name__}".encode("utf-8")
        ).hexdigest()[:12]
        logger.exception(
            "Podcast job failed; job_id=%s error_id=%s error_type=%s",
            job_id,
            error_id,
            type(exc).__name__,
        )
        safe_message = str(exc) if isinstance(exc, ValueError) else "Podcast analysis failed."
        _update_podcast_job(
            job_id,
            status="failed",
            stage="Podcast analysis failed.",
            progress=100,
            error={
                "message": safe_message[:300],
                "code": "podcast_job_failed",
                "reference": error_id,
            },
            completed_at=time.time(),
        )
    finally:
        podcast_job_tasks.pop(job_id, None)


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
        return {"status": "extracted", "method": "playwright", "text": cap_extracted_text(rendered_text)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Playwright extraction failed: {exc}") from exc


@app.post("/article-jobs", status_code=202)
async def create_article_job(payload: ArticleJobRequest, request: Request):
    """Queue backend-owned article extraction, classification, and analysis."""
    _cleanup_article_jobs()
    client_id = client_identifier(request)
    content_key = _article_content_key(payload, client_id)
    existing_id = article_jobs_by_content.get(content_key)
    existing = article_jobs.get(existing_id or "")
    if existing and existing.get("status") in {"queued", "running", "complete"}:
        return {
            "job_id": existing_id,
            "status": existing["status"],
            "stage": existing["stage"],
            "created_at": existing["created_at"],
            "reused": True,
        }

    if any(
        job.get("client_id") == client_id and job.get("status") in {"queued", "running"}
        for job in article_jobs.values()
    ):
        raise HTTPException(
            status_code=409,
            detail="An article analysis is already running for this client.",
        )

    job_id = secrets.token_urlsafe(24)
    created_at = time.time()
    article_jobs[job_id] = {
        "job_id": job_id,
        "client_id": client_id,
        "content_key": content_key,
        "page_url": str(payload.page_url),
        "status": "queued",
        "stage": "Article analysis queued.",
        "progress": 0,
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    article_jobs_by_content[content_key] = job_id
    task = asyncio.create_task(run_article_job(job_id, payload))
    article_job_tasks[job_id] = task
    return {
        "job_id": job_id,
        "status": "queued",
        "stage": "Article analysis queued.",
        "created_at": created_at,
        "reused": False,
    }


@app.get("/article-jobs/{job_id}")
async def get_article_job(job_id: str):
    """Return article progress, a completed result, or preserved partial output."""
    _cleanup_article_jobs()
    job = article_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Article job was not found. The backend may have restarted; "
                "start the analysis again."
            ),
        )
    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
        "result": job.get("result"),
    }


@app.post("/podcast-jobs", status_code=202)
async def create_podcast_job(payload: PodcastJobRequest, request: Request):
    """Queue current-page podcast discovery, transcription, and analysis."""
    _cleanup_podcast_jobs()
    page_url = str(payload.page_url)
    url_key = _normalized_podcast_url(page_url)
    existing_id = podcast_jobs_by_url.get(url_key)
    existing = podcast_jobs.get(existing_id or "")
    if existing and existing.get("status") in {"queued", "running", "complete"}:
        return {
            "job_id": existing_id,
            "status": existing["status"],
            "stage": existing["stage"],
            "created_at": existing["created_at"],
            "reused": True,
        }
    client_id = client_identifier(request)
    if any(
        job.get("client_id") == client_id and job.get("status") in {"queued", "running"}
        for job in podcast_jobs.values()
    ):
        raise HTTPException(
            status_code=409,
            detail="A podcast analysis is already running for this client.",
        )
    job_id = secrets.token_urlsafe(24)
    created_at = time.time()
    podcast_jobs[job_id] = {
        "job_id": job_id,
        "client_id": client_id,
        "url_key": url_key,
        "page_url": page_url,
        "status": "queued",
        "stage": "Podcast analysis queued.",
        "progress": 0,
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
        "result": None,
        "segments": [],
        "error": None,
    }
    podcast_jobs_by_url[url_key] = job_id
    task = asyncio.create_task(run_podcast_job(job_id, payload))
    podcast_job_tasks[job_id] = task
    return {
        "job_id": job_id,
        "status": "queued",
        "stage": "Podcast analysis queued.",
        "created_at": created_at,
        "reused": False,
    }


@app.get("/podcast-jobs/{job_id}")
async def get_podcast_job(job_id: str):
    """Return compact progress or the final podcast analysis."""
    _cleanup_podcast_jobs()
    job = podcast_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Podcast job was not found. The backend may have restarted; "
                "start the analysis again."
            ),
        )
    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
        "result": job.get("result") if job["status"] == "complete" else None,
    }


@app.get("/podcast-jobs/{job_id}/segments")
async def get_podcast_job_segments(
    job_id: str,
    cursor: int = 0,
    limit: int = 100,
):
    """Page through speaker turns without storing the transcript in Chrome."""
    _cleanup_podcast_jobs()
    job = podcast_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Podcast job was not found.")
    if job.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Podcast transcript is not ready yet.")
    if cursor < 0 or limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Invalid transcript pagination.")
    segments = list(job.get("segments") or [])
    page = segments[cursor : cursor + limit]
    next_cursor = cursor + len(page)
    return {
        "job_id": job_id,
        "segments": page,
        "cursor": cursor,
        "next_cursor": next_cursor if next_cursor < len(segments) else None,
        "total": len(segments),
    }


@app.post("/classify-fact-opinion")
async def classify_fact_opinion(req: FactOpinionRequest):
    """Classify locally, resolving only ambiguous items through OpenAI."""
    try:
        return await resolve_fact_opinion_items(req.items, req.title)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The local fact-opinion classifier is unavailable.",
        ) from exc


@app.post("/analyze")
async def analyze(article: AnalyzeRequest):
    """Classify once, then run bias and research on resolved facts only."""
    text = normalize_analysis_text(article.text)
    if len(text) < MIN_EXTRACT_CHARS:
        raise HTTPException(status_code=400, detail="Not enough text to analyze.")

    try:
        fact_opinion = await classify_article_fact_opinion(text, article.title)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The fact-opinion classifier is unavailable.",
        ) from exc

    factual_text, factual_quotes = build_factual_content(fact_opinion, text)
    bias_text, bias_quotes = build_bias_content(fact_opinion, text)
    candidate_claim_count = _research_candidate_count(fact_opinion)
    article_input_truncated = len(str(article.text or "").strip()) > len(text)
    if bias_text:
        bias_raw = await analyze_bias(bias_text, article.title, bias_quotes)
        if "error" in bias_raw:
            raise HTTPException(
                status_code=502,
                detail=model_error_detail(bias_raw, "Bias analysis failed."),
            )
        bias_result = validate_ai_bias(bias_raw, bias_text)
    else:
        bias_result = no_factual_bias_result()

    if factual_text:
        research_raw = await researcher_ai(
            factual_text,
            article.title,
            factual_quotes,
            bias_result=bias_result,
            candidate_claim_count=candidate_claim_count,
            article_input_truncated=article_input_truncated,
        )
        if "error" in research_raw:
            raise HTTPException(
                status_code=502,
                detail=model_error_detail(research_raw, "Research verification failed."),
            )
        research_result = validate_ai_research(
            research_raw,
            candidate_claim_count=candidate_claim_count,
            total_factual_characters=len(factual_text),
            article_input_truncated=article_input_truncated,
        )
    else:
        research_result = no_factual_research_result(article_input_truncated)

    return {
        "status": "analyzed",
        "ai_result": bias_result,
        "ai_research": research_result,
        "fact_opinion": fact_opinion,
    }


@app.post("/analyze-bias")
async def receive_bias(article: AnalyzeRequest):
    """Classify the article and analyze resolved factual text for bias."""
    text = normalize_analysis_text(article.text)
    if len(text) < MIN_EXTRACT_CHARS:
        raise HTTPException(status_code=400, detail="Not enough text to analyze.")

    try:
        fact_opinion = await classify_article_fact_opinion(text, article.title)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The fact-opinion classifier is unavailable.",
        ) from exc

    bias_text, bias_quotes = build_bias_content(fact_opinion, text)
    if bias_text:
        bias_raw = await analyze_bias(bias_text, article.title, bias_quotes)
        if "error" in bias_raw:
            raise HTTPException(
                status_code=502,
                detail=model_error_detail(bias_raw, "Bias analysis failed."),
            )
        bias_result = validate_ai_bias(bias_raw, bias_text)
    else:
        bias_result = no_factual_bias_result()
    return {
        "status": "bias_analyzed",
        "ai_result": bias_result,
        "fact_opinion": fact_opinion,
    }


@app.post("/research")
async def receive_research(article: ResearchRequest):
    """Research resolved facts, reusing a matching prior classification."""
    text = normalize_analysis_text(article.text)
    if len(text) < MIN_EXTRACT_CHARS:
        raise HTTPException(status_code=400, detail="Not enough text to research.")

    try:
        fact_opinion = await ensure_article_classification(
            text, article.title, article.fact_opinion
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The fact-opinion classifier is unavailable.",
        ) from exc

    factual_text, factual_quotes = build_factual_content(fact_opinion, text)
    candidate_claim_count = _research_candidate_count(fact_opinion)
    article_input_truncated = len(str(article.text or "").strip()) > len(text)
    if factual_text:
        research_raw = await researcher_ai(
            factual_text,
            article.title,
            factual_quotes,
            bias_result=article.bias_result,
            candidate_claim_count=candidate_claim_count,
            article_input_truncated=article_input_truncated,
        )
        if "error" in research_raw:
            raise HTTPException(
                status_code=502,
                detail=model_error_detail(research_raw, "Research verification failed."),
            )
        research_result = validate_ai_research(
            research_raw,
            candidate_claim_count=candidate_claim_count,
            total_factual_characters=len(factual_text),
            article_input_truncated=article_input_truncated,
        )
    else:
        research_result = no_factual_research_result(article_input_truncated)
    return {
        "status": "researched",
        "ai_research": research_result,
        "fact_opinion": fact_opinion,
    }


if __name__ == "__main__":
    # Local development entrypoint.
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
