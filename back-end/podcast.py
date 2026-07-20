"""Podcast discovery, transcript normalization, and bounded audio transcription."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


PODCAST_NAMESPACE = "https://podcastindex.org/namespace/1.0"
TRANSCRIPT_MIME_TYPES = {
    "text/plain",
    "text/html",
    "text/vtt",
    "application/json",
    "application/x-subrip",
    "text/srt",
}
AUDIO_MIME_PREFIXES = ("audio/", "video/mp4")
DEFAULT_SPEAKER = "Speaker A"


class PodcastHints(BaseModel):
    """Best-effort source hints collected from the active browser tab."""

    model_config = ConfigDict(extra="forbid")

    feed_urls: list[HttpUrl] = Field(default_factory=list, max_length=5)
    transcript_urls: list[HttpUrl] = Field(default_factory=list, max_length=5)
    audio_urls: list[HttpUrl] = Field(default_factory=list, max_length=5)


class PodcastJobRequest(BaseModel):
    """Create-job request for the current open podcast episode page."""

    model_config = ConfigDict(extra="forbid")

    page_url: HttpUrl = Field(..., max_length=2048)
    hints: PodcastHints = Field(default_factory=PodcastHints)


class PodcastSegment(BaseModel):
    """One exact speaker turn in canonical transcript coordinates."""

    model_config = ConfigDict(extra="forbid")

    id: str
    speaker: str = Field(..., min_length=1, max_length=100)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    text: str = Field(..., min_length=1)
    start_offset: int = Field(..., ge=0)
    end_offset: int = Field(..., ge=1)
    classification: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.end_offset <= self.start_offset:
            raise ValueError("Podcast segment offsets are invalid.")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("Podcast segment timestamps are invalid.")
        return self


class PodcastTranscript(BaseModel):
    """Canonical, offset-preserving transcript used by downstream analysis."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=300)
    page_url: str
    source: Literal[
        "rss_transcript", "page_transcript", "openai_audio"
    ]
    language: str | None = Field(default=None, max_length=40)
    duration_seconds: float | None = Field(default=None, ge=0)
    text: str = Field(..., min_length=1)
    segments: list[PodcastSegment] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_segment_text(self):
        for segment in self.segments:
            if self.text[segment.start_offset : segment.end_offset] != segment.text:
                raise ValueError("Podcast segment offsets do not slice exact text.")
        return self


@dataclass(frozen=True)
class PodcastPageInfo:
    title: str
    canonical_url: str
    published_date: str | None
    embedded_transcript: str
    feed_urls: tuple[str, ...]
    transcript_urls: tuple[str, ...]
    audio_urls: tuple[str, ...]


@dataclass(frozen=True)
class RssEpisodeInfo:
    title: str
    published_date: str | None
    transcript_urls: tuple[tuple[str, str, str | None], ...]
    audio_url: str | None
    language: str | None


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start_seconds: float
    end_seconds: float


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_speaker(value: Any, fallback: str = DEFAULT_SPEAKER) -> str:
    speaker = _clean_text(value).strip("-:[]")[:100]
    return speaker or fallback


def _canonical_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    hostname = (parsed.hostname or "").lower()
    if parsed.port and parsed.port not in {80, 443}:
        hostname = f"{hostname}:{parsed.port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), hostname, path, parsed.query, ""))


def _normalized_date(value: Any) -> str | None:
    """Reduce common page/RSS date forms to an ISO calendar date."""
    raw = _clean_text(value)
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        match = re.match(r"^(\d{4}-\d{2}-\d{2})", raw)
        return match.group(1) if match else None


def _unique_urls(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        try:
            normalized = _canonical_url(value)
        except Exception:
            continue
        if normalized not in seen:
            seen.add(normalized)
            result.append(value)
    return tuple(result)


def inspect_podcast_page(
    html: str,
    page_url: str,
    hints: PodcastHints | None = None,
) -> PodcastPageInfo:
    """Collect publisher transcript/feed/audio sources without trusting them yet."""
    hints = hints or PodcastHints()
    soup = BeautifulSoup(str(html or ""), "html.parser")
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    title = _clean_text(
        (soup.find("meta", property="og:title") or {}).get("content")
        or (soup.title.string if soup.title and soup.title.string else "")
    )
    title = title or "Podcast Analysis"

    feed_urls = [str(url) for url in hints.feed_urls]
    transcript_urls = [str(url) for url in hints.transcript_urls]
    audio_urls = [str(url) for url in hints.audio_urls]
    embedded_transcripts: list[str] = []
    published_date: str | None = None

    for selector, attribute in (
        ("meta[property='article:published_time']", "content"),
        ("meta[name='date']", "content"),
        ("time[datetime]", "datetime"),
    ):
        node = soup.select_one(selector)
        if node:
            published_date = _normalized_date(node.get(attribute))
            if published_date:
                break

    for link in soup.find_all("link", href=True):
        href = urljoin(page_url, link.get("href"))
        link_type = _clean_text(link.get("type")).lower()
        rel = " ".join(link.get("rel") or []).lower()
        if link_type in {"application/rss+xml", "application/atom+xml"} or "alternate" in rel and "rss" in link_type:
            feed_urls.append(href)
        if link_type in TRANSCRIPT_MIME_TYPES or "transcript" in rel:
            transcript_urls.append(href)

    for media in soup.select("audio[src], audio source[src], video[src], video source[src]"):
        src = media.get("src")
        if src and not str(src).startswith(("blob:", "data:")):
            audio_urls.append(urljoin(page_url, src))

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or "")
        except (TypeError, json.JSONDecodeError):
            continue
        stack = list(payload if isinstance(payload, list) else [payload])
        while stack:
            item = stack.pop()
            if not isinstance(item, dict):
                continue
            for nested in item.values():
                if isinstance(nested, dict):
                    stack.append(nested)
                elif isinstance(nested, list):
                    stack.extend(value for value in nested if isinstance(value, dict))
            kind = item.get("@type")
            kinds = set(kind if isinstance(kind, list) else [kind])
            if kinds.intersection({"PodcastEpisode", "AudioObject", "MediaObject"}):
                if not published_date:
                    published_date = _normalized_date(
                        item.get("datePublished") or item.get("uploadDate")
                    )
                transcript = item.get("transcript")
                if isinstance(transcript, str):
                    if transcript.startswith(("http://", "https://")):
                        transcript_urls.append(transcript)
                    elif len(_clean_text(transcript)) >= 200:
                        embedded_transcripts.append(transcript)
                for key in ("contentUrl", "embedUrl"):
                    value = item.get(key)
                    if isinstance(value, str) and value.startswith(("http://", "https://")):
                        audio_urls.append(value)

    transcript_nodes = soup.select(
        "[id*='transcript' i], [class*='transcript' i], [data-testid*='transcript' i]"
    )
    for node in transcript_nodes:
        candidate = node.get_text("\n", strip=True)
        if len(_clean_text(candidate)) >= 500:
            embedded_transcripts.append(candidate)

    canonical_url = page_url
    if canonical and canonical.get("href"):
        canonical_url = urljoin(page_url, canonical.get("href"))
    embedded = max(embedded_transcripts, key=len, default="")
    return PodcastPageInfo(
        title=title[:300],
        canonical_url=canonical_url,
        published_date=published_date,
        embedded_transcript=embedded,
        feed_urls=_unique_urls(feed_urls),
        transcript_urls=_unique_urls(transcript_urls),
        audio_urls=_unique_urls(audio_urls),
    )


def _element_text(element: ET.Element | None) -> str:
    return _clean_text(element.text if element is not None else "")


def _child_text(item: ET.Element, local_name: str) -> str:
    for child in list(item):
        if child.tag.rsplit("}", 1)[-1].lower() == local_name.lower():
            return _element_text(child)
    return ""


def select_rss_episode(
    xml_text: str,
    *,
    page_url: str,
    page_title: str,
    page_date: str | None = None,
) -> RssEpisodeInfo | None:
    """Select one unambiguous RSS item for the active episode page."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    channel = root.find("channel")
    language = _child_text(channel if channel is not None else root, "language") or None
    page_key = _canonical_url(page_url)
    normalized_title = _clean_text(page_title).casefold()
    normalized_page_date = _normalized_date(page_date)
    scored: list[tuple[int, ET.Element]] = []
    for item in root.iter():
        if item.tag.rsplit("}", 1)[-1].lower() != "item":
            continue
        score = 0
        link = _child_text(item, "link")
        guid = _child_text(item, "guid")
        for candidate in (link, guid):
            if candidate.startswith(("http://", "https://")):
                try:
                    if _canonical_url(candidate) == page_key:
                        score = max(score, 100)
                except Exception:
                    pass
        item_title = _child_text(item, "title")
        normalized_item_title = item_title.casefold()
        if normalized_title and normalized_item_title == normalized_title:
            score = max(score, 60)
        elif (
            len(normalized_title) >= 12
            and len(normalized_item_title) >= 12
            and (
                normalized_title in normalized_item_title
                or normalized_item_title in normalized_title
            )
        ):
            score = max(score, 40)
        item_date = _normalized_date(
            _child_text(item, "pubDate") or _child_text(item, "date")
        )
        if score and normalized_page_date and item_date == normalized_page_date:
            score += 20
        if score:
            scored.append((score, item))
    if not scored:
        return None
    best_score = max(score for score, _ in scored)
    best = [item for score, item in scored if score == best_score]
    if len(best) != 1:
        return None
    item = best[0]
    transcript_urls: list[tuple[str, str, str | None]] = []
    audio_url: str | None = None
    for child in list(item):
        local = child.tag.rsplit("}", 1)[-1].lower()
        namespace = child.tag[1:].split("}", 1)[0] if child.tag.startswith("{") else ""
        if local == "transcript" and namespace in {
            PODCAST_NAMESPACE,
            "https://github.com/Podcastindex-org/podcast-namespace/blob/main/docs/1.0.md",
        }:
            url = child.attrib.get("url", "").strip()
            mime = child.attrib.get("type", "text/plain").split(";", 1)[0].lower()
            if url and mime in TRANSCRIPT_MIME_TYPES:
                transcript_urls.append((url, mime, child.attrib.get("language")))
        if local == "enclosure":
            mime = child.attrib.get("type", "").lower()
            url = child.attrib.get("url", "").strip()
            if url and mime.startswith(AUDIO_MIME_PREFIXES):
                audio_url = url
    return RssEpisodeInfo(
        title=_child_text(item, "title") or page_title,
        published_date=_normalized_date(
            _child_text(item, "pubDate") or _child_text(item, "date")
        ),
        transcript_urls=tuple(transcript_urls),
        audio_url=audio_url,
        language=language,
    )


def _timestamp_seconds(value: str) -> float | None:
    raw = value.strip().replace(",", ".")
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = "0"
            minutes, seconds = parts
        else:
            return None
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def _speaker_and_text(lines: list[str], default: str = DEFAULT_SPEAKER) -> tuple[str, str]:
    joined = _clean_text(" ".join(lines))
    voice = re.match(r"^<v(?:\.[^ >]+)*\s+([^>]+)>(.*)$", joined, re.IGNORECASE)
    if voice:
        return _safe_speaker(voice.group(1), default), _clean_text(voice.group(2))
    joined = re.sub(r"<[^>]+>", "", joined)
    label = re.match(r"^([\w .'-]{1,60}):\s+(.+)$", joined)
    if label:
        return _safe_speaker(label.group(1), default), _clean_text(label.group(2))
    return default, joined


def parse_vtt(text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", str(text or "").replace("\r", "\n"))
    segments: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        left, right = [part.strip().split(" ", 1)[0] for part in lines[timing_index].split("-->", 1)]
        speaker, cue_text = _speaker_and_text(lines[timing_index + 1 :])
        if cue_text:
            segments.append(
                {
                    "speaker": speaker,
                    "start_seconds": _timestamp_seconds(left),
                    "end_seconds": _timestamp_seconds(right),
                    "text": cue_text,
                }
            )
    return segments


def parse_srt(text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", str(text or "").replace("\r", "\n"))
    segments: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        left, right = [part.strip().split(" ", 1)[0] for part in lines[timing_index].split("-->", 1)]
        speaker, cue_text = _speaker_and_text(lines[timing_index + 1 :])
        if cue_text:
            segments.append(
                {
                    "speaker": speaker,
                    "start_seconds": _timestamp_seconds(left),
                    "end_seconds": _timestamp_seconds(right),
                    "text": cue_text,
                }
            )
    return segments


def parse_plain_transcript(text: str, html: bool = False) -> list[dict[str, Any]]:
    value = BeautifulSoup(text, "html.parser").get_text("\n") if html else str(text or "")
    segments: list[dict[str, Any]] = []
    for line in value.replace("\r", "\n").splitlines():
        line = _clean_text(line)
        if not line:
            continue
        speaker, cue_text = _speaker_and_text([line])
        if cue_text:
            segments.append(
                {
                    "speaker": speaker,
                    "start_seconds": None,
                    "end_seconds": None,
                    "text": cue_text,
                }
            )
    return segments


def parse_json_transcript(text: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        candidates = payload.get("segments") or payload.get("cues") or payload.get("transcript")
    else:
        candidates = payload
    if not isinstance(candidates, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        cue_text = _clean_text(item.get("text") or item.get("body") or item.get("content"))
        if not cue_text:
            continue
        start_value = item.get("start") if "start" in item else item.get("startTime")
        end_value = item.get("end") if "end" in item else item.get("endTime")
        segments.append(
            {
                "speaker": _safe_speaker(item.get("speaker") or item.get("voice")),
                "start_seconds": _coerce_seconds(start_value),
                "end_seconds": _coerce_seconds(end_value),
                "text": cue_text,
            }
        )
    return segments


def _coerce_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return _timestamp_seconds(str(value))


def parse_publisher_transcript(text: str, mime_type: str) -> list[dict[str, Any]]:
    mime = str(mime_type or "text/plain").split(";", 1)[0].lower()
    if mime == "text/vtt":
        return parse_vtt(text)
    if mime in {"application/x-subrip", "text/srt"}:
        return parse_srt(text)
    if mime == "application/json":
        return parse_json_transcript(text)
    return parse_plain_transcript(text, html=mime == "text/html")


def canonicalize_transcript(
    raw_segments: list[dict[str, Any]],
    *,
    title: str,
    page_url: str,
    source: Literal["rss_transcript", "page_transcript", "openai_audio"],
    language: str | None = None,
    duration_seconds: float | None = None,
) -> PodcastTranscript:
    text_parts: list[str] = []
    segments: list[PodcastSegment] = []
    cursor = 0
    for raw in raw_segments:
        cue_text = _clean_text(raw.get("text"))
        if not cue_text:
            continue
        if text_parts:
            text_parts.append("\n")
            cursor += 1
        start_offset = cursor
        text_parts.append(cue_text)
        cursor += len(cue_text)
        segments.append(
            PodcastSegment(
                id=f"podcast-segment-{len(segments) + 1:05d}",
                speaker=_safe_speaker(raw.get("speaker")),
                start_seconds=_coerce_seconds(raw.get("start_seconds")),
                end_seconds=_coerce_seconds(raw.get("end_seconds")),
                text=cue_text,
                start_offset=start_offset,
                end_offset=cursor,
            )
        )
    if not segments:
        raise ValueError("The publisher transcript contained no usable speech.")
    return PodcastTranscript(
        title=_clean_text(title) or "Podcast Analysis",
        page_url=page_url,
        source=source,
        language=language,
        duration_seconds=duration_seconds,
        text="".join(text_parts),
        segments=segments,
    )


async def fetch_public_bytes(
    url: str,
    *,
    validate_url: Callable[[str], Awaitable[None]],
    max_bytes: int,
    timeout_seconds: float = 30.0,
    destination: Path | None = None,
) -> tuple[bytes | None, str, str]:
    """Fetch a public URL with redirect validation and a streamed byte cap."""
    current_url = str(url)
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        headers={"User-Agent": "FactGPT/1.0 podcast analyzer"},
    ) as http:
        for _ in range(6):
            await validate_url(current_url)
            async with http.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Podcast source redirect omitted its location.")
                    current_url = urljoin(str(response.url), location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                total = 0
                chunks: list[bytes] = []
                output = destination.open("wb") if destination is not None else None
                try:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError("Podcast source exceeds the configured byte limit.")
                        if output is not None:
                            output.write(chunk)
                        else:
                            chunks.append(chunk)
                finally:
                    if output is not None:
                        output.close()
                return (None if destination is not None else b"".join(chunks)), content_type, current_url
    raise ValueError("Podcast source exceeded the redirect limit.")


async def _run_process(*args: str) -> tuple[str, str]:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"Media processing failed ({Path(args[0]).name}).")
    return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


async def probe_duration(path: Path) -> float:
    stdout, _ = await _run_process(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    )
    try:
        duration = float(stdout.strip())
    except ValueError as exc:
        raise ValueError("Podcast audio duration could not be determined.") from exc
    if duration <= 0:
        raise ValueError("Podcast audio has no usable duration.")
    return duration


async def _silence_boundaries(path: Path) -> list[float]:
    _, stderr = await _run_process(
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(path),
        "-af",
        "silencedetect=noise=-35dB:d=0.5",
        "-f",
        "null",
        "-",
    )
    return [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", stderr)]


def choose_chunk_boundaries(
    duration: float,
    silences: list[float],
    *,
    target_seconds: float = 2700.0,
    search_seconds: float = 120.0,
) -> list[float]:
    boundaries = [0.0]
    while duration - boundaries[-1] > target_seconds:
        target = boundaries[-1] + target_seconds
        candidates = [value for value in silences if abs(value - target) <= search_seconds and value > boundaries[-1] + 60]
        boundary = min(candidates, key=lambda value: abs(value - target)) if candidates else target
        boundaries.append(min(boundary, duration))
    boundaries.append(duration)
    return boundaries


async def transcode_audio_chunks(
    source: Path,
    output_dir: Path,
    *,
    duration_seconds: float,
) -> list[AudioChunk]:
    """Create speech-optimized, silence-aligned MP3 chunks below the API cap."""
    silences = await _silence_boundaries(source)
    boundaries = choose_chunk_boundaries(duration_seconds, silences)
    chunks: list[AudioChunk] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        target = output_dir / f"podcast-chunk-{index:04d}.mp3"
        await _run_process(
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "48k",
            str(target),
        )
        if target.stat().st_size >= 20_000_000:
            raise ValueError("A transcoded podcast chunk still exceeds 20 MB.")
        chunks.append(AudioChunk(target, start, end))
    return chunks


def _object_value(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _speaker_label(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return f"Speaker {alphabet[index]}"
    return f"Speaker {index + 1}"


async def _reference_clip(
    chunk: AudioChunk,
    *,
    start: float,
    end: float,
    target: Path,
) -> str:
    clip_start = max(0.0, start)
    clip_duration = min(10.0, max(2.0, end - start))
    await _run_process(
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{clip_start:.3f}",
        "-t",
        f"{clip_duration:.3f}",
        "-i",
        str(chunk.path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    )
    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return f"data:audio/mpeg;base64,{encoded}"


async def transcribe_audio_chunks(
    openai_client: Any,
    chunks: list[AudioChunk],
    *,
    reference_dir: Path,
) -> list[dict[str, Any]]:
    """Transcribe chunks while preserving global time and safe speaker labels."""
    merged: list[dict[str, Any]] = []
    known_references: dict[str, str] = {}
    used_labels: set[str] = set()
    for chunk_index, chunk in enumerate(chunks):
        request: dict[str, Any] = {
            "model": "gpt-4o-transcribe-diarize",
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        }
        if known_references:
            request["known_speaker_names"] = list(known_references)
            request["known_speaker_references"] = list(known_references.values())
        with chunk.path.open("rb") as audio_file:
            response = await openai_client.audio.transcriptions.create(
                file=audio_file,
                **request,
            )
        raw_segments = list(_object_value(response, "segments", []) or [])
        chunk_speaker_map: dict[str, str] = {}
        for raw in raw_segments:
            raw_speaker = _clean_text(_object_value(raw, "speaker")) or "unknown"
            if raw_speaker in known_references:
                safe_speaker = raw_speaker
            elif raw_speaker in chunk_speaker_map:
                safe_speaker = chunk_speaker_map[raw_speaker]
            else:
                next_index = 0
                while _speaker_label(next_index) in used_labels:
                    next_index += 1
                safe_speaker = _speaker_label(next_index)
                used_labels.add(safe_speaker)
                chunk_speaker_map[raw_speaker] = safe_speaker
            start = float(_object_value(raw, "start", 0.0) or 0.0)
            end = float(_object_value(raw, "end", start) or start)
            cue_text = _clean_text(_object_value(raw, "text"))
            if not cue_text:
                continue
            merged.append(
                {
                    "speaker": safe_speaker,
                    "start_seconds": chunk.start_seconds + start,
                    "end_seconds": chunk.start_seconds + end,
                    "text": cue_text,
                }
            )

        if len(known_references) < 4:
            candidates: dict[str, tuple[float, float]] = {}
            for raw in raw_segments:
                raw_speaker = _clean_text(_object_value(raw, "speaker")) or "unknown"
                safe = raw_speaker if raw_speaker in known_references else chunk_speaker_map.get(raw_speaker)
                if not safe or safe in known_references:
                    continue
                start = float(_object_value(raw, "start", 0.0) or 0.0)
                end = float(_object_value(raw, "end", start) or start)
                if end - start >= 2 and (
                    safe not in candidates or end - start > candidates[safe][1] - candidates[safe][0]
                ):
                    candidates[safe] = (start, end)
            for safe, (start, end) in candidates.items():
                if len(known_references) >= 4:
                    break
                target = reference_dir / f"reference-{chunk_index}-{len(known_references)}.mp3"
                known_references[safe] = await _reference_clip(
                    chunk, start=start, end=end, target=target
                )

    return deduplicate_adjacent_segments(merged)


def deduplicate_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact repeated boundary cues if a chunker/provider overlaps them."""
    result: list[dict[str, Any]] = []
    for segment in segments:
        if result:
            previous = result[-1]
            same_text = _clean_text(previous.get("text")).casefold() == _clean_text(segment.get("text")).casefold()
            previous_end = _coerce_seconds(previous.get("end_seconds"))
            current_start = _coerce_seconds(segment.get("start_seconds"))
            overlaps = previous_end is not None and current_start is not None and current_start <= previous_end + 1.0
            if same_text and overlaps:
                previous["end_seconds"] = max(
                    float(previous.get("end_seconds") or 0),
                    float(segment.get("end_seconds") or 0),
                )
                continue
        result.append(dict(segment))
    return result
