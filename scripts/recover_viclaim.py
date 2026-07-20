#!/usr/bin/env python3
"""Resume-safe ViClaim audio recovery using yt-dlp and faster-whisper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel
from tqdm import tqdm


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def merge_checkpoint(source: pd.DataFrame, output: Path) -> pd.DataFrame:
    source = source.copy()
    source["sentence"] = ""
    source["recovery_status"] = "pending"
    source["recovery_error"] = ""
    if not output.exists():
        return source

    checkpoint = pd.read_csv(output).fillna("")
    keys = ["clip_id", "sentence_start_millis", "sentence_end_millis"]
    saved_columns = keys + ["sentence", "recovery_status", "recovery_error"]
    if not set(saved_columns).issubset(checkpoint.columns):
        return source
    checkpoint = checkpoint[saved_columns].drop_duplicates(keys, keep="last")
    merged = source.merge(checkpoint, on=keys, how="left", suffixes=("", "_saved"))
    for column in ("sentence", "recovery_status", "recovery_error"):
        saved = merged.pop(f"{column}_saved").fillna("").astype(str)
        merged[column] = saved.where(saved.str.strip().ne(""), merged[column])
    return merged


def find_audio(clip_dir: Path) -> Path | None:
    ignored = {".part", ".ytdl", ".json", ".tmp"}
    candidates = [
        path
        for path in clip_dir.glob("*")
        if path.is_file() and path.suffix.casefold() not in ignored
    ]
    return max(candidates, key=lambda path: path.stat().st_size) if candidates else None


def download_audio(clip_id: str, clip_dir: Path, cookies_browser: str | None) -> Path:
    clip_dir.mkdir(parents=True, exist_ok=True)
    existing = find_audio(clip_dir)
    if existing is not None:
        return existing

    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--continue",
        "--no-overwrites",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--sleep-interval",
        "1",
        "--max-sleep-interval",
        "3",
        "--js-runtimes",
        "node",
        "-f",
        "bestaudio/best",
        "-o",
        str(clip_dir / "%(id)s.%(ext)s"),
    ]
    if cookies_browser:
        command.extend(["--cookies-from-browser", cookies_browser])
    command.append(f"https://www.youtube.com/watch?v={clip_id}")
    subprocess.run(command, check=True)
    audio = find_audio(clip_dir)
    if audio is None:
        raise FileNotFoundError(f"yt-dlp produced no audio for {clip_id}")
    return audio


def transcribe_words(
    model: WhisperModel,
    audio: Path,
    language: str,
    words_cache: Path,
) -> list[dict[str, object]]:
    if words_cache.exists():
        return json.loads(words_cache.read_text(encoding="utf-8"))
    segments, _ = model.transcribe(
        str(audio),
        language=language,
        task="transcribe",
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    words: list[dict[str, object]] = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            words.append(
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word,
                    "probability": float(word.probability),
                }
            )
    atomic_json(words, words_cache)
    return words


def map_words(rows: pd.DataFrame, words: list[dict[str, object]]) -> list[str]:
    sentences: list[str] = []
    for row in rows.itertuples(index=False):
        start = float(row.sentence_start_millis) / 1000.0
        end = float(row.sentence_end_millis) / 1000.0
        selected = [
            str(word["word"])
            for word in words
            if start <= (float(word["start"]) + float(word["end"])) / 2.0 < end
        ]
        sentences.append("".join(selected).strip())
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("VICLAIM_MODEL", "small"))
    parser.add_argument("--device", default=os.getenv("VICLAIM_DEVICE", "cpu"))
    parser.add_argument(
        "--compute-type", default=os.getenv("VICLAIM_COMPUTE_TYPE", "int8")
    )
    parser.add_argument("--clips", nargs="*")
    args = parser.parse_args()

    frame = merge_checkpoint(pd.read_csv(args.input), args.output)
    requested = set(args.clips) if args.clips else None
    missing = frame["sentence"].fillna("").astype(str).str.strip().eq("")
    clip_ids = frame.loc[missing, "clip_id"].drop_duplicates().tolist()
    if requested is not None:
        clip_ids = [clip_id for clip_id in clip_ids if clip_id in requested]

    print(
        f"Loading faster-whisper model={args.model} device={args.device} "
        f"compute_type={args.compute_type}"
    )
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    failures: list[dict[str, str]] = []
    cookies_browser = os.getenv("VICLAIM_COOKIES_BROWSER")

    for clip_id in tqdm(clip_ids, unit="clip", desc="Recovering ViClaim"):
        mask = frame["clip_id"].eq(clip_id)
        clip_rows = frame.loc[mask]
        languages = clip_rows["language"].dropna().astype(str).unique().tolist()
        language = languages[0] if languages else "en"
        clip_dir = args.cache_dir / str(clip_id)
        try:
            audio = download_audio(str(clip_id), clip_dir, cookies_browser)
            words = transcribe_words(model, audio, language, clip_dir / "words.json")
            sentences = map_words(clip_rows, words)
            frame.loc[mask, "sentence"] = sentences
            recovered = pd.Series(sentences).str.strip().ne("").to_numpy()
            frame.loc[mask, "recovery_status"] = [
                "recovered" if value else "empty_interval" for value in recovered
            ]
            frame.loc[mask, "recovery_error"] = ""
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            frame.loc[mask, "recovery_status"] = "failed"
            frame.loc[mask, "recovery_error"] = message
            failures.append({"clip_id": str(clip_id), "error": message})
        atomic_csv(frame, args.output)
        atomic_csv(
            pd.DataFrame(failures, columns=["clip_id", "error"]),
            args.output.with_name("viclaim_recovery_failures.csv"),
        )

    status_counts = frame["recovery_status"].value_counts().to_dict()
    report = {
        "input_rows": len(frame),
        "requested_clips": len(clip_ids),
        "rows_with_text": int(frame["sentence"].astype(str).str.strip().ne("").sum()),
        "status_counts": status_counts,
        "failures_this_run": len(failures),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
    }
    atomic_json(report, args.output.with_name("viclaim_recovery_report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
