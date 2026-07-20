#!/usr/bin/env python3
"""Normalize an acquired news-vs-opinion corpus for automatic ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "0": "fact",
    "fact": "fact",
    "news": "fact",
    "objective": "fact",
    "1": "opinion",
    "opinion": "opinion",
    "editorial": "opinion",
    "op-ed": "opinion",
    "subjective": "opinion",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column")
    parser.add_argument("--split-column")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/fact_opinion/fact_vs_opinion_articles/recovered/fact_vs_opinion.csv"
        ),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {args.text_column, args.label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing input columns: {sorted(missing)}")

    output = pd.DataFrame(
        {
            "text": frame[args.text_column].fillna("").astype(str).str.strip(),
            "label": frame[args.label_column]
            .astype(str)
            .str.strip()
            .str.casefold()
            .map(LABEL_MAP),
        }
    )
    if args.group_column:
        output["group_id"] = frame[args.group_column].astype(str)
    if args.split_column:
        output["split"] = frame[args.split_column].astype(str)
    output = output[output["text"].ne("") & output["label"].notna()].copy()
    output = output.drop_duplicates(subset="text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} rows to {args.output}")


if __name__ == "__main__":
    main()
