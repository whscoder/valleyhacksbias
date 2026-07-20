#!/usr/bin/env python3
"""Import the public Webis and iDebate replacements for missing FactVsOp data."""

from __future__ import annotations

import html
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERED = REPO_ROOT / "data/fact_opinion/fact_vs_opinion_articles/recovered"
WEBIS_ZIP = RECOVERED / "corpus-webis-editorials-16.zip"
IDEBATE_JSON = RECOVERED / "idebate-annotation.json"
OUTPUT = RECOVERED / "fact_vs_opinion.csv"


def clean_text(value: object) -> str:
    text = html.unescape(str(value))
    text = text.replace("\\'", "'")
    return re.sub(r"\s+", " ", text).strip()


def load_webis() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with zipfile.ZipFile(WEBIS_ZIP) as archive:
        member = "corpus-webis-editorials-16/unannotated.csv"
        frame = pd.read_csv(
            io.BytesIO(archive.read(member)), sep="\t", dtype=str, keep_default_na=False
        )
        split_by_id: dict[str, str] = {}
        pattern = re.compile(
            r"annotated-txt/split-for-evaluation-final/"
            r"(training|validation|test)/([^/]+)\.txt$"
        )
        for name in archive.namelist():
            match = pattern.search(name)
            if match:
                split_by_id[match.group(2)] = {
                    "training": "train",
                    "validation": "validation",
                    "test": "test",
                }[match.group(1)]
        for row in frame.itertuples(index=False):
            article_id = str(getattr(row, "_0", ""))
            # itertuples renames spaced headers, so use positional access.
            values = list(row)
            article_id = str(values[0])
            records.append(
                {
                    "text": clean_text(values[6]),
                    "label": "opinion",
                    "group_id": f"webis:{article_id}",
                    "split": split_by_id.get(article_id, "train"),
                    "source_detail": "webis_editorials_16",
                }
            )
    return records


def load_idebate() -> list[dict[str, str]]:
    topics = json.loads(IDEBATE_JSON.read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []
    for topic in topics:
        topic_id = topic["_topic_ID"]
        split = {"valid": "validation"}.get(topic["_split"], topic["_split"])
        for annotation in topic["_annotation"]:
            body = annotation["_doc_body"]
            for line_id, labels in zip(
                annotation["_labeled_lineIDs"], annotation["_labels"]
            ):
                label_set = {str(label).casefold() for label in labels}
                if label_set == {"factual"}:
                    target = "fact"
                elif label_set == {"opinion"}:
                    target = "opinion"
                else:
                    continue
                records.append(
                    {
                        "text": clean_text(body[int(line_id)]),
                        "label": target,
                        "group_id": f"idebate:{topic_id}",
                        "split": split,
                        "source_detail": "idebate_acl2017",
                    }
                )
    return records


def main() -> None:
    missing = [str(path) for path in (WEBIS_ZIP, IDEBATE_JSON) if not path.exists()]
    if missing:
        raise SystemExit(f"Missing downloaded source files: {missing}")
    frame = pd.DataFrame.from_records(load_webis() + load_idebate())
    frame = frame[frame["text"].ne("")].copy()
    label_counts = frame.groupby(frame["text"].str.casefold())["label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    frame = frame[~frame["text"].str.casefold().isin(conflicts)]
    frame = frame.drop_duplicates(subset="text", keep="first")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(frame):,} public replacement rows to {OUTPUT}")
    print(pd.crosstab(frame["source_detail"], frame["label"]).to_string())


if __name__ == "__main__":
    main()
