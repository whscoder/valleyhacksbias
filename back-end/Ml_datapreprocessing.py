"""Build a clean, unified fact-vs-opinion dataset.

This module only handles data preparation. It does not split data, train a
model, tune hyperparameters, make predictions, or calculate model metrics.

Every retained example is written with the same schema:

``text``
    The normalized text to classify.
``label``
    The binary target: ``fact`` or ``opinion``.
``source``
    The source corpus, retained for auditing.
``group_id``
    A document/debate identifier that a later training module can use to avoid
    placing related rows in different data splits.

Raw source files are read-only. The generated CSV and its JSON data-quality
report are written under ``data/fact_opinion/processed`` by default.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "fact_opinion"
DEFAULT_OUTPUT = DATA_ROOT / "processed" / "fact_opinion.csv"

OUTPUT_COLUMNS = ["text", "label", "source", "group_id"]
VALID_LABELS = {"fact", "opinion"}

SOURCE_PRIORITY = {
    "claimbuster_groundtruth": 0,
    "claimbuster_crowdsourced": 1,
    "factvsop_recovered": 2,
    "viclaim_recovered": 3,
    "what_to_factcheck": 4,
    "subj": 5,
    "claimbuster_ncs_weak_fact": 6,
    "what_to_factcheck_argument_weak": 7,
}


def _empty_dataset() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _normalize_text(value: object) -> str:
    """Normalize Unicode and collapse all whitespace to single spaces."""
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", normalized).strip()


def load_claimbuster(data_root: Path) -> pd.DataFrame:
    """Load labeled ClaimBuster sentences and discard speaker metadata."""
    dataset_dir = (
        data_root
        / "claimbuster_v2"
        / "extracted"
        / "ClaimBuster_Datasets"
        / "datasets"
    )
    label_map = {-1: "opinion", 0: "fact", 1: "fact"}
    frames: list[pd.DataFrame] = []

    for filename, source in (
        ("groundtruth.csv", "claimbuster_groundtruth"),
        ("crowdsourced.csv", "claimbuster_crowdsourced"),
    ):
        path = dataset_dir / filename
        frame = pd.read_csv(path, usecols=["Text", "Verdict", "File_id"])
        _require_columns(frame, {"Text", "Verdict", "File_id"}, path)
        frames.append(
            pd.DataFrame(
                {
                    "text": frame["Text"],
                    "label": frame["Verdict"].map(label_map),
                    "source": source,
                    "group_id": "claimbuster:" + frame["File_id"].astype(str),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]


def _claimbuster_dataset_dir(data_root: Path) -> Path:
    return (
        data_root
        / "claimbuster_v2"
        / "extracted"
        / "ClaimBuster_Datasets"
        / "datasets"
    )


def _claimbuster_labeled_ids(dataset_dir: Path) -> set[int]:
    labeled_ids: set[int] = set()
    for filename in ("groundtruth.csv", "crowdsourced.csv"):
        frame = pd.read_csv(dataset_dir / filename, usecols=["Sentence_id"])
        labeled_ids.update(frame["Sentence_id"].dropna().astype(int))
    return labeled_ids


def load_claimbuster_ncs_weak_facts(data_root: Path) -> pd.DataFrame:
    """Load positive NCS annotations not already present in labeled CSVs."""
    dataset_dir = _claimbuster_dataset_dir(data_root)
    positive_ids: set[int] = set()

    for path in sorted(dataset_dir.glob("*xNCS.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        positive_ids.update(
            int(record["sentence_id"])
            for record in records
            if int(record.get("label", 0)) == 1
        )

    positive_ids.difference_update(_claimbuster_labeled_ids(dataset_dir))
    all_sentences_path = dataset_dir / "all_sentences.csv"
    frame = pd.read_csv(
        all_sentences_path, usecols=["Sentence_id", "Text", "File_id"]
    )
    selected = frame[frame["Sentence_id"].isin(positive_ids)]

    return pd.DataFrame(
        {
            "text": selected["Text"],
            "label": "fact",
            "source": "claimbuster_ncs_weak_fact",
            "group_id": "claimbuster:" + selected["File_id"].astype(str),
        }
    )[OUTPUT_COLUMNS]


def load_subj(data_root: Path) -> pd.DataFrame:
    """Map SUBJ objective/subjective examples to fact/opinion."""
    dataset_dir = data_root / "subj" / "csv"
    label_map = {0: "fact", 1: "opinion"}
    frames: list[pd.DataFrame] = []

    for filename in ("train.csv", "test.csv"):
        path = dataset_dir / filename
        frame = pd.read_csv(path, usecols=["text", "label"])
        _require_columns(frame, {"text", "label"}, path)
        partition = path.stem
        frames.append(
            pd.DataFrame(
                {
                    "text": frame["text"],
                    "label": frame["label"].map(label_map),
                    "source": "subj",
                    "group_id": [
                        f"subj:{partition}:{row_id}" for row_id in range(len(frame))
                    ],
                }
            )
        )

    return pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]


def _read_fact_checked_segments(path: Path) -> list[str]:
    """Read the corpus's unquoted multiline TSV format safely."""
    texts: list[str] = []
    current_text: list[str] | None = None

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line_number == 1 and line == "start\tend\ttext":
                continue

            match = re.match(r"^(\d+)\t(\d+)\t(.*)$", line)
            if match:
                if current_text is not None:
                    texts.append("\n".join(current_text))
                current_text = [match.group(3)]
            elif current_text is not None:
                current_text.append(line)
            elif line.strip():
                raise ValueError(f"Unexpected content in {path} at line {line_number}")

    if current_text is not None:
        texts.append("\n".join(current_text))
    return texts


def load_what_to_factcheck(data_root: Path) -> pd.DataFrame:
    """Load positive fact-checked spans as fact examples."""
    segment_dir = data_root / "what_to_factcheck" / "data" / "fact-checked_segments"
    records: list[dict[str, str]] = []

    for path in sorted(segment_dir.glob("*.tsv")):
        article_id = path.stem
        for text in _read_fact_checked_segments(path):
            records.append(
                {
                    "text": text,
                    "label": "fact",
                    "source": "what_to_factcheck",
                    "group_id": f"what_to_factcheck:{article_id}",
                }
            )

    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def load_argument_annotations_as_weak_facts(data_root: Path) -> pd.DataFrame:
    """Load argumentative claim and premise spans as weak fact examples."""
    annotation_dir = data_root / "what_to_factcheck" / "data" / "argument_annotations"
    allowed_types = {
        "MajorClaim",
        "Claim",
        "Premise",
        "SuperAnnotatorMajorClaim",
        "SuperAnnotatorClaim",
        "SuperAnnotatorPremise",
    }
    records: list[dict[str, str]] = []

    for path in sorted(annotation_dir.glob("*.ann")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("T"):
                    continue
                fields = line.rstrip("\n").split("\t", 2)
                if len(fields) != 3 or fields[1].split()[0] not in allowed_types:
                    continue
                records.append(
                    {
                        "text": fields[2],
                        "label": "fact",
                        "source": "what_to_factcheck_argument_weak",
                        "group_id": f"what_to_factcheck:{path.stem}",
                    }
                )

    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def load_recovered_viclaim(data_root: Path) -> pd.DataFrame:
    """Load recovered ViClaim text and convert its soft scores to binary labels."""
    path = data_root / "processed" / "viclaim_transcribed.csv"
    if not path.exists():
        return _empty_dataset()

    frame = pd.read_csv(path)
    text_column = "sentence" if "sentence" in frame.columns else "text"
    required = {text_column, "label_fcw", "label_fnc", "label_opn", "clip_id"}
    _require_columns(frame, required, path)

    scores = frame[["label_fcw", "label_fnc", "label_opn"]].apply(
        pd.to_numeric, errors="coerce"
    )
    fact_score = scores[["label_fcw", "label_fnc"]].max(axis=1)
    opinion_score = scores["label_opn"]
    score_difference = (fact_score - opinion_score).abs()
    usable = fact_score.notna() & opinion_score.notna() & score_difference.gt(1e-12)

    result = pd.DataFrame(
        {
            "text": frame.loc[usable, text_column],
            "label": "fact",
            "source": "viclaim_recovered",
            "group_id": "viclaim:" + frame.loc[usable, "clip_id"].astype(str),
        }
    )
    result.loc[opinion_score.loc[usable] > fact_score.loc[usable], "label"] = "opinion"
    return result[OUTPUT_COLUMNS]


def load_recovered_factvsop(data_root: Path) -> pd.DataFrame:
    """Load the normalized original or replacement FactVsOp corpus."""
    path = (
        data_root
        / "fact_vs_opinion_articles"
        / "recovered"
        / "fact_vs_opinion.csv"
    )
    if not path.exists():
        return _empty_dataset()

    frame = pd.read_csv(path)
    _require_columns(frame, {"text", "label"}, path)
    label_map = {
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
    labels = frame["label"].astype(str).str.strip().str.casefold().map(label_map)

    if "group_id" in frame.columns:
        raw_groups = frame["group_id"].astype("string")
        groups = raw_groups.where(raw_groups.notna() & raw_groups.str.strip().ne(""))
        groups = groups.fillna(pd.Series(frame.index, index=frame.index).map(str))
    else:
        groups = pd.Series(frame.index, index=frame.index).map(str)

    return pd.DataFrame(
        {
            "text": frame["text"],
            "label": labels,
            "source": "factvsop_recovered",
            "group_id": "factvsop:" + groups.astype(str),
        }
    )[OUTPUT_COLUMNS]


def _load_all_sources(data_root: Path) -> list[pd.DataFrame]:
    loaders: tuple[Callable[[Path], pd.DataFrame], ...] = (
        load_claimbuster,
        load_claimbuster_ncs_weak_facts,
        load_subj,
        load_what_to_factcheck,
        load_argument_annotations_as_weak_facts,
        load_recovered_viclaim,
        load_recovered_factvsop,
    )
    return [loader(data_root) for loader in loaders]


def clean_and_combine(
    frames: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normalize sources and remove blank, invalid, conflicting, and duplicate rows."""
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("No source datasets were provided")

    combined = pd.concat(frame_list, ignore_index=True)[OUTPUT_COLUMNS].copy()
    input_rows = len(combined)
    combined["text"] = combined["text"].map(_normalize_text)
    combined["label"] = combined["label"].astype("string").str.strip().str.casefold()
    combined["source"] = combined["source"].map(_normalize_text)
    combined["group_id"] = combined["group_id"].map(_normalize_text)

    empty_text = combined["text"].eq("")
    invalid_label = ~combined["label"].isin(VALID_LABELS)
    invalid_metadata = combined["source"].eq("") | combined["group_id"].eq("")
    invalid_rows = empty_text | invalid_label | invalid_metadata
    dropped_empty_or_invalid = int(invalid_rows.sum())
    combined = combined.loc[~invalid_rows].copy()

    combined["_text_key"] = combined["text"].str.casefold()
    labels_per_text = combined.groupby("_text_key")["label"].nunique()
    conflict_keys = set(labels_per_text[labels_per_text > 1].index)
    conflicting_rows = combined["_text_key"].isin(conflict_keys)
    dropped_conflicting = int(conflicting_rows.sum())
    combined = combined.loc[~conflicting_rows].copy()

    combined["_priority"] = combined["source"].map(SOURCE_PRIORITY).fillna(99)
    combined = combined.sort_values(
        ["_priority", "source", "group_id"], kind="stable"
    )
    rows_before_deduplication = len(combined)
    combined = combined.drop_duplicates(subset="_text_key", keep="first")
    dropped_duplicates = rows_before_deduplication - len(combined)

    dataset = combined[OUTPUT_COLUMNS].reset_index(drop=True)
    validate_dataset(dataset)

    report: dict[str, object] = {
        "input_rows": input_rows,
        "output_rows": len(dataset),
        "dropped_empty_or_invalid_rows": dropped_empty_or_invalid,
        "dropped_conflicting_label_rows": dropped_conflicting,
        "dropped_duplicate_text_rows": dropped_duplicates,
        "columns": OUTPUT_COLUMNS,
        "label_counts": dataset["label"].value_counts().sort_index().to_dict(),
        "source_counts": dataset["source"].value_counts().sort_index().to_dict(),
        "excluded_data": {
            "claimbuster_unlabeled_sentences": (
                "Excluded because they have no fact/opinion target label."
            ),
            "mpqa_lexicon": (
                "Excluded because it is a word-level lexicon, not labeled text rows."
            ),
            "viclaim_rows_without_text": (
                "Excluded because a classifier cannot grade a row with no text."
            ),
            "viclaim_tied_scores": (
                "Excluded because equal fact/opinion scores do not define a binary target."
            ),
        },
    }
    return dataset, report


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Fail fast if the output violates the canonical dataset contract."""
    if list(dataset.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Output columns must be exactly {OUTPUT_COLUMNS}")
    if dataset.empty:
        raise ValueError("The normalized dataset is empty")
    if dataset.isna().any().any():
        raise ValueError("The normalized dataset contains empty cells")
    if set(dataset["label"].unique()) != VALID_LABELS:
        raise ValueError("The normalized dataset must contain fact and opinion labels")
    if dataset["text"].str.casefold().duplicated().any():
        raise ValueError("The normalized dataset contains duplicate text")


def build_dataset(data_root: Path = DATA_ROOT) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load every supported source and return one validated dataset."""
    return clean_and_combine(_load_all_sources(data_root))


def write_dataset(
    dataset: pd.DataFrame,
    report: dict[str, object],
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    """Write the dataset and report atomically so partial files are never exposed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_name("normalization_report.json")
    temporary_csv = output.with_suffix(output.suffix + ".tmp")
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")

    dataset.to_csv(temporary_csv, index=False)
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_csv.replace(output)
    temporary_report.replace(report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Source data directory (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    dataset, report = build_dataset(args.data_root)
    report_path = write_dataset(dataset, report, args.output)
    print(
        json.dumps(
            {
                "dataset": str(args.output),
                "report": str(report_path),
                **report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
