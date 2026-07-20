"""Paid, fixed-seed evaluation for every item routed to OpenAI review."""

from __future__ import annotations

import asyncio
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

from home import (
    FactOpinionItem,
    load_fact_opinion_classifier,
    local_review_reasons,
    resolve_fact_opinion_items,
)


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fact_opinion"
    / "processed"
    / "fact_opinion.csv"
)
SAMPLE_SIZE = 100
RANDOM_STATE = 2028
MINIMUM_SCORE = 0.85
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "testingcode"
    / "reports"
    / "openai-fallback-latest.json"
)


def held_out_routed_sample(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Recreate the untouched split and select balanced rows policy routes to review."""
    frame = pd.read_csv(DATA_PATH)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=2026)
    _, test_index = next(
        splitter.split(frame["text"], frame["label"], groups=frame["group_id"])
    )
    test = frame.iloc[test_index].copy()

    model = load_fact_opinion_classifier()
    probabilities = np.exp(model.predict_log_proba(test["text"]))
    test["local_confidence"] = probabilities.max(axis=1)
    classes = np.asarray(model.classes_, dtype=str)
    test["local_label"] = classes[np.argmax(probabilities, axis=1)]
    test["review_reasons"] = [
        local_review_reasons(
            str(text),
            str(label),
            float(confidence) >= float(model.confidence_threshold_),
        )
        for text, label, confidence in zip(
            test["text"], test["local_label"], test["local_confidence"]
        )
    ]
    routed = test[
        test["review_reasons"].map(bool)
        & test["text"].str.len().between(1, 5_000)
    ]

    per_label = sample_size // 2
    selected = []
    for label in ("fact", "opinion"):
        candidates = routed[routed["label"] == label]
        selected.append(
            candidates.sample(
                n=min(per_label, len(candidates)), random_state=RANDOM_STATE
            )
        )
    sample = pd.concat(selected)
    if len(sample) < sample_size:
        remaining = routed.drop(index=sample.index)
        sample = pd.concat(
            [
                sample,
                remaining.sample(
                    n=min(sample_size - len(sample), len(remaining)),
                    random_state=RANDOM_STATE,
                ),
            ]
        )
    return sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def selection_summary(sample: pd.DataFrame) -> dict:
    reason_counts: dict[str, int] = {}
    for reasons in sample["review_reasons"]:
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "rows": len(sample),
        "expected_label_counts": sample["label"].value_counts().to_dict(),
        "local_label_counts": sample["local_label"].value_counts().to_dict(),
        "review_reason_counts": reason_counts,
    }


async def evaluate(
    *,
    sample_size: int = SAMPLE_SIZE,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    dry_run: bool = False,
) -> int:
    sample = held_out_routed_sample(sample_size)
    if len(sample) != sample_size:
        raise RuntimeError(
            f"Only {len(sample)} eligible routed rows were available; "
            f"the acceptance run requires exactly {sample_size}."
        )
    summary = selection_summary(sample)
    print("Selection summary:", json.dumps(summary, indent=2, sort_keys=True))
    if dry_run:
        print("Dry run only: no OpenAI review requests were made.")
        return 0

    items = [
        FactOpinionItem(id=f"eval-{index:03d}", text=row.text)
        for index, row in sample.iterrows()
    ]
    result = await resolve_fact_opinion_items(items, "Held-out classifier evaluation")
    predictions = [
        item.final_prediction.label or "unresolved" for item in result.items
    ]
    expected = sample["label"].tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(
        expected,
        predictions,
        labels=["fact", "opinion"],
        zero_division=0,
    )
    macro_f1 = float(np.mean(f1))
    accuracy = float(accuracy_score(expected, predictions))
    passed = macro_f1 >= MINIMUM_SCORE and bool(np.all(recall >= MINIMUM_SCORE))

    rows = []
    for (_, row), item, prediction in zip(sample.iterrows(), result.items, predictions):
        rows.append(
            {
                "id": item.id,
                "text": row.text,
                "expected": row.label,
                "predicted": prediction,
                "local_label": row.local_label,
                "local_confidence": float(row.local_confidence),
                "review_reasons": list(row.review_reasons),
                "decision_source": item.final_prediction.source,
                "explanation": item.final_prediction.explanation,
                "opinion_excerpts": item.final_prediction.opinion_excerpts,
            }
        )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATA_PATH),
        "random_state": RANDOM_STATE,
        "minimum_score": MINIMUM_SCORE,
        "selection": summary,
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class_precision": dict(zip(["fact", "opinion"], map(float, precision))),
            "per_class_recall": dict(zip(["fact", "opinion"], map(float, recall))),
            "per_class_f1": dict(zip(["fact", "opinion"], map(float, f1))),
            "unresolved_count": predictions.count("unresolved"),
            "mixed_count": predictions.count("mixed"),
            "passed": passed,
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"Rows evaluated: {len(sample)}")
    print(f"Unresolved/invalid outputs: {predictions.count('unresolved')}")
    print(f"Unresolved rate: {predictions.count('unresolved') / len(sample):.1%}")
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(expected, predictions, digits=3, zero_division=0))
    print("Per-class precision:", dict(zip(["fact", "opinion"], precision)))
    print("Per-class recall:", dict(zip(["fact", "opinion"], recall)))
    print("Per-class F1:", dict(zip(["fact", "opinion"], f1)))
    print(f"Macro F1: {macro_f1:.3f}")
    print("Production acceptance:", "PASS" if passed else "FAIL")
    print(f"Saved report: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the routed sample without making paid OpenAI requests.",
    )
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            evaluate(
                sample_size=args.sample_size,
                output_path=args.output,
                dry_run=args.dry_run,
            )
        )
    )
