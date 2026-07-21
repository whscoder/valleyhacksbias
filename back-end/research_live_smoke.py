"""Opt-in live smoke test for the Responses web-search research integration.

Run only against a non-production key:
    FACTGPT_RUN_LIVE_RESEARCH_SMOKE=1 ../.venv/bin/python research_live_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import home


CASES = (
    (
        "Public-health statistic",
        "The World Health Organization says tobacco kills more than eight million people each year.",
    ),
    (
        "Government result",
        "The U.S. Census Bureau reported that the 2020 census counted more than 331 million people.",
    ),
    (
        "Attributed reporting",
        'The agency said "the bridge reopened on Monday," after inspectors completed repairs.',
    ),
)


async def main() -> int:
    if os.getenv("FACTGPT_RUN_LIVE_RESEARCH_SMOKE") != "1":
        print(
            "Refusing paid live requests. Set FACTGPT_RUN_LIVE_RESEARCH_SMOKE=1 "
            "to run this smoke test.",
            file=sys.stderr,
        )
        return 2

    failed = 0
    for title, text in CASES:
        raw = await home.researcher_ai(
            text,
            title,
            candidate_claim_count=1,
        )
        if "error" in raw:
            failed += 1
            print(f"FAIL {title}: {raw.get('error_code', 'unknown_error')}")
            continue
        try:
            result = home.validate_ai_research(
                raw,
                candidate_claim_count=1,
                total_factual_characters=len(text),
            )
        except Exception as exc:  # Keep this an executable smoke diagnostic.
            failed += 1
            print(f"FAIL {title}: invalid research payload ({type(exc).__name__})")
            continue
        print(f"PASS {title}: {len(result.claims)} checked claim(s)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
