"""Fill generations.prompt_text for rows left empty by batch ingest.

Batch ingest skips copying the original prompt (not returned by the Anthropic
Batch results stream). This script reads the dataset file once per (lang,
dataset) and backfills via prompt_id.

Idempotent: only updates rows where prompt_text = ''.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slop_bench.config import PROMPT_FILES  # noqa: E402
from slop_bench.db import connect  # noqa: E402
from slop_bench.prompts import load_prompts  # noqa: E402


def main() -> None:
    conn = connect()
    to_fill = conn.execute(
        "SELECT id, prompt_id, language, dataset FROM generations WHERE prompt_text = '' LIMIT 100000"
    ).fetchall()
    print(f"Rows needing backfill: {len(to_fill)}")
    cache: dict[tuple[str, str], dict[int, str]] = {}
    updated = 0
    for row in to_fill:
        pid = row["prompt_id"]
        key = (row["language"], row["dataset"])
        if key not in cache:
            cache[key] = {
                idx: text
                for idx, text in load_prompts(PROMPT_FILES[row["language"]][row["dataset"]])
            }
        try:
            _, _, row_idx = pid.split(":")
            text = cache[key].get(int(row_idx), "")
        except Exception:  # noqa: BLE001
            text = ""
        conn.execute(
            "UPDATE generations SET prompt_text=? WHERE id=?", (text, row["id"])
        )
        updated += 1
    conn.commit()
    print(f"Updated {updated} rows.")


if __name__ == "__main__":
    main()
