"""P0.3 driver — stratified ~19.5k prompts × 5 models.

Phase 1: submit Anthropic batches for Sonnet + Haiku (16 batches).
Phase 2: run sync for gpt-5.4-mini + deepseek-v3.2 (extends existing P0.1 results).
Phase 3: poll Anthropic batches, ingest as they complete.
Phase 4: offline IMPORTS extraction over new generations.
Phase 5: compute rates and emit headline table.

Idempotent: re-running picks up where it left off. `--phase {submit,sync,poll,ingest,all}`
lets you run pieces independently.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slop_bench.batch import (  # noqa: E402
    ensure_batches_table,
    ingest,
    list_batches,
    poll,
    submit,
)
from slop_bench.budget import Budget  # noqa: E402
from slop_bench.config import MODELS, PROMPT_FILES  # noqa: E402
from slop_bench.db import connect  # noqa: E402
from slop_bench.prompts import stratified_sample  # noqa: E402
from slop_bench.run_experiment import RunPlan, run_async  # noqa: E402


ANTHROPIC_KEYS = ["sonnet-4-6", "haiku-4-5"]
SYNC_KEYS = ["gpt-5.4-mini", "deepseek-v3.2"]  # Gemini added later once billing on
LANGUAGES = ["python", "javascript"]
DATASETS = ["SO_AT", "SO_LY", "LLM_AT", "LLM_LY"]
SAMPLE_FRACTION = 0.488  # ≈19,500 across all 8 files


def phase_submit() -> None:
    conn = connect()
    ensure_batches_table(conn)
    print(f"\n=== Phase 1: submit Anthropic batches (fraction={SAMPLE_FRACTION}) ===")
    total_submitted = 0
    total_skipped = 0
    for model_key, lang, dataset in itertools.product(ANTHROPIC_KEYS, LANGUAGES, DATASETS):
        # Skip if a batch for this exact (model, language, dataset) already submitted.
        existing = conn.execute(
            "SELECT id, batch_id, status, prompt_count FROM batches "
            "WHERE model_key=? AND language=? AND dataset=?",
            (model_key, lang, dataset),
        ).fetchall()
        if existing:
            ids = [r["batch_id"] for r in existing]
            print(f"  [exists] {model_key} {lang}/{dataset}: {len(existing)} batch(es) {ids}")
            continue
        prompts = stratified_sample(PROMPT_FILES[lang][dataset], fraction=SAMPLE_FRACTION)
        res = submit(
            conn,
            spec=MODELS[model_key],
            language=lang,
            dataset=dataset,
            prompts=prompts,
        )
        total_submitted += res.submitted_count
        total_skipped += res.skipped_existing
        print(
            f"  [OK]  {model_key:<11} {lang:<10} {dataset:<7}  "
            f"submitted={res.submitted_count} skipped_cached={res.skipped_existing}  "
            f"batch_id={res.batch_id}"
        )
    print(f"Total submitted: {total_submitted}, skipped-cached: {total_skipped}")


def phase_sync() -> None:
    print("\n=== Phase 2: sync runs for GPT-5.4-mini + DeepSeek V3.2 ===")
    for lang in LANGUAGES:
        for dataset in DATASETS:
            path = PROMPT_FILES[lang][dataset]
            full = list(stratified_sample(path, fraction=SAMPLE_FRACTION))
            limit = len(full)
            plan = RunPlan(
                language=lang,
                dataset=dataset,
                limit=limit,
                start_idx=0,
                model_keys=SYNC_KEYS,
            )
            print(f"\n--- {lang}/{dataset}  limit={limit}  models={SYNC_KEYS} ---")
            asyncio.run(run_async(plan))


def phase_poll(max_wait_sec: int = 3600, poll_every: int = 60) -> None:
    print("\n=== Phase 3: poll Anthropic batches ===")
    conn = connect()
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        batches = list_batches(conn)
        pending = [b for b in batches if b["status"] not in {"ended", "skipped_all_cached"}]
        if not pending:
            print("All batches ended.")
            return
        print(f"\n[{time.strftime('%H:%M:%S')}] {len(pending)} batch(es) still processing:")
        for b in pending:
            try:
                p = poll(conn, b["id"])
                print(
                    f"  db={p.db_id:>3} {b['model_key']:<11} {b['language']:<10} "
                    f"{b['dataset']:<7} status={p.status:<12} "
                    f"done={p.succeeded}/{b['prompt_count']}  err={p.errored}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  db={b['id']} poll error: {e}")
        time.sleep(poll_every)
    print("Timed out.")


def phase_ingest() -> None:
    print("\n=== Phase 4: ingest completed batches ===")
    conn = connect()
    budget = Budget.load()
    batches = list_batches(conn)
    for b in batches:
        if b["status"] not in {"ended", "skipped_all_cached"}:
            continue
        if b["ingested_at"]:
            continue
        print(f"  ingesting db={b['id']} ({b['model_key']} {b['language']}/{b['dataset']})...")
        res = ingest(conn, b["id"], budget=budget)
        print(
            f"    inserted={res.inserted_generations} errors={res.error_results} "
            f"cost=${res.cost_usd:.4f}"
        )
    print(f"Budget so far: ${budget.total:.2f} / ${budget.cap:.2f}")


def phase_report() -> None:
    """Offline IMPORTS pass + rate summary."""
    print("\n=== Phase 5: offline IMPORTS + rates ===")
    import subprocess

    subprocess.run(
        [sys.executable, "scripts/extract_imports_offline.py"], check=True
    )
    subprocess.run(
        [sys.executable, "scripts/preliminary_rates.py"], check=True
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["submit", "sync", "poll", "ingest", "report", "all"],
        default="all",
    )
    ap.add_argument("--max-wait", type=int, default=3600)
    ap.add_argument("--poll-every", type=int, default=60)
    args = ap.parse_args()

    if args.phase in {"submit", "all"}:
        phase_submit()
    if args.phase in {"sync", "all"}:
        phase_sync()
    if args.phase in {"poll", "all"}:
        phase_poll(max_wait_sec=args.max_wait, poll_every=args.poll_every)
    if args.phase in {"ingest", "all"}:
        phase_ingest()
    if args.phase == "report":
        phase_report()


if __name__ == "__main__":
    main()
