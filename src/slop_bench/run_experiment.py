"""CLI driver: generate code + run H1 + validate → SQLite.

Resume-safe: before firing a call, checks `generations` for existing
(prompt_id, model_key, n_sample). If present, reuses the row.

Examples
--------
# 10-prompt calibration across all 5 models
python -m slop_bench.run_experiment --language python --dataset SO_AT --limit 10

# 500-prompt MVP (P0.1)
python -m slop_bench.run_experiment --language python --dataset SO_AT --limit 500

# Full corpus on one model
python -m slop_bench.run_experiment --language python --dataset SO_AT --models sonnet-4-6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .budget import Budget, BudgetExceeded
from .config import LARGE_RUN_CONFIRM, MODELS, PROMPT_FILES
from .db import connect
from .heuristics import h1_extract, normalize
from .orchestrator import Orchestrator
from .prompts import load_prompts, prompt_id
from .validators.npm import NpmValidator
from .validators.pypi import PyPIValidator


# ─── Helpers ─────────────────────────────────────────────────────────────────
def already_generated(conn: sqlite3.Connection, pid: str, model_key: str, n_sample: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM generations WHERE prompt_id=? AND model_key=? AND n_sample=? LIMIT 1",
        (pid, model_key, n_sample),
    ).fetchone()
    return row is not None


def insert_generation(
    conn: sqlite3.Connection,
    *,
    prompt_id_str: str,
    language: str,
    dataset: str,
    model_key: str,
    n_sample: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    prompt_text: str,
    generated_code: str,
    finish_reason: str | None,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> int:
    cur = conn.execute(
        """INSERT INTO generations
        (prompt_id, language, dataset, model_key, n_sample, temperature, top_p,
         max_tokens, prompt_text, generated_code, finish_reason, input_tokens,
         output_tokens, cost_usd)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            prompt_id_str,
            language,
            dataset,
            model_key,
            n_sample,
            temperature,
            top_p,
            max_tokens,
            prompt_text,
            generated_code,
            finish_reason,
            input_tokens,
            output_tokens,
            cost_usd,
        ),
    )
    return cur.lastrowid


def insert_heuristic_hits(
    conn: sqlite3.Connection,
    *,
    generation_id: int,
    heuristic: str,
    raw_names: list[str],
    language: str,
) -> None:
    if not raw_names:
        return
    rows = [
        (generation_id, heuristic, raw, normalize(raw, language))
        for raw in raw_names
    ]
    conn.executemany(
        "INSERT INTO heuristic_hits (generation_id, heuristic, package_raw, package_norm) "
        "VALUES (?,?,?,?)",
        rows,
    )


def insert_api_call(
    conn: sqlite3.Connection,
    *,
    generation_id: int,
    provider: str,
    model_key: str,
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    batch: bool,
    cost_usd: float,
    latency_ms: int,
) -> None:
    conn.execute(
        """INSERT INTO api_calls
        (generation_id, provider, model_key, endpoint, input_tokens, output_tokens,
         batch, cost_usd, latency_ms)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            generation_id,
            provider,
            model_key,
            endpoint,
            input_tokens,
            output_tokens,
            1 if batch else 0,
            cost_usd,
            latency_ms,
        ),
    )


# ─── Runner ──────────────────────────────────────────────────────────────────
@dataclass
class RunPlan:
    language: str
    dataset: str
    limit: int | None
    start_idx: int
    model_keys: list[str]


async def _run_one(
    orchestrator: Orchestrator,
    conn: sqlite3.Connection,
    spec,
    language: str,
    dataset: str,
    row_idx: int,
    prompt: str,
    n_sample: int = 0,
) -> tuple[str, bool]:
    """Generate (if new), extract H1, persist. Returns (status, was_new)."""
    pid = prompt_id(language, dataset, row_idx)
    if already_generated(conn, pid, spec.key, n_sample):
        return "cached", False

    try:
        result = await orchestrator.generate_code(
            spec=spec, language=language, prompt=prompt, n_sample=n_sample
        )
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}:{str(e)[:120]}", False

    gen_id = insert_generation(
        conn,
        prompt_id_str=pid,
        language=language,
        dataset=dataset,
        model_key=spec.key,
        n_sample=n_sample,
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        prompt_text=prompt,
        generated_code=result.generated_code,
        finish_reason=result.finish_reason,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )
    raw_hits = h1_extract(result.generated_code, language)
    insert_heuristic_hits(
        conn,
        generation_id=gen_id,
        heuristic="H1",
        raw_names=raw_hits,
        language=language,
    )
    insert_api_call(
        conn,
        generation_id=gen_id,
        provider=spec.provider,
        model_key=spec.key,
        endpoint="code_gen",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        batch=False,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
    return "ok", True


async def run_async(plan: RunPlan) -> dict:
    budget = Budget.load()
    orchestrator = Orchestrator(budget)

    path = PROMPT_FILES[plan.language][plan.dataset]
    all_prompts = list(load_prompts(path))
    if plan.limit is not None:
        all_prompts = all_prompts[plan.start_idx : plan.start_idx + plan.limit]

    specs = [MODELS[k] for k in plan.model_keys]
    conn = connect()

    print(
        f"[plan] language={plan.language} dataset={plan.dataset} "
        f"prompts={len(all_prompts)} models={[s.key for s in specs]} "
        f"budget_remaining=${budget.remaining:.2f}"
    )

    # Fire all (prompt × model) calls concurrently; per-provider semaphore
    # inside the orchestrator rate-limits actual concurrency.
    tasks: list[asyncio.Task] = []
    for row_idx, text in all_prompts:
        for spec in specs:
            tasks.append(
                asyncio.create_task(
                    _run_one(orchestrator, conn, spec, plan.language, plan.dataset, row_idx, text)
                )
            )

    start_time = time.perf_counter()
    results: list[tuple[str, bool]] = []
    errors: list[str] = []
    with tqdm(total=len(tasks), desc="generating", unit="call") as pbar:
        for fut in asyncio.as_completed(tasks):
            try:
                status, was_new = await fut
            except BudgetExceeded as e:
                print(f"\n[BUDGET EXCEEDED] {e}", file=sys.stderr)
                # Cancel remaining tasks and bail.
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break
            results.append((status, was_new))
            if status.startswith("error"):
                errors.append(status)
            pbar.update(1)
            if pbar.n % 25 == 0:
                pbar.set_postfix_str(f"spend=${budget.total:.3f}")

    elapsed = time.perf_counter() - start_time
    conn.close()
    budget.save()
    summary = {
        "tasks": len(tasks),
        "completed": len(results),
        "ok": sum(1 for s, _ in results if s == "ok"),
        "cached": sum(1 for s, _ in results if s == "cached"),
        "errors": len(errors),
        "error_samples": errors[:5],
        "spend_usd": round(budget.total, 4),
        "per_provider": {k: round(v, 4) for k, v in budget.per_provider.items()},
        "elapsed_s": round(elapsed, 2),
    }
    print("\n[summary] " + json.dumps(summary, indent=2))
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────
def _parse_args() -> RunPlan:
    p = argparse.ArgumentParser(description="Slop-Bench run_experiment driver.")
    p.add_argument("--language", required=True, choices=["python", "javascript"])
    p.add_argument(
        "--dataset", required=True, choices=["SO_AT", "SO_LY", "LLM_AT", "LLM_LY"]
    )
    p.add_argument("--limit", type=int, default=None, help="Max prompts (default: all)")
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS.keys()),
        help="Model keys to run (default: all 5)",
    )
    args = p.parse_args()
    for k in args.models:
        if k not in MODELS:
            sys.exit(f"Unknown model key: {k}. Valid: {list(MODELS)}")
    return RunPlan(
        language=args.language,
        dataset=args.dataset,
        limit=args.limit,
        start_idx=args.start_idx,
        model_keys=args.models,
    )


def main() -> None:
    plan = _parse_args()
    asyncio.run(run_async(plan))


if __name__ == "__main__":
    main()
