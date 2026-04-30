"""Anthropic Batch API wrapper.

Covers the three operations we care about:

* ``submit(spec, language, dataset, prompts)`` — send one or more prompts as
  a single Messages Batch. Deduplicates against existing generations rows so
  you can safely re-submit the same stratified sample; prompts already in the
  DB are skipped.
* ``poll(batch_row_id)`` — check status and counts.
* ``ingest(batch_row_id)`` — stream succeeded results, insert into
  ``generations`` / ``api_calls``, charge the budget per actual token usage
  (batch discount applied).

We intentionally do NOT depend on LiteLLM's batch wrapper — the vendor SDK is
friendlier and future-proof against LiteLLM API changes.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import anthropic

from .budget import Budget
from .config import (
    CODE_GEN_KWARGS,
    MODELS,
    ModelSpec,
    build_code_user_message,
)
from .prompts import prompt_id as _prompt_id


# ── Cached client ──────────────────────────────────────────────────────────
_ANTHROPIC_CLIENT: anthropic.Anthropic | None = None


def _anthropic() -> anthropic.Anthropic:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = anthropic.Anthropic()
    return _ANTHROPIC_CLIENT


# ── Schema helpers ─────────────────────────────────────────────────────────
_BATCHES_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    provider        TEXT NOT NULL,
    model_key       TEXT NOT NULL,
    language        TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    batch_id        TEXT NOT NULL UNIQUE,
    prompt_count    INTEGER NOT NULL,
    status          TEXT NOT NULL,
    submitted_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at    TEXT,
    ingested_at     TEXT,
    cost_usd        REAL,
    raw_metadata    TEXT
);
CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
"""


def ensure_batches_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_BATCHES_SCHEMA)


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class SubmitResult:
    db_id: int
    batch_id: str
    submitted_count: int
    skipped_existing: int


@dataclass
class PollResult:
    db_id: int
    batch_id: str
    status: str           # in_progress | ended | canceling | ...
    succeeded: int
    errored: int
    expired: int
    processing: int
    canceled: int


@dataclass
class IngestResult:
    db_id: int
    inserted_generations: int
    error_results: int
    cost_usd: float


# ── Submit ─────────────────────────────────────────────────────────────────
def _already_done_prompt_ids(
    conn: sqlite3.Connection, model_key: str
) -> set[str]:
    """Return the set of prompt_ids already in generations for this model."""
    rows = conn.execute(
        "SELECT prompt_id FROM generations WHERE model_key=? AND n_sample=0",
        (model_key,),
    ).fetchall()
    return {r["prompt_id"] for r in rows}


def submit(
    conn: sqlite3.Connection,
    *,
    spec: ModelSpec,
    language: str,
    dataset: str,
    prompts: Iterable[tuple[int, str]],
) -> SubmitResult:
    """Submit prompts as one Anthropic Messages Batch. Must be anthropic model."""
    if spec.provider != "anthropic":
        raise ValueError(f"submit() is Anthropic-only (got {spec.provider})")
    ensure_batches_table(conn)

    done = _already_done_prompt_ids(conn, spec.key)
    requests: list[dict[str, Any]] = []
    skipped = 0
    model_id = spec.litellm_id.split("/", 1)[1]  # strip 'anthropic/'
    for row_idx, text in prompts:
        pid = _prompt_id(language, dataset, row_idx)
        if pid in done:
            skipped += 1
            continue
        user_content = build_code_user_message(language.capitalize(), text)
        # Anthropic 4.x rejects temperature+top_p together — omit top_p.
        params = {
            "model": model_id,
            "max_tokens": CODE_GEN_KWARGS["max_tokens"],
            "temperature": CODE_GEN_KWARGS["temperature"],
            "messages": [{"role": "user", "content": user_content}],
        }
        # custom_id must be ≤64 chars and match [a-zA-Z0-9_-].
        # We replace ':' with '-' because dataset names contain '_' (e.g. SO_AT)
        # and a naive "-" replacement would make parsing ambiguous.
        requests.append(
            {
                "custom_id": pid.replace(":", "-"),
                "params": params,
            }
        )

    if not requests:
        # No work to do. Record an empty batch row for bookkeeping.
        cur = conn.execute(
            """INSERT INTO batches
               (provider, model_key, language, dataset, batch_id, prompt_count, status,
                completed_at)
               VALUES (?, ?, ?, ?, ?, 0, 'skipped_all_cached', ?)""",
            (
                spec.provider,
                spec.key,
                language,
                dataset,
                f"skipped-{spec.key}-{language}-{dataset}-{int(time.time())}",
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
        return SubmitResult(
            db_id=cur.lastrowid,
            batch_id="skipped",
            submitted_count=0,
            skipped_existing=skipped,
        )

    batch = _anthropic().messages.batches.create(requests=requests)
    cur = conn.execute(
        """INSERT INTO batches
           (provider, model_key, language, dataset, batch_id, prompt_count, status,
            raw_metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            spec.provider,
            spec.key,
            language,
            dataset,
            batch.id,
            len(requests),
            batch.processing_status,
            json.dumps({"created_at": str(batch.created_at)}),
        ),
    )
    conn.commit()
    return SubmitResult(
        db_id=cur.lastrowid,
        batch_id=batch.id,
        submitted_count=len(requests),
        skipped_existing=skipped,
    )


# ── Poll ───────────────────────────────────────────────────────────────────
def poll(conn: sqlite3.Connection, db_id: int) -> PollResult:
    row = conn.execute(
        "SELECT batch_id, status FROM batches WHERE id=?", (db_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"batches.id={db_id} not found")
    if row["batch_id"] == "skipped":
        return PollResult(db_id, "skipped", "skipped_all_cached", 0, 0, 0, 0, 0)

    batch = _anthropic().messages.batches.retrieve(row["batch_id"])
    rc = batch.request_counts

    # Persist status changes
    conn.execute(
        "UPDATE batches SET status=? WHERE id=?",
        (batch.processing_status, db_id),
    )
    if batch.ended_at is not None:
        conn.execute(
            "UPDATE batches SET completed_at=? WHERE id=?",
            (str(batch.ended_at), db_id),
        )
    conn.commit()

    return PollResult(
        db_id=db_id,
        batch_id=batch.id,
        status=batch.processing_status,
        succeeded=rc.succeeded,
        errored=rc.errored,
        expired=rc.expired,
        processing=rc.processing,
        canceled=rc.canceled,
    )


# ── Ingest ─────────────────────────────────────────────────────────────────
def ingest(
    conn: sqlite3.Connection, db_id: int, *, budget: Budget
) -> IngestResult:
    """Stream successful results, insert into generations + api_calls, charge budget."""
    row = conn.execute(
        "SELECT batch_id, provider, model_key, language, dataset FROM batches WHERE id=?",
        (db_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"batches.id={db_id} not found")
    if row["batch_id"] == "skipped":
        return IngestResult(db_id=db_id, inserted_generations=0, error_results=0, cost_usd=0.0)

    already_ingested = conn.execute(
        "SELECT ingested_at FROM batches WHERE id=?", (db_id,)
    ).fetchone()["ingested_at"]
    if already_ingested:
        # idempotent
        return IngestResult(db_id=db_id, inserted_generations=0, error_results=0, cost_usd=0.0)

    spec = MODELS[row["model_key"]]
    language = row["language"]
    dataset = row["dataset"]

    inserted = 0
    errors = 0
    total_cost = 0.0
    # Precompute already-done set to be robust against double-ingest races.
    done_pids = _already_done_prompt_ids(conn, spec.key)

    client = _anthropic()
    for item in client.messages.batches.results(row["batch_id"]):
        custom_id = item.custom_id
        result = item.result
        # Reconstruct prompt_id from custom_id. Modern format uses '-' separator
        # ("python-SO_AT-500"); old format used '_' which was ambiguous because
        # dataset names contain '_'. We fall back to rsplit logic for old batches.
        if "-" in custom_id:
            parts = custom_id.split("-", 2)
        else:
            # Legacy encoding: "python_SO_AT_500" → rsplit the row_idx off first.
            lang_dataset, _, idx_str = custom_id.rpartition("_")
            if not idx_str:
                errors += 1
                continue
            lang, _, dataset = lang_dataset.partition("_")
            parts = [lang, dataset, idx_str]
        if len(parts) != 3 or not parts[2].isdigit():
            errors += 1
            continue
        pid = f"{parts[0]}:{parts[1]}:{parts[2]}"
        if pid in done_pids:
            continue  # safety

        if result.type != "succeeded":
            errors += 1
            continue
        message = result.message
        text = ""
        for block in message.content:
            if getattr(block, "type", None) == "text":
                text += block.text
        text = (text or "").strip()
        usage = message.usage
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cost = budget.project(
            provider=spec.provider,
            input_tokens=in_tok,
            output_tokens=out_tok,
            input_price=spec.input_price,
            output_price=spec.output_price,
            batch=True,
            discount=spec.batch_discount,
        )
        total_cost += cost

        # row_idx is parts[2]; we store prompt_id (already built). prompt_text
        # is backfilled later by scripts/backfill_prompt_text.py.
        conn.execute(
            """INSERT OR IGNORE INTO generations
            (prompt_id, language, dataset, model_key, n_sample, temperature, top_p,
             max_tokens, prompt_text, generated_code, finish_reason, input_tokens,
             output_tokens, cost_usd)
            VALUES (?,?,?,?,0,?,?,?,'',?,?,?,?,?)""",
            (
                pid,
                language,
                dataset,
                spec.key,
                CODE_GEN_KWARGS["temperature"],
                CODE_GEN_KWARGS["top_p"],  # stored even though unused at inference
                CODE_GEN_KWARGS["max_tokens"],
                text,
                message.stop_reason,
                in_tok,
                out_tok,
                cost,
            ),
        )
        gen_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO api_calls
               (generation_id, provider, model_key, endpoint, input_tokens, output_tokens,
                batch, cost_usd, latency_ms)
               VALUES (?,?,?,'code_gen',?,?,1,?,NULL)""",
            (
                gen_id,
                spec.provider,
                spec.key,
                in_tok,
                out_tok,
                cost,
            ),
        )
        inserted += 1

    budget.charge(provider=spec.provider, cost_usd=total_cost)
    conn.execute(
        "UPDATE batches SET ingested_at=?, cost_usd=? WHERE id=?",
        (datetime.now(UTC).isoformat(), total_cost, db_id),
    )
    conn.commit()
    return IngestResult(
        db_id=db_id,
        inserted_generations=inserted,
        error_results=errors,
        cost_usd=total_cost,
    )


def list_batches(conn: sqlite3.Connection) -> list[dict]:
    ensure_batches_table(conn)
    rows = conn.execute(
        "SELECT id, provider, model_key, language, dataset, batch_id, prompt_count, "
        "status, submitted_at, completed_at, ingested_at, cost_usd FROM batches "
        "ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]
