"""Streaming monitor for P0.3 Anthropic batches + sync progress.

Each iteration:
  1. Poll all non-ended Anthropic batches. If a batch just ended, ingest it
     immediately and print one ENDED line (triggers a notification).
  2. Print one HEARTBEAT line every iteration with aggregate status.
  3. Exit when all batches ended AND sync log shows completion.

Use via Monitor(); filter matches ENDED|HEARTBEAT|SYNC_DONE|ALL_DONE|ERR.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slop_bench.batch import ingest, list_batches, poll  # noqa: E402
from slop_bench.budget import Budget  # noqa: E402
from slop_bench.db import connect  # noqa: E402


SYNC_LOG = Path("/tmp/p0_3_sync.log")


def sync_done() -> bool:
    """Very conservative: does /tmp/p0_3_sync.log show a finished summary?"""
    if not SYNC_LOG.exists():
        return False
    try:
        data = SYNC_LOG.read_bytes().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return False
    # run_experiment prints "[summary]" JSON at the end of each run_async call.
    # There are 8 run_async calls (8 files), so 8 summaries means sync done.
    return data.count("[summary]") >= 8


def main() -> None:
    conn = connect()
    budget = Budget.load()
    ingested_ids: set[int] = set()
    start = time.time()
    iter_i = 0

    while True:
        iter_i += 1
        batches = list_batches(conn)
        not_ended = 0
        newly_ended: list[str] = []

        for b in batches:
            if b["status"] == "skipped_all_cached":
                continue
            if b["status"] == "ended" and b["id"] in ingested_ids:
                continue
            try:
                p = poll(conn, b["id"])
            except Exception as e:  # noqa: BLE001
                print(f"ERR poll db={b['id']} {type(e).__name__}: {e}", flush=True)
                continue

            if p.status == "ended" and b["id"] not in ingested_ids:
                try:
                    r = ingest(conn, p.db_id, budget=budget)
                    ingested_ids.add(b["id"])
                    newly_ended.append(
                        f"ENDED db={p.db_id} {b['model_key']}/{b['language']}/{b['dataset']} "
                        f"succ={p.succeeded}/{b['prompt_count']} "
                        f"ingested={r.inserted_generations} cost=${r.cost_usd:.4f}"
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"ERR ingest db={b['id']} {type(e).__name__}: {e}", flush=True)
            elif p.status != "ended":
                not_ended += 1

        for line in newly_ended:
            print(line, flush=True)

        elapsed = int(time.time() - start)
        print(
            f"HEARTBEAT iter={iter_i} elapsed={elapsed}s "
            f"not_ended={not_ended}/16 budget=${budget.total:.2f}",
            flush=True,
        )

        sd = sync_done()
        if not_ended == 0:
            if sd:
                print("ALL_DONE batches+sync complete", flush=True)
                return
            else:
                print("BATCHES_DONE waiting for sync", flush=True)
        elif sd:
            print("SYNC_DONE waiting for batches", flush=True)

        time.sleep(60)


if __name__ == "__main__":
    main()
