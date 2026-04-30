"""P0.4: live pypi.org / registry.npmjs.org HEAD verification of a random
sample of "hallucinated" packages, to ground-truth the master-list heuristic.

Writes reports/spot_check.md with a per-sample table and summary stats.

Polite client: custom User-Agent, httpx HTTP/2, diskcache to avoid hammering.
Pauses for 60s if >5 consecutive 5xx returns from either registry.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import diskcache
import httpx

from .config import CACHE_DIR, USER_AGENT
from .db import connect
from .validators.npm import NpmValidator
from .validators.pypi import PyPIValidator


PYPI_URL = "https://pypi.org/pypi/{name}/json"
NPM_URL = "https://registry.npmjs.org/{name}"


@dataclass
class SpotCheckHit:
    name_raw: str
    name_norm: str
    registry: str
    language: str
    model_key: str
    prompt_id: str
    heuristic: str
    live_status: int | None
    ok: bool                        # True if master-list hallucination confirmed
    note: str = ""


def _http_client() -> httpx.Client:
    return httpx.Client(
        http2=True,
        headers={"User-Agent": USER_AGENT},
        timeout=10.0,
        follow_redirects=True,
    )


def _get_cache() -> diskcache.Cache:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(CACHE_DIR / "registry_head"))


def _head_once(url: str, client: httpx.Client, cache) -> int:
    if url in cache:
        return cache[url]
    try:
        resp = client.head(url)
        # npm registry redirects HEAD to GET for some names; accept either.
        status = resp.status_code
    except httpx.HTTPError as e:
        status = -1
    cache[url] = status
    return status


def sample_hallucinated(n: int = 100, seed: int = 17) -> list[SpotCheckHit]:
    """Sample n random hallucinated package references (H1 ∪ IMPORTS)."""
    conn = connect()
    pypi = PyPIValidator()
    npm = NpmValidator()

    rows = conn.execute(
        """SELECT g.id AS gen_id, g.language, g.model_key, g.prompt_id,
                  h.heuristic, h.package_raw, h.package_norm
           FROM generations g
           JOIN heuristic_hits h ON h.generation_id = g.id
           WHERE h.heuristic IN ('H1','IMPORTS')"""
    ).fetchall()

    flagged: list[SpotCheckHit] = []
    for r in rows:
        if r["language"] == "python":
            norm, is_h = pypi.classify(r["package_raw"])
            registry = "pypi"
        else:
            norm, is_h = npm.classify(r["package_raw"])
            registry = "npm"
        if is_h:
            flagged.append(
                SpotCheckHit(
                    name_raw=r["package_raw"],
                    name_norm=norm,
                    registry=registry,
                    language=r["language"],
                    model_key=r["model_key"],
                    prompt_id=r["prompt_id"],
                    heuristic=r["heuristic"],
                    live_status=None,
                    ok=False,
                )
            )

    rng = random.Random(seed)
    rng.shuffle(flagged)
    return flagged[:n]


def live_verify(hits: list[SpotCheckHit]) -> list[SpotCheckHit]:
    """Attach live_status from the registry. Mutates and returns."""
    cache = _get_cache()
    consecutive_5xx_pypi = 0
    consecutive_5xx_npm = 0
    with _http_client() as client:
        for i, h in enumerate(hits):
            if h.registry == "pypi":
                url = PYPI_URL.format(name=h.name_norm)
            else:
                url = NPM_URL.format(name=h.name_norm)
            status = _head_once(url, client, cache)
            h.live_status = status

            # A 404 confirms the name really does not exist.
            # A 200 means the master-list was wrong (registry has it now).
            if status == 404:
                h.ok = True
                h.note = "confirmed hallucination (404)"
            elif status == 200:
                h.ok = False
                h.note = "false positive: registry has this name"
            elif status == -1:
                h.ok = False
                h.note = "network error"
            elif status >= 500:
                h.ok = False
                h.note = f"registry {status}"
                if h.registry == "pypi":
                    consecutive_5xx_pypi += 1
                else:
                    consecutive_5xx_npm += 1
                if consecutive_5xx_pypi > 5 or consecutive_5xx_npm > 5:
                    print("Pausing 60s: registry 5xx streak")
                    time.sleep(60)
                    consecutive_5xx_pypi = 0
                    consecutive_5xx_npm = 0
            else:
                h.ok = False
                h.note = f"status {status}"

            # Small delay to be polite (npm is aggressive about rate limiting).
            if i % 5 == 4:
                time.sleep(0.5)
    return hits


def write_report(hits: list[SpotCheckHit], path: Path = Path("reports/spot_check.md")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P0.4 — Spot check of flagged hallucinations",
        "",
        f"Sample size: {len(hits)}",
        "",
        "| # | Language | Model | Heuristic | Raw | Norm | Live HTTP | Verdict | Note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    summary = Counter()
    for i, h in enumerate(hits, 1):
        verdict = "✓ confirmed" if h.ok else "✗ false pos" if h.live_status == 200 else "? unknown"
        summary[verdict] += 1
        lines.append(
            f"| {i} | {h.language} | `{h.model_key}` | {h.heuristic} | `{h.name_raw}` | "
            f"`{h.name_norm}` | {h.live_status} | {verdict} | {h.note} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for verdict, n in summary.most_common():
        lines.append(f"- {verdict}: {n}")
    path.write_text("\n".join(lines) + "\n")


def run(n: int = 100) -> None:
    hits = sample_hallucinated(n=n)
    print(f"Sampled {len(hits)} hallucinations")
    hits = live_verify(hits)
    write_report(hits)
    print("Wrote reports/spot_check.md")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    run(n=args.n)
