"""Build reports/month1_table.md from runs.db.

For each model, computes (per Python / JavaScript / Combined):
  * Package-level hallucination rate (H1 ∪ IMPORTS union)
  * 95 % bootstrap CI over generations (n_bootstrap=1000)
  * Number of generations, validated packages, hallucinated packages
  * Per-model $ spent

Then maps to Spracklen '24 closest analog and reports Delta.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slop_bench.db import connect  # noqa: E402
from slop_bench.validators.npm import NpmValidator  # noqa: E402
from slop_bench.validators.pypi import PyPIValidator  # noqa: E402


# Mapping per brief
ANALOG_2024 = {
    "sonnet-4-6": ("Claude-3.5-Sonnet (2024)", None),         # rate TBD
    "haiku-4-5": ("CodeLlama-7B Python (2024)", 26.12),       # flagged as rough lower-bound
    "gpt-5.4-mini": ("GPT-4 Turbo", 3.59),
    "gemini-2.5-pro": ("commercial avg 2024", 5.20),
    "deepseek-v3.2": ("DeepSeek-Coder 33B", None),             # rate TBD
}


def compute(conn, language: str, validator) -> dict[str, dict]:
    """Return per-model stats with 95% CI via bootstrap."""
    rows = conn.execute(
        """SELECT g.id AS gen_id, g.model_key,
                  h.heuristic, h.package_raw, h.package_norm
           FROM generations g
           LEFT JOIN heuristic_hits h ON h.generation_id = g.id
           WHERE g.language = ?""",
        (language,),
    ).fetchall()

    # For each (model, gen_id) collect the set of {pkg_raw}, classify each.
    per_model_gen: dict[str, dict[int, list[tuple[str, bool]]]] = defaultdict(lambda: defaultdict(list))
    all_gens: dict[str, set[int]] = defaultdict(set)

    for r in rows:
        all_gens[r["model_key"]].add(r["gen_id"])
        if r["heuristic"] in ("H1", "IMPORTS") and r["package_raw"]:
            _, is_h = validator.classify(r["package_raw"])
            per_model_gen[r["model_key"]][r["gen_id"]].append((r["package_raw"], is_h))

    stats = {}
    for model, gen_ids in all_gens.items():
        gen_ids = list(gen_ids)
        total_pkgs = 0
        halluc_pkgs = 0
        for gid in gen_ids:
            seen_raw: set[str] = set()
            for raw, is_h in per_model_gen[model].get(gid, []):
                if raw in seen_raw:
                    continue
                seen_raw.add(raw)
                total_pkgs += 1
                if is_h:
                    halluc_pkgs += 1
        rate = 100 * halluc_pkgs / total_pkgs if total_pkgs else 0.0

        # Bootstrap 95% CI over generations (resample with replacement)
        rng = random.Random(42)
        n_b = 1000
        rates = []
        for _ in range(n_b):
            sampled = [rng.choice(gen_ids) for _ in range(len(gen_ids))]
            t = 0; h = 0
            for gid in sampled:
                seen_raw: set[str] = set()
                for raw, is_hal in per_model_gen[model].get(gid, []):
                    if raw in seen_raw:
                        continue
                    seen_raw.add(raw)
                    t += 1
                    if is_hal:
                        h += 1
            rates.append(100 * h / t if t else 0.0)
        rates.sort()
        ci_lo = rates[int(0.025 * n_b)]
        ci_hi = rates[int(0.975 * n_b)]

        stats[model] = {
            "n_gens": len(gen_ids),
            "n_pkgs": total_pkgs,
            "n_halluc": halluc_pkgs,
            "rate": rate,
            "ci": (ci_lo, ci_hi),
        }
    return stats


def spend_per_model(conn) -> dict[str, float]:
    rows = conn.execute(
        "SELECT model_key, SUM(cost_usd) AS c FROM generations GROUP BY model_key"
    ).fetchall()
    return {r["model_key"]: r["c"] or 0.0 for r in rows}


def main() -> None:
    conn = connect()
    pypi = PyPIValidator()
    npm = NpmValidator()

    py_stats = compute(conn, "python", pypi)
    js_stats = compute(conn, "javascript", npm)
    spend = spend_per_model(conn)

    all_models = sorted(set(py_stats) | set(js_stats))

    lines = [
        "# Month 1 headline — Slop-Bench replication vs extension",
        "",
        "Heuristic: **H1 ∪ IMPORTS** (pip/npm-install regex ∪ ast/tree-sitter import extraction).",
        "Bootstrap 95 % CI over generations, n_bootstrap=1000.",
        "",
        "| Model (dated snapshot) | Py rate % (CI) | JS rate % (CI) | Combined % | "
        "Spracklen '24 analog & rate | Delta | # gens | # validated | # halluc | $ spent |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in all_models:
        py = py_stats.get(model, {"rate": float("nan"), "ci": (float("nan"),)*2, "n_gens":0, "n_pkgs":0, "n_halluc":0})
        js = js_stats.get(model, {"rate": float("nan"), "ci": (float("nan"),)*2, "n_gens":0, "n_pkgs":0, "n_halluc":0})
        # Combined
        cp = py["n_pkgs"] + js["n_pkgs"]
        ch = py["n_halluc"] + js["n_halluc"]
        combined = 100 * ch / cp if cp else 0.0
        n_gens = py["n_gens"] + js["n_gens"]
        analog_name, analog_rate = ANALOG_2024.get(model, ("—", None))
        if analog_rate is None:
            delta = "—"
            analog_cell = f"{analog_name} (TBD)"
        else:
            delta = f"{combined - analog_rate:+.2f} pp"
            analog_cell = f"{analog_name} ({analog_rate:.2f}%)"

        def fmt_rate(s):
            if s["n_pkgs"] == 0:
                return "n/a"
            return f"{s['rate']:.2f} ({s['ci'][0]:.2f}–{s['ci'][1]:.2f})"

        lines.append(
            f"| `{model}` | {fmt_rate(py)} | {fmt_rate(js)} | {combined:.2f} "
            f"| {analog_cell} | {delta} | {n_gens} | {cp} | {ch} | ${spend.get(model,0):.2f} |"
        )

    lines.append("")
    lines.append("## Per-model interpretation")
    lines.append("")
    for model in all_models:
        py = py_stats.get(model); js = js_stats.get(model)
        if py is None or js is None:
            continue
        lines.append(f"### `{model}`")
        lines.append("")
        lines.append(
            f"- Python: {py['n_halluc']}/{py['n_pkgs']} packages hallucinated across "
            f"{py['n_gens']} generations → {py['rate']:.2f} %"
        )
        lines.append(
            f"- JS: {js['n_halluc']}/{js['n_pkgs']} packages hallucinated across "
            f"{js['n_gens']} generations → {js['rate']:.2f} %"
        )
        lines.append("")

    Path("reports/month1_table.md").write_text("\n".join(lines) + "\n")
    print("Wrote reports/month1_table.md")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
