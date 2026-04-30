"""Prompt dataset loader.

Spracklen ships two JSONL schemas:
  * SO_AT / SO_LY: dict-with-single-key — `{"0": "<prompt text>"}`
  * LLM_AT / LLM_LY: bare string — `"<prompt text>"`

`load_prompts(path)` yields `(row_index, prompt_text)` tuples, schema-agnostic.
`prompt_id(language, dataset, row_index)` gives a stable key for DB rows.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def load_prompts(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                yield idx, obj
            elif isinstance(obj, dict):
                # SO files: {"0": "..."} — take the single value
                if not obj:
                    continue
                yield idx, next(iter(obj.values()))
            else:
                raise ValueError(
                    f"{path}: unexpected JSONL object type {type(obj).__name__} at line {idx}"
                )


def prompt_id(language: str, dataset: str, row_index: int) -> str:
    return f"{language}:{dataset}:{row_index}"


def count_prompts(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def stratified_sample(
    path: Path, *, fraction: float | None = None, size: int | None = None
) -> list[tuple[int, str]]:
    """Return the first-N prompts by row index, deterministic by file order.

    If `size` is given, returns exactly that many. If `fraction` is given,
    returns round(total * fraction) prompts. Exactly one must be provided.

    Rationale: "first-N" is trivially reproducible without a random seed and
    lets a late-joining model (e.g. Gemini once billing activates) hit the
    same sample as everyone else.
    """
    if (fraction is None) == (size is None):
        raise ValueError("provide exactly one of fraction or size")
    all_rows = list(load_prompts(path))
    if size is None:
        size = round(len(all_rows) * fraction)  # type: ignore[arg-type]
    return all_rows[:size]
