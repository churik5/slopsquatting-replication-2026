import json
from pathlib import Path

from slop_bench.config import PROMPT_FILES
from slop_bench.prompts import count_prompts, load_prompts, prompt_id


def test_prompt_id_format():
    assert prompt_id("python", "SO_AT", 17) == "python:SO_AT:17"


def test_load_so_at():
    path = PROMPT_FILES["python"]["SO_AT"]
    rows = list(load_prompts(path))
    assert rows
    first_idx, first_text = rows[0]
    assert first_idx == 0
    assert first_text.startswith("Title:")
    assert "Body:" in first_text


def test_load_llm_at_bare_string():
    path = PROMPT_FILES["python"]["LLM_AT"]
    rows = list(load_prompts(path))
    assert rows
    _, first = rows[0]
    assert isinstance(first, str)
    assert first  # non-empty


def test_counts_expected(tmp_path: Path):
    # Sanity: Python totals
    counts = {
        name: count_prompts(PROMPT_FILES["python"][name])
        for name in ("SO_AT", "SO_LY", "LLM_AT", "LLM_LY")
    }
    assert sum(counts.values()) == 19084
