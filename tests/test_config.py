"""Smoke tests on config — these must pass before any API calls."""
import os

from slop_bench import config


def test_budget_sane():
    assert config.BUDGET_CAP == 160.0
    assert 0 < config.BUDGET_WARN < config.BUDGET_CAP


def test_models_populated():
    assert set(config.MODELS) == {
        "sonnet-4-6",
        "haiku-4-5",
        "gpt-5.4-mini",
        "gemini-2.5-pro",
        "deepseek-v3.2",
    }
    for spec in config.MODELS.values():
        assert spec.input_price > 0
        assert spec.output_price > 0
        assert spec.concurrency > 0


def test_system_prompt_has_language_placeholder_filled():
    out = config.build_code_user_message("Python", "Write a hello world.")
    assert out.startswith("You are a coding assistant that generates Python code.")
    assert "Write a hello world." in out


def test_h2_and_h3_use_system_role():
    msgs = config.build_h2_messages("Python", "print('hi')")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Which Python packages are required" in msgs[1]["content"]

    msgs3 = config.build_h3_messages("JavaScript", "Build a calculator.")
    assert "JavaScript" in msgs3[0]["content"]
    assert msgs3[1]["content"].startswith(
        "What JavaScript packages would be useful"
    )


def test_prompt_files_exist():
    for lang in ("python", "javascript"):
        for name in ("SO_AT", "SO_LY", "LLM_AT", "LLM_LY"):
            p = config.PROMPT_FILES[lang][name]
            assert p.exists(), f"missing prompt file: {p}"
            assert p.stat().st_size > 1000
