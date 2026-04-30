from pathlib import Path

import pytest

from slop_bench.budget import Budget, BudgetExceeded


def test_charge_accumulates(tmp_path: Path):
    b = Budget(cap=1.0, warn_at=0.8, path=tmp_path / "spend.json")
    b.charge(provider="openai", cost_usd=0.30)
    b.charge(provider="openai", cost_usd=0.40)
    b.charge(provider="anthropic", cost_usd=0.10)
    assert b.total == pytest.approx(0.80)
    assert b.remaining == pytest.approx(0.20)
    assert b.per_provider == {"openai": 0.70, "anthropic": 0.10}


def test_cap_enforced(tmp_path: Path):
    b = Budget(cap=1.0, path=tmp_path / "spend.json")
    b.charge(provider="openai", cost_usd=0.95)
    with pytest.raises(BudgetExceeded):
        b.charge(provider="openai", cost_usd=0.10)


def test_persistence_roundtrip(tmp_path: Path):
    p = tmp_path / "spend.json"
    b = Budget(cap=2.0, path=p)
    b.charge(provider="openai", cost_usd=0.25)
    b2 = Budget.load(p)
    assert b2.per_provider == {"openai": 0.25}
    assert b2.total == pytest.approx(0.25)


def test_projection():
    b = Budget(cap=10.0)
    # 1M input tokens at $3/M = $3; 0.5M output at $15/M = $7.5
    assert b.project(
        provider="anthropic",
        input_tokens=1_000_000,
        output_tokens=500_000,
        input_price=3.0,
        output_price=15.0,
    ) == pytest.approx(10.5)


def test_projection_batch_discount():
    b = Budget(cap=10.0)
    assert b.project(
        provider="anthropic",
        input_tokens=1_000_000,
        output_tokens=0,
        input_price=3.0,
        output_price=15.0,
        batch=True,
        discount=0.5,
    ) == pytest.approx(1.5)
