from slop_bench.validators.pypi import PyPIValidator


def test_pypi_known_packages_are_valid():
    v = PyPIValidator()
    # Some canonical names that should exist in any full PyPI dump
    for name in ["requests", "numpy", "pandas", "flask", "django"]:
        norm, is_hallucinated = v.classify(name)
        assert not is_hallucinated, f"{name} should be a valid PyPI package"


def test_pypi_fabricated_is_hallucinated():
    v = PyPIValidator()
    # Intentionally made-up name not in PyPI
    norm, is_hallucinated = v.classify("xyzzy-definitely-not-a-real-pkg-12345")
    assert is_hallucinated
    assert norm == "xyzzy-definitely-not-a-real-pkg-12345"


def test_pypi_false_positive_excluded(tmp_path):
    v = PyPIValidator(
        names=frozenset(),
        false_positives=frozenset({"somefp"}),
    )
    _, is_hallucinated = v.classify("somefp")
    assert not is_hallucinated


def test_pypi_whitespace_and_none_ignored():
    v = PyPIValidator()
    assert v.classify("None")[1] is False
    assert v.classify("nan")[1] is False
    assert v.classify("two words")[1] is False
