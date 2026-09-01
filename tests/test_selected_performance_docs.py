#!/usr/bin/env python3
"""Guard selected performance documentation claim boundaries."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DOC_MARKERS = {
    "README.md": (
        "make bench-canonical-report-freshness",
        "selected `bench_refactor_csc` row",
        "nos4.mtx --repeat 1",
        "without a timing threshold or portable performance claim",
        "not hosted CI proof, package proof, ABI proof, runtime-loader\nproof, external-library parity, OpenMP speedup evidence, backend superiority\nevidence, or state-of-the-art evidence",
    ),
    "benchmarks/README.md": (
        "The reviewed Linux hosted selected-performance lane",
        "support_tier=hosted_selected",
        "claim_boundary=hosted_selected_threshold_free",
        "baseline=n/a",
        "threshold=n/a",
        "not as portable\n  speed evidence or broad benchmark publication",
    ),
    "docs/maintainer_guide.md": (
        "sprint168-selected-performance-freshness",
        "build/bench-reports/canonical/bench_refactor_csc.csv",
        "build/bench-reports/canonical/index.tsv",
        "build/bench-reports/canonical/manifest.txt",
        "should remain `baseline=n/a`, `threshold=n/a`, and `status=measurement`",
        "hosted threshold-free freshness remain separate policy\n    surfaces",
    ),
    "tests/corpus/README.md": (
        "SRT-BENCH-REFACTOR-CSC-NOS4",
        "bench_refactor_csc",
        "tests/data/suitesparse/nos4.mtx --repeat 1",
        "baseline=n/a",
        "threshold=n/a",
        "no portable\nperformance, release benchmark, algorithmic superiority, platform parity,\npackage/ABI, or state-of-the-art claim",
    ),
    "tests/corpus/schemas/report_index_fields.md": (
        "SRT-BENCH-REFACTOR-CSC-NOS4",
        "bench_refactor_csc",
        "tests/data/suitesparse/nos4.mtx --repeat 1",
        "status=measurement",
        "baseline=n/a",
        "threshold=n/a",
        "not create pass/fail benchmark proof",
    ),
}

FORBIDDEN_PATTERNS = (
    re.compile(r"selected performance (?:proves|guarantees) portable performance", re.I),
    re.compile(r"selected performance (?:proves|is) state-of-the-art", re.I),
    re.compile(r"hosted selected performance (?:is|acts as) a timing gate", re.I),
    re.compile(
        r"sprint168-selected-performance-freshness (?:proves|guarantees) performance superiority",
        re.I,
    ),
    re.compile(r"bench-canonical-report-freshness (?:proves|guarantees) speedup", re.I),
)


def read_doc(relative_path: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and relative_path in overrides:
        return overrides[relative_path]
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def validate_docs(overrides: dict[str, str] | None = None) -> None:
    corpus = []
    for relative_path, markers in DOC_MARKERS.items():
        text = read_doc(relative_path, overrides)
        corpus.append(text)
        for marker in markers:
            if marker not in text:
                raise AssertionError(f"{relative_path} missing selected performance marker {marker!r}")

    combined = "\n".join(corpus)
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(combined)
        if match:
            raise AssertionError(
                f"unsupported selected performance claim found: {match.group(0)!r}"
            )


def assert_raises_with(fn, expected: str) -> None:
    try:
        fn()
    except AssertionError as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}") from exc
        return
    raise AssertionError(f"expected failure containing {expected!r}")


def test_current_docs_validate_selected_performance_claims() -> None:
    validate_docs()


def test_missing_required_marker_fails_clearly() -> None:
    relative_path = "docs/maintainer_guide.md"
    text = read_doc(relative_path).replace("sprint168-selected-performance-freshness", "", 1)
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "docs/maintainer_guide.md missing selected performance marker",
    )


def test_forbidden_selected_performance_overclaim_fails_clearly() -> None:
    relative_path = "README.md"
    text = (
        read_doc(relative_path)
        + "\nThe selected performance proves portable performance.\n"
    )
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "unsupported selected performance claim",
    )


def test_missing_threshold_free_policy_marker_fails_clearly() -> None:
    relative_path = "tests/corpus/schemas/report_index_fields.md"
    text = read_doc(relative_path).replace("threshold=n/a", "threshold=200.0", 1)
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "tests/corpus/schemas/report_index_fields.md missing selected performance marker "
        "'threshold=n/a'",
    )


def test_forbidden_hosted_timing_gate_overclaim_fails_clearly() -> None:
    relative_path = "benchmarks/README.md"
    text = read_doc(relative_path) + "\nHosted selected performance is a timing gate.\n"
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "unsupported selected performance claim",
    )


def main() -> int:
    test_current_docs_validate_selected_performance_claims()
    test_missing_required_marker_fails_clearly()
    test_forbidden_selected_performance_overclaim_fails_clearly()
    test_missing_threshold_free_policy_marker_fails_clearly()
    test_forbidden_hosted_timing_gate_overclaim_fails_clearly()
    print("test-selected-performance-docs: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
