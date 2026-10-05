#!/usr/bin/env python3
"""Guard selected performance documentation claim boundaries."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WS = r"\s+"

DOC_MARKERS = {
    "README.md": (
        "make bench-canonical-report-freshness",
        "selected `bench_refactor_csc` row",
        "nos4.mtx --repeat 1",
        "reviewed Linux and macOS hosted performance\n  lanes",
        "still without a timing threshold\n  or portable performance claim",
        "Promoting that selected canonical benchmark to a timing threshold would\n  "
        "require a documented stable runner class",
        "future timing-threshold promotion would need stable-runner,\ncompiler, "
        "repeat, warmup, variance, baseline, threshold, retained-artifact,\n"
        "and non-claim evidence",
        "Locally generated benchmark,\nsentinel, and normalized report-index artifacts stay under ignored `build/`\npaths and are not hosted CI proof by themselves",
        "Only the explicitly reviewed\nhosted lanes described below promote selected uploaded artifacts to hosted\nfreshness evidence",
    ),
    "INSTALL.md": (
        "Linux/macOS selected performance freshness",
        "`make bench-canonical-report-freshness`",
        "Linux/macOS hosted selected lanes",
        "No portable performance, timing threshold, release benchmark, platform parity, "
        "package/ABI proof, broad package-manager distribution, or state-of-the-art claim;",
        "timing-threshold promotion requires stable runner, compiler, repeat, "
        "warmup, variance, baseline, threshold, retained-artifact, and non-claim evidence",
    ),
    "benchmarks/README.md": (
        "The reviewed Linux and macOS hosted selected-performance lanes",
        "support_tier=hosted_selected",
        "claim_boundary=hosted_selected_threshold_free",
        "runner_context=github-actions-macos-latest",
        "baseline=n/a",
        "threshold=n/a",
        "backend_context=n/a",
        "warmup=none_configured",
        "variance=not_computed_single_sample",
        "`methodology_notes` includes `not_portable_performance_claim`",
        "not as portable\n  speed evidence or broad benchmark publication",
        "require stable runner-class, compiler, repeat, warmup, variance, baseline,\n  "
        "threshold, retained-artifact, and non-claim evidence",
    ),
    "docs/maintainer_guide.md": (
        "sprint168-selected-performance-freshness",
        "sprint202-macos-selected-performance-freshness",
        "build/bench-reports/canonical/bench_refactor_csc.csv",
        "build/bench-reports/canonical/index.tsv",
        "build/bench-reports/canonical/manifest.txt",
        "SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest",
        "Sprint 202 artifacts",
        "Linux/macOS hosted selected lanes are threshold-free methodology evidence",
        "should remain `baseline=n/a`, `threshold=n/a`, and `status=measurement`",
        "requires `methodology_notes` to include\n    "
        "`not_portable_performance_claim`",
        "rejects methodology-note tokens that promote portable performance",
        "stable runner-class evidence, compiler\n    evidence, repeat policy, warmup policy, variance rule, baseline\n    "
        "provenance, threshold value, retained-artifact policy, and updated\n    "
        "non-claim evidence",
        "Selected performance repair workflow:",
        "python3 tests/test_selected_report_targets_manifest.py",
        "hosted threshold-free freshness remain separate policy\n    surfaces",
    ),
    "tests/corpus/README.md": (
        "SRT-BENCH-REFACTOR-CSC-NOS4",
        "bench_refactor_csc",
        "tests/data/suitesparse/nos4.mtx --repeat 1",
        "baseline=n/a",
        "threshold=n/a",
        "limited to Linux and macOS for this same selected row",
        "no portable\nperformance, release benchmark, algorithmic superiority, platform parity,\npackage/ABI, broad package-manager distribution, or state-of-the-art claim",
    ),
    "tests/corpus/schemas/report_index_fields.md": (
        "SRT-BENCH-REFACTOR-CSC-NOS4",
        "bench_refactor_csc",
        "tests/data/suitesparse/nos4.mtx --repeat 1",
        "status=measurement",
        "baseline=n/a",
        "threshold=n/a",
        "covers Linux and macOS\n  for that exact selected row only",
        "not create pass/fail benchmark proof",
    ),
}

FORBIDDEN_PATTERNS = (
    re.compile(
        rf"selected{WS}performance{WS}(?:proves|guarantees)"
        rf"{WS}portable{WS}performance",
        re.I,
    ),
    re.compile(
        rf"selected{WS}performance{WS}(?:proves|is){WS}state-of-the-art",
        re.I,
    ),
    re.compile(
        rf"hosted{WS}selected{WS}performance{WS}"
        rf"(?:is|acts{WS}as|has|sets|provides|creates|defines|establishes|guarantees)"
        rf"{WS}a{WS}timing{WS}gate",
        re.I,
    ),
    re.compile(
        rf"selected{WS}(?:canonical{WS})?benchmark{WS}"
        rf"(?:is|acts{WS}as|has|sets|provides|creates|defines|establishes|guarantees){WS}"
        rf"a{WS}(?:hosted{WS})?timing(?:-(?:{WS})?|{WS})threshold",
        re.I,
    ),
    re.compile(
        rf"selected{WS}performance{WS}"
        rf"(?:is|acts{WS}as|has|sets|provides|creates|defines|establishes|guarantees){WS}"
        rf"a{WS}(?:hosted{WS})?timing(?:-(?:{WS})?|{WS})threshold",
        re.I,
    ),
    re.compile(
        rf"selected{WS}(?:performance|(?:canonical{WS})?benchmark){WS}"
        rf"(?:is|acts{WS}as|has|sets|provides|creates|defines|establishes|guarantees){WS}"
        rf"linux(?:/|{WS}and{WS})macos{WS}timing{WS}parity",
        re.I,
    ),
    re.compile(
        rf"sprint168-selected-performance-freshness{WS}"
        rf"(?:proves|guarantees){WS}performance{WS}superiority",
        re.I,
    ),
    re.compile(
        rf"bench-canonical-report-freshness{WS}(?:proves|guarantees)"
        rf"{WS}speedup",
        re.I,
    ),
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


def test_missing_future_threshold_prerequisite_marker_fails_clearly() -> None:
    relative_path = "benchmarks/README.md"
    marker = (
        "require stable runner-class, compiler, repeat, warmup, variance, baseline,\n  "
        "threshold, retained-artifact, and non-claim evidence"
    )
    text = read_doc(relative_path).replace(marker, "", 1)
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "benchmarks/README.md missing selected performance marker",
    )


def test_missing_maintainer_repair_marker_fails_clearly() -> None:
    relative_path = "docs/maintainer_guide.md"
    marker = "Selected performance repair workflow:"
    text = read_doc(relative_path).replace(marker, "", 1)
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "docs/maintainer_guide.md missing selected performance marker",
    )


def test_forbidden_hosted_timing_gate_overclaim_fails_clearly() -> None:
    relative_path = "benchmarks/README.md"
    claims = (
        "Hosted selected performance is a timing gate.",
        "Hosted selected performance acts as a timing gate.",
        "Hosted selected performance has a timing gate.",
        "Hosted selected performance sets a timing gate.",
        "Hosted selected performance provides a timing gate.",
        "Hosted selected performance creates a timing gate.",
        "Hosted selected performance defines a timing gate.",
        "Hosted selected performance establishes a timing gate.",
        "Hosted selected performance guarantees a timing gate.",
    )
    for claim in claims:
        text = read_doc(relative_path) + f"\n{claim}\n"
        assert_raises_with(
            lambda text=text: validate_docs({relative_path: text}),
            "unsupported selected performance claim",
        )


def test_forbidden_selected_timing_threshold_overclaim_fails_clearly() -> None:
    relative_path = "benchmarks/README.md"
    claims = (
        "The selected canonical benchmark is a timing-threshold.",
        "The selected canonical benchmark provides a timing threshold.",
        "The selected canonical benchmark provides a\ntiming threshold.",
        "The selected canonical benchmark provides a timing\nthreshold.",
        "The selected canonical benchmark provides a timing-\nthreshold.",
        "The selected canonical benchmark provides a hosted timing threshold.",
        "The selected benchmark provides a timing threshold.",
        "The selected benchmark sets a timing threshold.",
        "The selected canonical benchmark guarantees a timing threshold.",
        "The selected canonical benchmark has a timing threshold.",
        "The selected canonical benchmark sets a timing threshold.",
        "The selected performance is a timing threshold.",
        "The selected performance has a timing threshold.",
        "The selected performance provides a timing-\nthreshold.",
        "The selected performance provides a hosted timing threshold.",
        "The selected performance sets a timing threshold.",
        "The selected performance guarantees a timing threshold.",
    )
    for claim in claims:
        text = read_doc(relative_path) + f"\n{claim}\n"
        assert_raises_with(
            lambda text=text: validate_docs({relative_path: text}),
            "unsupported selected performance claim",
        )


def test_forbidden_selected_linux_macos_parity_overclaim_fails_clearly() -> None:
    relative_path = "README.md"
    claims = (
        "The selected performance guarantees linux/macos timing parity.",
        "The selected performance guarantees Linux and macOS timing parity.",
        "The selected performance guarantees Linux and macOS\ntiming parity.",
        "The selected canonical benchmark guarantees Linux and macOS timing parity.",
        "The selected benchmark guarantees Linux and macOS timing parity.",
    )
    positive_verbs = (
        "is",
        "acts as",
        "has",
        "sets",
        "provides",
        "creates",
        "defines",
        "establishes",
        "guarantees",
    )
    expanded_claims = claims + tuple(
        f"The selected canonical benchmark {verb} Linux and macOS timing parity."
        for verb in positive_verbs
    )
    for claim in expanded_claims:
        text = read_doc(relative_path) + f"\n{claim}\n"
        assert_raises_with(
            lambda text=text: validate_docs({relative_path: text}),
            "unsupported selected performance claim",
        )


def main() -> int:
    test_current_docs_validate_selected_performance_claims()
    test_missing_required_marker_fails_clearly()
    test_forbidden_selected_performance_overclaim_fails_clearly()
    test_missing_threshold_free_policy_marker_fails_clearly()
    test_missing_future_threshold_prerequisite_marker_fails_clearly()
    test_missing_maintainer_repair_marker_fails_clearly()
    test_forbidden_hosted_timing_gate_overclaim_fails_clearly()
    test_forbidden_selected_timing_threshold_overclaim_fails_clearly()
    test_forbidden_selected_linux_macos_parity_overclaim_fails_clearly()
    print("test-selected-performance-docs: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
