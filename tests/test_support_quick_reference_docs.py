#!/usr/bin/env python3
"""Guard Sprint 205 support, quick-reference, and diagnostics docs."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


DOC_MARKERS = {
    "README.md": (
        "docs/cookbook.md#problem-shape-quick-reference",
        "problem-local residuals",
        "run-local convergence fields",
        "local measurement artifacts",
        "INSTALL.md#support-readiness-matrix",
    ),
    "docs/cookbook.md": (
        "## Problem-Shape Quick Reference",
        "[support/readiness matrix](../INSTALL.md#support-readiness-matrix)",
        "solver-selection guide",
        "[README repeated-run direct workflow](../README.md#repeated-run-direct-workflow)",
        "[`example_analysis`](../examples/README.md#repeated-run-direct-example_analysis)",
        "[README runtime/backend controls](../README.md#runtime-and-backend-controls)",
        "[algorithm docs](algorithm.md)",
        "Local and selected hosted freshness do not prove portable performance",
        "Static-first install only; no shared-library, dynamic ABI, or package-manager support.",
    ),
    "examples/README.md": (
        "## Route Interpretation",
        "Local build-tree usage only; not an install or package-manager proof.",
        "support status remains owned by [INSTALL.md#support-readiness-matrix]",
        "Benchmark output is local or selected evidence, not portable performance proof.",
        "Examples do not widen package-manager, shared-library, dynamic ABI, Windows, hosted API, release, or state-of-the-art claims.",
    ),
    "docs/tutorial.md": (
        "cookbook.md#problem-shape-quick-reference",
        "support/readiness matrix",
        "problem-local residual",
        "run-local convergence fields",
        "QR-local rank",
        "SVD-local rank",
        "fresh`, `stale`, `skip`, or `defer`",
    ),
    "docs/solver_selection.md": (
        "problem-local residual",
        "run-local stagnation",
        "QR-local rank",
        "SVD-local rank",
        "Ritz residual for the requested eigenpairs",
    ),
    "benchmarks/README.md": (
        "threshold-free measurement artifacts",
        "not as support or timing proof",
        "optional data or prerequisites were unavailable by policy",
        "not as passing evidence",
    ),
    "docs/api_reference.md": (
        "selected local current-output proof",
        "The generated HTML tree is local-only generated output.",
        "not a hosted or source-controlled publication surface",
    ),
    "docs/maintainer_guide.md": (
        "### Support truth and quick-reference routing",
        "### Diagnostics vocabulary routing",
        "First-use docs should prefer \"problem-local residual\"",
        "Do not rewrite `skip` or `defer` as pass/fail evidence",
        "current generated output for the selected gate",
    ),
}


FORBIDDEN_PATTERNS = (
    re.compile(r"package-manager support is (?:available|supported|provided)", re.I),
    re.compile(r"shared-library (?:support|packaging) is (?:available|supported|provided)", re.I),
    re.compile(r"dynamic ABI (?:support|compatibility) is (?:available|supported|provided)", re.I),
    re.compile(r"generated API (?:is|docs are) hosted", re.I),
    re.compile(r"selected performance (?:proves|guarantees) portable performance", re.I),
    re.compile(r"benchmark output (?:proves|guarantees) portable performance", re.I),
    re.compile(r"state-of-the-art (?:support|performance|proof) is (?:available|provided)", re.I),
    re.compile(r"Windows selected freshness is promoted", re.I),
)


def read_doc(relative_path: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and relative_path in overrides:
        return overrides[relative_path]
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def has_marker(text: str, marker: str) -> bool:
    if marker in text:
        return True
    normalized_text = re.sub(r"\s+", " ", text)
    normalized_marker = re.sub(r"\s+", " ", marker)
    return normalized_marker in normalized_text


def validate_docs(overrides: dict[str, str] | None = None) -> None:
    corpus = []
    for relative_path, markers in DOC_MARKERS.items():
        text = read_doc(relative_path, overrides)
        corpus.append(text)
        for marker in markers:
            if not has_marker(text, marker):
                raise AssertionError(f"{relative_path} missing Sprint 205 docs marker {marker!r}")

    combined = "\n".join(corpus)
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(combined)
        if match:
            raise AssertionError(f"unsupported Sprint 205 documentation claim: {match.group(0)!r}")


def assert_raises_with(fn, expected: str) -> None:
    try:
        fn()
    except AssertionError as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}") from exc
        return
    raise AssertionError(f"expected failure containing {expected!r}")


def test_current_docs_validate_support_quick_reference_claims() -> None:
    validate_docs()


def test_missing_quick_reference_route_fails_clearly() -> None:
    relative_path = "README.md"
    text = read_doc(relative_path).replace("docs/cookbook.md#problem-shape-quick-reference", "")
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "README.md missing Sprint 205 docs marker",
    )


def test_missing_support_truth_route_fails_clearly() -> None:
    relative_path = "docs/cookbook.md"
    text = read_doc(relative_path).replace(
        "[support/readiness matrix](../INSTALL.md#support-readiness-matrix)",
        "support matrix",
    )
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "docs/cookbook.md missing Sprint 205 docs marker",
    )


def test_missing_examples_route_interpretation_fails_clearly() -> None:
    relative_path = "examples/README.md"
    text = read_doc(relative_path).replace(
        "Local build-tree usage only; not an install or package-manager proof.",
        "Local usage path.",
        1,
    )
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "examples/README.md missing Sprint 205 docs marker",
    )


def test_missing_diagnostics_vocabulary_fails_clearly() -> None:
    relative_path = "docs/tutorial.md"
    text = read_doc(relative_path).replace("run-local convergence fields", "convergence status", 1)
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "docs/tutorial.md missing Sprint 205 docs marker",
    )


def test_forbidden_package_manager_overclaim_fails_clearly() -> None:
    relative_path = "README.md"
    text = read_doc(relative_path) + "\nPackage-manager support is available.\n"
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "unsupported Sprint 205 documentation claim",
    )


def test_forbidden_portable_performance_overclaim_fails_clearly() -> None:
    relative_path = "benchmarks/README.md"
    text = read_doc(relative_path) + "\nBenchmark output proves portable performance.\n"
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "unsupported Sprint 205 documentation claim",
    )


def test_forbidden_hosted_api_overclaim_fails_clearly() -> None:
    relative_path = "docs/api_reference.md"
    text = read_doc(relative_path) + "\nGenerated API docs are hosted.\n"
    assert_raises_with(
        lambda: validate_docs({relative_path: text}),
        "unsupported Sprint 205 documentation claim",
    )


def main() -> int:
    test_current_docs_validate_support_quick_reference_claims()
    test_missing_quick_reference_route_fails_clearly()
    test_missing_support_truth_route_fails_clearly()
    test_missing_examples_route_interpretation_fails_clearly()
    test_missing_diagnostics_vocabulary_fails_clearly()
    test_forbidden_package_manager_overclaim_fails_clearly()
    test_forbidden_portable_performance_overclaim_fails_clearly()
    test_forbidden_hosted_api_overclaim_fails_clearly()
    print("test-support-quick-reference-docs: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
