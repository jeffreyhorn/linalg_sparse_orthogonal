#!/usr/bin/env python3
"""Guard Sprint 215 external-baseline wording in public and evidence docs."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md": ("was blocked before either solver ran", "day11-evidence-and-claim-matrix.md"),
    "INSTALL.md": ("External baseline comparison", "blocked before either solver ran"),
    "benchmarks/README.md": (
        "blocked before either solver ran",
        "not numerical or",
        "provisional `attempt.json` is not an accepted",
    ),
    "docs/solver_selection.md": ("preflight was blocked before either solver", "source-controlled-helper"),
    "docs/maintainer_guide.md": ("status_reason=schema_validation_unavailable", "Only a validated"),
    "tests/corpus/README.md": (
        "not an observed external comparison",
        "not `comparison_status=pass`",
    ),
    "tests/corpus/schemas/report_index_fields.md": (
        "not accepted `evidence.json`",
        "two-sided correctness result",
    ),
    "docs/planning/EPIC_19/PROJECT_PLAN.md": (
        "blocked before either solver ran",
        "provisional `attempt.json` are not accepted",
    ),
}
SUBJECT = (
    r"(?:sprint\s+215\s+(?:blueprint|design|pilot)|"
    r"(?:(?:selected|blocked)\s+)?cholmod\s+(?:pilot|preflight)|"
    r"(?:selected|blocked)\s+(?:pilot|preflight)|"
    r"(?:a\s+)?single\s+(?:cholmod\s+)?measurement)"
)
POSITIVE = r"(?:is|proves?|guarantees?|establishes?|demonstrates?|confirms?|shows?)"
BROAD = (
    r"(?:state[- ]of[- ]the[- ]art|portable\s+(?:speed|performance)|"
    r"(?:linux\s*(?:and|/)\s*macos|cross[- ]platform|platform)\s*"
    r"(?:timing\s*)?parity|cross[- ]platform\s+(?:performance|speed))"
)
UNSUPPORTED = re.compile(rf"\b{SUBJECT}\s+{POSITIVE}\s+(?!(?:not|never)\b)[^.!?]{{0,80}}?{BROAD}\b", re.I)


def validate_docs(overrides: dict[str, str] | None = None) -> None:
    for path, markers in REQUIRED.items():
        content = (overrides or {}).get(path)
        if content is None:
            content = (ROOT / path).read_text(encoding="utf-8")
        normalized = " ".join(content.split())
        for marker in markers:
            if " ".join(marker.split()) not in normalized:
                raise AssertionError(f"{path}: missing external-baseline boundary {marker!r}")
        match = UNSUPPORTED.search(normalized)
        if match:
            raise AssertionError(f"{path}: unsupported external-baseline claim {match.group(0)!r}")


class ExternalBaselineClaimDocsTests(unittest.TestCase):
    def test_current_docs(self) -> None:
        validate_docs()

    def test_missing_blocker_boundary(self) -> None:
        for path, marker in (
            ("README.md", "was blocked before either solver ran"),
            ("INSTALL.md", "blocked before either solver ran"),
        ):
            with self.subTest(path=path):
                content = (ROOT / path).read_text(encoding="utf-8")
                with self.assertRaisesRegex(AssertionError, "missing external-baseline boundary"):
                    validate_docs({path: content.replace(marker, "was prepared", 1)})

    def test_unsupported_promotions_on_every_surface(self) -> None:
        claims = (
            "The Sprint 215 blueprint proves state-of-the-art performance.",
            "The blocked CHOLMOD pilot guarantees Linux and macOS timing parity.",
            "A single CHOLMOD measurement establishes portable speed.",
            "The selected CHOLMOD pilot demonstrates cross-platform performance.",
            "The blocked pilot is state-of-the-art.",
            "The selected pilot proves platform parity.",
        )
        for path in REQUIRED:
            content = (ROOT / path).read_text(encoding="utf-8")
            for claim in claims:
                with self.subTest(path=path, claim=claim):
                    with self.assertRaisesRegex(AssertionError, "unsupported external-baseline claim"):
                        validate_docs({path: content + "\n" + claim + "\n"})

    def test_missing_provisional_boundary(self) -> None:
        for path, marker in (
            ("benchmarks/README.md", "provisional `attempt.json` is not an accepted"),
            ("tests/corpus/README.md", "not an observed external comparison"),
            ("tests/corpus/schemas/report_index_fields.md", "not accepted `evidence.json`"),
            ("docs/planning/EPIC_19/PROJECT_PLAN.md", "provisional `attempt.json` are not accepted"),
        ):
            with self.subTest(path=path):
                content = (ROOT / path).read_text(encoding="utf-8")
                with self.assertRaisesRegex(AssertionError, "missing external-baseline boundary"):
                    validate_docs({path: content.replace(marker, "is a result", 1)})

    def test_explicit_nonclaim_is_allowed(self) -> None:
        for path in REQUIRED:
            content = (ROOT / path).read_text(encoding="utf-8")
            for claim in (
                "The Sprint 215 blueprint does not prove state-of-the-art performance.",
                "The blocked pilot is not state-of-the-art.",
            ):
                with self.subTest(path=path, claim=claim):
                    validate_docs({path: content + "\n" + claim + "\n"})


if __name__ == "__main__":
    unittest.main()
