#!/usr/bin/env python3
"""Guard Sprint 215 external-baseline wording in public and maintainer docs."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md": ("was blocked before either solver ran", "day11-evidence-and-claim-matrix.md"),
    "benchmarks/README.md": ("blocked before either solver ran", "not numerical or"),
    "docs/solver_selection.md": ("preflight was blocked before either solver", "source-controlled-helper"),
    "docs/maintainer_guide.md": ("status_reason=schema_validation_unavailable", "Only a validated"),
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
        for marker in markers:
            if marker not in content:
                raise AssertionError(f"{path}: missing external-baseline boundary {marker!r}")
        match = UNSUPPORTED.search(" ".join(content.split()))
        if match:
            raise AssertionError(f"{path}: unsupported external-baseline claim {match.group(0)!r}")


class ExternalBaselineClaimDocsTests(unittest.TestCase):
    def test_current_docs(self) -> None:
        validate_docs()

    def test_missing_blocker_boundary(self) -> None:
        path = "README.md"
        content = (ROOT / path).read_text(encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "missing external-baseline boundary"):
            validate_docs({path: content.replace("was blocked before either solver ran", "was prepared", 1)})

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

    def test_explicit_nonclaim_is_allowed(self) -> None:
        path = "README.md"
        content = (ROOT / path).read_text(encoding="utf-8")
        for claim in (
            "The Sprint 215 blueprint does not prove state-of-the-art performance.",
            "The blocked pilot is not state-of-the-art.",
        ):
            with self.subTest(claim=claim):
                validate_docs({path: content + "\n" + claim + "\n"})


if __name__ == "__main__":
    unittest.main()
