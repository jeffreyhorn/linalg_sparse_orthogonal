#!/usr/bin/env python3
"""Recompute a Sprint 225 decision from retained raw evidence before citing it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from check_refactor_csc_timing import (
    DEFAULT_BASELINE, ROOT, EvidenceError, file_json, verify_report_against_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="extracted raw candidate directory or collector ZIP")
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--expected-candidate-commit")
    parser.add_argument("--verify-checkout", action="store_true")
    args = parser.parse_args()
    if args.verify_checkout and not args.expected_candidate_commit:
        parser.error("--verify-checkout requires --expected-candidate-commit")
    try:
        report = file_json(args.report, "S225_REPORT_INVALID")
        code = verify_report_against_evidence(
            report, args.baseline_root, args.candidate, args.expected_candidate_commit,
            ROOT if args.verify_checkout else None,
        )
    except EvidenceError as exc:
        print(f"refactor-csc-report: {exc.rule}: {exc.detail}", file=sys.stderr)
        return 2
    print(f"refactor-csc-report: verified {report['primary_rule']} from raw evidence")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
