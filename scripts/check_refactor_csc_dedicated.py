#!/usr/bin/env python3
"""Recompute provisional s225-v4 timing results from three offline raw bundles."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import median

import check_refactor_csc_pair as paired
import check_refactor_csc_timing as common


ROOT = Path(__file__).resolve().parents[1]
POLICY = "s225-v4"


def decide(root: Path, candidate_commit: str, checkout: Path,
           collection_id: str) -> tuple[dict, int]:
    report: dict[str, object] = {
        "schema_version": 1, "policy_version": POLICY,
        "baseline_commit": paired.BASELINE_COMMIT,
        "candidate_commit": candidate_commit, "collection_id": collection_id,
        "status": "blocked", "primary_rule": "S225_V4_MISSING",
        "claim_scope": "one_dedicated_refactor_csc_nos4_cell_on_one_intel_mac",
        "non_claims": common.NON_CLAIMS, "jobs": [], "between_job_spread": None,
        "reference_median_ratio": None, "independent_review": "pending",
        "detail": "raw dedicated evidence missing",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", candidate_commit):
            paired.reject("invalid candidate source commit")
        if not collection_id.isdecimal():
            paired.reject("invalid collection identity")
        current = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if current != candidate_commit:
            paired.reject("candidate checkout differs from claimed commit")
        jobs = []
        for replica in (1, 2, 3):
            path = root / f"sprint225-v4-pair-{collection_id}-1-{replica}"
            if not path.exists():
                path = path.with_suffix(".zip")
            result, manifest_hash = paired.check_pair(
                path, replica, candidate_commit, checkout, collection_id, "1",
                dedicated=True,
            )
            result["member_manifest_sha256"] = manifest_hash
            jobs.append(result)
        first = jobs[0]
        for previous, current_job in zip(jobs, jobs[1:]):
            previous_end = datetime.fromisoformat(previous["ended_at_utc"])
            current_start = datetime.fromisoformat(current_job["started_at_utc"])
            if previous_end > current_start:
                paired.reject("replica run windows overlap or are out of order")
        for job in jobs[1:]:
            if any(job[field] != first[field] for field in (
                "runner", "host_id", "operator", "baseline_build", "candidate_build",
                "baseline_executable_sha256", "candidate_executable_sha256",
                "run_id", "attempt",
            )):
                paired.reject("replicas differ in host, operator, build, binary, or collection identity")
        medians = [Decimal(job["median_ratio"]) for job in jobs]
        reference = median(medians)
        spread = (max(medians) - min(medians)) / reference
        report.update({
            "jobs": jobs, "between_job_spread": str(spread),
            "reference_median_ratio": str(reference),
        })
        if (any(Decimal(job["mad_ratio"]) > Decimal("0.10") or
                Decimal(job["p90_ratio"]) > Decimal("1.50") for job in jobs)
                or spread > Decimal("0.10")):
            paired.reject("paired ratios exceed locked dispersion limits", "S225_V4_DISPERSION")
        if any(value > Decimal("1.25") for value in medians):
            paired.reject("a replica median candidate/baseline ratio exceeds 1.25",
                          "S225_V4_REGRESSION")
        report.update({
            "status": "provisional_pass", "primary_rule": "S225_V4_PROVISIONAL_PASS",
            "detail": "three dedicated replicas pass numerical gates; independent evidence review pending",
        })
        return report, 0
    except common.EvidenceError as exc:
        report.update({
            "primary_rule": exc.rule, "detail": exc.detail,
            "status": "unstable" if exc.rule == "S225_V4_DISPERSION" else
                      "regression" if exc.rule == "S225_V4_REGRESSION" else "blocked",
        })
        return report, 1 if exc.rule in ("S225_V4_DISPERSION", "S225_V4_REGRESSION") else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--checkout", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, code = decide(args.bundle_root, args.candidate_commit, args.checkout,
                          args.collection_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"refactor-csc-dedicated: {report['primary_rule']}: {report['detail']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
