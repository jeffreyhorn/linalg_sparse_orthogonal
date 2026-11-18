#!/usr/bin/env python3
"""Record diagnostic host context before the Sprint 225 baseline preflight."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from collect_refactor_csc_timing import POLICY_VERSION, git_value, host_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = {
        "purpose": "timing_preflight_diagnostic_only",
        "policy_version": POLICY_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_value("rev-parse", "HEAD"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "workflow_job_key": os.environ.get("GITHUB_JOB", ""),
        "runner": host_metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"timing host context: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
