#!/usr/bin/env python3
"""Guard the reviewed same-machine Sprint 225 timing collection and gate."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/sprint225-timing.yml"


def problems(workflow: dict) -> list[str]:
    issues: list[str] = []
    triggers = workflow.get("on", {})
    if "master" not in triggers.get("push", {}).get("branches", []) or "master" not in triggers.get("pull_request", {}).get("branches", []):
        issues.append("paired timing must run on master pushes and pull requests")
    jobs = workflow.get("jobs", {})
    collect = jobs.get("collect-pair", {})
    decide = jobs.get("canonical-refactor-csc-timing", {})
    if not collect or not decide:
        return issues + ["collection or decision job missing"]
    if (collect.get("runs-on") != "macos-15-intel" or collect.get("timeout-minutes") != "20"
            or collect.get("env", {}).get("OMP_NUM_THREADS") != "1"):
        issues.append("reviewed macOS class, serial mode, and 20-minute cap required")
    matrix = collect.get("strategy", {}).get("matrix", {})
    if matrix.get("replica") != ["1", "2", "3"] or collect.get("strategy", {}).get("fail-fast") != "false":
        issues.append("three independent pairs required")
    steps = collect.get("steps", [])
    if (not steps or steps[0].get("with", {}).get("fetch-depth") != "0" or
            "github.event.pull_request.head.sha" not in steps[0].get("with", {}).get("ref", "")):
        issues.append("full history required to check out the baseline source")
    named = {step.get("name"): step for step in steps}
    raw = named.get("Collect same-machine raw baseline and candidate trials", {})
    upload = named.get("Upload paired raw trial bundle", {})
    if not all(token in raw.get("run", "") for token in (
        "collect_refactor_csc_pair.py", "--replica", "--output-dir",
    )) or raw.get("continue-on-error") == "true":
        issues.append("paired raw collector must fail closed")
    with_fields = upload.get("with", {})
    if (upload.get("if") != "always()" or upload.get("uses") != "actions/upload-artifact@v4"
            or with_fields.get("if-no-files-found") != "error"
            or with_fields.get("retention-days") != "90"
            or "${{ github.run_id }}" not in with_fields.get("name", "")
            or "${{ github.run_attempt }}" not in with_fields.get("name", "")
            or "${{ matrix.replica }}" not in with_fields.get("name", "")
            or with_fields.get("path") != "build/sprint225-pair/"):
        issues.append("every pair needs its own always-uploaded raw artifact")
    if (decide.get("needs") != "collect-pair" or decide.get("if") != "always()"
            or decide.get("timeout-minutes") != "5"):
        issues.append("decision must run after all three jobs, including failure")
    steps = decide.get("steps", [])
    if (not steps or steps[0].get("with", {}).get("fetch-depth") != "0" or
            "github.event.pull_request.head.sha" not in steps[0].get("with", {}).get("ref", "")):
        issues.append("decision requires source history for provenance checks")
    named = {step.get("name"): step for step in steps}
    download = named.get("Download three paired raw bundles", {})
    gate = named.get("Recompute canonical paired timing decision", {})
    decision_upload = named.get("Upload paired timing decision", {})
    if (download.get("uses") != "actions/download-artifact@v4"
            or not all(token in download.get("with", {}).get("pattern", "") for token in (
                "${{ github.run_id }}", "${{ github.run_attempt }}", "*",
            ))):
        issues.append("download all three raw pair artifacts")
    if (gate.get("if") != "always()" or gate.get("continue-on-error") == "true"
            or not all(token in gate.get("run", "") for token in (
                "check_refactor_csc_pair.py", "--bundle-root", "--candidate-commit",
                "--run-id", "--run-attempt", "--output",
            ))):
        issues.append("independent raw decision must fail closed")
    if (decision_upload.get("if") != "always()" or
            decision_upload.get("with", {}).get("if-no-files-found") != "error" or
            "build/sprint225-timing-decision.json" not in decision_upload.get("with", {}).get("path", "")):
        issues.append("always upload timing decision or blocked receipt")
    if any("check_bench_canonical_freshness" in step.get("run", "") for job in jobs.values() for step in job.get("steps", [])):
        issues.append("threshold-free freshness cannot decide timing")
    return issues


class TimingWorkflowTests(unittest.TestCase):
    def test_real_workflow_preserves_paired_gate(self):
        workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(problems(workflow), [])

    def test_mutations_fail_closed(self):
        original = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
        for name, change in (
            ("one-job", lambda w: w["jobs"]["collect-pair"]["strategy"]["matrix"].update(replica=["1"])),
            ("wrong-runner", lambda w: w["jobs"]["collect-pair"].update({"runs-on": "macos-latest"})),
            ("shallow-source", lambda w: w["jobs"]["collect-pair"]["steps"][0]["with"].update({"fetch-depth": "1"})),
            ("no-raw-upload", lambda w: w["jobs"]["collect-pair"]["steps"].pop()),
            ("success-only-decision", lambda w: w["jobs"]["canonical-refactor-csc-timing"].update({"if": "success()"})),
            ("swallow-decision", lambda w: w["jobs"]["canonical-refactor-csc-timing"]["steps"][-2].update({"continue-on-error": "true"})),
            ("freshness-substitute", lambda w: w["jobs"]["canonical-refactor-csc-timing"]["steps"][-2].update({"run": "python3 scripts/check_bench_canonical_freshness.py"})),
        ):
            with self.subTest(name=name):
                workflow = copy.deepcopy(original)
                change(workflow)
                self.assertTrue(problems(workflow), name)


if __name__ == "__main__":
    unittest.main()
