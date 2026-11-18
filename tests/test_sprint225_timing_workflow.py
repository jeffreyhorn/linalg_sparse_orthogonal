#!/usr/bin/env python3
"""Guard the independent Sprint 225 timing job and its failure artifact route."""

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
        issues.append("timing job must run on master pushes and pull requests")
    jobs = workflow.get("jobs", {})
    job = jobs.get("canonical-refactor-csc-timing")
    if not isinstance(job, dict):
        return issues + ["separate timing job missing"]
    if job.get("runs-on") != "macos-latest" or job.get("timeout-minutes") != "20":
        issues.append("locked macOS runner and 20-minute cap required")
    steps = job.get("steps", [])
    if any(step.get("continue-on-error") == "true" for step in steps):
        issues.append("timing failure must fail the job")
    named = {step.get("name"): step for step in steps}
    clean = named.get("Clean candidate build tree before receipts", {})
    host = named.get("Record hosted timing runner context", {})
    baseline = named.get("Verify accepted baseline before candidate collection", {})
    build = named.get("Build pinned canonical benchmark from clean tree", {})
    collect = named.get("Collect raw candidate trial bundle", {})
    decide = named.get("Decide canonical candidate timing", {})
    upload = named.get("Upload candidate raw trials and timing decision", {})
    if "record_refactor_csc_timing_host.py" not in host.get("run", ""):
        issues.append("diagnostic runner context required before baseline preflight")
    if "make clean" not in clean.get("run", "") or (clean and host and steps.index(clean) > steps.index(host)):
        issues.append("clean build tree before writing attempt receipts")
    if host and baseline and steps.index(host) > steps.index(baseline):
        issues.append("runner context must precede baseline preflight")
    if baseline.get("id") != "baseline" or "--preflight" not in baseline.get("run", ""):
        issues.append("accepted-baseline preflight required")
    if ("make clean" in build.get("run", "") or "SPARSE_IDX_BITS=32" not in build.get("run", "")
            or "sprint225-build-config" not in build.get("run", "")):
        issues.append("clean pinned build required")
    if "--role candidate" not in collect.get("run", "") or "--build-config" not in collect.get("run", ""):
        issues.append("candidate raw collection required")
    if not all(token in decide.get("run", "") for token in (
        "check_refactor_csc_timing.py", "--candidate", "--verify-checkout",
        "--expected-candidate-commit", "--output",
    )) or "always()" not in decide.get("if", ""):
        issues.append("independent candidate decision required even after collection failure")
    if any("check_bench_canonical_freshness" in step.get("run", "") for step in steps):
        issues.append("selected freshness cannot decide timing")
    with_fields = upload.get("with", {})
    if (upload.get("uses") != "actions/upload-artifact@v4" or upload.get("if") != "always()"
            or with_fields.get("if-no-files-found") != "error"
            or with_fields.get("retention-days") != "90"
            or "${{ github.run_id }}" not in with_fields.get("name", "")
            or "${{ github.run_attempt }}" not in with_fields.get("name", "")
            or not all(path in with_fields.get("path", "") for path in (
                "build/sprint225-candidate/", "build/sprint225-timing-decision.json",
                "build/sprint225-host-provenance.json",
            ))):
        issues.append("always-uploaded raw bundle and decision receipt required")
    return issues


class TimingWorkflowTests(unittest.TestCase):
    def test_real_workflow_preserves_independent_failure_route(self):
        workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(problems(workflow), [])

    def test_mutations_cannot_remove_timing_or_artifact_requirement(self):
        original = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)

        def mutate_no_upload(workflow):
            workflow["jobs"]["canonical-refactor-csc-timing"]["steps"].pop()

        def mutate_success_only_upload(workflow):
            workflow["jobs"]["canonical-refactor-csc-timing"]["steps"][-1]["if"] = "success()"

        def mutate_missing_raw(workflow):
            workflow["jobs"]["canonical-refactor-csc-timing"]["steps"][-1]["with"]["path"] = "build/sprint225-timing-decision.json"

        def mutate_disposable_receipts(workflow):
            steps = workflow["jobs"]["canonical-refactor-csc-timing"]["steps"]
            steps[4]["run"] = "make clean\n" + steps[4]["run"]

        def mutate_no_build_receipt(workflow):
            steps = workflow["jobs"]["canonical-refactor-csc-timing"]["steps"]
            steps[4]["run"] = steps[4]["run"].replace("sprint225-build-config", "missing-config")

        def mutate_freshness_substitution(workflow):
            workflow["jobs"]["canonical-refactor-csc-timing"]["steps"][-2]["run"] = "python3 scripts/check_bench_canonical_freshness.py"

        def mutate_swallowed_failure(workflow):
            workflow["jobs"]["canonical-refactor-csc-timing"]["steps"][-2]["continue-on-error"] = "true"

        def mutate_no_master(workflow):
            workflow["on"]["push"]["branches"] = ["sprint-225"]

        for name, mutate in (
            ("no-upload", mutate_no_upload),
            ("success-only-upload", mutate_success_only_upload),
            ("missing-raw", mutate_missing_raw),
            ("disposable-receipts", mutate_disposable_receipts),
            ("no-build-receipt", mutate_no_build_receipt),
            ("freshness-substitution", mutate_freshness_substitution),
            ("swallowed-failure", mutate_swallowed_failure),
            ("no-master", mutate_no_master),
        ):
            with self.subTest(name=name):
                workflow = copy.deepcopy(original)
                mutate(workflow)
                self.assertTrue(problems(workflow), name)


if __name__ == "__main__":
    unittest.main()
