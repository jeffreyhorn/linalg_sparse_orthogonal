#!/usr/bin/env python3
"""Compare the frozen API-docs guard fixtures with baseline and structured policy."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from api_docs_workflow_policy import PolicyViolation, check_direct_policy
from api_docs_workflow_policy_inputs import read_policy_inputs
from api_docs_workflow_reader import WorkflowReadError


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/planning/EPIC_20/SPRINT_220/artifacts/day2-fixture-classification.tsv"
FIXTURES = ROOT / "tests/test_api_docs_local_only_guard.py"
EDGE_MATRIX = ROOT / "docs/planning/EPIC_20/SPRINT_220/artifacts/day3-edge-fixture-matrix.tsv"
EDGE_FIXTURES = ROOT / "tests/fixtures/api_docs_yaml_edges"
NON_WORKFLOW = frozenset((
    "test_current_tree_passes_guard", "test_fixture_passes_guard",
    "test_tracked_generated_api_file_fails_clearly",
    "test_staged_generated_api_file_fails_clearly",
    "test_visible_untracked_generated_api_file_fails_clearly",
    "test_missing_local_only_wording_fails_clearly",
))
MIXED_VARIANTS = {
    "test_workflow_powershell_docs_publication_commands": (
        *(["REJECT"] * 11), *(["ALLOW"] * 4),
    ),
    "test_workflow_powershell_backtick_continuations": (
        "REJECT", "REJECT", "ALLOW", "REJECT", "REJECT", "ALLOW",
    ),
    "test_workflow_powershell_backtick_windows_paths": ("REJECT", "ALLOW"),
    "test_workflow_folded_blank_line_continuation_stages_docs": (
        "REJECT", "ALLOW", "ALLOW",
    ),
}
MIXED_DIAGNOSTICS = {
    "test_workflow_powershell_docs_publication_commands": (
        *(["stages docs"] * 3), *(["archives docs"] * 2),
        *(["dynamic staging paths"] * 6), *(["-"] * 4),
    ),
    "test_workflow_powershell_backtick_continuations": (
        "stages docs", "stages docs", "-", "stages docs", "stages docs", "-",
    ),
    "test_workflow_powershell_backtick_windows_paths": ("stages docs", "-"),
    "test_workflow_folded_blank_line_continuation_stages_docs": (
        "stages docs", "-", "-",
    ),
}
LOCATION = re.compile(r"(?P<path>\.github/workflows/[^:\s]+):(?P<line>\d+):(?P<column>\d+)")
UNEXPECTED = frozenset((
    "current_guard_regression", "structured_false_positive", "structured_missed_bypass",
    "parser_error", "unclassified_disagreement", "edge_stage_mismatch",
))


def load_oracle() -> tuple[str, dict[str, dict[str, str]], tuple[str, ...]]:
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    revision = re.search(r"at ([0-9a-f]{40})", lines[0])
    if revision is None:
        raise ValueError("frozen manifest lacks a full baseline revision")
    rows = tuple(csv.DictReader((line for line in lines if not line.startswith("#")), delimiter="\t"))
    oracle = {row["test_function"]: row for row in rows}
    if len(oracle) != len(rows):
        raise ValueError("duplicate frozen fixture name")
    tree = ast.parse(FIXTURES.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    registered = tuple(
        node.value.func.id for node in main.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    )
    if set(registered) != set(oracle) or len(registered) != len(oracle):
        raise ValueError("registered fixtures differ from the frozen manifest")
    if {name for name, row in oracle.items() if row["expected"] == "MIXED"} != set(MIXED_VARIANTS):
        raise ValueError("mixed-variant expectations differ from the frozen manifest")
    if any(len(MIXED_VARIANTS[name]) != len(MIXED_DIAGNOSTICS[name]) for name in MIXED_VARIANTS):
        raise ValueError("mixed-variant diagnostics differ from frozen verdicts")
    return revision.group(1), oracle, registered


def _redact_root(value: str, root: Path) -> str:
    return value.replace(str(root.resolve()), "<fixture>").replace(str(root), "<fixture>")


def _diagnostic(result: subprocess.CompletedProcess[str], root: Path) -> str:
    output = result.stderr.strip() if result.returncode else result.stdout.strip().splitlines()[-1]
    return _redact_root(output, root)


def _run_legacy(root: Path, baseline: str) -> tuple[str, str]:
    env = {**os.environ, "API_DOCS_PYTHON": sys.executable}
    if root.resolve() == ROOT:
        # This one fixture uses the checkout itself. Redirect root lookup in a
        # temporary script instead of writing a baseline file into the checkout.
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "legacy_guard.sh"
            root_line = 'ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"'
            if baseline.count(root_line) != 1:
                raise ValueError("baseline root declaration changed")
            script.write_text(baseline.replace(root_line, f"ROOT_DIR={shlex.quote(str(root))}"), encoding="utf-8")
            result = subprocess.run(["bash", str(script)], cwd=root, env=env,
                                    text=True, capture_output=True, check=False)
    else:
        script = root / "scripts" / "legacy_guard.sh"
        script.write_text(baseline, encoding="utf-8")
        try:
            result = subprocess.run(["bash", str(script)], cwd=root, env=env,
                                    text=True, capture_output=True, check=False)
        finally:
            script.unlink()
    return ("REJECT" if result.returncode else "ALLOW", _diagnostic(result, root))


def _run_structured(root: Path) -> tuple[str, str]:
    try:
        check_direct_policy(read_policy_inputs(root))
    except (WorkflowReadError, PolicyViolation) as exc:
        return "REJECT", _redact_root(str(exc), root)
    return "ALLOW", ""


def _classify(expected: str, old: str, new: str, current: str, diagnostic: str) -> list[str]:
    findings = []
    if current != expected:
        findings.append("current_guard_regression")
    if new != expected:
        findings.append("structured_false_positive" if expected == "ALLOW" else "structured_missed_bypass")
    if old != expected:
        findings.append("legacy_false_positive" if expected == "ALLOW" else "legacy_missed_bypass")
    if new == "REJECT" and "parse:" in diagnostic and expected == "ALLOW":
        findings.append("parser_error")
    if old != new and not findings:
        findings.append("unclassified_disagreement")
    return findings


def _unexpected_count(counts: Counter[str]) -> int:
    return sum(counts[label] for label in UNEXPECTED)


def _edge_stage_matches(stage: str, diagnostic: str) -> bool:
    if stage == "-":
        return not diagnostic
    if stage == "policy":
        return (
            "jobs." in diagnostic
            and ": parse:" not in diagnostic
            and ": unsupported:" not in diagnostic
        )
    return f": {stage}:" in diagnostic


def run_edges(output: Path) -> tuple[int, Counter[str]]:
    revision, _, _ = load_oracle()
    baseline = subprocess.run(
        ["git", "show", f"{revision}:scripts/check_api_docs_local_only.sh"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout
    sys.path.insert(0, str(ROOT / "tests"))
    from test_api_docs_local_only_guard import run_guard, write_fixture

    with EDGE_MATRIX.open(encoding="utf-8") as stream:
        matrix = tuple(csv.DictReader(stream, delimiter="\t"))
    rows: list[dict[str, object]] = []
    for case in matrix:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            (root / ".github/workflows/api-docs.yml").write_text(
                (EDGE_FIXTURES / case["fixture"]).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            current = run_guard(root)
            old_verdict, old_detail = _run_legacy(root, baseline)
            new_verdict, new_detail = _run_structured(root)
            current_verdict = "REJECT" if current.returncode else "ALLOW"
            location = LOCATION.search(new_detail) or LOCATION.search(old_detail)
            classification = _classify(
                case["expected_class"], old_verdict, new_verdict,
                current_verdict, new_detail,
            )
            if not _edge_stage_matches(case["failure_stage"], new_detail):
                classification.append("edge_stage_mismatch")
            rows.append({
                "fixture": f"edge:{case['fixture']}", "variant": 1, "scope": "workflow",
                "baseline_revision": revision, "expected": case["expected_class"],
                "expected_diagnostic": case["failure_stage"],
                "legacy": {"verdict": old_verdict, "diagnostic": old_detail},
                "structured": {"verdict": new_verdict, "diagnostic": new_detail},
                "current": {"verdict": current_verdict, "diagnostic": _diagnostic(current, root)},
                "workflow_files": (".github/workflows/api-docs.yml", ".github/workflows/ci.yml"),
                "workflow_location": location.groupdict() if location else None,
                "classification": classification,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    counts = Counter(label for row in rows for label in row["classification"])
    print(f"api-docs-differential: {len(rows)} YAML edge fixtures, report {output}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    return _unexpected_count(counts), counts


def run(output: Path, selected: frozenset[str]) -> tuple[int, Counter[str]]:
    revision, oracle, registered = load_oracle()
    if selected - oracle.keys():
        raise ValueError(f"unknown frozen fixture(s): {', '.join(sorted(selected - oracle.keys()))}")
    baseline = subprocess.run(
        ["git", "show", f"{revision}:scripts/check_api_docs_local_only.sh"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout
    sys.path.insert(0, str(ROOT / "tests"))
    import test_api_docs_local_only_guard as suite

    original_run_guard = suite.run_guard
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    current_name = ""
    variant = 0

    def capture(root: Path) -> subprocess.CompletedProcess[str]:
        nonlocal variant
        variant += 1
        expected_class = oracle[current_name]["expected"]
        expected = (MIXED_VARIANTS[current_name][variant - 1]
                    if expected_class == "MIXED" else expected_class)
        expected_detail = (MIXED_DIAGNOSTICS[current_name][variant - 1]
                           if expected_class == "MIXED" else oracle[current_name]["required_diagnostic"])
        current = original_run_guard(root)
        old_verdict, old_detail = _run_legacy(root, baseline)
        if current_name in NON_WORKFLOW:
            new_verdict, new_detail = "NOT_APPLICABLE", "non-workflow contract"
        else:
            new_verdict, new_detail = _run_structured(root)
        current_verdict = "REJECT" if current.returncode else "ALLOW"
        workflow_files = tuple(sorted(str(path.relative_to(root)) for path in (root / ".github/workflows").glob("*.y*ml")))
        location = LOCATION.search(new_detail) or LOCATION.search(old_detail)
        comparison = [] if new_verdict == "NOT_APPLICABLE" else _classify(
            expected, old_verdict, new_verdict, current_verdict, new_detail,
        )
        if new_verdict == "NOT_APPLICABLE" and current_verdict != expected:
            comparison.append("current_guard_regression")
        rows.append({
            "fixture": current_name, "variant": variant, "scope": "non_workflow" if current_name in NON_WORKFLOW else "workflow",
            "baseline_revision": revision,
            "expected": expected, "expected_diagnostic": expected_detail,
            "legacy": {"verdict": old_verdict, "diagnostic": old_detail},
            "structured": {"verdict": new_verdict, "diagnostic": new_detail},
            "current": {"verdict": current_verdict, "diagnostic": _diagnostic(current, root)},
            "workflow_files": workflow_files,
            "workflow_location": location.groupdict() if location else None,
            "classification": comparison,
        })
        return current

    suite.run_guard = capture
    try:
        for name in registered:
            if selected and name not in selected:
                continue
            current_name = name
            variant = 0
            try:
                getattr(suite, name)()
            except Exception as exc:
                failures.append(f"{name}: {exc}")
            expected_count = len(MIXED_VARIANTS[name]) if name in MIXED_VARIANTS else 1
            if variant < expected_count:
                failures.append(f"{name}: only {variant}/{expected_count} variants captured")
    finally:
        suite.run_guard = original_run_guard

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    counts = Counter(label for row in rows for label in row["classification"])
    print(f"api-docs-differential: {len(rows)} variants, {len(failures)} fixture errors, report {output}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    for failure in failures:
        print(f"  ERROR {failure}", file=sys.stderr)
    return len(failures) + _unexpected_count(counts), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[], help="run one frozen fixture (repeatable)")
    parser.add_argument("--edges-only", action="store_true", help="compare the Day 3 YAML edge matrix")
    args = parser.parse_args()
    if args.edges_only and args.case:
        parser.error("--case cannot be used with --edges-only")
    errors, _ = run_edges(args.output) if args.edges_only else run(args.output, frozenset(args.case))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
