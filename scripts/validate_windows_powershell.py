#!/usr/bin/env python3
"""Validate owned Windows PowerShell workflow material.

This script validates the selected Windows CI PowerShell surface without
executing the workflow commands. Local runs without pwsh return exit 2 after
structural checks; hosted Windows runs should pass --require-pwsh and fail
closed if pwsh is unavailable or any snippet does not parse.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "windows-ci.yml"
SELECTED_TARGET_MANIFEST = (
    REPO_ROOT / "tests" / "corpus" / "manifests" / "selected_report_targets.tsv"
)
WINDOWS_DEFERRAL_RECORD = (
    REPO_ROOT
    / "docs"
    / "planning"
    / "EPIC_16"
    / "SPRINT_182"
    / "artifacts"
    / "windows-report-freshness-deferral-decision.md"
)

DEFERRAL_MARKER = "Windows report freshness remains formally deferred"
WORKFLOW_DEFERRAL_MARKER = "Sprint 182 formally defers Windows report freshness"
WINDOWS_RUNNER = "windows-2022"
FORBIDDEN_SELECTED_FRESHNESS = (
    "report-index-oracle-freshness",
    "report-index-comparison-freshness",
    "bench-canonical-report-freshness",
    "check_bench_canonical_freshness.py",
    "sprint159-oracle-freshness",
    "sprint175-linux-selected-comparison-freshness",
    "sprint175-macos-selected-comparison-freshness",
    "sprint168-selected-performance-freshness",
)
SELECTED_REPORT_FAMILIES = {"oracle", "comparison", "benchmark"}
HOSTED_VALIDATION_JOB = "powershell-validation"
HOSTED_VALIDATION_STEP_NAME = "Validate owned Windows PowerShell workflow material"
HOSTED_VALIDATION_COMMAND = "python scripts/validate_windows_powershell.py --require-pwsh"
CLAIM_BOUNDARY_MARKERS = {
    REPO_ROOT / "README.md": (
        "Windows still does not claim Makefile parity",
        "hosted PowerShell validation ownership\n  job",
        "workflow validation\nownership only",
        "Windows report\nfreshness is formally deferred by the Sprint 182 decision record",
        "Windows-safe generation path",
    ),
    REPO_ROOT / "INSTALL.md": (
        "hosted PowerShell validation ownership for selected Windows workflow snippets",
        "report freshness, package-manager support",
        "does not imply Windows Makefile parity",
        "runtime-loader behavior, or broad Windows parity",
    ),
    REPO_ROOT / "docs" / "maintainer_guide.md": (
        "Sprint 182 formally defers Windows report freshness",
        "make windows-powershell-validate",
        "python scripts/validate_windows_powershell.py --require-pwsh",
        "PowerShell check is unavailable, record that as an environment residual",
        "unavailable local PowerShell checks out of pass evidence",
    ),
    REPO_ROOT / "tests" / "corpus" / "README.md": (
        "Sprint 182 records Windows report freshness as formally deferred",
        "hosted Windows\nPowerShell validation lane owns selected workflow snippet parsing",
        "unavailable local PowerShell validation",
        "Do not reinterpret those states as pass evidence",
    ),
}
UNSUPPORTED_WINDOWS_CLAIM_PATTERNS = (
    re.compile(
        r"Windows report freshness (?:is |now )?(?:supported|promoted|complete|closed)",
        re.I,
    ),
    re.compile(r"PowerShell validation (?:proves|promotes|closes) Windows report freshness", re.I),
    re.compile(r"local unavailable PowerShell (?:is|counts as|proves) pass evidence", re.I),
    re.compile(r"Windows selected report artifacts? (?:are |now )?(?:published|uploaded)", re.I),
)


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:
    job_id: str
    name: str
    run: str
    shell: str


@dataclass(frozen=True)
class StepRequirement:
    job_id: str
    name_anchor: str
    tokens: tuple[str, ...]


STEP_REQUIREMENTS = (
    StepRequirement(
        "build-and-test",
        "Run enforced reviewed CMake configure path",
        ("cmake -S . -B build", "Visual Studio 17 2022"),
    ),
    StepRequirement(
        "build-and-test",
        "Run enforced reviewed CMake build path",
        ("cmake --build build", "Release"),
    ),
    StepRequirement(
        "build-and-test",
        "Inspect enforced Windows reviewed consumer CTest surface",
        ("EXPECTED_WINDOWS_CTEST_COUNT", "Total Tests:"),
    ),
    StepRequirement(
        "build-and-test",
        "Run enforced reviewed CMake execution path",
        ("ctest --test-dir build", "--output-on-failure"),
    ),
    StepRequirement(
        "install-and-downstream",
        "Run reviewed CMake install/downstream validation proof",
        ("sparse_lu_ortho.lib", "sparse.pc", "metadata-only", "find_package", "mismatch"),
    ),
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValidationError(f"missing required file: {rel(path)}") from exc


def fail(message: str) -> int:
    print(f"windows-powershell-validate: FAIL: {message}", file=sys.stderr, flush=True)
    return 1


def unavailable(message: str) -> int:
    print(
        f"windows-powershell-validate: UNAVAILABLE: {message}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "windows-powershell-validate: local unavailable PowerShell is not pass evidence",
        file=sys.stderr,
        flush=True,
    )
    return 2


def pass_msg(message: str) -> None:
    print(f"windows-powershell-validate: {message} ok", flush=True)


def find_job_block(text: str, job_id: str) -> str:
    marker = f"  {job_id}:\n"
    start = text.find(marker)
    if start == -1:
        raise ValidationError(f"windows workflow missing job {job_id!r}")
    next_start = len(text)
    search_pos = start + len(marker)
    for line_start in line_offsets(text, search_pos):
        line_end = text.find("\n", line_start)
        line = text[line_start : line_end + 1 if line_end != -1 else len(text)]
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            next_start = line_start
            break
    return text[start:next_start]


def line_offsets(text: str, start: int = 0) -> list[int]:
    offsets = []
    pos = text.find("\n", start)
    while pos != -1 and pos + 1 < len(text):
        offsets.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return offsets


def field_value(block: str, field: str) -> str:
    prefix = f"    {field}:"
    for line in block.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"')
    raise ValidationError(f"job block missing {field!r}")


def parse_steps(job_id: str, job_block: str) -> list[Step]:
    lines = job_block.splitlines()
    steps: list[Step] = []
    current: dict[str, str] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("      - name: "):
            if current is not None:
                steps.append(
                    Step(
                        job_id=job_id,
                        name=current.get("name", ""),
                        run=current.get("run", ""),
                        shell=current.get("shell", ""),
                    )
                )
            current = {"name": line.split(":", 1)[1].strip()}
            i += 1
            continue
        if current is None:
            i += 1
            continue
        if line.startswith("        shell: "):
            current["shell"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("        run: |"):
            block_lines: list[str] = []
            i += 1
            while i < len(lines) and (
                lines[i].startswith("          ") or lines[i].strip() == ""
            ):
                block_lines.append(lines[i][10:] if lines[i].startswith("          ") else "")
                i += 1
            current["run"] = "\n".join(block_lines)
            continue
        if line.startswith("        run: "):
            current["run"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        i += 1
    if current is not None:
        steps.append(
            Step(
                job_id=job_id,
                name=current.get("name", ""),
                run=current.get("run", ""),
                shell=current.get("shell", ""),
            )
        )
    return steps


def split_manifest_values(value: str) -> list[str]:
    if value == "none":
        return []
    return [part for part in value.split(";") if part]


def selected_report_targets() -> list[dict[str, str]]:
    with SELECTED_TARGET_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def selected_report_freshness_tokens(rows: list[dict[str, str]]) -> tuple[str, ...]:
    tokens = set(FORBIDDEN_SELECTED_FRESHNESS)
    for row in rows:
        for field in ("generator_command", "workflow_job", "workflow_artifact"):
            tokens.update(split_manifest_values(row[field]))
    return tuple(sorted(token for token in tokens if token != "none"))


def validate_selected_report_references(rows: list[dict[str, str]]) -> None:
    for row in rows:
        target_id = row["target_id"]
        if row["family"] not in SELECTED_REPORT_FAMILIES:
            raise ValidationError(
                f"selected_report_targets.tsv has unexpected selected family "
                f"{row['family']!r}: {target_id}"
            )
        workflow_files = split_manifest_values(row["workflow_file"])
        workflow_jobs = split_manifest_values(row["workflow_job"])
        workflow_artifacts = split_manifest_values(row["workflow_artifact"])
        workflow_platforms = split_manifest_values(row["workflow_platforms"])
        if not workflow_files:
            raise ValidationError(f"{target_id} is missing workflow_file metadata")
        if not workflow_jobs:
            raise ValidationError(f"{target_id} is missing workflow_job metadata")
        if not workflow_artifacts:
            raise ValidationError(f"{target_id} is missing workflow_artifact metadata")
        if not workflow_platforms:
            raise ValidationError(f"{target_id} is missing workflow_platforms metadata")
        if len(workflow_artifacts) not in (1, len(workflow_platforms)):
            raise ValidationError(
                f"{target_id} workflow_artifact must contain one shared artifact "
                "or one artifact per workflow platform"
            )
        for workflow_file in workflow_files:
            workflow_path = REPO_ROOT / workflow_file
            if not workflow_path.is_file():
                raise ValidationError(
                    f"{target_id} references missing workflow_file {workflow_file!r}"
                )
        for workflow_artifact in workflow_artifacts:
            if not workflow_artifact.startswith("sprint"):
                raise ValidationError(
                    f"{target_id} workflow_artifact is not sprint-scoped: "
                    f"{workflow_artifact!r}"
                )
    pass_msg(f"selected report manifest references ({len(rows)} rows)")


def validate_manifest_windows_deferral(rows: list[dict[str, str]]) -> None:
    for row in rows:
        platforms = split_manifest_values(row["workflow_platforms"])
        if "windows" in platforms:
            raise ValidationError(
                "selected_report_targets.tsv must not list windows while "
                f"Windows report freshness is deferred: {row['target_id']}"
            )
    pass_msg(f"selected manifest has no windows workflow platforms ({len(rows)} rows)")


def validate_deferral_record() -> None:
    text = read_text(WINDOWS_DEFERRAL_RECORD)
    if DEFERRAL_MARKER not in text:
        raise ValidationError("Windows report freshness deferral marker is missing")
    pass_msg("Windows report freshness deferral record")


def validate_workflow_structure(
    text: str,
    forbidden_selected_freshness: tuple[str, ...] = FORBIDDEN_SELECTED_FRESHNESS,
) -> list[Step]:
    if WORKFLOW_DEFERRAL_MARKER not in text:
        raise ValidationError("windows workflow missing Sprint 182 deferral comment")
    pass_msg("windows workflow deferral comment")

    steps: list[Step] = []
    for job_id in ("build-and-test", "install-and-downstream", HOSTED_VALIDATION_JOB):
        block = find_job_block(text, job_id)
        runner = field_value(block, "runs-on")
        if runner != WINDOWS_RUNNER:
            raise ValidationError(f"{job_id} must run on {WINDOWS_RUNNER}, got {runner!r}")
        pass_msg(f"{job_id} runner")
        steps.extend(parse_steps(job_id, block))

    for needle in forbidden_selected_freshness:
        if needle in text:
            raise ValidationError(
                f"windows workflow must not run or upload selected report freshness {needle!r}"
            )
    if "actions/upload-artifact" in text:
        raise ValidationError(
            "windows workflow must not publish hosted artifacts while selected "
            "Windows report evidence is absent"
        )
    pass_msg("windows selected report freshness non-promotion")

    return steps


def validate_claim_boundaries(overrides: dict[Path, str] | None = None) -> None:
    for path, markers in CLAIM_BOUNDARY_MARKERS.items():
        text = overrides[path] if overrides and path in overrides else read_text(path)
        for marker in markers:
            if marker not in text:
                raise ValidationError(
                    f"{rel(path)} missing Windows/PowerShell non-claim marker {marker!r}"
                )
        for pattern in UNSUPPORTED_WINDOWS_CLAIM_PATTERNS:
            match = pattern.search(text)
            if match:
                raise ValidationError(
                    f"{rel(path)} contains unsupported Windows/PowerShell claim "
                    f"{match.group(0)!r}"
                )
    pass_msg(f"Windows/PowerShell claim boundaries ({len(CLAIM_BOUNDARY_MARKERS)} files)")


def validate_hosted_validation_wiring(text: str) -> None:
    block = find_job_block(text, HOSTED_VALIDATION_JOB)
    runner = field_value(block, "runs-on")
    if runner != WINDOWS_RUNNER:
        raise ValidationError(
            f"{HOSTED_VALIDATION_JOB} must run on {WINDOWS_RUNNER}, got {runner!r}"
        )
    steps = parse_steps(HOSTED_VALIDATION_JOB, block)
    matches = [step for step in steps if step.name == HOSTED_VALIDATION_STEP_NAME]
    if len(matches) != 1:
        raise ValidationError(
            f"{HOSTED_VALIDATION_JOB} expected one step named "
            f"{HOSTED_VALIDATION_STEP_NAME!r}, got {len(matches)}"
        )
    step = matches[0]
    if HOSTED_VALIDATION_COMMAND not in step.run:
        raise ValidationError(
            f"{HOSTED_VALIDATION_JOB}:{step.name} must run "
            f"{HOSTED_VALIDATION_COMMAND!r}"
        )
    if step.shell != "cmd":
        raise ValidationError(
            f"{HOSTED_VALIDATION_JOB}:{step.name} must declare shell: cmd, "
            f"got {step.shell!r}"
        )
    pass_msg("hosted Windows PowerShell validation wiring")


def find_required_step(steps: list[Step], requirement: StepRequirement) -> Step:
    matches = [
        step
        for step in steps
        if step.job_id == requirement.job_id and requirement.name_anchor in step.name
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"{requirement.job_id} expected one step containing "
            f"{requirement.name_anchor!r}, got {len(matches)}"
        )
    return matches[0]


def validate_required_steps(steps: list[Step]) -> list[Step]:
    selected: list[Step] = []
    for requirement in STEP_REQUIREMENTS:
        step = find_required_step(steps, requirement)
        if step.shell != "pwsh":
            raise ValidationError(
                f"{step.job_id}:{step.name} must declare shell: pwsh, got {step.shell!r}"
            )
        if not step.run:
            raise ValidationError(f"{step.job_id}:{step.name} is missing run text")
        for token in requirement.tokens:
            if token not in step.run:
                raise ValidationError(f"{step.job_id}:{step.name} missing token {token!r}")
        selected.append(step)

    selected_keys = {(step.job_id, step.name) for step in selected}
    unowned_pwsh_steps = [
        f"{step.job_id}:{step.name}"
        for step in steps
        if step.shell == "pwsh" and (step.job_id, step.name) not in selected_keys
    ]
    if unowned_pwsh_steps:
        raise ValidationError(
            "windows workflow has unowned PowerShell steps: "
            + ", ".join(unowned_pwsh_steps)
        )
    pass_msg(f"selected PowerShell workflow steps ({len(selected)})")
    return selected


def parse_with_pwsh(pwsh: str, steps: list[Step]) -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-windows-pwsh-") as tmp:
        tmpdir = Path(tmp)
        for index, step in enumerate(steps, start=1):
            snippet = tmpdir / f"snippet-{index}.ps1"
            snippet.write_text(step.run + "\n", encoding="utf-8")
            env = os.environ.copy()
            env["SPARSE_PWSH_SNIPPET"] = str(snippet)
            result = subprocess.run(
                [
                    pwsh,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    (
                        "$ErrorActionPreference = 'Stop'; "
                        "$text = Get-Content -Raw -LiteralPath $env:SPARSE_PWSH_SNIPPET; "
                        "[scriptblock]::Create($text) | Out-Null"
                    ),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise ValidationError(
                    f"PowerShell parse failed for {step.job_id}:{step.name}: {detail}"
                )
    pass_msg(f"PowerShell parse validation ({len(steps)} snippets)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate selected Windows CI PowerShell workflow material."
    )
    parser.add_argument(
        "--require-pwsh",
        action="store_true",
        help="fail if pwsh is unavailable; intended for hosted Windows CI",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        selected_targets = selected_report_targets()
        workflow_text = read_text(WINDOWS_WORKFLOW)
        validate_selected_report_references(selected_targets)
        steps = validate_workflow_structure(
            workflow_text,
            selected_report_freshness_tokens(selected_targets),
        )
        validate_claim_boundaries()
        validate_hosted_validation_wiring(workflow_text)
        selected_steps = validate_required_steps(steps)
        validate_deferral_record()
        validate_manifest_windows_deferral(selected_targets)
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            if args.require_pwsh:
                raise ValidationError("pwsh not found but --require-pwsh was set")
            return unavailable("pwsh not found; structural checks passed")
        parse_with_pwsh(pwsh, selected_steps)
    except ValidationError as exc:
        return fail(str(exc))

    print(
        "windows-powershell-validate: passed "
        f"({len(STEP_REQUIREMENTS)} selected PowerShell snippets)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
