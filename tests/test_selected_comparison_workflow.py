#!/usr/bin/env python3
"""Guard selected report freshness workflow integration."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from normalize_report_index import (  # noqa: E402
    expected_int,
    selected_report_targets,
    split_manifest_values,
)


LINUX_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MACOS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "macos-ci.yml"
WINDOWS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "windows-ci.yml"
SELECTED_TARGET_MANIFEST = REPO_ROOT / "tests" / "corpus"
WINDOWS_DEFERRAL_RECORD = (
    REPO_ROOT
    / "docs"
    / "planning"
    / "EPIC_16"
    / "SPRINT_182"
    / "artifacts"
    / "windows-report-freshness-deferral-decision.md"
)
ORACLE_UPLOAD_PATHS = [
    "build/corpus/oracle/corpus.oracle.tsv",
    "build/corpus-reports/index.tsv",
    "build/corpus-reports/skips.tsv",
    "build/corpus-reports/manifest.txt",
]
BENCHMARK_UNSELECTED_CONTEXT_FILES = [
    "build/bench-reports/canonical/bench_chol_csc.csv",
    "build/bench-reports/canonical/bench_iterative_reuse.csv",
    "build/bench-reports/canonical/bench_eigs_reuse.csv",
]
WINDOWS_FORBIDDEN_SELECTED_FRESHNESS = [
    "report-index-oracle-freshness",
    "report-index-comparison-freshness",
    "bench-canonical-report-freshness",
    "check_bench_canonical_freshness.py",
    "sprint159-oracle-freshness",
    "sprint175-linux-selected-comparison-freshness",
    "sprint175-macos-selected-comparison-freshness",
    "sprint168-selected-performance-freshness",
]
WINDOWS_DEFERRAL_REQUIRED_TEXT = [
    "Windows report freshness remains formally deferred",
    "no reviewed Windows Makefile parity",
    "no Windows-safe CMake/MSVC project probe path",
    "no Windows-native canonical benchmark report generator",
    "selected target manifest rows do not list `windows`",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_contains(text: str, needle: str, *, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} missing {needle!r}")


def assert_raises_with(fn, expected: str) -> None:
    try:
        fn()
    except AssertionError as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}") from exc
        return
    raise AssertionError(f"expected failure containing {expected!r}")


def workflow_rows(workflow_path: Path, job_id: str) -> list[dict[str, str]]:
    rows = []
    workflow_file = str(workflow_path.relative_to(REPO_ROOT))
    for row in selected_report_targets(SELECTED_TARGET_MANIFEST):
        files = split_manifest_values(row["workflow_file"])
        jobs = split_manifest_values(row["workflow_job"])
        if workflow_file in files and job_id in jobs:
            rows.append(row)
    return rows


def comparison_rows(workflow_path: Path, job_id: str) -> list[dict[str, str]]:
    rows = [row for row in workflow_rows(workflow_path, job_id) if row["family"] == "comparison"]
    if not rows:
        raise AssertionError(
            f"{workflow_path.name}:{job_id} expected selected comparison rows from manifest"
        )
    return rows


def single_row(workflow_path: Path, job_id: str, family: str) -> dict[str, str]:
    rows = [row for row in workflow_rows(workflow_path, job_id) if row["family"] == family]
    if len(rows) != 1:
        raise AssertionError(
            f"{workflow_path.name}:{job_id} expected one selected {family} row, got {len(rows)}"
        )
    return rows[0]


def job_block(text: str, job_id: str, *, label: str) -> str:
    match = re.search(rf"(?m)^  {re.escape(job_id)}:\n", text)
    if match is None:
        raise AssertionError(f"{label} missing job {job_id!r}")
    next_job = re.search(r"(?m)^  [A-Za-z0-9_-]+:\n", text[match.end() :])
    end = match.end() + next_job.start() if next_job else len(text)
    return text[match.start() : end]


def upload_block(job: str, artifact_name: str, *, label: str) -> str:
    name_index = job.find(f"name: {artifact_name}")
    if name_index == -1:
        raise AssertionError(f"{label} missing upload artifact name {artifact_name!r}")
    block_start = job.rfind("- name:", 0, name_index)
    if block_start == -1:
        raise AssertionError(f"{label} missing upload step for {artifact_name!r}")
    next_step = job.find("\n      - ", name_index)
    block_end = next_step if next_step != -1 else len(job)
    return job[block_start:block_end]


def assert_upload_fail_closed(job: str, artifact_name: str, *, label: str) -> str:
    block = upload_block(job, artifact_name, label=label)
    assert_contains(block, "uses: actions/upload-artifact@v4", label=label)
    assert_contains(block, "if-no-files-found: error", label=label)
    return block


def comparison_directory(row: dict[str, str]) -> str:
    return str(Path(row["artifact_pattern"]).parent)


def workflow_artifact_name(row: dict[str, str], platform: str) -> str:
    artifacts = split_manifest_values(row["workflow_artifact"])
    platforms = split_manifest_values(row["workflow_platforms"])
    if len(artifacts) == 1:
        return artifacts[0]
    if len(artifacts) != len(platforms):
        raise AssertionError(
            f"{row['target_id']} has mismatched workflow_artifact/workflow_platforms"
        )
    if platform not in platforms:
        raise AssertionError(f"{row['target_id']} missing workflow platform {platform!r}")
    return artifacts[platforms.index(platform)]


def shared_workflow_artifact_name(rows: list[dict[str, str]], platform: str) -> str:
    names = {workflow_artifact_name(row, platform) for row in rows}
    if len(names) != 1:
        raise AssertionError(f"expected one shared {platform} workflow artifact, got {names}")
    return names.pop()


def assert_selected_targets(job: str, rows: list[dict[str, str]], *, label: str) -> None:
    for row in rows:
        assert_contains(
            job,
            f'("{row["target_key"]}", Path("{comparison_directory(row)}"), '
            f"{expected_int(row, 'expected_rows')})",
            label=label,
        )


def assert_comparison_upload_paths(
    job: str,
    rows: list[dict[str, str]],
    artifact_name: str,
    *,
    label: str,
) -> None:
    block = assert_upload_fail_closed(job, artifact_name, label=label)
    if "build/comparison/**" in block:
        raise AssertionError(f"{label} must not use a broad comparison upload path")
    for row in rows:
        directory = comparison_directory(row)
        for filename in split_manifest_values(row["required_files"]):
            assert_contains(block, f"{directory}/{filename}", label=label)


def assert_summary_fail_closed(job: str, *, label: str) -> None:
    for needle in [
        "uploaded_files = [",
        "missing uploaded artifact",
        "expected {expected_rows} selected rows",
        "expected {expected_rows} pass rows",
        "missing manifest {key}",
        "selected_targets={len(targets)}",
        "total_selected_rows={total_rows}",
        "total_pass_rows={total_pass_rows}",
        '"source_commit", "source_branch", "platform"',
    ]:
        assert_contains(job, needle, label=label)


def assert_linux_guard_runs_outside_validated_lane(text: str) -> None:
    build_job = job_block(text, "build-and-test", label="linux")
    freshness_job = job_block(text, "generated-report-freshness", label="linux")
    guard_step = "Run selected comparison workflow guard"
    guard_command = "python3 tests/test_selected_comparison_workflow.py"

    assert_contains(build_job, guard_step, label="linux build-and-test")
    assert_contains(build_job, guard_command, label="linux build-and-test")
    if guard_step in freshness_job or guard_command in freshness_job:
        raise AssertionError("linux generated-report-freshness must not host its own guard")


def assert_no_report_freshness_lane(text: str, *, label: str) -> None:
    for needle in WINDOWS_FORBIDDEN_SELECTED_FRESHNESS:
        if needle in text:
            raise AssertionError(
                f"{label} must not run or upload selected report freshness {needle!r}"
            )


def assert_windows_workflow_contract(text: str) -> None:
    build_job = job_block(text, "build-and-test", label="windows")
    install_job = job_block(text, "install-and-downstream", label="windows")
    assert_contains(
        build_job,
        "Run enforced reviewed CMake configure path",
        label="windows build-and-test",
    )
    assert_contains(
        install_job,
        "Run reviewed CMake install/downstream validation proof",
        label="windows install-and-downstream",
    )
    assert_contains(
        text,
        "Sprint 182 formally defers Windows report freshness",
        label="windows workflow",
    )
    assert_contains(
        text,
        "generated report freshness",
        label="windows workflow",
    )


def assert_no_windows_selected_manifest_platform() -> None:
    for row in selected_report_targets(SELECTED_TARGET_MANIFEST):
        platforms = split_manifest_values(row["workflow_platforms"])
        if "windows" in platforms:
            raise AssertionError(
                "selected_report_targets.tsv must not list windows for selected "
                f"report freshness while Sprint 182 deferral is active: {row['target_id']}"
            )


def assert_windows_deferral_record(path: Path = WINDOWS_DEFERRAL_RECORD) -> None:
    try:
        text = read_text(path)
    except FileNotFoundError as exc:
        raise AssertionError(f"windows deferral record missing file: {path}") from exc
    for needle in WINDOWS_DEFERRAL_REQUIRED_TEXT:
        assert_contains(text, needle, label="windows deferral record")


def test_linux_selected_oracle_lane() -> None:
    text = read_text(LINUX_WORKFLOW)
    job = job_block(text, "generated-report-freshness", label="linux")
    row = single_row(LINUX_WORKFLOW, "generated-report-freshness", "oracle")
    artifact_name = workflow_artifact_name(row, "linux")

    assert_contains(job, "Linux reviewed hosted oracle/comparison freshness", label="linux")
    assert_contains(job, "Run reviewed hosted oracle freshness", label="linux")
    assert_contains(job, row["generator_command"], label="linux")
    block = assert_upload_fail_closed(job, artifact_name, label="linux oracle upload")
    for path in ORACLE_UPLOAD_PATHS:
        assert_contains(block, path, label="linux oracle upload")
    if "build/corpus/**" in block or "build/report-index/**" in block:
        raise AssertionError("linux oracle upload must not use broad generated-report paths")


def test_linux_selected_comparison_lane() -> None:
    text = read_text(LINUX_WORKFLOW)
    job = job_block(text, "generated-report-freshness", label="linux")
    rows = comparison_rows(LINUX_WORKFLOW, "generated-report-freshness")
    artifact_name = shared_workflow_artifact_name(rows, "linux")

    assert_contains(job, "Run reviewed hosted selected comparison freshness", label="linux")
    assert_linux_guard_runs_outside_validated_lane(text)
    assert_contains(job, "make report-index-comparison-freshness", label="linux")
    assert_selected_targets(job, rows, label="linux")
    assert_comparison_upload_paths(
        job,
        rows,
        artifact_name,
        label="linux selected comparison upload",
    )
    assert_summary_fail_closed(job, label="linux")


def test_linux_selected_performance_lane() -> None:
    text = read_text(LINUX_WORKFLOW)
    job = job_block(text, "hosted-performance-freshness", label="linux")
    row = single_row(LINUX_WORKFLOW, "hosted-performance-freshness", "benchmark")
    artifact_name = workflow_artifact_name(row, "linux")
    selected_artifact = split_manifest_values(row["expected_row_ids"])[0]
    selected_path = row["artifact_pattern"]

    assert_contains(job, "Linux reviewed hosted selected performance freshness", label="linux")
    assert_contains(job, "Run reviewed hosted selected performance report", label="linux")
    assert_contains(job, "make bench-canonical-report", label="linux")
    assert_contains(job, "check_bench_canonical_freshness.py", label="linux")
    assert_contains(job, f'row["artifact"] == "{selected_artifact}"', label="linux")
    block = assert_upload_fail_closed(job, artifact_name, label="linux performance upload")
    assert_contains(block, selected_path, label="linux performance upload")
    for path in BENCHMARK_UNSELECTED_CONTEXT_FILES:
        assert_contains(block, path, label="linux performance upload")
    if "build/bench-reports/**" in block or "build/bench-reports/canonical/**" in block:
        raise AssertionError("linux performance upload must not use broad benchmark paths")


def test_macos_selected_comparison_lane() -> None:
    text = read_text(MACOS_WORKFLOW)
    job = job_block(text, "selected-comparison-freshness", label="macos")
    rows = comparison_rows(MACOS_WORKFLOW, "selected-comparison-freshness")
    artifact_name = shared_workflow_artifact_name(rows, "macos")

    assert_contains(job, "macOS reviewed selected comparison freshness", label="macos")
    assert_contains(job, "Run reviewed selected comparison freshness", label="macos")
    assert_contains(job, "make report-index-comparison-freshness", label="macos")
    assert_selected_targets(job, rows, label="macos")
    assert_comparison_upload_paths(
        job,
        rows,
        artifact_name,
        label="macos selected comparison upload",
    )
    assert_summary_fail_closed(job, label="macos")
    for non_claim in [
        "Windows report freshness",
        "external-library parity",
        "package/ABI support",
        "performance superiority",
        "state-of-the-art status",
    ]:
        assert_contains(text, non_claim, label="macos")


def test_windows_report_freshness_remains_formally_deferred() -> None:
    text = read_text(WINDOWS_WORKFLOW)
    assert_windows_workflow_contract(text)
    assert_no_report_freshness_lane(text, label="windows")
    assert_no_windows_selected_manifest_platform()
    assert_windows_deferral_record()
    assert_contains(text, "Windows does not claim Makefile parity", label="windows")
    assert_contains(text, "package-manager support", label="windows")


def test_windows_drift_selected_command_fails_clearly() -> None:
    text = read_text(WINDOWS_WORKFLOW)
    drifted = text + "\n# drift\nrun: make report-index-comparison-freshness\n"
    assert_raises_with(
        lambda: assert_no_report_freshness_lane(drifted, label="windows"),
        "windows must not run or upload selected report freshness "
        "'report-index-comparison-freshness'",
    )


def test_windows_drift_selected_artifact_fails_clearly() -> None:
    text = read_text(WINDOWS_WORKFLOW)
    drifted = text + "\n# drift\nname: sprint175-macos-selected-comparison-freshness\n"
    assert_raises_with(
        lambda: assert_no_report_freshness_lane(drifted, label="windows"),
        "windows must not run or upload selected report freshness "
        "'sprint175-macos-selected-comparison-freshness'",
    )


def test_windows_deferral_record_missing_blocker_fails_clearly() -> None:
    original = read_text(WINDOWS_DEFERRAL_RECORD)
    drifted = original.replace("no Windows-native canonical benchmark report generator", "")
    assert_raises_with(
        lambda: [
            assert_contains(drifted, needle, label="windows deferral record")
            for needle in WINDOWS_DEFERRAL_REQUIRED_TEXT
        ],
        "windows deferral record missing "
        "'no Windows-native canonical benchmark report generator'",
    )


def test_windows_deferral_record_missing_file_fails_clearly() -> None:
    missing_path = WINDOWS_DEFERRAL_RECORD.with_name("missing-windows-deferral.md")
    assert_raises_with(
        lambda: assert_windows_deferral_record(missing_path),
        f"windows deferral record missing file: {missing_path}",
    )


def test_windows_workflow_missing_reviewed_job_fails_clearly() -> None:
    text = read_text(WINDOWS_WORKFLOW)
    job = job_block(text, "install-and-downstream", label="windows")
    drifted = text.replace(job, "", 1)
    assert_raises_with(
        lambda: assert_windows_workflow_contract(drifted),
        "windows missing job 'install-and-downstream'",
    )


def test_windows_workflow_missing_deferral_comment_fails_clearly() -> None:
    text = read_text(WINDOWS_WORKFLOW)
    drifted = text.replace(
        "# Sprint 182 formally defers Windows report freshness; this workflow must stay\n",
        "",
        1,
    )
    assert_raises_with(
        lambda: assert_windows_workflow_contract(drifted),
        "windows workflow missing 'Sprint 182 formally defers Windows report freshness'",
    )


def test_workflow_drift_missing_job_fails_clearly() -> None:
    text = "jobs:\n  build-and-test:\n    steps: []\n"
    assert_raises_with(
        lambda: job_block(text, "generated-report-freshness", label="linux"),
        "linux missing job 'generated-report-freshness'",
    )


def test_workflow_drift_wrong_upload_artifact_fails_clearly() -> None:
    text = read_text(LINUX_WORKFLOW)
    job = job_block(text, "generated-report-freshness", label="linux")
    rows = comparison_rows(LINUX_WORKFLOW, "generated-report-freshness")
    artifact_name = shared_workflow_artifact_name(rows, "linux")
    drifted = job.replace(
        f"          name: {artifact_name}",
        "          name: drifted-selected-comparison-freshness",
        1,
    )
    assert_raises_with(
        lambda: assert_upload_fail_closed(
            drifted,
            artifact_name,
            label="linux selected comparison upload",
        ),
        "linux selected comparison upload missing upload artifact name",
    )


def test_workflow_drift_missing_fail_closed_setting_fails_clearly() -> None:
    text = read_text(LINUX_WORKFLOW)
    job = job_block(text, "generated-report-freshness", label="linux")
    rows = comparison_rows(LINUX_WORKFLOW, "generated-report-freshness")
    artifact_name = shared_workflow_artifact_name(rows, "linux")
    block = upload_block(job, artifact_name, label="linux selected comparison upload")
    drifted_block = block.replace("          if-no-files-found: error\n", "", 1)
    drifted = job.replace(block, drifted_block, 1)
    assert_raises_with(
        lambda: assert_upload_fail_closed(
            drifted,
            artifact_name,
            label="linux selected comparison upload",
        ),
        "linux selected comparison upload missing 'if-no-files-found: error'",
    )


def test_workflow_drift_broad_comparison_upload_fails_clearly() -> None:
    text = read_text(LINUX_WORKFLOW)
    job = job_block(text, "generated-report-freshness", label="linux")
    rows = comparison_rows(LINUX_WORKFLOW, "generated-report-freshness")
    artifact_name = shared_workflow_artifact_name(rows, "linux")
    drifted = job.replace(
        "            build/comparison/qr_minnorm/project_observations.tsv",
        "            build/comparison/**",
        1,
    )
    assert_raises_with(
        lambda: assert_comparison_upload_paths(
            drifted,
            rows,
            artifact_name,
            label="linux selected comparison upload",
        ),
        "linux selected comparison upload must not use a broad comparison upload path",
    )


def test_workflow_drift_missing_required_upload_file_fails_clearly() -> None:
    text = read_text(LINUX_WORKFLOW)
    job = job_block(text, "generated-report-freshness", label="linux")
    rows = comparison_rows(LINUX_WORKFLOW, "generated-report-freshness")
    artifact_name = shared_workflow_artifact_name(rows, "linux")
    drifted = job.replace(
        "            build/comparison/qr_minnorm/project_observations.tsv\n",
        "",
        1,
    )
    assert_raises_with(
        lambda: assert_comparison_upload_paths(
            drifted,
            rows,
            artifact_name,
            label="linux selected comparison upload",
        ),
        "linux selected comparison upload missing "
        "'build/comparison/qr_minnorm/project_observations.tsv'",
    )


def main() -> int:
    test_linux_selected_oracle_lane()
    test_linux_selected_comparison_lane()
    test_linux_selected_performance_lane()
    test_macos_selected_comparison_lane()
    test_windows_report_freshness_remains_formally_deferred()
    test_windows_drift_selected_command_fails_clearly()
    test_windows_drift_selected_artifact_fails_clearly()
    test_windows_deferral_record_missing_blocker_fails_clearly()
    test_windows_deferral_record_missing_file_fails_clearly()
    test_windows_workflow_missing_reviewed_job_fails_clearly()
    test_windows_workflow_missing_deferral_comment_fails_clearly()
    test_workflow_drift_missing_job_fails_clearly()
    test_workflow_drift_wrong_upload_artifact_fails_clearly()
    test_workflow_drift_missing_fail_closed_setting_fails_clearly()
    test_workflow_drift_broad_comparison_upload_fails_clearly()
    test_workflow_drift_missing_required_upload_file_fails_clearly()
    print("test-selected-comparison-workflow: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
