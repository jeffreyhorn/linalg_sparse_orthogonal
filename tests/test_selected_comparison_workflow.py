#!/usr/bin/env python3
"""Guard selected comparison freshness workflow integration."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LINUX_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MACOS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "macos-ci.yml"

SELECTED_TARGETS = [
    ("qr-minnorm", "build/comparison/qr_minnorm", 6),
    ("qr-compatible-ls", "build/comparison/qr_compatible_ls", 6),
    ("partial-svd-diag6-k2", "build/comparison/partial_svd_diag6_k2", 10),
    ("lu-nonsym-square-5", "build/comparison/lu_nonsym_square_5", 6),
]
SELECTED_FILES = [
    "project_observations.tsv",
    "baseline_observations.tsv",
    "dependency_status.tsv",
    "study.tsv",
    "summary.md",
    "manifest.tsv",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_contains(text: str, needle: str, *, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} missing {needle!r}")


def assert_selected_targets(text: str, *, label: str) -> None:
    for target, directory, expected_rows in SELECTED_TARGETS:
        assert_contains(
            text,
            f'("{target}", Path("{directory}"), {expected_rows})',
            label=label,
        )


def upload_block(text: str, artifact_name: str, *, label: str) -> str:
    name_index = text.find(f"name: {artifact_name}")
    if name_index == -1:
        raise AssertionError(f"{label} missing upload artifact name {artifact_name!r}")
    block_start = text.rfind("- name:", 0, name_index)
    if block_start == -1:
        raise AssertionError(f"{label} missing upload step for {artifact_name!r}")
    next_step = text.find("\n      - ", name_index)
    next_job_match = re.search(r"\n  [A-Za-z0-9_-]+:", text[name_index:])
    next_job = name_index + next_job_match.start() if next_job_match else -1
    candidates = [index for index in [next_step, next_job] if index != -1]
    block_end = min(candidates) if candidates else len(text)
    return text[block_start:block_end]


def assert_selected_upload_fail_closed(text: str, artifact_name: str, *, label: str) -> None:
    block = upload_block(text, artifact_name, label=label)
    assert_contains(block, "uses: actions/upload-artifact@v4", label=label)
    assert_contains(block, "if-no-files-found: error", label=label)
    for _target, directory, _expected_rows in SELECTED_TARGETS:
        for filename in SELECTED_FILES:
            assert_contains(block, f"{directory}/{filename}", label=label)


def assert_summary_fail_closed(text: str, *, label: str) -> None:
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
        assert_contains(text, needle, label=label)


def assert_linux_guard_runs_outside_validated_lane(text: str) -> None:
    build_job_start = text.index("  build-and-test:")
    freshness_job_start = text.index("  generated-report-freshness:")
    next_job_match = re.search(r"\n  [A-Za-z0-9_-]+:", text[build_job_start + 1 :])
    if next_job_match is None:
        raise AssertionError("linux build-and-test missing following job boundary")
    next_job_start = build_job_start + 1 + next_job_match.start()
    build_job_end = min(next_job_start, freshness_job_start)
    guard_step = "Run selected comparison workflow guard"
    guard_command = "python3 tests/test_selected_comparison_workflow.py"
    build_job = text[build_job_start:build_job_end]
    freshness_job = text[freshness_job_start:]

    assert_contains(build_job, guard_step, label="linux build-and-test")
    assert_contains(build_job, guard_command, label="linux build-and-test")
    if guard_step in freshness_job or guard_command in freshness_job:
        raise AssertionError("linux generated-report-freshness must not host its own guard")


def test_linux_selected_comparison_lane() -> None:
    text = read_text(LINUX_WORKFLOW)
    assert_contains(text, "Linux reviewed hosted oracle/comparison freshness", label="linux")
    assert_contains(text, "Run reviewed hosted selected comparison freshness", label="linux")
    assert_linux_guard_runs_outside_validated_lane(text)
    assert_contains(text, "make report-index-comparison-freshness", label="linux")
    assert_selected_targets(text, label="linux")
    assert_selected_upload_fail_closed(
        text,
        "sprint175-linux-selected-comparison-freshness",
        label="linux selected comparison upload",
    )
    assert_summary_fail_closed(text, label="linux")


def test_macos_selected_comparison_lane() -> None:
    text = read_text(MACOS_WORKFLOW)
    assert_contains(text, "macOS reviewed selected comparison freshness", label="macos")
    assert_contains(text, "Run reviewed selected comparison freshness", label="macos")
    assert_contains(text, "make report-index-comparison-freshness", label="macos")
    assert_selected_targets(text, label="macos")
    assert_selected_upload_fail_closed(
        text,
        "sprint175-macos-selected-comparison-freshness",
        label="macos selected comparison upload",
    )
    assert_summary_fail_closed(text, label="macos")
    for non_claim in [
        "Windows report freshness",
        "external-library parity",
        "package/ABI support",
        "performance superiority",
        "state-of-the-art status",
    ]:
        assert_contains(text, non_claim, label="macos")


def main() -> int:
    test_linux_selected_comparison_lane()
    test_macos_selected_comparison_lane()
    print("test-selected-comparison-workflow: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
