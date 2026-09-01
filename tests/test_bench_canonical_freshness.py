#!/usr/bin/env python3
"""Regression tests for selected canonical benchmark freshness checks."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_bench_canonical_freshness as checker  # noqa: E402
from normalize_report_index import (  # noqa: E402
    expected_int,
    selected_report_targets,
    split_manifest_values,
)


CHECKER = REPO_ROOT / "scripts" / "check_bench_canonical_freshness.py"
REPORT_DIR = REPO_ROOT / "build" / "bench-reports" / "canonical"
SELECTED_TARGET_ID = "SRT-BENCH-REFACTOR-CSC-NOS4"


def selected_benchmark_target() -> dict[str, str]:
    rows = selected_report_targets(REPO_ROOT / "tests" / "corpus")
    selected = [row for row in rows if row["target_id"] == SELECTED_TARGET_ID]
    if len(selected) != 1:
        raise AssertionError(
            f"expected exactly one {SELECTED_TARGET_ID} row, found {len(selected)}"
        )
    return selected[0]


SELECTED_ARTIFACT = checker.selected_benchmark_artifact(selected_benchmark_target())


def run_command(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    command_env = os.environ.copy()
    command_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        command_env.update(env)
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=command_env,
        text=True,
        capture_output=True,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(args)}")
    return result


def run_checker(report_dir: Path, mode: str = "local", *, expect_success: bool = True) -> str:
    result = run_command(
        [
            "python3",
            str(CHECKER),
            "--report-dir",
            str(report_dir),
            "--mode",
            mode,
        ],
        expect_success=expect_success,
    )
    return result.stdout + result.stderr


def generate_local_report() -> None:
    run_command(["make", "bench-canonical-report-freshness"])


def generate_hosted_report() -> None:
    run_command(
        ["make", "bench-canonical-report"],
        env={
            "BENCH_CANONICAL_REPORT_LABEL": "sprint-169-freshness-test",
            "SPARSE_CANONICAL_SUPPORT_TIER": "hosted_selected",
            "SPARSE_CANONICAL_CLAIM_BOUNDARY": "hosted_selected_threshold_free",
            "SPARSE_CANONICAL_RUNNER_CONTEXT": "github-actions-ubuntu-latest",
            "SPARSE_CANONICAL_BUILD_FLAGS": "default_make_flags",
            "SPARSE_CANONICAL_CPU_MODEL": "unknown",
            "SPARSE_CANONICAL_BUILD_MODE": "serial",
        },
    )


def copy_report(tmp_path: Path) -> Path:
    copied = tmp_path / "canonical"
    shutil.copytree(REPORT_DIR, copied)
    return copied


def read_index(report_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    with (report_dir / "index.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_index(report_dir: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with (report_dir / "index.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def mutate_selected_field(report_dir: Path, field: str, value: str) -> None:
    fieldnames, rows = read_index(report_dir)
    for row in rows:
        if row["artifact"] == SELECTED_ARTIFACT:
            row[field] = value
    write_index(report_dir, fieldnames, rows)


def mutate_artifact_field(report_dir: Path, artifact: str, field: str, value: str) -> None:
    fieldnames, rows = read_index(report_dir)
    for row in rows:
        if row["artifact"] == artifact:
            row[field] = value
    write_index(report_dir, fieldnames, rows)


def mutate_manifest_value(report_dir: Path, key: str, value: str) -> None:
    path = report_dir / "manifest.txt"
    lines = []
    replaced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            lines.append(f"{key}={value}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remove_selected_row_field(report_dir: Path) -> None:
    lines = (report_dir / "index.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0]
    data = lines[1:]
    mutated = []
    for line in data:
        if "\tbench_refactor_csc\t" in line:
            mutated.append("\t".join(line.split("\t")[:-1]))
        else:
            mutated.append(line)
    (report_dir / "index.tsv").write_text(
        "\n".join([header, *mutated]) + "\n",
        encoding="utf-8",
    )


def assert_fails_with(report_dir: Path, expected: str, mode: str = "local") -> None:
    output = run_checker(report_dir, mode, expect_success=False)
    if expected not in output:
        raise AssertionError(f"expected {expected!r} in checker output:\n{output}")


def selected_index_row(report_dir: Path) -> dict[str, str]:
    _fieldnames, rows = read_index(report_dir)
    selected = [row for row in rows if row["artifact"] == SELECTED_ARTIFACT]
    if len(selected) != 1:
        raise AssertionError(f"expected one selected row, found {len(selected)}")
    return selected[0]


def selected_benchmark_csv_row(report_dir: Path) -> dict[str, str]:
    path = report_dir / "bench_refactor_csc.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise AssertionError(f"expected one bench_refactor_csc row, found {len(rows)}")
    return rows[0]


def write_selected_benchmark_csv(report_dir: Path, rows: list[dict[str, str]]) -> None:
    path = report_dir / "bench_refactor_csc.csv"
    if not rows:
        path.write_text("benchmark,matrix,n,scenario\n", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mutate_selected_benchmark_csv_field(report_dir: Path, field: str, value: str) -> None:
    path = report_dir / "bench_refactor_csc.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if field not in fieldnames:
        raise AssertionError(f"selected benchmark CSV missing field {field!r}")
    for row in rows:
        row[field] = value
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remove_selected_benchmark_csv_column(report_dir: Path, field: str) -> None:
    path = report_dir / "bench_refactor_csc.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = [name for name in list(reader.fieldnames or []) if name != field]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def test_positive_local_report() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        output = run_checker(report)
        assert "bench-canonical-freshness: passed" in output


def test_selected_benchmark_manifest_matches_checker_contract() -> None:
    target = selected_benchmark_target()
    assert target["family"] == "benchmark"
    assert target["subfamily"] == "canonical"
    assert target["target_key"] == SELECTED_ARTIFACT
    assert target["support_tier"] == checker.HOSTED_SUPPORT_TIER
    assert target["freshness_policy"] == "generated_local_advisory"
    assert target["generator_command"] == "make bench-canonical-report-freshness"
    assert checker.selected_benchmark_relative_path(target) == "bench_refactor_csc.csv"
    assert tuple(split_manifest_values(target["required_files"])) == checker.required_artifacts(
        target
    )
    assert target["workflow_artifact"] == "sprint168-selected-performance-freshness"
    assert expected_int(target, "expected_rows") == 1
    assert "threshold-free methodology fields" in target["claim_scope"]
    for non_claim in [
        "no portable performance claim",
        "no release benchmark claim",
        "no algorithmic superiority claim",
        "no platform parity",
        "no state-of-the-art claim",
        "no package or ABI support claim",
    ]:
        assert non_claim in split_manifest_values(target["non_claims"])


def test_selected_benchmark_csv_matches_index_fixture_contract() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        index_row = selected_index_row(report)
        csv_row = selected_benchmark_csv_row(report)
        assert index_row["artifact"] == csv_row["benchmark"] == "bench_refactor_csc"
        assert index_row["fixture_or_workload"] == csv_row["matrix"] == "nos4.mtx"
        assert index_row["command"] == "tests/data/suitesparse/nos4.mtx --repeat 1"
        assert index_row["matrix_size"] == f"n={csv_row['n']}" == "n=100"
        assert index_row["repeat_semantics"] == "configured_repeat_1"
        assert csv_row["scenario"] == "chol_spd"
        assert csv_row["nnz"] == "594"
        assert csv_row["ldlt_dense_backend_request"] == "n/a"
        assert csv_row["ldlt_dense_backend_selected"] == "n/a"
        assert csv_row["ldlt_dense_backend_fallback"] == "n/a"


def test_selected_benchmark_csv_wrong_fixture_fails() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_selected_benchmark_csv_field(report, "matrix", "bcsstk14.mtx")
        assert_fails_with(
            report,
            "benchmark_selected_csv_value: artifact=bench_refactor_csc.csv "
            "field=matrix expected=nos4.mtx observed=bcsstk14.mtx",
        )


def test_selected_benchmark_csv_missing_required_column_fails() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        remove_selected_benchmark_csv_column(report, "scenario")
        assert_fails_with(
            report,
            "benchmark_selected_csv_schema: artifact=bench_refactor_csc.csv "
            "missing_columns=scenario",
        )


def test_selected_benchmark_csv_extra_row_fails() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        row = selected_benchmark_csv_row(report)
        write_selected_benchmark_csv(report, [row, row])
        assert_fails_with(
            report,
            "benchmark_selected_csv_rows: artifact=bench_refactor_csc.csv "
            "expected_rows=1 observed_rows=2",
        )


def test_selected_matrix_size_is_required() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_selected_field(report, "matrix_size", "not_recorded")
        assert_fails_with(report, "field=matrix_size expected=n=100 observed=not_recorded")


def test_selected_warmup_is_required() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_selected_field(report, "warmup", "not_recorded")
        assert_fails_with(report, "field=warmup expected=none_configured observed=not_recorded")


def test_selected_variance_is_required() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_selected_field(report, "variance", "not_recorded")
        assert_fails_with(
            report,
            "field=variance expected=not_computed_single_sample observed=not_recorded",
        )


def test_selected_baseline_stays_threshold_free() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_selected_field(report, "baseline", "100.0")
        assert_fails_with(report, "field=baseline expected=n/a observed=100.0")


def test_selected_threshold_stays_threshold_free() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_selected_field(report, "threshold", "200.0")
        assert_fails_with(report, "field=threshold expected=n/a observed=200.0")


def test_selected_status_cannot_become_performance_pass_claim() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_selected_field(report, "status", "pass")
        assert_fails_with(report, "field=status expected=measurement observed=pass")


def test_manifest_selected_matrix_size_must_match() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_manifest_value(report, "selected_matrix_size", "n=101")
        assert_fails_with(report, "field=matrix_size row=n=100 manifest=n=101")


def test_row_width_mismatch_is_rejected() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        remove_selected_row_field(report)
        assert_fails_with(report, "freshness: error: benchmark_selected_schema")


def test_unselected_rows_cannot_be_hosted_selected() -> None:
    generate_local_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        mutate_artifact_field(report, "bench_chol_csc", "support_tier", "hosted_selected")
        assert_fails_with(report, "freshness: error: benchmark_unselected_claim_boundary")


def test_positive_hosted_report_keeps_unselected_rows_local() -> None:
    generate_hosted_report()
    with tempfile.TemporaryDirectory() as tmp:
        report = copy_report(Path(tmp))
        output = run_checker(report, "hosted")
        assert "bench-canonical-freshness: passed" in output
        _fieldnames, rows = read_index(report)
        for row in rows:
            if row["artifact"] == SELECTED_ARTIFACT:
                assert row["support_tier"] == "hosted_selected"
                assert row["claim_boundary"] == "hosted_selected_threshold_free"
            else:
                assert row["support_tier"] == "local_only"
                assert row["claim_boundary"] == "local_threshold_free"


def test_generator_rejects_tsv_control_characters_in_methodology_metadata() -> None:
    output = run_command(
        ["make", "bench-canonical-report"],
        env={"BENCH_CANONICAL_REPORT_LABEL": "bad\tlabel"},
        expect_success=False,
    )
    combined = output.stdout + output.stderr
    if "BENCH_CANONICAL_REPORT_LABEL must not contain tabs or newlines" not in combined:
        raise AssertionError(f"expected control-character failure:\n{combined}")


def main() -> None:
    tests = [
        test_positive_local_report,
        test_selected_benchmark_manifest_matches_checker_contract,
        test_selected_benchmark_csv_matches_index_fixture_contract,
        test_selected_benchmark_csv_wrong_fixture_fails,
        test_selected_benchmark_csv_missing_required_column_fails,
        test_selected_benchmark_csv_extra_row_fails,
        test_selected_matrix_size_is_required,
        test_selected_warmup_is_required,
        test_selected_variance_is_required,
        test_selected_baseline_stays_threshold_free,
        test_selected_threshold_stays_threshold_free,
        test_selected_status_cannot_become_performance_pass_claim,
        test_manifest_selected_matrix_size_must_match,
        test_row_width_mismatch_is_rejected,
        test_unselected_rows_cannot_be_hosted_selected,
        test_positive_hosted_report_keeps_unselected_rows_local,
        test_generator_rejects_tsv_control_characters_in_methodology_metadata,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: passed")


if __name__ == "__main__":
    main()
