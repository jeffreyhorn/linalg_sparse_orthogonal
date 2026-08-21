#!/usr/bin/env python3
"""Focused tests for the normalized report index generator."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "normalize_report_index.py"
ORACLE_SCRIPT = REPO_ROOT / "scripts" / "run_corpus_oracle.py"
CORPUS_ROOT = REPO_ROOT / "tests" / "corpus"
REPORT_FAMILIES = CORPUS_ROOT / "manifests" / "report_families.tsv"
SPRINT151_PARTIAL_SVD_ROW_COUNTS = {
    "partial_svd_rankdef_diag6x4_k2_range_projector_v1": 7,
    "partial_svd_lowrank_rect5x7_k3_sparse_output_v1": 6,
    "partial_svd_fail_closed_diag6_k2_v1": 5,
}
SELECTED_ORACLE_FIXTURE_KEYS = {
    "partial_svd_clustered_repeated_diag8x6_k3_v1",
    "partial_svd_fail_closed_diag6_k2_v1",
    "partial_svd_lowrank_rect5x7_k3_sparse_output_v1",
    "partial_svd_rankdef_diag6x4_k2_range_projector_v1",
    "qr_minnorm_3x6_exact_values",
    "qr_minnorm_5x10_exact_values",
    "qr_rank_deficient_6x4_nullspace_v1",
    "qr_rankdef_dependent_row_4x3_v1",
    "qr_rankdef_duplicate_5x4_v1",
    "qr_underdetermined_minnorm_2x4",
}
SELECTED_COMPARISON_ROW_IDS = [
    "comparison_qr_underdetermined_minnorm_2x4_project_status_v1",
    "comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1",
    "comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1",
    "comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1",
    "comparison_qr_underdetermined_minnorm_2x4_solution_values_v1",
    "comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1",
    "comparison_qr_overdetermined_compatible_5x3_project_status_v1",
    "comparison_qr_overdetermined_compatible_5x3_baseline_status_v1",
    "comparison_qr_overdetermined_compatible_5x3_residual_norm_v1",
    "comparison_qr_overdetermined_compatible_5x3_solution_norm_v1",
    "comparison_qr_overdetermined_compatible_5x3_solution_values_v1",
    "comparison_qr_overdetermined_compatible_5x3_project_vs_baseline_max_abs_delta_v1",
    "comparison_partial_svd_diag6_k2_project_status_v1",
    "comparison_partial_svd_diag6_k2_baseline_status_v1",
    "comparison_partial_svd_diag6_k2_singular_value_0_v1",
    "comparison_partial_svd_diag6_k2_singular_value_1_v1",
    "comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1",
    "comparison_partial_svd_diag6_k2_residual_norm_v1",
    "comparison_partial_svd_diag6_k2_u_orthogonality_v1",
    "comparison_partial_svd_diag6_k2_v_orthogonality_v1",
    "comparison_partial_svd_diag6_k2_u_projector_diag_v1",
    "comparison_partial_svd_diag6_k2_v_projector_diag_v1",
    "comparison_lu_nonsym_square_5_project_status_v1",
    "comparison_lu_nonsym_square_5_baseline_status_v1",
    "comparison_lu_nonsym_square_5_residual_norm_v1",
    "comparison_lu_nonsym_square_5_solution_norm_v1",
    "comparison_lu_nonsym_square_5_solution_values_v1",
    "comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1",
]
SELECTED_PARTIAL_SVD_COMPARISON_ROW_IDS = {
    row_id for row_id in SELECTED_COMPARISON_ROW_IDS if "partial_svd_diag6_k2" in row_id
}
SELECTED_LU_COMPARISON_ROW_IDS = {
    row_id for row_id in SELECTED_COMPARISON_ROW_IDS if "lu_nonsym_square_5" in row_id
}
SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC = (
    "artifacts=build/comparison/qr_minnorm/study.tsv,"
    "build/comparison/qr_compatible_ls/study.tsv,"
    "build/comparison/partial_svd_diag6_k2/study.tsv,"
    "build/comparison/lu_nonsym_square_5/study.tsv"
)
ORACLE_FIELDS = [
    "oracle_row_id",
    "fixture_key",
    "solver_family",
    "operation",
    "comparison_kind",
    "command",
    "source_commit",
    "source_branch",
    "generated_at_utc",
    "platform",
    "compiler",
    "configuration",
    "support_tier",
    "expected_result_kind",
    "expected_result",
    "observed_result",
    "tolerance_kind",
    "tolerance_value",
    "comparison_status",
    "failure_class",
    "skip_or_defer_reason",
    "claim_scope",
    "non_claims",
]
COMPARISON_STUDY_FIELDS = [
    "comparison_row_id",
    "report_family",
    "subfamily",
    "row_kind",
    "fixture_key",
    "operation",
    "metric",
    "baseline_name",
    "baseline_type",
    "baseline_version",
    "baseline_command",
    "baseline_python_executable",
    "baseline_python_version",
    "project_name",
    "project_version",
    "project_command",
    "source_commit",
    "source_branch",
    "worktree_state",
    "platform",
    "compiler",
    "configuration",
    "expected_value",
    "project_value",
    "baseline_value",
    "delta_value",
    "tolerance_kind",
    "tolerance_value",
    "status",
    "status_reason",
    "caveat",
    "artifact_path",
    "generated_at_utc",
    "support_tier",
    "claim_scope",
    "non_claims",
]


def run_command(args: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True)
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(args)}")
    return result


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def assert_sorted(rows: list[dict[str, str]]) -> None:
    keys = [
        (
            row["report_family"],
            row["subfamily"],
            row["row_origin"],
            row["row_meaning"],
            row["native_row_id"],
            row["artifact_path"],
            row["row_id"],
        )
        for row in rows
    ]
    if keys != sorted(keys):
        raise AssertionError("normalized rows are not deterministically sorted")


def generated_oracle_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["row_id"].startswith("oracle_")]


def test_current_repo_no_generated() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "normalized-index.tsv"
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--no-generated",
                "--output",
                str(output),
            ]
        )
        rows = read_tsv(output)
        row_ids = {row["row_id"] for row in rows}
        assert "report_contract_runtime_backend_governance_runtime_backend_governance_policy_v1" in row_ids
        assert "corpus_fixture_qr_rank_deficient_6x4_nullspace_v1_v1" in row_ids
        assert "corpus_expected_qr_rank_deficient_6x4_nullspace_v1_rank_v1" in row_ids
        assert "corpus_optional_suitesparse_rank_deficient_qr_subset_v1_v1" in row_ids
        assert "report_missing_oracle_generated_reference_observed_oracle_comparison_v1" in row_ids
        assert "report_contract_package_static_install_package_install_proof_owner_v1" in row_ids
        assert_sorted(rows)
        for row in rows:
            if row["row_id"].startswith("report_contract_"):
                assert row["status"] != "pass"
                if row["status"] != "defer":
                    assert row["freshness_status"] == "source_controlled"
                assert row["non_claims"]


def test_git_metadata_is_independent_of_caller_cwd() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output = tmp_path / "normalized-index.tsv"
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--no-generated",
                "--output",
                str(output),
            ],
            cwd=tmp_path,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                "command failed outside repo root:"
                f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        rows = read_tsv(output)
        commits = {row["source_commit"] for row in rows}
        branches = {row["source_branch"] for row in rows}
        assert commits == {current_commit()}
        assert "unknown" not in branches


def test_family_filter_and_required_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "oracle.tsv"
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--family",
                "oracle",
                "--no-generated",
                "--output",
                str(output),
            ]
        )
        rows = read_tsv(output)
        assert {row["report_family"] for row in rows} == {"oracle"}
        assert any(row["freshness_status"] == "not_generated" for row in rows)

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--family",
                "oracle",
                "--no-generated",
                "--require-generated",
                "oracle",
                "--check",
            ],
            expect_success=False,
        )
        assert "required generated family missing: oracle" in result.stdout


def test_generated_artifact_presence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus_root = tmp_path / "corpus"
        build_root = tmp_path / "build"
        shutil.copytree(CORPUS_ROOT / "manifests", corpus_root / "manifests")
        artifact = build_root / "bench-reports" / "canonical" / "index.tsv"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("metric\tvalue\nplaceholder\t1\n")
        output = tmp_path / "index.tsv"

        run_command(
            [
                "python3",
                str(SCRIPT),
                "--corpus-root",
                str(corpus_root),
                "--build-root",
                str(build_root),
                "--family",
                "benchmark",
                "--output",
                str(output),
            ]
        )
        rows = read_tsv(output)
        assert any(row["freshness_status"] == "generated_present_unchecked" for row in rows)
        assert not any(row["freshness_status"] == "not_generated" for row in rows)


def write_runtime_fixture(build_root: Path) -> None:
    canonical = build_root / "bench-reports" / "canonical"
    canonical.mkdir(parents=True)
    (canonical / "index.tsv").write_text(
        "\t".join(
            [
                "surface",
                "category",
                "report_label",
                "generated_at_utc",
                "git_commit",
                "git_branch",
                "platform",
                "compiler",
                "build_mode",
                "omp_num_threads",
                "artifact",
                "relative_path",
                "command",
                "methodology_notes",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "canonical",
                "measurement",
                "test",
                "2026-08-07T00:00:00Z",
                "abc123",
                "sprint-141",
                "linux-x86_64",
                "cc",
                "serial",
                "unset",
                "bench_refactor_csc",
                "bench_refactor_csc.csv",
                "tests/data/suitesparse/nos4.mtx --repeat 1",
                "threshold_free_local_measurement;not_portable_performance_claim",
            ]
        )
        + "\n"
    )
    (canonical / "manifest.txt").write_text(
        "bench-canonical-report\n"
        "generated_at_utc=2026-08-07T00:00:00Z\n"
        "git_commit=abc123\n"
        "git_branch=sprint-141\n"
        "platform=linux-x86_64\n"
        "compiler=cc\n"
    )

    sentinels = build_root / "bench-reports" / "sentinels"
    sentinels.mkdir(parents=True)
    (sentinels / "sentinels.tsv").write_text(
        "\t".join(
            [
                "report_family",
                "sentinel_id",
                "status",
                "support_tier",
                "claim_boundary",
                "command",
                "build_mode",
                "omp_num_threads",
                "matrix_or_fixture",
                "metric",
                "value",
                "baseline",
                "threshold",
                "artifact",
                "backend_request",
                "backend_selected",
                "backend_fallback",
                "dense_kernel",
                "panel_solver",
                "notes",
                "methodology_notes",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "sentinel",
                "S5",
                "pass",
                "reviewed_thresholded",
                "local_wall_gate",
                "make wall-check",
                "serial",
                "unset",
                "bcsstk14",
                "qg_amd_reorder_ms",
                "1.0",
                "2.0",
                "2x",
                "wall_check.txt",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "existing_threshold_gate_passed",
                "thresholded_local_wall_gate;not_portable_performance_claim",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "sentinel",
                "S6",
                "pass",
                "reviewed_thresholded",
                "local_selected_regression_gate",
                "build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1",
                "serial",
                "unset",
                "nos4.mtx",
                "refactor_csc_ms",
                "0.5",
                "500.0",
                "500.0",
                "bench_refactor_csc_nos4.csv",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "selected_local_smoke_ceiling_passed",
                "selected_local_large_regression_gate;not_portable_performance_claim",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "sentinel",
                "S2",
                "report",
                "reviewed_threshold_free",
                "local_threshold_free",
                "build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1",
                "serial",
                "unset",
                "nos4.mtx",
                "factor_csc_ms",
                "3.0",
                "n/a",
                "n/a",
                "bench_chol_csc_nos4.csv",
                "unset",
                "builtin",
                "n/a",
                "builtin",
                "left_looking",
                "threshold_free",
                "threshold_free_local_backend_context;not_backend_superiority_claim",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "sentinel",
                "S3",
                "report",
                "reviewed_threshold_free",
                "local_threshold_free",
                "build/bench_refactor_csc --indefinite-kkt --repeat 1",
                "serial",
                "unset",
                "kkt-150",
                "refactor_csc_ms",
                "4.0",
                "n/a",
                "n/a",
                "bench_refactor_csc_kkt.csv",
                "external",
                "builtin",
                "yes",
                "n/a",
                "n/a",
                "threshold_free;ldlt_env=external;scenario=ldlt_kkt",
                "threshold_free_local_ldlt_backend_context;not_backend_superiority_claim",
            ]
        )
        + "\n"
    )
    (sentinels / "manifest.txt").write_text(
        "performance-sentinels\n"
        "generated_at_utc=2026-08-07T00:00:00Z\n"
        "git_commit=abc123\n"
        "git_branch=sprint-141\n"
        "platform=linux-x86_64\n"
        "compiler=cc\n"
    )

    guardrails = build_root / "bench-reports" / "large-matrix-guardrails"
    guardrails.mkdir(parents=True)
    (guardrails / "index.tsv").write_text(
        "lane_id\tstatus\tcategory\tcommand\tartifact\tnotes\n"
        "G1\tpass\treviewed\tbuild/test_reorder_amd_qg\tamd_qg.txt\tstructural guardrail\n"
        "S1\tskip\tsupplemental\tbuild/bench_reorder --skip-factor\tn/a\tset SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 to run\n"
    )
    (guardrails / "manifest.txt").write_text(
        "large-matrix-guardrails\n"
        "generated_at_utc=2026-08-07T00:00:00Z\n"
        "git_commit=abc123\n"
        "git_branch=sprint-141\n"
        "platform=linux-x86_64\n"
        "compiler=cc\n"
        "supplemental=0\n"
    )


def test_runtime_report_rows_preserve_boundaries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        write_runtime_fixture(build_root)
        output = tmp_path / "runtime-index.tsv"
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "benchmark",
                "--family",
                "sentinel",
                "--family",
                "guardrail",
                "--output",
                str(output),
            ]
        )
        rows = read_tsv(output)
        by_family = {}
        for row in rows:
            by_family.setdefault(row["report_family"], []).append(row)
        benchmark_rows = [
            row for row in by_family["benchmark"] if row["native_row_id"] == "bench_refactor_csc"
        ]
        assert len(benchmark_rows) == 1
        assert (
            "methodology_notes=threshold_free_local_measurement%3Bnot_portable_performance_claim"
            in benchmark_rows[0]["configuration"]
        )
        assert "not_portable_performance_claim" not in {
            part.split("=", 1)[0] for part in benchmark_rows[0]["configuration"].split(";")
        }
        assert any(
            row["row_meaning"] == "sentinel_hard_gate" and row["status"] == "pass"
            for row in by_family["sentinel"]
        )
        s6_rows = [
            row
            for row in by_family["sentinel"]
            if row["native_row_id"] == "S6_nos4.mtx_refactor_csc_ms"
        ]
        assert len(s6_rows) == 1
        assert s6_rows[0]["row_meaning"] == "sentinel_hard_gate"
        assert "claim_boundary=local_selected_regression_gate" in s6_rows[0]["configuration"]
        assert "threshold=500.0" in s6_rows[0]["configuration"]
        assert "not_portable_performance_claim" in s6_rows[0]["configuration"]
        assert any(
            row["row_meaning"] == "sentinel_advisory_measurement"
            and row["status"] == "advisory"
            for row in by_family["sentinel"]
        )
        s3_rows = [
            row for row in by_family["sentinel"] if row["native_row_id"] == "S3_kkt-150_refactor_csc_ms"
        ]
        assert len(s3_rows) == 1
        assert s3_rows[0]["status"] == "advisory"
        assert "backend_request=external" in s3_rows[0]["configuration"]
        assert "backend_selected=builtin" in s3_rows[0]["configuration"]
        assert "backend_fallback=yes" in s3_rows[0]["configuration"]
        assert (
            "methodology_notes=threshold_free_local_ldlt_backend_context%3B"
            "not_backend_superiority_claim"
            in s3_rows[0]["configuration"]
        )
        assert "not_backend_superiority_claim" not in {
            part.split("=", 1)[0] for part in s3_rows[0]["configuration"].split(";")
        }
        assert any(row["native_row_id"] == "G1" and row["status"] == "pass" for row in by_family["guardrail"])
        assert any(row["native_row_id"] == "S1" and row["status"] == "skip" for row in by_family["guardrail"])
        for row in rows:
            if row["row_id"].startswith(("benchmark_", "sentinel_", "guardrail_")):
                assert row["freshness_status"] == "generated_present_unchecked"
                assert row["non_claims"]
                assert (
                    "state-of-the-art claim" in row["non_claims"]
                    or "performance" in row["non_claims"]
                    or "platform" in row["non_claims"]
                )


def test_quality_and_package_rows_preserve_scope() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        deadcode_dir = build_root / "deadcode"
        deadcode_dir.mkdir(parents=True)
        (deadcode_dir / "report.tsv").write_text(
            "bucket\ttool\tsymbol\tpath\tline\tdetail\tdisposition\n"
            "coverage-gap\tcoverage-notes\tbench_missing\tbenchmarks\t\tAbsent from compile database\tdefer-until-compile-db-expanded\n"
            "secondary-candidate-signal\tcppcheck\t\tsrc/example.c\t\tunusedFunction count=1\tsummarize-only-supporting-evidence\n"
        )
        output = tmp_path / "quality-index.tsv"
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "deadcode",
                "--family",
                "package",
                "--family",
                "coverage",
                "--no-generated",
                "--output",
                str(output),
            ]
        )
        no_generated_rows = read_tsv(output)
        assert any(
            row["report_family"] == "coverage"
            and row["freshness_status"] == "not_generated"
            for row in no_generated_rows
        )
        assert any(row["row_id"] == "package_make_install_pkg_config_v1" for row in no_generated_rows)
        assert any(row["row_id"] == "package_cmake_install_export_v1" for row in no_generated_rows)
        for row in no_generated_rows:
            if row["row_id"].startswith("package_"):
                assert row["status"] == "advisory"
                assert "shared-library ABI support" in row["non_claims"]
                assert row["freshness_status"] == "source_controlled"

        generated_output = tmp_path / "quality-generated-index.tsv"
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "deadcode",
                "--output",
                str(generated_output),
            ]
        )
        generated_rows = read_tsv(generated_output)
        assert any(row["report_family"] == "deadcode" for row in generated_rows)
        assert any(row["native_row_id"].startswith("coverage-gap_coverage-notes") for row in generated_rows)
        for row in generated_rows:
            if row["row_id"].startswith("deadcode_"):
                assert row["status"] == "advisory"
                assert row["freshness_status"] == "generated_present_unchecked"
                assert "zero-dead-code guarantee" in row["non_claims"]

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--family",
                "coverage",
                "--no-generated",
                "--require-generated",
                "coverage",
                "--check",
            ],
            expect_success=False,
        )
        assert "required generated family missing: coverage" in result.stdout


def test_freshness_missing_generated_and_deferred_rows() -> None:
    result = run_command(
        [
            "python3",
            str(SCRIPT),
            "--family",
            "oracle",
            "--no-generated",
            "--check-freshness",
        ]
    )
    assert "freshness: warning:" in result.stdout
    assert "not_generated: local generated report is absent" in result.stdout

    result = run_command(
        [
            "python3",
            str(SCRIPT),
            "--family",
            "oracle",
            "--no-generated",
            "--require-generated",
            "oracle",
            "--check-freshness",
        ],
        expect_success=False,
    )
    assert "freshness: error:" in result.stdout
    assert "required generated family missing: oracle" in result.stdout

    result = run_command(
        [
            "python3",
            str(SCRIPT),
            "--family",
            "runtime_backend",
            "--check-freshness",
        ]
    )
    assert "freshness: advisory:" in result.stdout
    assert "source-controlled row is governed by schema and Git review" in result.stdout


def test_freshness_stale_and_advisory_runtime_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        oracle_dir = build_root / "corpus" / "oracle"
        report_dir = build_root / "corpus-reports"
        oracle_output = tmp_path / "oracle-index.tsv"
        runtime_output = tmp_path / "runtime-index.tsv"

        run_command(
            [
                "python3",
                str(ORACLE_SCRIPT),
                "--oracle-dir",
                str(oracle_dir),
                "--report-dir",
                str(report_dir),
            ]
        )
        oracle_path = oracle_dir / "qr_rank_deficient_6x4_nullspace_v1.oracle.tsv"
        oracle_path.write_text(oracle_path.read_text().replace("\t" + current_commit() + "\t", "\toldcommit\t"))

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--check-freshness",
            ]
        )
        assert "freshness: warning:" in result.stdout
        assert "stale: source_commit does not match current HEAD" in result.stdout

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--strict-generated",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "stale: source_commit does not match current HEAD" in result.stdout

        write_runtime_fixture(build_root)
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "benchmark",
                "--check-freshness",
                "--output",
                str(runtime_output),
            ]
        )
        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "benchmark",
                "--check-freshness",
            ]
        )
        assert "freshness: advisory:" in result.stdout
        assert "local measurement freshness is advisory" in result.stdout

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "benchmark",
                "--strict-generated",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "stale: source_commit does not match current HEAD" in result.stdout

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "benchmark",
                "--strict-generated",
                "--advisory-ok",
                "--check-freshness",
            ]
        )
        assert "freshness: advisory:" in result.stdout
        assert "local measurement freshness is advisory" in result.stdout

        sentinel_path = build_root / "bench-reports" / "sentinels" / "sentinels.tsv"
        sentinel_path.write_text(sentinel_path.read_text().replace("\tpass\t", "\tfail\t", 1))
        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "sentinel",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "generated hard-gate or guardrail row reports fail" in result.stdout


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def selected_oracle_fixture_sequence() -> list[tuple[str, str]]:
    sequence: list[tuple[str, str]] = []
    sequence.extend([("unknown", "qr_rank_deficient_6x4_nullspace_v1")] * 3)
    qr_fixtures = [
        "qr_rank_deficient_6x4_nullspace_v1",
        "qr_rankdef_duplicate_5x4_v1",
        "qr_rankdef_dependent_row_4x3_v1",
        "qr_underdetermined_minnorm_2x4",
        "qr_minnorm_3x6_exact_values",
        "qr_minnorm_5x10_exact_values",
    ]
    partial_svd_fixtures = [
        "partial_svd_clustered_repeated_diag8x6_k3_v1",
        "partial_svd_rankdef_diag6x4_k2_range_projector_v1",
        "partial_svd_lowrank_rect5x7_k3_sparse_output_v1",
        "partial_svd_fail_closed_diag6_k2_v1",
    ]
    for index in range(23):
        sequence.append(("qr", qr_fixtures[index % len(qr_fixtures)]))
    for index in range(26):
        sequence.append(("partial_svd", partial_svd_fixtures[index % len(partial_svd_fixtures)]))
    return sequence


def write_selected_oracle_rows(
    build_root: Path,
    *,
    drop_last: bool = False,
    omit_solver_family: str = "",
    stale_first: bool = False,
    fail_first: bool = False,
    remap_fixture_from: str = "",
    remap_fixture_to: str = "",
) -> None:
    oracle_dir = build_root / "corpus" / "oracle"
    report_dir = build_root / "corpus-reports"
    oracle_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    rows = []
    selected_rows = selected_oracle_fixture_sequence()
    if omit_solver_family:
        selected_rows = [
            row for row in selected_rows if row[0] != omit_solver_family
        ]
    if drop_last:
        selected_rows.pop()
    for index, (solver_family, fixture_key) in enumerate(selected_rows):
        if fixture_key == remap_fixture_from:
            fixture_key = remap_fixture_to
        rows.append(
            {
                "oracle_row_id": f"{fixture_key}_synthetic_{index}",
                "fixture_key": fixture_key,
                "solver_family": solver_family,
                "operation": "synthetic",
                "comparison_kind": "value",
                "command": (
                    "python3 scripts/run_corpus_oracle.py "
                    "--include-solver-qr --include-partial-svd"
                ),
                "source_commit": "oldcommit" if stale_first and index == 0 else current_commit(),
                "source_branch": "sprint-152",
                "generated_at_utc": "2026-08-11T00:00:00Z",
                "platform": "test",
                "compiler": "test",
                "configuration": "proof_owner=synthetic_selected_oracle",
                "support_tier": "local_only",
                "expected_result_kind": "scalar",
                "expected_result": "1",
                "observed_result": "1",
                "tolerance_kind": "absolute",
                "tolerance_value": "0",
                "comparison_status": "fail" if fail_first and index == 0 else "pass",
                "failure_class": "synthetic_failure" if fail_first and index == 0 else "",
                "skip_or_defer_reason": "",
                "claim_scope": "selected generated oracle freshness proof",
                "non_claims": "synthetic test rows only; no solver correctness claim",
            }
        )
    with (oracle_dir / "corpus.oracle.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORACLE_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "manifest.txt").write_text(
        "corpus-oracle\n"
        f"git_commit={current_commit()}\n"
        "command=python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd\n"
    )


def write_selected_comparison_rows(
    build_root: Path,
    *,
    drop_last: bool = False,
    stale_first: bool = False,
    defer_first: bool = False,
    fail_first: bool = False,
    skip_first: bool = False,
    duplicate_first: bool = False,
    unexpected_first: bool = False,
) -> None:
    row_ids = list(SELECTED_COMPARISON_ROW_IDS)
    if drop_last:
        row_ids.remove("comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1")
    if unexpected_first:
        row_ids[0] = "comparison_partial_svd_diag6_k2_unexpected_metric_v1"
    if duplicate_first:
        row_ids.append("comparison_partial_svd_diag6_k2_project_status_v1")
    if stale_first or defer_first or fail_first or skip_first:
        selected = "comparison_partial_svd_diag6_k2_project_status_v1"
        row_ids.remove(selected)
        row_ids.insert(0, selected)

    rows_by_subfamily: dict[str, list[dict[str, str]]] = {
        "qr_minnorm": [],
        "qr_compatible_ls": [],
        "partial_svd_diag6_k2": [],
        "lu_nonsym_square_5": [],
    }
    for index, row_id in enumerate(row_ids):
        if "partial_svd_diag6_k2" in row_id:
            subfamily = "partial_svd_diag6_k2"
            fixture_key = "partial_svd_diag6_k2"
            operation = "partial_svd"
            artifact_path = "build/comparison/partial_svd_diag6_k2/study.tsv"
            non_claims = (
                "synthetic test rows only; no broad partial-SVD correctness; "
                "no raw singular-vector identity claim"
            )
        elif "lu_nonsym_square_5" in row_id:
            subfamily = "lu_nonsym_square_5"
            fixture_key = "lu_nonsym_square_5"
            operation = "square_solve"
            artifact_path = "build/comparison/lu_nonsym_square_5/study.tsv"
            non_claims = (
                "synthetic test rows only; no broad LU correctness; "
                "no broad nonsymmetric solve parity"
            )
        elif "qr_overdetermined_compatible_5x3" in row_id:
            subfamily = "qr_compatible_ls"
            fixture_key = "qr_overdetermined_compatible_5x3"
            operation = "least_squares_solve"
            artifact_path = "build/comparison/qr_compatible_ls/study.tsv"
            non_claims = "synthetic test rows only; no broad QR parity claim"
        else:
            subfamily = "qr_minnorm"
            fixture_key = "qr_underdetermined_minnorm_2x4"
            operation = "minnorm_solve"
            artifact_path = "build/comparison/qr_minnorm/study.tsv"
            non_claims = "synthetic test rows only; no broad QR parity claim"
        status = "pass"
        status_reason = "synthetic_pass"
        if index == 0 and defer_first:
            status = "defer"
            status_reason = "synthetic_dependency_deferred"
        if index == 0 and fail_first:
            status = "fail"
            status_reason = "synthetic_comparison_failure"
        if index == 0 and skip_first:
            status = "skip"
            status_reason = "synthetic_comparison_skipped"
        rows_by_subfamily[subfamily].append(
            {
                "comparison_row_id": row_id,
                "report_family": "comparison",
                "subfamily": subfamily,
                "row_kind": "metric_comparison",
                "fixture_key": fixture_key,
                "operation": operation,
                "metric": f"synthetic_metric_{index}",
                "baseline_name": "dense_reference",
                "baseline_type": "source_controlled",
                "baseline_version": "synthetic",
                "baseline_command": "synthetic baseline",
                "baseline_python_executable": "python3",
                "baseline_python_version": "synthetic",
                "project_name": "sparse_lu_ortho",
                "project_version": "synthetic",
                "project_command": "synthetic project",
                "source_commit": "oldcommit" if stale_first and index == 0 else current_commit(),
                "source_branch": "sprint-159",
                "worktree_state": "clean",
                "platform": "test",
                "compiler": "test",
                "configuration": "proof_owner=synthetic_selected_comparison",
                "expected_value": "1",
                "project_value": "1",
                "baseline_value": "1",
                "delta_value": "0",
                "tolerance_kind": "absolute",
                "tolerance_value": "0",
                "status": status,
                "status_reason": status_reason,
                "caveat": "",
                "artifact_path": artifact_path,
                "generated_at_utc": "2026-08-16T00:00:00Z",
                "support_tier": "local_only",
                "claim_scope": "selected generated comparison freshness proof",
                "non_claims": non_claims,
            }
        )
    for subfamily, rows in rows_by_subfamily.items():
        if not rows:
            continue
        comparison_dir = build_root / "comparison" / subfamily
        comparison_dir.mkdir(parents=True, exist_ok=True)
        with (comparison_dir / "study.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPARISON_STUDY_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


def test_generated_oracle_rows_are_preserved() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        oracle_dir = build_root / "corpus" / "oracle"
        report_dir = build_root / "corpus-reports"
        output = tmp_path / "oracle-index.tsv"

        run_command(
            [
                "python3",
                str(ORACLE_SCRIPT),
                "--include-partial-svd",
                "--oracle-dir",
                str(oracle_dir),
                "--report-dir",
                str(report_dir),
            ]
        )
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--output",
                str(output),
            ]
        )
        rows = read_tsv(output)
        native_ids = {row["native_row_id"] for row in rows}
        oracle_rows = generated_oracle_rows(rows)
        assert "qr_rank_deficient_6x4_nullspace_v1_rank" in native_ids
        assert "partial_svd_clustered_repeated_diag8x6_k3_v1_singular_values" in native_ids
        for fixture_key, expected_count in SPRINT151_PARTIAL_SVD_ROW_COUNTS.items():
            fixture_rows = [
                row
                for row in oracle_rows
                if row["native_row_id"].startswith(fixture_key)
            ]
            assert len(fixture_rows) == expected_count
            for row in fixture_rows:
                assert row["report_family"] == "oracle"
                assert row["subfamily"] == "solver_backed"
                assert row["row_origin"] == "generated_local"
                assert row["support_tier"] == "local_only"
                assert row["status"] == "pass"
                assert "solver_family=partial_svd" in row["configuration"]
                assert f"fixture_key={fixture_key}" in row["configuration"]
                assert "proof_owner=generated_partial_svd_reference" in row["configuration"]
                assert "solver_execution=none" in row["configuration"]
                assert (
                    "broad partial-SVD correctness" in row["non_claims"]
                    or "broad sparse-output correctness" in row["non_claims"]
                )
                assert "external-library parity" in row["non_claims"]
                assert "performance" in row["non_claims"]
        assert not any(row["freshness_status"] == "not_generated" for row in rows)
        for row in oracle_rows:
            assert row["status"] == "pass"
            assert "fixture_key=" in row["configuration"]
            assert "broad" in row["non_claims"]
            assert row["freshness_status"] == "generated_present_unchecked"


def test_sprint151_partial_svd_oracle_freshness_strictness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        oracle_dir = build_root / "corpus" / "oracle"
        report_dir = build_root / "corpus-reports"
        output = tmp_path / "oracle-index.tsv"

        run_command(
            [
                "python3",
                str(ORACLE_SCRIPT),
                "--include-partial-svd",
                "--oracle-dir",
                str(oracle_dir),
                "--report-dir",
                str(report_dir),
            ]
        )
        oracle_path = oracle_dir / "corpus.oracle.tsv"
        oracle_text = oracle_path.read_text()
        fixture_key = "partial_svd_rankdef_diag6x4_k2_range_projector_v1"
        if fixture_key not in oracle_text:
            raise AssertionError(f"missing generated fixture rows for {fixture_key}")
        oracle_path.write_text(oracle_text.replace("\t" + current_commit() + "\t", "\toldcommit\t"))

        run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--output",
                str(output),
            ]
        )
        rows = read_tsv(output)
        stale_fixture_rows = [
            row
            for row in generated_oracle_rows(rows)
            if row["native_row_id"].startswith(fixture_key)
        ]
        assert len(stale_fixture_rows) == SPRINT151_PARTIAL_SVD_ROW_COUNTS[fixture_key]
        assert {row["source_commit"] for row in stale_fixture_rows} == {"oldcommit"}

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--check-freshness",
            ]
        )
        assert "freshness: warning:" in result.stdout
        assert "stale: source_commit does not match current HEAD" in result.stdout
        assert fixture_key in result.stdout

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--strict-generated",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "stale: source_commit does not match current HEAD" in result.stdout
        assert fixture_key in result.stdout


def test_selected_oracle_required_freshness_requires_complete_family_set() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        write_selected_oracle_rows(build_root)

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--require-generated",
                "oracle",
                "--check-freshness",
            ]
        )
        assert "oracle_selected_row_count" not in result.stdout
        assert "oracle_selected_fixture_keys" not in result.stdout
        assert "freshness: warning:" not in result.stdout
        assert "generated row exists but strict freshness comparison is pending" not in result.stdout

        output = tmp_path / "oracle-index.tsv"
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--output",
                str(output),
            ]
        )
        rows = read_tsv(output)
        assert len(generated_oracle_rows(rows)) == 52
        observed_fixture_keys = {
            part.removeprefix("fixture_key=")
            for row in generated_oracle_rows(rows)
            for part in row["configuration"].split(";")
            if part.startswith("fixture_key=")
        }
        assert SELECTED_ORACLE_FIXTURE_KEYS <= observed_fixture_keys


def test_selected_oracle_required_freshness_rejects_partial_family_set() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        write_selected_oracle_rows(build_root, drop_last=True)

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--require-generated",
                "oracle",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "oracle_selected_row_count" in result.stdout
        assert "row_count_mismatch" in result.stdout
        assert "make report-index-oracle-freshness" in result.stdout
        assert "python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd" in result.stdout
        assert f"resolved_artifact={build_root / 'corpus' / 'oracle'}/*.tsv" in result.stdout


def test_selected_oracle_required_freshness_reports_missing_artifacts() -> None:
    result = run_command(
        [
            "python3",
            str(SCRIPT),
            "--family",
            "oracle",
            "--no-generated",
            "--require-generated",
            "oracle",
            "--check-freshness",
        ],
        expect_success=False,
    )
    assert "freshness: error:" in result.stdout
    assert "required generated family missing: oracle" in result.stdout
    assert "artifact=build/corpus/oracle/*.tsv" in result.stdout
    assert "make report-index-oracle-freshness" in result.stdout
    assert "python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd" in result.stdout


def test_selected_oracle_required_freshness_rejects_stale_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        write_selected_oracle_rows(build_root, stale_first=True)

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--require-generated",
                "oracle",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "stale: source_commit does not match current HEAD" in result.stdout
        assert "recorded=oldcommit" in result.stdout
        assert "current=" in result.stdout
        assert "artifact=" in result.stdout
        assert "make report-index-oracle-freshness" in result.stdout


def test_selected_oracle_required_freshness_rejects_failed_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        write_selected_oracle_rows(build_root, fail_first=True)

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--require-generated",
                "oracle",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "generated oracle row reports fail" in result.stdout
        assert "fixture_key=qr_rank_deficient_6x4_nullspace_v1" in result.stdout
        assert "artifact=" in result.stdout
        assert "make report-index-oracle-freshness" in result.stdout


def test_selected_oracle_required_freshness_rejects_missing_solver_family() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        write_selected_oracle_rows(build_root, omit_solver_family="qr")

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--require-generated",
                "oracle",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "oracle_selected_solver_families" in result.stdout
        assert "missing=qr" in result.stdout
        assert "observed=partial_svd,unknown" in result.stdout
        assert f"resolved_artifact={build_root / 'corpus' / 'oracle'}/*.tsv" in result.stdout


def test_selected_oracle_required_freshness_rejects_missing_fixture_key() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_root = tmp_path / "build"
        missing_fixture = "partial_svd_fail_closed_diag6_k2_v1"
        replacement_fixture = "partial_svd_clustered_repeated_diag8x6_k3_v1"
        write_selected_oracle_rows(
            build_root,
            remap_fixture_from=missing_fixture,
            remap_fixture_to=replacement_fixture,
        )

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "oracle",
                "--require-generated",
                "oracle",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "oracle_selected_row_count" not in result.stdout
        assert "oracle_selected_fixture_keys" in result.stdout
        assert f"missing={missing_fixture}" in result.stdout
        assert "manifest=build/corpus-reports/manifest.txt" in result.stdout
        assert f"resolved_manifest={build_root / 'corpus-reports' / 'manifest.txt'}" in result.stdout


def test_selected_oracle_gate_preserves_advisory_and_source_controlled_families() -> None:
    coverage_result = run_command(
        [
            "python3",
            str(SCRIPT),
            "--family",
            "coverage",
            "--check-freshness",
        ]
    )
    assert "freshness: advisory:" in coverage_result.stdout
    assert "local generated advisory report is absent" in coverage_result.stdout

    coverage_required = run_command(
        [
            "python3",
            str(SCRIPT),
            "--family",
            "coverage",
            "--require-generated",
            "coverage",
            "--check-freshness",
        ],
        expect_success=False,
    )
    assert "required generated family missing: coverage" in coverage_required.stdout

    package_result = run_command(
        [
            "python3",
            str(SCRIPT),
            "--family",
            "package",
            "--check-freshness",
        ]
    )
    assert "freshness: advisory:" in package_result.stdout
    assert "source-controlled row is governed by schema and Git review" in package_result.stdout


def test_selected_comparison_required_freshness_accepts_complete_row_set() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        build_root = Path(tmp) / "build"
        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "required generated family missing: comparison" in result.stdout
        assert SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC in result.stdout

        write_selected_comparison_rows(build_root)
        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ]
        )
        assert "comparison_selected_rows" not in result.stdout
        assert "comparison_selected_status" not in result.stdout
        assert "freshness: warning:" not in result.stdout
        assert "generated row exists but strict freshness comparison is pending" not in result.stdout

        output = Path(tmp) / "comparison-index.tsv"
        run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--output",
                str(output),
            ]
        )
        rows = read_tsv(output)
        partial_svd_rows = [
            row
            for row in rows
            if row["subfamily"] == "partial_svd_diag6_k2"
            and row["row_origin"] == "generated_local"
            and row["row_id"].startswith("comparison_")
        ]
        assert {row["row_id"] for row in partial_svd_rows} == SELECTED_PARTIAL_SVD_COMPARISON_ROW_IDS
        assert {row["status"] for row in partial_svd_rows} == {"pass"}
        assert {row["support_tier"] for row in partial_svd_rows} == {"local_only"}
        assert all(
            row["artifact_path"].endswith("comparison/partial_svd_diag6_k2/study.tsv")
            for row in partial_svd_rows
        )
        assert all("raw singular-vector identity" in row["non_claims"] for row in partial_svd_rows)

        lu_rows = [
            row
            for row in rows
            if row["subfamily"] == "lu_nonsym_square_5"
            and row["row_origin"] == "generated_local"
            and row["row_id"].startswith("comparison_")
        ]
        assert {row["row_id"] for row in lu_rows} == SELECTED_LU_COMPARISON_ROW_IDS
        assert {row["status"] for row in lu_rows} == {"pass"}
        assert {row["support_tier"] for row in lu_rows} == {"local_only"}
        assert all(
            row["artifact_path"].endswith("comparison/lu_nonsym_square_5/study.tsv")
            for row in lu_rows
        )
        assert all("no broad LU correctness" in row["non_claims"] for row in lu_rows)


def test_selected_comparison_manifest_support_tiers_remain_bounded() -> None:
    rows = read_tsv(REPORT_FAMILIES)
    comparison_rows = {
        row["subfamily"]: row for row in rows if row["report_family"] == "comparison"
    }
    expected_subfamilies = {
        "qr_minnorm",
        "qr_compatible_ls",
        "partial_svd_diag6_k2",
        "lu_nonsym_square_5",
    }
    assert set(comparison_rows) == expected_subfamilies
    for subfamily, row in comparison_rows.items():
        assert row["row_origin"] == "generated_local", subfamily
        assert row["support_tier"] == "local_only", subfamily
        assert row["freshness_policy"] == "generated_compare_inputs", subfamily
        assert row["artifact_pattern"] == f"build/comparison/{subfamily}/study.tsv"
        assert "no hosted CI proof from generated-local row metadata" in row["non_claims"]
        assert "no broad platform portability proof" in row["non_claims"]
        assert "no Windows report freshness" in row["non_claims"]
        assert "no package-manager proof" in row["non_claims"]
        assert "no shared-library ABI proof" in row["non_claims"]
        assert "no performance superiority" in row["non_claims"]
        assert "no state-of-the-art claim" in row["non_claims"]

    ci_row = next(
        row
        for row in rows
        if row["report_family"] == "ci" and row["subfamily"] == "reviewed_lanes"
    )
    assert ci_row["support_tier"] == "reviewed_cross_platform"
    assert ci_row["freshness_policy"] == "hosted_ci_external"
    assert "Linux selected oracle/comparison freshness" in ci_row["claim_scope"]
    assert "macOS selected comparison freshness" in ci_row["claim_scope"]
    assert "no local report freshness proof from CI metadata alone" in ci_row["non_claims"]
    assert "no Windows report freshness" in ci_row["non_claims"]
    assert "no benchmark release claim" in ci_row["non_claims"]


def test_selected_comparison_required_freshness_rejects_row_set_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        build_root = Path(tmp) / "build"
        write_selected_comparison_rows(build_root, drop_last=True)

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "comparison_selected_rows" in result.stdout
        assert "row_set_mismatch" in result.stdout
        assert SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC in result.stdout
        assert (
            "missing=comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1"
            in result.stdout
        )

        write_selected_comparison_rows(build_root, unexpected_first=True)
        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "unexpected=comparison_partial_svd_diag6_k2_unexpected_metric_v1" in result.stdout
        assert SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC in result.stdout


def test_selected_comparison_required_freshness_rejects_duplicate_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        build_root = Path(tmp) / "build"
        write_selected_comparison_rows(build_root, duplicate_first=True)

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "duplicate normalized row_id" in result.stderr


def test_selected_comparison_required_freshness_rejects_stale_and_invalid_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        build_root = Path(tmp) / "build"
        write_selected_comparison_rows(build_root, stale_first=True)

        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "stale: source_commit does not match current HEAD" in result.stdout
        assert "run make report-index-comparison-freshness" in result.stdout

        write_selected_comparison_rows(build_root, fail_first=True)
        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "generated comparison row reports fail" in result.stdout
        assert "comparison_selected_status" in result.stdout
        assert SELECTED_COMPARISON_ARTIFACT_DIAGNOSTIC in result.stdout

        write_selected_comparison_rows(build_root, defer_first=True)
        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "comparison_selected_status" in result.stdout
        assert "comparison_selected_rows: skip_or_defer_not_proof" in result.stdout
        assert "comparison_optional_rows" not in result.stdout

        write_selected_comparison_rows(build_root, skip_first=True)
        result = run_command(
            [
                "python3",
                str(SCRIPT),
                "--build-root",
                str(build_root),
                "--family",
                "comparison",
                "--require-generated",
                "comparison",
                "--check-freshness",
            ],
            expect_success=False,
        )
        assert "freshness: error:" in result.stdout
        assert "comparison_selected_status" in result.stdout
        assert "comparison_selected_rows: skip_or_defer_not_proof" in result.stdout
        assert "comparison_optional_rows" not in result.stdout


def main() -> int:
    test_current_repo_no_generated()
    test_git_metadata_is_independent_of_caller_cwd()
    test_family_filter_and_required_missing()
    test_generated_artifact_presence()
    test_runtime_report_rows_preserve_boundaries()
    test_quality_and_package_rows_preserve_scope()
    test_freshness_missing_generated_and_deferred_rows()
    test_freshness_stale_and_advisory_runtime_rows()
    test_generated_oracle_rows_are_preserved()
    test_sprint151_partial_svd_oracle_freshness_strictness()
    test_selected_oracle_required_freshness_requires_complete_family_set()
    test_selected_oracle_required_freshness_rejects_partial_family_set()
    test_selected_oracle_required_freshness_reports_missing_artifacts()
    test_selected_oracle_required_freshness_rejects_stale_rows()
    test_selected_oracle_required_freshness_rejects_failed_rows()
    test_selected_oracle_required_freshness_rejects_missing_solver_family()
    test_selected_oracle_required_freshness_rejects_missing_fixture_key()
    test_selected_oracle_gate_preserves_advisory_and_source_controlled_families()
    test_selected_comparison_required_freshness_accepts_complete_row_set()
    test_selected_comparison_manifest_support_tiers_remain_bounded()
    test_selected_comparison_required_freshness_rejects_row_set_mismatch()
    test_selected_comparison_required_freshness_rejects_duplicate_rows()
    test_selected_comparison_required_freshness_rejects_stale_and_invalid_rows()
    print("test-normalize-report-index: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
