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
SPRINT151_PARTIAL_SVD_ROW_COUNTS = {
    "partial_svd_rankdef_diag6x4_k2_range_projector_v1": 7,
    "partial_svd_lowrank_rect5x7_k3_sparse_output_v1": 6,
    "partial_svd_fail_closed_diag6_k2_v1": 5,
}


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
        assert any(row["native_row_id"] == "bench_refactor_csc" for row in by_family["benchmark"])
        assert any(
            row["row_meaning"] == "sentinel_hard_gate" and row["status"] == "pass"
            for row in by_family["sentinel"]
        )
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
    print("test-normalize-report-index: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
