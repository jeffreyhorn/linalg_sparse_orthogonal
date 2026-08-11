#!/usr/bin/env python3
"""Validate maintained corpus TSV skeletons.

This intentionally checks schema shape and row semantics only. It does not run
solver or oracle comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "tests" / "corpus"
FIXTURE_REQUIRED = {
    "fixture_key",
    "fixture_family",
    "storage_kind",
    "generator_key",
    "rows",
    "cols",
    "nnz",
    "symmetry",
    "definiteness",
    "rank_status",
    "conditioning_class",
    "scale_class",
    "sparsity_class",
    "rhs_policy",
    "expected_behavior",
    "claim_scope",
    "non_claims",
    "support_tier",
    "validation_command",
    "owner",
    "introduced_in",
}
FIXTURE_CONDITIONAL = {
    "matrix_path",
    "expected_rank",
    "nullity",
}
GENERATOR_REQUIRED = {
    "generator_key",
    "generator_version",
    "algorithm",
    "seed",
    "parameters",
    "expected_structure_hash",
    "expected_value_hash",
    "canonical_format",
    "floating_policy",
    "regeneration_command",
    "change_policy",
}
OPTIONAL_DATA_REQUIRED = {
    "optional_data_key",
    "source_name",
    "source_url_or_reference",
    "license_or_terms",
    "expected_location",
    "availability_state",
    "skip_reason",
    "defer_reason",
    "fixture_keys",
    "validation_command",
    "pass_interpretation",
    "skip_interpretation",
    "claim_boundary",
}
REPORT_FAMILY_REQUIRED = {
    "report_family",
    "subfamily",
    "row_meaning",
    "row_origin",
    "status",
    "support_tier",
    "freshness_policy",
    "generator_command",
    "artifact_pattern",
    "claim_scope",
    "non_claims",
    "owner",
    "introduced_in",
}
EXPECTED_REQUIRED = {
    "oracle_row_id",
    "fixture_key",
    "operation",
    "comparison_kind",
    "expected_result_kind",
    "expected_result",
    "tolerance_kind",
    "tolerance_value",
    "claim_scope",
    "non_claims",
    "status",
}

STORAGE_KINDS = {"inline", "generated", "matrix_market", "optional_external"}
EXPECTED_BEHAVIORS = {"success", "diagnostic_failure", "unsupported", "non_convergence", "skip"}
SUPPORT_TIERS = {
    "reviewed_linux",
    "reviewed_cross_platform",
    "supplemental_macos",
    "supplemental_windows",
    "local_only",
    "optional_data",
    "staged",
}
COMPARISON_KINDS = {
    "value",
    "residual_norm",
    "rank",
    "nullity",
    "subspace_distance",
    "status",
    "diagnostic",
    "local_measurement",
}
EXPECTED_RESULT_KINDS = {
    "value",
    "residual_norm",
    "rank",
    "nullity",
    "subspace_distance",
    "status",
    "diagnostic",
    "performance_local",
}
TOLERANCE_KINDS = {
    "exact",
    "absolute",
    "relative",
    "mixed",
    "projector",
    "status_only",
    "not_applicable",
}
EXPECTED_STATUSES = {
    "placeholder_pending_generator",
    "placeholder_pending_oracle_command",
    "placeholder_pending_oracle_schema",
    "ready_for_oracle",
}
OPTIONAL_AVAILABILITY = {"available", "unavailable", "disabled", "deferred"}
REPORT_ROW_ORIGINS = {
    "source_controlled",
    "generated_local",
    "generated_ci",
    "external_optional",
    "documentation",
}
REPORT_ROW_MEANINGS = {
    "fixture_metadata",
    "generator_metadata",
    "optional_data_policy",
    "expected_result",
    "observed_oracle_comparison",
    "solver_backed_fixture_proof",
    "benchmark_measurement",
    "sentinel_hard_gate",
    "sentinel_advisory_measurement",
    "guardrail_lane",
    "deadcode_classification",
    "coverage_summary",
    "package_install_proof_owner",
    "ci_lane_definition",
    "documentation_advisory",
    "runtime_backend_governance_policy",
    "not_generated",
    "deferred_governance",
}
REPORT_STATUSES = {
    "pass",
    "fail",
    "skip",
    "defer",
    "unsupported",
    "xfail",
    "unknown",
    "advisory",
}
REPORT_FRESHNESS_POLICIES = {
    "source_controlled",
    "generated_compare_inputs",
    "generated_local_advisory",
    "hosted_ci_external",
    "optional_data_skip",
    "deferred_governance",
}
CANONICAL_FORMAT = "coo_zero_based_row_col_value_f64_text_v1"
STRUCTURE_FORMAT = "coo_zero_based_row_col_text_v1"


def qr_rank_deficient_6x4_nullspace_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 1.0),
        (0, 3, 1.0),
        (1, 1, 1.0),
        (1, 3, 1.0),
        (2, 2, 1.0),
        (3, 0, 1.0),
        (3, 1, 1.0),
        (3, 3, 2.0),
        (4, 1, 1.0),
        (4, 2, 1.0),
        (4, 3, 1.0),
        (5, 0, 1.0),
        (5, 2, 1.0),
        (5, 3, 1.0),
    ]


def partial_svd_clustered_repeated_diag8x6_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 10.0),
        (1, 1, 10.0),
        (2, 2, 9.999999),
        (3, 3, 4.0),
        (4, 4, 1.0),
    ]


def partial_svd_rankdef_diag6x4_k2_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 9.0),
        (1, 1, 6.0),
    ]


def partial_svd_lowrank_rect5x7_k3_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 8.0),
        (1, 1, 4.0),
        (2, 2, 2.0),
        (3, 3, 1.0),
    ]


def partial_svd_fail_closed_diag6_k2_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 9.0),
        (1, 1, 6.0),
        (2, 2, 3.0),
        (3, 3, 1.0),
        (4, 4, 0.5),
        (5, 5, 0.25),
    ]


def qr_rankdef_duplicate_5x4_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 1.0),
        (0, 2, 2.0),
        (1, 1, 1.0),
        (1, 2, -1.0),
        (1, 3, 1.0),
        (2, 0, 2.0),
        (2, 1, -1.0),
        (2, 3, -1.0),
        (3, 0, 1.0),
        (3, 1, 1.0),
        (3, 2, 1.0),
        (3, 3, 1.0),
        (4, 0, 3.0),
        (4, 2, -2.0),
    ]


def qr_rankdef_dependent_row_4x3_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 1.0),
        (0, 2, 1.0),
        (1, 1, 1.0),
        (1, 2, 2.0),
        (2, 0, 1.0),
        (2, 1, 1.0),
        (2, 2, 3.0),
        (3, 0, 2.0),
        (3, 1, -1.0),
    ]


def qr_underdetermined_minnorm_2x4_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 1.0),
        (0, 2, 1.0),
        (1, 1, 1.0),
        (1, 3, 1.0),
    ]


def qr_minnorm_3x6_exact_values_entries() -> list[tuple[int, int, float]]:
    return [
        (0, 0, 2.0),
        (0, 3, 1.0),
        (1, 1, 3.0),
        (1, 4, 1.0),
        (2, 2, 1.0),
        (2, 5, 2.0),
    ]


def qr_minnorm_5x10_exact_values_entries() -> list[tuple[int, int, float]]:
    entries: list[tuple[int, int, float]] = []
    for index in range(5):
        entries.append((index, index, 2.0))
        entries.append((index, index + 5, 1.0))
    return entries


GENERATED_FIXTURES = {
    "qr_rank_deficient_6x4_nullspace_generator_v1": {
        "algorithm": "fixed_columns_c3_equals_c0_plus_c1",
        "rows": 6,
        "cols": 4,
        "expected_rank": 3,
        "nullity": 1,
        "parameters": "rows=6;cols=4;expected_rank=3;nullity=1;dependency=c3-c0-c1",
        "entries": qr_rank_deficient_6x4_nullspace_entries,
    },
    "qr_rankdef_duplicate_5x4_generator_v1": {
        "algorithm": "fixed_rankdef_duplicate_5x4",
        "rows": 5,
        "cols": 4,
        "expected_rank": 3,
        "nullity": 1,
        "parameters": "rows=5;cols=4;expected_rank=3;nullity=1;duplicate_column=c3-c1",
        "entries": qr_rankdef_duplicate_5x4_entries,
    },
    "qr_rankdef_dependent_row_4x3_generator_v1": {
        "algorithm": "fixed_rankdef_dependent_row_4x3",
        "rows": 4,
        "cols": 3,
        "expected_rank": 2,
        "nullity": 1,
        "parameters": "rows=4;cols=3;expected_rank=2;nullity=1;dependent_row=r2-r0-r1",
        "entries": qr_rankdef_dependent_row_4x3_entries,
    },
    "qr_underdetermined_minnorm_2x4_generator_v1": {
        "algorithm": "fixed_underdetermined_minnorm_2x4",
        "rows": 2,
        "cols": 4,
        "expected_rank": 2,
        "nullity": 2,
        "parameters": "rows=2;cols=4;expected_rank=2;nullity=2;rhs=1,1;expected_norm=1.0",
        "entries": qr_underdetermined_minnorm_2x4_entries,
    },
    "qr_minnorm_3x6_exact_values_generator_v1": {
        "algorithm": "fixed_minnorm_3x6_exact_values",
        "rows": 3,
        "cols": 6,
        "expected_rank": 3,
        "nullity": 3,
        "parameters": "rows=3;cols=6;expected_rank=3;nullity=3;expected_norm=sqrt(8.4)",
        "entries": qr_minnorm_3x6_exact_values_entries,
    },
    "qr_minnorm_5x10_exact_values_generator_v1": {
        "algorithm": "fixed_minnorm_5x10_exact_values",
        "rows": 5,
        "cols": 10,
        "expected_rank": 5,
        "nullity": 5,
        "parameters": "rows=5;cols=10;expected_rank=5;nullity=5;expected_norm=sqrt(11.0)",
        "entries": qr_minnorm_5x10_exact_values_entries,
    },
    "partial_svd_clustered_repeated_diag8x6_generator_v1": {
        "algorithm": "fixed_diagonal_clustered_repeated_partial_svd",
        "rows": 8,
        "cols": 6,
        "expected_rank": 5,
        "nullity": 1,
        "parameters": "rows=8;cols=6;k=3;diag=10,10,9.999999,4,1,0;expected_rank=5;nullity=1",
        "entries": partial_svd_clustered_repeated_diag8x6_entries,
    },
    "partial_svd_rankdef_diag6x4_k2_range_projector_generator_v1": {
        "algorithm": "fixed_partial_svd_rankdef_diag6x4_k2_range_projector",
        "rows": 6,
        "cols": 4,
        "expected_rank": 2,
        "nullity": 2,
        "parameters": "rows=6;cols=4;k=2;diag=9,6,0,0;expected_rank=2;nullity=2",
        "entries": partial_svd_rankdef_diag6x4_k2_entries,
    },
    "partial_svd_lowrank_rect5x7_k3_sparse_output_generator_v1": {
        "algorithm": "fixed_partial_svd_lowrank_rect5x7_k3_sparse_output",
        "rows": 5,
        "cols": 7,
        "expected_rank": 4,
        "nullity": 3,
        "parameters": "rows=5;cols=7;k=3;diag=8,4,2,1,0;drop_tol=0;expected_rank=4;nullity=3",
        "entries": partial_svd_lowrank_rect5x7_k3_entries,
    },
    "partial_svd_fail_closed_diag6_k2_generator_v1": {
        "algorithm": "fixed_partial_svd_fail_closed_diag6_k2",
        "rows": 6,
        "cols": 6,
        "expected_rank": 6,
        "nullity": 0,
        "parameters": "rows=6;cols=6;k=2;diag=9,6,3,1,0.5,0.25;tight_max_iter=1;expected_rank=6;nullity=0",
        "entries": partial_svd_fail_closed_diag6_k2_entries,
    },
}


class CorpusValidationError(RuntimeError):
    pass


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        raise CorpusValidationError(f"{path}: empty TSV")
    width = len(rows[0])
    for index, row in enumerate(rows, start=1):
        if len(row) != width:
            raise CorpusValidationError(
                f"{path}:{index}: expected {width} columns, got {len(row)}"
            )
    header = rows[0]
    if len(set(header)) != len(header):
        raise CorpusValidationError(f"{path}: duplicate header fields")
    return [dict(zip(header, row)) for row in rows[1:]]


def require_fields(
    path: Path,
    rows: Iterable[dict[str, str]],
    required: set[str],
    allow_empty: Optional[set[str]] = None,
) -> None:
    allow_empty = allow_empty or set()
    header = set(read_header(path))
    missing = sorted(required - header)
    if missing:
        raise CorpusValidationError(f"{path}: missing required headers: {', '.join(missing)}")
    for line, row in enumerate(rows, start=2):
        for field in required:
            if field in allow_empty:
                continue
            if row.get(field, "") == "":
                raise CorpusValidationError(f"{path}:{line}: required field {field} is empty")


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        return next(reader)


def assert_enum(path: Path, line: int, field: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise CorpusValidationError(
            f"{path}:{line}: invalid {field}={value!r}; expected one of {sorted(allowed)}"
        )


def is_lower_snake(value: str) -> bool:
    if not value:
        return False
    return all(ch.islower() or ch.isdigit() or ch == "_" for ch in value) and "__" not in value


def canonical_structure_text(rows: int, cols: int, entries: list[tuple[int, int, float]]) -> str:
    canonical_entries = sorted(entries, key=lambda entry: (entry[0], entry[1]))
    lines = [
        f"format {STRUCTURE_FORMAT}",
        f"rows {rows}",
        f"cols {cols}",
        f"nnz {len(canonical_entries)}",
    ]
    lines.extend(f"{row} {col}" for row, col, _value in canonical_entries)
    return "\n".join(lines) + "\n"


def canonical_value_text(rows: int, cols: int, entries: list[tuple[int, int, float]]) -> str:
    canonical_entries = sorted(entries, key=lambda entry: (entry[0], entry[1]))
    lines = [
        f"format {CANONICAL_FORMAT}",
        f"rows {rows}",
        f"cols {cols}",
        f"nnz {len(canonical_entries)}",
    ]
    lines.extend(f"{row} {col} {value:.16f}" for row, col, value in canonical_entries)
    return "\n".join(lines) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_known_generator(path: Path, line: int, row: dict[str, str]) -> None:
    generator = GENERATED_FIXTURES.get(row["generator_key"])
    if not generator:
        return
    entries = generator["entries"]()
    expected_structure_hash = sha256_text(
        canonical_structure_text(generator["rows"], generator["cols"], entries)
    )
    expected_value_hash = sha256_text(
        canonical_value_text(generator["rows"], generator["cols"], entries)
    )
    checks = {
        "algorithm": generator["algorithm"],
        "parameters": generator["parameters"],
        "expected_structure_hash": expected_structure_hash,
        "expected_value_hash": expected_value_hash,
        "canonical_format": CANONICAL_FORMAT,
    }
    for field, expected in checks.items():
        if row[field] != expected:
            raise CorpusValidationError(
                f"{path}:{line}: {field} {row[field]!r} does not match generated value "
                f"{expected!r}"
            )


def validate(root: Path) -> None:
    manifests = root / "manifests"
    expected = root / "expected"
    fixtures_path = manifests / "fixtures.tsv"
    generators_path = manifests / "generators.tsv"
    optional_path = manifests / "optional_data.tsv"
    report_families_path = manifests / "report_families.tsv"

    fixture_rows = read_tsv(fixtures_path)
    generator_rows = read_tsv(generators_path)
    optional_rows = read_tsv(optional_path)
    report_family_rows = read_tsv(report_families_path)
    require_fields(
        fixtures_path,
        fixture_rows,
        FIXTURE_REQUIRED | FIXTURE_CONDITIONAL,
        allow_empty={"generator_key", "matrix_path", "expected_rank", "nullity"},
    )
    require_fields(generators_path, generator_rows, GENERATOR_REQUIRED)
    require_fields(
        optional_path,
        optional_rows,
        OPTIONAL_DATA_REQUIRED,
        allow_empty={"skip_reason", "defer_reason"},
    )
    require_fields(report_families_path, report_family_rows, REPORT_FAMILY_REQUIRED)

    fixture_keys = {row["fixture_key"] for row in fixture_rows}
    generator_keys = {row["generator_key"] for row in generator_rows}
    if len(fixture_keys) != len(fixture_rows):
        raise CorpusValidationError(f"{fixtures_path}: duplicate fixture_key")
    if len(generator_keys) != len(generator_rows):
        raise CorpusValidationError(f"{generators_path}: duplicate generator_key")
    report_family_keys = {
        (row["report_family"], row["subfamily"], row["row_meaning"])
        for row in report_family_rows
    }
    if len(report_family_keys) != len(report_family_rows):
        raise CorpusValidationError(
            f"{report_families_path}: duplicate report_family/subfamily/row_meaning"
        )

    generators_by_key = {row["generator_key"]: row for row in generator_rows}
    generator_lines_by_key = {
        row["generator_key"]: line for line, row in enumerate(generator_rows, start=2)
    }

    for line, row in enumerate(fixture_rows, start=2):
        assert_enum(fixtures_path, line, "storage_kind", row["storage_kind"], STORAGE_KINDS)
        assert_enum(
            fixtures_path,
            line,
            "expected_behavior",
            row["expected_behavior"],
            EXPECTED_BEHAVIORS,
        )
        assert_enum(fixtures_path, line, "support_tier", row["support_tier"], SUPPORT_TIERS)
        if row["storage_kind"] == "generated":
            if row["generator_key"] == "":
                raise CorpusValidationError(
                    f"{fixtures_path}:{line}: generated rows require generator_key"
                )
            if row["matrix_path"] != "":
                raise CorpusValidationError(
                    f"{fixtures_path}:{line}: generated rows must leave matrix_path empty"
                )
            if row["generator_key"] not in generator_keys:
                raise CorpusValidationError(
                    f"{fixtures_path}:{line}: generator_key {row['generator_key']!r} not found"
                )
        else:
            if row["generator_key"] != "":
                raise CorpusValidationError(
                    f"{fixtures_path}:{line}: non-generated rows must leave generator_key empty"
                )
            if row["matrix_path"] == "":
                raise CorpusValidationError(
                    f"{fixtures_path}:{line}: stored matrix rows require matrix_path"
                )
        if row["generator_key"] in GENERATED_FIXTURES:
            generated = GENERATED_FIXTURES[row["generator_key"]]
            entries = generated["entries"]()
            if row["rows"] != str(generated["rows"]):
                raise CorpusValidationError(f"{fixtures_path}:{line}: generated row count mismatch")
            if row["cols"] != str(generated["cols"]):
                raise CorpusValidationError(f"{fixtures_path}:{line}: generated col count mismatch")
            if row["nnz"] != str(len(entries)):
                raise CorpusValidationError(f"{fixtures_path}:{line}: generated nnz mismatch")
            if row["expected_rank"] != str(generated["expected_rank"]):
                raise CorpusValidationError(f"{fixtures_path}:{line}: generated rank mismatch")
            if row["nullity"] != str(generated["nullity"]):
                raise CorpusValidationError(f"{fixtures_path}:{line}: generated nullity mismatch")
            validate_known_generator(
                generators_path,
                generator_lines_by_key[row["generator_key"]],
                generators_by_key[row["generator_key"]],
            )
    for line, row in enumerate(optional_rows, start=2):
        assert_enum(
            optional_path,
            line,
            "availability_state",
            row["availability_state"],
            OPTIONAL_AVAILABILITY,
        )
        if row["availability_state"] != "available" and row["skip_reason"] == "":
            raise CorpusValidationError(
                f"{optional_path}:{line}: unavailable optional data requires skip_reason"
            )
        if row["availability_state"] == "deferred" and row["defer_reason"] == "":
            raise CorpusValidationError(
                f"{optional_path}:{line}: deferred optional data requires defer_reason"
            )
        if (
            row["availability_state"] != "available"
            and "pass" in row["skip_interpretation"].lower()
        ):
            raise CorpusValidationError(
                f"{optional_path}:{line}: skip_interpretation must not describe pass evidence"
            )
        if (
            row["availability_state"] != "available"
            and "parity" not in row["claim_boundary"].lower()
        ):
            raise CorpusValidationError(
                f"{optional_path}:{line}: claim_boundary must preserve external-parity non-claim"
            )

    for line, row in enumerate(report_family_rows, start=2):
        for field in ("report_family", "subfamily", "row_meaning", "status"):
            if not is_lower_snake(row[field]):
                raise CorpusValidationError(
                    f"{report_families_path}:{line}: {field} must be lowercase snake case"
                )
        assert_enum(
            report_families_path,
            line,
            "row_origin",
            row["row_origin"],
            REPORT_ROW_ORIGINS,
        )
        assert_enum(
            report_families_path,
            line,
            "row_meaning",
            row["row_meaning"],
            REPORT_ROW_MEANINGS,
        )
        assert_enum(
            report_families_path,
            line,
            "status",
            row["status"],
            REPORT_STATUSES,
        )
        assert_enum(
            report_families_path,
            line,
            "support_tier",
            row["support_tier"],
            SUPPORT_TIERS,
        )
        assert_enum(
            report_families_path,
            line,
            "freshness_policy",
            row["freshness_policy"],
            REPORT_FRESHNESS_POLICIES,
        )
        if row["status"] == "pass":
            raise CorpusValidationError(
                f"{report_families_path}:{line}: contract rows must not be pass evidence"
            )
        if "state-of-the-art" in row["claim_scope"].lower():
            raise CorpusValidationError(
                f"{report_families_path}:{line}: claim_scope must not assert state-of-the-art"
            )
        if row["non_claims"] == "":
            raise CorpusValidationError(
                f"{report_families_path}:{line}: non_claims must preserve boundaries"
            )

    for path in sorted(expected.glob("*.tsv")):
        expected_rows = read_tsv(path)
        require_fields(path, expected_rows, EXPECTED_REQUIRED, allow_empty={"tolerance_value"})
        for line, row in enumerate(expected_rows, start=2):
            fixture_key = row["fixture_key"]
            if fixture_key not in fixture_keys:
                raise CorpusValidationError(
                    f"{path}:{line}: fixture_key {fixture_key!r} not found"
                )
            assert_enum(path, line, "comparison_kind", row["comparison_kind"], COMPARISON_KINDS)
            assert_enum(
                path,
                line,
                "expected_result_kind",
                row["expected_result_kind"],
                EXPECTED_RESULT_KINDS,
            )
            assert_enum(path, line, "tolerance_kind", row["tolerance_kind"], TOLERANCE_KINDS)
            assert_enum(path, line, "status", row["status"], EXPECTED_STATUSES)
            if row["tolerance_kind"] in {"exact", "absolute", "relative", "mixed", "projector"}:
                if row["tolerance_value"] == "":
                    raise CorpusValidationError(
                        f"{path}:{line}: {row['tolerance_kind']} requires tolerance_value"
                    )
            elif row["tolerance_value"] != "":
                raise CorpusValidationError(
                    f"{path}:{line}: {row['tolerance_kind']} requires empty tolerance_value"
                )
            if "pass" in row["status"]:
                raise CorpusValidationError(
                    f"{path}:{line}: expected-result skeleton status must not be pass evidence"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_CORPUS_ROOT,
        type=Path,
        help="corpus root directory",
    )
    args = parser.parse_args()
    validate(args.root)
    print(f"validate-corpus-schema: {args.root} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
