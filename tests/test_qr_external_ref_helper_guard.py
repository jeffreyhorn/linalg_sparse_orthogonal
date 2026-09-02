#!/usr/bin/env python3
"""Guard Sprint 193 QR external-reference helper ownership checks."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_qr_external_ref_helper_guard.sh"

MOVED_MARKERS = [
    "static int read_qr_basis_external_reference",
    "static int read_qr_threshold_external_reference",
    "static void test_qr_external_reference_readers_reject_invalid_arguments(void) {",
    "static void test_qr_external_reference_readers_reject_unsupported_fixtures(void) {",
    "static void test_qr_external_dense_reference_rank1_4x3_nullspace_projector(void) {",
    "static void test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector(void) {",
    "static void test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector(void) {",
    "static void test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace(void) {",
    "static void test_qr_external_dense_reference_rank_threshold_diag4_family(void) {",
    "static void test_qr_external_dense_reference_rank_threshold_diag4_scaled_family(void) {",
    "static void test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family(void) {",
    "test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family(void) {",
]

RUN_TEST_MARKERS = [
    "RUN_TEST(test_qr_external_reference_readers_reject_invalid_arguments);",
    "RUN_TEST(test_qr_external_reference_readers_reject_unsupported_fixtures);",
    "RUN_TEST(test_qr_external_dense_reference_rank1_4x3_nullspace_projector);",
    "RUN_TEST(test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector);",
    "RUN_TEST(test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector);",
    "RUN_TEST(test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace);",
    "RUN_TEST(test_qr_external_dense_reference_rank_threshold_diag4_family);",
    "RUN_TEST(test_qr_external_dense_reference_rank_threshold_diag4_scaled_family);",
    "RUN_TEST(test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family);",
    "RUN_TEST(test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family);",
]


def write_fixture(root: Path) -> None:
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "build-metadata").mkdir()
    (root / "docs").mkdir()

    (root / "scripts" / SCRIPT.name).write_text(
        SCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "Makefile").write_text(
        "TEST_SRCS := $(TESTDIR)/test_qr.c\n\n"
        ".PHONY: qr-external-ref-helper-guard\n"
        "qr-external-ref-helper-guard:\n"
        "\t@bash scripts/check_qr_external_ref_helper_guard.sh\n",
        encoding="utf-8",
    )
    (root / "CMakeLists.txt").write_text("add_sparse_test(test_qr)\n", encoding="utf-8")
    (root / "build-metadata" / "library_sources.txt").write_text(
        "src/sparse_qr.c\n",
        encoding="utf-8",
    )
    (root / "docs" / "maintainer_guide.md").write_text(
        "Sprint 193 QR external-reference helper boundary\n"
        "`tests/test_qr_external_ref_helpers.h` owns the selected QR\n"
        "`tests/test_qr.c` remains the registered QR proof-owner binary\n"
        "`make qr-external-ref-helper-guard`\n"
        "no-behavior-change review-surface reduction\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_qr.c").write_text(
        '#include "test_qr_external_ref_helpers.h"\n\n'
        "static void test_qr_external_dense_reference_economy_projector_5x3(void) {}\n"
        + "\n".join(RUN_TEST_MARKERS)
        + "\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_qr_external_ref_helpers.h").write_text(
        "#ifndef TEST_QR_EXTERNAL_REF_HELPERS_H\n"
        "#define TEST_QR_EXTERNAL_REF_HELPERS_H\n\n"
        '#include "test_qr_helpers.h"\n'
        '#include "test_solver_helpers.h"\n\n'
        + "\n".join(MOVED_MARKERS)
        + "\n#endif\n",
        encoding="utf-8",
    )


def run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/check_qr_external_ref_helper_guard.sh"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def assert_guard_fails_with(mutator, expected: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_fixture(root)
        mutator(root)
        result = run_guard(root)
        if result.returncode == 0:
            raise AssertionError("expected guard failure")
        message = result.stdout + result.stderr
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}")


def test_current_tree_passes_guard() -> None:
    result = run_guard(REPO_ROOT)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def test_fixture_passes_guard() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_fixture(root)
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_missing_helper_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_qr.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_qr_external_ref_helpers.h"\n',
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "must include test_qr_external_ref_helpers.h exactly once",
    )


def test_missing_qr_helper_dependency_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_qr_external_ref_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_qr_helpers.h"\n',
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must include test_qr_helpers.h")


def test_moved_definition_in_test_qr_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_qr.c"
        path.write_text(
            path.read_text(encoding="utf-8") + MOVED_MARKERS[0] + "\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "tests/test_qr.c still owns moved selected-cluster definition",
    )


def test_economy_body_moved_to_helper_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        test_path = root / "tests" / "test_qr.c"
        helper_path = root / "tests" / "test_qr_external_ref_helpers.h"
        economy = "static void test_qr_external_dense_reference_economy_projector_5x3(void) {}"
        test_path.write_text(
            test_path.read_text(encoding="utf-8").replace(economy + "\n", ""),
            encoding="utf-8",
        )
        helper_path.write_text(
            helper_path.read_text(encoding="utf-8") + economy + "\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must retain the economy external-reference proof-owner body")


def test_helper_makefile_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8") + "EXTRA := tests/test_qr_external_ref_helpers.h\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must remain header-only")


def test_missing_maintainer_docs_marker_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "maintainer_guide.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "`make qr-external-ref-helper-guard`\n",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "missing the QR helper guard command")


if __name__ == "__main__":
    test_current_tree_passes_guard()
    test_fixture_passes_guard()
    test_missing_helper_include_fails_clearly()
    test_missing_qr_helper_dependency_include_fails_clearly()
    test_moved_definition_in_test_qr_fails_clearly()
    test_economy_body_moved_to_helper_fails_clearly()
    test_helper_makefile_registration_fails_clearly()
    test_missing_maintainer_docs_marker_fails_clearly()
