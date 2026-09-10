#!/usr/bin/env python3
"""Guard Sprint 201 SVD helper ownership checks."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_svd_helper_guard.sh"
SHARED_HELPER = "test_svd_helpers.h"
SELECTED_HELPER = "test_svd_selected_helpers.h"

MOVED_MARKERS = [
    "static inline void tf_svd_test_rank_full(void) {",
    "static inline void tf_svd_test_rank_deficient(void) {",
    "static inline void tf_svd_test_rank_nearly_singular(void) {",
    "static inline void tf_svd_test_rank_diagonal_threshold_fixture(void) {",
    "static inline void tf_svd_test_qr_rank_dependent_row_fixture(void) {",
    "static inline void tf_svd_test_rank_null(void) {",
    "static inline void tf_svd_test_pinv_diagonal(void) {",
    "static inline void tf_svd_test_pinv_moore_penrose(void) {",
    "static inline void tf_svd_test_pinv_null(void) {",
    "static inline void tf_svd_test_pinv_rectangular(void) {",
    "static inline void tf_svd_test_pinv_underdetermined_minnorm_solution(void) {",
    "static inline void tf_svd_test_lowrank_diagonal(void) {",
    "static inline void tf_svd_test_lowrank_error_bound(void) {",
    "static inline void tf_svd_test_lowrank_errors(void) {",
]

RUN_TEST_MARKERS = [
    "RUN_TEST(test_svd_rank_full);",
    "RUN_TEST(test_svd_rank_deficient);",
    "RUN_TEST(test_svd_rank_nearly_singular);",
    "RUN_TEST(test_svd_rank_diagonal_threshold_fixture);",
    "RUN_TEST(test_svd_qr_rank_dependent_row_fixture);",
    "RUN_TEST(test_svd_rank_null);",
    "RUN_TEST(test_pinv_diagonal);",
    "RUN_TEST(test_pinv_moore_penrose);",
    "RUN_TEST(test_pinv_null);",
    "RUN_TEST(test_pinv_rectangular);",
    "RUN_TEST(test_pinv_underdetermined_minnorm_solution);",
    "RUN_TEST(test_lowrank_diagonal);",
    "RUN_TEST(test_lowrank_error_bound);",
    "RUN_TEST(test_lowrank_errors);",
]


def write_fixture(root: Path) -> None:
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "build-metadata").mkdir()

    (root / "scripts" / SCRIPT.name).write_text(
        SCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "Makefile").write_text(
        "TEST_SRCS := $(TESTDIR)/test_svd.c\n\n"
        "$(BUILDDIR)/test_svd: $(TESTDIR)/test_svd.c "
        "$(TESTDIR)/test_svd_helpers.h "
        "$(TESTDIR)/test_svd_selected_helpers.h $(LIB) | $(BUILDDIR)\n"
        "\t$(CC) $< -o $@\n\n"
        ".PHONY: svd-helper-guard\n"
        "svd-helper-guard:\n"
        "\t@bash scripts/check_svd_helper_guard.sh\n",
        encoding="utf-8",
    )
    (root / "CMakeLists.txt").write_text("add_sparse_test(test_svd)\n", encoding="utf-8")
    (root / "build-metadata" / "library_sources.txt").write_text(
        "src/sparse_svd.c\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_svd.c").write_text(
        f'#include "{SHARED_HELPER}"\n'
        f'#include "{SELECTED_HELPER}"\n\n'
        + "\n".join(RUN_TEST_MARKERS)
        + "\n",
        encoding="utf-8",
    )
    (root / "tests" / SHARED_HELPER).write_text(
        "#ifndef TEST_SVD_HELPERS_H\n"
        "#define TEST_SVD_HELPERS_H\n\n"
        '#include "sparse_matrix.h"\n'
        '#include "sparse_svd.h"\n'
        '#include "test_framework.h"\n\n'
        "static inline void tf_svd_shared_fixture(void) {}\n"
        "#endif\n",
        encoding="utf-8",
    )
    (root / "tests" / SELECTED_HELPER).write_text(
        "#ifndef TEST_SVD_SELECTED_HELPERS_H\n"
        "#define TEST_SVD_SELECTED_HELPERS_H\n\n"
        '#include "sparse_qr.h"\n'
        '#include "sparse_svd.h"\n'
        '#include "sparse_vector.h"\n'
        f'#include "{SHARED_HELPER}"\n\n'
        + "\n".join(MOVED_MARKERS)
        + "\n#endif\n",
        encoding="utf-8",
    )


def run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/check_svd_helper_guard.sh"],
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


def test_missing_shared_helper_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_svd.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                f'#include "{SHARED_HELPER}"\n',
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        f"must include {SHARED_HELPER} exactly once",
    )


def test_missing_selected_helper_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_svd.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                f'#include "{SELECTED_HELPER}"\n',
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        f"must include {SELECTED_HELPER} exactly once",
    )


def test_missing_qr_dependency_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / SELECTED_HELPER
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "sparse_qr.h"\n',
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must include sparse_qr.h")


def test_missing_svd_dependency_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / SELECTED_HELPER
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "sparse_svd.h"\n',
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must include sparse_svd.h")


def test_missing_vector_dependency_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / SELECTED_HELPER
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "sparse_vector.h"\n',
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must include sparse_vector.h")


def test_moved_definition_in_test_svd_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_svd.c"
        path.write_text(
            path.read_text(encoding="utf-8") + MOVED_MARKERS[0] + "\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "tests/test_svd.c still owns moved selected-cluster definition",
    )


def test_moved_definition_in_shared_helper_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / SHARED_HELPER
        path.write_text(
            path.read_text(encoding="utf-8") + MOVED_MARKERS[0] + "\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "tests/test_svd_helpers.h still owns moved selected-cluster definition",
    )


def test_missing_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_svd.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(RUN_TEST_MARKERS[0] + "\n", ""),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must retain proof-owner registration")


def test_reordered_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_svd.c"
        reordered = [RUN_TEST_MARKERS[1], RUN_TEST_MARKERS[0], *RUN_TEST_MARKERS[2:]]
        path.write_text(
            f'#include "{SHARED_HELPER}"\n'
            f'#include "{SELECTED_HELPER}"\n\n'
            + "\n".join(reordered)
            + "\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "selected RUN_TEST registrations changed order")


def test_missing_makefile_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "TEST_SRCS := $(TESTDIR)/test_svd.c",
                "TEST_SRCS :=",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "Makefile no longer registers test_svd.c")


def test_missing_cmake_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "CMakeLists.txt"
        path.write_text("", encoding="utf-8")

    assert_guard_fails_with(mutate, "CMakeLists.txt no longer registers test_svd")


def test_missing_selected_helper_makefile_prerequisite_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                " $(TESTDIR)/test_svd_selected_helpers.h",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must list test_svd_selected_helpers.h")


def test_selected_helper_prerequisite_moved_to_other_rule_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        text = path.read_text(encoding="utf-8")
        text = text.replace(" $(TESTDIR)/test_svd_selected_helpers.h", "")
        text += "\n$(BUILDDIR)/other_test: $(TESTDIR)/test_svd_selected_helpers.h\n"
        path.write_text(text, encoding="utf-8")

    assert_guard_fails_with(mutate, "test_svd prerequisite rule")


def test_selected_helper_included_by_second_translation_unit_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "tests" / "test_svd_partial_corpus.c").write_text(
            f'#include "{SELECTED_HELPER}"\n',
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must be included only by tests/test_svd.c")


def test_selected_helper_cmake_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "CMakeLists.txt"
        path.write_text(
            path.read_text(encoding="utf-8") + "add_sparse_test(test_svd_selected_helpers)\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not become a separate CMake test")


def test_selected_helper_library_source_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "build-metadata" / "library_sources.txt"
        path.write_text(
            path.read_text(encoding="utf-8") + "tests/test_svd_selected_helpers.h\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not be listed as a library source")


if __name__ == "__main__":
    test_current_tree_passes_guard()
    test_fixture_passes_guard()
    test_missing_shared_helper_include_fails_clearly()
    test_missing_selected_helper_include_fails_clearly()
    test_missing_qr_dependency_include_fails_clearly()
    test_missing_svd_dependency_include_fails_clearly()
    test_missing_vector_dependency_include_fails_clearly()
    test_moved_definition_in_test_svd_fails_clearly()
    test_moved_definition_in_shared_helper_fails_clearly()
    test_missing_run_test_registration_fails_clearly()
    test_reordered_run_test_registration_fails_clearly()
    test_missing_makefile_registration_fails_clearly()
    test_missing_cmake_registration_fails_clearly()
    test_missing_selected_helper_makefile_prerequisite_fails_clearly()
    test_selected_helper_prerequisite_moved_to_other_rule_fails_clearly()
    test_selected_helper_included_by_second_translation_unit_fails_clearly()
    test_selected_helper_cmake_registration_fails_clearly()
    test_selected_helper_library_source_registration_fails_clearly()
