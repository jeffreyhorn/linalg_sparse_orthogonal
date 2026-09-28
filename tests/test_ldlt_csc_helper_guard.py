#!/usr/bin/env python3
"""Guard Sprint 211 LDLT CSC helper ownership checks."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_ldlt_csc_helper_guard.sh"

HELPERS = [
    "test_ldlt_csc_fixtures.h",
    "test_ldlt_csc_native_parity_helpers.h",
    "test_ldlt_csc_oracle_helpers.h",
    "test_ldlt_csc_supernode_helpers.h",
]

RUN_TEST_MARKERS = [
    "RUN_TEST(test_native_1x1_diagonal_matches_wrapper);",
    "RUN_TEST(test_native_1x1_tridiagonal_matches_wrapper);",
    "RUN_TEST(test_native_1x1_mixed_indefinite_matches_wrapper);",
    "RUN_TEST(test_native_1x1_with_swap_matches_wrapper);",
    "RUN_TEST(test_native_1x1_tridiag_large_matches_wrapper);",
    "RUN_TEST(test_native_detects_near_zero_1x1_pivot);",
    "RUN_TEST(test_native_1x1_identity_matches_wrapper);",
    "RUN_TEST(test_native_2x2_forced_matches_wrapper);",
    "RUN_TEST(test_native_2x2_nonadjacent_partner_matches_wrapper);",
    "RUN_TEST(test_native_mixed_pivots_matches_wrapper);",
    "RUN_TEST(test_native_mixed_pivots_larger_matches_wrapper);",
    "RUN_TEST(test_native_2x2_solve_matches_linked_list);",
    "RUN_TEST(test_native_2x2_inertia_matches_wrapper);",
]

MOVED_DEFINITION_MARKERS = [
    "static void test_native_1x1_diagonal_matches_wrapper(void) {",
    "static void test_native_1x1_tridiagonal_matches_wrapper(void) {",
    "static void test_native_1x1_mixed_indefinite_matches_wrapper(void) {",
    "static void test_native_1x1_with_swap_matches_wrapper(void) {",
    "static void test_native_1x1_tridiag_large_matches_wrapper(void) {",
    "static void test_native_detects_near_zero_1x1_pivot(void) {",
    "static void test_native_1x1_identity_matches_wrapper(void) {",
    "static void test_native_2x2_forced_matches_wrapper(void) {",
    "static void test_native_2x2_nonadjacent_partner_matches_wrapper(void) {",
    "static void test_native_mixed_pivots_matches_wrapper(void) {",
    "static void test_native_mixed_pivots_larger_matches_wrapper(void) {",
    "static void test_native_2x2_solve_matches_linked_list(void) {",
    "static void test_native_2x2_inertia_matches_wrapper(void) {",
]


def helper_guard(name: str) -> str:
    return name.upper().replace(".", "_")


def write_fixture(root: Path) -> None:
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "build-metadata").mkdir()

    (root / "scripts" / SCRIPT.name).write_text(
        SCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    helper_prereqs = " ".join(f"$(TESTDIR)/{name}" for name in HELPERS)
    (root / "Makefile").write_text(
        "TEST_SRCS := $(TESTDIR)/test_ldlt_csc.c\n\n"
        f"$(BUILDDIR)/test_ldlt_csc: $(TESTDIR)/test_ldlt_csc.c {helper_prereqs} "
        "$(LIB) | $(BUILDDIR)\n"
        "\t$(CC) $< -o $@\n\n"
        ".PHONY: ldlt-csc-helper-guard\n"
        "ldlt-csc-helper-guard:\n"
        "\t@bash scripts/check_ldlt_csc_helper_guard.sh\n",
        encoding="utf-8",
    )
    (root / "CMakeLists.txt").write_text(
        "add_sparse_test(test_ldlt_csc)\n",
        encoding="utf-8",
    )
    (root / "build-metadata" / "library_sources.txt").write_text(
        "src/sparse_ldlt_csc.c\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_ldlt_csc.c").write_text(
        "".join(f'#include "{name}"\n' for name in HELPERS)
        + "\n"
        + "\n".join(RUN_TEST_MARKERS)
        + "\nRUN_TEST(test_solve_null_args);\n",
        encoding="utf-8",
    )
    for name in HELPERS:
        guard = helper_guard(name)
        definitions = ""
        if name == "test_ldlt_csc_native_parity_helpers.h":
            definitions = "\n".join(f"{marker} }}\n" for marker in MOVED_DEFINITION_MARKERS)
        (root / "tests" / name).write_text(
            f"#ifndef {guard}\n"
            f"#define {guard}\n\n"
            f"static inline void {name.replace('.', '_')}_fixture(void) {{}}\n"
            f"{definitions}"
            "#endif\n",
            encoding="utf-8",
        )


def run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/check_ldlt_csc_helper_guard.sh"],
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


def test_missing_native_helper_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"\n',
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must include test_ldlt_csc_native_parity_helpers.h")


def test_line_commented_native_helper_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"',
                '// #include "test_ldlt_csc_native_parity_helpers.h"',
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "exactly once as an active include")


def test_block_commented_native_helper_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"',
                '/* #include "test_ldlt_csc_native_parity_helpers.h" */',
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "exactly once as an active include")


def test_missing_native_helper_makefile_prerequisite_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                " $(TESTDIR)/test_ldlt_csc_native_parity_helpers.h",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must list test_ldlt_csc_native_parity_helpers.h")


def test_native_helper_extra_makefile_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "TEST_SRCS += $(TESTDIR)/test_ldlt_csc_native_parity_helpers.h\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "exactly once, only as a test_ldlt_csc prerequisite")


def test_native_helper_duplicate_same_line_makefile_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "$(TESTDIR)/test_ldlt_csc_native_parity_helpers.h",
                "$(TESTDIR)/test_ldlt_csc_native_parity_helpers.h "
                "$(TESTDIR)/test_ldlt_csc_native_parity_helpers.h",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "exactly once, only as a test_ldlt_csc prerequisite")


def test_native_helper_bare_makefile_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "TEST_SRCS += test_ldlt_csc_native_parity_helpers.h\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "exactly once, only as a test_ldlt_csc prerequisite")


def test_native_helper_second_translation_unit_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "tests" / "test_other.c").write_text(
            '#include "test_ldlt_csc_native_parity_helpers.h"\n',
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not be included by tests/test_other.c")


def test_missing_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(RUN_TEST_MARKERS[0] + "\n", ""),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must retain proof-owner registration")


def test_line_commented_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"// {RUN_TEST_MARKERS[0]}",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "as an active RUN_TEST line")


def test_block_commented_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"/* {RUN_TEST_MARKERS[0]} */",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "as an active RUN_TEST line")


def test_if_zero_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"#if 0\n{RUN_TEST_MARKERS[0]}\n#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "as an active RUN_TEST line")


def test_reordered_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        reordered = [RUN_TEST_MARKERS[1], RUN_TEST_MARKERS[0], *RUN_TEST_MARKERS[2:]]
        path.write_text(
            "".join(f'#include "{name}"\n' for name in HELPERS)
            + "\n"
            + "\n".join(reordered)
            + "\nRUN_TEST(test_solve_null_args);\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "selected RUN_TEST registrations changed order")


def test_selected_registration_after_solve_block_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            "".join(f'#include "{name}"\n' for name in HELPERS)
            + "\n"
            + "\n".join(RUN_TEST_MARKERS[:-1])
            + "\nRUN_TEST(test_solve_null_args);\n"
            + RUN_TEST_MARKERS[-1]
            + "\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must remain before Day 9 solve registration")


def test_moved_definition_missing_from_native_helper_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                f"{MOVED_DEFINITION_MARKERS[0]} }}\n",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must own moved selected-cluster definition")


def test_block_commented_moved_definition_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                MOVED_DEFINITION_MARKERS[0],
                f"/* {MOVED_DEFINITION_MARKERS[0]} */",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must own moved selected-cluster definition")


def test_if_zero_moved_definition_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                MOVED_DEFINITION_MARKERS[0],
                f"#if 0\n{MOVED_DEFINITION_MARKERS[0]}\n#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must own moved selected-cluster definition")


def test_moved_definition_in_proof_owner_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8") + f"\n{MOVED_DEFINITION_MARKERS[0]} }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not retain moved selected-cluster definition")


def test_moved_definition_in_wrong_helper_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_oracle_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8") + f"\n{MOVED_DEFINITION_MARKERS[0]} }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not own moved selected-cluster definition")


def test_native_helper_cmake_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "CMakeLists.txt"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "add_sparse_test(test_ldlt_csc_native_parity_helpers)\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not become a separate CMake test")


def test_native_helper_library_source_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "build-metadata" / "library_sources.txt"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "tests/test_ldlt_csc_native_parity_helpers.h\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not be listed as a library source")


if __name__ == "__main__":
    test_current_tree_passes_guard()
    test_fixture_passes_guard()
    test_missing_native_helper_include_fails_clearly()
    test_line_commented_native_helper_include_fails_clearly()
    test_block_commented_native_helper_include_fails_clearly()
    test_missing_native_helper_makefile_prerequisite_fails_clearly()
    test_native_helper_extra_makefile_registration_fails_clearly()
    test_native_helper_duplicate_same_line_makefile_registration_fails_clearly()
    test_native_helper_bare_makefile_registration_fails_clearly()
    test_native_helper_second_translation_unit_include_fails_clearly()
    test_missing_run_test_registration_fails_clearly()
    test_line_commented_run_test_registration_fails_clearly()
    test_block_commented_run_test_registration_fails_clearly()
    test_if_zero_run_test_registration_fails_clearly()
    test_reordered_run_test_registration_fails_clearly()
    test_selected_registration_after_solve_block_fails_clearly()
    test_moved_definition_missing_from_native_helper_fails_clearly()
    test_block_commented_moved_definition_fails_clearly()
    test_if_zero_moved_definition_fails_clearly()
    test_moved_definition_in_proof_owner_fails_clearly()
    test_moved_definition_in_wrong_helper_fails_clearly()
    test_native_helper_cmake_registration_fails_clearly()
    test_native_helper_library_source_registration_fails_clearly()
