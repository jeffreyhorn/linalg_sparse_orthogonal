#!/usr/bin/env python3
"""Guard Sprint 211 LDLT CSC helper ownership checks."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_ldlt_csc_helper_guard.sh"
BEHAVIOR_SCRIPT = REPO_ROOT / "tests" / "test_ldlt_csc_native_parity_behavior.py"

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
    (root / "tests" / BEHAVIOR_SCRIPT.name).write_text(
        BEHAVIOR_SCRIPT.read_text(encoding="utf-8"),
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
        "\t@bash scripts/check_ldlt_csc_helper_guard.sh\n"
        "\t@python3 tests/test_ldlt_csc_helper_guard.py\n"
        "\t@python3 tests/test_ldlt_csc_native_parity_behavior.py\n\n"
        ".PHONY: quality-review-compile\n"
        "quality-review-compile:\n"
        "\t@$(MAKE) ldlt-csc-helper-guard\n\n"
        ".PHONY: quality-review-full\n"
        "quality-review-full:\n"
        "\t@$(MAKE) quality-review-compile\n",
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


def assert_guard_passes_with(mutator) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_fixture(root)
        mutator(root)
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


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


def test_if_zero_else_native_helper_include_passes_guard() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"',
                "#if 0\n"
                '#include "test_ldlt_csc_native_parity_helpers.h"\n'
                "#else\n"
                '#include "test_ldlt_csc_native_parity_helpers.h"\n'
                "#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_passes_with(mutate)


def test_parenthesized_zero_native_helper_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"',
                "#if (0)\n"
                '#include "test_ldlt_csc_native_parity_helpers.h"\n'
                "#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "exactly once as an active include")


def test_path_qualified_native_helper_include_passes_guard() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"',
                '#include "helpers/test_ldlt_csc_native_parity_helpers.h"',
            ),
            encoding="utf-8",
        )

    assert_guard_passes_with(mutate)


def test_angle_bracket_native_helper_include_passes_guard() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"',
                "#include <test_ldlt_csc_native_parity_helpers.h>",
            ),
            encoding="utf-8",
        )

    assert_guard_passes_with(mutate)


def test_ifdef_native_helper_include_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"',
                "#ifdef ENABLE_NATIVE_HELPER\n"
                '#include "test_ldlt_csc_native_parity_helpers.h"\n'
                "#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "exactly once as an active include")


def test_ifndef_native_helper_include_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"',
                "#ifndef SKIP_NATIVE_HELPER\n"
                '#include "test_ldlt_csc_native_parity_helpers.h"\n'
                "#endif",
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

    assert_guard_fails_with(
        mutate,
        "must list exact prerequisite token $(TESTDIR)/test_ldlt_csc_native_parity_helpers.h",
    )


def test_commented_native_helper_makefile_prerequisite_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "$(TESTDIR)/test_ldlt_csc_native_parity_helpers.h ",
                "# $(TESTDIR)/test_ldlt_csc_native_parity_helpers.h ",
                1,
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "exactly once, only as a test_ldlt_csc prerequisite")


def test_suffix_native_helper_makefile_prerequisite_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "$(TESTDIR)/test_ldlt_csc_native_parity_helpers.h ",
                "$(TESTDIR)/test_ldlt_csc_native_parity_helpers.h.extra ",
                1,
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "must list exact prerequisite token $(TESTDIR)/test_ldlt_csc_native_parity_helpers.h",
    )


def test_multiline_makefile_prerequisite_rule_passes_guard() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(
                "$(BUILDDIR)/test_ldlt_csc: $(TESTDIR)/test_ldlt_csc.c "
                "$(TESTDIR)/test_ldlt_csc_fixtures.h "
                "$(TESTDIR)/test_ldlt_csc_native_parity_helpers.h "
                "$(TESTDIR)/test_ldlt_csc_oracle_helpers.h "
                "$(TESTDIR)/test_ldlt_csc_supernode_helpers.h $(LIB) | $(BUILDDIR)",
                "$(BUILDDIR)/test_ldlt_csc: $(TESTDIR)/test_ldlt_csc.c \\\n"
                "\t$(TESTDIR)/test_ldlt_csc_fixtures.h \\\n"
                "\t$(TESTDIR)/test_ldlt_csc_native_parity_helpers.h \\\n"
                "\t$(TESTDIR)/test_ldlt_csc_oracle_helpers.h \\\n"
                "\t$(TESTDIR)/test_ldlt_csc_supernode_helpers.h $(LIB) | $(BUILDDIR)",
            ),
            encoding="utf-8",
        )

    assert_guard_passes_with(mutate)


def test_makefile_guard_target_runs_python_suite_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\t@python3 tests/test_ldlt_csc_helper_guard.py\n",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "ldlt-csc-helper-guard target must run tests/test_ldlt_csc_helper_guard.py",
    )


def test_makefile_guard_target_runs_behavior_suite_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\t@python3 tests/test_ldlt_csc_native_parity_behavior.py\n",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "ldlt-csc-helper-guard target must run tests/test_ldlt_csc_native_parity_behavior.py",
    )


def test_quality_review_compile_runs_helper_guard_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\t@$(MAKE) ldlt-csc-helper-guard\n",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "quality-review-compile target must run ldlt-csc-helper-guard",
    )


def test_quality_review_full_runs_compile_gate_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\t@$(MAKE) quality-review-compile\n",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "quality-review-full target must run quality-review-compile",
    )


def test_makefile_guard_target_ignores_commented_python_suite_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\t@python3 tests/test_ldlt_csc_helper_guard.py\n",
                "\t# @python3 tests/test_ldlt_csc_helper_guard.py\n",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "ldlt-csc-helper-guard target must run tests/test_ldlt_csc_helper_guard.py",
    )


def test_quality_review_compile_ignores_commented_helper_guard_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\t@$(MAKE) ldlt-csc-helper-guard\n",
                "\t# @$(MAKE) ldlt-csc-helper-guard\n",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "quality-review-compile target must run ldlt-csc-helper-guard",
    )


def test_makefile_guard_target_ignores_echoed_python_suite_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\t@python3 tests/test_ldlt_csc_helper_guard.py\n",
                "\t@echo python3 tests/test_ldlt_csc_helper_guard.py\n",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "ldlt-csc-helper-guard target must run tests/test_ldlt_csc_helper_guard.py",
    )


def test_quality_review_compile_ignores_inline_comment_helper_guard_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "\t@$(MAKE) ldlt-csc-helper-guard\n",
                "\t@true # $(MAKE) ldlt-csc-helper-guard\n",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "quality-review-compile target must run ldlt-csc-helper-guard",
    )


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


def test_native_helper_angle_bracket_second_translation_unit_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "tests" / "test_other.c").write_text(
            "#include <test_ldlt_csc_native_parity_helpers.h>\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not be included by tests/test_other.c")


def test_native_helper_examples_translation_unit_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "examples").mkdir()
        (root / "examples" / "example_other.c").write_text(
            '#include "test_ldlt_csc_native_parity_helpers.h"\n',
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not be included by examples/example_other.c")


def test_native_helper_path_qualified_examples_include_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "examples").mkdir()
        (root / "examples" / "example_other.c").write_text(
            '#include "../tests/test_ldlt_csc_native_parity_helpers.h"\n',
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not be included by examples/example_other.c")


def test_native_helper_unknown_ifdef_second_translation_unit_include_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / "examples").mkdir()
        (root / "examples" / "example_other.c").write_text(
            "#ifdef ENABLE_NATIVE_HELPER\n"
            '#include "test_ldlt_csc_native_parity_helpers.h"\n'
            "#endif\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not be included by examples/example_other.c")


def test_native_helper_unknown_ifdef_duplicate_owner_include_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '#include "test_ldlt_csc_native_parity_helpers.h"\n',
                '#include "test_ldlt_csc_native_parity_helpers.h"\n'
                "#ifdef ENABLE_NATIVE_HELPER\n"
                '#include "test_ldlt_csc_native_parity_helpers.h"\n'
                "#endif\n",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "no additional possible-active includes")


def test_missing_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(RUN_TEST_MARKERS[0] + "\n", ""),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must retain proof-owner registration")


def test_duplicate_same_line_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"{RUN_TEST_MARKERS[0]} {RUN_TEST_MARKERS[0]}",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "as an active RUN_TEST line")


def test_unknown_ifdef_duplicate_run_test_registration_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"{RUN_TEST_MARKERS[0]}\n"
                "#ifdef ENABLE_NATIVE_REGISTRATION\n"
                f"{RUN_TEST_MARKERS[0]}\n"
                "#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "no additional possible-active registrations")


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


def test_octal_zero_run_test_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"#if 00\n{RUN_TEST_MARKERS[0]}\n#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "as an active RUN_TEST line")


def test_if_zero_else_run_test_registration_passes_guard() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"#if 0\n{RUN_TEST_MARKERS[0]}\n#else\n{RUN_TEST_MARKERS[0]}\n#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_passes_with(mutate)


def test_if_zero_elif_run_test_registration_passes_guard() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"#if 0\n{RUN_TEST_MARKERS[0]}\n#elif 1\n{RUN_TEST_MARKERS[0]}\n#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_passes_with(mutate)


def test_if_one_run_test_registration_passes_guard() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                f"#if 1\n{RUN_TEST_MARKERS[0]}\n#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_passes_with(mutate)


def test_unknown_primary_if_run_test_registration_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                "#if ENABLE_NATIVE_REGISTRATION\n"
                f"{RUN_TEST_MARKERS[0]}\n"
                "#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "as an active RUN_TEST line")


def test_if_zero_unknown_elif_run_test_registration_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RUN_TEST_MARKERS[0],
                "#if 0\n"
                f"{RUN_TEST_MARKERS[0]}\n"
                "#elif ENABLE_NATIVE_REGISTRATION\n"
                f"{RUN_TEST_MARKERS[0]}\n"
                "#endif",
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


def test_missing_solve_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc.c"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "RUN_TEST(test_solve_null_args);\n",
                "",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must retain Day 9 solve registration")


def test_behavior_suite_missing_selected_marker_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_behavior.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '"test_native_1x1_diagonal_matches_wrapper"',
                '"test_native_1x1_diagonal_drifted"',
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "behavior suite must pin selected native parity test",
    )


def test_behavior_suite_commented_selected_marker_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_behavior.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '    "test_native_1x1_diagonal_matches_wrapper",\n',
                '    # "test_native_1x1_diagonal_matches_wrapper",\n',
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "behavior suite must pin selected native parity tests in SELECTED_TESTS",
    )


def test_behavior_suite_summary_contract_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_behavior.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '"Assertions": 3556',
                '"Assertions": 0',
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "behavior suite must preserve 'Assertions' summary value 3556",
    )


def test_behavior_suite_commented_summary_contract_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_behavior.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '    "Assertions": 3556,\n',
                '    # "Assertions": 3556,\n    "Assertions": 0,\n',
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "behavior suite must preserve 'Assertions' summary value 3556",
    )


def test_behavior_suite_output_diagnostics_contract_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_behavior.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace("output_excerpt(output)", "output"),
            encoding="utf-8",
        )

    assert_guard_fails_with(
        mutate,
        "behavior suite must preserve command-output diagnostics on failure",
    )


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


def test_duplicate_same_line_moved_definition_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                MOVED_DEFINITION_MARKERS[0],
                f"{MOVED_DEFINITION_MARKERS[0]} {MOVED_DEFINITION_MARKERS[0]}",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must own moved selected-cluster definition")


def test_unknown_ifdef_duplicate_moved_definition_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                MOVED_DEFINITION_MARKERS[0],
                f"{MOVED_DEFINITION_MARKERS[0]}\n"
                "#ifdef ENABLE_NATIVE_DEFINITION\n"
                f"{MOVED_DEFINITION_MARKERS[0]}\n"
                "#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "no additional possible-active definitions")


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


def test_hex_zero_moved_definition_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                MOVED_DEFINITION_MARKERS[0],
                f"#if 0x0\n{MOVED_DEFINITION_MARKERS[0]}\n#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must own moved selected-cluster definition")


def test_if_zero_else_moved_definition_passes_guard() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                MOVED_DEFINITION_MARKERS[0],
                f"#if 0\n{MOVED_DEFINITION_MARKERS[0]}\n"
                f"#else\n{MOVED_DEFINITION_MARKERS[0]}\n#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_passes_with(mutate)


def test_ifndef_moved_definition_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                MOVED_DEFINITION_MARKERS[0],
                "#ifndef SKIP_NATIVE_DEFINITION\n"
                f"{MOVED_DEFINITION_MARKERS[0]}\n"
                "#endif",
            ),
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must own moved selected-cluster definition")


def test_ifdef_moved_definition_fails_closed() -> None:
    def mutate(root: Path) -> None:
        path = root / "tests" / "test_ldlt_csc_native_parity_helpers.h"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                MOVED_DEFINITION_MARKERS[0],
                "#ifdef ENABLE_NATIVE_DEFINITION\n"
                f"{MOVED_DEFINITION_MARKERS[0]}\n"
                "#endif",
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


def test_moved_definition_in_examples_translation_unit_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "examples").mkdir()
        (root / "examples" / "example_other.c").write_text(
            f"{MOVED_DEFINITION_MARKERS[0]} }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "examples/example_other.c must not own")


def test_unknown_ifdef_moved_definition_outside_native_helper_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / "examples").mkdir()
        (root / "examples" / "example_other.c").write_text(
            "#ifdef ENABLE_NATIVE_DEFINITION\n"
            f"{MOVED_DEFINITION_MARKERS[0]} }}\n"
            "#endif\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "examples/example_other.c must not own")


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


def test_native_helper_bare_library_source_registration_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "build-metadata" / "library_sources.txt"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "test_ldlt_csc_native_parity_helpers.h\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "must not be listed as a library source")


if __name__ == "__main__":
    test_current_tree_passes_guard()
    test_fixture_passes_guard()
    test_missing_native_helper_include_fails_clearly()
    test_line_commented_native_helper_include_fails_clearly()
    test_block_commented_native_helper_include_fails_clearly()
    test_if_zero_else_native_helper_include_passes_guard()
    test_parenthesized_zero_native_helper_include_fails_clearly()
    test_path_qualified_native_helper_include_passes_guard()
    test_angle_bracket_native_helper_include_passes_guard()
    test_ifdef_native_helper_include_fails_closed()
    test_ifndef_native_helper_include_fails_closed()
    test_missing_native_helper_makefile_prerequisite_fails_clearly()
    test_commented_native_helper_makefile_prerequisite_fails_clearly()
    test_suffix_native_helper_makefile_prerequisite_fails_clearly()
    test_multiline_makefile_prerequisite_rule_passes_guard()
    test_makefile_guard_target_runs_python_suite_fails_clearly()
    test_makefile_guard_target_runs_behavior_suite_fails_clearly()
    test_quality_review_compile_runs_helper_guard_fails_clearly()
    test_quality_review_full_runs_compile_gate_fails_clearly()
    test_makefile_guard_target_ignores_commented_python_suite_fails_clearly()
    test_quality_review_compile_ignores_commented_helper_guard_fails_clearly()
    test_makefile_guard_target_ignores_echoed_python_suite_fails_clearly()
    test_quality_review_compile_ignores_inline_comment_helper_guard_fails_clearly()
    test_native_helper_extra_makefile_registration_fails_clearly()
    test_native_helper_duplicate_same_line_makefile_registration_fails_clearly()
    test_native_helper_bare_makefile_registration_fails_clearly()
    test_native_helper_second_translation_unit_include_fails_clearly()
    test_native_helper_angle_bracket_second_translation_unit_include_fails_clearly()
    test_native_helper_examples_translation_unit_include_fails_clearly()
    test_native_helper_path_qualified_examples_include_fails_clearly()
    test_native_helper_unknown_ifdef_second_translation_unit_include_fails_closed()
    test_native_helper_unknown_ifdef_duplicate_owner_include_fails_closed()
    test_missing_run_test_registration_fails_clearly()
    test_duplicate_same_line_run_test_registration_fails_clearly()
    test_unknown_ifdef_duplicate_run_test_registration_fails_closed()
    test_line_commented_run_test_registration_fails_clearly()
    test_block_commented_run_test_registration_fails_clearly()
    test_if_zero_run_test_registration_fails_clearly()
    test_octal_zero_run_test_registration_fails_clearly()
    test_if_zero_else_run_test_registration_passes_guard()
    test_if_zero_elif_run_test_registration_passes_guard()
    test_if_one_run_test_registration_passes_guard()
    test_unknown_primary_if_run_test_registration_fails_closed()
    test_if_zero_unknown_elif_run_test_registration_fails_closed()
    test_reordered_run_test_registration_fails_clearly()
    test_selected_registration_after_solve_block_fails_clearly()
    test_missing_solve_registration_fails_clearly()
    test_behavior_suite_missing_selected_marker_fails_clearly()
    test_behavior_suite_commented_selected_marker_fails_clearly()
    test_behavior_suite_summary_contract_fails_clearly()
    test_behavior_suite_commented_summary_contract_fails_clearly()
    test_behavior_suite_output_diagnostics_contract_fails_clearly()
    test_moved_definition_missing_from_native_helper_fails_clearly()
    test_block_commented_moved_definition_fails_clearly()
    test_duplicate_same_line_moved_definition_fails_clearly()
    test_unknown_ifdef_duplicate_moved_definition_fails_closed()
    test_if_zero_moved_definition_fails_clearly()
    test_hex_zero_moved_definition_fails_clearly()
    test_if_zero_else_moved_definition_passes_guard()
    test_ifndef_moved_definition_fails_closed()
    test_ifdef_moved_definition_fails_closed()
    test_moved_definition_in_proof_owner_fails_clearly()
    test_moved_definition_in_wrong_helper_fails_clearly()
    test_moved_definition_in_examples_translation_unit_fails_clearly()
    test_unknown_ifdef_moved_definition_outside_native_helper_fails_closed()
    test_native_helper_cmake_registration_fails_clearly()
    test_native_helper_library_source_registration_fails_clearly()
    test_native_helper_bare_library_source_registration_fails_clearly()
