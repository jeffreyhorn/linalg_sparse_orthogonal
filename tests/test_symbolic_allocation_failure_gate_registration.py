#!/usr/bin/env python3
"""Guard Sprint 195 symbolic Cholesky allocation-failure gate registration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
CMAKE = ROOT / "CMakeLists.txt"
TEST_ETREE = ROOT / "tests" / "test_etree.c"


def require_contains(text: str, needle: str, *, owner: Path) -> None:
    if needle not in text:
        raise AssertionError(f"{owner.relative_to(ROOT)} missing: {needle}")


def main() -> None:
    makefile = MAKEFILE.read_text()
    cmake = CMAKE.read_text()
    test_etree = TEST_ETREE.read_text()

    require_contains(
        makefile,
        ".PHONY: symbolic-allocation-failure-gate",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "symbolic-allocation-failure-gate: $(BUILDDIR)/test_etree",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "python3 tests/test_symbolic_allocation_failure_gate_registration.py",
        owner=MAKEFILE,
    )

    require_contains(cmake, "add_sparse_test(test_etree)", owner=CMAKE)
    require_contains(
        cmake,
        'set_tests_properties(test_etree PROPERTIES LABELS "etree;symbolic;allocation_failure")',
        owner=CMAKE,
    )

    required_tests = [
        "RUN_TEST(test_symbolic_cholesky_allocation_hook_reaches_empty_col_ptr);",
        "RUN_TEST(test_symbolic_cholesky_allocation_hook_reaches_nonempty_col_ptr);",
        "RUN_TEST(test_symbolic_cholesky_allocation_failures_clear_partial_state);",
        "RUN_TEST(test_symbolic_cholesky_allocation_failures_recover_on_retry);",
    ]
    for test_name in required_tests:
        require_contains(test_etree, test_name, owner=TEST_ETREE)

    required_cases = [
        '{"col_ptr", 0}',
        '{"row_idx", 1}',
        '{"child_head", 2}',
        '{"child_next", 3}',
        '{"marker", 4}',
        '{"tmp", 5}',
        '{"col_rows", 6}',
        '{"col_nrows", 7}',
        '{"propagated row set", 8}',
    ]
    for case in required_cases:
        require_contains(test_etree, case, owner=TEST_ETREE)

    require_contains(
        test_etree,
        "assert_symbolic_failure_free_safe(&sym);",
        owner=TEST_ETREE,
    )

    print("symbolic-allocation-failure-gate-registration: passed")


if __name__ == "__main__":
    main()
