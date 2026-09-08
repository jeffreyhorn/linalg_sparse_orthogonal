#!/usr/bin/env python3
"""Guard selected symbolic LU allocation-failure gate registration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
TEST_ETREE = ROOT / "tests" / "test_etree.c"


def require_contains(text: str, needle: str, *, owner: Path) -> None:
    if needle not in text:
        raise AssertionError(f"{owner.relative_to(ROOT)} missing: {needle}")


def main() -> None:
    makefile = MAKEFILE.read_text()
    test_etree = TEST_ETREE.read_text()

    require_contains(
        makefile,
        ".PHONY: symbolic-lu-allocation-failure-gate",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "symbolic-lu-allocation-failure-gate: $(BUILDDIR)/test_etree",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "SPARSE_TEST_SYMBOLIC_LU_ALLOCATION_ONLY=1 $(BUILDDIR)/test_etree",
        owner=MAKEFILE,
    )

    require_contains(
        test_etree,
        'tf_env_enabled("SPARSE_TEST_SYMBOLIC_LU_ALLOCATION_ONLY")',
        owner=TEST_ETREE,
    )

    required_tests = [
        "RUN_TEST(test_symbolic_lu_allocation_failures_clear_outputs);",
        "RUN_TEST(test_symbolic_lu_allocation_failures_cleanup_sweep);",
        "RUN_TEST(test_symbolic_lu_allocation_failures_recover_on_retry);",
    ]
    for test_name in required_tests:
        require_contains(test_etree, test_name, owner=TEST_ETREE)

    required_lu_cases = [
        '{"perm seen", 6, 1}',
        '{"perm inverse", 7, 1}',
        '{"row_cols", 6, 0}',
        '{"parent", 7, 0}',
        '{"postorder", 8, 0}',
        '{"cc", 9, 0}',
        '{"sym_full col_ptr", 10, 0}',
        '{"sym_full row_idx", 11, 0}',
        '{"sym_full child_head", 12, 0}',
        '{"sym_full child_next", 13, 0}',
        '{"sym_full marker", 14, 0}',
        '{"sym_full tmp", 15, 0}',
        '{"sym_full col_rows", 16, 0}',
        '{"sym_full col_nrows", 17, 0}',
        '{"sym_full propagated row set", 18, 0}',
        '{"sym_U u_cnt", 19, 0}',
        '{"sym_U col_ptr", 20, 0}',
        '{"sym_U row_idx", 21, 0}',
    ]
    for case in required_lu_cases:
        require_contains(test_etree, case, owner=TEST_ETREE)

    required_assertions = [
        "assert_symbolic_failure_free_safe(&sym_L);",
        "assert_symbolic_failure_free_safe(&sym_U);",
        "assert_allocation_hook_probe_after_reset();",
        "assert_symbolic_lu_retry_output_fresh(A, &sym_L, &sym_U);",
    ]
    for assertion in required_assertions:
        require_contains(test_etree, assertion, owner=TEST_ETREE)

    print("symbolic-lu-allocation-failure-gate-registration: passed")


if __name__ == "__main__":
    main()
