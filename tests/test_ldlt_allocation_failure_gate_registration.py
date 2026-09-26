#!/usr/bin/env python3
"""Guard selected linked-list LDLT allocation-failure gate registration."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
CMAKE = ROOT / "CMakeLists.txt"
TEST_LDLT = ROOT / "tests" / "test_ldlt.c"


def require_contains(text: str, needle: str, *, owner: Path) -> None:
    if needle not in text:
        raise AssertionError(f"{owner.relative_to(ROOT)} missing: {needle}")


def require_active_run_test_once(text: str, test_name: str) -> None:
    pattern = re.compile(rf"^\s*RUN_TEST\({re.escape(test_name)}\);\s*$", re.MULTILINE)
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise AssertionError(
            "tests/test_ldlt.c must retain active proof-owner registration "
            f"RUN_TEST({test_name}) exactly once; found {len(matches)}"
        )


def main() -> None:
    makefile = MAKEFILE.read_text()
    cmake = CMAKE.read_text()
    test_ldlt = TEST_LDLT.read_text()

    require_contains(
        makefile,
        ".PHONY: ldlt-linked-list-allocation-failure-gate",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "ldlt-linked-list-allocation-failure-gate: $(BUILDDIR)/test_ldlt",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "python3 tests/test_ldlt_allocation_failure_gate_registration.py",
        owner=MAKEFILE,
    )

    require_contains(cmake, "add_sparse_test(test_ldlt)", owner=CMAKE)
    require_contains(
        cmake,
        'set_tests_properties(test_ldlt PROPERTIES LABELS "ldlt;linked_list;allocation_failure")',
        owner=CMAKE,
    )

    required_tests = [
        "test_ldlt_linked_list_allocation_failures_clear_outputs",
        "test_ldlt_linked_list_allocation_failures_recover_on_retry",
        "test_ldlt_linked_list_retry_matches_success_baseline",
        "test_ldlt_linked_list_allocation_failure_cleanup_repeatable",
        "test_ldlt_linked_list_allocation_failures_clear_stale_outputs",
        "test_ldlt_linked_list_success_cleanup_free_safe",
    ]
    for test_name in required_tests:
        require_active_run_test_once(test_ldlt, test_name)

    required_cases = [
        '{"D output array", 0}',
        '{"D_offdiag output array", 1}',
        '{"pivot_size output array", 2}',
        '{"working copy entry buffer", 3}',
        '{"working copy column-tail scratch", 11}',
        '{"L row headers", 12}',
        '{"permutation output array", 18}',
        '{"column accumulator workspace", 19}',
        '{"pivot-candidate nonzero list workspace", 24}',
    ]
    for case in required_cases:
        require_contains(test_ldlt, case, owner=TEST_LDLT)

    required_assertions = [
        "ASSERT_EQ((idx_t)ldlt_allocation_failure_case_count, 25);",
        "assert_ldlt_failure_output_free_safe(&ldlt);",
        "assert_ldlt_success_output_free_safe(&ldlt);",
        "assert_ldlt_stale_output_sentinel_seeded(&ldlt);",
        "assert_ldlt_success_outputs_match(&expected, &actual);",
        "assert_ldlt_allocation_hook_probe_after_reset();",
    ]
    for assertion in required_assertions:
        require_contains(test_ldlt, assertion, owner=TEST_LDLT)

    print("ldlt-allocation-failure-gate-registration: passed")


if __name__ == "__main__":
    main()
