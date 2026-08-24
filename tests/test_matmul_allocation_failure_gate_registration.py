#!/usr/bin/env python3
"""Guard Sprint 178 matrix multiply allocation-failure gate registration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
CMAKE = ROOT / "CMakeLists.txt"
TEST_MATMUL = ROOT / "tests" / "test_matmul.c"


def require_contains(text: str, needle: str, *, owner: Path) -> None:
    if needle not in text:
        raise AssertionError(f"{owner.relative_to(ROOT)} missing: {needle}")


def main() -> None:
    makefile = MAKEFILE.read_text()
    cmake = CMAKE.read_text()
    test_matmul = TEST_MATMUL.read_text()

    require_contains(
        makefile,
        ".PHONY: matmul-allocation-failure-gate",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "matmul-allocation-failure-gate: $(BUILDDIR)/test_matmul",
        owner=MAKEFILE,
    )
    require_contains(
        makefile,
        "python3 tests/test_matmul_allocation_failure_gate_registration.py",
        owner=MAKEFILE,
    )

    require_contains(cmake, "add_sparse_test(test_matmul)", owner=CMAKE)
    require_contains(
        cmake,
        'set_tests_properties(test_matmul PROPERTIES LABELS "matmul;allocation_failure")',
        owner=CMAKE,
    )

    required_tests = [
        "RUN_TEST(test_matmul_error_precedence_clears_stale_output);",
        "RUN_TEST(test_matmul_acc_allocation_failure_clears_stale_output);",
        "RUN_TEST(test_matmul_remaining_workspace_allocation_failures_clear_stale_output);",
        "RUN_TEST(test_matmul_workspace_allocation_failure_recovers);",
    ]
    for test_name in required_tests:
        require_contains(test_matmul, test_name, owner=TEST_MATMUL)

    print("matmul-allocation-failure-gate-registration: passed")


if __name__ == "__main__":
    main()
