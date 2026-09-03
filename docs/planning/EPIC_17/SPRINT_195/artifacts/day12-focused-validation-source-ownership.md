# Sprint 195 Day 12: Focused Validation and Source Ownership

## Purpose

Run the focused Sprint 195 reliability gate, source/build ownership checks,
formatting checks, CMake label proof, and claim-boundary checks before the full
Day 13 quality gate.

## Validation Commands

```sh
make source-list-check
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
cmake -S . -B build/sprint195-day12-cmake
cmake --build build/sprint195-day12-cmake --target test_etree --parallel 1
ctest -N --test-dir build/sprint195-day12-cmake -L symbolic
ctest --test-dir build/sprint195-day12-cmake -L symbolic --output-on-failure
rg -n "symbolic-allocation-failure-gate|sparse_symbolic_cholesky\(\)|broad allocation-failure|state-of-the-art reliability|Local selected allocation-failure proof|ctest --test-dir <build-dir> -L symbolic" README.md INSTALL.md docs/maintainer_guide.md docs/planning/EPIC_17/SPRINT_195
make format-check
git diff --check
```

## Results

| Check | Result |
| --- | --- |
| Source-list ownership | `make source-list-check` passed with 49 library sources. |
| Registration guard | `python3 tests/test_symbolic_allocation_failure_gate_registration.py` passed. |
| Focused Make gate | `make symbolic-allocation-failure-gate` passed; `test_etree` ran 101 tests, 0 failures, 0 skips, and 1262 assertions. |
| CMake configure | `cmake -S . -B build/sprint195-day12-cmake` passed. |
| CMake selected build | `cmake --build build/sprint195-day12-cmake --target test_etree --parallel 1` passed. |
| CTest selector listing | `ctest -N --test-dir build/sprint195-day12-cmake -L symbolic` selected only `test_etree`. |
| CTest selector execution | `ctest --test-dir build/sprint195-day12-cmake -L symbolic --output-on-failure` passed with 1 of 1 tests passing. |
| Claim-boundary grep | Targeted grep found the new symbolic gate and non-claim wording in `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and Sprint 195 artifacts. |
| Formatting | `make format-check` passed. |
| Whitespace | `git diff --check` passed. |

## Fix Log

No Day 12 fixes were required after validation. The source-list, CMake label,
Make gate, focused registration guard, documentation wording, formatting, and
diff checks were already synchronized after Days 10 and 11.

## Remaining Risk

Day 12 does not replace the Day 13 full quality gate. Because Sprint 195
changed `.c` files, Day 13 still needs `make format`, `make lint`, and
`make test`.
