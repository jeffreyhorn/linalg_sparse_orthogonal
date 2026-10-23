# Sprint 217 Day 13: Focused And Instrumented Validation

**Scope:** [Day 13 plan](../PLAN.md), the selected norm/ILUT/iterative
changes from Days 3-10, and the [Day 1 baseline](day1-norm-consumer-inventory.md).
All commands below ran locally on 2026-10-20 on branch `sprint-217`.

These counts describe the Day 13 tree, before the later PR #241 review fixes;
they are not counts for the final reviewed tree. See the
[post-fix validation](day14-validation-claim-closeout.md#post-fix-validation-2026-10-21)
for the subsequent Make gate and updated affected-binary counts. The UBSan,
ASan, and selected CTest results below were not rerun after those fixes.

## Focused Make Tests

`make -s build/test_sparse_vector build/test_ilu build/test_iterative
build/test_minres build/test_bicgstab build/test_block_solvers
build/test_stagnation build/test_suitesparse` exited 0. Each binary was then
run directly and exited 0:

| Binary | Tests run | Failed | Skipped | Assertions |
| --- | ---: | ---: | ---: | ---: |
| `test_sparse_vector` | 40 | 0 | 0 | 85 |
| `test_ilu` | 37 | 0 | 0 | 520 |
| `test_iterative` | 93 | 0 | 0 | 783 |
| `test_minres` | 46 | 0 | 0 | 719 |
| `test_bicgstab` | 52 | 0 | 0 | 387 |
| `test_block_solvers` | 17 | 0 | 0 | 179 |
| `test_stagnation` | 46 | 0 | 0 | 309 |
| `test_suitesparse` | 10 | 0 | 0 | 40 |
| **Total** | **341** | **0** | **0** | **3022** |

The ILUT `west0067` diagnostic now reports `finite_z=67/67`,
`max_abs_z=1.914849e+274`, `||z||_2=1.915e+274`, and the independent
`sqrt(67)*max|z|` upper bound `1.567371e+275`. Its factor-system backward
error is `8.249302e-80` against a `1e-12` limit. Both ILUT-preconditioned
GMRES variants remain nonconverged after 500 iterations (right-preconditioned
residual `6.228e+16`; pivoted residual `2.151e+05`). These values establish
a finite factor solve, not an accurate `A*z=r` solve or useful convergence.

## Instrumented Make Tests

| Target and command | Exit | Framework totals | Diagnostics |
| --- | ---: | --- | --- |
| `make -s sanitize BUILDDIR=/tmp/sprint217-day13-ubsan COVDIR=/tmp/sprint217-day13-cov` | 0 | 2001 run, 0 failed, 4 skipped, 1,177,540 assertions | No UBSan runtime diagnostic in `/tmp/sprint217-day13-ubsan.log` |
| `make -s asan CC=/usr/local/opt/llvm/bin/clang BUILDDIR=/tmp/sprint217-day13-asan COVDIR=/tmp/sprint217-day13-asan-cov` | 0 | 2001 run, 0 failed, 4 skipped, 1,177,540 assertions | No ASan report in `/tmp/sprint217-day13-asan.log` |

The Make targets rebuild and run the full test suite. The ASan target sets
`ASAN_OPTIONS=detect_leaks=0`, so these results do **not** establish leak
freedom. Both runs counted the same four framework skips:
`test_skip_macro_marks_current_test_skipped` (intentional framework self-test),
`test_slow_wrapper_body` (needs `SPARSE_TEST_SLOW=1`),
`test_experimental_wrapper_body` (needs `SPARSE_TEST_EXPERIMENTAL=1`), and
`test_analysis_typed_nd_sep_lift_weight_overrides_env` (the `bcsstk04` env
balance weight did not differ from the typed hybrid baseline). Two additional
test-internal `[SKIP]` messages are not included in the framework count:
large-matrix cases need `SPARSE_TEST_LARGE=1`, and
`test_concurrent_insert` needs `-DSPARSE_MUTEX`. None is counted as a pass.

The selected numeric-error fixtures exercised nonfinite RHS/factors,
overflowing ILUT substitution, computed solver failures, and invalid
preconditioner output under both instruments. `test_s217_ilut_nonfinite_solve_status`
checks that an invalid RHS leaves `z` unchanged in its one-element case and
that a later solve succeeds after restoring valid factors. Source inspection
of `sparse_ilu_solve` shows its allocated permutation scratch buffer is freed
on numeric-error returns; it writes only to `z`, not to a non-aliased `r`.
Other failures can leave `z` partially written, as the public contract says;
no failure output is used as a solution. Solver fixtures check
`converged=0` and, for selected computed errors, a NaN residual diagnostic.
This is focused failure-path evidence, not an exhaustive allocation-failure
or aliasing proof.

## CMake And CTest

`cmake -S . -B /tmp/sprint217-day13-cmake -DCMAKE_BUILD_TYPE=Debug` and the
build of the 12 selected test targets both exited 0. The following CTest
selection exited 0, with **12/12 passed** in 16.06 seconds:

```sh
ctest --test-dir /tmp/sprint217-day13-cmake --output-on-failure \
  -R '^test_(sparse_vector|ilu|iterative|minres|bicgstab|bicgstab_block|block_solvers|stagnation|suitesparse|integration|qr_corpus|svd_partial_corpus)$'
```

This covers the affected registrations plus integration, SuiteSparse, QR,
and selected SVD corpus consumers. The Apple Clang CMake build emitted
nonfatal `-Wdouble-promotion` warnings for float-typed `NAN`/`INFINITY`
macros in double-valued expressions; they were warnings, not failed checks.
CTest's process-level result does not erase the sanitizer-run skip ledger.

## Baseline And Day 14 Handoff

Day 1 passed smaller focused suites but printed `west0067: ILUT solve
norm=inf`, without testing whether `z` was finite. The current finite
`z`, finite scaled norm, and independent factor-system check resolve that
specific ambiguity: the old naive square/sum norm overflowed, while the
selected factor solve is finite. The newly covered numeric failures and
expanded focused test counts are the basis for the updated claim; no broad
solver-quality claim follows.

Sanitizer targets invoke `clean`. Their `BUILDDIR` and `COVDIR` were set to
fresh `/tmp` paths to avoid deleting the existing repository `build/` and
`coverage/` artifacts. The normal `build/test_ilu` links only `libSystem`,
not a sanitizer runtime, and the CMake build is also outside the repository.
No instrumented object remains in the normal `build/` tree. Day 14 can start
the required `make format && make lint && make test` sequence from the
ordinary build state. No Day 13 source edit was needed; the only nonfatal
build warnings and the explicit skip conditions above remain visible.
