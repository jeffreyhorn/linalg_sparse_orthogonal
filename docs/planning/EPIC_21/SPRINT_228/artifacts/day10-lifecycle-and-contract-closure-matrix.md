# Sprint 228 Day 10: Lifecycle and Contract Closure Matrix

## Outcome

The sparse low-rank public API now has an in-process lifecycle regression that
alternates maintained-default and compatibility successes with selector,
allocation, and numeric failures. No prior call changes a later call's route,
status, values, publication state, allocator state, or output ownership.

This closes implementation lifecycle work before Day 11 measurement. It does
not add a performance or whole-SVD memory claim.

## Mixed-call replay

`test_s228_lowrank_mixed_route_failure_lifecycle` runs two complete cycles on
one input and process:

1. unset-selector default success, comparison with a pinned baseline, destroy;
2. malformed-selector `SPARSE_ERR_BADARG` with NULL output;
3. exact `on` in the first cycle and exact `off` in the second;
4. allocation failure at the first checked allocation with NULL output;
5. immediate success without an intervening allocator reset, comparison, destroy;
6. non-finite-input `SPARSE_ERR_NUMERIC` with NULL output;
7. immediate success on the original input and destroy;
8. finite `DBL_MAX` threshold producing a valid empty matrix and destroy; and
9. all-zero input producing a second valid empty matrix and destroy.

The original diagonal input is checked cell-for-cell after each cycle. The
empty outputs pass the same map, metadata, row-chain, column-chain, count, and
ownership validator used by nonempty differential results.

## Frozen contract-to-test matrix

| Frozen clause | Expected behavior | Automated owner |
| --- | --- | --- |
| NULL result pointer | `SPARSE_ERR_NULL`, no dereference | `test_s228_lowrank_frozen_error_precedence` |
| Valid result publication | set `*result = NULL` before inspecting later arguments | precedence, selector, numeric, and allocation regressions |
| NULL input | `SPARSE_ERR_NULL`, NULL output | `test_s228_lowrank_frozen_error_precedence` |
| Invalid rank | `SPARSE_ERR_BADARG` before matrix state, tolerance, or selector | `test_s228_lowrank_frozen_error_precedence`, Day 9 public corpus |
| Factored or nonidentity input state | `SPARSE_ERR_BADARG` before tolerance or selector | `test_s228_lowrank_frozen_error_precedence` |
| Non-finite tolerance | `SPARSE_ERR_BADARG` before selector/SVD | precedence and Day 9 public corpus |
| Selector parsing | unset/`on` maintained; `off` compatibility; all other values fail | selector validation/isolation and mixed lifecycle tests |
| Non-finite matrix data | `SPARSE_ERR_NUMERIC`, NULL output | Day 9 public corpus and mixed lifecycle test |
| Economy-SVD allocation failure | propagated `SPARSE_ERR_ALLOC`, NULL output | Day 9 exhaustive public sweeps |
| All-zero input | fresh caller-owned empty matrix after validation | degenerate differential and mixed lifecycle tests |
| Finite tolerance `<= 0` | default `DBL_EPSILON * sigma[0]` | Day 9 negative-tolerance and differential tests |
| Positive high threshold | fresh structurally valid empty matrix | mixed lifecycle test |
| Exact threshold equality | value retained | one-ULP threshold differential test |
| Non-finite factor/product/sum/cell | `SPARSE_ERR_NUMERIC`, partial output destroyed | outer hardening regressions |
| Allocation failure at any reached site | `SPARSE_ERR_ALLOC`, partial state destroyed | Day 9 exhaustive sweeps |
| Failure ownership | output NULL; input unchanged; all temporary owners released | Day 9 sweeps, sanitizer run, mixed lifecycle test |
| Success ownership | new identity-mapped, unfactored matrix; caller frees independently | all differential and mixed lifecycle tests |
| Sparse structure | exact shape/nnz; sorted unique row/column links; finite nonzero nodes | shared structural validator across deterministic corpus |
| Retry | clean or immediate retry reproduces accepted output | outer numeric retry, exhaustive sweeps, mixed lifecycle test |

`SPARSE_ERR_NOT_CONVERGED` originates in the shared SVD implementation and is
returned directly before reconstruction ownership begins. Dedicated partial-SVD
tests exercise that status; the sparse low-rank wrapper has no deterministic
public option for forcing full-SVD nonconvergence, so the closure claim does not
invent an unreachable injection seam.

## Verification gate

The focused 32-bit and isolated 64-bit binaries each report 135 tests, 19,333
assertions, zero failures, and zero skips. An isolated Homebrew Clang 22.1.5
ASan+UBSan build with leak detection passes the same suite without a report.
The repository-wide format, lint, test, API-documentation, and diff checks also
pass; exact commands and the cumulative sprint state are recorded in
`WORKING_NOTES.md`.

## Day 11 handoff

Retain the lifecycle and exhaustive failure suites unchanged while adding the
frozen reconstruction-only allocation counters and fresh-process RSS packet.
Measurement instrumentation must not broaden the accepted claim beyond removal
of the additional default-route `m*n` reconstruction accumulator.
