# Sprint 103 Day 4 Helper and Reporting Boundary

## Purpose

Day 4 freezes the helper and reporting boundary before Sprint 103 adds
iterative, eigensolver, LOBPCG, thick-restart, or SVD comparison evidence. It
reviews Sprint 102 helper patterns, identifies repeated residual and
convergence reporting, and selects the smallest safe reuse strategy without
widening public APIs or hiding solver-family numerical semantics.

## Sprint 102 Helper Pattern Reviewed

Sprint 102 introduced an opt-in test-support helper in
`tests/test_solver_helpers.h` behind `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`:

```c
typedef enum {
    TF_EXTERNAL_REFERENCE_ERROR = -1,
    TF_EXTERNAL_REFERENCE_SKIP = 0,
    TF_EXTERNAL_REFERENCE_OK = 1
} tf_external_reference_status_t;
```

```c
tf_external_reference_status_t tf_read_external_reference_vector(
    const char *cmd,
    const char *label,
    double *x_out,
    idx_t n,
    char *reason,
    size_t reason_cap);
```

The helper owns only subprocess/vector parsing:

- `popen` / `_popen` command execution;
- `OK n` header parsing;
- `SKIP reason` and `ERROR reason` handling;
- vector entry parsing;
- dimension mismatch, truncation, parse, trailing-data, and non-zero-exit
  failure handling;
- CRLF-safe reason trimming.

It intentionally does not own:

- fixture construction;
- matrix loading;
- solver invocation;
- RHS or target construction;
- residual computation;
- convergence thresholds;
- eigenvalue, eigenvector, singular-value, rank, or reconstruction checks;
- public API behavior.

## Repeated Sprint 103 Reporting Patterns

| pattern | current owners | Day 4 assessment |
|---|---|---|
| true residual via `tf_relative_residual_l2(...)` | `tests/test_iterative.c`, `tests/test_minres.c`, `tests/test_bicgstab.c`, integration tests | reusable reporting concept, but not a new helper extraction target because each family has different tolerance and status semantics |
| convergence tuple printing | CG, GMRES, MINRES, BiCGSTAB, stagnation tests | standardize artifact wording first; do not add a printf helper |
| stagnation and residual-history checks | `tests/test_stagnation.c` | keep local because accepted alternatives differ by solver and fixture |
| Ritz/eigenpair residual checks | `tests/test_eigs.c`, `tests/test_eigs_lobpcg.c`, `tests/test_eigs_thick_restart.c` | candidate for later consolidation, but Day 4 keeps it local until Day 8 spectral design freezes thresholds |
| orthogonality checks | LOBPCG and SVD tests | conceptually shared, but matrix layout and tolerance ownership differ |
| SVD reconstruction/rank checks | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h` | keep local until Day 10/11 SVD scope freeze |
| external vector helper status checks | direct-solver external helpers today; possible future iterative helper | safe to reuse only when helper output is exactly vector-valued `OK n` / `SKIP` / `ERROR` |

## Boundary Decision

Day 4 selects **no new C helper extraction** for Day 5.

Rationale:

- The highest-ranked Day 5 candidate is BiCGSTAB comparison design, not helper
  implementation.
- Existing residual helpers already cover the common scalar residual need.
- Spectral and SVD helper extraction would be premature before Day 8 and Day
  10 freeze residual, orthogonality, singular-value, and reconstruction
  thresholds.
- Sprint 102's external-reference vector parser is safe to reuse later only if
  a future family-local helper emits the same vector output contract.
- A reporting artifact template gives Day 5 enough structure without touching
  `.c` or `.h` files.

## Reuse Rules

| candidate reuse | allowed in Sprint 103? | rule |
|---|---|---|
| `tf_read_external_reference_vector(...)` | yes, conditionally | only for helper commands that emit `OK n` plus exactly `n` scalar entries or `SKIP`/`ERROR` reason lines |
| `TF_EXTERNAL_REFERENCE_*` status enum | yes, conditionally | compare against named constants; do not use magic numbers |
| `tf_relative_residual_l2(...)` | yes | residual thresholds must be fixture-specific and recorded before implementation |
| family-local Ritz residual helpers | yes | keep local unless Day 8 proves a shared helper removes meaningful duplication |
| SVD orthogonality/reconstruction helpers | yes | keep local until SVD scope freeze |
| local `printf` convergence summaries | yes | output is diagnostic evidence only, not performance proof |
| external helper skip behavior | yes | skip earns no oracle claim and must include a reason |

## Convergence Reporting Contract

Every new Sprint 103 comparison artifact should report:

| field | required content |
|---|---|
| solver path | exact public API or internal helper path under test |
| fixture key and taxonomy class | Day 3 class plus construction or load path |
| profile class | fast, slow, stagnation-sensitive, restart-sensitive, tolerance-sensitive, orthogonality-sensitive, or rank-sensitive |
| reference behavior | constructed solution, direct-solver cross-check, closed-form spectrum, dense tridiagonal reference, external helper, or internal invariant |
| tolerance model | residual, solution, eigenvalue, eigenpair residual, orthogonality, singular-value, reconstruction, or rank threshold |
| expected status | success, skip, expected non-convergence, stagnation, breakdown, or expected failure |
| iteration or basis budget | threshold only if declared before implementation; otherwise descriptive |
| validation command | focused test command and full gate if `.c` or `.h` changes |
| non-claims | broad parity, portable timing, and unsupported cases that remain unearned |

## Skip/Error Behavior Table

| condition | status | required test behavior | claim impact |
|---|---|---|---|
| external helper not runnable | `TF_EXTERNAL_REFERENCE_SKIP` | `SKIP_TEST(reason)` if the lane is optional | no oracle claim earned |
| helper emits `SKIP reason` | `TF_EXTERNAL_REFERENCE_SKIP` | skip with copied reason | no oracle claim earned |
| helper emits `ERROR reason` | `TF_EXTERNAL_REFERENCE_ERROR` | fail the comparison | helper/reference failure, not a solver pass |
| malformed helper output | `TF_EXTERNAL_REFERENCE_ERROR` | fail the comparison | no oracle claim earned |
| expected non-convergence fixture | family-local status | assert declared non-converged/stagnated/breakdown status | supports failure-mode claim only |
| residual above threshold on expected-success fixture | family-local failure | fail the test | solver or threshold regression |
| iteration count differs without threshold | diagnostic only | do not fail unless predeclared | no performance claim |

## Day 5 Focused Validation Plan

Day 5 is expected to remain design/documentation-only unless the iterative
batch design uncovers an unavoidable helper gap.

If Day 5 changes planning documentation only:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_103
```

If Day 5 changes BiCGSTAB tests:

```sh
make build/test_bicgstab
./build/test_bicgstab
make format && make lint && make test
git diff --check
```

If Day 5 changes shared test helpers:

```sh
make build/test_bicgstab build/test_iterative build/test_minres
./build/test_bicgstab
./build/test_iterative
./build/test_minres
make format && make lint && make test
git diff --check
```

If Day 5 adds an external helper script:

```sh
python3 <helper> <fixture-key>
make build/<affected-test>
./build/<affected-test>
make format && make lint && make test
git diff --check
```

## Day 5 Implementation Boundary

Day 5 should freeze the first iterative comparison batch before implementation.
Recommended scope:

- primary family: BiCGSTAB;
- preferred fixture classes: `nonsym-known-solution`, `nonsym-mm-medium`, and
  selected `ill-conditioned-scale` only if tolerance is predeclared;
- preferred reference behavior: constructed solution plus LU or GMRES
  cross-check; external dense helper only if its output contract is frozen;
- no new shared C helper unless repeated code blocks are proven meaningful and
  validation cost is accepted.

## Explicit Non-Extraction

Sprint 103 Day 4 does not extract:

- generic iterative solver oracle logic;
- generic convergence result printers;
- shared Ritz residual helpers;
- shared SVD rank or reconstruction helpers;
- external ARPACK, LAPACK, NumPy, SciPy, PETSc, or SuiteSparse integration;
- public API helpers;
- build-system registration for new tests.

## Non-Claims Preserved

This boundary does not claim:

- any new comparison implementation has landed;
- BiCGSTAB, eigensolver, LOBPCG, thick-restart, or SVD has external helper
  parity;
- grow-m parity is an independent thick-restart oracle;
- local iteration counts are portable performance evidence;
- residual, orthogonality, or reconstruction thresholds are universal outside
  named fixtures.

## Day 4 Conclusion

Sprint 103 should proceed to Day 5 with a reporting contract, not a new helper
extraction. The safe reusable implementation surface is Sprint 102's
external-reference vector parser, and only for future helper commands that
match its exact status and vector-output contract. Residual, orthogonality,
rank, reconstruction, and convergence semantics remain family-local until a
later design artifact proves a narrower extraction is justified.
