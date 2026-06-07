# Sprint 57 Day 3 - direct-solver giant-test seam audit

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Reduce the strongest direct-solver giant tests to a concrete refactor map by
separating their live ownership bands, ranking the real bounded seam options,
and fixing the strongest first target before any permanent test refactor lands.

## Audited files

Primary direct-solver giant-test surfaces:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_integration.c` = `1803`

These were audited using:

- top-of-file ownership/comments
- helper/test function inventory
- internal section/comment boundaries
- grouped `RUN_TEST(...)` blocks in `main()`
- the existing shared `tests/test_solver_helpers.h` seam

## Ownership shapes

### `tests/test_chol_csc.c`

Live seam classes:

- matrix/setup and conversion scaffolding
  - alloc/grow
  - from/to sparse round-trips
  - permutation and validate checks
- scalar-path numeric proof
  - workspace
  - scalar eliminate
  - solve / factor-solve
- supernodal/backend proof
  - supernode detection
  - postorder regression
  - dense helper cross-checks
  - supernode extract/writeback
  - supernode eliminate diag/panel
  - supernodal residual sweeps
  - writeback and dispatch tail

Key fact:

- this file has `137` tests but only a small local helper layer, so the main
  problem is proof-cluster overconcentration rather than missing helper reuse

### `tests/test_ldlt_csc.c`

Live seam classes:

- alloc / row-adjacency / conversion scaffolding
- analysis-aware repeated-run and supernodal indefinite proof
- native-kernel / symmetric-swap / pivot-behavior proof
- solve / inertia / singularity proof

Key fact:

- this file has `96` tests and a denser local helper/builder layer than the
  Cholesky test, which makes it a strong later helper-oriented target but not
  the cleanest first split

### `tests/test_integration.c`

Live seam classes:

- baseline LU workflows
- progress/cancel coverage
- explicit analysis/factor lifecycle parity checks
- public repeated-run direct lifecycle error and preservation checks
- QR / iterative / eigensolver progress coverage tail

Key fact:

- this file is smaller and already grouped by caller-story intent, so it is
  better treated as a later regression-expansion surface than as the first
  giant-test refactor target

## Ranked seam options

### Rank 1: `tests/test_chol_csc.c` supernodal / writeback / dispatch cluster

Why it ranks first:

- largest direct-solver giant test in the repo
- strongest contiguous high-mass proof cluster
- helper extraction alone would not relieve the main review cost
- cleanest bounded split by behavior family:
  - scalar-path proof can remain
  - supernodal/writeback/dispatch proof can become the first owned refactor
    boundary

Likely first owned seam:

- supernodal detection, postorder, dense-helper cross-checks, extract/writeback
  proof, elimination residual sweeps, and dispatch tail

### Rank 2: `tests/test_ldlt_csc.c` analysis-aware supernodal / native cluster

Why it ranks second:

- still very large
- already has clearer local helper/builder density
- indefinite, native-kernel, and analysis-aware repeated-run proof are all
  real families

Why it does not rank first:

- the family boundary is less clean because supernodal indefinite,
  symmetric-swap, native kernel, and solve/inertia proof are more tightly
  interleaved

### Rank 3: `tests/test_integration.c` lifecycle proof helper normalization

Why it ranks third:

- smaller than the giant CSC family binaries
- caller-story grouping is already fairly good
- strongest value is later additive lifecycle/factor-many coverage, not
  immediate giant-file relief

## Rejected first moves

Rejected as the first Sprint 57 direct-test refactor:

- generic assertion/helper extraction only
  - too low-impact on the main giant-file review cost
- early `test_integration.c` split
  - risks scattering cross-family caller-story proof with low maintainability
    payoff
- widening `tests/test_solver_helpers.h` into a broad CSC utility layer
  - fights the existing narrow-helper policy and would blur ownership instead
    of clarifying it

## Recommended landing order

1. Design and land a bounded `test_chol_csc.c` supernodal/writeback/dispatch
   refactor boundary
2. Follow with a bounded `test_ldlt_csc.c` family-local seam
3. Leave `test_integration.c` mostly intact until the lifecycle/factor-many
   expansion days prove that a smaller local helper move is justified

## Conclusion

The direct-solver giant-test problem is now concrete:

- the first Sprint 57 target should be `test_chol_csc.c`
- the first boundary should follow a real proof-family split, not a mechanical
  helper cleanup
- `test_ldlt_csc.c` is the strongest second target
- `test_integration.c` should remain a later caller-story coverage surface

That gives Day 4 a clear starting point for the first bounded direct-solver
test refactor design.
