# Sprint 121 Retrospective

**Sprint:** 121 - SVD, QR & Rank-Deficient Numerical Oracle Expansion
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 121 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Epic 11 Sprint 121 scope and inherited Sprint 120 oracle
      architecture patterns.
- [x] Audited SVD, partial-SVD, low-rank, rank, and pseudoinverse evidence
      owners.
- [x] Audited QR, least-squares, rectangular, and rank-deficient evidence
      owners.
- [x] Designed deterministic matrix taxonomy coverage for rank,
      conditioning, rectangularity, sparsity, scaling, and expected failures.
- [x] Planned bounded helper extraction before touching test code.
- [x] Extracted reusable SVD fixture, reconstruction, orthogonality, rank,
      low-rank, and pseudoinverse proof helpers.
- [x] Extracted reusable QR fixture, reconstruction, residual,
      rank-deficient, least-squares, and generated-RHS proof helpers.
- [x] Expanded rank-deficient and threshold-rank fixture coverage for QR and
      SVD.
- [x] Expanded compatible, incompatible, and minimum-norm QR solve evidence.
- [x] Expanded pseudoinverse, low-rank, and partial-SVD fixture evidence.
- [x] Designed one bounded SVD external dense-reference pilot with explicit
      skip behavior, tolerance, and non-claim boundaries.
- [x] Implemented the bounded external-reference pilot in
      `tests/svd_external_dense_reference.py` and `tests/test_svd.c`.
- [x] Validated focused QR, QR solve, SVD, and external-reference surfaces.
- [x] Ran the required full C quality gate for the branch's `.c` and `.h`
      changes: `make format && make lint && make test`.
- [x] Published build-system, CTest, source-list, and membership non-impact
      evidence.
- [x] Published explicit non-claims for LAPACK, SciPy, NumPy, SuiteSparse,
      PETSc, Trilinos, Eigen, external parity, performance, platform, ABI, and
      state-of-the-art positioning.
- [x] Published residual SVD, QR, partial-SVD, helper-ownership, and
      documentation queues for future sprints.
- [x] Finalized this retrospective and ran focused documentation hygiene.

## What Went Well

1. **The sprint separated taxonomy, helper extraction, and proof expansion.**
   The first five days built the evidence map, fixture taxonomy, and helper
   movement plan before implementation. That kept the later C/header changes
   grounded in named proof owners instead of opportunistic cleanup.

2. **Helper extraction reduced duplication without hiding numerical meaning.**
   `tests/test_svd_helpers.h`, `tests/test_svd_partial_helpers.h`, and
   `tests/test_qr_helpers.h` now hold reusable builders and measurement
   helpers, while rank, residual, tolerance, and non-claim assertions remain
   visible in scenario tests.

3. **Rank-deficient and rectangular evidence improved across related owners.**
   QR rank fixtures, SVD rank fixtures, least-squares cases, pseudoinverse
   cases, low-rank fixtures, and partial-SVD vector checks now share a clearer
   deterministic matrix vocabulary.

4. **The external-reference pilot stayed bounded.**
   The pilot compares full-SVD singular values for one fixed 6x4 fixture
   against a pure-Python standard-library dense reference. It avoids
   NumPy/SciPy/LAPACK dependencies and is documented as a fixture-local
   validation lane rather than broad dense-library parity.

5. **Validation was consolidated before closeout.**
   Day 13 collected the focused QR/SVD package and full
   `make format && make lint && make test` evidence. Day 14 then closed the
   sprint from that evidence instead of adding new claims.

6. **Non-claims were carried throughout the sprint.**
   Each major artifact preserved the distinction between deterministic
   in-repository evidence and public claims about external libraries,
   performance, package support, or state-of-the-art behavior.

## What Did Not Go Well

1. **External-reference coverage is intentionally narrow.**
   The Sprint 121 pilot is useful, but it covers one small full-SVD
   singular-value fixture only. It does not cover SVD vectors, QR, partial SVD,
   broad matrix families, or external library parity.

2. **Some helper candidates remain too semantic to extract safely.**
   Generic assertion wrappers, Bidiagonal/Golub-Kahan helpers, and
   minimum-norm ownership helpers still risk hiding tolerance or ownership
   meaning. Deferring them was correct, but the duplication remains visible.

3. **Public-facing guidance did not change.**
   Sprint 121 added internal validation evidence, not a new support guarantee.
   That means README, solver-selection, package, and install docs remain
   unchanged until broader evidence justifies public wording.

4. **Platform-specific external-helper behavior still relies on policy, not
   broad local proof.**
   The helper follows existing external-reference skip behavior, including
   Windows skip policy, but local validation did not prove every platform lane.

5. **The work touched dense existing test owners.**
   `tests/test_qr.c`, `tests/test_qr_solve.c`, and `tests/test_svd.c` remain
   substantial files even after helper extraction. Sprint 121 improved
   ownership boundaries, but it did not make those files small.

## Final Metrics

### Validation

| Metric | Sprint 121 close state |
|---|---:|
| focused SVD external helper | passed, emitted 4 singular values |
| focused QR tests | 65 passed, 0 failed |
| focused QR assertions | 603 |
| focused QR solve tests | 13 passed, 0 failed |
| focused QR solve assertions | 1014 |
| focused SVD tests | 104 passed, 0 failed |
| focused SVD assertions | 1685 |
| SVD external-reference pilot max difference | `6.217e-15` |
| required full Make quality | `make format && make lint && make test` passed |
| full Make test final result | `All tests passed.` |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 121 docs |
| Day 14 C quality rerun | not required; documentation-only closeout |

### Sprint Artifact Package

| Metric | Sprint 121 close state |
|---|---:|
| artifact files under `SPRINT_121/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| new test helper headers | 2 |
| new external-reference helper scripts | 1 |
| modified existing test files | 4 |
| Makefile/CMake/CTest registration changes | 0 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| SVD helper ownership | Completed first reusable helper batch in `tests/test_svd_helpers.h` and `tests/test_svd_partial_helpers.h`. |
| QR helper ownership | Completed first reusable helper batch in `tests/test_qr_helpers.h`. |
| Rank-deficient fixtures | Expanded QR and SVD deterministic rank and threshold-rank coverage. |
| Least-squares evidence | Expanded compatible, incompatible, and underdetermined minimum-norm QR solve evidence. |
| Pseudoinverse evidence | Expanded deterministic underdetermined pseudoinverse evidence. |
| Low-rank and partial-SVD evidence | Expanded rectangular low-rank and partial-SVD vector evidence. |
| External-reference pilot | Completed bounded pure-Python full-SVD singular-value comparison. |
| Build registration | Unchanged; no new test executable or library source was added. |
| Public API | Unchanged. |
| Public documentation claims | Unchanged; closeout records non-claims. |
| Benchmarks/performance | Not claimed and not refreshed. |
| External-library parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Decide whether to add more SVD external fixtures beyond the fixed 6x4 pilot.
- Design a QR external dense-reference lane only after fixture size, tolerance,
  skip behavior, and failure interpretation are explicit.
- Design partial-SVD external parity separately from full-SVD parity because
  vector/subspace and convergence semantics differ.
- Revisit minimum-norm helper ownership migration only when a future QR solve
  or minimum-norm proof owner is being consolidated.
- Keep Bidiagonal/Golub-Kahan helper extraction separate from general SVD
  helpers because those checks encode specialized transpose and reconstruction
  semantics.
- Update public solver-selection wording only after broader external or
  support-level evidence lands.

Still consciously constrained rather than silently solved:

- no LAPACK parity claim;
- no SciPy or NumPy parity claim;
- no SuiteSparse, PETSc, Trilinos, or Eigen parity claim;
- no broad external dense-library parity claim;
- no singular-vector or subspace external parity claim;
- no QR external parity claim;
- no partial-SVD external parity claim;
- no low-rank or pseudoinverse global optimality claim;
- no package/install/platform/ABI claim;
- no performance or scalability claim;
- no state-of-the-art claim;
- no public API claim.

Not carried forward as unresolved Sprint 121 debt:

- SVD/partial-SVD/low-rank/rank/pseudoinverse audit;
- QR/least-squares/rank-deficient audit;
- matrix taxonomy design;
- bounded helper extraction plan;
- first SVD helper extraction batch;
- first QR helper extraction batch;
- deterministic rank-deficient fixture expansion;
- least-squares and pseudoinverse fixture expansion;
- low-rank and partial-SVD fixture expansion;
- bounded SVD external-reference pilot design and implementation;
- final validation and closeout package.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-svd-evidence-audit.md](./artifacts/day2-svd-evidence-audit.md)
- [day3-qr-rank-deficient-evidence-audit.md](./artifacts/day3-qr-rank-deficient-evidence-audit.md)
- [day4-matrix-taxonomy-design.md](./artifacts/day4-matrix-taxonomy-design.md)
- [day5-helper-extraction-plan.md](./artifacts/day5-helper-extraction-plan.md)
- [day6-svd-helper-extraction.md](./artifacts/day6-svd-helper-extraction.md)
- [day7-qr-helper-extraction.md](./artifacts/day7-qr-helper-extraction.md)
- [day8-rank-deficient-fixture-expansion.md](./artifacts/day8-rank-deficient-fixture-expansion.md)
- [day9-ls-pinv-expansion.md](./artifacts/day9-ls-pinv-expansion.md)
- [day10-lowrank-partial-svd-expansion.md](./artifacts/day10-lowrank-partial-svd-expansion.md)
- [day11-reference-pilot-design.md](./artifacts/day11-reference-pilot-design.md)
- [day12-reference-pilot-implementation.md](./artifacts/day12-reference-pilot-implementation.md)
- [day13-validation-package.md](./artifacts/day13-validation-package.md)
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md)
