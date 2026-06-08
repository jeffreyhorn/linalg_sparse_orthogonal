# Sprint 59 Day 11 - Epic 5 summary and handoff batch

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Land the main Epic 5 handoff draft from the measured Sprint 59 state without
opening a project-level file rewrite yet.

This artifact is the high-signal summary layer that the final Sprint 59
closeout will build on. It is intentionally organized around stable work bands
and preserved contracts rather than a day-by-day sprint chronology.

## Epic 5 summary

Epic 5 started from a structurally strong post-Epic-4 state and finished by
closing the largest remaining lifecycle, CSC, repeated-run, maintainability,
public-surface, and quality/platform productization seams without reopening
the core public API fence.

The work now reduces cleanly to eight closed bands.

### 1. Direct-solver lifecycle design fence

Sprint 50 fixed the public repeated-run direct contract and the compatibility
boundary before implementation:

- `sparse_analysis_t`
- `sparse_factors_t`
- analyze once
- factor / solve
- refactor / solve many
- free explicitly

It also fixed the main non-goals:

- no broad direct-handle redesign
- no demotion of one-shot direct APIs
- no raw CSC/native storage exposure

### 2. Public direct lifecycle implementation and deeper integration

Sprints 51-52 turned that design into a real validated public workflow:

- public direct lifecycle headers were refreshed
- LU / Cholesky / LDL^T one-shot wrappers were preserved
- the shared analysis/factor/refactor path was strengthened
- factor-many benchmark proof became explicit
- public repeated-run regression coverage expanded

Result:

- repeated direct solves now read as a real public workflow rather than an
  under-documented internal path

### 3. CSC direct-solver completion and dispatch follow-through

Sprint 53 closed the most important remaining CSC direct-solver seams:

- deeper analysis-aware indefinite LDL^T CSC completion
- tighter LDL^T CSC dispatch ownership
- truthful forced-CSC telemetry
- real indefinite repeated-run benchmark proof
- stronger CSC repeated-run regression evidence

Result:

- the CSC direct-solver story is now materially closer to the rest of the
  public repeated-run direct workflow

### 4. Public repeated-run solver lifecycle completion

Sprint 54 fixed the steady-state public repeated-run solver support boundary:

- iterative handles:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver handle:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`

And it preserved the intended exclusions:

- `BiCGSTAB`
- block iterative workflows

Result:

- the public repeated-run story is now explicit instead of implied

### 5. Large-source decomposition

Sprints 55-56 reduced the largest remaining implementation hotspots in bounded
owned slices:

- `src/sparse_eigs.c`: `3233 -> 1534`
- `src/sparse_iterative.c`: `2377 -> 1985`
- `src/sparse_ldlt_csc.c`: `2723 -> 2127`
- `src/sparse_chol_csc.c`: `2194 -> 1532`
- `src/sparse_svd.c`: `1728 -> 1319`

Representative extracted files:

- `src/sparse_eigs_lobpcg.c`
- `src/sparse_eigs_thick_restart.c`
- `src/sparse_iterative_minres.c`
- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_svd_partial.c`

Result:

- the highest-risk implementation hotspots are materially smaller and more
  cleanly owned

### 6. Giant-test refactor and lifecycle/factor-many regression expansion

Sprint 57 reduced the largest remaining proof-surface helper density and added
two high-value lifecycle regressions:

- helper seam extractions in:
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `tests/test_svd_partial_helpers.h`
  - `tests/test_iterative_handle_helpers.h`
- direct proof for:
  - repeated `sparse_factor_solve(...)` reuse
  - safe zeroed-state free behavior
  - same-pattern refactor-many parity with one-shot Cholesky

Result:

- the repeated-run direct story is better proven
- the giant-test queue is smaller and more explicit

### 7. Public-surface simplification

Sprint 58 simplified the highest-signal caller-facing surfaces:

- `README.md`
- `docs/tutorial.md`
- `include/sparse_eigs.h`
- `include/sparse_iterative.h`
- `examples/example_eigs.c`
- `examples/README.md`
- `benchmarks/README.md`

Result:

- the public workflow story is shorter, clearer, and more stable

### 8. Final quality/platform reconciliation and caller-story normalization

Sprint 59 Days 1-11 reduced the final residual queue and top-level drift:

- quality/platform residual dispositions are now explicit and current
- caller-story terminology now matches the stable repeated-run workflow
  categories more closely
- the Epic 5 finish now has a measured closeout-input set and a main handoff
  draft

Result:

- the branch is now positioned for a final measured validation sweep and
  explicit Epic-level closeout

## Preserved compatibility fence

Epic 5 preserved one stable public product fence throughout:

- one-shot APIs remain first-class/default workflows
- repeated-run direct solves remain the explicit analysis/factors lifecycle:
  - analyze once
  - factor / solve
  - refactor / solve many
- repeated-run iterative handles remain limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated-run eigensolver handle remains limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces
- no broad public API redesign, raw internal storage exposure, or generic
  universal solver handle was introduced

## Validation baseline carried into final closeout

As of Day 11, Sprint 59 has remained docs-only, so the strongest inherited
validation baseline is still the Sprint 58 Day 13 state:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained truthfulness anchors:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake count anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity:
  - `53 vs 53`
- full reviewed CMake `ctest`:
  - `53 / 53`
- latest inherited reviewed CMake total time:
  - `481.74 sec`

The Sprint 59 Day 13 validation sweep is the remaining step that should
supersede the time-sensitive measured baseline for final Epic 5 closeout.

## Consciously deferred residual queue

The remaining future-facing queue is now bounded and explicit.

### Quality/platform residuals

- dead-code execution remains serialized
- macOS dead-code remains staged pending fresh evidence
- broader Windows reviewed-wrapper parity remains deferred
- Windows dead-code remains deferred/excluded
- coverage calibration is no longer an active residual

### Maintainability residuals

- later iterative decomposition:
  - `GMRES`
  - shared block-wrapper scaffolding
- possible later eigensolver/private-header cleanup
- later CSC decomposition/comment cleanup if still justified
- deferred giant-test seams:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - intentionally retained dense `tests/test_integration.c`

### Public-surface density residuals

- deeper long-form `README.md` chronology/performance-history cleanup
- broader docs-density reduction outside the bounded Sprint 58-59 scope

### Non-goals that should remain closed by default

- generic direct-handle redesign
- raw CSC/native storage exposure
- broad repeated-run support-boundary expansion
- late benchmark/example workflow redesign

## Handoff position

Epic 5 no longer needs more design work to explain what the repo is supposed
to be.

What remains is:

- final project-level residual wording check, only if needed
- final Sprint 59 validation sweep
- final Sprint 59 closeout and retrospective

That is a materially smaller and more explicit finish than the Epic 5 review
queue implied at the start of Sprint 50.
