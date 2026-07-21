# Sprint 129 Day 11 Minimum-Norm Helper Movement Decision

## Purpose

Day 11 applies the Day 10 minimum-norm helper ownership gate. No helper is
moved. The tentative 2 x 4 helper candidate does not pass the Day 11
acceptance checklist once the concrete fixture layouts are compared.

This is an explicit deferral, not an unfinished implementation.

## Decision

Generic minimum-norm helper consolidation is deferred. No `.c` or `.h` files
are changed for Day 11.

The only tentative Day 10 candidate was a 2 x 4 split-constraint fixture
builder. Day 11 rejects moving it because the existing 2 x 4 lanes are not one
single behavior fixture:

| Owner | Fixture layout | Behavior use |
| --- | --- | --- |
| QR solve | `row0: x0 + x1 = 1`, `row1: x2 + x3 = 1` | External-reference QR minimum-norm fixture and exact `[0.5, 0.5, 0.5, 0.5]` solution. |
| COLAMD/minimum-norm | `row0: x0 + x2 = 1`, `row1: x1 + x3 = 1` | Owner-local exact solution and minimality lane. |
| SVD pseudoinverse | `row0: x0 + x1 = 1`, `row1: x2 + x3 = 1` | SVD-owned pseudoinverse minimum-norm check with storage-layout comments. |

A helper that covers all three would need a topology parameter or a generic
fixture name. That would hide the behavior owner and violate the Day 10 rule
against generic `tf_minnorm_*` or QR/SVD-neutral helpers. A helper that covers
only QR solve and SVD would cross the SVD pseudoinverse owner boundary without
enough SVD-local duplication to justify movement.

## Day 10 Checklist Result

| Checklist item | Result | Evidence |
| --- | --- | --- |
| Helper name encodes behavior and owner | Failed for a shared helper | A single 2 x 4 helper would either need a generic name or a topology parameter spanning QR/COLAMD/SVD owners. |
| Call sites keep shape, RHS, expected solution/norm, tolerance, and diagnostics visible | Passed only if no movement occurs | Current call sites keep expectations explicit. Moving construction would reduce visible fixture topology. |
| Helper does not turn SVD pseudoinverse into a global QR oracle | Failed for QR+SVD sharing | The SVD lane shares the QR-solve topology but has pseudoinverse storage/layout semantics that should stay SVD-owned. |
| Fallback, refinement, zero-row, COLAMD ordering, rank-deficient, and SuiteSparse lanes remain owner-specific | Passed by deferral | No helper movement touches those lanes. |
| Public/build surfaces unchanged unless separately gated | Passed | No public header, package header, ABI, CMake, or Makefile change. |
| Focused validation pinned before edits | Passed by deferral | No code edits require focused QR solve, COLAMD, or SVD execution. |
| Full gate for `.c` or `.h` edits | Not required | Day 11 changes documentation only. |

## Deferred Items

| Deferred item | Future owner | Promotion gate |
| --- | --- | --- |
| QR-solve-local 2 x 4 fixture builder | QR solve owner | Only if at least two QR-solve call sites share the exact same topology and expected values while keeping tolerance and diagnostics visible. |
| COLAMD-local interleaved 2 x 4 fixture builder | COLAMD/minimum-norm owner | Only if multiple COLAMD lanes share the exact interleaved topology and owner-specific options remain visible. |
| SVD pseudoinverse 2 x 4 fixture builder | SVD owner | Only if multiple SVD pseudoinverse lanes reuse the same topology and storage-layout comments remain at call sites. |
| SVD pseudoinverse application helper | SVD owner | Needs layout-specific naming, explicit dimensions, and focused SVD plus COLAMD QR-vs-SVD validation if shared. |
| SuiteSparse submatrix builder | COLAMD/minimum-norm or future corpus owner | Needs corpus support-tier, runtime, shape, and missing-data policy before movement. |
| Generic minimum-norm assertion/helper | No current owner | Rejected unless a future sprint defines a stronger owner boundary than Sprint 129 has. |

## Files Changed

| File | Change |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_129/WORKING_NOTES.md` | Added Day 11 deferral notes. |
| `docs/planning/EPIC_11/SPRINT_129/artifacts/day11-minnorm-helper-movement-decision.md` | Recorded helper movement decision, checklist result, deferred items, validation, and non-claims. |

No C source, header, Python helper, Matrix Market data, build file, maintainer
guide, public API, or public wording file changed for Day 11.

## Maintainer Guide Decision

No maintainer-guide update is required on Day 11. The day records a helper
movement deferral and does not add an evidence lane, helper protocol, public
API behavior, external fixture key, or user-visible claim.

## Validation

Day 11 changes documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

No code quality gate is required for Day 11 because no `.c` or `.h` file
changed for this day.

## Non-Claims Preserved

- No generic QR/SVD/minimum-norm helper API.
- No SVD pseudoinverse-as-global-QR-oracle claim.
- No public API, package, ABI, CMake, Makefile, CI, CTest, install-header, or
  helper protocol claim.
- No broad QR minimum-norm, COLAMD, fallback, refinement, zero-row,
  SuiteSparse corpus, optional-data, platform, backend, performance,
  scalability, or memory claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity claim.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Helper ownership is clearer than before, or deferral is explicit. | Complete | The tentative helper is explicitly deferred because concrete 2 x 4 topologies differ by owner. |
| No generic QR/SVD/minimum-norm helper API is created accidentally. | Complete | No helper is moved and generic helper creation is rejected. |
| All required validation passes. | Complete | Day 11 is documentation-only; documentation hygiene is the required validation. |
