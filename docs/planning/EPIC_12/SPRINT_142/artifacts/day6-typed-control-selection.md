# Day 6 Typed-Control Selection

## Purpose

Day 6 selects which runtime/backend controls should be promoted to typed
public controls in Sprint 142, and which should be explicitly deferred as
compatibility, diagnostic, maintainer-only, test-only, or future-sprint
surfaces. The selection uses the Day 2 inventory, Day 3 dispatch audit, and
Day 4-5 precedence contract work.

The main decision is conservative: Sprint 142 should not add a new public ABI
surface for dense-helper, SVD, FM, OpenMP, package, or sentinel controls. The
highest-value work for Day 7 is to make the current boundaries explicit in
source-controlled artifacts and focused validation rather than silently
promoting environment variables into public API.

## Scoring Model

| Score | Meaning |
| --- | --- |
| 5 | Strongly favorable. |
| 3 | Mixed or moderate. |
| 1 | Weak or risky. |

Higher user value and validation readiness increase priority. Higher
implementation risk, documentation burden, and claim risk decrease priority.

## Candidate Matrix

| Candidate | User value | Implementation risk | Validation readiness | Documentation burden | Claim risk | Day 6 decision |
| --- | --- | --- | --- | --- | --- | --- |
| Cholesky dense-helper typed selector replacing or shadowing `SPARSE_CHOL_DENSE_BACKEND` | 2 | 2 | 3 | 2 | 2 | Defer. Useful mostly for local backend experiments and report context; public API would imply optional BLAS/Accelerate support expectations Sprint 142 should not create. |
| LDLT dense-helper typed selector replacing or shadowing `SPARSE_LDLT_DENSE_BACKEND` | 2 | 2 | 2 | 2 | 2 | Defer. Current report fields expose request/selected/fallback, but focused invalid-env parity is weaker than Cholesky and public promotion would widen backend support claims. |
| Public thread-count or runtime OpenMP policy API | 2 | 1 | 2 | 1 | 1 | Defer. `SPARSE_OPENMP` remains compile-time and `OMP_NUM_THREADS` remains caller/runtime-owned; a library-owned thread policy needs separate design. |
| Typed eigensolver reorth OpenMP threshold | 1 | 2 | 2 | 2 | 2 | Defer. It is a compile-time tuning constant, not a user workflow; promoting it would add runtime policy surface with limited value. |
| Typed SVD low-rank outer-product selector replacing `SPARSE_SVD_LOWRANK_OUTER` | 2 | 3 | 4 | 3 | 2 | Defer. Existing tests cover behavior, but the selector is advisory memory/performance experimentation and not part of Sprint 142's backend-governance priority. |
| Typed FM strategy and pass controls | 2 | 1 | 3 | 1 | 1 | Defer. These are graph/reorder tuning internals with high documentation and claim risk. |
| Typed ND/profile/debug controls | 1 | 1 | 2 | 1 | 1 | Defer. Profile/debug controls are maintainer diagnostics, not user-facing runtime policy. |
| Dispatch-only sentinel row for Cholesky/LDLT top-level selection | 3 | 4 | 4 | 4 | 4 | Select for sentinel design, not public typed promotion. Fits Days 8-9 better than Day 7 API work. |
| Eigensolver AUTO backend snapshot row | 3 | 4 | 4 | 4 | 4 | Select for sentinel design, not public typed promotion. Uses existing `backend_used` telemetry and avoids ABI changes. |
| Shift-invert LDLT route snapshot row | 3 | 4 | 4 | 4 | 4 | Select for sentinel design, not public typed promotion. Uses existing `used_csc_path_ldlt` telemetry. |
| Explicit maintainer-only deferral ledger | 5 | 5 | 5 | 5 | 5 | Select. It closes the governance ambiguity without changing ABI or over-claiming support. |
| LDLT dense helper focused invalid-env parity test | 3 | 4 | 3 | 4 | 4 | Select as optional Day 7 validation-only improvement if code/test budget remains. It does not require public API promotion. |

## Selected Day 7 Batch

| Batch item | Implementation shape | Files likely touched | Validation |
| --- | --- | --- | --- |
| Maintainer-only runtime control deferral ledger | Add a source-controlled Sprint 142 artifact that classifies dense-helper env selectors, SVD low-rank env selector, FM/debug/profile env vars, OpenMP runtime context, package/build controls, and sentinel/report controls as public typed, compatibility, maintainer-only, build-time, test-only, or future-sprint. | `docs/planning/EPIC_12/SPRINT_142/artifacts/day7-typed-control-batch.md`, `WORKING_NOTES.md` | Documentation hygiene. |
| Public typed-control non-expansion statement | Record that no new public ABI/API fields are added in Sprint 142 Day 7 unless a specific failing validation forces a narrow test-only change. | Day 7 artifact and later docs handoff | Documentation hygiene. |
| Optional LDLT dense helper invalid-env validation | If Day 7 has implementation room, add or identify focused validation that invalid `SPARSE_LDLT_DENSE_BACKEND` falls back to builtin without promoting it. Prefer a test-only assertion if a clean existing helper is available; otherwise document as a deferred proof owner. | Potentially `tests/test_ldlt*.c` only if clean; otherwise artifact-only | If C test changes: focused test plus `make format && make lint && make test`. |
| Sentinel candidates preserved for Days 8-9 | Carry dispatch-only Cholesky/LDLT, eigensolver AUTO, and shift-invert LDLT route snapshots forward as sentinel design inputs. | Day 7 artifact and Day 8 design | Documentation hygiene on Day 7; script/test validation on Days 8-9 if implemented. |

The selected batch is intentionally small enough to finish and validate in
Day 7. It avoids adding public option fields, enum values, package metadata,
or ABI changes before Sprint 143 package/ABI ownership.

## Explicit Deferral List

| Control | Classification | Deferral reason | Future owner |
| --- | --- | --- | --- |
| `SPARSE_CHOL_DENSE_BACKEND` | Maintainer/compatibility env selector | Useful for local dense-helper comparisons, but public typed promotion would imply optional BLAS/Accelerate support and platform semantics not earned here. | Day 7 deferral ledger; Sprint 143+ only if package/backend product scope changes. |
| `SPARSE_LDLT_DENSE_BACKEND` | Maintainer/compatibility env selector | Same support-risk profile as Cholesky dense helper, with less focused invalid-env parity proof today. | Day 7 validation-only consideration; Sprint 143+ for product promotion. |
| `SPARSE_SVD_LOWRANK_OUTER` | Maintainer/runtime experiment env selector | Existing tests validate the path, but it is an advisory memory/wall tradeoff and outside the selected backend-governance API surface. | Day 7 deferral ledger; possible future SVD productization sprint. |
| `SPARSE_FM_FINEST_STRATEGY`, `SPARSE_FM_ENSEMBLE_STRATEGIES`, pass-count and schedule vars | Maintainer graph/reorder tuning env vars | Internal strategy tuning has high documentation burden and can affect fill/performance claims. | Keep in maintainer guide; no public API in Sprint 142. |
| `SPARSE_FM_*_DEBUG`, `SPARSE_HCC_DEBUG`, `SPARSE_ND_PROFILE`, `SPARSE_QG_PROFILE` | Maintainer diagnostics | Diagnostic stderr/profile controls are not user runtime policy. | Keep maintainer-only. |
| `SPARSE_OPENMP` | Build-time feature flag | Already documented as compile-time; public runtime API would be a separate threading design. | Build/docs owner; no Sprint 142 typed promotion. |
| `OMP_NUM_THREADS` | External runtime context | Owned by OpenMP runtime and caller process, not library API. | Report context only. |
| `SPARSE_MUTEX` | Build-time safety flag | Matrix mutation option, not backend/runtime policy; package/link effects belong elsewhere. | Build/docs owner; Sprint 143 if packaging surface changes. |
| `SPARSE_CSC_THRESHOLD`, eigensolver AUTO thresholds, `SPARSE_EIGS_OMP_REORTH_MIN_N` | Compile-time tuning constants | Already shape deterministic AUTO/default behavior; runtime typed promotion is not needed for Sprint 142. | Keep header-documented compile-time constants. |
| `SPARSE_TEST_SLOW`, `SPARSE_TEST_EXPERIMENTAL`, `SPARSE_TEST_LARGE`, `RUN_BENCH` | Test/benchmark workflow controls | Developer workflow only. | Test docs only. |
| Package/pkg-config/CMake link behavior | Package/build surface | Runtime governance must not imply package, ABI, shared-library, or package-manager changes. | Sprint 143 handoff if affected. |

## Test And Documentation Plan

| Need | Owner | Day |
| --- | --- | --- |
| Preserve typed backend precedence coverage. | Existing focused tests from Day 5. | Day 11-12 validation if touched again. |
| Preserve analysis typed-vs-env precedence coverage. | `tests/test_reorder_nd.c`. | Day 11-12 validation if touched again. |
| Make maintainer-only deferrals explicit. | Day 7 typed-control batch artifact and working notes. | Day 7. |
| Decide whether LDLT dense invalid-env fallback needs a focused test. | Day 7 implementation pass. | Day 7. |
| Design sentinel rows for dispatch observability. | Day 8 sentinel design. | Day 8. |
| Implement selected sentinel rows without portable timing claims. | Day 9 sentinel implementation. | Day 9. |
| Update public/maintainer docs to match earned behavior and deferrals. | Day 10 docs/examples. | Day 10. |
| Re-run focused tests, report-index checks, and full quality gate if C/header changes occur. | Day 11-12 validation. | Days 11-12. |

## Day 6 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected batch is small enough to finish and validate. | Complete | Selected Day 7 batch avoids ABI/API expansion and focuses on explicit deferral plus optional test-only parity. |
| Deferred controls are not silently dropped. | Complete | Explicit deferral list records every candidate from Days 2-5. |
| Public API and maintainer-only boundaries are explicit. | Complete | Candidate matrix, selected batch, and deferral classifications. |
