# Sprint 104 Day 4 Descriptor Surface Boundary

## Purpose

Day 4 freezes the backend descriptor and runtime-selection boundary before any
source edits. It compares the Day 2 consumer audit with the Day 3 runtime
contract, decides which changes are public API, internal-only, test-support, or
documentation-only, and gives Day 5 a focused implementation sequence.

## Boundary Decision

Sprint 104 should not introduce a broad public vendor-backend API. The Day 5
implementation batch should stay inside the existing internal and benchmark
observability surfaces:

- preserve builtin fallback as the portable product truth;
- preserve existing public selector enums and struct layout;
- avoid adding public headers or ABI fields;
- improve selected/requested/fallback wording and diagnostics where existing
  benchmark/test/internal surfaces already expose them;
- keep graph/ND runtime controls out of the dense-backend descriptor batch;
- defer any broader runtime-control cleanup to Day 6/7.

## Surface Classification

| surface | current role | Day 4 classification | Day 5 disposition |
|---|---|---|---|
| `sparse_cholesky_opts_t::backend` | public path selector | public API, stable | no enum or layout change |
| `sparse_ldlt_opts_t::backend` and `used_csc_path` | public path selector and telemetry | public API, stable | no enum or layout change |
| `sparse_eigs_opts_t::backend` and `sparse_eigs_t::backend_used` | public algorithm selector/result telemetry | public API, stable | no enum or layout change |
| `chol_dense_kernels_t` | internal Cholesky CSC dense-kernel descriptor | internal-only | may receive docs/tests/benchmark wording only unless a tiny helper is needed |
| `chol_csc_supernodal_dense_kernels()` | internal descriptor accessor | internal-only/test-visible | keep existing fallback behavior |
| Cholesky dense-kernel test override | test-support | test-only | keep explicit `NULL` override error-path behavior |
| `ldlt_dense_factor_selected()` | internal LDLT dense-factor dispatch | internal-only | keep existing fallback behavior |
| `ldlt_dense_factor_backend_name()` | internal/benchmark diagnostic | diagnostic support | may be referenced by docs/benchmarks; no public claim expansion |
| `SPARSE_CHOL_DENSE_BACKEND` | compatibility/runtime env hook | maintainer/benchmark context | document request semantics and fallback |
| `SPARSE_LDLT_DENSE_BACKEND` | compatibility/runtime env hook | maintainer/benchmark context | document request semantics and fallback |
| benchmark CSV fields | measurement context | benchmark-facing diagnostics | wording/output alignment is allowed if scoped |
| OpenMP build/runtime settings | compile/runtime context | Day 6/8 input | no Day 5 source change unless descriptor work directly needs disclosure |
| graph/ND `SPARSE_ND_*` / `SPARSE_FM_*` env vars | reordering runtime controls | out of Day 5 descriptor scope | Day 6 audit only |

## Descriptor and Status Requirements

| requirement | decision |
|---|---|
| public API change | not allowed in Sprint 104 Day 5 unless a later explicit request reopens scope |
| ABI/layout change | not allowed |
| builtin fallback | must remain default and tested/documented as supported |
| optional backend request | best-effort request, not a hard requirement |
| unavailable optional provider | fallback to builtin remains valid behavior |
| selected backend observability | should be visible in benchmark/test diagnostics where existing surfaces already support it |
| fallback observability | required for benchmark interpretation; public programmatic API can remain deferred |
| invalid public enum | keep existing `SPARSE_ERR_BADARG` behavior |
| test-only backend contract error | preserve `SPARSE_ERR_BACKEND_CONTRACT` for deliberately broken internal descriptors |

## Candidate Changes

| candidate | classification | Day 4 decision | validation impact |
|---|---|---|---|
| Add public backend-provider enum | public API | reject/defer; too broad for Sprint 104 | none |
| Add public selected-dense-backend fields to factor structs | public ABI/API | reject/defer; ABI/layout risk not justified by current scope | none |
| Add docs describing `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND` as best-effort | documentation-only | allow | docs hygiene |
| Align benchmark wording around request/selected/fallback | documentation or benchmark-output touch | allow if limited to existing fields | docs hygiene or full code gate if benchmark `.c` changes |
| Add a tiny internal request-normalization helper | internal source | allow only if it removes duplicated behavior without changing semantics | full C quality gate |
| Add Cholesky benchmark fallback field parallel to LDLT | benchmark-output source | allow only if Day 5 keeps CSV compatibility risk explicit | full C quality gate plus focused benchmark build/run |
| Add tests for explicit builtin fallback env requests | test source | allow if implementation touches dense backend behavior | full C quality gate |
| Change fallback from silent to hard failure | behavior change | reject; conflicts with Day 3 contract |
| Move graph/ND env controls into dense-backend descriptor docs | scope expansion | reject/defer to Day 6 | none |

## Compatibility Checklist

Before any Day 5 source edit, confirm:

- existing zero-initialized Cholesky, LDLT, and eigensolver option structs keep
  their current behavior;
- existing enum values and numeric mappings remain unchanged;
- `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND` unset behavior
  remains builtin;
- unknown or unavailable optional dense-backend requests still fall back to
  builtin unless a test-only override deliberately forces contract failure;
- benchmark CSV changes, if any, either preserve existing column names or are
  documented as a bounded benchmark-surface update;
- tests clean up process-global environment variables;
- Windows remains builtin-only for dynamic dense backend probing;
- local benchmark timings disclose selected backend and thread context before
  being interpreted.

## Focused Validation Plan

| touched surface | focused validation |
|---|---|
| docs only | `git diff --check`; `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104` and touched public docs |
| `benchmarks/README.md` only | docs hygiene plus inspect affected section |
| `docs/maintainer_guide.md` only | docs hygiene plus inspect affected section |
| `benchmarks/bench_chol_csc.c` | `make build/bench_chol_csc`; run a small representative row if runtime is acceptable; then `make format && make lint && make test` because `.c` changed |
| `benchmarks/bench_refactor_csc.c` | `make build/bench_refactor_csc`; focused benchmark smoke if runtime is acceptable; then full C quality gate |
| `src/sparse_dense.c` or `src/sparse_ldlt_dense.c` | focused dense-backend tests (`build/test_chol_csc_supernodal`, `build/test_ldlt` where applicable); then full C quality gate |
| public headers | focused compile/test owner plus full C quality gate; require explicit compatibility note |
| tests touching env fallback | focused test binary and full C quality gate |

## Day 5 Implementation Sequence

Recommended order:

1. Prefer documentation and benchmark wording updates over source changes if
   they satisfy the Day 3 contract.
2. If source is needed, first adjust diagnostics/output that already exists;
   do not add public API.
3. Keep Cholesky and LDLT dense-backend semantics behavior-preserving.
4. Add focused tests only for behavior that changed or was previously
   untested.
5. Run focused validation first, then the full quality chain for any `.c` or
   `.h` touch.
6. Record exact validation and remaining non-claims in Day 5 notes.

## Deferred Items

| deferred item | reason |
|---|---|
| public vendor-backend selector API | would require product and ABI design beyond this sprint day |
| public selected-dense-backend result fields | ABI/layout risk; current benchmark/test diagnostics are enough for Sprint 104 |
| hard-fail behavior for unavailable optional providers | conflicts with portable builtin fallback contract |
| unified runtime-control registry for graph/ND/OpenMP/dense backends | larger architecture work; Day 6 can audit threading/runtime controls first |
| portable performance threshold based on optional acceleration | requires Day 8/9 sentinel design, baseline, and machine-class assumptions |

## Completion Check

| criterion | status |
|---|---|
| descriptor fields and diagnostics identified | complete |
| public API, internal-only, test-support, and docs-only decisions made | complete |
| compatibility requirements recorded | complete |
| source changes scoped before implementation starts | complete |
| builtin fallback remains first-class | complete |
| focused validation plan written | complete |
