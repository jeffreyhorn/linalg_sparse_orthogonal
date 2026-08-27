# Sprint 184 Day 2: Declaration Baseline Capture

**Sprint:** 184 - Public Header Coherence Batch 3
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_184/`
**Status:** Complete

## Purpose

Day 2 captures the current public declaration surface for the QR, SVD, and
LDLT candidate families before Sprint 184 selects one family or edits public
headers. It also records preservation rules and the initial guard strategy for
future comment cleanup, declaration organization, docs alignment, and
validation.

## Baseline Capture Command

The Day 2 baseline uses a line-numbered declaration-order capture for the three
candidate headers:

```sh
rg -n "^(typedef (struct|enum)|} sparse_|[a-zA-Z_][a-zA-Z0-9_ *]+\\s+sparse_(qr|svd|pinv|cond|ldlt)[a-zA-Z0-9_]*\\()" include/sparse_qr.h include/sparse_svd.h include/sparse_ldlt.h
```

This capture is intentionally simple. It records named type closures and public
function declaration starts in current order. It is enough to detect accidental
type/function additions, removals, renames, and order movement during
comment-cleanup work. If Sprint 184 intentionally reorganizes declarations,
the before/after capture must be recorded in that later artifact.

## Baseline Checksums

| Scope | SHA-256 |
| --- | --- |
| `include/sparse_qr.h` declaration-order baseline | `77a53e6bc780d79907bad9a040310bb0d63f93dce3fdd3beb0ed8cfdfd0279bc` |
| `include/sparse_svd.h` declaration-order baseline | `51d334c7cc7681a3b53f0af3e5a3d0bdf4d890e0734fa4da1b424c54604c3025` |
| `include/sparse_ldlt.h` declaration-order baseline | `b99ed791daeb2e9a6d411cb0bccad486a897aa42b9c130f0f45eb58d0cf547a7` |
| Combined QR/SVD/LDLT declaration-order baseline | `765b4711e1a62006566b1a0a7f6187401b958753fbae4cb902f540c6e98ed45e` |

## Candidate Declaration Inventory

### QR: `include/sparse_qr.h`

| Line | Declaration surface |
| ---: | --- |
| 24 | `typedef struct` for `sparse_qr_opts_t` |
| 51 | `sparse_qr_opts_t` type closure |
| 63 | `typedef struct` for `sparse_qr_t` |
| 75 | `sparse_qr_t` type closure |
| 90 | `sparse_qr_factor(...)` |
| 101 | `sparse_qr_factor_opts(...)` |
| 116 | `sparse_qr_apply_q(...)` |
| 133 | `sparse_qr_form_q(...)` |
| 163 | `sparse_qr_solve(...)` |
| 185 | `sparse_qr_refine(...)` |
| 211 | `sparse_qr_rank(...)` |
| 230 | `sparse_qr_nullspace(...)` |
| 238 | `sparse_qr_free(...)` |
| 258 | `sparse_qr_diag_r(...)` |
| 263 | `typedef struct` for `sparse_qr_rank_info_t` |
| 270 | `sparse_qr_rank_info_t` type closure |
| 299 | `sparse_qr_rank_info(...)` |
| 314 | `sparse_qr_condest(...)` |
| 340 | `sparse_qr_solve_minnorm(...)` |
| 369 | `sparse_qr_refine_minnorm(...)` |

Current grouping:

- file-level workflow contract and `sparse_matrix.h` include;
- options struct;
- factor object struct;
- factorization/apply/solve/rank/nullspace/free declarations;
- rank-revealing diagnostics section;
- minimum-norm solve/refinement declarations.

Contract categories already present:

- lifecycle/free: `sparse_qr_free()`, zeroed-struct safety, overwrite warning;
- ownership/output: factor output, dense Q, solution vectors, residual output,
  nullspace basis caller allocation;
- error codes: `SPARSE_ERR_NULL`, `SPARSE_ERR_BADARG`, `SPARSE_ERR_ALLOC`,
  `SPARSE_ERR_CANCELLED`, shape checks;
- tolerance: rank thresholds and manual R-diagonal selection;
- workspace: sparse mode comment and dense Q caller allocation;
- options/results: `sparse_qr_opts_t`, `sparse_qr_t`,
  `sparse_qr_rank_info_t`;
- cancellation: progress callback in `sparse_qr_opts_t`.

Docs/example mismatch risks:

- QR has multiple public examples (`example_least_squares.c`,
  `example_minnorm.c`, `example_colamd.c`) and solver-selection evidence
  boundary wording; stale minimum-norm, rank, or broad QR parity language would
  be high-impact.
- QR public docs discuss selected oracle/comparison freshness, so header
  cleanup must not widen into raw QR basis parity, broad external-library
  parity, or global rank-threshold claims.

### SVD / Partial SVD: `include/sparse_svd.h`

| Line | Declaration surface |
| ---: | --- |
| 21 | `typedef struct` for `sparse_svd_opts_t` |
| 31 | `sparse_svd_opts_t` type closure |
| 39 | `typedef struct` for `sparse_svd_t` |
| 49 | `sparse_svd_t` type closure |
| 75 | `sparse_svd_compute(...)` |
| 83 | `sparse_svd_free(...)` |
| 96 | `sparse_svd_extract_uv(...)` |
| 133 | `sparse_svd_partial(...)` |
| 156 | `sparse_svd_rank(...)` |
| 174 | `sparse_pinv(...)` |
| 192 | `sparse_svd_lowrank(...)` |
| 225 | `sparse_svd_lowrank_sparse(...)` |
| 241 | `sparse_cond(...)` |

Current grouping:

- file-level SVD workflow contract and includes;
- options struct;
- result struct;
- full SVD/free/extract/partial SVD declarations;
- `SVD applications` section for rank, pseudoinverse, low-rank, sparse
  low-rank, and condition number.

Contract categories already present:

- lifecycle/free: `sparse_svd_free()` and result ownership;
- ownership/output: caller-owned dense arrays, sparse output returned as
  `SparseMatrix *`, `NULL` on failure;
- error codes: `SPARSE_ERR_NULL`, `SPARSE_ERR_BADARG`, `SPARSE_ERR_ALLOC`,
  `SPARSE_ERR_NOT_CONVERGED`;
- tolerance: SVD convergence tolerance, rank tolerance, pseudoinverse
  tolerance, sparse low-rank drop tolerance;
- workspace: dense temporary accumulator note for sparse low-rank output;
- options/results: `sparse_svd_opts_t`, `sparse_svd_t`;
- cancellation: no public callback/cancellation contract in this header.

Docs/example mismatch risks:

- SVD docs contain fixture-local partial-SVD evidence boundaries. Header
  cleanup must not imply broad partial-SVD correctness, raw singular-vector
  identity, sparse-output optimality, external-library parity, performance, or
  partial-result guarantees.
- The sparse low-rank alternative path has important non-bit-level-equivalence
  wording that must stay scoped.

### LDLT: `include/sparse_ldlt.h`

| Line | Declaration surface |
| ---: | --- |
| 65 | `typedef struct` for `sparse_ldlt_t` |
| 85 | `sparse_ldlt_t` type closure |
| 107 | `typedef enum` for `sparse_ldlt_backend_t` |
| 111 | `sparse_ldlt_backend_t` enum closure |
| 131 | `typedef struct` for `sparse_ldlt_opts_t` |
| 166 | `sparse_ldlt_opts_t` type closure |
| 201 | `sparse_ldlt_factor(...)` |
| 219 | `sparse_ldlt_factor_opts(...)` |
| 244 | `sparse_ldlt_solve(...)` |
| 251 | `sparse_ldlt_free(...)` |
| 273 | `sparse_ldlt_inertia(...)` |
| 294 | `sparse_ldlt_refine(...)` |
| 313 | `sparse_ldlt_condest(...)` |

Current grouping:

- file-level factor/solve usage pattern and `sparse_matrix.h` include;
- factor object struct;
- backend enum;
- options struct;
- factor/factor_opts/solve/free/inertia/refine/condest declarations.

Contract categories already present:

- lifecycle/free: factor object overwrite warning, zeroed-struct safety,
  `sparse_ldlt_free()`;
- ownership/output: owned factor internals, telemetry pointer, solution
  vector, inertia outputs, condition-estimate output;
- error codes: `SPARSE_ERR_NULL`, `SPARSE_ERR_SHAPE`, `SPARSE_ERR_NOT_SPD`,
  `SPARSE_ERR_BADARG`, `SPARSE_ERR_SINGULAR`, `SPARSE_ERR_ALLOC`,
  `SPARSE_ERR_CANCELLED`;
- tolerance: pivot/drop tolerance, solve/refine/condest reuse, CSC backend
  tolerance caveat, refinement tolerance;
- workspace: temporary solve/refine/condition-estimator workspace allocation;
- options/results: `sparse_ldlt_t`, `sparse_ldlt_backend_t`,
  `sparse_ldlt_opts_t`, backend telemetry;
- cancellation: linked-list progress callback cancellation, CSC no-progress
  caveat.

Docs/example mismatch risks:

- LDLT docs touch backend dispatch, KKT-style systems, inertia, tolerance,
  thread safety, and backend telemetry. Cleanup must not imply backend
  superiority, portable performance, broad external comparison parity, or
  shared-library/ABI compatibility.
- The header currently includes a substantial usage example. If LDLT is
  selected, Day 3 should decide whether that example stays in the header or is
  shortened in favor of `examples/example_ldlt.c` and tutorial/cookbook docs.

## Existing Guard Surface

| Guard or target | Current role | Sprint 184 relevance |
| --- | --- | --- |
| `make docs-check` | Runs Doxygen plus API docs coverage. | Required if public header comments or API docs inputs change and Doxygen is available. |
| `make api-docs-coverage` | Runs `scripts/check_api_docs_coverage.py`. | Confirms checked-in public headers remain represented in API reference coverage expectations. |
| `make api-docs-local-only` | Runs `scripts/check_api_docs_local_only.sh`. | Preserves generated API HTML local-only status. |
| `scripts/check_api_docs_coverage.py` | Public API documentation coverage guard. | Existing broad docs coverage guard, not family-specific declaration protection. |
| `scripts/check_api_docs_local_only.sh` | Generated API HTML local-only guard. | Protects Sprint 179 publication decision while header docs are cleaned. |
| `scripts/check_lu_header_docs_guard.sh` | LU-specific section/declaration/docs drift guard from Sprint 172. | Best local precedent for a selected-family QR/SVD/LDLT guard. |
| `make format && make lint && make test` | Full C quality gate. | Required after any `.c` or `.h` change. |

There is no maintained generic declaration checksum helper today. Sprint 184
should either add a selected-family guard modeled on
`scripts/check_lu_header_docs_guard.sh` or document why the declaration-order
baseline plus full C/docs gates are sufficient for the selected family.

## Declaration-Preservation Rules

| Surface | Rule |
| --- | --- |
| Function signatures | Must remain byte-for-byte equivalent unless a later artifact explicitly rejects declaration preservation and records full validation. |
| Struct fields | No additions, removals, renames, type changes, or order changes during comment cleanup. |
| Enum values | No additions, removals, renames, value changes, or order changes during comment cleanup. |
| Typedef names | No renames or alias changes. |
| Macro/include guards | No guard, include, or installed-header-name changes unless explicitly selected and validated. |
| Comments | May be normalized for contract clarity, generated-doc input quality, and unsupported-claim boundaries. |
| Section headings | May change if they do not move declarations; declaration reordering waits for an explicit organization artifact and before/after baseline. |
| Examples/docs | May align with the selected family only; must not widen package, ABI, platform, performance, external parity, or state-of-the-art claims. |

## Candidate Narrowing

| Rank | Candidate | Day 2 rationale |
| ---: | --- | --- |
| 1 | QR | Largest candidate header, broadest public function surface, multiple examples, strong tutorial/solver-selection/README visibility, and sensitive rank/nullspace/minimum-norm claim boundaries. Best fit for a high-impact public-header coherence batch if Day 3 confirms declaration cleanup is bounded. |
| 2 | SVD / partial SVD | Strong docs and evidence sensitivity, especially around convergence, singular vectors, sparse low-rank output, and partial-SVD non-claims. Slightly smaller public declaration surface than QR. |
| 3 | LDLT | Valuable direct-solver/backend cleanup candidate, but current header already carries detailed lifecycle/backend wording and recent Sprint 183 work deferred LDLT comparison expansion. Still viable if Day 3 decides backend/tolerance wording has higher current risk than QR. |

Day 2 narrows QR to the provisional front-runner but does not make the final
selection. Day 3 should confirm or reject QR after comparing cleanup value,
guard feasibility, and docs alignment cost.

## Day 3 Handoff

Day 3 should:

- decide the selected family explicitly;
- record rejected alternatives and non-goals;
- capture the selected family declaration baseline in the Day 3 decision
  record;
- map lifecycle, ownership, error-code, tolerance, workspace, option/result,
  and cancellation wording for the selected family;
- define the first cleanup checklist without editing declarations.

## Validation

Day 2 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Baseline data can detect unintended declaration drift. | Complete | Declaration-order capture command and candidate checksums are recorded. |
| The selected guard strategy is clear enough to use before edits. | Complete | Use declaration-order baselines before edits, full C gate after header edits, docs guards for documentation changes, and evaluate a selected-family LU-style guard. |
| Remaining family-selection uncertainty is explicit and bounded. | Complete | QR is the provisional front-runner; SVD and LDLT remain documented alternatives for Day 3. |
