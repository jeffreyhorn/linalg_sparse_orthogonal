# Sprint 184 Day 3: Family Selection Decision and Contract Map

**Sprint:** 184 - Public Header Coherence Batch 3
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_184/`
**Status:** Complete

## Purpose

Day 3 selects the Sprint 184 public header family, records rejected
alternatives and non-goals, maps the selected family's contract wording
surface, and prepares the cleanup checklist for Days 4-5. No public headers
are edited on Day 3.

## Selection Decision

Sprint 184 selects:

| Field | Decision |
| --- | --- |
| Selected family | QR |
| Selected public header | `include/sparse_qr.h` |
| Day 2 declaration baseline | `77a53e6bc780d79907bad9a040310bb0d63f93dce3fdd3beb0ed8cfdfd0279bc` |
| Primary docs surfaces | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/api_reference.md`, `examples/README.md` |
| Primary examples | `examples/example_least_squares.c`, `examples/example_minnorm.c`, `examples/example_colamd.c` |
| Primary tests/evidence surfaces | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_qr_corpus.c`, `tests/qr_external_dense_reference.py`, selected oracle/comparison freshness gates |

## Selection Rationale

| Criterion | QR assessment |
| --- | --- |
| Adoption risk | High. QR is the public route for rectangular, least-squares, rank-sensitive, nullspace, and minimum-norm workflows. |
| Public API visibility | High. The header exposes options, factor objects, factorization, apply/form-Q, solve/refine, rank/nullspace, diagnostics, condition estimate, minimum-norm solve, and free helpers. |
| Documentation impact | High. QR appears in README, tutorial, cookbook, solver-selection, API reference, examples README, and three runnable examples. |
| Claim sensitivity | High. QR docs carry fixture-local corpus and comparison evidence boundaries that must not widen into broad QR correctness, raw basis parity, global rank-threshold policy, external-library parity, platform/package/ABI, performance, or state-of-the-art claims. |
| Guard feasibility | Good. Day 2 declaration-order baseline is small enough to review, and the existing LU header/docs guard gives a family-specific guard precedent. |
| Cleanup fit | Good. Most anticipated changes are comment-only contract cleanup and section-heading refinement. Declaration reordering can be deferred until guardrails are stronger. |

## Rejected Alternatives

| Candidate | Decision | Reason |
| --- | --- | --- |
| SVD / partial SVD | Defer | SVD is a strong future candidate, but its header is smaller than QR and its most sensitive partial-SVD evidence wording is already explicit in solver-selection/docs. Keep it for a focused future pass on convergence, vectors, low-rank output, and partial-result non-claims. |
| LDLT | Defer | LDLT is a viable direct-solver/backend candidate, but the current header already has detailed lifecycle, tolerance, backend, telemetry, and cancellation wording. Its backend/KKT surface should be cleaned in a direct-solver or backend-focused pass rather than displacing QR. |

## Explicit Non-Goals

Sprint 184 QR cleanup will not:

- change QR public function signatures, typedef names, struct fields, field
  order, enum values, macros, include guards, or installed header names;
- broaden QR correctness, raw Q/R basis parity, global rank-threshold policy,
  broad rank-deficient solve, broad minimum-norm behavior, or external-library
  parity claims;
- change implementation behavior in `src/sparse_qr.c` or
  `src/sparse_qr_householder.c`;
- add new QR tests or corpus rows unless a later header edit exposes a real
  mismatch that cannot be documented safely;
- claim package-manager support, shared-library support, dynamic ABI stability,
  runtime-loader behavior, broad platform parity, portable performance, or
  state-of-the-art sparse linear algebra coverage;
- publish or stage generated API HTML.

## Selected QR Declaration Baseline

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

## QR Contract Map

| Section or declaration | Lifecycle | Ownership/output | Errors | Tolerance | Workspace | Options/results | Cancellation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| File-level contract | States QR owns API contracts and points runnable workflows to examples/docs. | Mentions output-buffer sizes and cleanup at a high level. | Not detailed. | Mentions rank/residual diagnostics. | Not detailed. | Not detailed. | Not detailed. |
| `sparse_qr_opts_t` | Options are used at factorization time. | `progress_user` is passed through unchanged. | Cancellation returns `SPARSE_ERR_CANCELLED`. | No rank tolerance. | `sparse_mode` describes O(m) per-column working memory. | `reorder`, `economy`, `sparse_mode`, callback fields. | Progress callback can cancel at Householder column iterations. |
| `sparse_qr_t` | Factor functions overwrite without freeing prior contents; `sparse_qr_free()` required before reuse. | Owns `R`, `betas`, `v_vectors`, `col_perm`; caller owns struct storage. | Not detailed. | Stores rank set during factorization. | Owns factor storage. | Factor object/result. | Not applicable. |
| `sparse_qr_factor(...)` | Creates factor object from original matrix. | `A` not modified; `qr` must be freed. | Lists NULL, non-identity permutation, allocation errors. | Internal rank threshold only. | Factor allocation. | Default options path. | No callback. |
| `sparse_qr_factor_opts(...)` | Same as factor with options. | `A` not modified; `qr` output. | Currently shorter than `sparse_qr_factor()` and should be aligned. | Options do not expose rank tolerance. | Reorder/sparse mode may change workspace. | `opts == NULL` defaults. | Uses callback fields when provided. |
| `sparse_qr_apply_q(...)` | Uses existing factorization. | `x`/`y` length m; `y` may alias `x`. | Only generic success listed today. | Not applicable. | No explicit temporary workspace contract. | Uses factor object. | Not applicable. |
| `sparse_qr_form_q(...)` | Uses existing factorization. | Caller allocates dense Q as m*m or m*k. | Only generic success listed today. | Not applicable. | Explicit dense output allocation. | Economy/full behavior from factor object. | Not applicable. |
| `sparse_qr_solve(...)` | Uses existing factorization. | Caller provides `b`, `x`, optional residual. | Only generic success listed today. | Rank-deficient components zeroed according to factor rank. | No explicit temporary workspace contract. | Solve result through caller buffers. | Not applicable. |
| `sparse_qr_refine(...)` | Refines existing solution using factorization and original matrix. | `x` modified in place; optional residual output. | Lists NULL, shape, allocation errors. | `max_refine == 0` computes residual. | Temporary residual/correction workspace implied. | Uses existing factor object. | Not applicable. |
| `sparse_qr_rank(...)` | Read-only diagnostic on factor object. | Return value only. | No error-code channel. | Defines default and caller tolerance behavior. | None. | Uses factor object rank data. | Not applicable. |
| `sparse_qr_nullspace(...)` | Uses factorization and rank threshold. | Caller allocates `n * (n - rank)` dense scalars; outputs dimension. | Only generic success listed today. | Same as `sparse_qr_rank()`. | Dense basis output. | Uses factor object. | Not applicable. |
| `sparse_qr_free(...)` | Frees QR factorization data; zeroed struct safe. | Caller retains struct storage. | No return value. | Not applicable. | Releases owned factor storage. | Clears factor state. | Not applicable. |
| `sparse_qr_diag_r(...)` | Read-only diagnostic on factor object. | Caller allocates `min(m,n)` entries. | Lists NULL and unfactored errors. | Enables manual threshold selection. | None beyond output buffer. | Diagnostic output. | Not applicable. |
| `sparse_qr_rank_info_t` | Passive result struct. | Caller owns struct storage. | Not applicable. | Stores threshold-derived rank diagnostics. | None. | Result struct. | Not applicable. |
| `sparse_qr_rank_info(...)` | Read-only diagnostic on factor object. | Fills caller-provided `info`. | Lists NULL and unfactored errors. | Defines default and problem-specific tolerance guidance. | None. | Rank diagnostics result. | Not applicable. |
| `sparse_qr_condest(...)` | Read-only diagnostic on factor object. | Return value only. | Negative sentinel for NULL/unfactored/rank-zero. | Uses rank-determined R diagonal. | None. | Diagnostic scalar. | Not applicable. |
| `sparse_qr_solve_minnorm(...)` | One-shot QR(A^T) path, not existing factor reuse. | `A` not modified; caller provides `b`, `x`, optional opts. | Lists NULL, badarg, allocation errors. | Near-zero R diagonals zero components, not error. | Builds A^T and factorization internally. | Optional QR opts. | Callback behavior is inherited through opts but not stated. |
| `sparse_qr_refine_minnorm(...)` | Repeated one-shot minimum-norm corrections. | `x` modified in place; optional residual output. | Lists NULL, badarg, allocation errors. | Stops when residual stops decreasing or iteration budget reached. | Rebuilds A^T and QR each iteration. | Optional QR opts. | Callback behavior is inherited through opts but not stated. |

## Inconsistent Or Missing Contract Wording

| Topic | Current issue | Cleanup direction |
| --- | --- | --- |
| `sparse_qr_factor_opts()` errors | Shorter than `sparse_qr_factor()` and does not list NULL/allocation/cancellation behavior explicitly. | Align error-code wording with `sparse_qr_factor()` and callback cancellation semantics without changing behavior. |
| Output buffer sizes | Most APIs state lengths, but not all distinguish caller-owned output from optional output consistently. | Normalize caller-owned buffer wording for `apply_q`, `form_q`, `solve`, `nullspace`, diagnostics, and minimum-norm paths. |
| `opts == NULL` defaults | Defaults are mentioned in some places but not consistently across option-consuming APIs. | Use one clear phrase for NULL options/default behavior. |
| Rank tolerance | `sparse_qr_rank()` and `sparse_qr_rank_info()` both explain defaults, while `sparse_qr_solve()` references internal rank behavior. | Keep both public tolerance contracts but avoid implying a global rank-threshold policy. |
| Workspace wording | `sparse_mode` and `form_q` mention memory; solve/refine/minnorm paths have less explicit temporary allocation wording. | Clarify allocation and workspace failures only where public return codes already support them. |
| Cancellation | Factor options describe cancellation; minnorm routines accept opts but do not state how callback cancellation propagates. | Clarify callback scope only if implementation confirms propagation; otherwise avoid adding a promise. |
| Evidence boundaries | Docs carry selected QR corpus/comparison boundaries; header currently points to docs rather than repeating them. | Keep broad evidence/non-claim wording in docs. Header comments should stay API-local and not become evidence claims. |

## Documentation And Example Audit

| Surface | Day 3 finding | Likely Sprint 184 action |
| --- | --- | --- |
| `README.md` | QR public API list and evidence boundary are present. | Recheck after header cleanup; update only if terminology changes. |
| `docs/api_reference.md` | Summary row points at `sparse_qr.h`. | Likely no Day 4-5 edit unless coverage guard requires a clearer summary. |
| `docs/tutorial.md` | QR walkthrough states identity-permutation precondition and minnorm distinction. | Recheck after header cleanup; likely align wording only. |
| `docs/cookbook.md` | QR routing and fixture-local confidence wording are scoped. | Recheck claim wording after header cleanup. |
| `docs/solver_selection.md` | QR evidence boundary is explicit and narrow. | Preserve as the authority for evidence boundaries. |
| `examples/README.md` | QR examples are documented and separate teaching examples from corpus proof. | Recheck after header cleanup. |
| `examples/example_least_squares.c` | Demonstrates overdetermined QR, rank, residual, and cleanup. | Likely no code edit on Day 4-5. |
| `examples/example_minnorm.c` | Demonstrates one-shot minimum-norm and refinement. | Recheck whether free/cleanup wording should mention no factor object is retained. |
| `examples/example_colamd.c` | Demonstrates QR options and rank info after COLAMD. | Recheck option/default wording after header cleanup. |

## Initial Cleanup Checklist

### File-Level And Options

- Clarify that `include/sparse_qr.h` owns API-local contracts while evidence
  and workflow interpretation live in docs.
- Normalize `sparse_qr_opts_t` wording for defaults, `sparse_mode` workspace,
  and callback cancellation.
- Avoid claiming COLAMD/AMD/RCM/ND performance or superiority.

### Factor Object And Lifecycle

- Preserve `sparse_qr_t` layout exactly.
- Clarify factor-object ownership, overwrite behavior, zeroed-struct safety,
  and `sparse_qr_free()` expectations.
- Ensure factorization error paths are described without promising unstated
  cleanup behavior beyond current implementation.

### Factorization And Options APIs

- Align `sparse_qr_factor()` and `sparse_qr_factor_opts()` error-code and
  identity-permutation wording.
- Keep declarations unchanged.
- Preserve the Day 2 QR declaration checksum unless a later organization
  artifact records an intentional change.

### Apply/Form-Q And Solve APIs

- Normalize caller-owned input/output buffer language and aliasing notes.
- Keep dense Q output sizing explicit for full and economy modes.
- Clarify `sparse_qr_solve()` underdetermined behavior versus
  `sparse_qr_solve_minnorm()`.

### Rank, Nullspace, And Diagnostics

- Keep tolerance defaults consistent across `sparse_qr_rank()` and
  `sparse_qr_rank_info()`.
- Avoid global rank-threshold policy claims.
- Clarify nullspace basis output sizing and original-column-order wording.
- Keep `sparse_qr_condest()` as a rough diagnostic estimate, not a broad
  conditioning guarantee.

### Minimum-Norm APIs

- Keep one-shot minimum-norm solve/refine behavior distinct from reusing an
  existing `sparse_qr_t`.
- Clarify temporary factorization/workspace behavior where supported by
  existing error returns.
- Do not widen broad minimum-norm or external-library parity claims.

### Docs And Examples

- Recheck README, tutorial, cookbook, solver-selection, API reference, examples
  README, and QR examples after header cleanup.
- Preserve solver-selection as the main QR evidence-boundary authority.
- Avoid staging generated API HTML.

## Day 4 Handoff

Day 4 should begin comment-only cleanup in `include/sparse_qr.h` for lifecycle,
ownership/output, and error-code wording. Before editing, rerun or copy the
Day 2 QR declaration-order capture. After editing, compare the capture and run
the required full C gate because a public `.h` file will change.

## Validation

Day 3 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| One public header family is selected for the sprint. | Complete | QR and `include/sparse_qr.h` are selected above. |
| Cleanup areas are mapped without changing declarations. | Complete | Contract map and cleanup checklist are recorded; no public header was edited. |
| The checklist covers every contract wording category from item 184.2. | Complete | Lifecycle, ownership, error-code, tolerance, workspace, option/result, and cancellation are all covered in the QR contract map. |
