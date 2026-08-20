# Sprint 172 Day 3: Header Family Selection Decision

## Purpose

Day 3 selects exactly one public header family for Sprint 172 cleanup and
defines the support, non-claim, and validation boundaries before any public
header edit.

## Selection Decision

Sprint 172 selects the general LU public header family:

```text
include/sparse_lu.h
```

The selected family is limited to the checked-in public LU header and its
directly related user-facing documentation/examples. Day 3 does not authorize
edits to declarations, implementation, tests, build files, install metadata, or
generated API output.

## Why `include/sparse_lu.h`

Day 2 ranked `include/sparse_analysis.h`, `include/sparse_ldlt.h`, and
`include/sparse_lu.h` as the strongest direct-solver candidates. Day 3
reconciles that ranking with prior cleanup history:

| Candidate | Day 2 signal | Day 3 decision |
| --- | --- | --- |
| `include/sparse_analysis.h` | Largest remaining direct lifecycle header, adoption-visible. | Defer/recheck only. Sprint 155 already cleaned it as part of a selected public-header batch. |
| `include/sparse_ldlt.h` | Highest remaining risk-term density, backend/lifecycle visible. | Defer/recheck only. Sprint 155 already cleaned it as part of a selected public-header batch. |
| `include/sparse_lu.h` | General one-shot direct-solver first-use surface with factor, solve, block solve, transpose solve, condition estimate, refinement, and helper APIs. | Select. It remains high adoption value and was deferred in Sprint 155 because no tutorial gap targeted it. |
| `include/sparse_qr.h` | Strong docs visibility and rank/minimum-norm sensitivity. | Defer. Prior Sprint 155 records identify QR as recently cleaned, and Sprint 172 can close higher-value LU one-shot coherence first. |
| `include/sparse_types.h` | Foundational error/type/macro surface. | Defer. Cleanup risk is higher because enum, typedef, macro, version, and callback wording sits close to ABI-sensitive public declarations. |
| `include/sparse_lu_csr.h` | Specialized CSR LU working-format surface and prior Doxygen warning owner. | Defer. Best handled in a CSR/LU-CSR-specific batch, not mixed into one-shot linked-list LU cleanup. |

The LU header is the best Day 3 target because it is visible in README
first-use material, examples, cookbook workflows, API summaries, maintainer
proof notes, and broad tests, while still being feasible as a declaration-
preserving comment cleanup.

## Selected Family Scope

In scope for Days 4-10:

- `include/sparse_lu.h` comment cleanup;
- LU one-shot lifecycle wording;
- matrix ownership and in-place mutation wording;
- factor/solve/refine/condition-estimate error contracts;
- tolerance semantics for factorization, solve, refinement, and condition
  estimation where existing header text already covers them;
- output buffer, aliasing, and caller allocation wording where existing
  comments or tests already support it;
- progress/cancellation callback wording for `sparse_lu_opts_t`;
- declaration section organization if Day 6 proves it is non-behavioral and
  reviewable;
- README/API/tutorial/cookbook/example wording only where required to keep LU
  public-header comments and user-facing docs coherent;
- lightweight guard/check updates only if Day 10 finds a practical way to
  prevent LU header documentation drift.

Out of scope for this sprint selection:

- `include/sparse_lu_csr.h` public API cleanup;
- `include/sparse_analysis.h` repeated-run lifecycle cleanup;
- `include/sparse_ldlt.h`, `include/sparse_cholesky.h`, and other direct
  solver header cleanup;
- implementation behavior changes in `src/sparse_lu.c`;
- test behavior changes;
- generated API HTML refresh/publication;
- package/install metadata changes;
- shared-library, dynamic ABI, loader, symbol-visibility, or provider-package
  work.

## Supported Contract-Language Scope

The selected cleanup may clarify existing LU API semantics for:

| Area | Allowed clarification |
| --- | --- |
| Ownership and lifecycle | LU factorization mutates the caller-owned `SparseMatrix`; callers should use `sparse_copy()` when they need the original coefficients; repeated stable-pattern workflows belong to `sparse_analysis.h`. |
| Error handling | Existing `SPARSE_OK`, `SPARSE_ERR_NULL`, `SPARSE_ERR_SHAPE`, `SPARSE_ERR_BADARG`, `SPARSE_ERR_SINGULAR`, `SPARSE_ERR_ALLOC`, and `SPARSE_ERR_CANCELLED` wording may be normalized where already documented. |
| Tolerance | Existing factor pivot tolerance, solve singularity check, refinement tolerance, and condition-estimate wording may be made more consistent without changing numeric behavior. |
| Workspace/allocation | Caller-owned input/output buffers and allocation-failure behavior may be clarified where the existing header or tests support it. |
| Threading | Existing same-matrix mutation/read-only solve threading notes may be normalized. |
| Callbacks | `progress_cb` and `progress_user` behavior may be shortened and aligned with `sparse_progress_cb_t` without changing callback semantics. |
| Declaration organization | Section headings may be clarified; declaration moves require Day 6 approval and before/after declaration evidence. |

If Day 4 cannot prove a contract detail from current header text,
implementation, tests, or docs, the cleanup must record a gap rather than
inventing behavior.

## Unsupported Claims

Cleanup of `include/sparse_lu.h` must not imply:

- dynamic ABI stability;
- shared-library builds, installs, import/export macros, symbol allowlists, or
  runtime-loader support;
- package-manager availability or provider registry support;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- portable LU performance superiority;
- broad external-library parity;
- LU CSR public-solve parity;
- state-of-the-art sparse linear algebra coverage;
- generated API HTML publication or freshness.

Any package, ABI, platform, performance, external-comparison, or generated-doc
wording change must remain an explicit non-claim or be deferred.

## Impacted File List

Day 3 identifies these likely impacted surfaces for later days:

| Surface | Expected role |
| --- | --- |
| `include/sparse_lu.h` | Selected cleanup target. |
| `README.md` | LU first-use example, API overview, in-place factorization note, and header index may need coherence checks. |
| `docs/api_reference.md` | Public header/API index may need a small LU wording alignment if header terminology changes. |
| `docs/tutorial.md` | First-solve and direct-solver handoff language may need a coherence check. |
| `docs/cookbook.md` | LU solve recipe may need wording alignment if header lifecycle language changes. |
| `examples/example_basic_solve.c` | Primary maintained LU first-use example for cross-checking public wording. |
| `examples/example_condition.c` | Condition-estimate example for cross-checking condest wording. |
| `examples/example_matrix_market.c` | LU solve example over Matrix Market input for adoption wording. |
| `examples/example_colamd.c` | Reordered LU example for cross-checking reorder language. |
| `tests/test_sparse_lu.c` | Primary linked-list LU behavior proof owner. |
| `tests/test_edge_cases.c` | LU edge-case, singularity, repeated solve, and bad-state proof owner. |
| `tests/test_lu_csr.c` | Cross-check only; do not widen Sprint 172 into LU CSR cleanup. |
| `docs/maintainer_guide.md` | Claim and proof wording cross-check only; public-header policy already exists. |

Day 4 should narrow this list before any edit. Day 5 must not touch every
listed surface by default.

## Behavior-Preservation Constraints

The Sprint 172 LU cleanup must preserve:

- all public function declarations and signatures;
- typedef names and layout;
- enum names, values, and order;
- struct field names, order, types, and layout;
- public macros and numeric values;
- include guards;
- installed header names;
- required includes unless separately proven safe;
- existing ownership/freeing rules;
- existing input mutation and output overwrite behavior;
- existing error returns and default-option behavior;
- existing callback/cancellation behavior;
- static-first package and ABI non-claim boundaries.

Before any Day 5 header edit, Day 4 should capture or define the declaration-
preservation command set for `include/sparse_lu.h`.

## Day 4 Handoff

Day 4 should design cleanup for `include/sparse_lu.h` only. The design should:

1. map current LU declarations into workflow sections;
2. identify comments to keep, shorten, move, or rewrite;
3. prove each ownership, lifecycle, error, tolerance, workspace, threading, and
   callback clarification from current implementation/tests/docs;
4. define declaration-preservation commands;
5. decide whether any declaration reordering is worth proposing for Day 6;
6. define exact docs/examples that may need alignment after the header cleanup.

## Validation Notes

Day 3 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 3.

## Completion Check

- Exactly one public header family is selected: `include/sparse_lu.h`.
- The decision accounts for Day 2 ranking and prior cleanup history.
- Supported contract-language scope is explicit.
- Unsupported package, ABI, platform, performance, and generated-doc claims
  remain non-claims.
- Impacted docs/examples/tests are identified for Day 4 design.
- Behavior-preservation constraints are explicit before implementation.
