# Sprint 145 Day 8 Solver Front Door

## Purpose

Align solver-selection, diagnostics, and advanced-control guidance with the
README, examples, cookbook, and INSTALL ladders created earlier in Sprint 145.
The goal is a short first-use route for choosing a solver and interpreting the
first local result before changing backends, tolerances, preconditioners, or
benchmark settings.

## Changed Surfaces

| Surface | Change | Owner |
| --- | --- | --- |
| `docs/solver_selection.md` | Added `First-Use Solver Route`, `Diagnostics Handoff`, and `Advanced-Control Escalation` sections. | Solver-selection front door |
| `README.md` | Routed the diagnostics adoption-map row and quick-start next step to the solver-selection diagnostics handoff. | README front door |
| `examples/README.md` | Linked example-local diagnostics to the solver-selection escalation view. | Runnable example front door |
| `docs/cookbook.md` | Linked the data-first diagnostic ladder to solver-selection diagnostics before backend/tolerance changes. | Data-first cookbook |

No source files, public headers, build rules, package metadata, or CI workflow
files were changed.

## First-Use Solver Route

| Step | Route | Boundary |
| --- | --- | --- |
| Confirm matrix entry | CSR, CSC, Matrix Market, hand-written, or existing matrix shell. | Storage format does not choose the solver after a `SparseMatrix *` exists. |
| Choose by shape | LU, Cholesky, LDLT, QR, iterative, eigensolver, or SVD. | Mathematical assumptions own the first solver choice. |
| Run one maintained example | `example_basic_solve`, `example_compressed_input`, then family examples. | Examples teach public API usage, not broad numerical parity. |
| Inspect diagnostics locally | Return codes, constructor diagnostics, residuals, convergence, rank, condition, Ritz residuals, or report context. | Diagnostics belong to the workflow that produced them. |
| Escalate after the first result | Typed backend/options, preconditioners, tolerances, benchmarks, reports. | Advanced controls do not create portable performance or platform claims. |

## Diagnostics Ownership

| Workflow | Diagnostic owner |
| --- | --- |
| CSR/CSC construction | `NULL` result or explicit `sparse_err_t` from `sparse_from_*`. |
| Matrix Market input | `sparse_errno()` after load failure. |
| One-shot direct solve | Factorization/solve return code and local residual. |
| Repeated direct lifecycle | Analyze/factor/refactor return codes, same-pattern invariant, and solve residual. |
| Iterative solve | Convergence status, residual history, iteration count, stagnation, and breakdown. |
| QR | Rank, residual, nullity, and nullspace output within the bounded QR workflow. |
| SVD and partial SVD | Rank, condition, triplet residuals, convergence status, and fail-closed status. |
| Eigensolver | Ritz residual, convergence count, selected backend, peak basis size, and shift-invert/preconditioner status. |
| Benchmarks/reports | Matrix corpus, compiler, backend, thread settings, generated index, and manifest context. |

## Claim Boundary Review

| Area | Day 8 result |
| --- | --- |
| Direct solvers | Solver choice remains assumption-based: general square, SPD, symmetric indefinite, rectangular/rank-sensitive, or stable-pattern reuse. |
| Iterative solvers | Diagnostics and preconditioners remain local workflow aids, not convergence-rate or universal speedup guarantees. |
| QR | Existing `qr_rank_deficient_6x4_nullspace_v1` language remains fixture-local and rejects broad QR, external-library, performance, and state-of-the-art parity. |
| Partial-SVD | Existing `partial_svd_clustered_repeated_diag8x6_k3_v1` language remains fixture-local and rejects broad repeated-spectrum, vector identity, parity, performance, package, ABI, and state-of-the-art claims. |
| Runtime/backend | Defaults remain first-use guidance; explicit backend/options are advanced controls after diagnostics justify them. |
| Benchmarks/reports | Measurement remains local and configuration-sensitive, not portable runtime proof. |
| Package/platform | No install support tier wording changed; static-first and platform boundaries remain owned by `INSTALL.md`. |

## Validation

| Check | Result |
| --- | --- |
| solver-selection anchor scan | Passed |
| front-door diagnostics link scan | Passed |
| unsupported numerical/state-of-the-art claim scan | Passed: matches are explicit non-claims or bounded evidence |
| `git diff --check` | Passed |
| untracked artifact whitespace scan | Passed |
| `.c` / `.h` changed-file scan | Passed: no paths |

`make format && make lint && make test` was not required because Day 8 changed
only documentation.

## Day 8 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| First-use users can identify the likely solver path and diagnostic path. | Complete | `First-Use Solver Route` and `Diagnostics Handoff` now precede detailed solver-family sections. |
| QR and partial-SVD wording stays bounded to earned evidence. | Complete | Existing fixture-local evidence sections remain intact and are referenced from diagnostics. |
| Unsupported numerical claim scans pass. | Complete | Claim scan matches are explicit non-claims or bounded evidence statements. |

## Day 9 Handoff

Day 9 should design the public-header cleanup pass. Focus on comments that
make first-use API contracts harder to read, while preserving ownership,
return-code, mutation, ABI, and numerical-boundary semantics in the headers.
