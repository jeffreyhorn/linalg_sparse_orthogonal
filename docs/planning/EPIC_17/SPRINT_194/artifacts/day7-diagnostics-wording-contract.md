# Sprint 194 Day 7 Diagnostics Wording Contract

## Objective

Define the wording standard for diagnostics documentation before broad cleanup
across direct, iterative, QR/SVD, eigensolver, Matrix I/O, example, and report
surfaces. This contract is documentation-only: it must not imply a status-code,
tolerance, backend, or solver-behavior change.

## Current Wording Inventory

| Surface | Current diagnostic owner | Existing wording pattern | Cleanup risk |
| --- | --- | --- | --- |
| Core error codes | `include/sparse_types.h` | `SPARSE_OK` means success; other `sparse_err_t` values are errors; `sparse_strerror()` supplies text. | Public docs must not invent status names or collapse distinct errors into generic failure. |
| Matrix construction | `include/sparse_matrix.h`, `docs/cookbook.md`, `docs/solver_selection.md` | Simple constructors return `NULL`; explicit constructors return `sparse_err_t`; Matrix Market failures may expose `sparse_errno()`. | Avoid promising cause-specific diagnostics for constructor families that only return `NULL`. |
| Direct solvers | `include/sparse_lu.h`, `include/sparse_cholesky.h`, `include/sparse_ldlt.h`, `include/sparse_analysis.h` | Factor, solve, refine, and condest calls return `sparse_err_t`; examples compute residuals for shown systems. | Avoid saying residual checks prove broad correctness, backend superiority, or portable accuracy. |
| Iterative solvers | `include/sparse_iterative.h`, `README.md`, `docs/solver_selection.md`, `examples/README.md` | `sparse_iter_result_t` reports iterations, final relative residual, convergence, stagnation, residual-history count, and breakdown. | Avoid treating `SPARSE_ERR_NOT_CONVERGED` as a hard failure with no approximate solution; result fields are only specified where headers say they are populated. |
| QR | `include/sparse_qr.h`, `docs/solver_selection.md`, `examples/README.md` | QR APIs report rank, nullity/nullspace, R-diagonal diagnostics, least-squares residuals, and minimum-norm outputs. | Avoid broad QR correctness, raw QR-basis parity, global rank-threshold policy, or broad least-squares claims. |
| SVD | `include/sparse_svd.h`, `docs/solver_selection.md`, `docs/cookbook.md`, `examples/README.md` | SVD APIs report singular values, rank, condition, pseudoinverse/low-rank outputs, and `SPARSE_ERR_NOT_CONVERGED` on SVD iteration failure. | Avoid broad partial-SVD correctness, vector identity, repeated-spectrum ordering, sparse-output optimality, or performance claims. |
| Symmetric eigensolver | `include/sparse_eigs.h`, `README.md`, `docs/solver_selection.md`, `examples/README.md` | `sparse_eigs_t` reports residual norm, backend used, convergence count, peak basis size, and shift-invert/preconditioner telemetry. | Avoid converting AUTO routing into a superiority claim or treating shift-invert residuals as original-A residuals without recomputation. |
| Examples | `examples/README.md`, `examples/*.c` | Examples report local residuals, convergence, rank, condition, or solution summaries. | Keep examples framed as teaching output for shown systems, not broad external-library parity or release evidence. |
| Benchmarks/reports | `benchmarks/README.md`, `docs/cookbook.md`, `docs/solver_selection.md`, report scripts/tests | Generated rows are discovery/freshness diagnostics tied to corpus, compiler, backend, thread, index, and manifest context. | Avoid using report rows as broad performance, package, platform, or state-of-the-art evidence. |
| Maintainer proof interpretation | `docs/maintainer_guide.md` | Explains which guards own claims and where evidence remains local or hosted. | Keep maintainer wording separate from user workflow instructions. |

## Term Normalization Table

| Term | Use when | Do not use as |
| --- | --- | --- |
| `return code` | A public function returns `sparse_err_t`. | A synonym for result-struct fields. |
| `SPARSE_OK` | The API completed successfully under that function's documented contract. | A claim that residual, rank, or benchmark quality is globally optimal. |
| `SPARSE_ERR_NOT_CONVERGED` | Iterative, SVD, or eigensolver iteration budget ended without convergence, as documented by that API. | A generic direct-solver failure or proof that the method is unsuitable for all related problems. |
| `SPARSE_ERR_SINGULAR` | LU/LDL/analysis or shift-invert paths encounter a singular or near-singular pivot/factorization condition. | A universal rank diagnostic for QR/SVD paths. |
| `SPARSE_ERR_NOT_SPD` | Cholesky, LDL symmetry checks, or SPD-specific paths reject non-SPD or non-symmetric input as documented. | A general "bad matrix" status for unsymmetric LU, QR, or iterative workflows. |
| `SPARSE_ERR_BADARG` | Options, dimensions, permutations, tolerances, or state preconditions are invalid. | A substitute for shape, singularity, I/O, or convergence wording. |
| `SPARSE_ERR_SHAPE` | Matrix dimensions or solver shape requirements are incompatible. | A rank-deficiency or numerical failure status. |
| `SPARSE_ERR_ALLOC` | Required workspace or object allocation failed. | A recoverability or retry guarantee beyond the local call. |
| `SPARSE_ERR_IO` / `SPARSE_ERR_PARSE` | Matrix Market or stream paths fail with I/O or format diagnostics. | Solver diagnostics after a matrix is successfully loaded. |
| `SPARSE_ERR_CANCELLED` | A progress callback requested cancellation. | A guarantee that in-place factorization inputs are bit-identical after cancellation. |
| `residual` | A norm or vector computed for a specific solve/decomposition workflow. | Broad correctness, backend parity, or performance proof. |
| `relative residual` | The API or example explicitly defines scaling, such as `||b - A*x|| / ||b||`. | A generic residual when the source reports an absolute norm. |
| `converged` | A result field or return contract says the tolerance was met within the budget. | A quality statement independent of tolerance, scaling, backend, or problem assumptions. |
| `non-converged` | `SPARSE_ERR_NOT_CONVERGED` or a result field reports budget exhaustion. | A crash, invalid input, or proof that no approximate solution exists. |
| `stagnated` | `sparse_iter_result_t.stagnated` is set according to the configured stagnation window. | A universal non-convergence synonym. |
| `breakdown` | `sparse_iter_result_t.breakdown` records solver-specific Krylov breakdown. | A generic numerical failure across all solvers. |
| `rank` / `numerical rank` | QR or SVD APIs compute rank under their documented tolerance rules. | A global rank policy for every decomposition or matrix family. |
| `condition estimate` | LU/LDL/QR/SVD APIs produce the documented estimate. | Exact condition number unless the API says exact. |
| `backend used` | A result field reports the selected backend, especially AUTO dispatch. | Evidence that the backend is universally faster or more accurate. |
| `freshness diagnostic` | A generated report row or guard confirms a selected artifact is current. | Broad report coverage, performance, platform, package, ABI, or state-of-the-art evidence. |

## Workflow-Specific Wording Rules

### Construction and Matrix I/O

- Say simple constructors return `NULL` for invalid input or allocation
  failure only when the constructor family has no explicit error out-parameter.
- Say explicit constructors return `sparse_err_t` when the API exposes that
  owner.
- For Matrix Market paths, mention `sparse_errno()` only after
  `sparse_load_mm(...)` or stream I/O failure paths that document errno
  capture.
- Do not promise that a later solver diagnostic can identify the original
  construction or parse issue once input construction succeeded.

### Direct Solvers

- Use `factorization return code`, `solve return code`, `refinement return
  code`, or `condition-estimate return code` for LU, Cholesky, LDL^T, QR, and
  repeated direct lifecycle APIs that return `sparse_err_t`.
- Use `singular or near-singular pivot` for LU/LDL singularity wording when
  matching the public header contract.
- Use `not symmetric positive-definite` for Cholesky/SPD paths and avoid
  shortening it to generic "invalid matrix."
- Treat example residuals as local confidence checks for the shown system.
- Do not use residual examples as broad accuracy, backend superiority, or
  state-of-the-art evidence.

### Iterative Solvers

- Name `sparse_iter_result_t` when describing iteration count, final relative
  residual, convergence, stagnation, residual-history count, or breakdown.
- Say result fields are populated on `SPARSE_OK` and
  `SPARSE_ERR_NOT_CONVERGED` where the headers document that behavior.
- Say `x` is an approximate solution on `SPARSE_OK` or
  `SPARSE_ERR_NOT_CONVERGED` for APIs that document that contract.
- Use `iteration budget` or `max_iter` for non-convergence. Do not call it a
  singularity or invalid-input failure.
- Keep preconditioners framed as acceleration/tuning tools, not convergence
  guarantees.

### QR

- Use `least-squares residual`, `minimum-norm output`, `rank`, `nullity`,
  `nullspace`, and `R-diagonal diagnostics` for QR paths.
- Say QR rank is tolerance-local and API-local. Do not state a global rank
  policy for all workflows.
- Preserve bounded corpus wording for QR fixture rows.
- Avoid raw QR-basis parity, sign/orientation identity, broad least-squares
  parity, or external-library parity unless a specific proof owns it.

### SVD

- Use `singular values`, `numerical rank`, `condition estimate`,
  `pseudoinverse`, `low-rank output`, `triplet residual`, and
  `orthogonality` only in the SVD contexts that expose those diagnostics.
- For partial SVD, keep fixture-local corpus wording and selected comparison
  wording separate from broad correctness.
- Do not imply raw singular-vector identity, vector sign/orientation identity,
  repeated-spectrum ordering, broad sparse-output optimality, or portable
  performance.

### Symmetric Eigensolver

- Use `Ritz residual`, `n_converged`, `backend_used`, `peak_basis_size`,
  `used_csc_path_ldlt`, `shift-invert`, and `preconditioner status` only when
  referring to the public eigensolver result contract.
- Keep `SPARSE_EIGS_BACKEND_AUTO` wording as routing policy. Do not say AUTO is
  universally best.
- When shift-invert is involved, distinguish transformed-operator residuals
  from original-A residuals unless docs explicitly recompute
  `||A v - lambda v||`.
- Keep nonsymmetric eigensolver support excluded.

### Benchmarks and Reports

- Use `local measurement`, `selected freshness diagnostic`, `fixture-local
  evidence`, and `hosted evidence` according to the support/readiness matrix.
- Include corpus, compiler, backend, thread, manifest, and generated index
  context when describing report evidence.
- Do not convert benchmark/report diagnostics into portable performance,
  package, ABI, platform parity, broad external-library parity, or
  state-of-the-art claims.

## Claim-Risk Notes

The following wording patterns should be avoided during Days 8 and 9 cleanup:

- "proves correctness" without naming the bounded API, fixture, tolerance, or
  evidence owner;
- "best", "optimal", or "state of the art" outside a formally documented
  mathematical property already owned by a public API;
- "converged" without stating tolerance/budget/result-field context;
- "failed" when `SPARSE_ERR_NOT_CONVERGED` still leaves an approximate
  solution according to the API contract;
- "singular" for every non-success direct or decomposition path;
- "residual" without saying local, absolute, relative, Ritz, least-squares, or
  transformed-operator context when the distinction matters;
- "Windows support" without CMake/MSVC and non-claim boundaries;
- "report evidence" without selected target, local/hosted, and freshness
  context.

## Cleanup Target Map

| Future day | Scope | Apply this contract by |
| --- | --- | --- |
| Day 8 | Direct and iterative docs | Normalize return-code, residual, convergence, stagnation, breakdown, and preconditioner wording. |
| Day 9 | QR/SVD/eigensolver docs | Normalize rank, nullity, condition, residual, backend, shift-invert, and fixture-local evidence wording. |
| Day 10+ | Cross-doc adoption polish | Keep examples, cookbook, solver selection, API docs, and maintainer guide using the same terms without changing API behavior. |

## Completion Criteria

- Diagnostic terms map to existing public APIs, result fields, tests, or report
  guards.
- Wording cleanup remains documentation-only.
- No new solver preference, tolerance, return-code, result-field, backend, or
  support behavior is implied.
