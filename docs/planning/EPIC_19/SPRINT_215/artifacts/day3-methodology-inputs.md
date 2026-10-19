# Sprint 215 Day 3: Methodology Inputs

## Scope And Cell Identity

This is the input contract for the selected Day 3 **correctness-only** pilot.
One result row must contain the complete comparison-cell identity, not just
a fixture name and baseline version:

| Identity component | Frozen pilot value or required execution field |
| --- | --- |
| Matrix | `tests/data/symmetric_4.mtx`, SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`; 4 x 4 real symmetric, seven stored / ten expanded entries. |
| Problem and RHS | SPD linear solve `A*x=b`; `x_ref=[1,1,1,1]`, `b=[13,14,17,16]`, one RHS, one solve; no transposition or repeated solve. |
| Project solver | Exact pilot Git SHA; `sparse_cholesky_factor_opts` + `sparse_cholesky_solve`; no reorder, linked-list backend, `used_csc_path=0`; default 32-bit indices and float64 scalar. |
| External solver | SuiteSparse `v7.12.2` commit `42151688813c45846a597edcb601435a0e38f3dd`, CHOLMOD 5.3.4; absolute loaded library path and SHA-256, `cholmod_analyze` + `cholmod_factorize` + `cholmod_solve`, natural ordering, postorder off, simplicial, real double; record effective mode and index ABI. |
| Build and execution | OS/architecture, compiler/version/flags, debug/release or optimization mode, dependency/BLAS library versions and paths, environment, effective thread count. Project default `-O2` is a proposed local build, not an observed pilot build. External Homebrew build flags are unresolved and must be captured. |
| Evidence state | `not-run`, `blocked`, `invalid-input`, `correctness-fail`, or `correctness-pass` with raw status/output/provenance. Timing fields are absent or explicitly `not-run`. |

Represent the cell key as the ordered tuple
`(matrix_sha256, problem_class, rhs_sha256_or_exact_values, operation,
project_git_sha, project_solver_path, project_backend, project_ordering,
external_project_and_commit, external_binary_sha256, external_solver_path,
external_backend, external_ordering, precision, index_abi, threads,
project_build_mode, external_build_mode, host_and_dependencies)`.
The exact RHS values above may be stored directly instead of a derived hash.
Do not merge rows when any key component differs. Compare statuses and
residuals within the cell; factor layouts are not required to be identical.

## Input Construction And Conversion Checks

1. Verify file SHA-256 before loading. Require Matrix Market `coordinate real symmetric`, dimensions 4 x 4, seven stored entries, finite values and valid one-based indices. Reject changed bytes, duplicate coordinates, missing diagonals, bad shape or unsupported field/symmetry rather than silently repairing the input.
2. Expand each off-diagonal stored entry **once** into its transpose; do not double diagonal values. Independently check expanded `nnz=10`, symmetry and the complete coefficients. The project `sparse_load_mm` performs symmetric expansion; make a fresh copy for in-place factorization. Keep an untouched `A` for RHS and residual checks.
3. Construct external lower-triangular CSC in canonical column order with sorted row indices and one copy of each stored lower-triangle entry. Set CHOLMOD `stype=-1` and real-double representation; do not feed both triangles with a symmetric-storage flag. Check CSC column pointers, indices, values and reconstructed dense entries against the project-loaded `A` before factoring.
4. Form the single float64 RHS from the untouched expanded matrix and `x_ref`, and verify it equals `[13,14,17,16]`. Check both sides receive identical RHS bytes and dimensions. Do not let either solver's in-place factor overwrite the reference `A` or RHS.
5. Record return codes, CHOLMOD common status, output finiteness, selected ordering/backend and any input mutation. A successful process exit is not enough to classify numerical correctness. Raw `x`, `A*x-b`, reference-solution error and denominator quantities must be retained for Day 4's predeclared metrics.

The existing symmetric-loader test in `tests/test_sparse_io.c` checks the
mirrored values. It does not prove CHOLMOD conversion, external execution or
the complete cell identity.

## Eligibility And Inapplicable Cases

The selected cell is SPD only. A non-square, nonsymmetric, complex, nonfinite
or malformed input is `invalid-input` for this cell; do not coerce it to SPD.
A zero/singular or rank-deficient matrix is `inapplicable` to the selected
unregularized Cholesky comparison and must not count as a solver failure or
successful baseline match. An indefinite or merely symmetric-but-not-positive
matrix is likewise `inapplicable`; the expected rejection/status can be
studied in a separate negative-test cell, not pooled with SPD solve rows.
Missing RHS, matrix bytes, external package, binary provenance, thread
control or license/publication evidence is `blocked` at the appropriate
execution or publication gate. Do not substitute a different fixture or
silently change algorithm family to obtain a result. A genuinely failing
factorization on a verified eligible SPD input is `correctness-fail`, with
raw diagnostics retained; do not relabel it `inapplicable` after observing the
result.

## Assumptions, Exclusions And Open Decisions

- Assumption: the checked-in 4 x 4 matrix is SPD by strict diagonal dominance with positive diagonal; eligibility must still be checked from loaded coefficients before an accepted result.
- Assumption: the local macOS x86-64 CHOLMOD package can be used for a narrow local pilot. Runtime load path, ABI, dependency graph, compile options, module composition and effective thread count are unverified.
- Exclusion: existing source-controlled dense helpers, `HB/nos4` canonical freshness, optional SuiteSparse corpus, SciPy/SuperLU, factor-entry matching, timing and cross-platform results are not this cell.
- Day 4 decision: predeclare solver-status mapping, residual/backward-error and solution-error formulas, finite-value rules and numeric tolerances before any pass/fail classification. Do not set thresholds from observed outputs.
- Day 5 decision: predeclare measurement phases (conversion, analysis, numeric factor, solve), warmup/repeats, clock, dispersion, allocation/thread controls and host provenance before any runtime collection or speed claim. A correctness-only row must not inherit the selected canonical benchmark's one-repeat freshness semantics.
- Publication gate: confirm fixture authorship/file terms and CHOLMOD module/license terms before distributing a third-party comparison artifact. Preserve source and binary hashes and attribution in any later report.

The Day 3 decision and candidate exclusions are in
`day3-baseline-selection-decision.md`. This record defines inputs for a
future harness; it is not a test report.
