# Sprint 215 State-Of-The-Art Evidence Blueprint

## Decision And Earned Scope

This is an executable *design* for external-baseline evidence, not a
state-of-the-art result. Sprint 215 selected one local, correctness-only
CHOLMOD cell. Its [Day 10 attempt](artifacts/day10-selected-pilot-outcome.md)
stopped before either solver ran because the proposed v1 schema and validator
do not exist. [Day 13 validation](artifacts/day13-integrated-validation.md)
passed the repository quality and compatibility checks, but produced no
external numerical or timing evidence. The retained generated `attempt.json`
is a blocked preflight, not an accepted `evidence.json`.

The existing selected comparison rows use source-controlled helpers. The
canonical `bench_refactor_csc`/`nos4.mtx` row is local, repeat-one and
threshold-free. Neither is a third-party result or a substitute for this
cell. No selected manifest, report normalizer, workflow or benchmark contract
was changed for Sprint 215.

## Exact Selected Cell

| Field | Frozen choice |
| --- | --- |
| Selection | `s215-cholmod-v7_12_2-symmetric4-spdsolve-v1` |
| Baseline | SuiteSparse `v7.12.2`, commit `42151688813c45846a597edcb601435a0e38f3dd`, CHOLMOD 5.3.4 C API; discovered local Homebrew package, not runtime-verified |
| Input | `tests/data/symmetric_4.mtx`, SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`; 4 x 4 real symmetric, seven stored and ten expanded entries |
| Problem | Real float64 SPD `A*x=b`, `x_ref=[1,1,1,1]`, `b=[13,14,17,16]`; one factorization and one solve |
| Project path | Fresh matrix; `sparse_cholesky_factor_opts` with `SPARSE_REORDER_NONE` and `SPARSE_CHOL_BACKEND_LINKED_LIST`, then `sparse_cholesky_solve`; default 32-bit index ABI |
| Baseline path | Lower-triangular CSC; CHOLMOD natural ordering, no postorder, simplicial mode, `cholmod_solve(CHOLMOD_A,...)`; record actual type, factor mode, ABI and loaded library |
| Initial environment | Local macOS x86-64 only; correctness-only, no timing or hosted inference |

The exact matrix conversion and complete identity tuple are in the
[Day 3 inputs](artifacts/day3-methodology-inputs.md) and
[selection decision](artifacts/day3-baseline-selection-decision.md).
Changing matrix bytes, RHS, precision, baseline build, backend, ordering or
operation requires a new selection/cell decision, not reuse of this ID.
The discovered dylib hash is a candidate fingerprint, not proof of the
library actually loaded by a solver process.

## Method And Acceptance

1. Verify the checked-in input hash, shape, symmetry expansion, finite
   coefficients, identical matrix/RHS on both sides, solver options, actual
   execution path and complete build/library/dependency/thread provenance.
   Missing identity or ineligible input is `blocked` or `non_comparable`, not
   a pass.
2. Preserve per-phase returns and diagnostics. Each solver independently
   needs successful factorization and solve, finite solution and metrics,
   plus the predeclared [Day 4 numerical limits](artifacts/day4-correctness-and-tolerance-contract.md).
   With `T=32*4*DBL_EPSILON` and `K=17/7`, compute the infinity-norm
   residual and oracle error from untouched `A`, `b` and each raw `x`.
   Require `R <= T*max(1,||b||_inf)+T*(||A||_inf*||x||_inf+||b||_inf)` and
   `E <= T*K*max(1,||x_ref||_inf)+T*K*||x_ref||_inf`. Retain backward error
   and the cross-solver difference. A successful process exit alone is not
   numerical correctness; do not tune tolerances after seeing output.
3. The selected pilot has `timing=not_run`. A separate, approved local
   timing run must first pass both correctness gates and the
   [Day 5 protocol](artifacts/day5-external-comparison-methodology.md):
   prepared-input phase boundaries, five paired warmups, 50 timed pairs in
   alternating AB/BA order, raw nanosecond samples, clock-resolution checks,
   per-trial numerical validation, median and dispersion, and complete host,
   binary and thread provenance. Failed or resolution-limited trials are
   retained and cannot yield a speed verdict. The tiny selected cell does
   not authorize a winner or portable speed ratio even if a local timing
   record later validates.
4. Keep fixture attribution and CHOLMOD module/dependency license review as
   a separate publication gate. Local feasibility and a license expression
   are not publication approval.

## Evidence Contract

The [Day 6 design](artifacts/day6-external-report-schema-design.md) and
[Day 7 validation rules](artifacts/day7-schema-validation-design.md) specify
an isolated `external_baseline_evidence_v1` JSON schema. The
[planning example](artifacts/day7-external-evidence-blueprint.example.json)
has `record_kind=blueprint`, `evidence_status=planned`, null `cell_id`, two
`not_run` sides, `timing.state=not_run` and no generated artifacts. It parses
as JSON; it is not semantically validated by a v1 implementation.

The future validator must reject duplicate/unknown fields, nonfinite values,
selection drift, incomplete observed-cell identity, inconsistent statuses or
units, unsafe/missing raw paths, hash mismatches, and unsupported claims.
Generated observed records require complete project/baseline binaries and
dependency identity, methodology hashes, raw full-precision solutions and
metrics, independent numerical gates, and a complete canonical cell-ID test
vector. `correctness_observed` needs both verified passes; `measured` also
needs the Day 5 sample contract. `planned`, `not_run`, `blocked`, failed and
non-comparable states cannot be mapped to an existing selected `pass` row.

The *implemented* [opt-in runner](../../../../scripts/run_cholmod_pilot.py)
currently writes only provisional `attempt.json` with
`provisional_status`, `sides` and scalar `timing`; it does not emit the
proposed v1 `evidence.json`. Its Day 10 record has `blocked`,
`schema_validation_unavailable`, empty `sides`, no solver commands and no
timing. Do not rename it or translate its status into a v1 observation.

## Conditional Rerun

The commands below are a handoff sequence, **not** a claim that the missing
validator exists or that a solver has run. Use a fresh run ID each time;
the runner refuses to overwrite a bundle.

1. Implement `tests/corpus/schemas/external_baseline_evidence_v1.json`,
   `scripts/validate_external_baseline_evidence.py`, its `--self-test`,
   canonical full-cell-ID test vector and Day 7 positive/negative fixtures.
   Make the validator accept the planning example only as `planned`. Pass
   corpus, manifest, normalizer, comparison, benchmark and docs regressions
   without altering selected-row meanings.
2. Recheck fixture/package/binary hashes, actual loaded and transitive
   libraries, project archive/flags, ABI, ordering, simplicial backend and
   effective threads. If any gate is unavailable, retain a named blocked
   attempt. With gates implemented, run the following on a machine where the
   selected package is present:

   ```sh
   python3 scripts/validate_external_baseline_evidence.py --schema tests/corpus/schemas/external_baseline_evidence_v1.json --self-test
   shasum -a 256 tests/data/symmetric_4.mtx
   pkg-config --modversion cholmod
   RUN_ID=local-correctness-new-001
   python3 scripts/run_cholmod_pilot.py --fixture tests/data/symmetric_4.mtx --output "build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/$RUN_ID" --correctness-only --execute
   ```

3. Inspect and hash the generated raw command logs, actual loaded libraries,
   factor diagnostics, full solutions and Day 4 metrics. The runner's
   `attempt.json` remains provisional even if it reports a two-sided
   candidate. Add a reviewed transformation/validator path before creating
   `evidence.json`; freeze the canonical cell ID from *complete* identity,
   validate the record and its referenced files, and retain the original
   attempt. The planned validation interface is:

   ```sh
   python3 scripts/validate_external_baseline_evidence.py --schema tests/corpus/schemas/external_baseline_evidence_v1.json --input "build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/$RUN_ID/evidence.json" --bundle-root "build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/$RUN_ID"
   ```

   This final command is **not runnable** until both the validator and a
   separately reviewed `evidence.json` producer exist. Any mismatch stays
   blocked/failed with raw artifacts; no row is silently promoted.

## Expansion And Claim Gates

One validated local cell could support only a narrowly identified
correctness observation. A cross-matrix claim first needs distinct,
provenance-cleared matrices with separate predeclared eligibility and
numerical rules, not merely more repeats of `symmetric_4.mtx`. Candidates
are `tridiagonal_20.mtx` and `nos4.mtx`, after their source/terms and
operation compatibility are reviewed. A cross-solver-family claim needs a
second independently pinned external solver and a matching operation; the
deferred SciPy/SuperLU LU candidate needs installed/locked dependencies and
a genuinely nonsymmetric canonical fixture. Neither candidate is an
executed Sprint 215 comparison.

Linux/macOS parity requires separately validated same-cell observations and
branch-specific hosted provenance on **each** named platform, with matched
methodology, package/build/ABI/thread controls and raw logs. A performance
claim additionally needs valid Day 5-style paired samples on each claimed
host, representative workloads and explicit host-comparability reasoning;
one tiny cell or the threshold-free canonical row cannot establish it.
State-of-the-art claims would require a separately approved representative
matrix/solver/platform suite, current pinned baselines, accepted raw
correctness and measurement records, statistical treatment and publication
review. Two matrices or two platforms are necessary for some comparisons,
but never automatically sufficient for broad claims. See the
[residual queue](RESIDUAL_QUEUE.md) for owner decisions and stop conditions.
