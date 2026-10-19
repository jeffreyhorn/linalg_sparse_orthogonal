# Sprint 215 Day 5: External Comparison Methodology

## Scope And Evidence Boundary

This completes item 215.2's **predeclared** methodology for the one selected
Day 3 cell. The input is the exact SHA-256-pinned `symmetric_4.mtx` file,
real-float64 SPD `A*x=b` with `x_ref=[1,1,1,1]` and
`b=[13,14,17,16]`. The project path is no-reorder linked-list Cholesky;
the external path is SuiteSparse `v7.12.2` / CHOLMOD 5.3.4, natural-order
simplicial `CHOLMOD_A`. See `day3-baseline-selection-decision.md` for the
package/fixture selection, `day3-methodology-inputs.md` for the complete cell
key and conversion checks, and
`day4-correctness-and-tolerance-contract.md` for numerical pass/fail rules.

The **selected pilot remains correctness-only**. The timing protocol below
governs a *separate, later optional* local measurement, not a Day 5 run or a
retrospective interpretation of existing benchmark data. A timing record
cannot become external performance evidence without the complete cell key,
both independent correctness passes, verified environment and valid raw
samples. Nothing here creates a hosted threshold, Linux/macOS timing parity,
or a broad solver-speed ranking.

## Workload And Phase Boundaries

Start each trial from the same verified, untouched full matrix and RHS. For
each side, prepare an independent mutable input and a fresh factor state;
never time a solve against a factor left by an earlier trial. An external
lower-triangular CSC view is checked against the full matrix as specified on
Day 3. Run project `sparse_cholesky_factor_opts` and
`sparse_cholesky_solve` with the frozen options; run CHOLMOD
`cholmod_analyze`, `cholmod_factorize`, and `cholmod_solve(CHOLMOD_A,...)`
with the frozen method controls. Record effective factor ordering/mode, not
just requested options.

| Window | Start and stop | Selected-cell treatment |
| --- | --- | --- |
| Input/setup | Before reading/hash-checking fixture through parsing, symmetry expansion, CSC conversion, RHS preparation and CHOLMOD common/input allocation; stop when both ready-to-factor inputs exist. | Record separately as workflow/setup cost if implemented. It is **not** comparable solver time: source formats and conversion APIs differ. Exclude from the solver call windows. |
| Factor ready | Project: immediately before `sparse_cholesky_factor_opts` to its return. CHOLMOD: immediately before `cholmod_analyze` to `cholmod_factorize` return, including symbolic analysis. | Closest call-level factorization workload, descriptive only. Do not compare CHOLMOD numeric factorization alone with the project's integrated factor call or claim equal internal phase boundaries. Input conversion and factor destruction are outside both windows; allocations performed inside APIs remain included. |
| One solve | Immediately before each solve API call to its return, with a newly and successfully factored state and the fixed RHS; output allocation inside an API remains included. | Same `A*x=b` operation, but allocator/API overhead can differ. Report call-level latency, not a pure triangular-solve-kernel speedup. Validation and output destruction are outside this window. |
| Factor plus first solve | Immediately before factor/analysis to immediately after the first successful solve in the **same trial**. Include required inter-call status handling and API-internal allocations; exclude input setup, independent validation and factor destruction. | Primary descriptive prepared-input workload if a later timing run is permitted. Measure directly; do not add independently measured medians. Different internal analysis/factor boundaries remain disclosed. |
| Reuse/refactor | No second RHS, retained symbolic factor or numeric refactor in this cell. | `not_applicable`, not zero time. A reuse claim needs a new operation and methodology. |
| Full program/end-to-end | Process startup, dynamic loading, file I/O, input conversion, solver, reporting and teardown. | Optional diagnostic only. Do not combine with prepared-input call latencies or use for a solver ranking. |

Compile-time setup and package installation are never timed. The project
default Makefile `-O2` and discovered Homebrew library are proposed local
inputs, not verified equivalent build modes. If one side cannot provide the
frozen backend, ABI, threading or full operation, the cell is blocked or
`not_comparable`; do not change algorithms to obtain a number.

## Separate Optional Timing Protocol

1. Run the Day 3 input/provenance and Day 4 numerical checks on both sides before timing. Freeze the complete cell key, commands, binaries, environment and methodology version. Any failure stops the measurement path; retain failure evidence without a speed verdict.
2. Use one host and one process/session with the same monotonic clock (`clock_gettime(CLOCK_MONOTONIC)` where available); record clock identity and measured `clock_getres`. Capture integer nanosecond deltas; reject zero, negative or nonfinite durations. Do not subtract a measured "timer overhead" from individual samples.
3. Set `OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before loading either library. Record the actual linked BLAS/OpenMP libraries and verify effective thread counts during the run; environment variables alone are not proof of single-thread execution. If verification is unavailable, mark thread provenance `unverified` and timing `not-evidence`.
4. For each reported window, perform five paired warmup trials, then 50 paired timed trials. Alternate project-first and CHOLMOD-first order on successive pairs (AB, BA). Recreate the mutable matrix, factor and RHS for every trial outside the relevant timer. Do not retain symbolic/numeric state across trials or discard slow samples after seeing them.
5. Use a 30-second timeout per side per trial and a 15-minute cap for the optional run. A timeout, failed allocation, non-OK solver status, missing output, input mutation or nonfinite result invalidates that trial and the aggregate; record the raw failure, not a replacement sample or zero. Reapply the Day 4 finite/status/residual/oracle checks to every timed trial after stopping its timer. Do not time only the successful subset.
6. Preserve each ordered raw duration and status. For each side/window, report sample count, median, minimum, maximum, Q1, Q3, IQR and median absolute deviation in nanoseconds; state the quartile convention used by the implementation. Report warmup outcomes separately. If the median is below 100 times observed clock resolution, or any duration is zero/nonmonotonic, label the window `resolution_limited` and do not interpret its latency. No post-hoc outlier trimming or tolerance tuning is permitted.

The 4 x 4 problem is dominated easily by dispatch, allocation, cache and
timer effects. Even valid local samples are **descriptive call latencies**
for this one host/build/cell. Do not publish a winner, speedup ratio or
performance threshold for the selected pilot. A larger, decision-worthy
performance study needs new matrices and a separately approved protocol.
Run order, clock resolution, thermal/power state and competing workload must
be recorded; if they cannot be controlled or described, timing is exploratory
and `not-evidence`, not an absence of cost.

## Memory, Fill And Missing Values

Record factor nonzeros or fill only when each API supplies a clearly defined
quantity, with the exact definition and factor type (`is_ll`, `is_super` for
CHOLMOD). Do not equate CHOLMOD's analysis estimate with the project's
post-factor count, or compare LL and LDL storage as identical fill. Peak RSS
may be recorded only with a named OS measurement method and process scope;
one combined process cannot attribute peak RSS to each solver. Allocation
count and phase-specific memory are optional until an instrumented method is
defined. Store an unavailable measurement as `null` plus
`availability=not_collected|not_exposed|not_applicable|failed` and a reason;
never use 0 or an empty string for unavailable time, memory or fill.

## Required Provenance And Raw Record

The future report schema is Day 6 work. Regardless of its final home, a
raw run bundle must contain the following before a measured row is accepted:

| Group | Required fields and evidence |
| --- | --- |
| Cell and method | Matrix path and file SHA-256; dimensions, field/symmetry and expanded-nnz checks; full Day 3 cell key; exact RHS/oracle; method artifact version or SHA-256; Day 4 tolerance version; phase definitions and timing parameters. |
| Project build | Git commit and dirty-tree status; compiler executable/version, full compile/link flags, `SPARSE_IDX_BITS`, optimization/debug mode, OpenMP/mutex configuration; project executable and library SHA-256. A dirty tree requires a patch/diff hash or is blocked for reproducible evidence. |
| External package | SuiteSparse tag/commit, CHOLMOD version, package source/formula/revision, prefix, actual loaded library path and SHA-256, CHOLMOD index ABI, relevant build options/module composition and transitive-library paths/versions/hashes (AMD/COLAMD, BLAS/LAPACK, OpenMP and SuiteSparse config). A matching `pkg-config` version alone is insufficient. |
| Host and execution | OS/version/build, architecture and CPU model/core count, memory, compiler runtime, host identifier (privacy-safe), power/thermal state if available, concurrent load, clock identity/resolution, environment thread variables and observed thread count, locale, full commands, working directory, timestamps and trial order. A Linux/macOS or two-host comparison is a different evidence set. |
| Results and integrity | Per-side phase statuses, factor diagnostics, raw `x`, residuals and oracle errors, Day 4 dispositions; ordered warmup/timed samples with units, timeouts and failures; summary algorithm/version; hashes of raw logs, harness source/binary, manifest and any generated output. Preserve raw values before summarizing. |

Capture the actual runtime-loaded library and dependencies, not only the
linker's intended search path. If Homebrew replaces the discovered
`libcholmod.5.3.4.dylib` (Day 2 SHA-256
`0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`),
or any identity component changes, record a new cell rather than overwriting
the old one. Unknown external build flags or module terms must be stated as
unknown/blocking where they affect reproducibility or publication; never
invent a release/debug label. Fixture authorship and module licenses are
separate publication gates.

## Rerun Checklist And Acceptance

1. Check out the exact recorded project commit and preserve any recorded patch; verify `git rev-parse HEAD`, `git status --porcelain=v1`, and `shasum -a 256 tests/data/symmetric_4.mtx`. Verify the RHS and the full Day 3 conversion invariants before calling either solver.
2. Restore the recorded SuiteSparse package/build, and verify `brew list --versions suite-sparse`, `pkg-config --modversion cholmod`, `pkg-config --variable=prefix cholmod`, `pkg-config --cflags --libs cholmod`, the loaded dylib digest, and its dependency graph. On macOS, `otool -L` is useful linkage evidence but does not replace a runtime load-path check. Stop on any identity mismatch.
3. Reproduce the exact project build and external harness commands/flags, record binary hashes, set and verify thread controls, and capture the host/clock state. If the harness or dependency provenance is unavailable, report `blocked`, not a partial timing pass.
4. Run the Day 4 correctness-only gate on fresh inputs and retain raw outputs. For the selected pilot, stop here and report only numerical outcomes. A later optional timing run must use the fixed protocol above and verify correctness again for every trial.
5. Hash the raw input, commands/configuration, log and generated report artifacts. A reviewer checks cell identity, per-side numerical pass, complete provenance, all raw samples and the absence of missing-value-as-zero substitutions before accepting descriptive local timing.

This checklist specifies what a future harness must capture; no external
harness, package rebuild, solver run or timed sample was produced on Day 5.
The existing Sprint 212 selected canonical benchmark remains a separate,
threshold-free freshness check with one repeat. Its rows cannot be promoted
to this external-baseline methodology, and this local method cannot imply
hosted, cross-platform or state-of-the-art evidence.
