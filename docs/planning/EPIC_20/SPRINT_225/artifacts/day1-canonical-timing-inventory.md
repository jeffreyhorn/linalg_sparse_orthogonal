# Day 1: Canonical Timing Cell Inventory

## Scope and evidence reviewed

This intake covers only selected target `SRT-BENCH-REFACTOR-CSC-NOS4`, artifact `bench_refactor_csc`, and command `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`. The fixture in this checkout has SHA-256 `6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80`. The source revision inspected was `5de34925dc72ce896e2ba4f8e70d5f73741166d5` on `sprint-225` (the branch contains an uncommitted Sprint 225 plan).

Sources: [Sprint 212 decision](../../../EPIC_19/SPRINT_212/RETROSPECTIVE.md), [runner and variance inventory](../../../EPIC_19/SPRINT_212/artifacts/day3-runner-and-variance-inventory.md), [threshold criteria](../../../EPIC_19/SPRINT_212/artifacts/day4-decision-criteria.md), [Sprint 218 lessons](../../SPRINT_218/RETROSPECTIVE.md), the [selected target manifest](../../../../../tests/corpus/manifests/selected_report_targets.tsv), `benchmarks/bench_refactor_csc.c`, `scripts/bench_canonical_report.sh`, `scripts/check_bench_canonical_freshness.py`, `Makefile`, and the Linux/macOS workflows.

## Current data flow

| Stage | Current behavior | Timing-gate consequence |
| --- | --- | --- |
| Selected identity | Manifest has one `bench_refactor_csc` row for `nos4.mtx --repeat 1`, expected CSV row count 1, and required `bench_refactor_csc.csv;index.tsv;manifest.txt`. | Reuse this identity, but maintain a separate timing decision and evidence bundle. |
| Build | `make bench-canonical-report` builds four benchmark binaries using normal Make flags. The hosted freshness jobs set `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags` and `SPARSE_CANONICAL_BUILD_MODE=serial`. | These are report labels, not an independently pinned compiler/build recipe or binary digest. |
| Generate | `scripts/bench_canonical_report.sh` runs `bench_refactor_csc` once on `nos4.mtx --repeat 1` and redirects stdout to `bench_refactor_csc.csv`; it also runs three other benchmarks and writes `index.tsv` and `manifest.txt`. | There is one process invocation and one selected CSV measurement, no replicated trial series. |
| Metadata | Index/manifest record timestamp, short commit, branch, platform, compiler string, runner context, build flags, CPU model, build mode, thread setting, workload, `configured_repeat_1`, `none_configured`, `not_computed_single_sample`, and `baseline=n/a`, `threshold=n/a`. | Useful provenance context, but inadequate for a threshold. Capture full commit, executable and fixture hashes, exact run/job identity, and all raw trial records for the new gate. |
| Validate | `make bench-canonical-report-freshness` or the hosted checker validates selected files, metadata, and claim boundary; the checker explicitly does not compare timing values. | A freshness pass remains a threshold-free result. |
| Upload | Linux `hosted-performance-freshness` and macOS `selected-performance-freshness` upload only the selected CSV, index, and manifest; both specify seven-day artifact retention. | No separate per-invocation stdout/stderr, exit status, warmup record, or durable raw-sample bundle is retained. |

The local `performance-sentinels` S6 `nos4` ceiling is a separate single-run smoke check with a configurable `500.0` ms default. It is not a hosted baseline or the Sprint 225 acceptance threshold.

## Benchmark timing semantics

`bench_refactor_csc` selects one SPD/Cholesky matrix when given the fixture path. Its default is 10 internal refactors, but the canonical command overrides that to **one** with `--repeat 1`. That internal repeat count is not the number of independent benchmark processes. The CSV contains a header and one `chol_spd` data row. The primary candidate field is `refactor_csc_ms`; other columns include `analyze_ms`, `refactor_public_ms`, both solve times, `speedup_refactor`, and both residuals. Times are emitted to three decimal places in milliseconds, so the current CSV loses sub-microsecond detail. `speedup_refactor` is a within-process ratio, not a candidate-versus-baseline regression statistic.

The benchmark uses `clock_gettime(CLOCK_MONOTONIC)` for interval timestamps. It times `sparse_analyze` separately. Matrix loading, RHS construction, initial numeric-factor priming, and per-repeat matrix copying/value perturbation occur outside the refactor intervals. `refactor_csc_ms` times conversion from sparse matrix with analysis plus supernodal elimination; it includes CSC construction and symbolic materialization. Public refactor and each solve have separate intervals. Residual computation and cleanup are outside the CSC refactor interval. The benchmark divides each accumulated interval by internal repeat count; with `--repeat 1`, it reports the one interval. A failed internal operation exits nonzero after an error on stderr and does not produce the selected completed data row. Today the report script retains redirected stdout for a successful run, while stderr and failed-run evidence are not uploaded as separate raw trials.

Day 2 must decide whether the timing gate uses the rounded `refactor_csc_ms` field or a higher-resolution measurement. Any benchmark timing change needs explicit compatibility review and a fresh baseline. Whole-process elapsed time would measure a different workload because it includes loading, setup and other paths.

## Runner and retention feasibility

| Candidate route | Observed configuration | Open decision |
| --- | --- | --- |
| Linux hosted freshness | `.github/workflows/ci.yml`: `ubuntu-latest`, 10-minute job cap, `cc` through default Make flags, serial report label, `/proc/cpuinfo` CPU model, seven-day artifact. Sprint 212 retained runs showed AMD EPYC 9V74, AMD EPYC 7763, and Intel Xeon 8573C under the same runner label. | The label alone is not a stable timing class. Pin/verify actual image, CPU class, compiler, build flags, and execution context, or use a same-machine comparison protocol. Confirm ownership and durable artifact route. |
| macOS hosted freshness | `.github/workflows/macos-ci.yml`: `macos-latest`, 10-minute job cap, `cc` through default Make flags, serial report label, `sysctl` CPU model, seven-day artifact. Sprint 212 observed Apple M1 (Virtual), yet historical times still varied. | Consider only as an alternative named class after variance review; do not mix with Linux baseline or infer parity. |
| Local S6 sentinel | `scripts/performance_sentinels.sh` has a local one-run `nos4` smoke ceiling. | Keep separate from the hosted selected timing gate. |

The existing upload mechanism (`actions/upload-artifact@v4`, `if: always()`) can carry a new raw-trial bundle, but the present seven-day retention is short for baseline review. No longer-lived, immutable baseline/sample route is established by the inspected configuration. The Sprint 218 measurement used 60 local process invocations with 180 hashed raw files; that is a useful retention and comparability pattern, not a baseline for this different kernel or host.

## Day 1 decision and blockers

- **Decision:** Keep `SRT-BENCH-REFACTOR-CSC-NOS4` as the sole timing-gate candidate and leave selected freshness unchanged. The current canonical report is valid threshold-free evidence only.
- **Blocker:** No reviewed stable runner/compiler class, process-trial protocol, warmup/dispersion rule, versioned baseline, threshold, or durable raw-sample route exists yet. Timing-gate promotion is blocked.
- **Day 2 input:** Pre-register the runner/CPU identity policy, exact build, trial and clock semantics, raw-log format, invalid-trial treatment, baseline update/expiry, statistic, dispersion, threshold, and retention. Historical Sprint 212 values are context only; they cannot be combined as baseline samples because commits and runner assignments differ.

## Validation

Read-only source/plan inspection, `git rev-parse HEAD`, and `shasum -a 256 tests/data/suitesparse/nos4.mtx` supplied this inventory. No benchmark or C/header source changed on Day 1. `git diff --check` is the documentation gate for this artifact.
