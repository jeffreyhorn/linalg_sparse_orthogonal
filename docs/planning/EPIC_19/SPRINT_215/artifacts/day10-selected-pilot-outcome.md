# Sprint 215 Day 10: Selected Pilot Outcome

## Outcome

**Blocked before solver execution.** The selected local correctness-only
CHOLMOD cell was not run. Day 8 requires the v1 schema and semantic validator
to reject the Day 7 negative fixtures before solving or accepting an observed
record. Neither
`tests/corpus/schemas/external_baseline_evidence_v1.json` nor
`scripts/validate_external_baseline_evidence.py` exists on this branch. The
Day 9 wrapper was made fail-closed at that gate and produced a local
`attempt.json` with `provisional_status=blocked`,
`status_reason=schema_validation_unavailable`,
`schema_validation.status=not_run`, empty `sides` and `timing=not_run`.
It produced no `evidence.json`, solution, residual, correctness pass or timing
sample. This is **not** an executed external comparison.

The only supported claim is that the selected fixture and discovered local
CHOLMOD package still matched their pinned discovery identities at preflight,
and that the guard stopped before running either solver when validation was
unavailable. It does not establish numerical correctness, solver parity,
portable speed, a hosted threshold, Linux/macOS timing parity, or
state-of-the-art performance. Publication review remains pending.

## Reproducible Blocker Proof

On Darwin 24.6.0 x86-64, both file-existence checks returned exit 1:

```sh
test -f tests/corpus/schemas/external_baseline_evidence_v1.json
test -f scripts/validate_external_baseline_evidence.py
```

The opt-in wrapper preflight was then invoked on a new generated bundle:

```sh
python3 scripts/run_cholmod_pilot.py --fixture tests/data/symmetric_4.mtx --output build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-preflight-20261019 --correctness-only --execute
```

It exited **2** with
`blocked: schema_validation_unavailable`. Its generated local bundle is
`build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-preflight-20261019/`.
The `attempt.json` SHA-256 was
`ef5ba0af8d878141a993cc4348efa9c5f2ac9fd5b99f1bef6da3d45b3dc0568a`.
The bundle is ignored generated output, not checked-in numerical evidence.
Reproduction should use a **new** run ID because the wrapper refuses to
overwrite an attempt directory.

The attempt captured these pre-solve observations:

| Field | Value or boundary |
| --- | --- |
| Selection | `s215-cholmod-v7_12_2-symmetric4-spdsolve-v1`; one real-float64 SPD solve on `tests/data/symmetric_4.mtx`. |
| Matrix | SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335` matched the Day 3 selection. No solver-side conversion was executed. |
| Baseline discovery | `pkg-config` reported CHOLMOD `5.3.4`, prefix `/usr/local/Cellar/suite-sparse/7.12.2`, include flag `-I/usr/local/Cellar/suite-sparse/7.12.2/include/suitesparse` and link flags `-L/usr/local/Cellar/suite-sparse/7.12.2/lib -lcholmod`. Candidate dylib SHA-256 matched `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`; no runtime-loaded library identity was claimed. |
| Project/toolchain | Git HEAD `07e482875d35c51d4f50de922508fabd5963be15` with untracked Sprint 215 work; Apple clang `17.0.0 (clang-1700.6.4.2)` targeting `x86_64-apple-darwin24.6.0`. Project archive/build flags were not captured because the gate stopped before building the pilot helper. |
| Environment | Platform and the wrapper's explicit thread/library-loader/CFLAGS variables were captured in `attempt.json`; effective thread counts were not observed. |
| Commands/raw files | `pkg-config`, `git`, compiler-version and `otool -L` commands all returned 0. Their stdout/stderr are separate SHA-256-referenced files in the bundle. No `project` or `baseline` command key exists. |
| Correctness/timing | No factorization, solve, `x`, residual, oracle error, raw sample or timing result. The Day 4 rules and Day 5 sampling policy remain specifications only. |
| Schema validation | `not_run` because the v1 schema and validator are absent; no record is eligible for promotion to `correctness_observed` or `measured`. |

The invariant was checked with:

```sh
jq -e '.provisional_status == "blocked" and .status_reason == "schema_validation_unavailable" and .schema_validation.status == "not_run" and .timing == "not_run" and .sides == {} and (.commands | has("project") | not) and (.commands | has("baseline") | not)' build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-preflight-20261019/attempt.json
```

It returned `true` with exit 0. This checks the **absence** of solver
execution; it is not a numerical pass.

## Negative Mutation And Fallback

The focused synthetic test
`test_changed_baseline_version_blocks_before_solver` changes the discovered
CHOLMOD version from `5.3.4` to `5.3.3` while retaining the selected ID.
The wrapper reports `baseline_version_mismatch`, `blocked`, empty `sides` and
`timing=not_run` before any build or solver invocation. A changed package
cannot silently reuse the selected cell key. The Day 9 tests also cover a
changed fixture hash, failing CHOLMOD minor/status, malformed output,
nonfinite solution and failed numerical limits. These are synthetic checks,
not external observations.

The executable fallback remains the Day 9 opt-in C/Python harness. Before a
future attempt can solve or be accepted, implement the isolated v1 schema and
validator with the Day 7 positive/negative fixture matrix and a canonical
complete-cell-ID test vector. Integrate its `--self-test` preflight in the
wrapper; then verify project build flags/archive, actual loaded CHOLMOD and
transitive dependency hashes, module/build options, effective ABI/ordering/
backend/threads, identical matrix/RHS and Day 4 diagnostics. If any gate
remains unavailable, retain a named blocked attempt with raw logs. Do not
promote `attempt.json` by renaming it, infer a zero metric, or change selected
report/canonical benchmark semantics. Publication separately requires fixture
attribution and module-license review.

## Validation Scope

`python3 tests/test_run_cholmod_pilot.py` passed nine synthetic tests,
including the Day 10 baseline-version mutation. The generated attempt
parsed as JSON, the blocked/no-solver invariant passed, and corpus/docs
checks are recorded in working notes. No `.c` or `.h` file changed on Day
10, so the Day 9 `make format`, `make lint`, `make test` result remains the
last C quality gate; Day 10 does not reinterpret it as an external pilot.
