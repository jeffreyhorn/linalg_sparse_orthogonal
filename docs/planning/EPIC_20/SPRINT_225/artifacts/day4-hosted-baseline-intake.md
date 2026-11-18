# Day 4: Hosted Baseline Trial Intake

## Outcome

Collected and independently inspected **one** `s225-v1` hosted macOS baseline job from exact commit `7e49ef3a96e00dcc815da3ba83b006934815c242`. It contains all five warmups and 20 measured process trials, with no invalid trials. This is **intake evidence**, not an accepted baseline: the locked policy requires three compatible independent hosted jobs, dispersion review, versioned storage, and an explicit approval decision. The timing gate remains blocked.

## Run, job, and artifact identity

| Field | Verified value |
| --- | --- |
| Workflow | `Sprint 225 baseline intake`, `.github/workflows/sprint225-baseline.yml`, branch push event |
| Run | [44395634285](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44395634285), attempt `1`, success |
| Job | `collect-baseline`, API ID `135124640857`, success; started `2026-11-18T20:52:59Z`, completed `2026-11-18T20:53:27Z` |
| Source | `7e49ef3a96e00dcc815da3ba83b006934815c242`; `run.json` says clean checkout and `dry_run=false` |
| Artifact | [sprint225-baseline-44395634285-1](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44395634285/artifacts/14469044864), ID `14469044864`, ZIP size 15,860 bytes |
| ZIP SHA-256 | `3093366692075f3fedabc0735bf6425b2b23ffb7880403919f338cf6cbc100fe`; downloaded ZIP matched the artifact API digest and upload log |
| Retention | API expiry `2027-02-16T20:52:53Z` (90-day requested retention) |
| Local audit copy | `/tmp/sprint225-baseline-44395634285.zip`; ignored local copy, not source-controlled baseline data |

**Day 5 retention update:** A GitHub rerun removed this attempt's hosted artifact before its configured expiry. Its previously downloaded ZIP was rehashed and copied to [the rejected baseline evidence set](day5-rejected-baseline/attempt-1.zip). The link above may therefore no longer resolve; the recorded digest and retained ZIP remain the audit trail. This attempt was not promoted to an accepted baseline.

The hosted log shows a clean build, `make build/bench_refactor_csc SPARSE_IDX_BITS=32`, the normal `-O2` Make compile flags, a successful 25-trial collector invocation, and upload of 54 files. The actual Make compile line included `-DSPARSE_IDX_BITS=32` and the Xcode 26.6 macOS SDK path. Every workflow step succeeded. The source tree was clean during collection; Day 4 notes added afterward do not change the measured commit.

## Runner and build provenance

| Field | Recorded value |
| --- | --- |
| Resolved image / version | `macos26` / `20261102.0473.1` |
| OS / architecture | `macOS-26.7.2-arm64-arm-64bit-Mach-O` / `arm64` |
| CPU | `Apple M1 (Virtual)` |
| Compiler | `/usr/bin/cc`; Apple clang `21.0.0 (clang-2100.1.1.101)`, target `arm64-apple-darwin25.6.0` |
| Build recipe | `make build/bench_refactor_csc SPARSE_IDX_BITS=32`; clean build in workflow |
| Executable SHA-256 | `22e6b76858e281dbb09a06173dab7d2437def5de1f1f9566b1144a301f7dadb0` |
| Library SHA-256 | `82aeeaea33d76e2334a44857842d6ef1cfd083048cac1dd836e8d646f15eedc8` |
| Fixture SHA-256 | `6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80` |

The runner name is an ephemeral GitHub Actions assignment and is recorded for traceability, not used as the comparison class. The image, OS/architecture, CPU, compiler, build, fixture and policy identity are the proposed class fields. The full per-trial provenance and compiler-version string remain in the raw artifact.

## Raw-bundle audit

Downloaded the artifact ZIP with `gh api`, hashed the ZIP, and inspected it as a ZIP without rewriting the member bytes. The ZIP has **54 unique members**: `run.json`, `trials.tsv`, `decision.json`, `hashes.sha256`, and 25 stdout plus 25 stderr files. `hashes.sha256` lists the other **53** files exactly once; every listed SHA-256 matched the downloaded member bytes. No absolute or parent-traversal member was present.

The `trials.tsv` ordinals are exactly `1` through `25`, in order: `1`-`5` warmup, `6`-`25` measured. Every row has the exact `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` argv, `exit_code=0`, `timed_out=False`, and `validity=valid`. Per-row stdout and stderr hashes match the raw streams. Every stderr file is empty. Every stdout file has one CSV data row with identity `bench_refactor_csc,proof,nos4.mtx,chol_spd,100,594` and a `refactor_csc_ms` equal to its `trials.tsv` value. There are **zero** missing, failed, timed-out, malformed, or silently omitted trials.

The 20 measured `refactor_csc_ms` values range from `0.026` to `0.034` ms, with median `0.029` ms. These are descriptive intake figures only. `decision.json` correctly says `pending_baseline_review`, `baseline_accepted=false`, and `freshness_result=separate`; no timing pass was emitted.

## Collection and validation path

The baseline intake workflow was added as a branch-only push job. Its first push included the prior Sprint 225 plan, Day 1-3 notes, the collector, and workflow inventory-test updates in commit `7e49ef3a`. Local workflow reader and policy-input tests passed after updating the expected four-workflow, 115-input, ten-upload inventory. The local API-docs workflow guard also passed. Those checks establish that the new upload step does not violate the local-only API-docs rule; they do not establish timing validity.

## Remaining conditions

1. Run two more independent hosted jobs against **the same exact source commit** and require the same resolved runner/compiler/build fingerprint. Do not combine a later source commit with this baseline group.
2. Recompute each job's median, MAD/median and p90/median, then the across-job spread under `s225-v1`. If any job or group is noisy, mark the attempted baseline invalid and retain its evidence.
3. Review and version all accepted baseline raw files in source control. Until then, the 90-day hosted artifact alone is intake evidence, not a durable accepted baseline.
4. Add independent validation of build recipe, source/fixture identity, raw manifest, and statistics before a candidate timing gate can be enforced. This Day 4 audit is manual and does not replace the validator planned for later days.

## Validation

Verified hosted run/job success, artifact ID/digest/expiry, all 54 ZIP members, 53 member hashes, 25 ordered trial rows, exact command/CSV identity, raw stream hashes and no invalid trials. Day 4 added documentation only after the hosted commit; no `.c` or `.h` files changed. No baseline was approved.
