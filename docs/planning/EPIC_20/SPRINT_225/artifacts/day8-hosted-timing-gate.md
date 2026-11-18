# Day 8: Hosted Canonical Timing Gate

## Separate job and current disposition

Added `.github/workflows/sprint225-timing.yml` with the independent check `macOS enforced refactor CSC timing decision (s225-v1)`. It runs on pushes to `sprint-225` or `master` and pull requests to `master`, on `macos-latest`, with a 20-minute job cap. The selected benchmark freshness jobs and their threshold-free `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and `variance=not_computed_single_sample` fields are unchanged. A freshness pass cannot satisfy the timing job.

The first step invokes `scripts/check_refactor_csc_timing.py --preflight`. It verifies the active baseline pointer, reviewed manifest, and three raw bundles before any candidate build. The current branch has no accepted baseline; preflight writes `build/sprint225-timing-decision.json` with `status=blocked`, `primary_rule=S225_BASELINE_MISSING`, and exits 2. The job therefore fails and uploads that explicit receipt. It does not present a skipped or green timing result. The rejected Day 5 ZIPs remain outside the active-baseline path.

Once an independently reviewed baseline exists, a successful preflight returns `S225_BASELINE_READY`, which is only permission to collect a candidate and is not a timing pass. The job then runs clean serial `make build/bench_refactor_csc SPARSE_IDX_BITS=32`, captures the build log, collects 5 warmup and 20 measured candidate processes with `--role candidate`, and invokes the raw-evidence validator with the exact checkout commit. The validator's nonzero exit fails the job. It also runs after a build or collection failure when preflight succeeded, so the artifact includes a failure decision. The collector preserves raw streams and attempted trial records for failed or timed-out processes. A partial bundle is retained for diagnosis and cannot pass validation.

The upload step uses `if: always()` and `if-no-files-found: error` for the candidate directory, decision JSON, and build log. Its artifact name is `sprint225-timing-<run_id>-<run_attempt>` with requested 90-day retention. If preflight fails, the decision JSON alone supplies the required clear failure receipt. Upload does not turn a failing decision into a pass. An infrastructure timeout can still interrupt upload; the 20-minute cap is the locked upper bound, and that case remains a failed hosted check.

## Commands and rerun procedure

Local preflight and decision checks:

```sh
python3 tests/test_check_refactor_csc_timing.py
python3 scripts/check_refactor_csc_timing.py --preflight --output /tmp/sprint225-preflight.json
python3 scripts/check_refactor_csc_timing.py \
  --candidate build/sprint225-candidate \
  --expected-candidate-commit "$(git rev-parse HEAD)" \
  --output build/sprint225-timing-decision.json
```

The second command currently exits 2 and records `S225_BASELINE_MISSING`. The candidate decision command is for a complete official hosted bundle; a local dry run cannot pass. The hosted job executes on push or pull request, so after a valid accepted baseline is committed, push that commit and inspect the new timing check. To repeat an infrastructure attempt on the **same** checkout, use `gh run rerun <run-id>` and inspect the new attempt separately. To change the baseline, runner, policy, or source, use a new commit and new run; never overwrite an old attempt's raw data. Retrieve evidence with `gh run download <run-id> -n sprint225-timing-<run-id>-<run-attempt>` and verify `hashes.sha256` and the uploaded report before drawing any timing conclusion. An upload URL or green selected-freshness job is insufficient.

## Verification and remaining gate condition

Local Python compilation and 11 timing-validator tests passed. Tests include missing-baseline preflight CLI output and a synthetic accepted-baseline preflight that reports `ready`, never `pass`. The structured workflow inventory now counts the additional six steps and one artifact upload; its eight tests and 50 policy tests passed, and `scripts/check_api_docs_local_only.sh` accepted all 121 workflow records. No `.c` or `.h` files changed; the C format/lint/test sequence is not triggered.

The first hosted push of commit `56cd9fc34009d75e39878d9cc5e0e721e34bac4a` ran [workflow run 44400684440](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44400684440), job `135142533000`. Its baseline preflight failed with exit 2 and `S225_BASELINE_MISSING`; the build, collection, and candidate decision steps were skipped, while the upload step passed. Artifact `sprint225-timing-44400684440-1` has ID `14471524335`. The downloaded ZIP SHA-256 is `9834e61dbdcbd6d263cad6bf481ac172de0d12b543da73e1e3af62abc5f4f361`, matching the GitHub artifact digest. The ZIP contains exactly one file, `sprint225-timing-decision.json`, whose `status=blocked`, `primary_rule=S225_BASELINE_MISSING`, `policy_version=s225-v1`, and null reference/threshold fields match the intended no-baseline result.

An accepted three-job baseline and independent verification of build/archive provenance are still missing, so this gate has no passing performance result. The Day 8 implementation provides a separate, fail-closed CI status and retains the evidence available on each attempt.
