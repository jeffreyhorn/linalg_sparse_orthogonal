# Day 9: Workload and Baseline Drift Rejections

## Decision

The timing decision now binds an official candidate bundle to the checkout and build present at decision time. The hosted workflow passes `--verify-checkout` and the exact checkout commit to `scripts/check_refactor_csc_timing.py`. The validator compares SHA-256 for `nos4.mtx`, `bench_refactor_csc.c`, `Makefile`, `build/bench_refactor_csc`, and `build/libsparse_lu_ortho.a` with the independently hashed candidate receipt. It also requires the Make index-width stamp to say `32`. A changed file after collection or a claimed executable hash unrelated to the tested binary fails with `S225_CANDIDATE_IDENTITY`. The locked fixture digest is checked separately.

This extends the Day 7 comparison of baseline/candidate metadata. The baseline source commit, runner/compiler class, fixture, benchmark source, Makefile digest, argv, backend/thread environment, serial build mode and index width must still match. A candidate source commit can differ from the baseline's source commit; it must match the exact candidate checkout specified by CI. The real branch has no accepted baseline, so these tests use synthetic accepted bundles only and do not constitute a timing performance pass.

## Expected versus observed fixture ledger

All observations below came from `python3 tests/test_check_refactor_csc_timing.py`; mutations were rehashed so a hash-manifest error could not hide the intended rejection. The positive fixture uses three synthetic baseline jobs, one synthetic candidate, `0.100` ms measured values, the locked `nos4.mtx` bytes, and a matching synthetic checkout.

| Fixture or mutation | Expected rule | Observed rule |
| --- | --- | --- |
| Canonical synthetic baseline, candidate, and checkout | `S225_TIMING_PASS` | `S225_TIMING_PASS` |
| Candidate fixture digest changed | `S225_CANDIDATE_IDENTITY` | `S225_CANDIDATE_IDENTITY` |
| Candidate benchmark-source or Makefile digest changed | `S225_CANDIDATE_IDENTITY` | `S225_CANDIDATE_IDENTITY` |
| Candidate command changed; repeat count `1` changed to `2` | `S225_CANDIDATE_IDENTITY` | `S225_CANDIDATE_IDENTITY` |
| Candidate backend override or parallel build mode | `S225_CANDIDATE_IDENTITY` | `S225_CANDIDATE_IDENTITY` |
| Candidate CPU or compiler version changed | `S225_CANDIDATE_IDENTITY` | `S225_CANDIDATE_IDENTITY` |
| Candidate host set to `unknown` | `S225_CANDIDATE_IDENTITY` | `S225_CANDIDATE_IDENTITY` |
| Checkout fixture, benchmark executable, library archive, benchmark source, or Makefile bytes changed after collection | `S225_CANDIDATE_IDENTITY` | `S225_CANDIDATE_IDENTITY` |
| Checkout source commit differs from candidate receipt; width stamp changed to `64` | `S225_CANDIDATE_IDENTITY` | `S225_CANDIDATE_IDENTITY` |
| Active pointer, baseline manifest, or baseline raw ZIP missing | `S225_BASELINE_MISSING` | `S225_BASELINE_MISSING` |
| Baseline expired, superseded, malformed schema, or wrong policy version | `S225_BASELINE_INVALID` | `S225_BASELINE_INVALID` |
| Active-pointer manifest digest or baseline source commit disagrees | `S225_BASELINE_INVALID` | `S225_BASELINE_INVALID` |

The earlier Day 7 tests still reject a corrupted raw stream, invalid metric, failed trial, timeout, noisy candidate, over-threshold median, missing reviewer approval, and a forged accepted pointer around a rejected Day 5 ZIP. Baseline preflight remains blocked on the actual branch with `S225_BASELINE_MISSING`.

## Verification and limit

Python compilation and all 14 timing-validator test methods passed. `git diff --check` found no whitespace errors. The hosted workflow edit adds one argument to its candidate decision; its separate timing status and artifact route are unchanged. No `.c` or `.h` files changed, so the C format/lint/test sequence is not triggered.

This file-to-receipt comparison proves consistency at decision time. It does not by itself prove how the compiler produced the candidate executable; the clean Make command and build log remain the hosted build evidence, and actual candidate build execution is still blocked by the absent accepted baseline. Day 10 covers weak-sample and claim-boundary mutations.
