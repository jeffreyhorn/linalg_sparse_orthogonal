# Day 10: Weak Evidence and Claim Boundary Rejections

## Decision

The Sprint 225 timing validator rejects inadequate samples before computing a timing result, and keeps invalid, unstable, regressed, and missing-input outcomes distinct. It now classifies a collector receipt with fewer than 25 trials as `S225_CANDIDATE_INVALID`. A missing raw warmup member is `S225_CANDIDATE_MISSING`. The decision remains separate from selected benchmark freshness.

The machine-readable decision uses report schema 2. It includes the fixed scope `one_canonical_refactor_csc_nos4_cell_on_matched_macos_runner` and explicit non-claims for portable speed, cross-platform timing parity, external-library ranking, and state-of-the-art status. `scripts/verify_refactor_csc_timing_report.py` recomputes a retained report from the accepted baseline and raw candidate bundle. Any changed claim field, freshness-derived pass, missing run provenance, absent raw-hash reference, or missing candidate evidence disagrees with the recomputation and is rejected as `S225_REPORT_INVALID`. The verifier is an evidence-review tool; a report alone is not proof of an uploaded artifact.

The workflow mutation suite checks the separate macOS job, 20-minute cap, baseline preflight, clean pinned build, candidate collector, raw decision with checkout verification, and `if: always()` artifact upload. Removing the upload, removing raw candidate files from its path, making upload success-only, swallowing a decision error, substituting selected freshness for the timing decision, or dropping master coverage fails the suite. The test is registered in the Linux and macOS structured-workflow checks. A real upload failure remains a failed CI job; successful upload and artifact bytes must still be inspected independently.

## Expected versus observed fixture ledger

The fixtures use an explicitly synthetic accepted baseline. Every candidate failure is checked against the expected rule and exit code. The test snapshots available raw candidate files before and after validation to verify that the checker does not edit or discard them.

| Fixture | Expected rule and exit | Observed rule and exit | Raw evidence |
| --- | --- | --- | --- |
| One measured sample with receipt count `6` | `S225_CANDIDATE_INVALID`, 2 | `S225_CANDIDATE_INVALID`, 2 | Preserved |
| One measured TSV row omitted | `S225_CANDIDATE_INVALID`, 2 | `S225_CANDIDATE_INVALID`, 2 | Preserved |
| First warmup stdout removed | `S225_CANDIDATE_MISSING`, 2 | `S225_CANDIDATE_MISSING`, 2 | Remaining files preserved |
| First warmup marked timed out | `S225_CANDIDATE_INVALID`, 2 | `S225_CANDIDATE_INVALID`, 2 | Preserved |
| Malformed trial UTC start time | `S225_CANDIDATE_INVALID`, 2 | `S225_CANDIDATE_INVALID`, 2 | Preserved |
| Measured raw stdout changed without hash update | `S225_CANDIDATE_INVALID`, 2 | `S225_CANDIDATE_INVALID`, 2 | Preserved |
| Uniform candidate at exact `0.125` ms ceiling for `0.100` ms reference | `S225_TIMING_PASS`, 0 | `S225_TIMING_PASS`, 0 | Bundle unchanged |
| Uniform candidate at `0.126` ms | `S225_TIMING_REGRESSION`, 1 | `S225_TIMING_REGRESSION`, 1 | Bundle unchanged |
| Ten `0.100` and ten `0.200` ms samples | `S225_CANDIDATE_DISPERSION`, 1 | `S225_CANDIDATE_DISPERSION`, 1 | Bundle unchanged |
| Pass report with altered freshness, portable-speed scope, missing parity non-claims, raw-hash reference, or hosted run ID | `S225_REPORT_INVALID`, 2 | `S225_REPORT_INVALID`, 2 | Recomputed from unchanged bundle |
| Pass report with candidate bundle missing at review time | `S225_REPORT_INVALID`, 2 | `S225_REPORT_INVALID`, 2 | Missing input detected |
| Timing workflow without required raw artifact upload or with freshness substituted | Workflow check failure | Workflow check failure | Upload path guarded |

The earlier Day 7 and Day 9 positive fixtures still pass schema, identity, exact-statistic, and checkout checks. This synthetic `S225_TIMING_PASS` is a unit-test oracle only; the real branch still has no accepted baseline and its hosted timing preflight fails with `S225_BASELINE_MISSING`.

## Verification and open condition

`python3 tests/test_check_refactor_csc_timing.py` passed all 17 methods. `tests/test_sprint225_timing_workflow.py` passed both methods and their workflow mutations. Python compilation, eight structured-workflow inventory tests, 50 structured-policy tests, `scripts/check_api_docs_local_only.sh`, and `git diff --check` passed. No `.c` or `.h` files changed; the C format/lint/test sequence is not triggered.

The remaining two planned hours of item 225.4 belong to Day 11's cross-check of the complete negative matrix. The hosted artifact and candidate result remain unverified until a compatible three-job baseline is reviewed and accepted. No numerical limit or claim boundary was relaxed in response to these tests.
