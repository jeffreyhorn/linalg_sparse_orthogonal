# Day 6: Baseline Disposition and Timing Gate Design

## Baseline acceptance disposition

**No accepted baseline exists for `s225-v1`.** The [Day 5 worksheet](day5-baseline-variance-decision.md) and its three [retained raw ZIPs](day5-rejected-baseline/) are the complete review record for the attempted collection. Attempts 1, 3, and 4 each have 5 warmups, 20 valid measured trials, 54 ZIP members, and 53 verified member hashes. They cannot form one baseline: their resolved macOS image versions and CPU models differ, attempt 3 has MAD/median `0.111111... > 0.10`, and the three medians have spread `0.126984... > 0.10`. Attempt 2 has no retained artifact and is excluded. The diagnostic median `0.0315` ms is **not** a reference value.

Item 225.2's planned 28-hour allocation is accounted for through intake, repeated collection, independent audit, and this rejection decision. Its product outcome remains **blocked**: no baseline version is approved, no active-baseline pointer exists, and no timing comparison may pass. A new collection needs three compatible jobs under the locked class. A change to the class definition, counts, or limits requires a new reviewed policy version and fresh data; this design does not retroactively relax `s225-v1`.

## Separate authorities

| Surface | Authority | Required behavior |
| --- | --- | --- |
| Selected canonical freshness | Existing selected-target manifest, `scripts/bench_canonical_report.sh`, and `scripts/check_bench_canonical_freshness.py` | Continue to report `measurement`, `baseline=n/a`, `threshold=n/a`, `none_configured` warmup and `not_computed_single_sample` variance. A freshness pass is neither an input nor a fallback for the timing decision. |
| Timing policy | [Day 3 `s225-v1` lock](day3-methodology-lock-and-collector.md) | Own one workload, runner-class equality rule, 5+20 trial schedule, 30-second process timeout, 20-minute job cap, statistics, and numerical limits. |
| Raw collection | `scripts/collect_refactor_csc_timing.py` and a future candidate-role extension | Capture byte-preserving streams, ordinal/phase/status and hashes. Collection status is not timing acceptance. |
| Accepted baseline | A future source-controlled `benchmarks/baselines/refactor-csc-nos4/s225-v1/<baseline-id>/` directory plus an explicit active-baseline pointer | Exist only after independent verification and review of three compatible hosted jobs; rejected Day 5 ZIPs remain outside this loader path. |
| Candidate decision | Future `scripts/check_refactor_csc_timing.py` (Day 7) | Verify baseline and candidate from raw evidence, recompute statistics, produce a distinct timing decision and nonzero status on missing evidence or failed rule. |
| Hosted CI | Future separate timing job (Day 8) | Build, collect, validate and upload all raw candidate data even on failure. Do not change the threshold-free freshness jobs or their selected report fields. |

## Accepted-baseline input contract

The timing validator accepts **one explicit baseline directory or active-baseline pointer**, never a search through arbitrary artifacts or the newest-looking ZIP. Proposed pointer: `benchmarks/baselines/refactor-csc-nos4/active.json`, containing `baseline_id`, `policy_version`, `baseline_manifest_sha256`, and `status=accepted`. The pointer does not exist while baseline collection is blocked.

An accepted baseline directory must contain a manifest and all **three** original raw job bundles. The manifest records schema/policy version, selected target, metric name/units, exact fixture digest, baseline source SHA, `benchmarks/bench_refactor_csc.c` digest, Makefile/build-recipe digest, full runner/compiler/build fingerprint, three distinct hosted run/attempt/job identities, timestamps and raw bundle SHA-256 values, 5+20 trial counts per job, per-job medians/MAD/p90, between-job spread, reference median, `accepted_at_utc`, `expires_at_utc` (30 days after collection under the reviewed rule), and reviewer decision. The manifest must be hashed by the pointer. No blank, `unknown`, `n/a`, dirty, local or dry-run provenance is allowed in required fields.

The validator must verify bytes rather than trust manifest statistics: reject unsafe or duplicate ZIP paths; require exact file membership and the 53-member hash list per job; recompute raw stream hashes and parsed `refactor_csc_ms` for all 25 ordinals; require five warmups and 20 measured values; recompute each job's median, MAD/median and nearest-rank p90/median using decimal arithmetic; and recompute the three-job spread/reference. All three jobs must use the same baseline source commit, workload, policy, image/version, OS/arch, CPU model, compiler path/full version, Make recipe and index width. The ephemeral runner name and workflow run/attempt ID are provenance, not class equality keys. Record library and executable hashes; for one baseline source/build, executable hashes must agree. Archive hashes currently differ across Day 5 runs and need investigation before using them as a byte-equality key. A candidate library/executable hash may differ because candidate source changes, while the measurement harness and build recipe must remain compatible.

The source-controlled manifest must name each hosted ZIP digest and the API upload digest or a separately retained upload receipt. A URL alone cannot substitute for a downloaded, verified bundle. If any of the three bundles is missing, mismatched, too old, from a different class, or above a dispersion limit, the baseline is unavailable and the CI timing job must fail closed. Day 5's `s225-v1-rejected-2026-11-18` set cannot be promoted by changing its directory or adding an `accepted` flag.

## Candidate input contract

The candidate collector must add `role=candidate` while preserving Day 3's raw byte and trial format. It runs on one exact candidate checkout with five warmups and 20 measured processes. It records full source SHA, clean state, hosted run ID/attempt/job key, resolved runner/compiler/build fingerprint, fixture and measurement-harness digests, executable/library digests, exact argv, timeout, all 25 trial streams, and `trials.tsv`. The validator must verify the candidate's raw ZIP or directory and hash manifest from scratch; `decision.json` emitted by the collector is a pending collection receipt, not an authorization to pass. A separate `timing-decision.json` must contain the independently calculated result so the raw bundle remains immutable.

Identity comparisons are exact for policy version, selected target/fixture SHA, argv, metric/units, 5+20 schedule, timeout, benchmark source digest, Make recipe and flags/index width, resolved image/version, OS/arch, CPU model, compiler path/full version, thread/backend environment, and the supported build mode. Baseline and candidate **source SHA and executable SHA are expected to differ** across code changes; each is recorded and verified within its own bundle. A changed benchmark timing implementation, fixture or build recipe requires a new baseline. A candidate from an unknown host, a dirty checkout, an unrecognized support tier, or a different runner class is rejected before numerical comparison. The validator checks baseline age against candidate collection time, using UTC timestamps with explicit timezone information.

Each candidate trial must have one ordinal in order, correct phase, exact argv, both raw streams, matching hashes, a zero exit, no timeout, a single selected CSV row, and a positive finite decimal metric. Reject a missing or bad **warmup** as well as a bad measured trial. No retry, omitted sample, outlier deletion, or substitution is allowed. External process wall time is diagnostic only; the selected metric remains the benchmark's `refactor_csc_ms` interval. Validate all numerical field strings before Decimal arithmetic and report a rule code rather than a traceback.

## Evaluation order and diagnostics

The first failure in this order is the primary result; record all safe-to-compute secondary findings in the report without changing precedence. Errors are classified as `blocked` for unavailable/incompatible evidence, `invalid` for malformed collection, `unstable` for excessive dispersion, and `regression` for a valid over-threshold candidate. All four are **non-pass** CI outcomes.

| Order | Rule code | Condition |
| ---: | --- | --- |
| 1 | `S225_BASELINE_MISSING` | No active pointer, manifest, or one of three required raw bundles. This is the current branch state. |
| 2 | `S225_BASELINE_INVALID` | Pointer/manifest/hash mismatch, unapproved status, expired age, bad provenance, unequal baseline class/source/build, bad trial count, or failed baseline dispersion. |
| 3 | `S225_CANDIDATE_MISSING` | Candidate bundle or required raw member absent. |
| 4 | `S225_CANDIDATE_IDENTITY` | Candidate source/run identity, workload, metric, build recipe, runner class or required environment differs from policy/baseline. |
| 5 | `S225_CANDIDATE_INVALID` | Duplicate/extra/missing member or ordinal, mismatched hash, failed/timed-out trial, wrong CSV row, or nonfinite/nonpositive metric. |
| 6 | `S225_CANDIDATE_DISPERSION` | Candidate MAD/median `> 0.10` or p90/median `> 1.50`. |
| 7 | `S225_TIMING_REGRESSION` | Candidate median `> 1.25 * accepted reference median`. |
| 8 | `S225_TIMING_PASS` | Every previous rule passed; one-cell median and dispersion are within limits. |

Compare exact decimal values with `<=`; no rounding before a decision. The candidate median uses positions 10/11, MAD uses the median absolute deviation, and nearest-rank p90 uses position 18. Baseline reference is the median of three independently valid job medians. A positive but noisy candidate is `unstable`, not a performance regression. A missing baseline does **not** become a skip or pass.

## Decision report and CI behavior

Proposed `timing-decision.json` fields: `schema_version`, `policy_version`, `target_id`, `status`, `primary_rule`, `secondary_rules`, `baseline_id`, `baseline_manifest_sha256`, three baseline ZIP digests, candidate bundle digest, candidate source/run/job identity, fixture and runner/build fingerprint, `warmup_count`, `measured_count`, `invalid_count`, all raw trial hashes or a reference to `hashes.sha256`, per-job baseline statistics, candidate median/MAD/p90 and exact ratios, reference median, allowed ceiling, collection/expiry timestamps, and `freshness_result="separate"`. Decimal values are strings, never binary floats. On a missing input, use explicit `null` fields and a `blocked` status rather than inventing statistics. Write the report before returning an error. Exit `0` only for `S225_TIMING_PASS`, `1` for valid threshold/dispersion failure, and `2` for missing/invalid evidence. Keep collector failures nonzero as well.

The future CI timing job must use the locked 20-minute cap, clean serial Make build, candidate collector, validator, and an `if: always()` upload of raw candidate files **plus** `timing-decision.json` or a preflight failure receipt. The upload name must include run ID and attempt; retention target is 90 days, subject to repository policy. Downloaded artifact digest and post-upload ID belong in a separate receipt because they cannot be present before upload. Never edit the hashed raw bundle after deciding. The CI check name and report artifact must be distinct from `hosted-performance-freshness`, `selected-performance-freshness`, and the local S6 sentinel. While `active.json` is absent, the timing job must report `S225_BASELINE_MISSING` and fail; it must not claim an enforced passing gate.

## Day 7-8 implementation handoff

1. Implement baseline loader and raw verifier, then candidate verifier/statistic calculation in a separate timing validator. Start with a missing-baseline fixture that yields `S225_BASELINE_MISSING`, and a synthetic accepted baseline for decision-unit tests; the rejected Day 5 ZIPs must never pass as accepted evidence.
2. Extend the collector with candidate role, safe failure receipts, and machine-readable build inputs; verify Make flags from actual build evidence rather than trusting the current recipe label.
3. Add one separate candidate CI job and upload route only when the validator fails closed. Preserve existing threshold-free report and freshness tests.
4. Investigate archive-hash differences and choose a verifiable library provenance rule. Recollection on a genuinely stable runner class is still required before any threshold can pass.

## Validation

Day 6 changed planning documentation only. Checked that the design never maps a selected freshness pass, local S6 smoke result, or rejected Day 5 statistic to `S225_TIMING_PASS`. No `.c` or `.h` files changed; Markdown link and whitespace checks close this day's artifact.
