# Day 5: Baseline Variance and Provenance Decision

## Decision

**Reject the attempted `s225-v1` baseline group.** Three independently executed, exact-source hosted jobs produced complete trial bundles, but their resolved runner fingerprints disagree and the locked dispersion criteria are not met. No baseline reference value, threshold, or timing pass is approved. The selected canonical report remains threshold-free; the separate timing gate remains blocked.

This is a versioned **rejected evidence set**, `s225-v1-rejected-2026-11-18`, not an accepted baseline. Original downloaded ZIP bytes are retained in [day5-rejected-baseline](day5-rejected-baseline/) in the worktree for review. They are not yet committed; the hosted artifacts for earlier rerun attempts were removed by GitHub, making the local ZIP copies the only retained byte evidence for those attempts.

## Exact-source hosted attempts

All attempts belong to workflow run [44395634285](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44395634285), source commit `7e49ef3a96e00dcc815da3ba83b006934815c242`, policy `s225-v1`, branch `sprint-225`, and `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` on macOS. `gh run rerun` kept the source commit fixed while creating new job attempts.

| Attempt | Job API ID | Artifact ID | ZIP SHA-256 | Evidence state |
| ---: | ---: | ---: | --- | --- |
| 1 | `135124640857` | `14469044864` | `3093366692075f3fedabc0735bf6425b2b23ffb7880403919f338cf6cbc100fe` | Downloaded and verified on Day 4; original hosted artifact later removed by rerun; [retained ZIP](day5-rejected-baseline/attempt-1.zip). |
| 2 | `135126631128` | Unrecorded | Unavailable | Hosted job succeeded, but rerun attempt 3 began before its artifact was downloaded. GitHub's run artifact list then returned zero artifacts. Excluded from all numerical and acceptance calculations; **do not infer its values**. |
| 3 | `135127263270` | `14470446180` | `1bdf522ce473a23c6a877527e31a9950787717cf15ca5be1f3fc20bf4cf0d8f4` | Downloaded and verified before rerun attempt 4; original hosted artifact subsequently removed; [retained ZIP](day5-rejected-baseline/attempt-3.zip). |
| 4 | `135128028384` | `14470216501` | `137ed6c27088acf3408df9f7573b258e7d9a79d834a715f4bbc2ef24cc7cb1f2` | Downloaded and verified; [retained ZIP](day5-rejected-baseline/attempt-4.zip). |

The stored ZIP hashes above matched GitHub's artifact API digests at download time. Attempt 4's artifact was configured with 90-day retention; that does not make the earlier rerun artifacts durable. The run/job IDs remain useful for provenance, while the retained ZIPs are the inspectable source for attempts 1, 3, and 4.

## Independent bundle audit

For **each** retained attempt 1, 3, and 4:

- ZIP has exactly 54 unique, safe members: `run.json`, `trials.tsv`, `decision.json`, `hashes.sha256`, and 25 pairs of raw stdout/stderr.
- The 53 paths in `hashes.sha256` exactly match the other ZIP members, and every member digest rehashes correctly.
- `trials.tsv` has ordinals 1-25 in order: five warmups then 20 measured process runs. All 25 have `exit_code=0`, `timed_out=False`, and `validity=valid`. No trial was dropped or replaced.
- Every trial's argv matches `run.json`; its raw stdout/stderr hashes match the row; its raw CSV has exactly one `bench_refactor_csc,proof,nos4.mtx,chol_spd,100` data row; and the raw `refactor_csc_ms` equals the TSV value.
- `run.json` records full source SHA `7e49ef3a96e00dcc815da3ba83b006934815c242`, clean checkout, `dry_run=false`, locked fixture SHA-256 `6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80`, build recipe, and 32-bit indices. `decision.json` says `pending_baseline_review` and `baseline_accepted=false`.

The verified raw files support the statistics below. They do **not** certify a comparable runner class or an accepted baseline.

## `s225-v1` calculation worksheet

Compute from the 20 measured decimal CSV values in each ZIP, excluding the first five warmups. For sorted values, median is the mean of positions 10 and 11; MAD is the median of absolute deviations from that median; p90 is position 18. Compare exact decimal ratios without rounding before the policy check.

| Attempt | Resolved image version | CPU model | Measured range (ms) | Median (ms) | MAD / median | p90 / median | Per-job rule |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `macos26 / 20261102.0473.1` | Apple M1 (Virtual) | `0.026`-`0.034` | `0.029` | `0.001 / 0.029 = 0.03448276` | `0.032 / 0.029 = 1.10344828` | Passes dispersion only. |
| 3 | `macos26 / 20261112.0504.1` | Apple M2 Pro (Virtual) | `0.026`-`0.037` | `0.0315` | `0.0035 / 0.0315 = 0.11111111` | `0.036 / 0.0315 = 1.14285714` | **Fails** MAD / median `<= 0.10`. |
| 4 | `macos26 / 20261112.0504.1` | Apple M1 (Virtual) | `0.028`-`0.047` | `0.033` | `0.0025 / 0.033 = 0.07575758` | `0.039 / 0.033 = 1.18181818` | Passes dispersion only. |

The per-job p90 values and ratios are shown for transparency; all three are below `1.50`. Attempt 3 fails the MAD rule regardless of its CPU mismatch. The three medians are `0.029`, `0.0315`, and `0.033` ms. Their median is `0.0315` ms, but the cross-job spread is `(0.033 - 0.029) / 0.0315 = 0.1269841269...`, above the locked `0.10` limit. **That median is a rejected-group diagnostic, not a baseline reference.**

Runner class also fails independently. Attempt 1 and attempts 3/4 have different image versions; attempt 3 has a different CPU model from attempts 1/4. The benchmark executable SHA-256 matched across all three attempts (`22e6b76858e281dbb09a06173dab7d2437def5de1f1f9566b1144a301f7dadb0`), but library archive hashes differed. The archive-hash discrepancy needs investigation before any binary-provenance claim; identical executable bytes alone do not waive runner-class or variance requirements.

## Implications and next collection rule

The current `macos-latest` label did not deliver one stable `s225-v1` fingerprint in these attempts. Do not pool the measurements by label, drop the noisy job, change the threshold, or accept only the favorable two runs. A viable next collection needs either a genuinely pinned runner/CPU class or a **new, reviewed policy version** with a same-machine comparison protocol. Changing the class definition, dispersion limit, or trial schedule requires fresh baseline evidence under that version. The existing `s225-v1` numbers may guide design review but cannot be retroactively approved.

The rerun artifact deletion is also an operational issue. For any future repeated collection, download and verify each artifact **before** rerunning, then retain reviewed raw bytes in source control or another durable store. A successful job conclusion without its raw bundle, as in attempt 2, is insufficient.

## Validation

The independent ZIP/TSV/raw-CSV audit and decimal worksheet were run against the three retained ZIPs. No C or header files changed. The Day 5 output is a rejected baseline decision and evidence preservation, not a timing-gate implementation or a performance claim.
