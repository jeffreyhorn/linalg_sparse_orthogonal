# Sprint 192 Retrospective

**Sprint:** 192 - Methodology-Bound Performance Evidence Lane
**Duration:** 14 days (Days 1-14 landed on branch `sprint-192`)
**Status:** Complete; one threshold-free hosted selected performance evidence
lane landed with explicit limits and residuals documented

## Source Artifact Note

Sprint 192 was executed from the Epic 17 project-plan section for Sprint 192
and lives under `docs/planning/EPIC_17/SPRINT_192/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 192 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Selected exactly one benchmark lane:
      `SRT-BENCH-REFACTOR-CSC-NOS4`, covering `bench_refactor_csc` on
      `tests/data/suitesparse/nos4.mtx --repeat 1`.
- [x] Confirmed the selected lane remains methodology-bound evidence, not a
      timing threshold, speedup, portability, release, or state-of-the-art
      claim.
- [x] Retained the threshold-free policy with `status=measurement`,
      `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and
      `variance=not_computed_single_sample`.
- [x] Narrowed hosted CI publication to the exact selected bundle:
      `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`.
- [x] Added workflow guards for hosted timeout, selected metadata, upload
      artifact name, retention, exact upload paths, broad upload rejection,
      unselected CSV rejection, and missing-path drift.
- [x] Hardened `scripts/check_bench_canonical_freshness.py` so the selected
      CSV content must match the selected target contract, not just exist.
- [x] Preserved selected benchmark methodology fields through normalized
      benchmark report-index rows.
- [x] Added regression coverage for selected CSV content, missing benchmark
      artifacts, workflow drift, docs non-claims, and normalized methodology
      metadata.
- [x] Updated maintainer, corpus, and report-index schema documentation to
      describe the selected lane and retained non-claims consistently.
- [x] Ran the final focused validation set and recorded that no `.c` or `.h`
      files changed, so the full C quality gate was not required.

## What Went Well

1. **The sprint stayed selected and bounded.** The work promoted one existing
   selected benchmark target instead of expanding the benchmark surface or
   creating partial evidence across several lanes.

2. **Publication scope became easier to review.** Hosted CI now uploads only
   the selected CSV plus `index.tsv` and `manifest.txt`, which keeps the
   hosted evidence bundle aligned with the manifest-required artifact list.

3. **Artifact meaning is now checked.** The selected freshness checker validates
   the selected CSV row contents for benchmark name, matrix, size, scenario,
   and LDLT backend fields, closing the gap where a present CSV could still
   carry the wrong measurement row.

4. **The normalizer stayed advisory.** `normalize_report_index.py` preserves
   methodology metadata for benchmark rows without becoming the hard owner of
   selected performance policy. The dedicated freshness checker remains the
   enforcement boundary.

5. **Documentation became executable.** The selected-performance docs guard
   verifies required non-claim markers and rejects broad performance,
   portability, release, and state-of-the-art wording.

6. **Threshold-free semantics stayed explicit.** The sprint did not introduce a
   hosted timing threshold without a reviewed baseline, variance model, and
   machine-class policy.

## What Didn't Go Well

1. **The lane is still statistically thin.** The selected hosted row remains a
   single repeat with no warmup and no computed variance. That is acceptable
   for methodology-bound freshness evidence but not enough for performance
   claims.

2. **Benchmark evidence is split between local generation and hosted
   publication.** The canonical generator still emits additional local CSVs,
   while hosted CI publishes only the selected bundle. The distinction is now
   guarded, but future edits must keep it intact.

3. **The policy touches several surfaces.** Workflow YAML, the freshness
   checker, normalizer behavior, manifest/schema docs, maintainer docs, corpus
   docs, and tests all need to move together when the selected lane changes.

4. **No hosted threshold was added.** This was intentional, but it leaves
   regression timing ownership for a later sprint with stronger baseline and
   variance evidence.

5. **Hosted scope is Linux-only.** Sprint 192 did not add Windows or macOS
   selected benchmark freshness claims.

## Final Metrics

### Validation

| Metric | Sprint 192 close state |
| --- | --- |
| canonical benchmark freshness | passed through `make bench-canonical-report-freshness` |
| selected-performance docs guard | passed |
| selected workflow guard | passed |
| selected report target schema validation | passed |
| normalized benchmark report-index freshness | passed |
| selected benchmark freshness regression tests | passed |
| report-index normalization regression tests | passed |
| Python syntax compilation | passed |
| final `git diff --check` | passed |
| generated report output status | ignored under `build/` |
| full C quality gate | not required; no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 192 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 13 |
| Sprint closeout artifacts added | 1 |
| Sprint retrospective files added | 1 |
| Workflow files changed | 1 |
| Python scripts changed | 1 |
| Python tests changed | 4 |
| Maintainer/corpus/schema docs changed | 3 |
| C source files changed | 0 |
| Public header files changed | 0 |
| Hosted selected benchmark targets added | 0 |
| Hosted selected benchmark targets hardened | 1 |
| Hosted uploaded selected artifacts | 3 |

### Claim Governance

| Metric | Sprint 192 close state |
| --- | ---: |
| methodology-bound hosted selected performance lane claims hardened | 1 |
| hosted timing threshold claims added | 0 |
| portable performance claims added | 0 |
| release benchmark claims added | 0 |
| algorithmic superiority claims added | 0 |
| platform parity claims added | 0 |
| package-manager proof claims added | 0 |
| shared-library ABI claims added | 0 |
| external-library parity claims added | 0 |
| OpenMP speedup claims added | 0 |
| backend superiority claims added | 0 |
| state-of-the-art performance claims added | 0 |

## Closed Claim

Sprint 192 closes this bounded implementation claim:

One methodology-bound hosted selected performance evidence lane is now hardened
for `SRT-BENCH-REFACTOR-CSC-NOS4`, using `bench_refactor_csc` on
`tests/data/suitesparse/nos4.mtx --repeat 1`, exact hosted artifact upload
scope, selected CSV content validation, manifest and report-index methodology
metadata, threshold-free policy, workflow drift guards, documentation
non-claim guards, and final local validation.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day2-candidate-benchmark-lane-audit.md](./artifacts/day2-candidate-benchmark-lane-audit.md);
- [day3-methodology-contract.md](./artifacts/day3-methodology-contract.md);
- [day4-generator-fixture-alignment.md](./artifacts/day4-generator-fixture-alignment.md);
- [day5-methodology-metadata-hardening.md](./artifacts/day5-methodology-metadata-hardening.md);
- [day6-report-index-normalization.md](./artifacts/day6-report-index-normalization.md);
- [day7-hosted-lane-design.md](./artifacts/day7-hosted-lane-design.md);
- [day8-hosted-lane-implementation.md](./artifacts/day8-hosted-lane-implementation.md);
- [day9-regression-policy-decision.md](./artifacts/day9-regression-policy-decision.md);
- [day10-claim-calibration.md](./artifacts/day10-claim-calibration.md);
- [day11-failure-and-drift-coverage.md](./artifacts/day11-failure-and-drift-coverage.md);
- [day12-integrated-local-validation.md](./artifacts/day12-integrated-local-validation.md);
- [day13-review-surface-audit.md](./artifacts/day13-review-surface-audit.md);
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md).

No portable performance, hosted threshold, release benchmark, algorithmic
superiority, platform parity, package-manager proof, shared-library ABI proof,
external-library parity, OpenMP speedup, backend superiority, or
state-of-the-art performance claim was added.

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Hosted timing threshold | Future performance-governance owner | Add a reviewed baseline, variance model, machine-class policy, flake budget, and failure wording before any hosted timing threshold can become a gate. |
| Portable performance evidence | Future benchmark methodology owner | Add multi-platform, multi-machine, repeated, variance-aware evidence with explicit compiler, CPU, threading, and backend context. |
| Windows/macOS selected benchmark freshness | Platform CI owner | Add hosted platform evidence and selected artifact validation without broadening current Linux selected claims. |
| Unselected canonical CSV publication | Benchmark publication owner | Keep unselected CSVs local-only unless a future sprint selects, documents, and guards each promoted row. |
| Release benchmark claim | Release engineering owner | Define release benchmark fixtures, reproducible environments, archived artifacts, and acceptance criteria separately from Sprint 192 freshness evidence. |

## Next-Sprint Readiness

Sprint 193 can build from a hardened selected-performance evidence pattern
without inheriting an accidental timing claim.

| Future need | Sprint 192 handoff |
| --- | --- |
| Additional hosted performance lane | Select exactly one target, keep artifact uploads exact, validate CSV content, preserve methodology metadata, and update docs guards before publication. |
| Hosted threshold policy | Start from Day 9's threshold-free decision and add baseline/variance evidence before changing `baseline=n/a` or `threshold=n/a`. |
| Report-index changes | Keep normalized benchmark rows advisory unless the dedicated freshness checker intentionally delegates policy ownership. |
| Documentation changes | Keep selected-performance wording paired with threshold-free, non-portable, non-release, and non-state-of-the-art markers. |
| Workflow changes | Preserve the three-file hosted selected bundle unless a future sprint updates the manifest, docs, tests, and claim boundary together. |

## Validation Retrospective

The final validation set was sufficient for Sprint 192 because the sprint
changed workflow, Python, tests, and documentation, but no C source or public
header files. The focused checks exercised the selected benchmark freshness
path, selected workflow guard, docs guard, corpus schema, normalized benchmark
index, Python syntax, and whitespace hygiene.
