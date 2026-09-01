# Sprint 191 Retrospective

**Sprint:** 191 - Bounded External Comparison Family
**Duration:** 14 days (Days 1-14 landed on branch `sprint-191`)
**Status:** Complete; one additional bounded local-only comparison family
landed with residuals documented

## Source Artifact Note

Sprint 191 was executed from the Epic 17 project-plan section for Sprint 191
and lives under `docs/planning/EPIC_17/SPRINT_191/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 191 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited candidate comparison families and selected exactly one bounded
      family: `qr-incompatible-ls`.
- [x] Defined the deterministic
      `qr_overdetermined_incompatible_4x2` fixture contract, expected solution,
      expected nonzero residual, metric rows, tolerances, and claim boundary.
- [x] Reused the source-controlled dense QR reference helper without adding a
      package-manager or optional package dependency claim.
- [x] Added runner support for `qr-incompatible-ls`, including nonzero expected
      residual handling for observation and study rows.
- [x] Added generated project, baseline, dependency, study, summary, and
      manifest artifacts under `build/comparison/qr_incompatible_ls/`.
- [x] Added `qr_incompatible_ls` report-family metadata and
      `SRT-COMP-QR-INCOMPATIBLE-LS` selected-target metadata with six exact
      expected generated row IDs.
- [x] Integrated the target into `make report-index-comparison-freshness`.
- [x] Added target-specific freshness diagnostics and regression coverage for
      missing artifacts, Windows-style artifact paths, stale rows, failed rows,
      and dependency-only rows.
- [x] Added Linux and macOS workflow artifact upload paths for the six exact QR
      incompatible comparison files.
- [x] Left Windows selected comparison metadata unchanged because no MSVC proof
      was added for this QR incompatible target.
- [x] Updated public, maintainer, solver, corpus, and schema documentation to
      describe six selected local comparison families without broadening QR,
      least-squares, ecosystem, package, platform, ABI, performance, release,
      or state-of-the-art claims.
- [x] Ran the final focused validation set and recorded that no `.c` or `.h`
      files changed, so the full C quality gate was not required.

## What Went Well

1. **The sprint stayed bounded.** The work added one target, one fixture, one
   subfamily, one artifact directory, and six selected rows instead of
   spreading partial evidence across multiple comparison families.

2. **The fixture added real evidence value.** The selected fixture covers an
   intentionally incompatible least-squares solve with a nonzero expected
   residual, which was not covered by the existing QR minimum-norm and
   compatible least-squares rows.

3. **Residual semantics became explicit.** The runner now supports
   target-level `expected_residual_norm`, so intentionally nonzero residuals
   are checked as valid expected outcomes instead of being treated as failed
   near-zero residual solves.

4. **The reference path remained source-controlled.** Sprint 191 avoided new
   optional dependency ambiguity by using `tests/qr_external_dense_reference.py`
   as the required baseline helper and keeping NumPy/SciPy rows deferred.

5. **Freshness validation became target-aware for the new family.** The
   normalizer can validate only `qr-incompatible-ls` and still catches missing,
   stale, incomplete, deferred, failed, duplicate, or path-form mismatches.

6. **Docs and tests reinforce the same claim boundary.** Active docs, schema
   docs, manifest metadata, workflow guards, runner tests, normalizer tests,
   and the QR docs guard all describe fixture-local evidence rather than broad
   QR or external-library parity.

## What Didn't Go Well

1. **The new family remains local-only.** Linux and macOS workflow artifact
   scopes were extended, but Windows selected QR incompatible freshness was
   intentionally not promoted without hosted MSVC evidence.

2. **The comparison count wording had several coupled surfaces.** Updating from
   five to six selected comparison families required synchronized changes in
   public docs, corpus docs, maintainer docs, workflow summaries, manifest
   metadata, and guard wording.

3. **Target-specific freshness can race generation if commands are run in
   parallel.** One Day 14 freshness invocation started before the generated
   `study.tsv` existed. The same command passed after generation completed,
   and the closeout artifact records the ordering requirement.

4. **Explicit tests add review volume.** The sprint kept repeated QR
   incompatible constants in tests because they make row identity and target
   scope reviewable, but this increases the size of future comparison-family
   diffs.

5. **Generated artifacts remain local evidence only.** The authoritative
   outputs are ignored under `build/`, so reviewers must regenerate or inspect
   CI-uploaded artifacts rather than rely on committed generated files.

## Final Metrics

### Validation

| Metric | Sprint 191 close state |
| --- | --- |
| direct `qr-incompatible-ls` generator | passed |
| QR solve owner test | passed, 19 tests, 0 failures, 1104 assertions |
| external comparison runner tests | passed |
| target-specific QR incompatible freshness | passed, six generated rows fresh |
| normalized report-index tests | passed |
| QR header/docs guard | passed |
| corpus schema validation | passed |
| selected report target manifest validation | passed |
| selected comparison workflow guard | passed |
| aggregate selected comparison freshness | passed, 46 normalized rows |
| Python syntax compilation | passed |
| active-doc stale wording scan | passed; only intended cookbook phrase and guard assertion remained |
| final `git diff --check` | passed |
| generated report output status | ignored under `build/` |
| full C quality gate | not required; no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 191 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 13 |
| Sprint closeout artifacts added | 1 |
| Sprint retrospective files added | 1 |
| Workflow files changed | 2 |
| Makefile targets changed | 1 |
| Python scripts changed | 2 |
| Python tests changed | 2 |
| Public/maintainer/report docs changed | 6 |
| Manifest/schema files changed | 3 |
| C source files changed | 0 |
| Public header files changed | 0 |
| Selected comparison families added | 1 |
| Selected generated comparison rows added | 6 |
| Selected Windows metadata rows added | 0 |

### Claim Governance

| Metric | Sprint 191 close state |
| --- | ---: |
| fixture-local QR incompatible least-squares comparison claims added | 1 |
| broad QR parity claims added | 0 |
| broad least-squares parity claims added | 0 |
| global rank-threshold claims added | 0 |
| broad rank-deficient solve claims added | 0 |
| NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity claims added | 0 |
| Windows selected QR incompatible freshness claims added | 0 |
| package-manager proof claims added | 0 |
| shared-library ABI claims added | 0 |
| performance superiority claims added | 0 |
| release proof claims added | 0 |
| state-of-the-art claims added | 0 |

## Closed Claim

Sprint 191 closes this bounded implementation claim:

One local-only QR incompatible least-squares selected comparison family is now
source-controlled for `qr-incompatible-ls`, using the
`qr_overdetermined_incompatible_4x2` fixture, the source-controlled dense QR
reference helper, nonzero expected residual handling, six exact generated study
rows, selected report metadata, target-specific freshness diagnostics,
Linux/macOS exact artifact upload scope, focused failure coverage, and
calibrated documentation.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day2-candidate-family-audit.md](./artifacts/day2-candidate-family-audit.md);
- [day3-fixture-metric-contract.md](./artifacts/day3-fixture-metric-contract.md);
- [day4-reference-dependency-policy.md](./artifacts/day4-reference-dependency-policy.md);
- [day5-fixture-material-implementation.md](./artifacts/day5-fixture-material-implementation.md);
- [day6-reference-execution.md](./artifacts/day6-reference-execution.md);
- [day7-project-observation.md](./artifacts/day7-project-observation.md);
- [day8-study-integration.md](./artifacts/day8-study-integration.md);
- [day9-freshness-integration.md](./artifacts/day9-freshness-integration.md);
- [day10-failure-coverage.md](./artifacts/day10-failure-coverage.md);
- [day11-claim-calibration.md](./artifacts/day11-claim-calibration.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-review-surface-audit.md](./artifacts/day13-review-surface-audit.md);
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md).

No broad QR parity, broad least-squares parity, global rank-threshold policy,
broad rank-deficient solve behavior, NumPy/SciPy/LAPACK/SuiteSparse/Eigen
parity, Windows selected QR incompatible freshness, package-manager proof,
shared-library ABI proof, performance superiority, release proof, or
state-of-the-art claim was added.

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Windows selected QR incompatible freshness | Hosted Windows comparison owner | Add an MSVC/CMake proof for `qr-incompatible-ls`, inspect hosted artifacts, and promote only the exact target metadata if the evidence passes. |
| Optional NumPy/SciPy package baselines | External comparison baseline owner | Select and justify package-backed baselines separately; until then, keep NumPy/SciPy rows deferred and advisory. |
| Broader QR least-squares parity | Future comparison-family owner | Add additional bounded fixtures and references one at a time with exact row IDs, tolerances, freshness diagnostics, and claim calibration. |
| Generated local comparison evidence | Reviewer/CI evidence owner | Regenerate ignored `build/comparison/qr_incompatible_ls/` artifacts locally or inspect uploaded CI artifacts before using generated rows as review evidence. |
| Future comparison-family review volume | Test/report infrastructure owner | Consider extracting shared selected-target constants only if it preserves reviewer-visible row identity and failure diagnostics. |

## Next-Sprint Readiness

Sprint 192 can start from a completed comparison-family pattern rather than a
candidate decision.

| Future need | Sprint 191 handoff |
| --- | --- |
| Additional comparison family | Reuse the Sprint 191 sequence: candidate audit, fixture contract, dependency policy, runner integration, manifest rows, freshness diagnostics, workflow artifact scope, docs calibration, and final validation. |
| Windows promotion | Keep Windows metadata absent for QR incompatible LS until hosted MSVC evidence is reviewed. |
| Optional package baselines | Keep package rows unavailable/deferred unless the sprint explicitly owns package-manager proof semantics. |
| Docs updates | Continue pairing every selected comparison claim with fixture key, reference helper, local-only support tier, and retained non-claims. |
| Workflow changes | Keep artifact uploads exact; do not broaden selected comparison uploads to `build/comparison/**`. |

## Validation Retrospective

Sprint 191 changed workflows, Makefile orchestration, Python report scripts,
Python tests, selected report manifests, docs, schema docs, and sprint planning
documentation but no C source or public headers. The selected validation set
was therefore:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
make build/test_qr_solve
./build/test_qr_solve
python3 tests/test_run_external_comparison.py
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
python3 tests/test_normalize_report_index.py
bash scripts/check_qr_header_docs_guard.sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
make report-index-comparison-freshness
python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py
rg -n 'five fixture|five selected|five generated|minimum-norm and compatible|QR minimum-norm and compatible|compatible least-squares rows from|selected generated comparisons for `qr_underdetermined_minnorm_2x4` and `qr_overdetermined_compatible_5x3`' README.md INSTALL.md docs/maintainer_guide.md docs/solver_selection.md docs/cookbook.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md scripts/check_qr_header_docs_guard.sh
git diff --check
git diff --name-only -- '*.c' '*.h'
```

The first target-specific freshness attempt on Day 14 raced artifact
generation and failed before `study.tsv` existed. The command was rerun after
generation and passed. Future closeout runs should generate selected artifacts
before launching target-specific freshness checks in parallel.

Any future `.c` or `.h` change must run:

```sh
make format
make lint
make test
```
