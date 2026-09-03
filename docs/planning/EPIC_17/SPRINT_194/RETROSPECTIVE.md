# Sprint 194 Retrospective

**Sprint:** 194 - Adoption and API Coherence Simplification
**Duration:** 14 days (Days 1-14 landed on branch `sprint-194`)
**Status:** Complete; adoption and API coherence guidance simplified with
support/readiness truth consolidated and evidence boundaries preserved

## Source Artifact Note

Sprint 194 was executed from the Epic 17 project-plan section for Sprint 194
and lives under `docs/planning/EPIC_17/SPRINT_194/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 194 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited adoption friction across README, install docs, tutorial,
      cookbook, solver-selection docs, examples, API reference, maintainer
      guide, public headers, package/install templates, selected report
      evidence, and prior planning artifacts.
- [x] Selected `INSTALL.md` as the compact user-facing support/readiness matrix
      owner.
- [x] Added an evidence-bounded support/readiness matrix that separates
      supported, validated, local-only, hosted-evidence, deferred, not-claimed,
      and residual surfaces.
- [x] Added installed-consumer tutorial guidance for Unix Make/`pkg-config`
      and installed CMake consumers while preserving source-tree tutorial
      routing.
- [x] Normalized direct, iterative, QR, SVD, and eigensolver diagnostic wording
      across maintained user docs.
- [x] Reduced tutorial-style adoption narrative in six public headers while
      preserving declaration-adjacent API contracts and Doxygen coverage.
- [x] Re-ran documentation, install, example, selected report, and generated
      artifact hygiene checks.
- [x] Ran the full C quality gate because public `.h` files changed.
- [x] Confirmed the sprint did not add broad package-manager, shared-library,
      dynamic ABI, broad Windows parity, portable performance,
      external-library parity, release readiness, or state-of-the-art claims.

## What Went Well

1. **Support truth now has a user-facing owner.** `INSTALL.md` owns a compact
   support/readiness matrix that gives consumers one place to distinguish
   supported static install paths, validated CMake/Windows lanes, selected
   evidence, deferred package/ABI surfaces, and explicit non-claims.

2. **The sprint preserved evidence boundaries while simplifying prose.**
   README, tutorial, cookbook, solver-selection, API reference, examples, and
   maintainer docs now route to the matrix or proof owners instead of
   repeating long caveat blocks in every adoption surface.

3. **Installed-consumer guidance became more usable.** The new tutorial path
   gives downstream users copy-pasteable Unix Make/`pkg-config` and CMake
   consumer flows that match `tests/test_install.sh`,
   `tests/test_cmake_install.sh`, `sparse.pc.in`, and
   `cmake/SparseConfig.cmake.in`.

4. **Diagnostic wording became more coherent.** Direct solver docs now keep
   `sparse_err_t` return-code ownership clear, iterative docs describe
   `sparse_iter_result_t` fields directly, and QR/SVD/eigs docs keep
   tolerance, rank, residual, and backend wording local to the relevant API.

5. **Header cleanup stayed bounded.** Six public headers lost duplicated
   tutorial/adoption narrative, but declarations, typedefs, enum values,
   public constants, struct fields, signatures, status-code mappings,
   ownership rules, callback contracts, and cleanup helpers were intentionally
   unchanged.

6. **Validation matched the changed surface.** The sprint ran the full C gate
   after header edits and also covered docs, Doxygen/API coverage, examples,
   install/export, selected report freshness, selected performance docs, and
   generated-output hygiene.

## What Didn't Go Well

1. **The support matrix necessarily became dense.** Compressing many
   historical support boundaries into one table improves discoverability, but
   the table still carries substantial non-claim detail because Windows,
   package-manager, shared-library, ABI, selected-report, and performance
   boundaries are easy to over-read.

2. **The documentation graph remains broad.** The sprint reduced duplication,
   but support/evidence interpretation still spans README, INSTALL,
   maintainer guide, corpus docs, benchmark docs, selected manifests, CI
   workflows, and validation scripts.

3. **Header comments still need careful review discipline.** Comment-only
   public-header cleanup has low runtime risk, but the files remain API-facing
   and require full C quality gate evidence when touched.

4. **Local PowerShell execution remains unavailable.** The Windows PowerShell
   ownership guard passes structurally, but local `pwsh` is not installed, so
   actual PowerShell execution evidence remains hosted-CI-owned.

5. **Markdown link checking is still not a first-class local gate.** The sprint
   inspected anchors and routes through available checks, but no dedicated
   Markdown link-check target exists in the current Makefile/script surface.

## Final Metrics

### Validation

| Metric | Sprint 194 close state |
| --- | --- |
| final `make format` | passed |
| final `make lint` | passed strict build, tooling/example compile, `clang-tidy`, and `cppcheck` checks |
| final `make test` | passed with final `All tests passed.` |
| Doxygen/API docs coverage | passed for checked-in public headers |
| generated API local-only guard | passed |
| QR header docs guard | passed |
| source-list check | passed with 49 library sources |
| LDLT CSC helper guard | passed |
| QR external-reference helper guard | passed |
| Windows PowerShell structural ownership guard | passed |
| corpus schema validation | passed |
| selected report target manifest validation | passed |
| selected performance docs validation | passed |
| report normalizer tests | passed |
| external comparison runner tests | passed |
| selected comparison workflow tests | passed |
| selected benchmark freshness tests | passed |
| tooling build | produced 16 benchmark binaries and 14 example binaries |
| Make install/`pkg-config` proof | passed 23 checks, 0 failures |
| CMake install/export proof | passed 27 checks, 0 failures, 0 skips |
| selected oracle freshness | passed for 54 generated rows |
| selected comparison freshness | passed for 46 generated rows |
| selected canonical benchmark freshness | passed |
| final `git diff --check` | passed |
| generated output status | `build/` and `docs/api/` ignored and untracked |

### Changed Surface

| Metric | Sprint 194 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| User-facing documentation files changed | 8 |
| Public header files changed | 6 |
| C implementation files changed | 0 |
| Public declarations intentionally changed | 0 |
| Package template files changed | 0 |
| CI workflow files changed | 0 |
| Python scripts changed | 0 |
| Python tests changed | 0 |

### Claim Governance

| Metric | Sprint 194 close state |
| --- | ---: |
| support/readiness matrix owners added | 1 |
| installed-consumer tutorial paths clarified | 2 |
| public headers cleaned | 6 |
| broad package-manager support claims added | 0 |
| Homebrew/core or bottle claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI claims added | 0 |
| Windows Makefile parity claims added | 0 |
| Windows `pkg-config` execution parity claims added | 0 |
| broad Windows report freshness claims added | 0 |
| broad external-library parity claims added | 0 |
| portable performance claims added | 0 |
| release readiness claims added | 0 |
| state-of-the-art claims added | 0 |

## Closed Claim

Sprint 194 closes this bounded documentation/API coherence claim:

The project now has a consolidated, evidence-bounded user-facing
support/readiness matrix in `INSTALL.md`; installed-consumer tutorial routing
for Unix Make/`pkg-config` and CMake consumers; normalized diagnostic wording
across maintained user docs; reduced tutorial-style narrative in six public
headers; and final validation evidence for docs, install/export, examples,
selected report freshness, generated-output hygiene, and the full C quality
gate after public-header edits.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-adoption-intake.md](./artifacts/day1-adoption-intake.md);
- [day2-adoption-friction-audit.md](./artifacts/day2-adoption-friction-audit.md);
- [day3-support-matrix-contract.md](./artifacts/day3-support-matrix-contract.md);
- [day4-support-matrix-implementation.md](./artifacts/day4-support-matrix-implementation.md);
- [day5-consumer-tutorial-audit.md](./artifacts/day5-consumer-tutorial-audit.md);
- [day6-consumer-tutorial-implementation.md](./artifacts/day6-consumer-tutorial-implementation.md);
- [day7-diagnostics-wording-contract.md](./artifacts/day7-diagnostics-wording-contract.md);
- [day8-direct-iterative-diagnostics-cleanup.md](./artifacts/day8-direct-iterative-diagnostics-cleanup.md);
- [day9-qr-svd-eigs-diagnostics-cleanup.md](./artifacts/day9-qr-svd-eigs-diagnostics-cleanup.md);
- [day10-header-narrative-audit.md](./artifacts/day10-header-narrative-audit.md);
- [day11-header-narrative-cleanup.md](./artifacts/day11-header-narrative-cleanup.md);
- [day12-docs-examples-install-validation.md](./artifacts/day12-docs-examples-install-validation.md);
- [day13-full-quality-claim-calibration.md](./artifacts/day13-full-quality-claim-calibration.md);
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md).

No broad platform support, package-manager distribution, Homebrew/core
readiness, bottle support, shared-library support, dynamic ABI compatibility,
runtime-loader behavior, broad Windows parity, broad report freshness,
portable performance, release readiness, external-library parity, or
state-of-the-art claim was added.

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Local PowerShell execution unavailable | Windows validation owner | Install `pwsh` locally or continue relying on hosted CI for `--require-pwsh` execution evidence. |
| No dedicated Markdown link-check target | Documentation tooling owner | Add a link-check target, fixtures, and documented failure semantics before making link integrity part of the local acceptance gate. |
| Package-manager distribution remains unclaimed | Future package-provider owner | Complete provider-specific proof, license metadata review, package-manager guard updates, hosted/local validation, and user-doc support wording. |
| Shared-library and dynamic ABI support remain deferred | Future ABI/package owner | Add shared-library build/install artifacts, symbol visibility policy, ABI policy, loader validation, package metadata, and consumer tests. |
| Broad Windows parity remains unclaimed | Future Windows platform owner | Add reviewed Windows evidence for each desired surface without converting selected CMake or selected comparison lanes into broad platform claims. |
| Portable performance remains unclaimed | Future benchmark methodology owner | Add multi-platform, repeated, variance-aware evidence with reviewed machine/compiler/threading context and claim wording. |
| Public headers still contain detailed API contracts | API documentation owner | Keep declaration-adjacent contracts in headers; move only broad workflow narrative when Doxygen coverage and docs routing remain valid. |

## Next-Sprint Readiness

Sprint 195 can start from a clearer adoption baseline: consumers have one
support/readiness matrix, installed-consumer docs have direct tutorial routes,
diagnostic wording is more consistent, and header narrative has been reduced
without changing public declarations.

| Future need | Sprint 194 handoff |
| --- | --- |
| Additional docs simplification | Route support/evidence claims through `INSTALL.md#support-readiness-matrix` and preserve proof-owner links rather than duplicating caveat text. |
| Package-manager work | Keep package-manager support unclaimed until provider proof, metadata, docs, and guards all change together. |
| Shared-library or ABI work | Treat dynamic artifacts, loader behavior, ABI compatibility, and static/shared selection as new implementation work, not wording cleanup. |
| Windows support expansion | Keep Windows CMake-first and selected Cholesky comparison wording bounded unless new hosted lanes and guards are added. |
| Performance wording | Keep benchmark output methodology-bound and threshold-free unless a future sprint adds reviewed baseline and variance policy. |
| Header edits | Run the full C gate and Doxygen/API checks after any public-header comment or declaration edits. |

## Validation Retrospective

Sprint 194 changed public header comments, so the full C quality gate was
required and run. The final full quality command was:

```sh
make format && make lint && make test
```

It passed.

The final affected-surface validation also passed:

```sh
make docs-check api-docs-local-only qr-header-docs-guard \
  source-list-check ldlt-csc-helper-guard qr-external-ref-helper-guard \
  windows-powershell-guard tooling-build examples-build
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
bash tests/test_install.sh
bash tests/test_cmake_install.sh
make report-index-oracle-freshness report-index-comparison-freshness \
  bench-canonical-report-freshness
```

A final `git diff --check` passed after the closeout and retrospective docs
were added.
