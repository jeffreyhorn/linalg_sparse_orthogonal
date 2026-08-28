# Epic 16 Retrospective

**Epic:** 16 - Product Evidence, Failure-Path Breadth & Distribution Closure
**Sprints:** 177-186
**Status:** Complete with explicit residuals

## Epic Objective

Epic 16 started from the completed Epic 15 closeout and focused on a smaller
set of product-evidence gaps that could be closed with explicit support tiers,
validation owners, and non-claim boundaries. The epic broadened
allocation-failure evidence by one subsystem, closed generated API HTML status,
selected a package-manager proof path, centralized selected report metadata,
formally closed Windows report freshness as a guarded deferral, added one
bounded comparison family, cleaned one public header family, reduced one large
review surface, and recalibrated closeout claims.

The epic deliberately did not try to claim broad state-of-the-art sparse
linear algebra status, broad external ecosystem parity, portable performance,
package-manager availability, shared-library support, dynamic ABI
compatibility, broad Windows parity, or broad generated-report parity.

## Sprint Outcomes

| Sprint | Outcome |
| --- | --- |
| 177 | Established the Epic 16 residual queue, evidence/status matrix, selected closure targets, acceptance gates, quality surface map, and handoff records. |
| 178 | Added deterministic allocation-failure cleanup, stale-output suppression, and retry evidence for the selected `sparse_matmul()` workspace allocation path. |
| 179 | Closed generated API HTML status as strengthened local-only generated output with freshness, coverage, and staging guards. |
| 180 | Selected Homebrew as the local formula/tap proof path and preserved package-manager support as a non-claim, with standalone license metadata left as a proof blocker. |
| 181 | Centralized selected oracle, comparison, benchmark, workflow, and report metadata in `selected_report_targets.tsv` with manifest-driven guards. |
| 182 | Closed Windows report freshness as a formal guarded deferral while preserving Windows CMake/MSVC and static-first install support boundaries. |
| 183 | Added the selected `cholesky_spd_tridiag_5` comparison family with fixture, metrics, report integration, manifest registration, and bounded claims. |
| 184 | Cleaned the QR public header family with declaration preservation, docs/example alignment, and a focused QR header/docs guard. |
| 185 | Reduced the selected LDLT CSC test review surface through family-local helper headers while preserving behavior and registration. |
| 186 | Reconciled evidence, recalibrated claims, updated project-plan status, ran focused and full validation, and drafted this Epic retrospective. |

## Major Outcomes

| Area | Outcome |
| --- | --- |
| Evidence reconciliation | Sprint 186 reconciled 54 project-plan items from Sprints 177-185: 48 Complete, 2 Narrowed, 2 Deferred, 2 Residualized, and 0 Superseded. |
| Allocation failure | The project now has an additional selected deterministic allocation-failure proof for `sparse_matmul()` workspace allocation cleanup and retry behavior. |
| Generated API HTML | Generated API HTML has an enforced local-only status. `docs/api_reference.md` plus checked-in public headers remain the supported source-controlled API reference path. |
| Package-manager provider path | Homebrew is documented as a selected local formula/tap proof path, not provider support. Package-manager support remains guarded as a non-claim. |
| Selected report metadata | Selected report targets have one manifest authority used by schema checks, workflow guards, report normalizer tests, and freshness checks. |
| Windows report freshness | Windows report freshness is formally deferred with guard coverage; Windows support remains scoped to reviewed CMake/MSVC and static-first install/downstream validation. |
| External comparison | The selected comparison set now includes Cholesky SPD tridiagonal evidence alongside selected QR, partial-SVD, and LU comparison families. |
| Public headers/API | QR public header documentation is clearer and better organized without declaration-set drift. |
| Review surface | `tests/test_ldlt_csc.c` has a reduced review surface through family-local helper headers and a helper ownership guard. |
| Claim governance | README, INSTALL, API reference, maintainer guidance, package docs, report docs, planning artifacts, and guards now align on earned evidence and retained non-claims. |

## Project-Plan Status

| Status | Count | Rows |
| --- | ---: | --- |
| Complete | 53 | Sprints 177-185 complete rows plus Sprint 186 items 186.1-186.6. |
| Narrowed | 2 | 179.2, 180.2. |
| Deferred | 2 | 182.2, 182.3. |
| Residualized | 2 | 180.6, 182.6. |
| Superseded | 0 | No Epic 16 item was replaced by an incompatible later path. |
| Planned | 0 | No Epic 16 project-plan item remains planned after the Day 13 residual handoff. |

The evidence-linked project-plan status tables live in
[PROJECT_PLAN.md](./PROJECT_PLAN.md). The Sprint 186 Day 3 matrix remains the
source for the final status vocabulary used across Sprints 177-185.

## Validation Evidence

| Evidence | Result | Boundary |
| --- | --- | --- |
| Generated API validation | `make api-docs-validate` and `make api-docs-freshness` passed. Coverage reported 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. | Local generated Doxygen freshness and staging only; not hosted or committed generated HTML. |
| QR header/docs guard | `make qr-header-docs-guard` passed. | Declaration-preserving QR header/docs coherence only; not broad QR behavior or API expansion. |
| Package guards | `bash scripts/static_package_deferral_check.sh` and `bash scripts/package_manager_deferral_check.sh` passed. | Static-first package evidence and package-manager non-claims; not Homebrew/provider support. |
| Corpus and workflow guards | `python3 scripts/validate_corpus_schema.py`, `python3 tests/test_selected_report_targets_manifest.py`, `python3 tests/test_selected_comparison_workflow.py`, and `python3 tests/test_normalize_report_index.py` passed. | Selected manifest authority and workflow/report scope only. |
| Selected oracle freshness | `make report-index-oracle-freshness` passed with 54 normalized rows. | Selected local QR and partial-SVD oracle rows only. |
| Selected comparison freshness | `make report-index-comparison-freshness` passed with 39 normalized rows. | Selected QR, partial-SVD, LU, and Cholesky comparison rows only. |
| External comparison runner tests | `python3 tests/test_run_external_comparison.py` passed. | Runner behavior and selected comparison metadata only. |
| Allocation-failure gate | `make matmul-allocation-failure-gate` passed; `test_matmul` ran 18 tests, 0 failures, 0 skips, and 185 assertions. | Selected `sparse_matmul()` allocation-failure path only. |
| Review-surface guard | `make ldlt-csc-helper-guard` passed. | LDLT CSC helper ownership and registration boundaries only. |
| Source registration | `make source-list-check` passed with 49 library sources. | Make/CMake source-list synchronization only. |
| Full repository quality gate | `make format && make lint && make test` passed; the full Make test suite ended with `All tests passed.` | Local repository quality confidence; not hosted platform parity. |
| Final whitespace | `git diff --check` passed. | Changed-file whitespace hygiene. |

## Earned Claims

Epic 16 earns these claims with qualifiers:

- The Epic 16 selected-gap plan and evidence matrix were completed and
  reconciled against Sprint 177-185 artifacts.
- `sparse_matmul()` workspace allocation has selected deterministic
  allocation-failure cleanup, stale-output suppression, and retry evidence.
- Generated API HTML has a maintained local-only freshness and staging guard;
  source-controlled API reference remains `docs/api_reference.md` plus public
  headers under `include/`.
- Homebrew is the selected local formula/tap proof path, with source-controlled
  template material and claim-safe proof behavior.
- Selected report target metadata is centralized in
  `tests/corpus/manifests/selected_report_targets.tsv`.
- Windows report freshness has a formal guarded deferral.
- The selected comparison set includes `cholesky_spd_tridiag_5` as a bounded
  Cholesky comparison family.
- `include/sparse_qr.h` has declaration-preserving public-header coherence
  cleanup and aligned QR-facing docs.
- The selected LDLT CSC review surface was reduced through family-local helper
  headers while preserving the proof-owner binary, behavior, and registration.
- Public and maintainer documentation has been recalibrated to preserve
  evidence-backed claims and retained non-claims.

## Non-Claims

Epic 16 does not claim:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library or ecosystem parity against SuiteSparse, PETSc,
  Trilinos, Eigen, SciPy, LAPACK, NumPy, or package-manager ecosystems;
- broad solver correctness beyond named fixtures, maintained corpus rows, and
  selected proof owners;
- portable performance superiority, backend superiority, OpenMP speedup proof,
  portable runtime proof, or state-of-the-art performance;
- broad allocation-failure cleanup coverage beyond selected proof lanes;
- hosted generated API HTML publication;
- retained generated API CI artifacts;
- committed generated API HTML;
- broad generated report freshness;
- Windows selected report freshness;
- broad Windows package, Makefile, `pkg-config`, or platform parity;
- package-manager provider availability;
- Homebrew/core readiness, bottles, Linuxbrew support, or public tap support;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- release readiness.

## Residual Queue

| Priority | Residual | Why it remains | Next-epic closure target |
| ---: | --- | --- | --- |
| 1 | R186-PKG-LICENSE | Full Homebrew proof success remains blocked because no root `LICENSE`, `COPYING`, or `NOTICE` file exists. | Add approved standalone license metadata or select an alternate formula license strategy, then rerun Homebrew proof and package guards. |
| 2 | R186-WIN-PWSH | Local PowerShell parse/workflow validation could not run because `pwsh` is unavailable. | Run the PowerShell checks in a `pwsh`-equipped environment or assign hosted validation ownership. |
| 3 | R186-WIN-REPORT-FRESHNESS | Windows report freshness was formally deferred instead of promoted. | Select and prove one Windows-safe freshness lane or keep the deferral with refreshed blockers. |
| 4 | R186-HOSTED-API | Generated API HTML remains local-only. | Revisit hosted publication, retained artifact, or committed output only if product value justifies the extra publication and freshness guards. |
| 5 | R186-BROAD-COMPARISON | Comparison evidence remains selected-fixture-only. | Add future comparison evidence one bounded family at a time with fixture, metric, report, manifest, and claim evidence. |
| 6 | R186-REVIEW-SURFACE-NEXT | Other large review surfaces remain outside the selected Sprint 185 LDLT CSC reduction. | Select exactly one future large review surface and repeat the behavior-preserving extraction pattern. |

The detailed prioritized residual handoff with owner surfaces, expected
evidence, validation commands, and deferral horizons is published in
[EPIC_16_RESIDUAL_QUEUE.md](./EPIC_16_RESIDUAL_QUEUE.md).

## State-Of-The-Art Assessment

Epic 16 does not earn an unqualified state-of-the-art sparse linear algebra
claim.

The defensible assessment is narrower: Epic 16 improves evidence discipline,
failure-path confidence for one additional subsystem, generated API status
clarity, selected package/provider proof structure, selected report metadata,
guarded Windows report deferral, bounded comparison breadth, QR API-doc
coherence, and one large test review surface. These are useful engineering and
adoption improvements, but they are not comparative proof against mature sparse
linear algebra ecosystems.

A future state-of-the-art or broad parity claim would need named libraries,
versions, fixtures, metrics, tolerances, platforms, compilers, package
provenance, performance methodology, failure semantics, support-tier
boundaries, and reviewed hosted evidence for each selected claim.

## What Went Well

1. **The epic closed narrow gaps with traceable evidence.** Each sprint ended
   with artifacts, validation records, and explicit claim boundaries rather
   than broad aspirational statements.

2. **The final closeout separated status types.** Complete, narrowed,
   deferred, residualized, and planned rows are visible in the project plan and
   Sprint 186 evidence matrix.

3. **Generated API status became enforceable.** The local-only decision is
   backed by Doxygen generation, page coverage, and staging guards.

4. **Report target ownership became less repetitive.** The selected target
   manifest gives workflows, report checks, docs, and tests one source of
   selected metadata truth.

5. **Comparison work stayed bounded.** The added Cholesky family is useful
   because it is exact about fixtures, metrics, rows, support tiers, and
   non-parity wording.

6. **Validation matched the risk.** Documentation-only days used focused
   guards and whitespace checks; closeout still ran the full repository quality
   gate for confidence.

## Could Be Better

1. **Several residuals are product decisions, not implementation details.**
   Homebrew proof success, hosted API publication, and Windows report freshness
   require explicit future product choices before code can close them.

2. **Environment-dependent validation remains a local limitation.** The local
   environment has no `pwsh`, so PowerShell report validation stays residual
   rather than proof.

3. **Claim governance is still distributed.** README, INSTALL, maintainer
   docs, API reference, package docs, manifests, scripts, workflows, and sprint
   artifacts all carry pieces of the support-tier story.

4. **Full lint remains expensive.** The final gate passed, but `clang-tidy` and
   `cppcheck` across the full C/test surface remain the slowest local checks.

5. **Review-surface reduction is intentionally incremental.** Sprint 185
   improved the LDLT CSC cluster only; other large test and solver surfaces
   need separate selection and validation.

## Key Deliverables

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [SPRINT_177/RETROSPECTIVE.md](./SPRINT_177/RETROSPECTIVE.md)
- [SPRINT_178/RETROSPECTIVE.md](./SPRINT_178/RETROSPECTIVE.md)
- [SPRINT_179/RETROSPECTIVE.md](./SPRINT_179/RETROSPECTIVE.md)
- [SPRINT_180/RETROSPECTIVE.md](./SPRINT_180/RETROSPECTIVE.md)
- [SPRINT_181/RETROSPECTIVE.md](./SPRINT_181/RETROSPECTIVE.md)
- [SPRINT_182/RETROSPECTIVE.md](./SPRINT_182/RETROSPECTIVE.md)
- [SPRINT_183/RETROSPECTIVE.md](./SPRINT_183/RETROSPECTIVE.md)
- [SPRINT_184/RETROSPECTIVE.md](./SPRINT_184/RETROSPECTIVE.md)
- [SPRINT_185/RETROSPECTIVE.md](./SPRINT_185/RETROSPECTIVE.md)
- [SPRINT_186/PLAN.md](./SPRINT_186/PLAN.md)
- [SPRINT_186/WORKING_NOTES.md](./SPRINT_186/WORKING_NOTES.md)
- [SPRINT_186/artifacts/day3-reconciled-evidence-matrix.md](./SPRINT_186/artifacts/day3-reconciled-evidence-matrix.md)
- [SPRINT_186/artifacts/day4-public-claim-inventory.md](./SPRINT_186/artifacts/day4-public-claim-inventory.md)
- [SPRINT_186/artifacts/day8-project-plan-status-update.md](./SPRINT_186/artifacts/day8-project-plan-status-update.md)
- [SPRINT_186/artifacts/day10-focused-integrated-validation.md](./SPRINT_186/artifacts/day10-focused-integrated-validation.md)
- [SPRINT_186/artifacts/day11-full-repository-quality-gate.md](./SPRINT_186/artifacts/day11-full-repository-quality-gate.md)
- [SPRINT_186/artifacts/day13-residual-queue-handoff.md](./SPRINT_186/artifacts/day13-residual-queue-handoff.md)
- [SPRINT_186/artifacts/day14-closeout-review-pr-handoff.md](./SPRINT_186/artifacts/day14-closeout-review-pr-handoff.md)
- [EPIC_16_RESIDUAL_QUEUE.md](./EPIC_16_RESIDUAL_QUEUE.md)

## Completion

Epic 16 project-plan work is complete with explicit residuals as of the Day 13
residual handoff. Evidence reconciliation, claim calibration, project-plan
status, integrated validation, retrospective drafting, and next-epic residual
publication are complete. Sprint 186 Day 14 completed final closeout review
and PR handoff packaging.
