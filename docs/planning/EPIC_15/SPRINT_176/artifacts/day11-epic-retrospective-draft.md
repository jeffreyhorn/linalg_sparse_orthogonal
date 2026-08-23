# Day 11: Epic 15 Retrospective Draft

## Purpose

Day 11 drafts the source-backed structure for the final Epic 15 retrospective.
The final retrospective should be created only after Sprint 176 final
validation, but its claim structure can now be drafted from Sprint 167-176
artifacts and the Day 10 recalibrated claim surface.

## Draft Retrospective Structure

The final `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md` should use this
structure:

1. Epic summary and scope.
2. Source artifact note covering stale prompt paths versus active Epic 15
   sources.
3. Definition of done checklist across Sprints 167-176.
4. Completed objective summary by sprint.
5. Earned claims and evidence links.
6. Retained non-claims.
7. What went well.
8. What did not go well.
9. Validation summary.
10. Residual queue and Epic 16 candidates.
11. Final claim calibration.

## Completed Objective Summary Draft

| Sprint | Closed objective | Primary evidence |
| --- | --- | --- |
| 167 | Established the Epic 15 baseline, evidence ledger, selected gap list, acceptance gates, and non-claim register. | `SPRINT_167/RETROSPECTIVE.md`, `SPRINT_167/artifacts/day14-sprint-closeout.md`. |
| 168 | Promoted one hosted, methodology-bound selected performance lane for `bench_refactor_csc` on `nos4.mtx --repeat 1`. | `SPRINT_168/RETROSPECTIVE.md`, `SPRINT_168/artifacts/day14-sprint-closeout.md`. |
| 169 | Hardened selected performance methodology with stable metadata, local freshness tests, and bounded sentinel policy. | `SPRINT_169/RETROSPECTIVE.md`, `SPRINT_169/artifacts/day14-sprint-closeout.md`. |
| 170 | Closed the shared-library ABI product decision as static-first-only continuation with guarded shared/dynamic ABI deferral. | `SPRINT_170/RETROSPECTIVE.md`, `SPRINT_170/artifacts/day9-shared-library-abi-product-decision.md`. |
| 171 | Closed package-manager readiness as formal provider deferral with an executable non-claim guard. | `SPRINT_171/RETROSPECTIVE.md`, `SPRINT_171/artifacts/day5-package-manager-deferral.md`. |
| 172 | Cleaned one high-impact public header family, `include/sparse_lu.h`, and added a focused header/docs guard. | `SPRINT_172/RETROSPECTIVE.md`, `SPRINT_172/artifacts/day14-sprint-closeout.md`. |
| 173 | Closed generated API HTML publication as guarded local-only freshness rather than hosted, committed, or artifact-only publication. | `SPRINT_173/RETROSPECTIVE.md`, `SPRINT_173/artifacts/day14-sprint-closeout.md`. |
| 174 | Added one bounded external comparison family: linked-list LU square solve on `lu_nonsym_square_5`. | `SPRINT_174/RETROSPECTIVE.md`, `SPRINT_174/artifacts/day14-sprint-closeout.md`. |
| 175 | Promoted macOS selected comparison freshness and reconciled Linux selected comparison artifacts for the four selected targets. | `SPRINT_175/RETROSPECTIVE.md`, `SPRINT_175/artifacts/day14-sprint-closeout.md`. |
| 176 | Added a deterministic selected allocation-failure proof for iterative repeated-run handles and recalibrated claims. | `SPRINT_176/artifacts/day5-harness-implementation.md` through `day10-claim-recalibration.md`; final validation completed across Days 12-14. |

## Earned Claims Draft

Epic 15 may claim the following when the final Sprint 176 validation record is
complete:

- The project has a source-controlled Epic 15 evidence ledger and finite
  closeout plan for selected evidence/productization gaps.
- The project has a selected Linux hosted performance freshness lane for one
  threshold-free `bench_refactor_csc` canonical row with methodology metadata.
- The selected performance lane has stable methodology fields and a bounded
  local sentinel policy; timing rows remain methodology-bound.
- The project has a recorded static-first shared-library ABI product decision
  and rejects unsupported shared-library configuration rather than silently
  implying support.
- Package-manager provider support is formally deferred and guarded; source
  install through Make/CMake remains the maintained package path.
- One additional public header family, `sparse_lu.h`, has been normalized and
  mechanically checked.
- Generated API HTML has a maintained local freshness command and local-only
  staging guard.
- The selected comparison set includes QR minimum-norm, QR compatible
  least-squares, partial-SVD diagonal top-k, and linked-list LU
  nonsymmetric square-solve families.
- Linux and macOS hosted workflows provide selected comparison artifact
  freshness for the four selected comparison targets.
- Sprint 176 adds one deterministic allocation-failure proof for CG, GMRES,
  and MINRES repeated-run handle prepare/growth cleanup, with public cleanup
  invariants and focused validation.

## Retained Non-Claims Draft

Epic 15 should continue to state that it does not claim:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library ecosystem parity;
- portable performance superiority;
- broad benchmark publication;
- broad allocation-failure cleanup coverage across all solvers and allocation
  paths;
- broad solver correctness beyond named fixtures and maintained proof owners;
- shared-library support, dynamic ABI compatibility, or runtime-loader
  behavior;
- package-manager provider availability;
- broad platform parity;
- Windows Makefile parity or Windows `pkg-config` execution parity;
- Windows report freshness;
- selected oracle freshness on macOS;
- hosted publication of all generated reports;
- hosted generated API HTML publication;
- release evidence.

## What Went Well Draft

1. Epic 15 closed productization gaps by choosing narrow, complete lanes
   instead of spreading partial work across many unsupported claims.
2. The performance lane became reviewable because it gained methodology
   metadata, selected-row freshness, hosted artifacts, and explicit
   threshold-free interpretation.
3. Static-first packaging is now a product decision, not an accidental default.
4. Package-manager support is no longer ambiguous; it is explicitly deferred
   and mechanically guarded.
5. Generated API HTML has a clear local-only status, reducing publication
   ambiguity.
6. The selected comparison surface grew in one complete fixture-local step and
   then gained Linux/macOS hosted selected-artifact freshness.
7. Sprint 176 converted a broad allocation-failure gap into one maintainable,
   family-local proof with tests, docs, and a focused gate.

## What Did Not Go Well Draft

1. Many sprint prompts referenced older Epic 12 paths while the active plans
   lived under Epic 15. Each sprint recorded the mismatch, but the repetition
   added review overhead.
2. Claim governance remains distributed across README, INSTALL, maintainer
   guide, benchmarks, corpus manifests, workflows, scripts, and planning
   artifacts.
3. Hosted evidence remains selected-lane scoped, not broad publication.
4. Windows still has important retained gaps: report freshness, Makefile
   parity, and `pkg-config` execution parity.
5. Generated report and workflow target lists are explicit and guarded, but
   still repetitive to update.
6. Allocation-failure proof is valuable but narrow; most allocation-heavy
   subsystems still lack deterministic failure-path coverage.

## Validation Summary Draft

The final Epic 15 retrospective should summarize validation by sprint family:

| Area | Representative validation |
| --- | --- |
| Performance publication and methodology | `make bench-canonical-report-freshness`, hosted-mode checker, performance sentinel checks, benchmark/report normalizer tests. |
| Static package and ABI deferral | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`, package report-index checks. |
| Package-manager deferral | `scripts/package_manager_deferral_check.sh`, package report-index checks. |
| Public header cleanup | focused LU header/docs guard plus `make format && make lint && make test` after public header edits. |
| Generated API HTML local-only status | `make api-docs-freshness`, `make api-docs-local-only`, generated API claim scans. |
| External comparison and report freshness | `make report-index-comparison-freshness`, comparison runner tests, normalizer tests, workflow guard tests. |
| Allocation-failure proof | `make iterative-allocation-failure-gate`, `ctest -L allocation_failure`, and full C quality gate after source/header edits. |
| Documentation-only closeout | `git diff --check` and claim-surface scans by inspection. |

## Residual Queue Draft

| Residual | Why it remains | Epic 16 candidate handling |
| --- | --- | --- |
| Broad allocation-failure coverage | Sprint 176 proves one selected iterative repeated-run handle family only. | Pick the next allocation-heavy subsystem and repeat the deterministic proof pattern. |
| Windows report freshness | Sprint 175 promoted macOS selected comparison only. | Design a Windows-safe report-generation path or record a stronger deferral. |
| Selected oracle freshness beyond Linux | macOS selected comparison freshness does not include oracle freshness. | Add a separate macOS oracle lane only if runtime/dependency constraints are acceptable. |
| Hosted generated API HTML | Sprint 173 selected local-only generated API HTML. | Decide hosted URL/artifact/retention policy before promotion. |
| Package-manager provider support | Sprint 171 formally deferred provider support. | Select exactly one provider and build recipe/proof/cleanup/docs, or keep deferral. |
| Shared-library and dynamic ABI | Sprint 170 selected static-first-only continuation. | Reopen only with symbol visibility, SONAME/install-name/DLL policy, ABI tests, and loader validation. |
| Broad external-library parity | Sprint 174 added one LU fixture-local comparison only. | Add one bounded comparison family at a time with exact fixtures and non-claims. |
| Portable performance superiority | Epic 15 added selected methodology-bound evidence, not speed claims. | Broaden methodology only after multi-platform, multi-fixture, statistical policy exists. |
| Public header coherence breadth | Sprint 172 cleaned only `sparse_lu.h`. | Select another header family and add a focused guard. |
| Workflow target-list duplication | Sprint 175 guards explicit Linux/macOS selected comparison lists, but updates are repetitive. | Factor selected target inventory before adding more hosted comparison targets. |

## Final Claim Calibration Draft

Epic 15 should position the project as an evidence-disciplined, static-first C
sparse linear algebra library with selected hosted performance/report
freshness, bounded comparison proof, local generated API freshness, formal
package/ABI decisions, and one selected allocation-failure proof.

It should not position the project as a state-of-the-art replacement for
SuiteSparse, PETSc, Trilinos, Eigen, SciPy, or vendor sparse libraries. The
evidence is stronger and more discoverable after Epic 15, but it remains
selected, methodology-bound, and explicitly scoped.

## Validation

Day 11 changed planning artifacts only. No `.c` or `.h` files were modified
for this day, so the full C quality gate is not required.

Validation command:

```sh
git diff --check
```

Result: passed.
