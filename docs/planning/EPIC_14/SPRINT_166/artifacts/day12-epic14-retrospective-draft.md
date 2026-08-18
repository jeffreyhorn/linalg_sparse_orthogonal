# Day 12 Epic 14 Retrospective Draft

## Purpose

Day 12 drafts the Epic 14 retrospective structure and content from the
evidence reconciled in Sprint 166 Days 1 through 11. This is a closeout draft,
not the final `EPIC_14_RETROSPECTIVE.md`; Day 13 will publish the residual
queue and Day 14 will finalize closeout material.

## Proposed Retrospective Header

```md
# Epic 14 Retrospective

**Epic:** 14 - Generated API Publication Decisions, Hosted Evidence,
Comparison Families, Package Boundaries, Performance Methodology & Final
Claim Calibration
**Sprints:** 157-166
**Status:** Complete with explicit residuals
```

## Draft Epic Objective

Epic 14 started from the Epic 13 residual queue and focused on closing a
bounded set of product-evidence gaps rather than expanding broad numerical or
state-of-the-art claims. The epic targeted generated API reference ambiguity,
reviewed hosted paths for selected generated evidence, one additional bounded
QR comparison family, one bounded partial-SVD comparison family, Windows
package parity decision closure, methodology-bound performance publication,
public-header/API coherence, static-first package boundary hardening, and
final claim recalibration.

The epic deliberately did not attempt to prove broad sparse linear algebra
parity, broad external-library equivalence, portable performance superiority,
dynamic ABI compatibility, shared-library support, package-manager readiness,
or unqualified state-of-the-art status. Its working standard was narrower:
every earned claim must name the evidence owner, validation gate, support tier,
and retained non-claims.

## Draft Sprint Outcomes

| Sprint | Outcome |
| --- | --- |
| 157 | Established the Epic 14 baseline, residual target selection, evidence contract, claim target register, quality surface map, risk register, and Sprint 158 generated API handoff. |
| 158 | Closed generated API HTML ambiguity with an explicit local-only product decision, source-header-first API authority, Doxygen warning/coverage checks, and documentation alignment. |
| 159 | Promoted selected oracle freshness and the then-selected comparison freshness path into reviewed Linux hosted evidence while preserving advisory/unselected row boundaries. |
| 160 | Added the bounded `qr-compatible-ls` comparison family with descriptor-backed generation, normalized selected freshness rows, focused tests, and fixture-local non-parity wording. |
| 161 | Added the bounded `partial-svd-diag6-k2` comparison family with subspace-safe metrics, normalized selected freshness rows, focused tests, and explicit non-claims for broad SVD behavior. |
| 162 | Closed the Windows package parity decision by retaining Windows Makefile and Windows `pkg-config` execution parity as explicit guarded non-claims while preserving CMake-first package evidence. |
| 163 | Published methodology-bound local benchmark and sentinel report metadata while keeping performance superiority, hosted performance proof, and state-of-the-art performance claims out of scope. |
| 164 | Completed a declaration-preserving public-header/API coherence batch for `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h`, with generated reference and full C quality validation. |
| 165 | Hardened the static-first package boundary with stronger shared-library rejection, ABI/package non-claim audits, refreshed install/downstream proof, docs alignment, and package validation. |
| 166 | Reconciled final evidence, local validation, hosted CI scope, public claims, project-plan item status, success criteria, and closeout material for Epic 14. |

## Draft Major Outcomes

| Area | Outcome |
| --- | --- |
| Generated API reference | Ambiguity is closed by a local-only generated HTML decision, source-header-first API authority, and recurring Doxygen warning/coverage checks. |
| Hosted generated evidence | Selected oracle and comparison evidence has a reviewed Linux hosted path, with Sprint 166 Day 7 aligning hosted comparison artifacts to all selected comparison families. |
| QR comparison | The selected QR comparison surface includes `qr-minnorm` plus the bounded `qr-compatible-ls` family and normalized selected freshness checks. |
| Partial-SVD comparison | The selected partial-SVD comparison surface includes `partial-svd-diag6-k2` with subspace-safe metrics and normalized selected freshness checks. |
| Windows package decision | Windows package confidence remains CMake-first and static-first; Windows Makefile and Windows `pkg-config` execution parity are guarded non-claims. |
| Performance publication | Benchmark and sentinel report rows now carry methodology metadata and non-superiority boundaries; generated rows remain local-only unless a future hosted lane is added. |
| Public headers/API docs | A selected public-header batch is cleaner and declaration-preserving, and the user-facing docs match the selected header behavior. |
| Static-first package boundary | Static package support is better guarded against shared-library, ABI, runtime-loader, package-manager, Windows Makefile, and Windows `pkg-config` drift. |
| Claim governance | Public claims were audited against state-of-the-art, external-parity, performance, package, Windows, ABI, runtime-loader, and generated-report overreach. |

## Draft Validation Evidence

| Evidence | Result | Boundary |
| --- | --- | --- |
| Sprint 166 Day 5 local baseline | Passed `make format`, `make lint`, `make test`, corpus schema validation, report normalizer tests, comparison runner tests, Python compile checks, and `git diff --check`. | Local macOS baseline only; not hosted Linux/Windows proof or broad platform parity. |
| Sprint 166 Day 6 supplemental sweep | Passed generated API docs, oracle freshness, comparison freshness, report-index checks, package checks, install/export checks, benchmark reports, performance sentinels, claim scans, and `git diff --check`. | Generated outputs remain local-only unless named hosted evidence exists. |
| Sprint 166 Day 7 hosted CI reconciliation | Updated reviewed Linux hosted comparison wording, summary, and upload paths to cover QR min-norm, QR compatible least-squares, and partial-SVD diag6 k2 selected families. | Local workflow validation only until PR hosted CI runs. |
| Sprint 166 Day 8 public claim audit part 1 | Updated generated comparison evidence wording across README, solver-selection, maintainer guide, corpus docs, and report-index schema docs. | Selected comparison rows are hosted evidence only after the reviewed Linux hosted lane runs. |
| Sprint 166 Day 9 public claim audit part 2 | Static package deferral guard passed and Windows `pkg-config` wording was tightened to command execution parity. | Static-first support only; no shared-library, dynamic ABI, runtime-loader, package-manager, or broad Windows parity claim. |
| Sprint 166 Day 10 project-plan reconciliation I | Sprints 157-161 items were marked complete, narrowed, deferred, or residualized with evidence links. | Comparison claims remain fixture-local and generated API HTML remains local-only. |
| Sprint 166 Day 11 project-plan reconciliation II | Sprints 162-166 items and Epic 14 success criteria were reconciled with evidence-backed statuses. | Final state-of-the-art assessment remains for Day 12-14 closeout wording. |

## Draft Earned Claims

Epic 14 earns these claims with qualifiers:

- Generated API reference publication is no longer ambiguous: generated HTML is
  intentionally ignored/local-only, while checked-in public headers and
  recurring Doxygen coverage checks are the maintained API documentation
  authority.
- Selected generated oracle evidence has a reviewed Linux hosted freshness
  path.
- Selected generated comparison evidence has a reviewed Linux hosted freshness
  path for the selected QR min-norm, QR compatible least-squares, and
  partial-SVD diag6 k2 comparison families after the Sprint 166 Day 7 workflow
  reconciliation.
- The QR comparison surface includes one additional bounded fixture-local
  family, `qr-compatible-ls`, with normalized selected freshness rows.
- The partial-SVD comparison surface includes one bounded fixture-local
  diagonal top-k family, `partial-svd-diag6-k2`, with subspace-safe metrics and
  normalized selected freshness rows.
- Windows package parity has an explicit product decision: retained
  non-claims for Windows Makefile install/uninstall parity and Windows
  `pkg-config` command execution parity, with Windows support remaining
  CMake-first and static-first.
- Local benchmark and sentinel report rows are methodology-bound and more
  navigable through normalized report metadata.
- The selected public-header cleanup batch preserved declarations while
  improving ownership, lifetime, error/status, output-buffer, option/result,
  backend, and workflow wording.
- The static-first package boundary is hardened with stronger shared-library
  rejection, package metadata checks, downstream static package proof, and
  public non-claim wording.
- Public claims now better distinguish reviewed hosted evidence, local
  generated evidence, advisory rows, explicit product decisions, and retained
  non-claims.

## Draft Non-Claims

Epic 14 does not claim:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library or ecosystem parity against LAPACK, NumPy, SciPy,
  SuiteSparse, Eigen, PETSc, Trilinos, or package-manager ecosystems;
- broad QR, SVD, or partial-SVD correctness beyond selected fixture-local
  evidence;
- raw QR basis identity, QR sign/orientation/order parity, raw singular-vector
  identity, phase identity, or basis-order parity;
- broad report-index freshness for every generated family;
- proof from advisory, deferred, optional dependency, or source-controlled
  metadata rows alone;
- hosted proof for local-only generated API HTML, benchmark/sentinel rows, or
  unselected generated report families;
- portable performance, backend superiority, OpenMP speedup proof, portable
  runtime proof, or state-of-the-art performance evidence;
- package-manager distribution;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- static/shared package selector UX;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broad Windows package or platform parity;
- broad public-header cleanup beyond the selected Sprint 164 header batch;
- final PR-hosted CI success for Sprint 166 until the branch is pushed and CI
  results are reconciled.

## Draft State-Of-The-Art Assessment

Epic 14 does not earn an unqualified state-of-the-art sparse linear algebra
claim.

The defensible assessment is narrower: Epic 14 improves the project's
engineering maturity by closing ambiguity around generated API documentation,
moving selected generated evidence onto reviewed hosted paths, widening
fixture-local comparison coverage for QR and partial-SVD, making Windows
package limitations explicit and test-backed, improving methodology metadata
for local performance rows, cleaning selected public headers without signature
drift, and hardening static-first package boundaries.

Those outcomes improve trust, maintainability, and evidence discipline. They
are not a comparative proof against mature sparse linear algebra ecosystems.
A future broad or state-of-the-art claim would require named competing
libraries, versions, fixtures, numerical tolerances, platform/compiler
matrices, package provenance, performance methodology, failure semantics, and
reviewed hosted evidence for the selected claims.

## Draft Lessons Learned

1. **Product decisions can close gaps when they are explicit and guarded.**
   Sprints 158, 162, and 165 closed ambiguity by choosing local-only generated
   HTML, CMake-first Windows package support, and static-first package
   boundaries instead of silently implying unsupported surfaces.

2. **Hosted evidence has to track selected-family growth.** Sprint 166 Day 7
   was necessary because Sprint 160 and 161 expanded selected comparison
   families after Sprint 159 first introduced the hosted comparison lane.

3. **Fixture-local comparison work is useful only when the claim is narrow.**
   The QR and partial-SVD comparison additions are valuable because they name
   exact fixtures, metrics, references, and non-claims.

4. **Generated report rows need support-tier language everywhere they appear.**
   Local generated rows, source-controlled metadata, advisory rows, and hosted
   artifacts are different evidence classes and cannot share generic wording.

5. **Declaration-preserving header cleanup is the right default.** Sprint 164
   improved API usability without drifting signatures, which kept validation
   and review risk controlled.

6. **Static-first package support is stronger when unsupported dynamic
   surfaces fail loudly.** The package boundary is more coherent because
   shared-library, ABI, runtime-loader, package-manager, and Windows parity
   non-claims are checked instead of assumed.

## Draft Next-Epic Candidate Themes

| Priority | Candidate | Closure target |
| --- | --- | --- |
| 1 | Final hosted CI evidence reconciliation for Sprint 166 branch. | Confirm Linux/macOS/Windows PR results and update final closeout evidence if hosted behavior differs from local validation. |
| 2 | Hosted performance publication decision. | Either add a reviewed hosted performance/report lane with artifact upload and methodology fields or retain local-only performance rows as an explicit non-claim. |
| 3 | Broader public-header cleanup batch. | Apply the Sprint 164 declaration-preserving process to remaining high-risk public headers such as QR, SVD, ILU, IC, and LDLT. |
| 4 | Additional bounded comparison family. | Add one selected fixture-local family with source-controlled references, metrics, normalized rows, tests, and clear non-parity wording. |
| 5 | Shared-library ABI product design. | Decide whether to continue static-only support or fund symbol visibility, export/import, SONAME/install-name/DLL metadata, runtime-loader validation, and ABI policy. |
| 6 | Package-manager distribution readiness. | Define package-manager target scope, provenance, install layout, versioning, CI proof, and support-tier wording before claiming ecosystem distribution. |

## Draft Final Retrospective Inputs

The final Epic 14 retrospective should cite:

- [`PROJECT_PLAN.md`](../../PROJECT_PLAN.md)
- [`SPRINT_157/artifacts/day14-sprint-closeout-and-sprint158-handoff.md`](../../SPRINT_157/artifacts/day14-sprint-closeout-and-sprint158-handoff.md)
- [`SPRINT_158/artifacts/day14-closeout-handoff.md`](../../SPRINT_158/artifacts/day14-closeout-handoff.md)
- [`SPRINT_159/artifacts/day14-closeout.md`](../../SPRINT_159/artifacts/day14-closeout.md)
- [`SPRINT_160/artifacts/day14-closeout.md`](../../SPRINT_160/artifacts/day14-closeout.md)
- [`SPRINT_161/artifacts/day14-closeout.md`](../../SPRINT_161/artifacts/day14-closeout.md)
- [`SPRINT_162/artifacts/day14-closeout.md`](../../SPRINT_162/artifacts/day14-closeout.md)
- [`SPRINT_163/artifacts/day14-closeout.md`](../../SPRINT_163/artifacts/day14-closeout.md)
- [`SPRINT_164/artifacts/day14-closeout.md`](../../SPRINT_164/artifacts/day14-closeout.md)
- [`SPRINT_165/artifacts/day14-closeout-and-handoff.md`](../../SPRINT_165/artifacts/day14-closeout-and-handoff.md)
- [`day5-local-validation-baseline.md`](day5-local-validation-baseline.md)
- [`day6-supplemental-validation-sweep.md`](day6-supplemental-validation-sweep.md)
- [`day7-hosted-ci-evidence-reconciliation.md`](day7-hosted-ci-evidence-reconciliation.md)
- [`day8-public-claim-audit-performance-report.md`](day8-public-claim-audit-performance-report.md)
- [`day9-public-claim-audit-package-abi-windows.md`](day9-public-claim-audit-package-abi-windows.md)
- [`day10-project-plan-reconciliation-part1.md`](day10-project-plan-reconciliation-part1.md)
- [`day11-project-plan-reconciliation-part2.md`](day11-project-plan-reconciliation-part2.md)

## Completion Check

- The retrospective draft is evidence-backed.
- Earned claims and non-claims are separated.
- The state-of-the-art assessment does not exceed evidence.
- Residual candidates are framed as future closure targets rather than vague
  aspirations.

## Validation

- Documentation/planning artifact only for Day 12.
- No `.c` or `.h` files were modified for this Day 12 draft.
- `git diff --check` passed after the artifact and working-notes update.
