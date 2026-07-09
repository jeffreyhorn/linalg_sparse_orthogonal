# Sprint 117 Day 7 Final Comparison Package Inventory

## Purpose

Day 7 inventories the final Epic 10 comparison package for solver, reorder,
benchmark, coverage, and package evidence. It classifies each artifact as public
claim evidence, local measurement context, supplemental proof, or residual
background before Day 8 public-claim cleanup.

## Regeneration Decision

No Day 7 comparison artifact was regenerated. Sprint 117 has changed planning
documentation only, and Day 5 already ran the strongest local reviewed baseline:
`make quality-review-full` passed with Makefile review checks, CMake parity,
`54` registered CMake tests, and `54 / 54` CTest passes.

The current workspace has Day 5 dead-code outputs under `build/deadcode/`, but
no fresh `build/bench-reports/` or `coverage/` report files are present for
Day 7. Historic benchmark, guardrail, package, and comparison artifacts are
therefore recorded as evidence provenance, not as newly regenerated Day 7
outputs.

## Final Comparison Surface Inventory

| Surface | Artifact owners | Final artifacts | Classification | Command provenance | Day 8 boundary |
|---|---|---|---|---|---|
| Direct solver external oracles | Sprint 102; `tests/test_chol_csc.c`; `tests/test_ldlt_csc.c`; `tests/test_sparse_lu.c`; direct-solver helper scripts | `SPRINT_102/artifacts/day14-closeout-and-handoff.md`; Sprint 102 fixture-taxonomy and CSC/general-expansion artifacts; maintainer-guide direct-solver evidence tables | Public claim evidence, bounded to selected direct-solver families and fixtures | Sprint 102 focused helper/test commands and `make format && make lint && make test`; Day 5 current `make quality-review-full` | Keep claims selected and family-local. Do not claim broad external parity for every direct solver path. |
| Iterative, eigensolver, and SVD comparisons | Sprints 103, 113, and 114; iterative/eigensolver/SVD test owners | `SPRINT_103/artifacts/day14-closeout-and-handoff.md`; `SPRINT_113/artifacts/day14-closeout-and-handoff.md`; `SPRINT_114/artifacts/day14-validation-metrics-and-handoff.md` | Public claim evidence for fixture-local residual, reconstruction, and exact-case behavior; residual background for broad external parity | Sprint 103 full quality gate; Sprint 113/114 focused proof-owner validation and full quality gates; Day 5 current `make quality-review-full` | Preserve bounded residual/reconstruction language. Do not imply ARPACK, LAPACK, SciPy, PETSc, or Trilinos parity. |
| Backend, runtime, and performance sentinels | Sprint 104; benchmark and runtime maintainers | `SPRINT_104/artifacts/day14-closeout-and-handoff.md`; Sprint 104 sentinel/reporting artifacts | Local measurement context and supplemental proof | Sprint 104 `make bench-canonical-report` and `make performance-sentinels`; not regenerated on Day 7 | Treat timing rows as local calibration only. No portable performance superiority claim. |
| Reorder, fill, graph, and large-matrix guardrails | Sprint 105; reorder/graph benchmark owners | `SPRINT_105/artifacts/day14-closeout-and-handoff.md`; Sprint 105 guardrail contract, named-matrix, and implementation artifacts | Public claim evidence for governed report/guardrail existence; local measurement context for timing and memory | Sprint 105 `make large-matrix-guardrails` and full quality gate; Day 5 current test baseline only | Do not claim universal reorder/fill superiority or cross-platform max-RSS comparability. |
| Package, install, ABI, and platform tiers | Sprints 112 and 115; install/package maintainers | `SPRINT_112/artifacts/day14-closeout-handoff.md`; `SPRINT_115/artifacts/day14-validation-package-platform-handoff.md` | Public claim evidence for static-first package support and tiered platform scope; supplemental proof for local install scripts | Sprint 112 `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh`; not rerun on Day 7 because package surfaces were untouched | Keep static-first and tiered-platform wording. Do not claim shared-library ABI, package-manager support, Windows install parity, or full macOS install/export parity. |
| Coverage and dead-code completeness | Day 5 validation; Makefile coverage/dead-code targets | `SPRINT_117/artifacts/day5-validation-execution.md`; `SPRINT_117/artifacts/day6-final-validation-package.md`; `build/deadcode/report.md`; `build/deadcode/report.tsv` | Reviewed completeness support for dead-code reporting; coverage is residual background because no fresh coverage report exists | Day 5 `make quality-review-full` included `deadcode-check`; no `make coverage` run in Sprint 117 | Cite dead-code as supporting completeness context only. Do not cite absent coverage output as current proof. |
| Adoption and public-claim guardrails | Sprint 116; Sprint 117 Days 2-3 | `SPRINT_116/artifacts/day14-validation-handoff.md`; `SPRINT_117/artifacts/day2-end-state-claim-inventory.md`; `SPRINT_117/artifacts/day3-end-state-claim-decision.md` | Public claim boundary evidence | Sprint 116 adoption-surface validation; Sprint 117 Day 5 current validation baseline | Use Day 8 to re-check public wording against earned, partial, deferred, and non-claim decisions. |
| Final validation package | Sprint 117 Days 4-6 | `SPRINT_117/artifacts/day4-full-validation-design.md`; `SPRINT_117/artifacts/day5-validation-execution.md`; `SPRINT_117/artifacts/day6-final-validation-package.md` | Reviewed validation evidence for the current branch | Day 5 docs hygiene and `make quality-review-full` | Local macOS validation supports the branch state but does not replace CI-owned platform proof. |

## Command Provenance Table

| Command or check | Source | Status | Inventory use |
|---|---|---|---|
| `git diff --check` | Sprint 117 Day 5 | Passed | Current branch documentation hygiene. |
| `rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_117` | Sprint 117 Day 5 | Passed with no matches | Current Sprint 117 whitespace hygiene. |
| `make quality-review-full` | Sprint 117 Day 5 | Passed | Current strongest local reviewed validation baseline. |
| Direct-solver focused helper and test commands | Sprint 102 | Passed in Sprint 102 | Historic bounded direct-solver oracle provenance. |
| `make format && make lint && make test` | Sprints 102, 103, 113, 114 | Passed in those sprints | Historic proof-owner and comparison validation provenance. |
| `make bench-canonical-report` | Sprint 104 | Passed in Sprint 104 | Historic benchmark report provenance; not fresh Day 7 evidence. |
| `make performance-sentinels` | Sprint 104 | Passed in Sprint 104 | Historic local sentinel provenance; not portable performance proof. |
| `make large-matrix-guardrails` | Sprint 105 | Passed in Sprint 105 | Historic reorder/fill guardrail provenance. |
| `bash tests/test_install.sh` | Sprint 112 | Passed in Sprint 112 | Historic static-first install proof provenance. |
| `bash tests/test_cmake_install.sh` | Sprint 112 | Passed in Sprint 112 | Historic CMake install/export proof provenance. |
| `make coverage` | Sprint 117 | Not run | No fresh coverage claim is available for Day 7. |
| Benchmark report regeneration | Sprint 117 | Not run | No benchmark or report surface changed; no fresh report is cited. |

## Public Claim Versus Local Evidence Classification

| Classification | Included evidence | Public wording rule |
|---|---|---|
| Public claim evidence | Direct-solver bounded oracle tests; fixture-local iterative/eigensolver/SVD residual and reconstruction checks; static-first package support; tiered platform scope; adoption claim guardrails | May support public docs only when scope, fixtures, and platform tiers are explicit. |
| Local measurement context | Performance sentinels, benchmark reports, reorder/fill timing, local max-RSS observations | May explain local regression monitoring, not portable speed or superiority. |
| Supplemental proof | Install scripts, CMake install/export checks, optional backend/runtime lanes, guardrail bundles | May support maintainer confidence when labeled supplemental or local. |
| Residual background | Deferred package-manager support, shared-library ABI, Windows install parity, broad ecosystem solver parity, absent coverage refresh | Must remain residual or non-claim until implemented and validated. |

## Day 8 Package Checklist

- Re-check public docs and adoption surfaces against Day 3 claim decisions.
- Preserve selected/family-local wording for direct solver, iterative,
  eigensolver, and SVD comparison evidence.
- Treat benchmark and performance artifacts as local measurement context only.
- Keep package/platform claims static-first and tiered.
- Do not cite absent Day 7 benchmark or coverage outputs as refreshed evidence.
- Carry CI-owned Linux, macOS, and Windows platform proof separately from local
  Day 5 validation.
- Record any public wording cleanup as a Day 8 claim-calibration change.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| All final comparison surfaces have artifact owners. | Complete. |
| Regenerated evidence is classified before public claim cleanup. | Complete; no regeneration was required, and historic/current evidence is classified. |
| No local benchmark artifact is treated as portable performance proof. | Complete. |
