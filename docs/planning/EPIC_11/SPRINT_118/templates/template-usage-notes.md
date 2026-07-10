# Sprint 118 Evidence Template Usage Notes

## Purpose

These templates are reusable planning and closeout artifacts for Sprints
119-127. They preserve the Sprint 118 product truth map, residual owner map,
hotspot handoff, and non-claim discipline while implementation sprints move
source boundaries, expand oracles, adjust performance/report lanes, decide
package/ABI support, and simplify adoption surfaces.

## Template Selection

| Work type | Use this template |
|---|---|
| Source movement, private-owner extraction, internal-header reshaping, or giant-test split | `source-movement-evidence-template.md` |
| Direct, iterative, eigensolver, SVD, QR, rank, corpus, dense-reference, external-reference, or cross-solver proof expansion | `oracle-expansion-evidence-template.md` |
| Benchmark, report-index, coverage/report interpretation, performance sentinel, backend/runtime, OpenMP, or local timing work | `performance-sentinel-evidence-template.md` |
| Static-first package decision, shared-library support, ABI policy, install/export proof, package metadata, platform tier, or package-manager disposition | `package-abi-decision-template.md` |
| README, install docs, solver-selection docs, algorithm docs, examples, header wording, cookbook routes, public claim cleanup, or link/path cleanup | `adoption-cleanup-evidence-template.md` |

When a sprint touches multiple work types, fill the primary template and add
cross-links to the secondary template sections that matter. Do not collapse
correctness, performance, package, and public-claim evidence into one
undifferentiated note.

## Required Inputs

Future sprints should cite these Sprint 118 artifacts before filling a
template:

| Sprint 118 artifact | Use |
|---|---|
| `artifacts/day3-baseline-quality-recheck.md` | Baseline validation, CTest count, and Makefile/CMake parity evidence. |
| `artifacts/day4-ci-tier-platform-truth.md` | Platform tier, install, package, and staged-exclusion truth. |
| `artifacts/day6-residual-owner-map.md` | Residual owners, dependencies, proof gates, and future-epic deferrals. |
| `artifacts/day8-product-truth-map.md` | Current baseline claims, candidate claims, explicit non-claims, and evidence references. |
| `artifacts/day9-hotspot-metrics.md` | Source/test/file-count metrics and raw command reproducibility. |
| `artifacts/day10-hotspot-owner-handoff.md` | Ranked source/test movement candidates and no-move/defer guidance. |
| `artifacts/day11-evidence-template-design.md` | Design rationale for the refreshed template set. |

## Validation Rules

| Touched surface | Required validation |
|---|---|
| Documentation-only planning artifacts | `git diff --check` and focused trailing-whitespace scan over touched docs. |
| Public docs or support wording | Claim-boundary scan against the Day 8 product truth map and explicit non-claims. |
| `.c` or `.h` files | `make format && make lint && make test`. |
| Makefile, CMake, workflow, package, script, benchmark, or install surfaces | Relevant focused validation lane plus reviewed/supplemental/local classification. |
| CTest membership or expected-count changes | `ctest -N` or equivalent count proof, Makefile/CMake parity if relevant, and support wording updates if platform-facing. |
| Benchmark/report/performance changes | Focused report or benchmark command, local context, threshold/report-only status, and non-portable interpretation. |
| Package/platform changes | Install/export/downstream consumer proof, platform tier impact, staged exclusions, and non-claim updates. |

## Claim Discipline

Every filled template should answer:

1. What current baseline claim does this work preserve?
2. What candidate claim does this work attempt to earn?
3. What evidence proves the candidate claim, if any?
4. What public wording changed?
5. What explicit non-claims remain after success?
6. What residuals are deferred, and who owns them next?

Passing a focused lane does not automatically create broad ecosystem parity,
portable performance superiority, shared-library ABI support, package-manager
support, GPU support, distributed-memory support, or symmetric platform parity.

## Handoff Rules

- Fill templates before broad implementation where the template is being used
  as a design gate.
- Update templates after validation with observed commands, counts, and
  residuals.
- Link filled templates from sprint working notes and retrospectives.
- Carry incomplete template sections into residual deferred debt rather than
  dropping them.
- If a template reveals unclear requirements or failing required validation,
  stop and resolve before promoting claims or closing the sprint day.

## Sprint Owner Map

| Sprint | Expected template use |
|---|---|
| 119 | Source movement template for eigensolver private-owner movement and any deferred movement decisions. |
| 120 | Source movement template for direct/iterative test splits; oracle template for generated-RHS or dense-reference pilots. |
| 121 | Oracle template for SVD/QR/rank evidence; source movement template only if helper extraction changes source ownership. |
| 122 | Oracle template for corpus taxonomy; performance template for report-index, coverage, and report classification work. |
| 123 | Performance template for sentinel/backend/runtime work; source movement template only for graph/reorder ownership changes. |
| 124 | Package/ABI decision template for static-first or shared-library decisions and package-manager disposition. |
| 125 | Package/ABI decision template for platform install/export and staged-lane updates. |
| 126 | Adoption cleanup template for docs, examples, cookbook, and claim-boundary simplification. |
| 127 | Adoption cleanup and package/performance/oracle templates as needed for final claim recalibration and residual publication. |
