# Sprint 137 Working Notes

## Sprint Goal

Freeze the post-Epic-11 baseline, select the gaps Epic 12 will close
completely, and create the evidence contracts that govern later implementation
sprints.

Sprint 137 is the Epic 12 intake and contract sprint. It must treat Epic 11
residuals as candidate inputs, not as preselected scope. Later sprints may only
promote corpus, QR, partial-SVD, report, runtime, package, platform, adoption,
or state-of-the-art wording after the required evidence, validation, and claim
gates are present.

## Starting Constraints

- Treat Epic 11 as closed on bounded evidence, not on broad state-of-the-art
  parity.
- Treat the Epic 11 residual queue as the source of candidate gaps for Epic 12.
- Prefer complete closure of selected gaps over partial progress on many gaps.
- Preserve static-first packaging, explicit platform tiers, local benchmark
  interpretation, generated-report boundaries, and state-of-the-art non-claims
  unless future sprints earn stronger proof.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only Sprint 137 changes require `git diff --check` and
  focused Markdown hygiene/link checks.

## Input Artifact Inventory

| Input | Role in Sprint 137 |
| --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 137 | Defines the baseline, residual reconciliation, gap selection, evidence templates, quality map, claim freeze, and closeout items. |
| `docs/planning/EPIC_12/SPRINT_137/PLAN.md` | Provides day-level execution order and 166-hour budget. |
| `docs/planning/EPIC_12/reviews/review-codex-2026-08-03.md` | Provides the current code review, measured signals, state-of-the-art assessment, and highest-value gap closures. |
| `docs/planning/EPIC_12/reviews/todo-codex-2026-08-03.md` | Provides the Epic 12 execution sequence and completion definition. |
| `docs/planning/EPIC_11/EPIC_11_RETROSPECTIVE.md` | Provides Epic 11 earned claims, non-claims, validation summary, and future-epic candidates. |
| `docs/planning/EPIC_11/SPRINT_136/artifacts/day12-residual-queue-publication.md` | Provides the post-Epic-11 residual queue, explicit non-claim register, QR residual queue, and promotion criteria. |
| `README.md` | Public front door for capabilities, workflow selection, support tiers, package/install claims, CI claims, and report interpretation. |
| `INSTALL.md` | Static-first install, downstream-consumer, package, ABI, and platform support truth. |
| `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/matrix_market.md` | Adoption, compressed-first, solver-selection, and Matrix Market documentation surfaces. |
| `docs/algorithm.md`, `docs/algorithm_history.md` | Current algorithm reference and historical measurement surfaces. |
| `docs/maintainer_guide.md` | Maintainer support-tier, validation, package/platform, report, benchmark, and non-claim policy surface. |
| `.github/workflows/*.yml` | Hosted Linux, macOS, Windows, package, sanitizer, dead-code, benchmark, and platform support-tier evidence. |
| `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `tests/test_install.sh`, `tests/test_cmake_install.sh` | Build, install, package, pkg-config, CMake export, and downstream-consumer proof surfaces. |
| `scripts/*.sh`, `scripts/*.py`, `benchmarks/README.md`, `benchmarks/*.c` | Report, benchmark, sentinel, guardrail, dead-code, and performance-governance evidence surfaces. |
| `src/`, `include/`, `tests/`, `examples/` | Source, public API, proof-owner, and adoption-example surfaces for Day 2 metrics and later gap selection. |

## Day-Level Ownership

| Day | Owner focus | Project-plan items |
| --- | --- | --- |
| 1 | Scope setup, artifact baseline, inherited input map, item ownership, validation expectations | Items 1-7 |
| 2 | Source/test/benchmark/example size and maintainability baseline | Item 1 |
| 3 | Build, package, CI, report, benchmark, and support-tier baseline | Item 1 |
| 4 | Epic 11 residual intake and candidate grouping | Item 2 |
| 5 | Residual owners, dependencies, promotion gates, and non-goals | Item 2 |
| 6 | Gap-selection scoring rubric and complete-closure criteria | Item 3 |
| 7 | Epic 12 selected gap decision for Sprints 138-146 | Item 3 |
| 8 | Corpus fixture, generated-matrix, optional-data, and oracle row templates | Item 4 |
| 9 | Report-index, normalized metadata, and stale-report templates | Item 4 |
| 10 | Package/ABI, platform promotion, downstream proof, and public claim templates | Item 4 |
| 11 | Quality surface map by touched documentation, scripts, build, CI, package, report, and C/header changes | Item 5 |
| 12 | Public claim freeze across README, INSTALL, docs, examples, benchmarks, and maintainer guide | Item 6 |
| 13 | Handoff synthesis and Sprint 138 corpus-readiness package | Item 7 |
| 14 | Sprint 137 closeout, residuals, validation summary, and working-notes completion | Item 7 |

## Initial Validation Expectations

| Change type | Required validation |
| --- | --- |
| Sprint 137 planning artifacts only | `git diff --check` plus focused Markdown link/path validation under `docs/planning/EPIC_12`. |
| Public documentation wording | `git diff --check`, focused Markdown link/path validation, and claim-boundary scan against Epic 11/Epic 12 non-claims. |
| Script or report-generator edits | Syntax validation for the touched script, focused report command where feasible, and support-tier/freshness semantics review. |
| Makefile, CMake, pkg-config, install, or package edits | Relevant package/install/CMake proof commands plus static/shared support-boundary review. |
| CI workflow edits | Workflow syntax or structural review plus hosted-runner support-tier notes; do not treat unrun hosted lanes as passed local evidence. |
| Benchmark or generated report execution | Capture command, platform, compiler/configuration, source commit, row meaning, freshness, support tier, and skip/defer status. |
| `.c` or `.h` edits | `make format && make lint && make test` after any focused tests needed for the touched implementation. |

## Inherited Claim Fences

| Claim family | Sprint 137 boundary |
| --- | --- |
| State of the art | Epic 12 starts from an explicit non-claim; no broad state-of-the-art wording is earned by planning artifacts. |
| Numerical corpus | Corpus work is a future maintained lane; current evidence remains fixture-local unless Sprint 138+ adds row semantics and proof. |
| QR evidence | Deferred QR residuals need trust value, output semantics, tolerance, rank/nullity, and support-tier metadata before promotion. |
| Partial SVD | Partial-SVD residuals need edge-case, convergence-budget, ordering, vector/subspace, tolerance, and skip semantics before promotion. |
| Report indexes | Generated reports are freshness and traceability evidence, not broad correctness, release, coverage, or performance proof. |
| Runtime/backend | Benchmark and sentinel rows are local measurement evidence; no portable performance, backend parity, OpenMP speedup, scalability, or memory claim is earned. |
| Package/ABI | Static-first remains maintained; shared-library packaging, dynamic ABI, runtime-loader behavior, and package-manager support remain non-claims. |
| Platform support | Linux is strongest; macOS and Windows install/export confidence remain supplemental; Windows pthread/POSIX tests remain staged. |
| Adoption docs | Navigation improvements do not create new solver behavior, package support, platform parity, or report schema claims. |
| Coverage/dead-code | Coverage remains supplemental; dead-code output is triage/report-completeness evidence, not removal-ready proof. |

## Day 1 Notes

- Created the Sprint 137 working-notes baseline and artifact directory.
- Re-read the Sprint 137 section of `docs/planning/EPIC_12/PROJECT_PLAN.md`.
- Re-read the Epic 12 review and gap-closure todo.
- Re-read the Epic 11 retrospective and Sprint 136 residual queue publication.
- Mapped Sprint 137 Items 1-7 to day-level owners across Days 1-14.
- Recorded inherited inputs before Day 2 metrics and Day 4 residual intake.
- Recorded validation expectations for documentation-only, public-doc,
  script/report, build-system, CI, benchmark/report, and C/header changes.
- No source files, public documentation, workflows, scripts, or support claims
  were changed on Day 1 beyond Sprint 137 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

