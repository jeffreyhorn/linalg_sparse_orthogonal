# Day 11 Claim Target Register

## Scope

Day 11 converts the selected Epic 14 targets into precise claim statements,
evidence owners, rejected broad claims, and documentation ownership. This
register is prospective: a claim is accepted as an Epic 14 target only when the
named sprint lands its evidence contract and validation. Until then, current
public wording must keep the existing non-claims.

## Accepted Target Claim Register

| Claim ID | Accepted Epic 14 target claim | Target | Sprint | Evidence owner | Required evidence before claim is earned | Docs that must move together |
| --- | --- | --- | --- | --- | --- | --- |
| C157-01 | The API reference policy is explicit and reviewable: generated HTML is either published with freshness evidence or retained as guarded local-only output. | T157-01 | 158 | Documentation/API owner | `make docs`, warning triage, public-header page coverage, generated `sparse_version.h` policy, and publication decision artifact. | `docs/api_reference.md`, `docs/maintainer_guide.md`, README API/docs links, selected public-header comments. |
| C157-02 | Selected QR, partial-SVD, oracle, and comparison generated freshness has a reviewed hosted evidence path. | T157-02 | 159 | CI, corpus, comparison, and report owners | Hosted selected freshness lane, selected family list, runtime budget, artifact upload or deterministic summary, row count/freshness result, support-tier docs update. | `.github/workflows/*.yml`, `docs/maintainer_guide.md`, `tests/corpus/README.md`, report-family manifest, README support-tier text. |
| C157-03 | One additional QR comparison family is fixture-local, metric-bound, freshness-checked, and documented. | T157-03 | 160 | QR and comparison owners | Fixture selection, basis-invariant metrics, tolerances, dependency/provenance record, comparison freshness pass, normalized rows, focused tests if behavior changes. | `docs/solver_selection.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md`, comparison report-family rows, README if summary changes. |
| C157-04 | One partial-SVD comparison family is subspace-safe, freshness-checked, and documented. | T157-04 | 161 | SVD and comparison owners | Fixture selection, singular-value/projector/residual/orthogonality/fail-closed metrics, dependency status, row counts, selected freshness pass, focused tests if behavior changes. | README SVD sections, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md`. |
| C157-05 | The Windows package boundary is decided: selected parity is reviewed, or non-parity is explicitly retained with stronger checks and wording. | T157-05 | 162 | Platform/package owner | Windows package parity audit, product-scope decision, hosted proof or rejection guard, metadata checks, and docs/workflow synchronization. | `.github/workflows/windows-ci.yml`, README cross-platform contract, `INSTALL.md`, `docs/maintainer_guide.md`, package metadata comments. |
| C157-06 | Selected performance/report rows are published with methodology fields and explicit non-superiority caveats. | T157-06 | 163 | Benchmark and report owners | Selected report subset, methodology fields, row classification, command output, report artifact, validation summary. | `benchmarks/README.md`, README performance/report wording, `docs/maintainer_guide.md`, report schema docs. |
| C157-07 | A selected public-header batch is clearer without accidental API signature drift. | T157-07 | 164 | Header owners | Header selection, before/after normalized declarations, zero declaration diff or explicit API review, generated-doc policy application, required C/header quality gate. | Selected `include/*.h`, README, tutorial, cookbook, solver-selection, API reference, maintainer guide. |
| C157-08 | The static-first package boundary is hardened and shared-library/dynamic ABI deferrals remain test-backed. | T157-08 | 165 | Package/ABI owner | Package metadata audit, static deferral guard, install/export scripts, downstream consumer proof, docs claim audit. | README, `INSTALL.md`, `docs/maintainer_guide.md`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, workflow comments. |
| C157-09 | Final Epic 14 public claims are mapped to evidence, and remaining gaps are residualized or rejected. | T157-09 | 166 | Epic/product owner | Final evidence inventory, validation baseline, hosted CI reconciliation, public claim scan, project-plan reconciliation, retrospective, residual queue. | README, `INSTALL.md`, API docs, solver docs, benchmark docs, corpus docs, maintainer guide, project plan, retrospective. |

## Explicit Rejected Claim Register

| Rejected claim | Current status | Why rejected | Required future evidence |
| --- | --- | --- | --- |
| Unqualified state-of-the-art sparse linear algebra status | Rejected throughout Epic 14. | Current evidence is bounded by selected fixtures, static-first package support, local/hosted lanes, and narrow comparisons. | Broad recurring correctness, external parity, performance, platform, package, ABI, and release evidence. |
| Broad external ecosystem parity | Rejected except for selected bounded comparison families. | Epic 14 selects one QR and one partial-SVD family, not general LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, NumPy, or SciPy parity. | Multiple dependency-backed comparison families with provenance, tolerances, hosted freshness, and docs. |
| Portable performance superiority | Rejected. | Benchmark rows are local, methodology-bound, supplemental, advisory, or selected hard gates, not cross-machine superiority evidence. | Recurring benchmark matrix, variance policy, competitive baselines, thresholds, hosted publication, and claim audit. |
| Package-manager distribution | Rejected. | No external package recipe owners, release workflow, update/uninstall policy, channel validation, or support commitment. | Selected package-manager recipes, release ownership, install/uninstall validation, version policy, and support docs. |
| Full shared-library support | Rejected. | Shared build/install product, export/import macros, symbol visibility, loader metadata, shared consumers, and runtime-loader validation are absent. | Cross-platform shared-library product decision, symbol allowlist, loader metadata, installed shared consumer tests, and CI proof. |
| Dynamic ABI compatibility | Rejected. | Public structs, callbacks, enum values, allocator/lifetime rules, error state, and version metadata lack an ABI policy. | ABI stability level, compatibility window, binary compatibility tests, exported-header audit, and release checks. |
| Runtime-loader behavior | Rejected. | No shared-library loader contract or installed shared consumer proof exists. | Runtime-loader matrix across Linux, macOS, and Windows with shared artifacts and downstream consumers. |
| Broad Windows platform parity | Rejected. | Windows is reviewed CMake-first and package support is narrower than Unix-side Make/`pkg-config` proof. | Explicitly selected Windows Makefile and/or `pkg-config` product scope plus hosted proof and docs. |
| Windows Makefile parity | Rejected unless Sprint 162 explicitly selects and proves it. | Current Windows workflows do not run Makefile install/uninstall or reviewed Make wrappers. | Hosted Windows Makefile implementation and validation with install/uninstall semantics. |
| Windows `pkg-config` execution parity | Rejected unless Sprint 162 explicitly selects and proves it. | Current Windows lane inspects `sparse.pc` metadata but does not execute a Windows `pkg-config` downstream compile/link/run proof. | Provider selection, path/link policy, hosted compile/link/run proof, and docs updates. |
| Generated local files as pass evidence | Rejected. | Ignored outputs under `build/`, `docs/api/`, and `coverage/` depend on local context and freshness. | Selected publication or hosted freshness gate with support-tier update. |
| Coverage or dead-code rows as solver correctness proof | Rejected. | These rows are supplemental/advisory quality signals, not numerical behavior proof. | Separate selected correctness tests or hosted claim-bearing report semantics. |

## Evidence Owner Table

| Claim surface | Evidence owner | Required recurring check or artifact |
| --- | --- | --- |
| API docs publication | Documentation/API owner | `make docs`, warning triage, page coverage, publication policy artifact. |
| Hosted generated evidence | CI, corpus, comparison, and report owners | Hosted selected freshness lane and artifact/summary policy. |
| QR comparison | QR and comparison owners | Selected comparison freshness, normalized rows, metric contract, focused tests if touched. |
| Partial-SVD comparison | SVD and comparison owners | Subspace-safe comparison freshness, normalized rows, metric contract, focused tests if touched. |
| Windows package decision | Platform/package owner | Product decision, hosted proof or rejection guard, workflow/docs synchronization. |
| Performance publication | Benchmark and report owners | Selected report command, methodology artifact, row classification, docs caveats. |
| Header/API cleanup | Header owners | Declaration preservation, generated-doc policy, `make format && make lint && make test` for header edits. |
| Static package boundary | Package/ABI owner | Install/export scripts, static deferral guard, metadata audit, downstream proof. |
| Final claim audit | Epic/product owner | Public docs scan, evidence-owner mapping, hosted reconciliation, residual queue. |

## Documentation Ownership Checklist

| Docs surface | Owns | Must be updated when |
| --- | --- | --- |
| README | Public front door, compact support claims, install summary, performance/report summary, known limits. | Any user-facing claim changes for API docs, hosted evidence, comparison, package, platform, performance, or state-of-the-art posture. |
| `INSTALL.md` | Operational install, package, platform, static-first, Windows package, shared-library, and ABI boundaries. | Package metadata, install/export validation, Windows package decision, static/shared support, or ABI wording changes. |
| `docs/api_reference.md` | API reference entry point and generated HTML/source-header-first policy. | Generated API docs policy, public-header routing, or API docs publication changes. |
| `docs/maintainer_guide.md` | Detailed support-tier, evidence, claim, package, generated-report, benchmark, and maintainer policy. | Any accepted or rejected claim changes. |
| `docs/solver_selection.md` | Solver-family claim limits and QR/SVD evidence routing. | QR or partial-SVD comparison/corpus evidence changes. |
| `docs/tutorial.md` and `docs/cookbook.md` | First-use and workflow learning paths. | SVD/QR workflow wording, API adoption guidance, or header cleanup changes affect user flow. |
| `benchmarks/README.md` | Benchmark/report command semantics and performance non-claims. | Performance report rows, sentinels, guardrails, or methodology wording changes. |
| `tests/corpus/README.md` | Corpus, expected-row, oracle, comparison, generated-report, skip/defer, and non-claim semantics. | Corpus rows, oracle/comparison freshness, report-family rows, or selected hosted generated evidence changes. |
| `.github/workflows/*.yml` comments | Hosted lane support-tier truth. | CI lane scope, artifact semantics, expected counts, platform support, package proof, or generated evidence changes. |
| `sparse.pc.in` and CMake package comments/templates | Package metadata truth. | Static package metadata, link flags, version behavior, static/shared selectors, or unsupported package/ABI wording changes. |

## Claim Change Checklist

Before a later sprint widens or closes a claim, it must record:

1. The accepted claim ID or rejected claim being changed.
2. The exact source files, docs, workflow, manifest, script, or metadata owners.
3. The command, hosted lane, generated artifact, or product decision that proves
   the change.
4. The support tier after the change.
5. The public docs that were updated or intentionally left unchanged.
6. The non-claims preserved after the change.
7. The validation commands actually run.
8. Any residuals left behind with owner, blocker, prerequisite, and promotion
   gate.

## Day 12 Inputs

Day 12 should use this register to build the risk and Sprint 158 handoff:

- risks should reference claim IDs where overclaiming is possible;
- the Sprint 158 handoff should use C157-01 as its primary target;
- generated-output tracking, Doxygen warnings, public-header page coverage, and
  source-header-first policy should be treated as Sprint 158 stop conditions;
- claim ownership should guide which docs must move with the generated API
  docs decision.

## Completion Check

- Every accepted target claim has a planned proof owner and required evidence.
- Rejected claims are explicit enough for later audits.
- Public docs have known ownership for claim wording.
- Claim changes have a repeatable checklist for later sprints.
