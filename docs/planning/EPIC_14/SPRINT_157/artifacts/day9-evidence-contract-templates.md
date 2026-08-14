# Day 9 Evidence Contract Templates

## Scope

Day 9 defines reusable evidence formats for the Epic 14 targets selected on
Day 8. These templates are intentionally narrow. They make later sprint
artifacts comparable and keep pass evidence separate from advisory output,
generated-local files, planning notes, and unsupported public claims.

## Shared Evidence Contract Fields

Every selected Epic 14 sprint should include the fields below, either in one
artifact or across clearly linked artifacts.

| Field | Required content |
| --- | --- |
| Target | Day 8 target ID and sprint number. |
| Claim surface | The exact public or maintainer claim being changed, preserved, or rejected. |
| Source owner | Source files, docs, manifests, workflow, or scripts that own the behavior. |
| Evidence owner | Command, test, CI lane, generated artifact, or review artifact that proves the target. |
| Support tier | Reviewed, hosted, local-only, supplemental, advisory, deferred, or retained non-claim. |
| Freshness | Commit, branch, command, platform, compiler, generated timestamp, or hosted run context needed to interpret evidence. |
| Pass evidence | Binary pass/fail command output, hosted lane result, artifact checksum/row count, or explicit product decision. |
| Advisory output | Local timing rows, coverage summaries, dead-code rows, optional-data skips, generated-local files, or planning notes. |
| Claim update | Exact docs or metadata that may change if the evidence passes. |
| Non-claims | Claims that remain rejected even after the target closes. |
| Stop condition | Condition that requires narrowing scope, retaining the non-claim, or asking for review. |

## Template 1: API Documentation Publication

| Field | Template |
| --- | --- |
| Target | `T157-01`, Sprint 158. |
| Claim surface | Generated API reference availability and freshness. |
| Source owner | `Doxyfile`, `docs/api_reference.md`, `docs/maintainer_guide.md`, README API/docs links, public headers under `include/`, `include/sparse_version.h.in`. |
| Evidence owner | `make docs`, Doxygen warning log, generated-page inventory, page-coverage check, publication decision artifact. |
| Support tier | Source-header-first; generated HTML is either published evidence, CI-published artifact, or guarded local-only output. |
| Required pass evidence | Command used, Doxygen warning count and triage, intended public-header page coverage result, generated `sparse_version.h` policy, and selected publication decision. |
| Advisory output | Local `docs/api/html/` files before the publication decision, untriaged warnings, screenshots, or planning notes. |
| Claim update | `docs/api_reference.md`, `docs/maintainer_guide.md`, README links, and any generated-output review guidance. |
| Non-claims | No dynamic ABI, shared-library support, package-manager distribution, broad platform parity, external parity, portable performance, or state-of-the-art coverage. |
| Stop condition | Generated pages are missing intended public headers, Doxygen warnings are unexplained, or generated output is committed without a matching policy decision. |

## Template 2: Hosted Generated Report Promotion

| Field | Template |
| --- | --- |
| Target | `T157-02`, Sprint 159. |
| Claim surface | Selected QR, partial-SVD, oracle, and comparison generated freshness as reviewed hosted evidence. |
| Source owner | `tests/corpus/**`, `tests/corpus/manifests/report_families.tsv`, `scripts/run_corpus_oracle.py`, `scripts/run_external_comparison.py`, `scripts/normalize_report_index.py`, Makefile freshness targets, CI workflows. |
| Evidence owner | Hosted CI lane running selected freshness gates, artifact upload or deterministic summary, local preflight commands. |
| Support tier | Hosted reviewed for selected families only; local-only/advisory for non-selected families. |
| Required pass evidence | Selected family list, runtime budget, command list, hosted run result, artifact retention policy, stale/missing/failing row semantics, row counts, and support-tier docs update. |
| Advisory output | Local ignored files under `build/corpus/`, `build/corpus-reports/`, `build/comparison/`, and `build/report-index/` unless tied to a hosted run or explicit publication policy. |
| Claim update | `docs/maintainer_guide.md`, `tests/corpus/README.md`, `benchmarks/README.md` if comparison/report wording changes, README support-tier text, report-family rows. |
| Non-claims | No broad QR/SVD correctness, external-library parity, hosted proof for advisory families, platform portability proof, or performance proof. |
| Stop condition | Hosted runtime is unstable, selected rows are stale/missing/failing, artifact policy is ambiguous, or advisory families are accidentally promoted. |

## Template 3: QR Comparison Evidence

| Field | Template |
| --- | --- |
| Target | `T157-03`, Sprint 160. |
| Claim surface | One bounded QR comparison family beyond the current minimum-norm fixture. |
| Source owner | Selected QR corpus fixtures/expected rows, `tests/qr_external_dense_reference.py`, `scripts/run_external_comparison.py`, comparison report-family rows, QR tests when implementation behavior changes. |
| Evidence owner | Comparison harness self-check, selected comparison freshness command, normalized comparison rows, focused script/C tests if touched. |
| Support tier | Fixture-local comparison evidence for the selected family only. |
| Required pass evidence | Fixture selection rationale, metric contract, tolerance policy, dependency/provenance record, skip/defer semantics, generated row count, freshness result, and docs update. |
| Advisory output | Raw basis vectors, local unnormalized scratch output, optional dependency skips without selected policy, or timing rows. |
| Claim update | QR sections in `docs/solver_selection.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md`, README if public summary changes. |
| Non-claims | No broad QR parity, raw Q-basis identity, global rank-threshold policy, broad rank-deficient solve coverage, broad SuiteSparse corpus coverage, package/ABI proof, or performance proof. |
| Stop condition | Metrics depend on unstable raw basis identity, dependency provenance is missing, tolerances are unexplained, or docs imply broad QR parity. |

## Template 4: Partial-SVD Comparison Evidence

| Field | Template |
| --- | --- |
| Target | `T157-04`, Sprint 161. |
| Claim surface | First bounded partial-SVD comparison publication. |
| Source owner | Selected partial-SVD corpus fixtures/expected rows, `tests/svd_external_dense_reference.py`, comparison runner, normalizer, SVD tests when implementation behavior changes. |
| Evidence owner | Comparison harness, selected oracle/comparison freshness commands, normalizer checks, focused tests. |
| Support tier | Fixture-local partial-SVD comparison evidence for one selected subspace-safe family. |
| Required pass evidence | Fixture family, singular-value metrics, projector/subspace metrics, residual and orthogonality metrics, convergence/fail-closed fields, dependency status, row counts, freshness result, docs update. |
| Advisory output | Raw singular-vector identity, local scratch output, optional skipped baselines, timing-only rows. |
| Claim update | SVD sections in README, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md`. |
| Non-claims | No broad SVD parity, raw vector identity parity, convergence-rate claim, sparse-output/drop-tolerance optimality, hosted proof beyond selected lane, performance claim, platform claim, or state-of-the-art claim. |
| Stop condition | Metrics are not subspace-safe, tight-budget failures produce ambiguous partial results, optional baselines are treated as pass evidence, or docs imply broad SVD parity. |

## Template 5: Windows Package Parity Decision

| Field | Template |
| --- | --- |
| Target | `T157-05`, Sprint 162. |
| Claim surface | Windows package support boundary for Makefile and `pkg-config` parity. |
| Source owner | `.github/workflows/windows-ci.yml`, `INSTALL.md`, README, `docs/maintainer_guide.md`, `tests/test_install.sh`, `tests/test_cmake_install.sh`, `sparse.pc.in`, CMake package templates. |
| Evidence owner | Product decision artifact, hosted Windows CMake install/downstream lane, selected new proof or rejection guard, docs alignment. |
| Support tier | Reviewed Windows CMake install/downstream validation; Windows Makefile and `pkg-config` execution parity only if selected and proven. |
| Required pass evidence | Parity audit, selected product scope, provider/toolchain decision if promoting, hosted compile/link/run proof or strengthened non-claim guard, metadata checks, docs/workflow updates. |
| Advisory output | Installed `sparse.pc` metadata on Windows without `pkg-config` execution, local-only Windows experiments, or package-manager hints. |
| Claim update | Windows CI comments, README cross-platform contract, `INSTALL.md`, `docs/maintainer_guide.md`, package metadata comments. |
| Non-claims | No package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |
| Stop condition | Provider choice is unclear, hosted proof is flaky, Makefile and `pkg-config` parity are conflated, or docs imply package-manager/shared-library support. |

## Template 6: Methodology-Bound Performance Publication

| Field | Template |
| --- | --- |
| Target | `T157-06`, Sprint 163. |
| Claim surface | Selected benchmark/report rows as methodology-bound evidence. |
| Source owner | `benchmarks/`, benchmark scripts, `benchmarks/README.md`, `docs/maintainer_guide.md`, report-family rows, Makefile benchmark/report targets. |
| Evidence owner | Selected benchmark/report command, sentinel command, generated methodology artifact, validation summary. |
| Support tier | Local-only, hosted, supplemental, advisory, or hard-gate per selected row. |
| Required pass evidence | Selected surface list, platform, compiler, build mode, thread count, backend, fixture, repeats, variance/caveat fields, threshold classification, command output, report artifact. |
| Advisory output | Threshold-free timing rows, local-only machine snapshots, coverage/dead-code rows, skipped guardrails unless selected. |
| Claim update | `benchmarks/README.md`, README performance/report wording, `docs/maintainer_guide.md`, report schema docs. |
| Non-claims | No portable performance superiority, backend superiority, release benchmark claim, broad scalability claim, or state-of-the-art performance claim. |
| Stop condition | Methodology fields are missing, runtime is unstable, thresholds are implicit, or docs imply portability beyond the selected environment. |

## Template 7: Public Header Declaration Preservation

| Field | Template |
| --- | --- |
| Target | `T157-07`, Sprint 164. |
| Claim surface | Public API/header usability and documentation coherence without accidental signature drift. |
| Source owner | Selected `include/*.h`, README, tutorial, cookbook, solver-selection, API reference, maintainer guide. |
| Evidence owner | Before/after normalized declaration capture, declaration diff, quality gates, generated-doc policy application. |
| Support tier | Source-controlled API/docs evidence; generated HTML handled according to Sprint 158 policy. |
| Required pass evidence | Header selection rationale, baseline declarations, edited headers/docs, after declarations, zero diff or explicit API-review note, required C/header quality gate if headers changed. |
| Advisory output | Comment-only intent without declaration capture, generated HTML not tied to Sprint 158 policy, planning-only notes. |
| Claim update | Header comments, docs cross-links, API reference routing, maintainer guide ownership notes. |
| Non-claims | No ABI stability promise, package guarantee, platform parity, solver correctness expansion, or generated HTML completeness unless separately proven. |
| Stop condition | Normalized declarations drift unexpectedly, header wording implies ABI or broad support, or C/header gates are not run after header edits. |

## Template 8: Static-First Package Boundary Hardening

| Field | Template |
| --- | --- |
| Target | `T157-08`, Sprint 165. |
| Claim surface | Static-first package support and shared-library/dynamic ABI non-claims. |
| Source owner | `CMakeLists.txt`, `Makefile`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, install scripts, static deferral guard, README, `INSTALL.md`, maintainer guide, workflow comments. |
| Evidence owner | Package metadata audit, static deferral guard, install/export scripts, downstream consumer proof, docs claim audit. |
| Support tier | Reviewed static-first package support; shared-library and dynamic ABI remain deferred unless separately funded. |
| Required pass evidence | Metadata audit result, `BUILD_SHARED_LIBS=ON` rejection guard result, unsupported metadata absence, install/export proof, downstream consumer proof, docs synchronization list. |
| Advisory output | Package proof-owner rows without a fresh install run, local package experiments, package-manager wording, generated install trees outside validation context. |
| Claim update | README, `INSTALL.md`, maintainer guide, package comments, workflow comments, static deferral guard wording. |
| Non-claims | No package-manager distribution, shared-library support, dynamic ABI compatibility, runtime-loader behavior, static/shared selectors, or broad platform parity. |
| Stop condition | Shared metadata appears without product decision, install proof fails, docs imply ABI stability, or Windows/Unix package surfaces are conflated. |

## Template 9: Final Claim Audit And Residual Publication

| Field | Template |
| --- | --- |
| Target | `T157-09`, Sprint 166. |
| Claim surface | Final Epic 14 public claim posture and residual queue. |
| Source owner | All selected sprint artifacts, public docs, maintainer guide, install docs, benchmark docs, corpus docs, workflows, project plan. |
| Evidence owner | Final validation baseline, hosted CI reconciliation, public claim scan, project-plan reconciliation, Epic retrospective, residual queue. |
| Support tier | Earned evidence per selected target; retained non-claim for unsupported broad claims. |
| Required pass evidence | Evidence inventory, validation commands and results, hosted run references or explicit limitations, claim/non-claim register, updated residual queue. |
| Advisory output | Planning artifacts without validation, local generated rows not promoted, historical measurements, optional/skipped data. |
| Claim update | README, `INSTALL.md`, API docs, solver docs, benchmark docs, maintainer guide, project plan, retrospective. |
| Non-claims | Any unsupported state-of-the-art, broad external parity, portable performance, package-manager, shared-library, dynamic ABI, runtime-loader, or broad Windows wording. |
| Stop condition | A public claim lacks recurring evidence, hosted status is unknown where required, or residuals are omitted for unfunded work. |

## Pass Evidence Versus Advisory Output

| Evidence type | Can support selected claim? | Required interpretation |
| --- | --- | --- |
| Passing required local command on current branch | Yes, for local support tier. | Record command, date/context, branch, and touched surface. |
| Passing reviewed hosted CI lane | Yes, for reviewed/hosted tier. | Record lane, platform, commit, and artifact or summary policy. |
| Generated local files under ignored paths | Only after selected freshness/publication gate. | Treat as local context until promoted. |
| Source-controlled manifests or proof-owner rows | Not by themselves. | They define ownership and row meaning, not a fresh pass. |
| Advisory benchmark, coverage, dead-code, or optional-data rows | No, unless explicitly selected and scoped. | Preserve advisory/supplemental wording. |
| Product decision artifact | Yes, when the selected outcome is a binary support/non-support decision. | Synchronize docs, workflows, and guards. |
| Planning artifact | No direct product claim. | Use for traceability and handoff only. |

## Day 10 Inputs

Day 10 should turn these templates into a validation-command map by touched
surface:

- documentation-only;
- scripts and generated-report tooling;
- C/header code;
- build-system/package metadata;
- CI workflows;
- generated API docs and report artifacts;
- benchmark/performance reports;
- final claim audits.

## Completion Check

- Each Day 8 selected target has a reusable evidence template.
- Templates distinguish pass evidence from advisory output.
- Templates preserve local-only, hosted, reviewed, supplemental, advisory,
  deferred, and retained-non-claim support tiers.
- Later sprint artifacts can be compared consistently against these fields.
