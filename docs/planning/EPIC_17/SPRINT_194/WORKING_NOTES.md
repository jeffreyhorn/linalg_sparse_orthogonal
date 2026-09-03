# Sprint 194 Working Notes: Adoption and API Coherence Simplification

## Sprint Goal

Make the project easier to adopt by simplifying user-facing workflow guidance
and consolidating support/readiness truth.

## Day 1: Adoption Intake

### Scope Trace

| Epic item | Day 1 intake interpretation |
| --- | --- |
| 194.1 Adoption Audit | Inventory the user-facing and maintainer-facing docs that currently teach setup, first solve, solver choice, diagnostics, examples, install, support tiers, generated evidence, and claim boundaries. |
| 194.2 Support Matrix | Identify the current support/readiness truth sources before choosing a compact matrix owner. |
| 194.3 Installed Consumer Tutorial | Map Make/`pkg-config`, CMake, static archive, package-manager, and platform-specific install surfaces to current examples, tests, and docs. |
| 194.4 Diagnostics Coherence | Locate the docs and headers that explain `NULL` construction, `sparse_err_t`, direct residuals, iterative convergence/status, QR/SVD rank and residuals, eigensolver residuals, and benchmark/report diagnostics. |
| 194.5 Header Narrative Cleanup | Identify public headers with longer workflow or diagnostic narrative that may be moved into docs without changing declarations or Doxygen coverage. |
| 194.6 Validation | Record the validation owners for docs, Doxygen, examples, install, source-list, report freshness, package/PowerShell guards, and full C gates when headers change. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Sprint 194 is allocated 166 hours to simplify adoption, consolidate support truth, improve installed-consumer guidance, normalize diagnostics, clean selected public-header narrative, and validate docs/install/examples/API surfaces. |
| `docs/planning/EPIC_17/reviews/todo-codex-2026-08-28.md` | Epic 17 Phase 6 names the exact adoption/API simplification sequence: audit first, create compact support/readiness matrix, improve installed-consumer tutorial, normalize diagnostics, move narrative out of headers where possible, then validate. |
| `README.md` | The README is already the short front door, but it also repeats support-tier, selected evidence, installation, benchmark, API, and non-claim language that may be better consolidated. |
| `INSTALL.md` | INSTALL owns operational setup, static-first install, `pkg-config`, CMake installed consumers, supported platforms, and package-manager deferrals. It is the strongest current candidate for installed-consumer truth. |
| `docs/maintainer_guide.md` | Maintainer guide owns policy interpretation, support-surface ownership, reviewed baseline meaning, package/ABI boundaries, and platform proof interpretation. It should remain maintainer-facing rather than first-user documentation. |
| `docs/tutorial.md` | Tutorial repeats the start-here ladder and local build-tree link path, then routes installed consumers back to INSTALL. It should remain the learning path after README. |
| `docs/cookbook.md` | Cookbook owns data-first CSR, CSC, and Matrix Market workflows, and repeats diagnostic and evidence boundaries for selected QR/SVD/comparison lanes. |
| `docs/solver_selection.md` | Solver-selection guide owns problem-shape decisions and diagnostic handoff. It repeats detailed selected comparison and non-claim language that may need link-based consolidation. |
| `docs/api_reference.md` | API reference indexes public headers and local Doxygen freshness. It should stay declaration/ownership oriented and link out for support matrix truth. |
| `examples/README.md` | Examples README owns runnable example selection and example-local diagnostics. It should not become an install, benchmark, or support-tier policy page. |
| `benchmarks/README.md` | Benchmark docs own benchmark command groups, CSV schema, report artifact meaning, and performance caveats. They are relevant to support matrix evidence but should not be first-use adoption prose. |
| `tests/corpus/README.md` and `tests/corpus/manifests/selected_report_targets.tsv` | Corpus docs and selected target manifest own report-family/selected-target semantics, target row identities, workflow artifact metadata, and freshness non-claims. |
| `packaging/homebrew/README.md` and `packaging/homebrew/sparse-lu-ortho.rb.in` | Homebrew material exists as local provider-proof machinery, but support remains unclaimed until the proof and license metadata contract pass. |

### Adoption Surface Inventory

| Surface | Current role | Day 1 notes |
| --- | --- | --- |
| `README.md` | Short front door, quick start, workflow chooser, compact support summary, command map, API summary, install summary. | High user impact and high duplication risk because it repeats support/evidence caveats from deeper docs. |
| `INSTALL.md` | Operational setup, static-first install contract, `pkg-config`, CMake consumer, platform support, install verification. | Best user-facing owner for the compact support/readiness matrix or for the matrix's primary link target. |
| `docs/tutorial.md` | Fuller learning path after the README. | Should keep local build-tree examples and link to INSTALL for downstream installed consumers. |
| `docs/cookbook.md` | Data-first CSR/CSC/Matrix Market recipes and workflow handoffs. | Should stay problem/data focused; selected evidence caveats may be candidates for summary-plus-link treatment. |
| `docs/solver_selection.md` | Problem-shape solver chooser and diagnostics escalation. | Best owner for diagnostics vocabulary; selected evidence details may need consolidation. |
| `docs/api_reference.md` | Source-controlled API index and generated local Doxygen contract. | Should link to public headers for declarations and support matrix for readiness; avoid becoming a policy duplicate. |
| `docs/algorithm.md` and `docs/algorithm_history.md` | Current algorithm behavior and historical measurement notes. | Should not carry install/platform support truth except by link. |
| `docs/maintainer_guide.md` | Maintainer policy and quality-contract interpretation. | Best maintainer-facing source for support/readiness interpretation and validation ownership. |
| `examples/README.md` | Runnable examples, first outputs, example-local diagnostics. | Should keep examples small and route install users to INSTALL. |
| `examples/cmake_example/` | Installed CMake consumer example. | Primary candidate for installed-consumer tutorial proof alignment. |
| `sparse.pc.in` | Installed `pkg-config` metadata template. | Evidence owner for Make/`pkg-config` installed consumer semantics. |
| `cmake/SparseConfig.cmake.in` | Installed CMake package template. | Evidence owner for `find_package(Sparse)` installed consumer semantics. |
| `include/*.h` | Public declarations, Doxygen comments, API-local ownership and return-code contracts. | Candidate narrative cleanup surface; declarations and call-site contracts must not change. |
| `.github/workflows/*.yml` | Hosted Linux, macOS, and Windows quality/proof lanes. | Support matrix should point to workflow evidence without presenting workflow history as user truth. |
| `tests/test_install.sh` and `tests/test_cmake_install.sh` | Local install/downstream consumer proofs. | Validation owners for installed static archive, installed headers, `pkg-config`, and CMake package behavior. |
| `scripts/validate_windows_powershell.py` and `tests/test_validate_windows_powershell.py` | Windows PowerShell workflow snippet and claim-boundary guard. | Support matrix must preserve Windows non-claims that this guard enforces. |
| `scripts/normalize_report_index.py`, `tests/test_selected_report_targets_manifest.py`, `tests/test_selected_comparison_workflow.py`, `tests/test_selected_performance_docs.py` | Selected report target, comparison, and performance evidence enforcement. | Evidence owner for selected rows, not a broad support or performance claim source. |

### Support and Readiness Truth Sources

| Truth source | Current owner | Sprint 194 interpretation |
| --- | --- | --- |
| Static-first install contract | `INSTALL.md`, `tests/test_install.sh`, `tests/test_cmake_install.sh`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in` | User-facing support matrix should state this once and link to install verification. |
| Platform support tiers | `INSTALL.md#supported-platforms`, `docs/maintainer_guide.md`, `.github/workflows/*.yml` | Matrix should separate user-readable support status from maintainer workflow proof. |
| Package-manager deferral | `INSTALL.md`, `packaging/homebrew/README.md`, `scripts/package_manager_deferral_check.sh`, `scripts/static_package_deferral_check.sh`, Sprint 188 artifacts | Keep unclaimed until local provider proof and approved license metadata are complete. |
| Windows report freshness decision | Sprint 190 artifacts, Windows workflow, selected target manifest, maintainer guide | Only one bounded Windows selected Cholesky comparison lane is wired; broad Windows report freshness remains deferred. |
| Selected comparison evidence | `tests/corpus/manifests/selected_report_targets.tsv`, `scripts/run_external_comparison.py`, `scripts/normalize_report_index.py`, Linux/macOS CI workflows | Evidence is fixture-local and selected-target scoped. |
| Selected performance evidence | `benchmarks/README.md`, selected target manifest, `make bench-canonical-report-freshness`, `make performance-sentinels` | Evidence is threshold-free or bounded local sentinel proof, not portable performance superiority. |
| API/Doxygen freshness | `docs/api_reference.md`, `Doxyfile`, `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh`, `make api-docs-freshness` | Generated HTML is local-only; public headers remain declaration truth. |
| Residual queues and historical provenance | Sprint retrospectives and artifacts under `docs/planning/EPIC_16` and `docs/planning/EPIC_17` | Historical context should be linked only when it explains a current limitation or decision. |

### Adoption Personas and Workflows

| Persona | First need | Current path | Friction to audit |
| --- | --- | --- | --- |
| Source checkout developer | Build locally, run tests, run first solve. | `README.md` -> `make` / `make examples` -> `examples/README.md`. | README command map is long and mixes first-use steps with maintainer evidence commands. |
| Build-tree API learner | Write a small local program without installing. | `README.md#quick-start`, `docs/tutorial.md`, `examples/example_basic_solve.c`. | Local linking guidance appears in README and tutorial; ensure one canonical minimal path. |
| Installed static archive consumer | Install headers/library and link downstream. | `INSTALL.md`, `make install`, installed headers, `libsparse_lu_ortho.a` or `.lib`. | Static-first contract is repeated in README, INSTALL, maintainer docs, and examples. |
| `pkg-config` consumer | Use compiler/link flags after Unix-side install. | `INSTALL.md#using-via-pkg-config`, `sparse.pc.in`, `tests/test_install.sh`. | Windows metadata inspection vs execution parity needs clear one-row support wording. |
| CMake consumer | Use `find_package(Sparse)` after install. | `INSTALL.md#using-from-a-cmake-project`, `examples/cmake_example/`, `tests/test_cmake_install.sh`. | CMake build, CMake install/export, and Windows CMake-first support need clean separation. |
| Solver chooser | Pick LU, Cholesky, LDLT, QR, iterative, eigs, or SVD by problem shape. | README workflow chooser, `docs/solver_selection.md`, cookbook, examples. | Diagnostic terms and evidence caveats are repeated across solver-selection, cookbook, examples, and README. |
| Maintainer/release reviewer | Verify claims, proof ownership, and quality gates. | `docs/maintainer_guide.md`, Makefile targets, CI workflows, sprint artifacts. | Maintainer proof language leaks into first-use docs and makes adoption pages heavy. |

### Initial Duplication and Friction Register

| ID | Friction | Candidate owner direction |
| --- | --- | --- |
| D1-F1 | Support/platform wording appears in README, INSTALL, maintainer guide, corpus docs, benchmark docs, and sprint artifacts. | Make one compact user-facing support/readiness matrix and link to maintainer proof detail. |
| D1-F2 | Static-first package and non-claim wording is repeated in README, INSTALL, examples, API reference, and maintainer guide. | Keep installed-consumer commands in INSTALL; keep README summary short. |
| D1-F3 | Selected comparison and selected performance evidence details are repeated in README, solver-selection, cookbook, corpus docs, and benchmark docs. | Use selected target manifest as row truth and summarize in user docs by workflow. |
| D1-F4 | Diagnostics handoff appears in README, examples, cookbook, solver-selection, and headers. | Use solver-selection as diagnostics vocabulary owner and link from other docs. |
| D1-F5 | Public headers such as `sparse_eigs.h`, `sparse_cholesky.h`, `sparse_csr.h`, and `sparse_analysis.h` include longer narrative alongside exact call-site contracts. | Move broad workflow narrative into docs only when Doxygen coverage and API-local caveats remain intact. |
| D1-F6 | Installed consumer tutorial material is split across INSTALL, README, tutorial, examples README, `examples/cmake_example/`, package templates, and install tests. | Create or consolidate a minimal installed-consumer tutorial section around Make/`pkg-config` and CMake. |
| D1-F7 | Validation commands are available but scattered across Makefile help, README command map, maintainer guide, and sprint artifacts. | Add a compact validation owner map for Sprint 194 changes. |

### Initial Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Overclaiming support/readiness | Sprint 194 is explicitly about simplifying adoption, not widening platform/package/performance support. | Preserve current non-claims and only summarize already-proven support. |
| Hiding historical evidence | Removing repeated caveats can accidentally erase provenance needed by maintainers. | Link from user truth to maintainer/corpus/benchmark owners instead of deleting evidence ownership. |
| Breaking Doxygen coverage | Header narrative cleanup can remove comments needed by generated API pages. | Run `make api-docs-freshness` or at least `make docs-check` when public headers change. |
| Changing declarations while editing comments | Header cleanup must not affect ABI/API shape. | Keep public declarations byte-for-byte unless a later day explicitly records an API change, which is a non-goal. |
| Stale support links | Consolidation increases reliance on links and anchors. | Run markdown/link-oriented checks available in repo and inspect changed anchors manually. |
| Tutorial commands diverge from install proof | Installed-consumer docs must match `tests/test_install.sh`, `tests/test_cmake_install.sh`, templates, and CI surfaces. | Treat tests/templates as executable proof owners before changing tutorial text. |
| Platform-specific confusion | Windows CMake support, `pkg-config` metadata inspection, and Windows `pkg-config` execution non-claim are easy to blur. | Keep support matrix rows platform/toolchain specific. |
| Review noise | Large documentation rewrites can obscure claim semantics. | Sequence changes through audit, matrix contract, then focused edits. |

### Validation Owners

| Change type | Candidate validation |
| --- | --- |
| Markdown-only inventory or planning docs | `git diff --check` |
| User docs or anchors | `git diff --check`, manual link/anchor inspection, relevant docs guards if present |
| API/Doxygen wording | `make docs-check`, `make api-docs-freshness` |
| Public headers | `make format && make lint && make test`; add Doxygen checks when comments change |
| Installed consumer docs/templates | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `make quality-review-cmake` when CMake contract changes |
| Examples docs or sources | `make examples-build`, full C gate if `.c`/`.h` examples change |
| Support/claim wording | package deferral guards, Windows PowerShell guard, selected report target tests, selected performance docs tests |
| Report evidence wording | `python3 tests/test_selected_report_targets_manifest.py`, `python3 tests/test_selected_comparison_workflow.py`, `python3 tests/test_selected_performance_docs.py`, selected freshness targets as needed |

### Day 1 Validation

Source and planning checks:

```sh
git status --short --branch
sed -n '269,302p' docs/planning/EPIC_17/PROJECT_PLAN.md
sed -n '1,140p' docs/planning/EPIC_17/SPRINT_194/PLAN.md
sed -n '1,220p' README.md
sed -n '1,260p' INSTALL.md
sed -n '1,260p' docs/maintainer_guide.md
sed -n '1,260p' docs/tutorial.md
sed -n '1,280p' docs/solver_selection.md
sed -n '1,260p' examples/README.md
sed -n '1,220p' benchmarks/README.md
sed -n '1,220p' tests/corpus/README.md
sed -n '1,120p' tests/corpus/manifests/selected_report_targets.tsv
rg -n "support|readiness|install|pkg-config|CMake|Windows|Homebrew|selected|performance|comparison|state-of-the-art|shared|static|dynamic|ABI|residual|Doxygen|diagnostic|status|converged|rank|threshold" README.md INSTALL.md docs/{tutorial.md,cookbook.md,solver_selection.md,api_reference.md,maintainer_guide.md} examples/README.md
rg -n "Doxygen|diagnostic|status|residual|converg|rank|ownership|return|NULL|SPARSE_ERR|ABI|shared|package|Windows|CMake|pkg-config" include/*.h
git diff --check
```

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

### Day 2 Questions

1. Which file should own the compact user-facing support/readiness matrix:
   `INSTALL.md`, a new docs page linked from INSTALL/README, or
   `docs/maintainer_guide.md` with a user-facing excerpt?
2. Which support claims can be reduced to matrix rows without losing necessary
   non-claims for Windows, package managers, shared libraries, ABI, report
   freshness, and performance?
3. Which duplicated diagnostic phrases should be normalized first across
   README, examples, cookbook, solver selection, and public headers?
4. Which installed-consumer path needs the clearest tutorial treatment first:
   Unix Make/`pkg-config`, installed CMake, or Windows CMake-first?
5. Which public headers contain narrative that is safe to move into docs while
   preserving exact declarations and Doxygen coverage?

## Day 2: Adoption Friction Audit

### Audit Method

Day 2 ranked friction with five factors:

- user impact: how often a first-time user or downstream consumer sees it;
- duplication: how many maintained surfaces repeat the same claim;
- staleness risk: likelihood that future support/evidence changes require
  synchronized edits;
- evidence coupling: whether user docs embed maintainer proof or historical
  sprint detail directly;
- overclaim risk: likelihood that simplified wording could imply support not
  currently proven.

Scores use a 1-5 scale where 5 means highest impact or risk.

### Ranked Cleanup Candidates

| Rank | Candidate | User impact | Duplication | Staleness risk | Evidence coupling | Overclaim risk | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Compact support/readiness matrix with link-based proof owners | 5 | 5 | 5 | 5 | 5 | Highest priority. This is the prerequisite for safely reducing repeated support/platform/package caveats later. |
| 2 | Static installed-consumer guidance consolidation | 5 | 5 | 4 | 4 | 5 | High priority. README, INSTALL, tutorial, examples, templates, and install tests should converge on one minimal user path. |
| 3 | Diagnostics vocabulary owner and cross-doc normalization | 5 | 4 | 4 | 3 | 3 | High priority. Solver-selection should own the canonical handoff; README/examples/cookbook/headers should link or summarize. |
| 4 | Selected report/comparison/performance evidence summary cleanup | 4 | 5 | 5 | 5 | 5 | High priority after matrix contract. Keep selected manifest and benchmark/corpus docs as detail owners. |
| 5 | README command-map and evidence density reduction | 5 | 4 | 4 | 4 | 4 | Medium-high priority. Do only after the matrix and tutorial owners exist, otherwise README loses necessary context. |
| 6 | Public-header narrative cleanup | 3 | 3 | 3 | 3 | 4 | Medium priority. Delay until diagnostic wording and Doxygen validation plan are explicit. |
| 7 | Maintainer guide proof-detail boundary cleanup | 2 | 4 | 4 | 5 | 3 | Medium priority. It is intentionally detailed, so avoid shortening proof semantics needed by reviewers. |
| 8 | Benchmark/report handoff simplification in first-use docs | 3 | 3 | 4 | 4 | 4 | Medium priority. Keep benchmarks/README as detail owner. |
| 9 | Historical sprint-link pruning | 2 | 3 | 4 | 5 | 3 | Lower priority. Keep provenance where it explains current limitations. |

### Evidence-Backed Findings

| Finding | Current evidence | Cleanup boundary |
| --- | --- | --- |
| Support/platform status is repeated at several levels. | `README.md` summarizes CI/support tiers; `INSTALL.md#supported-platforms` carries a platform table; `docs/maintainer_guide.md` carries detailed platform proof interpretation; corpus and benchmark docs repeat report/platform non-claims. | Create one compact user-facing support/readiness matrix, likely in or linked from `INSTALL.md`, then replace repeated first-use prose with links. |
| Static-first package truth is repeated and easy to over-read. | `README.md` has install summary and non-claims; `INSTALL.md` owns static archive, `pkg-config`, CMake, and platform-specific boundaries; maintainer guide explains proof owners; examples mention installed-consumer examples. | Keep full installed-consumer commands and support boundary in `INSTALL.md`; README/tutorial/examples should only route users there. |
| Windows CMake support and Windows `pkg-config` non-claim need row-level precision. | `INSTALL.md` says Windows is CMake-first and `sparse.pc` inspection is metadata-only; maintainer guide and README repeat no Windows Makefile or `pkg-config` execution parity. | Matrix rows should separate Windows CMake install/downstream validation, Windows `sparse.pc` metadata inspection, and unclaimed Windows `pkg-config` execution. |
| Package-manager status is not user-support despite Homebrew proof material. | `INSTALL.md`, `packaging/homebrew/README.md`, package deferral guards, and maintainer guide describe the local Homebrew proof blocker. | Matrix should state package-manager support as not currently provided and link to provider-proof blocker context only for maintainers. |
| Selected comparison and performance evidence details are too dense for first-use pages. | README, solver-selection, cookbook, corpus README, benchmark README, and maintainer guide all repeat selected target caveats. | Keep target IDs, expected rows, artifacts, claim scopes, and non-claims in `tests/corpus/manifests/selected_report_targets.tsv` and detail docs; first-use pages should summarize by workflow. |
| Diagnostics are spread across user docs and headers. | README, examples README, cookbook, solver-selection, tutorial, and headers all explain `NULL`, `sparse_err_t`, residuals, convergence, rank, and status interpretation. | Make `docs/solver_selection.md#diagnostics-handoff` the canonical user vocabulary; public headers keep exact return-code and struct contracts. |
| Public headers mix exact contracts with some workflow narrative. | `include/sparse_eigs.h`, `include/sparse_cholesky.h`, `include/sparse_ldlt.h`, `include/sparse_csr.h`, and `include/sparse_analysis.h` include longer explanations around defaults, telemetry, cancellation, repeated workflows, and diagnostics. | Move only broad workflow narrative to docs; keep API-local defaults, preconditions, ownership, telemetry fields, and return codes in headers. |
| Validation guidance exists but is scattered. | Makefile targets cover docs, Doxygen, examples, install, package guards, report freshness, and C gates; README and maintainer guide repeat many commands. | Later Sprint 194 docs should include a compact validation-owner map instead of repeating full command prose everywhere. |

### Candidate Cleanup Boundaries

| Boundary | In scope for Sprint 194 | Out of scope |
| --- | --- | --- |
| Support matrix | Current support/readiness rows for build, install, platform, package manager, report evidence, performance evidence, API docs, shared-library/ABI status, and validation owner. | New platform support, new package-manager support, or broader hosted report freshness. |
| Installed consumers | Minimal Make/`pkg-config` and CMake downstream examples aligned with `tests/test_install.sh`, `tests/test_cmake_install.sh`, `sparse.pc.in`, and `cmake/SparseConfig.cmake.in`. | Shared-library tutorial, dynamic ABI promise, Windows `pkg-config` execution, or provider package installation. |
| Diagnostics | Canonical wording for constructor failure, `sparse_err_t`, direct residual, iterative convergence/status, QR/SVD rank/residuals, eigensolver Ritz residuals, and report freshness diagnostics. | Numerical behavior changes, tolerance changes, or new diagnostic fields. |
| Selected evidence | Link-based summaries that point to selected target manifest, corpus docs, benchmark docs, and maintainer guide. | Changing selected target rows, workflow promotion, row counts, support tiers, or generated artifact semantics without a dedicated implementation day. |
| Header comments | Comment-only relocation of non-contract workflow prose after identifying Doxygen coverage owners. | Declaration changes, ABI changes, option/result struct changes, or removal of API-local contracts. |

### Support-Tier Risk List

| Risk | Current vulnerable wording area | Required safeguard |
| --- | --- | --- |
| Windows CMake-first becomes broad Windows parity. | README/INSTALL support summaries and selected Windows report freshness prose. | Matrix must name Windows MSVC CMake subset and retained non-claims explicitly. |
| Windows `sparse.pc` metadata inspection becomes `pkg-config` execution support. | INSTALL package table and Windows notes. | Separate metadata inspection from command execution in the matrix. |
| Homebrew proof material becomes package-manager availability. | README/INSTALL/package docs. | Keep package-manager support as unprovided until proof blocker is resolved and guards change. |
| Selected comparison freshness becomes broad external-library parity. | README, solver-selection, cookbook, corpus docs. | Use fixture-local selected target wording and manifest links. |
| Selected performance freshness becomes portable performance claim. | README, benchmark docs, maintainer guide. | Preserve threshold-free methodology language and no portable performance claim. |
| API Doxygen freshness becomes hosted API publication. | API reference and README docs command sections. | Keep generated HTML local-only unless a future product decision changes it. |
| Static-first install becomes shared-library or dynamic ABI support. | README/INSTALL install summaries and package metadata. | Matrix row must keep shared-library packaging and dynamic ABI deferred. |

### Day 2 Decisions

- `INSTALL.md` is the leading candidate for the compact user-facing
  support/readiness matrix because it already owns install, package, platform,
  downstream consumer, and support-boundary detail.
- `docs/maintainer_guide.md` should remain the maintainer proof semantics
  owner, not the first user-facing matrix owner.
- `docs/solver_selection.md#diagnostics-handoff` should become the canonical
  user-facing diagnostics vocabulary owner.
- `tests/corpus/manifests/selected_report_targets.tsv` should remain the
  selected target row authority; first-use docs should not duplicate exact
  row IDs, artifact paths, or workflow artifacts unless necessary.
- README simplification should wait until the support matrix contract exists,
  so visible non-claims are not accidentally weakened.

### Day 2 Validation

Planning/doc checks:

```sh
git status --short --branch
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_194/PLAN.md
sed -n '1,260p' docs/planning/EPIC_17/SPRINT_194/WORKING_NOTES.md
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_194/artifacts/day1-adoption-intake.md
rg -n "Continuous integration|Supported platforms|Package-manager support|Shared-library packaging|selected comparison|selected performance|Windows|Homebrew|pkg-config|static-first|dynamic ABI|state-of-the-art|not broad|non-claim|does not claim" README.md INSTALL.md docs/maintainer_guide.md docs/api_reference.md docs/solver_selection.md docs/cookbook.md examples/README.md benchmarks/README.md tests/corpus/README.md
rg -n "Diagnostics Handoff|diagnostics|residual|status|convergence|rank|NULL|sparse_err_t|SPARSE_ERR" README.md docs/tutorial.md docs/cookbook.md docs/solver_selection.md examples/README.md include/*.h
rg -n "make install|cmake --install|find_package|pkg-config|Sparse::sparse_lu_ortho|BUILD_SHARED_LIBS|Libs.private|sparse.pc|libsparse_lu_ortho|CMAKE_INSTALL_PREFIX" README.md INSTALL.md docs/tutorial.md examples/README.md examples/cmake_example/CMakeLists.txt tests/test_install.sh tests/test_cmake_install.sh cmake/SparseConfig.cmake.in sparse.pc.in
git diff --check
```

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

## Day 3: Support Matrix Contract

### Contract Decision

Sprint 194 should add the compact user-facing support/readiness matrix to
`INSTALL.md`, near the existing support split and supported-platform sections.
That location already owns operational setup, installed consumers, package
shape, platform support, install validation, and package-manager boundaries.

`docs/maintainer_guide.md` remains the proof-semantics owner behind the
matrix. `tests/corpus/manifests/selected_report_targets.tsv`,
`benchmarks/README.md`, `tests/corpus/README.md`, CI workflows, install tests,
package guards, and Doxygen guards remain the detailed executable or
source-controlled evidence owners.

### Matrix Dimensions

The matrix should use these columns:

| Column | Meaning | Required behavior |
| --- | --- | --- |
| Surface | User-facing support area, such as local build, static install, Windows CMake, selected comparison, or API docs. | Keep row names short and stable enough to link from README/tutorial/cookbook/API docs. |
| Current user status | One of the approved status terms below. | Must describe what a user can rely on today, not historical sprint intent. |
| Primary user path | The doc, command, or workflow a user should start from. | Prefer user docs over planning artifacts. |
| Evidence owner | Test, script, manifest, workflow, or maintainer doc that owns proof details. | Must be precise enough for reviewers to find the source of truth. |
| Retained non-claims | Important support boundaries that remain unproven or deliberately out of scope. | Must be visible for high-risk rows. |

### Approved Status Vocabulary

| Term | Definition | Use when |
| --- | --- | --- |
| `supported` | The path is a normal documented user path and has maintained validation appropriate to its platform/scope. | Local source build, local examples, and static install surfaces with direct proof. |
| `validated` | The path has an explicit local or hosted validation command, but the row may be narrower than broad support. | CMake parity, install/export proof, selected workflow snippets, selected report freshness. |
| `local-only` | Evidence is generated or checked locally and should not be read as hosted, release, or portable proof. | Local report indexes, local benchmark snapshots, local Doxygen HTML. |
| `hosted-evidence` | A hosted CI lane runs a selected proof with scoped artifacts or metadata. | Reviewed Linux/macOS selected comparison, Linux selected performance, Windows selected Cholesky comparison. |
| `deferred` | The project intentionally does not claim the surface until named prerequisites are met. | Shared libraries, dynamic ABI, broad Windows parity, broad report freshness, package-manager distribution. |
| `not claimed` | The surface is outside the current support statement and has no active user path. | Package-manager availability, Homebrew/core, bottles, broad ecosystem parity, state-of-the-art claims. |
| `residual` | A known limitation or environment-dependent blocker remains recorded without being pass evidence. | Unavailable local PowerShell, missing optional dependencies, absent generated local reports. |

### Proposed Matrix Rows

| Surface | Current user status | Primary user path | Evidence owner | Retained non-claims |
| --- | --- | --- | --- | --- |
| Local source build and first solve | `supported` | `README.md`, `examples/README.md`, `make`, `make examples` | Makefile, examples sources, `make examples-build`, `make test` | Not an install, package-manager, performance, or platform-parity claim. |
| Unix Make static install | `supported` | `INSTALL.md#quick-start-makefile` | `tests/test_install.sh`, `sparse.pc.in`, Linux/macOS install lanes | No shared-library, dynamic ABI, runtime-loader, or package-manager support. |
| Unix `pkg-config` installed consumer | `validated` | `INSTALL.md#using-via-pkg-config` | `tests/test_install.sh`, `sparse.pc.in` | No Windows `pkg-config` execution parity; no package-manager distribution. |
| Installed CMake consumer | `supported` | `INSTALL.md#using-from-a-cmake-project`, `examples/cmake_example/` | `tests/test_cmake_install.sh`, `cmake/SparseConfig.cmake.in`, CMake package export | No shared-library or broad dynamic ABI promise. |
| Windows MSVC CMake install/downstream | `validated` | `INSTALL.md#windows-msvc`, `.github/workflows/windows-ci.yml` | Windows CMake install/downstream lane, `scripts/validate_windows_powershell.py` for workflow snippet ownership | No Windows Makefile parity, Windows `pkg-config` execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |
| Windows selected Cholesky comparison freshness | `hosted-evidence` | Windows selected comparison workflow, selected target manifest | Sprint 190 Windows workflow path, `scripts/normalize_report_index.py --selected-target cholesky-spd-tridiag-5`, PowerShell guard | No broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, or unselected Windows comparison families. |
| Linux/macOS selected comparison freshness | `hosted-evidence` | `make report-index-comparison-freshness` | `tests/corpus/manifests/selected_report_targets.tsv`, Linux/macOS hosted selected comparison lanes, comparison scripts/tests | No broad external-library parity, broad report freshness, package/ABI support, performance superiority, or state-of-the-art claim. |
| Linux selected performance freshness | `hosted-evidence` | `make bench-canonical-report-freshness` | `tests/corpus/manifests/selected_report_targets.tsv`, `benchmarks/README.md`, `scripts/check_bench_canonical_freshness.py`, Linux hosted lane | No portable performance claim, timing threshold claim, release benchmark claim, platform parity, package/ABI proof, or state-of-the-art claim. |
| Local benchmark and sentinel reports | `local-only` | `make bench-canonical-report`, `make performance-sentinels` | `benchmarks/README.md`, benchmark scripts, sentinel checks | Not hosted proof unless named by a hosted lane; no portable speedup claim. |
| Local generated API HTML | `local-only` | `docs/api_reference.md`, `make api-docs-freshness` | `Doxyfile`, `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh` | No hosted API publication, package-manager distribution, dynamic ABI proof, or completeness beyond checked-in public headers selected by Doxyfile. |
| Package-manager distribution | `not claimed` | Source install via Make or CMake instead | `packaging/homebrew/README.md`, `scripts/homebrew_local_formula_proof.sh`, `scripts/package_manager_deferral_check.sh`, Sprint 188 artifacts | No Homebrew/core readiness, bottle support, Linuxbrew support, public tap support, vcpkg/Conan/pkgsrc/distro package support, or broad provider support. |
| Shared-library and dynamic ABI support | `deferred` | Static install only | `INSTALL.md`, `scripts/static_package_deferral_check.sh`, Sprint 170 package/ABI decision | No `.so`, `.dylib`, `.dll`, import-library, loader, SONAME, install-name/RPATH, static/shared selector, or dynamic ABI support. |
| Broad state-of-the-art or ecosystem parity | `not claimed` | Use selected evidence docs only for scoped proof | Epic 17 review/todo, selected target manifest, benchmark/corpus docs | No broad SuiteSparse/PETSc/Trilinos/Eigen/SciPy parity, portable superiority, or unqualified state-of-the-art status. |

### Link Owner Map

| Source doc | Day 4 link behavior |
| --- | --- |
| `README.md` | Keep only a short current-support summary and link to the support/readiness matrix for detail. |
| `INSTALL.md` | Own the matrix and retain install-specific commands below it. |
| `docs/tutorial.md` | Link to the matrix only from install/downstream consumer handoff text. |
| `docs/cookbook.md` | Link to the matrix when mentioning package/platform/performance boundaries; keep data-first guidance local. |
| `docs/solver_selection.md` | Link to the matrix for platform/package/report boundaries; keep diagnostics and solver-choice wording local. |
| `docs/api_reference.md` | Link to the matrix for support/readiness; keep declaration and Doxygen-local contract wording local. |
| `examples/README.md` | Link to the matrix for installed-consumer and support boundary interpretation; keep example-local behavior local. |
| `benchmarks/README.md` | Keep benchmark semantics local; link back to the matrix only for support/readiness summary if needed. |
| `tests/corpus/README.md` | Keep selected target and report semantics local; do not become the user-facing support matrix. |
| `docs/maintainer_guide.md` | Link to the matrix as the user-facing support summary while retaining proof-owner interpretation. |

### Day 4 Implementation Guardrails

- Add the matrix without changing support level, selected target metadata,
  CI workflow behavior, package templates, or generated report semantics.
- Preserve visible non-claims for Windows breadth, package-manager breadth,
  shared libraries, dynamic ABI, broad report freshness, selected oracle or
  benchmark freshness on Windows, portable performance, and state-of-the-art
  status.
- Do not remove detailed proof-owner text until the matrix is present and
  links are in place.
- Keep planning artifacts as provenance, not the primary user path.

### Day 3 Validation

Planning/doc checks:

```sh
git status --short --branch
sed -n '70,140p' docs/planning/EPIC_17/SPRINT_194/PLAN.md
sed -n '120,190p' INSTALL.md
sed -n '340,415p' INSTALL.md
sed -n '210,330p' docs/maintainer_guide.md
sed -n '580,670p' docs/maintainer_guide.md
sed -n '1,12p' tests/corpus/manifests/selected_report_targets.tsv
rg -n "package_manager|static_package|homebrew|windows-powershell|report-index-comparison-freshness|bench-canonical-report-freshness|api-docs-freshness|docs-check|quality-review-cmake|test_install|test_cmake_install" Makefile scripts tests docs/maintainer_guide.md INSTALL.md
git diff --check
```

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

## Day 4: Support Matrix Implementation

### Implementation Summary

Day 4 implemented the compact support/readiness matrix in `INSTALL.md` and
added routing links from the major adoption docs. The implementation preserved
current support levels and did not change selected report target metadata, CI
workflow behavior, package templates, install rules, generated report
semantics, or public headers.

### Changed Files

| File | Change |
| --- | --- |
| `INSTALL.md` | Added `Support Readiness Matrix` as the active user-facing support truth and linked Start Here support reading to it. |
| `README.md` | Added an adoption-map row for current support/readiness, shortened the CI capability bullet to route to the matrix, and replaced repeated install/package non-claims with a matrix link. |
| `docs/tutorial.md` | Routed installed package and support/readiness status to INSTALL and the matrix while keeping the tutorial local-build-tree focused. |
| `docs/cookbook.md` | Routed install/downstream/support status to INSTALL and tied package/platform/performance boundaries to the matrix. |
| `docs/solver_selection.md` | Routed install/support readiness to the matrix and kept solver diagnostics local. |
| `docs/api_reference.md` | Added a support/readiness link for installed packages, local generated API HTML, platforms, package managers, shared libraries, and ABI boundaries. |
| `examples/README.md` | Routed installed-consumer support interpretation to the matrix while keeping example-local diagnostics local. |
| `docs/maintainer_guide.md` | Added the matrix to first-user starting points and clarified that INSTALL owns the user-facing support/readiness matrix. |

### Retained Claim Boundaries

| Boundary | Day 4 result |
| --- | --- |
| Windows support breadth | Matrix keeps Windows MSVC CMake install/downstream as `validated` and retains no Windows Makefile parity, Windows `pkg-config` execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |
| Windows selected report freshness | Matrix keeps only the bounded Cholesky selected comparison lane as hosted evidence and retains no broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, or unselected Windows comparison families. |
| Package-manager breadth | Matrix marks package-manager distribution as `not claimed` and points users to source install via Make or CMake. |
| Shared-library and dynamic ABI | Matrix marks shared-library and dynamic ABI support as `deferred` and keeps static install as the active path. |
| Selected comparison evidence | Matrix keeps Linux/macOS selected comparison freshness as fixture-local hosted evidence without external-library parity, broad report freshness, package/ABI, performance superiority, or state-of-the-art claims. |
| Selected performance evidence | Matrix keeps Linux selected performance freshness as methodology/freshness evidence without portable performance, timing threshold, release benchmark, platform parity, package/ABI, or state-of-the-art claims. |
| API docs | Matrix keeps generated API HTML as `local-only` and does not promote hosted API publication. |

### Day 4 Validation

Validation run:

```sh
git diff --check
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
make windows-powershell-guard
python3 tests/test_selected_performance_docs.py
```

All commands passed. `make windows-powershell-guard` includes internal
negative-case checks for unavailable or required PowerShell, but the target
completed successfully.

No `.c` or `.h` files were modified, so `make format && make lint && make
test` is not required for this day.

## Day 5: Consumer Tutorial Audit

### Audit Summary

Day 5 audited the current installed-consumer documentation and proof surfaces
for Sprint 194 Item 194.3. The audit found that the docs now route users to the
right owning surfaces, but the public installed-consumer walkthrough still
needs a fuller copy-pasteable tutorial that distinguishes source-tree local
builds from installed-prefix consumers.

### Evidence Reviewed

| Source | Finding |
| --- | --- |
| `INSTALL.md` | Owns the start-here flow, support/readiness matrix, installed files, `pkg-config`, CMake, Windows, and verification sections. |
| `README.md` | Provides only a compact static-first installation summary and routes support-boundary detail to `INSTALL.md`. |
| `docs/tutorial.md` | Correctly remains a local build-tree tutorial and routes installed consumers to `INSTALL.md`. |
| `examples/README.md` | Correctly routes installed downstream consumers to `examples/cmake_example/` and `INSTALL.md`. |
| `examples/cmake_example/` | Provides the maintained installed CMake consumer example using `find_package(Sparse)` and `Sparse::sparse_lu_ortho`. |
| `tests/test_install.sh` | Defines the Make install, installed headers, static archive, `sparse.pc`, `pkg-config`, compile/link/run, and uninstall proof. |
| `tests/test_cmake_install.sh` | Defines the CMake install/export, downstream `find_package`, exact-version, mismatched-version, and installed metadata proof. |
| `sparse.pc.in` | Defines the current `pkg-config` package name, cflags, libs, version, and static archive description. |
| `cmake/SparseConfig.cmake.in` | Defines the installed CMake package entrypoint. |
| `packaging/homebrew/README.md` | Remains provider-proof provenance and not a user-facing package-manager install path. |

### Tutorial Contracts Defined

Day 5 defined two minimal tutorial contracts in
`artifacts/day5-consumer-tutorial-audit.md`:

- Unix Make/`pkg-config` installed consumer:
  - staged local prefix install;
  - `PKG_CONFIG_PATH` setup;
  - installed includes as `<sparse/...>`;
  - compile/link through `pkg-config --cflags --libs sparse`;
  - expected `nnz: 1` and `OK` smoke output;
  - proof with `bash tests/test_install.sh`.
- Installed CMake consumer:
  - staged CMake install/export;
  - `find_package(Sparse REQUIRED)`;
  - `Sparse::sparse_lu_ortho`;
  - canonical example in `examples/cmake_example/`;
  - proof with `bash tests/test_cmake_install.sh`;
  - Windows route limited to CMake/MSVC.

### Retained Boundaries

The Day 5 contract explicitly excludes:

- package-manager support or Homebrew/core availability;
- shared-library packaging or dynamic ABI support;
- runtime-loader behavior;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- broad report freshness;
- portable performance superiority;
- state-of-the-art claims.

### Day 5 Validation

Validation run:

```sh
git diff --check
```

Day 5 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required for this
day.

## Day 6: Consumer Tutorial Implementation

### Implementation Summary

Day 6 implemented the installed-consumer tutorial scoped by Day 5. The new
canonical public path is `INSTALL.md#installed-consumer-tutorial`, with a
Unix Make/`pkg-config` consumer section and a CMake consumer section. README,
tutorial, examples, and maintainer-guide text now route to those public anchors
instead of preserving only the older short command summaries.

### Changed Files

| File | Change |
| --- | --- |
| `INSTALL.md` | Added the installed-consumer tutorial, updated Start Here and support matrix links, expanded CMake consumer guidance, and kept validation owners visible. |
| `README.md` | Linked installed downstream consumers to the new public tutorial while keeping the README summary compact. |
| `docs/tutorial.md` | Updated the local-build-tree handoff to point at the new Unix Make/`pkg-config` and CMake consumer anchors. |
| `examples/README.md` | Linked installed CMake consumer references to the CMake consumer and installed-consumer tutorial anchors. |
| `docs/maintainer_guide.md` | Linked support ownership text to the public installed-consumer tutorial. |
| `artifacts/day5-consumer-tutorial-audit.md` | Corrected the provisional smoke program to use the actual public matrix API names used by `tests/test_install.sh`. |
| `artifacts/day6-consumer-tutorial-implementation.md` | Recorded Day 6 changed files, tutorial behavior, retained boundaries, and validation plan. |

### Tutorial Details Implemented

Unix Make/`pkg-config` consumer guidance now includes:

- staged local install with `make install PREFIX="$PWD/_install"`;
- `PKG_CONFIG_PATH` setup and `pkg-config --exists sparse`;
- installed include paths as `<sparse/...>`;
- a minimal `main.c` using `sparse_create`, `sparse_insert`,
  `sparse_nnz`, and `sparse_free`;
- compile/link command using `pkg-config --cflags sparse` and
  `pkg-config --libs sparse`;
- expected output markers for version metadata, `nnz: 1`, and `OK`;
- failure wording for missing package metadata, headers, or static archive;
- validation with `bash tests/test_install.sh`.

CMake consumer guidance now includes:

- staged CMake install/export into `$PWD/_install`;
- minimal downstream `CMakeLists.txt`;
- `find_package(Sparse REQUIRED)`;
- installed target `Sparse::sparse_lu_ortho`;
- downstream configure/build/run commands using `CMAKE_PREFIX_PATH`;
- exact-version lookup wording;
- Windows/MSVC CMake install command;
- validation with `bash tests/test_cmake_install.sh`.

### Retained Boundaries

Day 6 preserved non-claims for package-manager support, Homebrew/core,
bottles, Linuxbrew, vcpkg, Conan, shared libraries, dynamic ABI, runtime-loader
behavior, Windows Makefile parity, Windows `pkg-config` execution parity,
broad platform parity, broad report freshness, portable performance
superiority, and state-of-the-art status.

### Day 6 Validation

Validation run:

```sh
git diff --check
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
make windows-powershell-guard
```

Day 6 changed documentation and planning artifacts only. No `.c` or `.h` files
were modified, so `make format && make lint && make test` is not required for
this day.

## Day 7: Diagnostics Wording Contract

### Contract Summary

Day 7 audited diagnostics language across the front-door docs, solver
selection, cookbook, examples, maintainer guide, and public headers. The result
is a documentation-only wording contract for Days 8 and 9 that normalizes how
docs talk about return codes, result structs, residuals, convergence,
non-convergence, singularity, unsupported inputs, QR/SVD rank diagnostics,
eigensolver telemetry, and generated report diagnostics.

### Evidence Reviewed

| Source | Diagnostic role |
| --- | --- |
| `include/sparse_types.h` | Owns `sparse_err_t`, `SPARSE_OK`, and public error-code vocabulary. |
| `include/sparse_matrix.h` | Owns constructor, Matrix Market, stream I/O, and matrix-operation error contracts. |
| `include/sparse_lu.h`, `include/sparse_cholesky.h`, `include/sparse_ldlt.h`, `include/sparse_analysis.h` | Own direct factor/solve/refine/condition and repeated lifecycle return-code wording. |
| `include/sparse_iterative.h` | Owns `sparse_iter_result_t`, convergence, stagnation, breakdown, residual-history, and approximate-solution wording. |
| `include/sparse_qr.h` | Owns QR rank, nullity/nullspace, least-squares residual, minimum-norm, and R-diagonal diagnostics. |
| `include/sparse_svd.h` | Owns SVD rank, condition, pseudoinverse, low-rank, and non-convergence wording. |
| `include/sparse_eigs.h` | Owns symmetric eigensolver residual, convergence count, backend, peak-basis, shift-invert, and preconditioner telemetry. |
| `README.md`, `docs/solver_selection.md`, `docs/cookbook.md`, `examples/README.md`, `docs/maintainer_guide.md` | User-facing and maintainer-facing diagnostic handoffs that need shared vocabulary. |

### Main Contract Decisions

- Use `return code` only for APIs returning `sparse_err_t`.
- Use result-struct field names when describing iterative or eigensolver
  telemetry.
- Treat `SPARSE_ERR_NOT_CONVERGED` as budget exhaustion under the relevant API,
  not as a generic hard failure.
- Treat residuals as workflow-local diagnostics unless a stronger proof owner
  is named.
- Keep QR/SVD rank language tolerance-local and API-local.
- Keep eigensolver AUTO backend language as routing policy, not superiority.
- Keep reports and benchmarks framed as local or selected hosted freshness
  diagnostics, not broad performance or state-of-the-art evidence.

### Artifact

The full Day 7 contract is recorded in
`artifacts/day7-diagnostics-wording-contract.md`.

### Day 7 Validation

Validation run:

```sh
git diff --check
```

Day 7 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required for this
day.

## Day 8: Direct and Iterative Diagnostics Cleanup

### Implementation Summary

Day 8 applied the Day 7 wording contract to direct and iterative solver docs.
The cleanup keeps direct diagnostics tied to `sparse_err_t` return codes and
iterative diagnostics tied to `sparse_iter_result_t` fields. It does not change
solver behavior, status-code semantics, tolerance defaults, preconditioner
behavior, or backend selection.

### Changed Files

| File | Change |
| --- | --- |
| `README.md` | Clarified LU/Cholesky return-code owners, added explicit `SPARSE_ERR_NOT_CONVERGED` handling to the iterative snippet, and documented `sparse_iter_result_t` field interpretation. |
| `docs/solver_selection.md` | Normalized direct diagnostic table wording, direct solver singularity/SPD wording, and iterative result-field/non-convergence wording. |
| `docs/cookbook.md` | Clarified direct function-specific return-code inspection and added iterative result-field guidance for compressed-input workflows. |
| `examples/README.md` | Aligned direct example residual wording and iterative example/result-field wording. |
| `docs/maintainer_guide.md` | Extended iterative evidence wording to all `sparse_iter_result_t` fields and clarified `SPARSE_ERR_NOT_CONVERGED` as iteration-budget exhaustion. |
| `artifacts/day8-direct-iterative-diagnostics-cleanup.md` | Recorded before/after terminology notes, retained semantics, and validation plan. |

### Retained Boundaries

- Direct solver residuals remain problem-local diagnostics.
- Iterative result fields remain tied to documented `SPARSE_OK` and
  `SPARSE_ERR_NOT_CONVERGED` population rules.
- `SPARSE_ERR_NOT_CONVERGED` is not described as singularity, invalid input, or
  broad method unsuitability.
- Preconditioners remain tuning tools, not convergence guarantees.
- No broad correctness, backend superiority, portable performance, or
  state-of-the-art claim was added.

### Day 8 Validation

Validation run:

```sh
git diff --check
python3 tests/test_selected_performance_docs.py
```

Day 8 changed documentation and planning artifacts only. No `.c` or `.h` files
were modified, so `make format && make lint && make test` is not required for
this day.

## Day 9: QR/SVD/Eigensolver Diagnostics Cleanup

### Implementation Summary

Day 9 applied the Day 7 diagnostics wording contract to QR, SVD, and symmetric
eigensolver documentation. The cleanup keeps QR rank/residual wording
QR-local, SVD rank/condition/residual/orthogonality wording SVD-local, and
eigensolver backend/residual wording tied to `sparse_eigs_t` telemetry. It
does not change API behavior, status codes, tolerance semantics, backend
selection, report freshness targets, or selected evidence scope.

### Changed Files

| File | Change |
| --- | --- |
| `README.md` | Normalized partial-SVD diagnostic scope, SVD API wording, QR rank/residual entries, and eigensolver `sparse_eigs_t` telemetry/AUTO-routing wording. |
| `docs/solver_selection.md` | Clarified QR fixture tolerance locality and SVD-local rank/fail-closed wording while preserving selected evidence boundaries. |
| `docs/cookbook.md` | Added SVD-local diagnostic scope and eigensolver `sparse_eigs_t` field guidance before backend/shift-invert/preconditioner escalation. |
| `examples/README.md` | Aligned QR example residual wording, SVD rank/condition wording, and eigensolver residual/backend handoff wording. |
| `docs/maintainer_guide.md` | Added explicit QR-local, SVD-local, and eigensolver AUTO-routing wording rules. |
| `artifacts/day9-qr-svd-eigs-diagnostics-cleanup.md` | Recorded before/after terminology notes, retained non-claims, retained semantics, and validation plan. |

### Retained Boundaries

- QR evidence remains fixture-local and tolerance-local.
- SVD and partial-SVD evidence remains fixture-local where named and API-local
  otherwise.
- Eigensolver AUTO remains routing policy, not backend superiority.
- Shift-invert wording does not treat transformed-operator residuals as
  original-A residuals unless recomputed by the owning docs/tests.
- No nonsymmetric eigensolver support, broad QR/SVD/eigs correctness,
  external-library parity, package/ABI proof, portable performance, or
  state-of-the-art claim was added.

### Day 9 Validation

Validation run:

```sh
git diff --check
python3 tests/test_selected_performance_docs.py
```

Day 9 changed documentation and planning artifacts only. No `.c` or `.h` files
were modified, so `make format && make lint && make test` is not required for
this day.

## Day 10: Header Narrative Audit

### Audit Summary

Day 10 audited public headers under `include/` for long-form workflow
narrative, duplicated examples, support/evidence interpretation, and adoption
guidance that could move to docs. No header files were edited. The result is a
bounded Day 11 candidate list plus preservation rules for generated API
documentation.

### Headers Reviewed

- `include/sparse_matrix.h`
- `include/sparse_csr.h`
- `include/sparse_iterative.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_eigs.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- representative direct, dense, reorder, vector, and type headers found by
  narrative search

### Main Findings

- Most public headers correctly own declarations, ownership, parameters,
  return values, result fields, callback lifetime, cleanup, and tolerance
  semantics.
- Several headers also contain cross-doc routing, workflow positioning,
  support/evidence caveats, benchmark-corpus references, or long usage examples
  that can stale independently of declarations.
- The safest Day 11 cleanup is narrow and declaration-preserving: trim routing
  paragraphs and move or shorten the longest `sparse_eigs.h` narrative, while
  preserving exact API contracts.

### Preservation Rules

Day 11 must preserve:

- declarations, typedefs, enum values, macro names, public constants, and
  struct fields;
- Doxygen parameter, return, note, see-also, ownership, and cleanup text;
- status-code mappings and result-field population rules;
- callback lifetime and cancellation semantics;
- tolerance formulas and default values;
- in-place mutation warnings;
- exact cleanup helper names and generated API anchors.

### Artifact

The full audit and candidate move list are recorded in
`artifacts/day10-header-narrative-audit.md`.

### Day 10 Validation

Validation run:

```sh
git diff --check
```

Day 10 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required for this
day.

## Day 11: Header Narrative Cleanup

### Cleanup Summary

Day 11 applied the selected cleanup from the Day 10 audit to six public
headers:

- `include/sparse_matrix.h`
- `include/sparse_csr.h`
- `include/sparse_iterative.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_eigs.h`

The cleanup removes or shortens tutorial-style routing, workflow-positioning,
benchmark/evidence examples, and adoption narrative that should live in the
documentation set rather than in declaration headers.

### Preserved API Surface

No declarations, typedefs, enum values, macro names, public constants, struct
fields, function signatures, status-code mappings, callback contracts, or
ownership/lifecycle requirements were intentionally changed. The edited
comments retain declaration-adjacent Doxygen coverage for parameters, return
values, option/result semantics, cleanup helpers, and local diagnostic scope.

### Artifact

The cleanup record is in `artifacts/day11-header-narrative-cleanup.md`.

### Day 11 Validation

Because public headers changed, Day 11 ran:

```sh
make format
make lint
make test
make api-docs-validate
```

Validation result: passed.

## Day 12: Documentation and Example Validation

### Validation Summary

Day 12 ran the documentation, examples, install/package, corpus/schema, report
freshness, and generated-output hygiene checks available on this machine.

### Commands Run

```sh
make docs-check api-docs-local-only qr-header-docs-guard \
  source-list-check ldlt-csc-helper-guard qr-external-ref-helper-guard \
  windows-powershell-guard
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
make tooling-build examples-build
bash tests/test_install.sh
bash tests/test_cmake_install.sh
make report-index-oracle-freshness report-index-comparison-freshness \
  bench-canonical-report-freshness
git status --short --ignored build docs/api
git ls-files docs/api build
git diff --check
```

### Results

- Doxygen/API docs coverage passed.
- Generated API local-only guard passed.
- QR header docs guard passed.
- Source-list, LDLT CSC helper, and QR external-reference helper guards passed.
- Windows PowerShell structural ownership guard passed; local `pwsh` execution
  remains unavailable and hosted-CI-owned.
- Corpus schema, selected report target manifest, selected performance docs,
  normalizer, selected comparison workflow, and benchmark freshness tests
  passed.
- Tooling build produced 16 benchmark binaries and 14 example binaries.
- Make install/`pkg-config` proof passed 23 checks.
- CMake install/export proof passed 27 checks, 0 failures, 0 skips.
- Selected oracle freshness passed for 54 generated rows.
- Selected comparison freshness passed for 46 generated rows.
- Canonical benchmark report freshness passed.
- Generated `build/` and `docs/api/` outputs remain ignored and untracked.
- `git diff --check` passed.

### Environment Residuals

- No dedicated Markdown link-check target was found in the current Makefile or
  validation scripts.
- Local PowerShell execution is unavailable because `pwsh` is not installed;
  this is recorded as an environment residual, not local hosted-execution
  evidence.

### Artifact

The Day 12 validation record is in
`artifacts/day12-docs-examples-install-validation.md`.

## Day 13: Full Quality Gate and Claim Calibration

### Validation Summary

Day 13 ran the final full quality gate because Day 11 changed public headers,
then re-ran affected documentation, install, example, guard, and selected
report freshness checks. It also audited the final documentation and header
surface for unsupported support, portability, package-manager, dynamic-library,
performance, release, or state-of-the-art claims.

### Commands Run

```sh
make format && make lint && make test

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

rg -n -i "state[- ]of[- ]the[- ]art|world[- ]class|best[- ]in[- ]class|production[- ]ready|fully supported|broad (windows|platform|package|performance)|homebrew/core|linuxbrew|vcpkg|conan|shared librar|dynamic ABI|runtime-loader|portable performance|performance guarantee|external-library parity|windows makefile parity|windows pkg-config" \
  README.md INSTALL.md docs/api_reference.md docs/cookbook.md \
  docs/maintainer_guide.md docs/solver_selection.md docs/tutorial.md \
  examples/README.md include
rg -n -i "supported|validated|not claimed|local-only|deferred|non-claim|claim boundary|package-manager|windows|performance" \
  README.md INSTALL.md docs/api_reference.md docs/cookbook.md \
  docs/maintainer_guide.md docs/solver_selection.md docs/tutorial.md \
  examples/README.md

find scripts/__pycache__ -type f -delete && rmdir scripts/__pycache__
git status --short --ignored build docs/api
git ls-files docs/api build
git diff --check
```

### Results

- Full `make format && make lint && make test` gate passed.
- Doxygen/API docs coverage, API local-only, QR header docs, source-list, LDLT
  CSC helper, QR external-reference helper, and Windows PowerShell ownership
  guards passed.
- Corpus schema, selected report target manifest, selected performance docs,
  normalizer, external-comparison runner, selected-comparison workflow, and
  selected benchmark freshness tests passed.
- Tooling build produced 16 benchmark binaries and 14 example binaries.
- Install proof passed 23 checks with 0 failures.
- CMake install/export proof passed 27 checks with 0 failures and 0 skips.
- Selected oracle freshness passed for 54 generated rows.
- Selected comparison freshness passed for 46 generated rows.
- Selected canonical benchmark freshness passed.
- Claim scan found only bounded or non-claim language; no unsupported promotion
  of package-manager availability, shared-library/dynamic ABI support, broad
  Windows parity, portable performance, external-library parity, release
  readiness, or state-of-the-art status was found.
- Generated `build/` and `docs/api/` outputs remain ignored and untracked.
- Python cache output generated during validation was removed.
- `git diff --check` passed.

### Environment Residuals

- Local PowerShell execution remains unavailable because `pwsh` is not
  installed; hosted CI owns the `--require-pwsh` execution evidence path.
- No dedicated Markdown link-check target was found in the current Makefile or
  validation scripts.

### Artifact

The Day 13 full-quality and claim-calibration record is in
`artifacts/day13-full-quality-claim-calibration.md`.

## Day 14: Closeout and Handoff

### Closeout Summary

Day 14 closed the Sprint 194 adoption/API coherence simplification scope. It
re-checked generated-output hygiene, final branch status, whitespace hygiene,
and claim boundaries, then recorded the final changed-file categories,
validation evidence, residuals, retrospective inputs, and PR-ready handoff
summary.

### Completed Scope

- Audited adoption friction across the maintained docs, examples, headers, and
  prior planning artifacts.
- Added an evidence-bounded install/support readiness matrix to `INSTALL.md`.
- Tightened consumer-facing wording around source builds, static install,
  CMake install/export, Windows CMake evidence, selected comparison freshness,
  selected benchmark freshness, package-manager non-claims, and shared-library
  / dynamic ABI deferrals.
- Improved first-solve and diagnostic routing in `README.md`, `docs/tutorial.md`,
  `docs/cookbook.md`, `docs/solver_selection.md`, `docs/api_reference.md`, and
  `examples/README.md`.
- Reduced tutorial-style adoption narrative in six public headers while
  preserving declaration-adjacent API contracts.
- Recorded full validation and claim-calibration evidence.

### Final Hygiene Commands

```sh
git status --short --ignored build docs/api
git ls-files docs/api build
git diff --check
find . -maxdepth 4 -type d -name __pycache__ -print
```

### Results

- `build/` and `docs/api/` remain ignored and untracked.
- No generated `build/` or `docs/api/` files are source-controlled.
- No Python `__pycache__` directories remain under the checked depth.
- `git diff --check` passed.
- Final claim scan continued to show bounded/non-claim language only.

### Retrospective Inputs

- The support/readiness matrix is the main consumer-facing simplification.
- Daily artifacts provide a clear audit-to-implementation-to-validation chain.
- Header cleanup reduced public API review surface without intentional
  declaration changes.
- Local `pwsh` remains unavailable; hosted CI owns PowerShell execution
  evidence.
- A dedicated Markdown link-check target remains a future validation candidate.
- Package-manager, shared-library/dynamic ABI, broad Windows parity, portable
  performance, release readiness, external-library parity, and state-of-the-art
  claims remain closed unless future sprints add validated support.

### Artifact

The Day 14 closeout and handoff record is in
`artifacts/day14-closeout-handoff.md`.
