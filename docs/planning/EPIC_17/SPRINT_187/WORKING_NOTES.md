# Sprint 187 Working Notes

## Sprint Goal

Freeze the post-Epic-16 baseline, convert the Codex review into a
source-controlled gap ledger, and select the exact Epic 17 closures.

## Branch Baseline

- Branch: `sprint-187`
- Starting point: current `master` after PR #207 merge.
- Epic 16 status: complete with explicit residuals.
- Epic 17 review package: merged under `docs/planning/EPIC_17/reviews/`.
- Sprint 187 plan status: day-by-day plan exists at
  `docs/planning/EPIC_17/SPRINT_187/PLAN.md`.

## Planning Source

| Field | Value |
| --- | --- |
| Project plan | `docs/planning/EPIC_17/PROJECT_PLAN.md` |
| Section | `Sprint 187: Epic 17 Baseline, Gap Ledger & Acceptance Gates` |
| Sprint duration | 14 days, approximately 166 hours |
| Review source | `docs/planning/EPIC_17/reviews/review-codex-2026-08-28.md` |
| Todo source | `docs/planning/EPIC_17/reviews/todo-codex-2026-08-28.md` |
| Prior residual source | `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md` |

## Sprint 187 Item Boundaries

| Item | Name | Sprint 187 interpretation |
| --- | --- | --- |
| 187.1 | Review Intake | Convert the Codex review findings into a prioritized Epic 17 gap ledger with owner files and claim risks. |
| 187.2 | Residual Reconciliation | Reconcile Epic 16 residuals with review findings and deduplicate package, Windows, comparison, performance, and review-surface gaps. |
| 187.3 | Closure Selection | Select the complete gaps Epic 17 will close and record non-goals for broad state-of-the-art, ABI, platform, package, and performance claims. |
| 187.4 | Acceptance Gates | Define validation commands, hosted/local ownership, artifact expectations, and support-tier wording for each selected closure. |
| 187.5 | Quality Surface Map | Map required checks for docs-only, script, workflow, package, generated-report, benchmark, and C/header changes. |
| 187.6 | Sprint Handoff | Create Sprint 187 artifacts, working notes, and handoff records for package and Windows work. |

## Day 1 Source Artifact Inventory

| Source family | Files | Day 1 use |
| --- | --- | --- |
| Epic 17 planning | `docs/planning/EPIC_17/PROJECT_PLAN.md`; `docs/planning/EPIC_17/SPRINT_187/PLAN.md` | Owns Sprint 187 scope, estimates, deliverables, and day-by-day execution boundary. |
| Codex review | `docs/planning/EPIC_17/reviews/review-codex-2026-08-28.md` | Primary source for review findings, state-of-the-art assessment, and prioritized gap list. |
| Codex todo | `docs/planning/EPIC_17/reviews/todo-codex-2026-08-28.md` | Primary source for step-by-step closure strategy and candidate validation commands. |
| Epic 16 closeout | `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md`; `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md` | Source of inherited residuals, earned claims, retained non-claims, and next-epic closure targets. |
| User docs | `README.md`; `INSTALL.md`; `docs/api_reference.md`; `docs/maintainer_guide.md`; `benchmarks/README.md`; `docs/solver_selection.md`; `docs/cookbook.md`; `docs/tutorial.md`; `docs/matrix_market.md` | Current public support, adoption, package, API, benchmark, and solver-selection truth. |
| Build/package owners | `Makefile`; `CMakeLists.txt`; `sparse.pc.in`; `cmake/SparseConfig.cmake.in`; `packaging/homebrew/`; `scripts/homebrew_local_formula_proof.sh`; package guard scripts | Package proof, install/export, static-first support, and package-manager non-claim ownership. |
| Platform owners | `.github/workflows/ci.yml`; `.github/workflows/macos-ci.yml`; `.github/workflows/windows-ci.yml`; `tests/test_install.sh`; `tests/test_cmake_install.sh` | Hosted validation, Windows support, install/downstream proof, and platform-support scope. |
| Report/evidence owners | `tests/corpus/manifests/`; `tests/corpus/schemas/`; `scripts/run_external_comparison.py`; `scripts/normalize_report_index.py`; report freshness tests | Selected report target authority, oracle/comparison freshness, and generated evidence workflow ownership. |
| Code/test owners | `include/`; `src/`; `tests/`; `benchmarks/`; `examples/` | Future maintainability, reliability, API, benchmark, and example/adoption closure owners. |

## Day 1 Owner Surface Map

| Closure family | Initial owner surfaces | Notes for later days |
| --- | --- | --- |
| Package proof | Root license metadata; `packaging/homebrew/`; `scripts/homebrew_local_formula_proof.sh`; `INSTALL.md`; `README.md`; package guards | Day 7 must define proof-promotion criteria without implying Homebrew/core, bottle, Linuxbrew, public tap, or broad provider support. |
| Windows validation | `.github/workflows/windows-ci.yml`; PowerShell snippets; selected workflow guards; manifest/schema tests; maintainer guide | Day 8 must separate PowerShell validation ownership from Windows report freshness promotion. |
| Windows report freshness | Windows workflow; selected report target manifest; report generator/normalizer scripts; report docs | Day 8 must define both promotion and renewed-deferral gates. |
| External comparison | `scripts/run_external_comparison.py`; comparison tests; selected manifest rows; report family docs | Day 9 must require exact fixture, metric, tolerance, dependency, artifact, and non-claim fields. |
| Performance evidence | `benchmarks/`; benchmark report scripts; report index normalization; CI freshness jobs; benchmark docs | Day 9 must distinguish methodology-bound evidence from portable performance claims. |
| Maintainability | Largest source/test candidates; helper guard scripts; source-list checks; maintainer guide | Day 10 must select one future cluster and require no-behavior-change invariants. |
| Adoption/API coherence | README, INSTALL, tutorial, cookbook, solver selection, API reference, examples, public headers | Day 11 must separate compact user truth from historical sprint evidence. |
| Reliability proof | Allocation/failure-path candidates; deterministic failure hooks; focused Make/CTest gates; README/maintainer docs | Day 10 must define owner-selection criteria before Sprint 195 work starts. |

## Day 1 Risks

| Risk | Mitigation |
| --- | --- |
| The Epic 17 ledger could duplicate Epic 16 residual IDs under new names. | Day 3 must preserve residual traceability and record a one-to-one or merged mapping. |
| Closure selection could spread effort across too many partial gaps. | Day 5 ranks feasibility and Day 6 selects only complete-gap targets. |
| Package proof could overclaim provider support after local Homebrew proof passes. | Day 7 acceptance gates must separate local proof from Homebrew/core, bottle, Linuxbrew, public tap, and broad provider support. |
| Windows validation could be confused with broad Windows parity. | Day 8 gates must keep PowerShell validation, report freshness, CMake support, Makefile parity, and `pkg-config` execution parity separate. |
| Comparison and performance work could imply state-of-the-art parity. | Day 9 gates must require named fixtures, metrics, tolerances, dependencies, support tiers, and explicit non-claims. |
| Maintainability extraction could change behavior while looking like cleanup. | Day 10 gates must require no-behavior-change invariants, focused tests, source-list checks, and full C gates for C/header changes. |

## Day 1 Open Questions

| Question | Day 1 disposition |
| --- | --- |
| Should Sprint 187 implement code changes? | No. Sprint 187 is a baseline, ledger, gate, quality-map, and handoff sprint. |
| Should Epic 17 try to earn a broad state-of-the-art claim? | No. The review recommends selected complete closures and final calibration rather than an unqualified state-of-the-art claim. |
| Should hosted API publication be selected in Epic 17? | Not by default. It remains an Epic 16 residual candidate, but closure selection must compare it against package, Windows, comparison, performance, maintainability, adoption, and reliability work. |
| Should Windows report freshness block Sprint 187 closeout? | No. Sprint 187 only defines the acceptance gates and handoff for later Windows work. |
| What validation is required for Day 1? | Documentation-only checks: `git diff --check`, direct whitespace scan of new untracked docs, and confirmation that no `.c` or `.h` files changed. |

## Day 2 Gap-Ledger Schema Draft

Day 2 should create the first structured gap ledger with these fields:

| Field | Purpose |
| --- | --- |
| Gap ID | Stable Epic 17 identifier, preserving Epic 16 residual IDs when applicable. |
| Source | Codex review section, Codex todo phase, Epic 16 residual, or current source/doc/CI evidence. |
| Area | Efficiency, maintainability, usability, documentation, coherence, test coverage, packaging, platform, performance, comparison, reliability, or state-of-the-art readiness. |
| Finding | Concise description of the gap or shortcoming. |
| Owner surfaces | Source, docs, tests, scripts, workflows, manifests, reports, examples, or planning files that own the gap. |
| Current evidence | Existing proof, guard, test, docs, CI lane, report row, or explicit non-claim. |
| Claim risk | What unsupported claim could be implied if the gap is mishandled. |
| User value | Adoption, correctness confidence, package usability, platform confidence, performance credibility, maintainability, or support clarity. |
| Closure candidate | Yes, no, or long-horizon, with rationale. |
| Candidate sprint | Tentative Sprint 188-195 target if selected. |
| Required validation | Commands or hosted checks expected for closure. |
| Non-goals | Explicit unsupported breadth that must remain out of scope. |

## Day 1 Validation

Day 1 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 2 Review Finding Extraction

Day 2 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day2-review-intake-matrix.md`
as the first structured Epic 17 gap ledger from the Codex review and todo.

### Day 2 Initial Ledger Summary

| Category | Count | Interpretation |
| --- | ---: | --- |
| Closure candidate | 9 | Candidate gaps that map directly to planned Sprint 188-195 closure families. |
| Long-horizon | 4 | Important gaps that remain too broad for full closure inside Epic 17 unless a later sprint explicitly narrows them. |
| Retained non-claim | 3 | Surfaces that should stay explicitly unsupported unless a later product decision changes scope. |

### Day 2 Area Coverage

| Area | Initial gap IDs |
| --- | --- |
| Packaging | E17-GAP-001, E17-GAP-014 |
| Platform and Windows | E17-GAP-002, E17-GAP-003, E17-GAP-015 |
| Comparison and performance | E17-GAP-004, E17-GAP-005, E17-GAP-010, E17-GAP-012 |
| Maintainability | E17-GAP-006, E17-GAP-011 |
| Usability and documentation | E17-GAP-007, E17-GAP-008 |
| Reliability and test coverage | E17-GAP-009, E17-GAP-013 |
| State-of-the-art readiness | E17-GAP-016 |

### Day 2 Long-Horizon Non-Goals

- Unqualified state-of-the-art sparse linear algebra status.
- Broad external ecosystem parity.
- Portable performance superiority.
- Shared-library and dynamic ABI support.
- Broad package-manager and broad Windows parity.

### Day 2 Validation

Day 2 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 3 Epic 16 Residual Reconciliation

Day 3 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day3-residual-reconciliation.md`
as the deduplicated mapping from Epic 16 residuals into the Epic 17 gap
ledger.

### Day 3 Residual Disposition Summary

| Disposition | Count | Residuals |
| --- | ---: | --- |
| Selected closure candidate | 5 | `R186-PKG-LICENSE`, `R186-WIN-PWSH`, `R186-WIN-REPORT-FRESHNESS`, `R186-BROAD-COMPARISON`, `R186-REVIEW-SURFACE-NEXT` |
| Long-horizon retained decision | 1 | `R186-HOSTED-API` |
| Dropped as duplicate | 0 | No Epic 16 residual was dropped; each maps to a current Epic 17 gap or retained decision. |

### Day 3 Reconciled Mapping

| Epic 16 residual | Epic 17 gap mapping | Day 3 disposition |
| --- | --- | --- |
| `R186-PKG-LICENSE` | `E17-GAP-001 / R186-PKG-LICENSE` | Selected for Sprint 188 package proof completion. |
| `R186-WIN-PWSH` | `E17-GAP-002 / R186-WIN-PWSH` | Selected for Sprint 189 PowerShell validation ownership. |
| `R186-WIN-REPORT-FRESHNESS` | `E17-GAP-003 / R186-WIN-REPORT-FRESHNESS` | Selected for Sprint 190 Windows report freshness decision. |
| `R186-HOSTED-API` | Folded into `E17-GAP-008` and long-horizon generated API publication deferral | Retained as a product-documentation decision, not selected by default for Sprint 188-195 closure. |
| `R186-BROAD-COMPARISON` | `E17-GAP-004 / R186-BROAD-COMPARISON` | Selected for Sprint 191 bounded external comparison family. |
| `R186-REVIEW-SURFACE-NEXT` | `E17-GAP-006 / R186-REVIEW-SURFACE-NEXT` plus broad `E17-GAP-011` context | Selected for Sprint 193 as one bounded review-surface reduction, not broad multi-module cleanup. |

### Day 3 Closure Candidate Split

- Package proof, PowerShell validation, Windows report freshness, bounded
  comparison, performance evidence, selected review-surface reduction,
  adoption/API coherence, documentation coherence, and one selected
  reliability proof remain valid Sprint 188-195 closure candidates.
- Hosted API publication stays long-horizon unless Day 5 and Day 6 ranking
  explicitly selects it over one of the currently planned complete closures.
- Broad storage replacement, broad numerical robustness, broad coverage,
  shared-library/dynamic ABI productization, broad Windows parity, and
  unqualified state-of-the-art positioning remain non-goals for Sprint 187
  selection.

### Day 3 Validation

Day 3 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 4 Owner Surface and Evidence Inventory

Day 4 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day4-owner-surface-inventory.md`
as the owner-file, validation-command, and environment-dependency inventory
for the reconciled Epic 17 closure candidates.

### Day 4 Owner Inventory Summary

| Closure family | Primary owner surfaces | Existing validation | Missing or future validation |
| --- | --- | --- | --- |
| Homebrew proof completion | Root license metadata, `packaging/homebrew/`, package guards, install docs | Homebrew proof script and package deferral guards exist | Full proof still needs license strategy and passing formula install/test/uninstall evidence |
| PowerShell validation ownership | `.github/workflows/windows-ci.yml`, PowerShell snippets, report workflow guards | Windows CMake/MSVC CI exists | Owned `pwsh` parse/workflow validation command |
| Windows report freshness | Windows CI, selected report manifest, report scripts, report docs | Manifest/schema/normalizer checks exist | Selected Windows freshness lane or strengthened deferral guard |
| Bounded comparison | Comparison runner, selected target manifest, external reference helpers, report docs | Current selected comparison freshness exists | One new bounded family with exact fixture, metrics, tolerance, and freshness evidence |
| Performance evidence | Benchmark drivers, canonical report scripts, performance sentinels, benchmark docs | `bench-canonical-report-freshness` and `performance-sentinels` exist | Methodology-bound hosted lane with explicit metadata and artifact policy |
| Review-surface reduction | Large source/test candidates, helper guards, source-list checks | Source-list and LDLT CSC helper guard patterns exist | One selected cluster guard and behavior-preserving focused tests |
| Adoption/API coherence | README, INSTALL, tutorial, cookbook, solver selection, API reference, examples, public headers | Docs and Doxygen checks exist | Compact support/readiness matrix and diagnostics coherence checks |
| Reliability proof | Allocation/failure hooks, selected owner tests, focused gates | Iterative and matmul allocation-failure gates exist | One new selected owner with deterministic failure-path proof |

### Day 4 Local Environment Notes

| Tool | Local status | Planning impact |
| --- | --- | --- |
| `brew` | Available at `/usr/local/bin/brew` | Sprint 188 can plan a local Homebrew proof path, subject to license metadata. |
| `pwsh` | Not available locally | Sprint 189 must rely on hosted Windows or document local skip behavior. |
| `cmake` / `ctest` | Available locally | CMake parity and install/downstream checks are locally inspectable. |
| `pkg-config` | Available locally | Unix-side static install proof remains locally inspectable. |
| `gh` | Available locally | Hosted PR/workflow evidence can be inspected when later sprints need it. |

### Day 4 Inputs To Day 5

Day 5 should rank closure candidates using:

- owner-surface clarity;
- existing validation availability;
- missing validation size;
- environment dependency risk;
- user/adoption value;
- claim-risk reduction;
- likelihood of complete closure inside one 14-day sprint.

### Day 4 Validation

Day 4 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 5 Gap Ranking and Feasibility

Day 5 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day5-gap-ranking-and-feasibility.md`
as the ranked gap ledger and feasibility record for Sprint 188 through Sprint
195 planning.

### Day 5 Ranked Shortlist

| Rank | Candidate | Target sprint | Feasibility | Primary reason |
| ---: | --- | --- | --- | --- |
| 1 | Homebrew proof completion | Sprint 188 | High | Clear blocker, existing proof script, local `brew` available, high adoption value. |
| 2 | PowerShell validation ownership | Sprint 189 | Medium-high | Hosted Windows can own proof even though local `pwsh` is unavailable. |
| 3 | Windows report freshness decision | Sprint 190 | Medium | Depends on Sprint 189; can close as selected lane or renewed deferral. |
| 4 | Bounded external comparison family | Sprint 191 | Medium-high | Existing comparison runner and manifest flow make one new family feasible. |
| 5 | Methodology-bound performance lane | Sprint 192 | Medium | Existing benchmark reports help, but hosted runtime/variance policy adds risk. |
| 6 | Selected review-surface reduction | Sprint 193 | Medium-high | Prior LDLT CSC pattern exists; must keep scope to one cluster. |
| 7 | Adoption and API coherence | Sprint 194 | High | Mostly documentation/examples with strong user value and clear validation. |
| 8 | Selected reliability proof | Sprint 195 | Medium | Valuable but requires careful owner selection and deterministic failure harness. |

### Day 5 Not Selected By Default

- `R186-HOSTED-API`: retained as a long-horizon generated API publication
  product decision.
- Broad storage-model replacement: too large for Epic 17 complete closure.
- Shared-library and dynamic ABI support: retained static-first non-claim.
- Broad Windows parity: narrowed to PowerShell validation and one report
  freshness decision.
- Broad state-of-the-art sparse linear algebra claim: retained non-claim until
  final calibration.

### Day 5 Dependencies

| Dependency | Effect |
| --- | --- |
| Sprint 188 before package claim updates | Homebrew proof completion controls whether package wording can be promoted. |
| Sprint 189 before Sprint 190 | PowerShell validation ownership is a prerequisite for Windows report freshness promotion. |
| Sprint 191 before final state-of-the-art calibration | New comparison evidence affects final claim boundaries. |
| Sprint 192 before final performance wording | Methodology-bound performance evidence controls any performance claim promotion. |
| Sprint 193 before Sprint 195 if selecting the same owner | Maintainability extraction should not overlap reliability proof unless Day 6 selects that coupling explicitly. |

### Day 5 Validation

Day 5 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 6 Closure Target Selection

Day 6 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day6-closure-target-selection.md`
as the Epic 17 closure-selection artifact. It converts the Day 5 ranked
shortlist into exact Sprint 188 through Sprint 195 closure targets, complete
definitions of done, explicit non-goals, and fallback rules.

### Day 6 Selected Closure Targets

| Sprint | Selected target | Complete definition of done |
| --- | --- | --- |
| Sprint 188 | Homebrew proof completion | The local formula path has a passing install/test/uninstall proof or a formally guarded license-strategy blocker, and package claims match the evidence. |
| Sprint 189 | PowerShell validation ownership | PowerShell snippets and scripts have an owned validation path in hosted Windows or explicit skip semantics when `pwsh` is absent locally. |
| Sprint 190 | Windows selected report freshness decision | One Windows report freshness lane is promoted with validation, or the deferral is renewed with stronger guards and revisit criteria. |
| Sprint 191 | Bounded external comparison family | One additional external comparison family has fixed fixtures, tolerances, manifests, freshness checks, and documentation. |
| Sprint 192 | Methodology-bound performance evidence lane | One bounded performance lane has methodology metadata, freshness validation, runtime policy, and calibrated claim wording. |
| Sprint 193 | Selected large review-surface reduction | One large review-surface cluster is reduced through behavior-preserving extraction and focused guard coverage. |
| Sprint 194 | Adoption and API coherence simplification | User-facing setup, support, examples, diagnostics, and API entry points are made consistent with the proven support tiers. |
| Sprint 195 | Selected reliability and failure-path proof | One owner gets deterministic failure-path evidence covering cleanup, stale output, retry, and no-global-state contamination. |

### Day 6 Explicit Non-Goals

- Unqualified state-of-the-art sparse linear algebra positioning.
- Shared-library, dynamic ABI, and binary compatibility guarantees.
- Broad Windows platform parity beyond the selected PowerShell and report
  freshness lanes.
- Broad package-manager distribution beyond the selected local Homebrew proof.
- Broad external parity across all candidate libraries or solver families.
- Portable performance leadership claims across architectures, compilers,
  matrix families, and dependency stacks.
- Hosted generated API publication unless a later Epic explicitly selects it.

### Day 6 Fallback Rules

Every selected target must close as either a promoted support claim backed by
passing validation or a retained non-claim backed by explicit blockers, guards,
and revisit criteria. Partial promotion without evidence is rejected for Epic
17.

### Day 6 Validation

Day 6 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 7 Package Acceptance Gates

Day 7 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day7-package-acceptance-gates.md`
as the Sprint 188 package and Homebrew gate definition. The gate keeps the
selected scope to local Homebrew source formula proof, static archive install
behavior, and claim-safe package documentation.

### Day 7 Gate Summary

| Gate | Acceptance requirement |
| --- | --- |
| License metadata | A standalone root `LICENSE`, `COPYING`, or `NOTICE` file exists and `SPARSE_HOMEBREW_LICENSE` maps to accurate Homebrew formula metadata, or the blocker remains guarded as unavailable. |
| Formula material | `packaging/homebrew/sparse-lu-ortho.rb.in` remains a temporary rendered local formula template with version, archive URL, checksum, and license placeholders. |
| Proof script | `scripts/homebrew_local_formula_proof.sh` proves render, archive, checksum, install, installed static surface, `brew test`, uninstall, and cleanup. |
| Package guards | `scripts/package_manager_deferral_check.sh` and `scripts/static_package_deferral_check.sh` preserve provider, shared-library, and dynamic ABI non-claims. |
| Documentation | README, INSTALL, Homebrew README, and maintainer guidance promote only the evidence-backed local proof state. |

### Day 7 Required Validation

- `SPARSE_HOMEBREW_LICENSE=<accurate-id> scripts/homebrew_local_formula_proof.sh`
  when the license strategy is complete.
- `scripts/package_manager_deferral_check.sh`.
- `scripts/static_package_deferral_check.sh`.
- Docs and C quality gates selected by the actual Sprint 188 file changes.

### Day 7 Retained Non-Claims

Homebrew/core, bottles, Linuxbrew, public taps, binary packages, hosted
provider registries, vcpkg, Conan, pkgsrc, distro packages, shared libraries,
dynamic ABI, and broad package-manager support remain outside Sprint 188.

### Day 7 Validation

Day 7 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 8 Windows Acceptance Gates

Day 8 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day8-windows-acceptance-gates.md`
as the Sprint 189 and Sprint 190 Windows gate definition. The gate separates
PowerShell validation ownership from Windows report freshness promotion and
keeps broad Windows parity as a non-claim.

### Day 8 Gate Summary

| Sprint | Gate | Acceptance requirement |
| --- | --- | --- |
| Sprint 189 | PowerShell validation ownership | A maintained validation command or hosted Windows lane parses/runs the selected PowerShell workflow material, with local `pwsh` absence handled as an explicit skip/residual. |
| Sprint 189 | Workflow owner boundary | `.github/workflows/windows-ci.yml` remains CMake-first and static-first while any new PowerShell validation step is scoped to validation ownership only. |
| Sprint 190 | Promotion path | Exactly one Windows-safe report freshness lane may be promoted only with manifest metadata, generator proof, upload scope, freshness checks, and docs alignment. |
| Sprint 190 | Deferral path | If promotion is rejected, the Sprint 182 deferral is renewed with explicit blockers, guards, and revisit criteria. |
| Sprint 190 | Stale-output protection | Windows selected report artifacts must use exact artifact names, `if-no-files-found: error`, manifest expected rows, and freshness diagnostics. |

### Day 8 Required Validation

- `python3 tests/test_selected_report_targets_manifest.py`.
- `python3 tests/test_selected_comparison_workflow.py`.
- `python3 scripts/validate_corpus_schema.py`.
- `python3 scripts/normalize_report_index.py --check-freshness` and selected
  `--family ... --require-generated ... --check-freshness` commands when
  report freshness metadata or generated report families change.
- Hosted Windows workflow evidence for any promoted PowerShell or freshness
  lane.

### Day 8 Retained Non-Claims

Windows Makefile parity, Windows `pkg-config` execution parity, Bash/POSIX
report generation, package-manager support, shared libraries, dynamic ABI,
runtime-loader behavior, broad generated report freshness, portable
performance, and broad Windows platform parity remain outside Sprints 189 and
190.

### Day 8 Validation

Day 8 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 9 Comparison and Performance Gates

Day 9 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day9-comparison-performance-gates.md`
as the Sprint 191 and Sprint 192 evidence-lane gate definition. The gate
requires one bounded external comparison family and one methodology-bound
performance lane, each with manifest metadata, freshness checks, and explicit
non-claims.

### Day 9 Gate Summary

| Sprint | Gate | Acceptance requirement |
| --- | --- | --- |
| Sprint 191 | Family selection | Exactly one new comparison family, fixture, reference path, metric set, tolerance policy, and dependency behavior are selected. |
| Sprint 191 | Report and manifest integration | `scripts/run_external_comparison.py`, `tests/corpus/manifests/selected_report_targets.tsv`, and normalizer checks agree on expected rows, required files, artifacts, claim scope, and non-claims. |
| Sprint 191 | Freshness evidence | `make report-index-comparison-freshness` and selected normalizer checks fail stale, missing, skipped, duplicate, or unexpected rows. |
| Sprint 192 | Lane selection | Exactly one benchmark lane, fixture, platform, repeat policy, runtime budget, support tier, and claim boundary are selected. |
| Sprint 192 | Methodology metadata | The benchmark report records compiler, flags, CPU, thread count, build mode, commit, branch, timestamp, fixture, repeats, warmup, variance, baseline, threshold, backend context, and methodology notes. |
| Sprint 192 | Hosted freshness | Hosted CI uploads exact selected benchmark artifacts and validates freshness without claiming portable performance. |

### Day 9 Required Validation

- `python3 scripts/validate_corpus_schema.py`.
- `python3 tests/test_selected_report_targets_manifest.py`.
- `python3 tests/test_selected_comparison_workflow.py`.
- `make report-index-comparison-freshness` for Sprint 191 comparison changes.
- `make bench-canonical-report-freshness` and
  `python3 scripts/check_bench_canonical_freshness.py --mode hosted` for
  Sprint 192 hosted performance promotion.
- `make format && make lint && make test` whenever `.c` or `.h` files change.

### Day 9 Retained Non-Claims

Broad external-library parity, broad solver correctness, raw factor/vector
identity, dependency ecosystem coverage, portable performance, performance
superiority, algorithmic superiority, platform parity, package proof, ABI
proof, release benchmark claims, and unqualified state-of-the-art status remain
outside Sprints 191 and 192.

### Day 9 Validation

Day 9 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 10 Maintainability and Reliability Gates

Day 10 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day10-maintainability-reliability-gates.md`
as the Sprint 193 and Sprint 195 code-quality gate definition. The gate uses
current large-file size, Sprint 185 helper extraction, source-list parity, and
allocation-failure proof patterns to define measurable completion criteria.

### Day 10 Gate Summary

| Sprint | Gate | Acceptance requirement |
| --- | --- | --- |
| Sprint 193 | Candidate ranking | Select exactly one high-risk source/test cluster using size, algorithm risk, helper ownership, current tests, and user-facing importance. |
| Sprint 193 | No-behavior-change invariants | Preserve public declarations, statuses, tolerances, fixtures, test names, registration, and claim boundaries unless a separate reviewed behavior change is selected. |
| Sprint 193 | Guard ownership | Add or update a focused guard like `make ldlt-csc-helper-guard` for helper presence, include ownership, registration, and source-list boundaries. |
| Sprint 195 | Reliability owner selection | Select exactly one allocation-heavy or failure-prone owner with deterministic failure hooks and user-visible cleanup semantics. |
| Sprint 195 | Failure-path proof | Cover failed allocation, cleanup, stale-output suppression, retry-after-reset, and global-state restoration where applicable. |
| Both | Required C validation | Any `.c` or `.h` change requires focused tests, source-list checks when registration changes, and `make format && make lint && make test`. |

### Day 10 Required Validation

- Focused proof-owner test binary for the selected cluster or reliability
  owner.
- Cluster-specific guard such as `make ldlt-csc-helper-guard`, or a new
  equivalent guard when Sprint 193 selects a different cluster.
- `make source-list-check` when library sources, test registrations, or CMake
  source lists change.
- `make iterative-allocation-failure-gate` or
  `make matmul-allocation-failure-gate` when those existing reliability owners
  are touched.
- `make format && make lint && make test` for any `.c` or `.h` change.

### Day 10 Retained Non-Claims

Sprint 193 reduces review surface only; it does not claim new numerical
behavior, public API, ABI, package/platform support, performance, external
parity, or state-of-the-art status. Sprint 195 proves one selected failure
surface only; it does not claim exhaustive allocation-failure coverage,
concurrency safety, all-solver reliability, or broad lifecycle correctness.

### Day 10 Validation

Day 10 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 11 Adoption and Documentation Gates

Day 11 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day11-adoption-documentation-gates.md`
as the Sprint 194 adoption and API coherence gate definition. The gate defines
how README, INSTALL, tutorial, cookbook, solver selection, API reference,
examples, public headers, and maintainer guidance should separate user-facing
truth from historical planning evidence.

### Day 11 Gate Summary

| Gate | Acceptance requirement |
| --- | --- |
| Adoption audit | Audit README, INSTALL, tutorial, cookbook, solver selection, API reference, examples, public headers, and maintainer guide for duplicate or contradictory workflow guidance. |
| Support/readiness matrix | Add a compact matrix that states build, install, package, platform, report, comparison, performance, API, and reliability support tiers without hiding non-claims. |
| Installed consumer tutorial | Provide a minimal external-consumer path for Make/`pkg-config` and CMake `find_package(Sparse)` without implying package-manager or shared-library support. |
| Diagnostics coherence | Normalize direct, iterative, QR/SVD, and eigensolver wording around status codes, residuals, convergence, retry, cleanup, and unsupported breadth. |
| Header narrative cleanup | Move broad tutorial or policy narrative out of selected public headers while preserving declarations, Doxygen coverage, and exact public semantics. |
| Validation map | Require links, Doxygen/API docs, examples, install checks, package/header guards, and full C quality gates when matching surfaces change. |

### Day 11 Required Validation

- `git diff --check` and Markdown link checks for docs-only changes.
- `make docs-check` and `make api-docs-freshness` when API docs, Doxygen
  inputs, or public headers change.
- `make examples` or the selected example build path when examples change.
- `bash tests/test_install.sh` and/or `bash tests/test_cmake_install.sh` when
  install or downstream-consumer docs change.
- Header-specific guards such as `bash scripts/check_qr_header_docs_guard.sh`
  and `bash scripts/check_lu_header_docs_guard.sh` when those surfaces change.
- `make format && make lint && make test` whenever `.c` or `.h` files change.

### Day 11 Retained Non-Claims

Sprint 194 improves adoption and documentation coherence only. It does not
promote new solver behavior, broad numerical correctness, package-manager
support, shared-library support, dynamic ABI, broad Windows parity, portable
performance, external-library parity, hosted API publication, or unqualified
state-of-the-art status.

### Day 11 Validation

Day 11 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 12 Quality Surface Map

Day 12 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day12-quality-surface-map.md`
as the reusable Epic 17 validation matrix. It maps planning, documentation,
script, workflow, package, report, benchmark, public-header, C implementation,
test, and generated-artifact changes to required checks, optional stronger
checks, hosted dependencies, and local skip rules.

### Day 12 Validation Policy Summary

| Surface | Minimum required validation |
| --- | --- |
| Planning-only docs | `git diff --check`, trailing-whitespace scan, and Markdown link check over the changed planning tree. |
| User docs | `git diff --check`, Markdown link check, and the surface-specific docs, install, example, or claim guard. |
| Scripts | Script syntax/targeted script test plus any owner guard named by the changed surface. |
| Workflows | Workflow guard tests plus hosted run evidence before any support promotion. |
| Package/install | Package guards, install tests, and Homebrew proof when Sprint 188 package wording is promoted. |
| Reports/manifests | Corpus schema, selected target manifest tests, normalizer freshness checks, and family-specific report freshness target. |
| Benchmarks | Benchmark report freshness, hosted-mode freshness checks for promoted hosted evidence, and workflow artifact guards. |
| Public headers | `make docs-check`, `make api-docs-freshness`, header docs guards, and full C gate. |
| C implementation/tests | Focused owner tests, registration/source-list checks as needed, and `make format && make lint && make test`. |

### Day 12 Mandatory C Gate

Any `.c` or `.h` change requires:

```sh
make format
make lint
make test
```

This requirement is not replaced by focused owner tests, docs checks, hosted
workflow evidence, or generated report checks.

### Day 12 Hosted and Skip Rules

Hosted evidence is required before promoting Windows, macOS/Linux hosted
comparison, hosted performance, or package/workflow claims. Local tool absence
such as missing `pwsh`, `brew`, optional external dependencies, or hosted-only
runner metadata must be recorded as unavailable/skip/residual, not pass
evidence.

### Day 12 Validation

Day 12 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 13 Implementation Handoffs

Day 13 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day13-implementation-handoffs.md`
as the Sprint 188 through Sprint 195 implementation handoff package. The
handoff links each selected closure to its gap source, acceptance-gate
artifact, owner files, first implementation steps, validation commands, done
state, retained non-goals, and pre-closeout open questions.

### Day 13 Handoff Summary

| Future sprint | Handoff focus | Primary gate source |
| --- | --- | --- |
| Sprint 188 | Resolve Homebrew license metadata and prove the local formula workflow. | Day 7 package acceptance gates |
| Sprint 189 | Add owned PowerShell validation without promoting report freshness. | Day 8 Windows acceptance gates |
| Sprint 190 | Promote one Windows-safe report freshness lane or renew formal deferral. | Day 8 Windows acceptance gates |
| Sprint 191 | Add one bounded external comparison family with manifest and freshness proof. | Day 9 comparison gates |
| Sprint 192 | Promote one methodology-bound hosted performance evidence lane. | Day 9 performance gates |
| Sprint 193 | Reduce exactly one large review surface with no-behavior-change validation. | Day 10 maintainability gates |
| Sprint 194 | Consolidate adoption docs, support truth, installed-consumer guidance, and API narrative. | Day 11 adoption gates |
| Sprint 195 | Add deterministic failure-path proof for one selected reliability owner. | Day 10 reliability gates |

### Day 13 Cross-Sprint Dependencies

- Sprint 188 package wording should land before Sprint 194 consolidates
  adoption and support-matrix truth.
- Sprint 189 PowerShell validation ownership should land before Sprint 190
  decides whether any Windows report freshness lane can be promoted.
- Sprint 190 Windows freshness or deferral wording should land before Sprint
  194 finalizes Windows support rows.
- Sprint 191 comparison work and Sprint 192 performance work must keep their
  evidence and claim boundaries separate.
- Sprint 193 review-surface selection should be known before Sprint 195
  chooses its reliability owner to avoid overlapping churn.

### Day 13 Pre-Closeout Open Questions

| Sprint | Open question |
| --- | --- |
| 188 | What exact license identifier and root metadata file are approved for Homebrew formula rendering? |
| 189 | What command name owns PowerShell validation, and where should local `pwsh` skip semantics be documented? |
| 190 | Which Windows-safe report lane should be promoted, or should the sprint renew deferral instead? |
| 191 | Which bounded comparison family, fixture, dependency, and tolerance set are selected? |
| 192 | Should the performance lane remain `bench_refactor_csc` on `nos4.mtx --repeat 1`, and should it stay threshold-free? |
| 193 | Which large source/test cluster is selected, and what exact invariants preserve behavior? |
| 194 | Where will the compact support/readiness matrix live, and which headers are eligible for narrative cleanup? |
| 195 | Which reliability owner is selected, and does it overlap with Sprint 193 changes? |
| 196 | Which earned claims survive final validation, and which remain explicit non-claims? |

### Day 13 Validation

Day 13 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

## Day 14 Sprint Closeout

Day 14 created
`docs/planning/EPIC_17/SPRINT_187/artifacts/day14-closeout-summary.md`
as the final Sprint 187 closeout record. The closeout reviews all artifacts
against project-plan items 187.1 through 187.6, confirms the ledger, residual
mapping, selection, gates, quality surface map, and handoffs are internally
consistent, and records retrospective inputs.

### Day 14 Project-Plan Closeout

| Item | Closeout status | Evidence owner |
| --- | --- | --- |
| 187.1 Review Intake | Complete. Review findings were converted into a 16-row Epic 17 gap ledger with owner files, claim risks, candidate sprints, validation, and non-goals. | `day2-review-intake-matrix.md` |
| 187.2 Residual Reconciliation | Complete. Epic 16 residuals were mapped to selected Epic 17 closures or retained long-horizon decisions. | `day3-residual-reconciliation.md` |
| 187.3 Closure Selection | Complete. Sprint 188-195 closure targets and broad non-goals were selected. | `day5-gap-ranking-and-feasibility.md`, `day6-closure-target-selection.md` |
| 187.4 Acceptance Gates | Complete. Package, Windows, comparison, performance, maintainability, adoption, and reliability gates were defined. | `day7-package-acceptance-gates.md` through `day11-adoption-documentation-gates.md` |
| 187.5 Quality Surface Map | Complete. Required checks, hosted evidence rules, local skip rules, and mandatory C gate policy were mapped by changed surface. | `day12-quality-surface-map.md` |
| 187.6 Sprint Handoff | Complete. Sprint 188-195 handoffs and PR/retrospective closeout notes are ready. | `day13-implementation-handoffs.md`, `day14-closeout-summary.md` |

### Day 14 Consistency Result

The Sprint 187 artifact chain is internally consistent:

- The gap ledger retains traceability to Codex review findings and Epic 16
  residuals.
- Selected closure targets map to Sprints 188 through 195 without adding
  broad, partial, or unsupported work.
- Acceptance gates point to owner files, validation commands, hosted evidence
  requirements, stop conditions, and non-goals.
- The Day 12 quality surface map preserves the full C gate for future `.c` and
  `.h` changes.
- The Day 13 handoff package gives each future sprint a ready starting point
  and keeps remaining open questions explicit.

### Day 14 Review-Ready Summary

Sprint 187 is ready for retrospective and PR preparation. The branch adds a
planning-only Sprint 187 package under
`docs/planning/EPIC_17/SPRINT_187/`, including the day-by-day plan, working
notes, daily artifacts, future-sprint handoffs, and final closeout summary.
It does not modify code, public headers, scripts, workflows, package material,
generated report output, or generated API output.

### Day 14 Retrospective Inputs

| Topic | Input |
| --- | --- |
| What worked | Daily artifacts made the baseline, review intake, residual mapping, target selection, acceptance gates, quality map, and implementation handoffs auditable. |
| What still needs attention | Future sprints must make concrete selections for license metadata, Windows report lane, comparison family, benchmark lane, maintainability cluster, and reliability owner. |
| Carry-forward risk | Claim wording can drift if support is promoted before the required local or hosted proof exists. |
| Validation expectation | Sprint 187 remains docs-only; future `.c` or `.h` changes must run `make format && make lint && make test`. |

### Day 14 Validation

Day 14 is planning documentation only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.
