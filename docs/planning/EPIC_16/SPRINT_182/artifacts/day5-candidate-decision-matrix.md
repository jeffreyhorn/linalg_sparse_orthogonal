# Sprint 182 Day 5: Candidate Decision Matrix

**Sprint:** 182 - Windows Report Freshness Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_182/`
**Status:** Complete

## Purpose

Day 5 compares Windows report freshness promotion and deferral options using
the evidence from Days 1-4. The outcome is a single Day 6 decision target, so
Sprint 182 stops carrying multiple parallel implementation paths.

## Inputs

| Input | Role |
| --- | --- |
| `day1-windows-freshness-scope-intake.md` | Shared decision criteria and inherited Sprint 181 boundaries. |
| `day2-windows-workflow-and-toolchain-audit.md` | Windows workflow, shell, toolchain, and fail-closed guard baseline. |
| `day3-report-command-compatibility-audit.md` | Selected command runtime and dependency blockers. |
| `day4-artifact-and-data-semantics-audit.md` | Artifact, newline, path, normalized-index, and upload-scope findings. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected target authority for commands, artifacts, workflow metadata, platforms, claims, and non-claims. |
| `tests/test_selected_comparison_workflow.py` | Current workflow guard for selected report freshness lanes and Windows non-claims. |

## Decision Criteria

| Criterion | Meaning for Day 5 |
| --- | --- |
| Windows CI feasibility | Candidate can run under the reviewed `windows-2022` PowerShell/CMake/MSVC lane without unreviewed setup. |
| Shell portability | Candidate avoids Makefile parity, Bash, POSIX utilities, and implicit Unix glob behavior unless newly reviewed. |
| Artifact stability | Candidate produces exact selected artifacts with stable row counts, LF-friendly TSV/CSV behavior, and fail-closed upload scope. |
| Dependency requirements | Candidate depends only on reviewed tools or source-controlled helpers and does not add package-manager requirements. |
| Runtime cost | Candidate is bounded enough for hosted Windows CI without performance-adjacent flake risk. |
| Maintenance cost | Candidate does not require broad platform-specific build/link/report maintenance. |
| User value | Candidate meaningfully clarifies Windows freshness without creating broad parity expectations. |
| Claim risk | Candidate preserves non-claims for broad report freshness, package-manager support, shared-library ABI, platform parity, and performance superiority. |
| Guardability | Candidate can be enforced with manifest/workflow guards that fail clearly on drift. |

## Candidate Matrix

| Candidate | Windows CI feasibility | Shell portability | Artifact stability | Dependency requirements | Runtime cost | Maintenance cost | User value | Claim risk | Guardability | Day 5 position |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Selected comparison direct Python promotion | Medium after refactor | Medium | High | Medium | Medium | Medium | Medium to high | Medium | Medium to high | Strongest future promotion candidate, but not selected for immediate implementation because Windows-safe probe/link behavior is missing. |
| Selected oracle promotion | Low to medium | Low to medium | Medium | Medium | Medium | Medium to high | Medium | Medium to high | Medium | Rejected as Day 6 target because it combines Unix probe assumptions with a broader 52-row selected surface and selected glob upload handling. |
| Selected benchmark promotion | Low | Low | Medium | Low to medium | Low to medium | High | Low to medium | High | Medium | Rejected as Day 6 target because Bash report generation, benchmark runtime, and performance-adjacent claims make it the riskiest promotion path. |
| Formal Windows report freshness deferral | High | High | High | High | High | Low | Medium | Low | High | Selected Day 6 target because it closes the product decision without widening Windows claims and preserves exact blockers for future promotion. |

## Ranking

| Rank | Candidate | Why |
| --- | --- | --- |
| 1 | Formal Windows report freshness deferral | Only option that is immediately feasible under the current reviewed Windows lane while preserving guard coverage and claim accuracy. |
| 2 | Selected comparison direct Python promotion | Best future promotion path because artifacts are exact and dependencies are source-controlled, but it requires Windows-safe CMake/MSVC probe/link work first. |
| 3 | Selected oracle promotion | Technically adjacent to comparison, but the selected scope is larger and upload semantics are harder because oracle uses a selected glob and 52-row contract. |
| 4 | Selected benchmark promotion | Highest implementation and claim risk because the generator is Bash-based and the output is performance-adjacent. |

## Selected Day 6 Decision Target

Day 5 selects **formal Windows report freshness deferral** for the Day 6
decision record.

The decision record should state that Windows continues to have reviewed
CMake build/test and static install/downstream proof, but generated report
freshness remains unpromoted. The blocker is not TSV data format. The blocker
is that current selected freshness commands depend on unreviewed Makefile
wrappers, Unix compiler/linker probe behavior, Bash report generation, or
performance-adjacent benchmark runtime assumptions.

## Smallest Future Promotion Candidate

If a later sprint promotes Windows report freshness, the smallest claim-safe
candidate should be one selected comparison target through a direct Python
command after these prerequisites exist:

- Windows-safe project probe build/link path using the reviewed CMake/MSVC
  toolchain;
- exact Python executable proof on the Windows runner;
- `.lib` and executable suffix handling;
- Windows-aware remediation text in freshness diagnostics;
- manifest workflow metadata for the Windows lane;
- exact selected artifact upload scope and guard allowlist.

## Rejected Option Rationale

| Option | Rejection rationale |
| --- | --- |
| Promote selected comparison now | Good artifact shape, but current probe build uses `cc`, Unix `.a`, `-lm`, extensionless temp executables, and fallback `make`. Promoting now would imply unreviewed Windows report freshness. |
| Promote selected oracle now | Carries the comparison probe/link blockers and also has broader artifact scope, selected glob upload handling, and 52 expected rows. |
| Promote selected benchmark now | Requires Bash report generation, Unix metadata commands, benchmark executable path assumptions, benchmark runtime proof, and careful performance non-claim wording. |
| Keep deciding between all paths | Rejected because Sprint 182 must converge before implementation. Formal deferral is the only evidence-backed path that can close the product decision now. |

## Exact Deferral Blockers

Formal deferral should name these blockers:

- no reviewed Windows Makefile parity for current selected freshness wrappers;
- no Windows-safe CMake/MSVC project probe path for comparison or oracle
  generators;
- no reviewed Windows `.lib`/MSVC link model for generated comparison/oracle
  probes;
- no Windows temp executable suffix handling in selected probe generators;
- no Windows-native canonical benchmark report generator;
- no Windows selected workflow metadata in
  `tests/corpus/manifests/selected_report_targets.tsv`;
- existing docs and guards correctly preserve Windows report freshness as a
  non-claim.

## Guard Requirements For The Selected Path

The deferral implementation should preserve or strengthen these guard
requirements:

- `.github/workflows/windows-ci.yml` must not run selected oracle,
  comparison, or benchmark freshness commands;
- Windows must not upload selected freshness artifact names used by Linux or
  macOS;
- Windows docs must not claim generated report freshness;
- selected target manifest rows must not list `windows` as a workflow
  platform unless a future promotion adds a reviewed Windows lane;
- any future exception must be manifest-backed and exact rather than broad.

## Day 6 Open Questions

| Question | Required Day 6 answer |
| --- | --- |
| Deferral artifact location | Choose where the formal Windows report freshness deferral record lives. |
| Manifest representation | Decide whether explicit deferral is represented in `selected_report_targets.tsv`, a separate artifact, docs only, or a guard-owned invariant. |
| Guard wording | Define exact guard text or checks that prove Windows remains unselected. |
| Documentation claim | Define the exact allowed Windows report freshness wording for README, INSTALL, maintainer guide, and workflow comments. |
| Future promotion gate | Define the minimum evidence required before selected comparison can be reconsidered. |

## Day 5 Decision

Formal Windows report freshness deferral is the recommended Day 6 decision
target. Selected comparison direct Python invocation remains the best future
promotion direction, but only after Windows-safe CMake/MSVC probe support and
manifest-backed workflow metadata exist.

## Validation

Day 5 changed planning artifacts only. Validation:

```sh
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
git diff --check
```

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 182 no longer carries multiple parallel implementation paths. | Complete | Formal Windows report freshness deferral is the selected Day 6 target. |
| Chosen path is justified by evidence rather than preference. | Complete | Candidate matrix, ranking, and exact deferral blockers summarize Days 1-4 evidence. |
| Rejected paths have concrete blockers or risk reasons. | Complete | Rejected option rationale and exact deferral blocker sections. |
