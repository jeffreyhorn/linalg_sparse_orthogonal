# Sprint 182 Day 6: Decision Record Design

**Sprint:** 182 - Windows Report Freshness Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_182/`
**Status:** Complete

## Purpose

Day 6 designs the exact contract for implementing the Day 5 decision:
formal Windows report freshness deferral.

This is a design artifact for Days 7-13. It records the decision-record
structure, blocker taxonomy, manifest approach, workflow guard plan, docs
plan, and revisit gates before implementation changes begin.

## Decision Record Structure

The formal deferral record should include these sections:

| Section | Required content |
| --- | --- |
| Status | Accepted for Sprint 182 implementation as formal deferral. |
| Decision | Windows generated report freshness remains unpromoted and explicitly deferred. |
| Context | Windows currently proves CMake build/test and static install/downstream validation, not generated report freshness. |
| Accepted evidence | Day 1-5 audit evidence showing data-format feasibility but runtime/workflow blockers. |
| Supported claims | Narrow Windows CMake/MSVC build/test and static install/downstream claims already proven. |
| Unsupported claims | Windows report freshness, Makefile parity, Bash report generation, package-manager support, shared-library ABI, broad platform parity, and performance superiority. |
| Rejected alternatives | Selected comparison, oracle, and benchmark promotion in Sprint 182. |
| Blocker taxonomy | Exact workflow, compiler/linker, artifact, manifest, documentation, and guard blockers. |
| Implementation boundaries | What Days 7-13 may change without implying promotion. |
| Guard contract | Required fail-closed behavior for Windows workflow and selected target metadata. |
| Revisit criteria | Minimum evidence before a future sprint can reconsider selected Windows freshness. |
| Validation gates | Tests and checks required before the sprint can close. |
| Pass/fail contract | Clear definition of what makes the deferral complete or invalid. |

## Decision

Sprint 182 should implement **formal Windows report freshness deferral**.

The selected decision means:

- Windows generated report freshness remains a non-claim.
- Windows continues to prove the reviewed CMake/MSVC build/test path.
- Windows continues to prove static-first CMake install/downstream behavior.
- No selected report freshness command or artifact upload is added to the
  Windows workflow.
- The product boundary is explicit, guarded, and documented.

This decision does not block future Windows report freshness. It requires a
future promotion to satisfy the revisit gates below before any Windows
freshness claim is added.

## Supported Windows Claims

The deferral record may preserve these existing claims:

- reviewed Windows CMake configure and build on the pinned hosted runner;
- reviewed Windows `ctest -N` registration count and full `ctest` execution;
- reviewed Windows static-first CMake install/downstream validation;
- installed static `.lib`, headers, CMake package metadata, exact-version
  behavior, mismatch-version rejection, and metadata-only `sparse.pc`
  inspection;
- absence of unsupported shared-library/imported runtime-loader metadata in
  the Windows static-first package lane.

These claims are not report freshness claims.

## Unsupported Scope

The formal deferral must continue to reject wording or artifacts that imply:

- selected oracle freshness on Windows;
- selected comparison freshness on Windows;
- selected benchmark freshness on Windows;
- broad generated report freshness on Windows;
- Windows Makefile parity;
- Windows Bash/POSIX shell report generation support;
- Windows `pkg-config` execution parity;
- Windows package-manager support;
- Windows shared-library, dynamic ABI, DLL/import-library, or runtime-loader
  support;
- broad Windows platform parity;
- portable performance or state-of-the-art claims from any Windows report.

## Blocker Taxonomy

| Blocker class | Exact blocker |
| --- | --- |
| Workflow shell | Existing Windows workflow is `pwsh`/CMake scoped; selected freshness wrappers are Makefile based. |
| Compiler/linker | Comparison and oracle probes assume `cc`, Unix static archive `.a`, `-lm`, and Unix-style compiler arguments. |
| Executable suffix | Probe generation currently uses extensionless temporary executables rather than Windows `.exe`-aware invocation. |
| Library artifact | Windows reviewed package evidence uses MSVC/CMake `.lib` outputs, not the Unix `build/libsparse_lu_ortho.a` path expected by selected probes. |
| Benchmark generator | Canonical benchmark report generation depends on Bash and POSIX metadata commands. |
| Artifact ownership | No Windows selected workflow artifact name or exact Windows upload scope exists in the selected target manifest. |
| Manifest status | No selected target row currently lists `windows` in `workflow_platforms`. |
| Documentation | Public docs correctly preserve Windows report freshness as a non-claim and need formal deferral wording before the sprint closes. |
| Guard behavior | Current guards reject Windows selected freshness strings; implementation must preserve or strengthen this boundary. |

## Manifest Change Plan

The selected target manifest should remain the authority for positive selected
freshness targets. Day 6 chooses this manifest policy:

- keep `windows` absent from `workflow_platforms` for all existing selected
  freshness rows;
- do not add a fake selected Windows target row to represent deferral;
- if implementation needs a source-controlled deferral marker, prefer a
  separate Sprint 182 deferral artifact plus guard/docs references;
- only add manifest fields or rows if they make support-tier status clearer
  without changing the manifest from selected-target authority into a general
  deferral registry;
- any future Windows promotion must add exact `workflow_file`, `workflow_job`,
  `workflow_artifact`, `workflow_platforms`, support-tier, claim-scope, and
  non-claim metadata.

## Workflow Guard Change Plan

The guard contract for formal deferral is:

- `.github/workflows/windows-ci.yml` must not contain selected freshness
  command names:
  - `report-index-oracle-freshness`
  - `report-index-comparison-freshness`
  - `bench-canonical-report-freshness`
  - `check_bench_canonical_freshness.py`
- `.github/workflows/windows-ci.yml` must not contain selected freshness
  artifact names:
  - `sprint159-oracle-freshness`
  - `sprint175-linux-selected-comparison-freshness`
  - `sprint175-macos-selected-comparison-freshness`
  - `sprint168-selected-performance-freshness`
- selected target manifest rows must not list `windows` as a selected
  freshness platform;
- guard diagnostics should describe this as formal deferral rather than an
  accidental absence;
- if a formal deferral artifact is added, a guard or docs check should verify
  that Windows deferral wording and workflow behavior stay aligned.

## Documentation Change Plan

Days 12-13 should update or confirm wording in:

| Surface | Required wording behavior |
| --- | --- |
| README | State that Windows report freshness is formally deferred while Windows CMake/install validation remains reviewed. |
| INSTALL | Preserve Windows static package/install scope and avoid package-manager or report freshness implications. |
| Maintainer guide | Explain the Sprint 182 deferral, blockers, guard behavior, and future promotion gate. |
| Workflow comments | Keep Windows workflow comments CMake/install scoped and name generated report freshness as deferred/non-claim. |
| Report-index language | Keep selected report freshness platform support limited to current Linux/macOS rows and avoid broad Windows freshness claims. |

Allowed public wording:

- "Windows report freshness is formally deferred."
- "Windows CI proves reviewed CMake build/test and static install/downstream
  behavior, not generated report freshness."
- "Selected Linux/macOS report freshness lanes do not imply Windows report
  freshness."

Disallowed wording:

- "Windows report freshness is supported."
- "Selected report freshness is cross-platform" unless Windows is excluded
  explicitly.
- "Windows has report parity with Linux/macOS."
- Any wording that turns benchmark metadata into a Windows performance claim.

## Future Promotion Gates

A future sprint may revisit selected Windows report freshness only after all
of these gates exist:

| Gate | Required evidence |
| --- | --- |
| Windows-safe command | Direct command that runs under reviewed Windows shell without Makefile or Bash assumptions. |
| CMake/MSVC probe support | Generated project probes build/link through CMake/MSVC or another reviewed Windows compiler path. |
| Library model | Probe path uses the reviewed Windows `.lib` artifact and avoids Unix `-lm` assumptions. |
| Executable handling | Temporary probe executables are `.exe`-aware and invoked explicitly. |
| Python proof | The exact Python executable used by the lane is proven on hosted Windows. |
| Artifact scope | Workflow uploads exact selected artifacts with `if-no-files-found: error`. |
| Manifest metadata | Selected target manifest records exact Windows workflow file, job, artifact, platform, support tier, claim scope, and non-claims. |
| Guard allowlist | Workflow guard changes from blanket rejection to manifest-backed allowlist for exactly one selected Windows path. |
| Documentation | README, INSTALL, maintainer guide, workflow comments, and report-index language describe only the proven Windows freshness scope. |
| Validation | Manifest tests, workflow guard tests, report checks, and Windows syntax review pass. |

## Implementation Boundaries For Days 7-13

| Area | Boundary |
| --- | --- |
| Decision artifact | Add the formal deferral record; do not add a Windows freshness lane. |
| Workflow | Keep existing Windows jobs CMake/install scoped. Workflow comment edits are allowed if they clarify deferral. |
| Manifest | Preserve selected positive-target authority and no Windows selected platform unless an explicit schema/docs change is justified. |
| Guards | Strengthen fail-closed checks for Windows report freshness absence and future promotion prerequisites. |
| Docs | Add formal deferral wording without weakening existing Windows CMake/install claims. |
| Code | Avoid report generator refactors unless needed for guard/docs validation; Sprint 182 selected deferral does not require a Windows probe implementation. |

## Validation Gates

The implemented deferral should pass:

```sh
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
git diff --check
```

If later implementation changes C or header files, the required quality gate
is:

```sh
make format && make lint && make test
```

## Pass/Fail Contract

The formal deferral passes when:

- exactly one Sprint 182 product path is implemented: Windows report
  freshness deferral;
- Windows workflow remains free of selected report freshness commands and
  selected artifact names;
- selected target manifest rows do not claim Windows selected freshness;
- docs state Windows report freshness is deferred while preserving reviewed
  CMake/install claims;
- future promotion blockers and gates are explicit;
- guard and manifest validation pass.

The formal deferral fails if:

- Windows workflow gains a selected report freshness command without
  manifest-backed promotion;
- selected target manifest lists `windows` without exact workflow metadata and
  evidence;
- docs imply Windows report freshness, broad platform parity, Makefile parity,
  package-manager support, shared-library ABI support, or performance
  superiority;
- guard checks allow broad selected report uploads or accidental Windows
  selected freshness strings.

## Day 6 Deliverables

- formal deferral decision-record design
- manifest change plan
- workflow guard change plan
- documentation change plan
- future promotion gates
- `docs/planning/EPIC_16/SPRINT_182/artifacts/day6-decision-record-design.md`

## Validation

Day 6 changed planning artifacts only. Validation:

```sh
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
git diff --check
```

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implementation work has one precise contract to follow. | Complete | Decision, implementation boundaries, and pass/fail contract sections. |
| Support-tier and freshness-policy wording are known before code changes. | Complete | Supported Windows claims, unsupported scope, manifest change plan, and docs plan. |
| Claim boundaries are explicit before workflow/docs edits. | Complete | Unsupported scope, guard plan, and future promotion gates. |
