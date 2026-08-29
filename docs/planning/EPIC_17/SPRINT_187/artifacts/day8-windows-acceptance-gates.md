# Sprint 187 Day 8: Windows Acceptance Gates

## Purpose

Define exact acceptance gates for Sprint 189 PowerShell validation ownership
and Sprint 190 Windows selected report freshness. These gates let Epic 17 close
the Windows validation gap without converting it into broad Windows parity.

## Current Windows Evidence Boundary

The current Windows workflow is `.github/workflows/windows-ci.yml`.

Accepted Windows evidence today:

- hosted `windows-2022` CMake configure and build with MSVC;
- hosted `ctest -N` count and full `ctest` execution;
- hosted CMake install/downstream validation;
- installed static `.lib`, headers, CMake package metadata, exact-version
  consumer behavior, mismatch-version rejection, and metadata-only
  `sparse.pc` inspection;
- absence of shared-library, loader, dynamic ABI, and static/shared selector
  metadata in the reviewed Windows package lane.

Current non-evidence:

- selected oracle freshness on Windows;
- selected comparison freshness on Windows;
- selected benchmark freshness on Windows;
- broad generated report freshness on Windows.

Sprint 182 formally deferred Windows report freshness in
`docs/planning/EPIC_16/SPRINT_182/artifacts/windows-report-freshness-deferral-decision.md`.
While that deferral remains active, `tests/corpus/manifests/selected_report_targets.tsv`
must not list `windows` in `workflow_platforms`, and Windows CI must not run or
upload selected report freshness artifacts.

## Sprint 189 Gate: PowerShell Validation Ownership

Sprint 189 closes only PowerShell validation ownership. It does not promote
Windows report freshness.

| Requirement | Acceptance criteria | Failure state |
| --- | --- | --- |
| PowerShell surface inventory | All PowerShell snippets and workflow-owned report-adjacent scripts in `.github/workflows/windows-ci.yml`, docs, and report workflow comments are listed with owner, purpose, and claim boundary. | Unknown PowerShell material remains outside validation ownership. |
| Validation command | A maintained command or script parses or dry-runs the selected PowerShell material when `pwsh` is available. | The sprint cannot claim local PowerShell validation ownership. |
| Hosted ownership | Hosted Windows CI runs the selected PowerShell validation lane, or records why local-only validation is intentionally unavailable. | The validation path remains an environment residual. |
| Local skip semantics | If local `pwsh` is absent, the command reports an explicit skip/unavailable state and does not count that state as pass evidence. | Local absence is hidden or treated as successful proof. |
| Report freshness boundary | New PowerShell validation does not run selected report generators, upload selected report artifacts, or add `windows` to selected manifest platforms. | Sprint 189 accidentally promotes report freshness and must be rejected or split. |
| Docs alignment | README, INSTALL, maintainer guide, report-index docs, and workflow comments explain PowerShell validation ownership and retained non-claims. | User-facing wording implies broader Windows support than validation proves. |

Accepted Sprint 189 outcomes:

1. PowerShell validation ownership passes in hosted Windows and, when possible,
   locally.
2. Local `pwsh` remains unavailable, but hosted validation owns the selected
   PowerShell surface and local absence is documented as skip/unavailable.

Rejected Sprint 189 outcomes:

- treating missing local `pwsh` as a passing validation signal;
- promoting selected Windows report freshness;
- weakening the existing CMake-first/static-first Windows support boundary;
- adding broad Windows Makefile, `pkg-config`, package-manager, shared-library,
  dynamic ABI, or runtime-loader claims.

## Sprint 190 Gate: Windows Report Freshness Decision

Sprint 190 must finish in exactly one of two accepted states:

1. Promote one Windows-safe selected report freshness lane with evidence.
2. Renew the formal deferral with stronger blockers, guards, and revisit
   criteria.

Partial promotion is not accepted.

## Sprint 190 Promotion Path

Promotion requires all gates below.

| Requirement | Acceptance criteria |
| --- | --- |
| One selected lane | Exactly one report freshness lane is selected: oracle, comparison, or benchmark. Broad report freshness remains out of scope. |
| Windows-safe generator | The command runs under hosted Windows without relying on Makefile, Bash, Unix archive/link assumptions, extensionless executables, Unix `-lm`, or POSIX path-only behavior. |
| CMake/MSVC probe path | Any generated probe builds and links through CMake/MSVC against the reviewed static package shape. |
| Executable handling | Temporary probe execution is `.exe` aware and records failures clearly. |
| Manifest metadata | The selected row in `tests/corpus/manifests/selected_report_targets.tsv` names exact `workflow_file`, `workflow_job`, `workflow_artifact`, `workflow_platforms=windows` or a platform list including `windows`, support tier, claim scope, and non-claims. |
| Artifact scope | The workflow uploads exactly the selected Windows artifacts and no broad report directory. |
| Artifact failure mode | Upload uses `if-no-files-found: error` or an equivalent hard failure. |
| Freshness check | A selected freshness command verifies required files, expected rows, expected row IDs, current commit freshness, and selected artifact paths. |
| Guard update | Existing tests that currently forbid Windows selected freshness are updated to allow only the exact selected lane and reject any broader lane. |
| Documentation | README, INSTALL, maintainer guide, report-index schema/docs, and workflow comments describe the promoted Windows lane and retained non-claims. |

The selected lane must still preserve these non-claims:

- no broad Windows report freshness;
- no Makefile parity;
- no Windows `pkg-config` execution parity;
- no package-manager support;
- no shared-library or dynamic ABI support;
- no portable performance, performance superiority, or state-of-the-art claim.

## Sprint 190 Renewed Deferral Path

If promotion is rejected, Sprint 190 still closes the decision by renewing the
deferral. The renewed deferral must include:

| Requirement | Acceptance criteria |
| --- | --- |
| Decision record | A refreshed decision artifact states that Windows report freshness remains formally deferred. |
| Blocker list | The blocker list names the exact remaining command, runtime, manifest, artifact, and claim blockers. |
| Guard evidence | Tests or scripts prove Windows CI does not run selected report freshness commands or upload selected report artifacts. |
| Manifest guard | Selected target manifest rows do not list `windows` while deferral is active. |
| Revisit criteria | Future promotion gates list the exact generator, CMake/MSVC probe, upload, freshness, and documentation requirements. |
| Docs alignment | Public and maintainer docs keep Windows report freshness unclaimed and point to the renewed deferral. |

Renewed deferral is acceptable only if it is explicit, guarded, and reviewable.
It is not acceptable to leave stale ambiguity.

## Owner Surfaces

| Surface | Sprint 189 role | Sprint 190 role |
| --- | --- | --- |
| `.github/workflows/windows-ci.yml` | Hosted PowerShell validation owner, while keeping CMake-first/static-first comments intact. | Promotion or deferral owner for exact Windows report freshness commands and artifacts. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Must keep `windows` absent during validation-only work. | Positive manifest authority for any promoted Windows selected lane. |
| `scripts/normalize_report_index.py` | Existing selected-target parser/freshness consumer; no promotion required in Sprint 189. | Freshness validation owner for required files, expected rows, expected row IDs, stale commit checks, and artifact diagnostics. |
| `tests/test_selected_report_targets_manifest.py` | Deferral guard that rejects `windows` while Sprint 182 deferral is active. | Must be updated to either retain deferral or allow exactly one promoted Windows row. |
| `tests/test_selected_comparison_workflow.py` | Workflow guard that blocks selected freshness commands/uploads in Windows CI while deferral is active. | Must become an allowlist for the selected Windows lane or remain a strengthened deferral guard. |
| `tests/corpus/schemas/report_index_fields.md` | Documents selected target manifest semantics and active Windows deferral split. | Documents any promoted Windows manifest fields and non-claim boundaries. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Explain PowerShell validation ownership without freshness promotion. | Explain the selected promotion or renewed deferral outcome. |

## Artifact Upload Scope

Any promoted Windows report freshness lane must define:

- workflow file and job;
- exact artifact name;
- exact required files;
- exact artifact path pattern;
- expected row count;
- expected row IDs;
- generator command;
- support tier;
- claim scope;
- non-claims;
- upload failure behavior;
- retention of ignored local/generated output policy.

Broad upload patterns such as a whole `build/` tree or every report family are
not acceptable for Sprint 190.

## Required Validation Commands

Minimum local/source validation for Sprint 189 and 190:

```sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
```

When selected report metadata or generated report checks change:

```sh
python3 scripts/normalize_report_index.py --check-freshness
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
```

When workflow behavior changes:

```sh
gh workflow run windows-ci.yml
gh run watch <run-id>
gh run view <run-id> --log-failed
```

Hosted Windows evidence is required before any Windows freshness promotion can
be treated as complete. Local-only checks are sufficient only for renewed
deferral or validation-script syntax/manifest guard work.

`make format && make lint && make test` remains required whenever Sprint 189 or
Sprint 190 modifies `.c` or `.h` files.

## Windows Non-Claim Register

Sprints 189 and 190 must retain these non-claims unless a later epic selects
separate evidence:

- Windows Makefile parity.
- Windows `pkg-config` command execution parity.
- Bash/POSIX report generation parity.
- Package-manager support on Windows.
- Shared-library package support.
- Dynamic ABI, DLL/import-library, SONAME, install-name, RPATH, or runtime
  loader behavior.
- Broad Windows platform parity.
- Broad generated report freshness.
- Cross-platform report freshness for every selected family.
- Portable performance, performance superiority, or state-of-the-art status
  from any Windows report artifact.

## Completion Gate

Sprint 189 is complete when PowerShell validation ownership is explicit,
validated, documented, and separated from report freshness.

Sprint 190 is complete when either:

1. one exact Windows-safe report freshness lane is promoted with hosted proof,
   manifest metadata, artifact upload, freshness checks, guards, and docs; or
2. the formal deferral is renewed with blocker evidence, guards, docs, and
   revisit criteria.

Any implicit Windows freshness claim, broad artifact upload, stale-output
acceptance, missing hosted proof for promotion, or weakened non-claim boundary
blocks completion.

## Validation

Day 8 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
