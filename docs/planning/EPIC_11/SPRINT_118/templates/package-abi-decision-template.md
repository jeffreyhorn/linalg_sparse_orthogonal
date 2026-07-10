# <Sprint/Day> Package And ABI Decision Evidence

## Scope

| Field | Value |
|---|---|
| Sprint/day | `<Sprint N Day N>` |
| Artifact owner | `<owner>` |
| Decision surface | `<static-first / shared library / ABI / package manager / platform install>` |
| Touched surfaces | `<files, workflows, docs, scripts>` |
| Explicitly out of scope | `<surfaces not touched>` |

## Baseline

| Baseline item | Current value |
|---|---|
| Current package contract | `<static-first / other>` |
| Current install proof | `<commands/artifacts>` |
| Current platform tier references | `<Day 4 or workflow references>` |
| Current product truth references | `<Sprint 118 Day 8 or other evidence>` |
| Current non-claims | `<dynamic ABI/package-manager/platform parity non-claims>` |

## Proof Values

| Proof value | Package, ABI, or platform behavior protected | Evidence before change |
|---|---|---|
| `<value>` | `<behavior>` | `<test/artifact/doc>` |

## Decision

| Decision question | Answer | Evidence |
|---|---|---|
| Preserve static-first only? | `<yes/no>` | `<evidence>` |
| Add shared-library support? | `<yes/no>` | `<evidence>` |
| Define dynamic ABI policy? | `<yes/no>` | `<evidence>` |
| Add package-manager support? | `<yes/no>` | `<evidence>` |
| Change platform tier? | `<yes/no>` | `<evidence>` |

## Static-First Contract

| Contract item | Expected state |
|---|---|
| Static archive install | `<state>` |
| Installed headers | `<state>` |
| `pkg-config` metadata | `<state>` |
| CMake package metadata | `<state>` |
| Explicitly absent artifacts | `<state>` |

## Shared-Library And ABI Contract

Complete this section even when the decision is to defer shared-library or ABI
support.

| Contract item | Decision |
|---|---|
| Shared library artifact | `<supported / absent / deferred>` |
| Symbol/version policy | `<policy or deferred>` |
| Loader/runtime proof | `<proof or deferred>` |
| ABI compatibility test | `<proof or deferred>` |
| Public claim wording | `<wording>` |

## Installed Artifact Expectations

| Artifact | Expected present? | Validation |
|---|---|---|
| `<artifact>` | `<yes/no>` | `<command>` |

## Package Metadata And Version Behavior

| Metadata surface | Expected behavior | Validation |
|---|---|---|
| `sparse.pc` | `<behavior>` | `<command>` |
| CMake config | `<behavior>` | `<command>` |
| CMake version | `<behavior>` | `<command>` |
| `VERSION` | `<behavior>` | `<command>` |

## Downstream Consumer Proof

| Consumer route | Compile | Link | Run | Notes |
|---|---|---|---|---|
| `pkg-config` | `<status>` | `<status>` | `<status>` | `<notes>` |
| CMake `find_package(Sparse)` | `<status>` | `<status>` | `<status>` | `<notes>` |

## Platform Tier Impact

| Platform | Reviewed lane | Supplemental lane | Staged exclusions | Impact |
|---|---|---|---|---|
| Linux | `<lane>` | `<lane>` | `<exclusions>` | `<impact>` |
| macOS | `<lane>` | `<lane>` | `<exclusions>` | `<impact>` |
| Windows | `<lane>` | `<lane>` | `<exclusions>` | `<impact>` |

## Expected Test Counts And Staged Exclusions

| Surface | Expected count | Observed count | Notes |
|---|---:|---:|---|
| Makefile tests | `<count>` | `<count>` | `<notes>` |
| CMake CTest registrations | `<count>` | `<count>` | `<notes>` |
| Windows CTest subset | `<count>` | `<count>` | `<notes>` |

## Validation Commands

| Command | Required because | Reviewed/supplemental/local | Result |
|---|---|---|---|
| `<command>` | `<reason>` | `<classification>` | `<pending/pass/fail>` |

Required trigger check:

- If any `.c` or `.h` file changed, run `make format && make lint && make test`.
- If install/export/package metadata changes, run relevant package/install
  consumer proof.
- If platform workflow scope or expected counts change, update support wording
  and staged exclusions.

## Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | `<none / update / fence>` | `<action>` |
| INSTALL | `<none / update / fence>` | `<action>` |
| Maintainer guide | `<none / update / fence>` | `<action>` |
| Workflows | `<none / update / fence>` | `<action>` |

## Package-Manager Disposition

| Manager | Status | Proof | Public claim |
|---|---|---|---|
| `<manager>` | `<unsupported / deferred / supported>` | `<proof>` | `<claim wording>` |

## Non-Claims Preserved

- Dynamic ABI support remains unclaimed unless explicitly implemented and
  validated.
- Package-manager support remains unclaimed unless real recipes and consumer
  proof exist.
- Symmetric platform parity remains unclaimed unless reviewed lanes prove it.
- `<additional non-claim>`

## Residual Handoff

| Residual | Next owner | Evidence link |
|---|---|---|
| `<residual>` | `<sprint/day/future epic>` | `<artifact>` |

## Completion Check

| Criterion | Status |
|---|---|
| Decision is explicit. | `<status>` |
| Static/shared/ABI contract is recorded. | `<status>` |
| Installed artifact expectations are recorded. | `<status>` |
| Downstream consumer proof is recorded. | `<status>` |
| Platform tier impact is recorded. | `<status>` |
| Validation commands are recorded. | `<status>` |
| Drift and non-claims are recorded. | `<status>` |
| Residual handoff is recorded. | `<status>` |
