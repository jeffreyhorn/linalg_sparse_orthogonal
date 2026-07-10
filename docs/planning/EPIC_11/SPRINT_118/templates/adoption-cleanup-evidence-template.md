# <Sprint/Day> Adoption Cleanup Evidence

## Scope

| Field | Value |
|---|---|
| Sprint/day | `<Sprint N Day N>` |
| Artifact owner | `<owner>` |
| Adoption surface | `<README / docs / examples / headers / install / benchmark docs>` |
| Touched surfaces | `<files>` |
| Explicitly out of scope | `<surfaces not touched>` |

## Baseline

| Baseline item | Current value |
|---|---|
| Current user-facing route | `<route>` |
| Current product truth references | `<Sprint 118 Day 8 or other evidence>` |
| Current candidate claims | `<candidate claims touched>` |
| Current non-claims | `<non-claims preserved before work>` |

## Proof Values

| Proof value | User behavior or claim protected | Evidence before change |
|---|---|---|
| `<value>` | `<behavior/claim>` | `<doc/test/artifact>` |

## User-Facing Route Changed

- Route:
- Intended audience:
- Before state:
- After state:
- What a new user should do first:

## Claim-Boundary Scan

| Claim or wording | Current truth source | Disposition |
|---|---|---|
| `<claim>` | `<artifact/doc/test>` | `<keep / fence / remove / future owner>` |

## Files And Links Changed

| File | Change | Link/path check |
|---|---|---|
| `<file>` | `<change>` | `<check>` |

## Example Or Cookbook Proof

| Example/workflow | Build/run proof | Notes |
|---|---|---|
| `<example>` | `<command/result>` | `<notes>` |

## Install, Package, And Platform Wording Impact

| Surface | Impact | Action |
|---|---|---|
| INSTALL | `<none / update / fence>` | `<action>` |
| Platform support table | `<none / update / fence>` | `<action>` |
| Package/ABI wording | `<none / update / fence>` | `<action>` |

## Benchmark And Performance Wording Impact

| Surface | Impact | Action |
|---|---|---|
| README benchmark text | `<none / update / fence>` | `<action>` |
| `benchmarks/README.md` | `<none / update / fence>` | `<action>` |
| Performance claims | `<none / update / fence>` | `<action>` |

## Validation Commands

| Command | Required because | Reviewed/supplemental/local | Result |
|---|---|---|---|
| `<command>` | `<reason>` | `<classification>` | `<pending/pass/fail>` |

Required trigger check:

- If docs only changed, run documentation hygiene.
- If examples, build metadata, install/package docs tied to proof, or code
  changed, run the relevant focused validation lane.
- If any `.c` or `.h` file changed, run `make format && make lint && make test`.

## Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | `<none / update / fence>` | `<action>` |
| INSTALL | `<none / update / fence>` | `<action>` |
| Solver-selection docs | `<none / update / fence>` | `<action>` |
| Examples/tutorial | `<none / update / fence>` | `<action>` |
| Benchmark/performance wording | `<none / update / fence>` | `<action>` |
| Platform/package wording | `<none / update / fence>` | `<action>` |

## Link And Path Checks

| Check | Result |
|---|---|
| Relative links | `<result>` |
| Example paths | `<result>` |
| Command paths | `<result>` |
| Artifact references | `<result>` |

## Non-Claims Preserved

- Compressed-first remains the product center; mutable shell remains
  compatibility unless future evidence changes this.
- Broad state-of-the-art replacement remains unclaimed.
- Ecosystem parity remains unclaimed.
- Portable performance superiority remains unclaimed.
- Dynamic ABI and package-manager support remain unclaimed unless proven.
- `<additional non-claim>`

## Residual Handoff

| Residual | Next owner | Evidence link |
|---|---|---|
| `<residual>` | `<sprint/day/future epic>` | `<artifact>` |

## Completion Check

| Criterion | Status |
|---|---|
| Current product truth references are recorded. | `<status>` |
| Claim-boundary scan is complete. | `<status>` |
| Files, links, and paths are checked. | `<status>` |
| Validation commands are recorded. | `<status>` |
| Package/platform/performance wording impact is recorded. | `<status>` |
| Non-claims are recorded. | `<status>` |
| Residual handoff is recorded. | `<status>` |
