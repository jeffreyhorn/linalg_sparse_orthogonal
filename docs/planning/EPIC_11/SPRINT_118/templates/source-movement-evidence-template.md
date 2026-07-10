# <Sprint/Day> Source Movement Evidence

## Scope

| Field | Value |
|---|---|
| Sprint/day | `<Sprint N Day N>` |
| Artifact owner | `<owner>` |
| Work type | `<source movement / private-owner extraction / internal-header reshaping / giant-test split>` |
| Touched surfaces | `<files, build metadata, tests, docs>` |
| Explicitly out of scope | `<surfaces not touched>` |

## Baseline

| Baseline item | Current value |
|---|---|
| Starting files | `<file list>` |
| Starting line counts | `<line counts>` |
| Starting CTest count | `<count>` |
| Current product truth references | `<Sprint 118 Day 8 or other evidence>` |
| Current non-claims | `<non-claims preserved before work>` |

## Starting Owner Metrics

| Owner | Lines | Function/test proxy count | Responsibility |
|---|---:|---:|---|
| `<file>` | `<count>` | `<count>` | `<current responsibility>` |

## Proof Values

| Proof value | Protected behavior or invariant | Evidence before change |
|---|---|---|
| `<value>` | `<behavior>` | `<test/artifact/doc>` |

## Behavior Boundary

- Boundary being moved:
- Boundary not being moved:
- Consumer paths affected:
- Unsupported or expected-failure behavior that must remain visible:

## Old/New File Plan

| Current file | Proposed file | Ownership after change | Notes |
|---|---|---|---|
| `<old>` | `<new>` | `<owner>` | `<notes>` |

## Internal Header And Private API Contract

| Contract item | Decision |
|---|---|
| Internal headers added or changed | `<headers>` |
| Private functions moved | `<functions>` |
| Public API impact | `<none / described change>` |
| ABI/package impact | `<none / described change>` |

## Source-List, Makefile, And CMake Impact

| Surface | Expected update | Validation |
|---|---|---|
| Makefile/source list | `<update>` | `<command>` |
| CMake | `<update>` | `<command>` |
| CTest membership | `<same / changed with reason>` | `<command>` |

## Change Plan

1. `<step>`
2. `<step>`
3. `<step>`

## Focused Consumer Proof

| Consumer path | Focused command | Expected result |
|---|---|---|
| `<path>` | `<command>` | `<result>` |

## Giant-Test Split Map

Use this section when the work splits a test owner.

| Current proof block | New proof owner | Fixture/helper ownership | Failure-localization improvement |
|---|---|---|---|
| `<block>` | `<file/function>` | `<owner>` | `<improvement>` |

## Validation Commands

| Command | Required because | Reviewed/supplemental/local | Result |
|---|---|---|---|
| `<command>` | `<reason>` | `<classification>` | `<pending/pass/fail>` |

Required trigger check:

- If any `.c` or `.h` file changed, run `make format && make lint && make test`.
- If Makefile, CMake, workflow, package, benchmark, or script surfaces changed,
  run the relevant focused validation lane and record the classification.
- If docs only changed, run documentation hygiene.

## Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | `<none / update / fence>` | `<action>` |
| INSTALL | `<none / update / fence>` | `<action>` |
| Solver/docs/examples | `<none / update / fence>` | `<action>` |
| Benchmark/performance wording | `<none / update / fence>` | `<action>` |

## Rollback Or Defer Plan

- Rollback path:
- Defer condition:
- Partial-move handling:

## Non-Claims Preserved

- `<non-claim>`
- `<non-claim>`

## Residual Handoff

| Residual | Next owner | Evidence link |
|---|---|---|
| `<residual>` | `<sprint/day/future epic>` | `<artifact>` |

## Completion Check

| Criterion | Status |
|---|---|
| Behavior boundary is explicit. | `<status>` |
| Old/new file plan is recorded. | `<status>` |
| Source-list and CMake impact are recorded. | `<status>` |
| Focused consumer proof is recorded. | `<status>` |
| Validation commands are recorded. | `<status>` |
| Drift and non-claims are recorded. | `<status>` |
| Residual handoff is recorded. | `<status>` |
