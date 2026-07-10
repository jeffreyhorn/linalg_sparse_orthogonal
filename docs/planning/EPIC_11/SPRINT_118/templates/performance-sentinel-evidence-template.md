# <Sprint/Day> Performance Sentinel Evidence

## Scope

| Field | Value |
|---|---|
| Sprint/day | `<Sprint N Day N>` |
| Artifact owner | `<owner>` |
| Benchmark/report/sentinel surface | `<surface>` |
| Touched surfaces | `<files, scripts, docs, reports>` |
| Explicitly out of scope | `<surfaces not touched>` |

## Baseline

| Baseline item | Current value |
|---|---|
| Existing command | `<command>` |
| Existing report path | `<path>` |
| Existing threshold state | `<none / threshold>` |
| Current product truth references | `<Sprint 118 Day 8 or other evidence>` |
| Current non-claims | `<non-claims preserved before work>` |

## Proof Values

| Proof value | Protected behavior | Evidence before change |
|---|---|---|
| `<value>` | `<behavior>` | `<test/artifact/doc>` |

## Machine, Compiler, Backend, And Thread Context

| Field | Value |
|---|---|
| Host/runner | `<machine or CI lane>` |
| OS | `<OS>` |
| Compiler | `<compiler/version>` |
| Dense/backend configuration | `<backend>` |
| OpenMP/thread count | `<threads>` |
| Build type | `<debug/release/options>` |

## Fixture And Runtime Budget

| Fixture or workload | Size | Runtime budget | Reason |
|---|---:|---:|---|
| `<fixture>` | `<size>` | `<budget>` | `<reason>` |

## Metrics And Units

| Metric | Unit | Collection source | Interpretation |
|---|---|---|---|
| `<metric>` | `<unit>` | `<command/report>` | `<local/report-only/thresholded>` |

## Threshold Or Report-Only Status

| Surface | Status | Baseline source | Failure action |
|---|---|---|---|
| `<surface>` | `<thresholded / report-only>` | `<source>` | `<action>` |

## Report Index Or Stale-Report Handling

| Report artifact | Index path | Stale-report check | Notes |
|---|---|---|---|
| `<artifact>` | `<index>` | `<check or none with reason>` | `<notes>` |

## Correctness Context

- Correctness owner:
- Residual/reconstruction columns present:
- Why benchmark evidence does or does not own correctness:

## Validation Commands

| Command | Required because | Reviewed/supplemental/local | Result |
|---|---|---|---|
| `<command>` | `<reason>` | `<classification>` | `<pending/pass/fail>` |

Required trigger check:

- If any `.c` or `.h` file changed, run `make format && make lint && make test`.
- If benchmark scripts or reports changed, run affected benchmark/report
  checks.
- Keep local timing evidence separate from portable performance claims.

## Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | `<none / update / fence>` | `<action>` |
| Benchmarks docs | `<none / update / fence>` | `<action>` |
| Maintainer guide | `<none / update / fence>` | `<action>` |
| Release/planning claims | `<none / update / fence>` | `<action>` |

## Non-Portable Interpretation

- Machine-specific assumptions:
- Compiler/backend/thread assumptions:
- What this evidence can support:
- What this evidence cannot support:

## Non-Claims Preserved

- Portable performance superiority remains unclaimed unless separately proven.
- Vendor-backend parity remains unclaimed unless separately proven.
- Universal reorder/fill superiority remains unclaimed unless separately
  proven.
- `<additional non-claim>`

## Residual Handoff

| Residual | Next owner | Evidence link |
|---|---|---|
| `<residual>` | `<sprint/day/future epic>` | `<artifact>` |

## Completion Check

| Criterion | Status |
|---|---|
| Local context is recorded. | `<status>` |
| Metrics and units are recorded. | `<status>` |
| Threshold or report-only status is clear. | `<status>` |
| Report index or stale-report handling is recorded. | `<status>` |
| Validation commands are recorded. | `<status>` |
| Drift and non-claims are recorded. | `<status>` |
| Residual handoff is recorded. | `<status>` |
