# Benchmark Interpretation Template

## Summary

| field | value |
|---|---|
| benchmark surface | TODO |
| benchmark binary or target | TODO |
| command | TODO |
| artifact owner | TODO |
| implementation owner | TODO |
| report owner | TODO |
| category | canonical maintained / regression-sensitive / exploratory |
| fixture scope | TODO |
| metric scope | local timing / throughput / memory / fill / residual / mixed |
| claim state before work | earned / candidate / blocked / non-goal |
| claim state after work | earned / candidate / blocked / non-goal |

## Claim Boundary

Bounded claim:

> TODO: write the exact benchmark interpretation this artifact supports.

Disallowed broader claim:

> TODO: write the portable or ecosystem-wide performance claim this artifact
> does not support.

## Command and Environment

| field | value |
|---|---|
| command | TODO |
| working directory | repository root |
| build mode | TODO |
| compiler and flags | TODO |
| platform | TODO |
| CPU / machine class | TODO |
| thread settings | TODO or `n/a` |
| BLAS / SuiteSparse / optional backend state | TODO or `n/a` |
| repeat count | TODO |
| random seed policy | TODO or `deterministic/no random seed` |

## Fixture Set

| fixture | source | dimensions | nnz | class | reason selected |
|---|---|---:|---:|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## Metrics

| metric | unit | emitted by | interpretation |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

Metric ownership checklist:

- [ ] local timing metrics are explicitly labeled local
- [ ] throughput or speedup metrics identify their comparison baseline
- [ ] residual or correctness fields identify the test/oracle owner
- [ ] fill, memory, or basis-size fields are separated from wall time
- [ ] unsupported rows or skipped workloads are represented explicitly

## Output Artifacts

| artifact | format | owner | interpretation |
|---|---|---|---|
| TODO | CSV / TSV / text / HTML / none | TODO | TODO |

## Acceptance or Reporting Mode

| field | value |
|---|---|
| mode | threshold-free report / thresholded sentinel / smoke runtime / exploratory |
| pass/fail threshold | TODO or `none` |
| threshold source | TODO or `n/a` |
| expected status | pass / report-only / skip / xfail |
| rerun guidance | TODO |

## Evidence Summary

| evidence type | result | notes |
|---|---|---|
| local timing | TODO | TODO |
| fill or memory | TODO | TODO |
| residual or correctness context | TODO | TODO |
| artifact generation | TODO | TODO |
| unsupported cases | TODO | TODO |

## Non-Claims

This artifact does not claim:

- TODO
- TODO
- TODO

## Follow-Up Work

| follow-up | owner | reason |
|---|---|---|
| TODO | TODO | TODO |
