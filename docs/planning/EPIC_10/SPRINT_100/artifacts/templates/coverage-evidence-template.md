# Coverage Evidence Template

## Summary

| field | value |
|---|---|
| coverage surface | TODO |
| command | TODO |
| artifact owner | TODO |
| backend | lcov / gcovr / other |
| reviewed status | reviewed / supplemental / exploratory |
| threshold | TODO |
| claim state before work | earned / candidate / blocked / non-goal |
| claim state after work | earned / candidate / blocked / non-goal |

## Claim Boundary

Bounded claim:

> TODO: write the exact coverage claim this artifact supports.

Disallowed broader claim:

> TODO: write what this coverage run does not prove.

## Command and Environment

| field | value |
|---|---|
| command | TODO |
| working directory | repository root |
| compiler | TODO |
| backend selection rule | TODO |
| build-tree behavior | tree-mutating / non-mutating |
| reset requirement after run | TODO |
| platform | TODO |
| required external tools | TODO |

## Scope

| scope element | included? | notes |
|---|---|---|
| `src/` | yes / no | TODO |
| `include/` or headers | yes / no | TODO |
| `tests/` | yes / no | TODO |
| `benchmarks/` | yes / no | TODO |
| examples | yes / no | TODO |
| generated docs | yes / no | TODO |

Known exclusions:

- TODO
- TODO

## Threshold and Output

| field | value |
|---|---|
| aggregate line threshold | TODO |
| file-level threshold | TODO or `none` |
| branch/function threshold | TODO or `none` |
| primary machine-readable artifact | TODO |
| primary human-readable artifact | TODO |
| parse rule | TODO |

## Reviewed vs Supplemental Interpretation

| question | answer |
|---|---|
| Is this part of `make quality-review-full`? | TODO |
| Is this part of mandatory PR validation? | TODO |
| Does it mutate the normal build tree? | TODO |
| What command restores normal reviewed-path state? | TODO |
| What does a pass mean? | TODO |
| What does a fail mean? | TODO |

## Evidence Summary

| evidence type | result | notes |
|---|---|---|
| command status | TODO | TODO |
| aggregate coverage | TODO | TODO |
| threshold check | TODO | TODO |
| exclusions | TODO | TODO |
| reviewed status | TODO | TODO |

## Non-Claims

This artifact does not claim:

- TODO
- TODO
- TODO

## Follow-Up Work

| follow-up | owner | reason |
|---|---|---|
| TODO | TODO | TODO |
