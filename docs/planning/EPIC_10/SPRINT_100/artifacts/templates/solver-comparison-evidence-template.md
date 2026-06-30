# Solver Comparison Evidence Template

## Summary

| field | value |
|---|---|
| comparison family | direct / iterative / eigensolver / SVD / reorder / graph |
| solver or algorithm path | TODO |
| artifact owner | TODO |
| implementation owner | TODO |
| test owner | TODO |
| external oracle owner | TODO or `none` |
| benchmark owner | TODO or `none` |
| validation command | TODO |
| claim state before work | earned / candidate / blocked / non-goal |
| claim state after work | earned / candidate / blocked / non-goal |

## Claim Evaluated

Bounded claim:

> TODO: write the exact claim this comparison is allowed to support.

Disallowed broader claim:

> TODO: write the broader claim this artifact does not support.

## Fixture Set

| fixture | source | dimensions | nnz | class | reason selected |
|---|---|---:|---:|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

Fixture-class checklist:

- [ ] symmetry recorded
- [ ] definiteness recorded
- [ ] conditioning or scale notes recorded
- [ ] sparsity pattern notes recorded
- [ ] expected success/failure recorded
- [ ] runtime cost bounded
- [ ] fixture availability documented

## Problem Construction

| field | value |
|---|---|
| matrix construction or load path | TODO |
| RHS / target vector construction | TODO |
| initial guess or block initialization | TODO or `n/a` |
| preconditioner construction | TODO or `n/a` |
| eigen/SVD target definition | TODO or `n/a` |
| reorder/backend/runtime options | TODO |
| random seed policy | TODO or `deterministic/no random seed` |

## Oracle or Reference Behavior

| field | value |
|---|---|
| oracle type | external process / dense reference / direct solver cross-check / deterministic property / benchmark-only |
| oracle command | TODO or `n/a` |
| oracle implementation owner | TODO |
| success output contract | TODO |
| skip output contract | TODO |
| error output contract | TODO |
| platform exclusions | TODO |

## Acceptance Criteria

### Correctness

| metric | threshold | rationale |
|---|---:|---|
| residual norm | TODO | TODO |
| max solution/eigenvector/singular-vector difference | TODO | TODO |
| reconstruction error | TODO | TODO |
| rank/eigenvalue/singular-value agreement | TODO | TODO |

### Convergence

| metric | threshold | rationale |
|---|---:|---|
| converged flag | TODO | TODO |
| iteration count | TODO or `not thresholded` | TODO |
| stagnation/breakdown status | TODO | TODO |
| residual history behavior | TODO | TODO |

### Timing

| metric | threshold | interpretation |
|---|---:|---|
| wall time | TODO or `not captured` | local context only unless justified |
| memory or basis size | TODO or `not captured` | TODO |

Timing caveat:

> Local timing is not portable performance superiority unless a separate
> benchmark-sentinel artifact explicitly defines machine class, fixture,
> threshold, and non-claim wording.

## Unsupported, Skipped, and Expected-Failure Cases

| case | expected status | reason | claim impact |
|---|---|---|---|
| TODO | pass / skip / fail / xfail | TODO | TODO |

## Validation Command

```sh
TODO
```

Expected result:

```text
TODO
```

If `.c` or `.h` files changed, required broader validation:

```sh
make format && make lint && make test
```

## Evidence Summary

| evidence type | result | notes |
|---|---|---|
| correctness | TODO | TODO |
| convergence | TODO | TODO |
| timing | TODO | TODO |
| unsupported cases | TODO | TODO |
| platform behavior | TODO | TODO |

## Non-Claims

This artifact does not claim:

- TODO
- TODO
- TODO

## Follow-Up Work

| follow-up | owner | reason |
|---|---|---|
| TODO | TODO | TODO |

