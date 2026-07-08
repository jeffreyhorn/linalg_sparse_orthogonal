# Day 8 Partial-Result Publication Exhaustion Proof

## Purpose

Day 8 proves the grow-m Lanczos `m_cap` exhaustion publication path. The proof
focuses on public result shape and bounded behavior after the final allowed
Lanczos run fails the requested tolerance.

## Exhaustion Path

The grow-m backend computes:

- `m_cap = min(max_iterations, n)`, with a lower bound of `min(2*k + 10, n)`;
- `m_init = min(3*k + 30, m_cap)`;
- `m_grow = min(k + 20, m_cap)`.

For the Day 8 fixture:

| Parameter | Value | Effect |
|---|---:|---|
| `n` | 80 | Large enough that `max_iterations` controls `m_cap`. |
| `k` | 3 | Requires `max_iterations >= 16`. |
| `max_iterations` | 16 | Sets `m_cap = m_init = 16`, so exactly one bounded Lanczos run occurs. |
| `tol` | `1e-18` | Intentionally tighter than the final residual, forcing `SPARSE_ERR_NOT_CONVERGED`. |
| Backend | `SPARSE_EIGS_BACKEND_LANCZOS` | Keeps proof on the grow-m path. |
| Fixture | shifted tridiagonal SPD | Deterministic non-diagonal spectrum without invariant-subspace early exit. |

## Implemented Proof

| File | Test | Evidence |
|---|---|---|
| `tests/test_eigs.c` | `test_s114_mcap_exhaustion_publishes_bounded_partial_result` | Asserts `SPARSE_ERR_NOT_CONVERGED`, `n_requested == k`, `n_converged == k`, `backend_used`, `peak_basis_size == 16`, `iterations == 16`, finite residual larger than tolerance, one progress callback at step 0, finite descending published values, nonzero finite vector columns, and untouched sentinel slots beyond `k`. |

## Publication Invariants

- The backend publishes only through caller-owned indices
  `[0, n_converged)`.
- Sentinel values and vector columns beyond `k` remain untouched.
- The result reports the bounded memory shape through `peak_basis_size`.
- The result reports the actual single-run work through `iterations`.
- `residual_norm` remains finite even when the return code is
  `SPARSE_ERR_NOT_CONVERGED`.

## Helper-Movement Assessment

Partial-publication helper movement remains blocked. Day 8 proves the grow-m
`m_cap` exhaustion branch, but thick-restart has a distinct restart-state
fallthrough, and Day 9 still needs shift-invert grow-m conversion proof before
Day 10 can make an eigensolver movement decision.

## Validation Plan

Day 8 modifies C tests, so the required quality gate is:

```sh
make build/test_eigs
./build/test_eigs
make format && make lint && make test
```

## Completion Criteria

- Bounded `m_cap` exits publish a documented partial result shape.
- Result count, memory, iteration, residual, and non-overrun invariants are
  explicit.
- Shift-invert grow-m proof can proceed with grow-m partial publication
  behavior pinned.
