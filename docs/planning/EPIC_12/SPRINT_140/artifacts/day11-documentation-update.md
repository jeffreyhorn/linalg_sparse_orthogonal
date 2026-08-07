# Day 11 Documentation Update

## Scope

Day 11 updates SVD-facing documentation with the earned Sprint 140
partial-SVD wording. The wording is intentionally fixture-local and points to
reproducible evidence instead of broadening public claims.

## Documentation Updated

| Surface | Update |
| --- | --- |
| `README.md` | Adds a concise maintained partial-SVD corpus proof bullet and updates the API summary for `sparse_svd_partial(...)`. |
| `include/sparse_svd.h` | Adds the public-header boundary for the generated 8x6 clustered/repeated fixture and preserved non-claims. |
| `docs/cookbook.md` | Adds the corpus fixture note near SVD route selection and ownership guidance. |
| `docs/tutorial.md` | Adds a bounded partial-SVD evidence note after the partial-SVD example. |
| `docs/solver_selection.md` | Adds solver-selection wording for the fixture-local proof owner and oracle command. |
| `docs/algorithm.md` | Adds the algorithm-facing boundary for the generated clustered/repeated fixture. |
| `examples/README.md` | Clarifies that example output is separate from the partial-SVD corpus proof. |
| `tests/corpus/README.md` | Adds the Sprint 140 partial-SVD lane, validation commands, generated-output expectations, stale-report signals, and residual-register updates. |
| `docs/maintainer_guide.md` | Updates the SVD trust-boundary row and Sprint 140 handoff wording. |

## Earned Wording

The docs now state that Sprint 140 earned fixture-local confidence for
`partial_svd_clustered_repeated_diag8x6_k3_v1`:

- generated 8x6 diagonal matrix
- `k = 3`
- clustered/repeated leading singular values
- top-3 singular-value checks
- left/right top-k subspace-projector checks
- triplet residual checks
- orthogonality checks
- default-budget success
- tight-budget fail-closed behavior
- no partial arrays on tight-budget failure

## Preserved Non-Claims

The documentation continues to fence off:

- broad partial-SVD correctness
- raw singular-vector identity
- broad repeated-spectrum coverage
- broad rank-deficient/null-space behavior
- external-library parity
- SuiteSparse parity
- platform parity
- package or ABI claims
- performance claims
- partial-result guarantees
- state-of-the-art claims

## Validation Plan

Because Day 11 updates `include/sparse_svd.h`, the full C quality gate remains
required:

```sh
make format
make lint
make test
```

Documentation and corpus hygiene checks:

```sh
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-partial-svd
git diff --check
```
