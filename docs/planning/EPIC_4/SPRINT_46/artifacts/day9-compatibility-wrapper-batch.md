# Sprint 46 Day 9 Artifact: Compatibility Wrapper Batch

## Purpose

Normalize the public eigensolver one-shot entry so it reads explicitly as a
compatibility-preserving wrapper over the already-migrated reusable internal
backend paths, without widening Sprint 46 into another solver, workspace,
benchmark, or public-API redesign batch.

## Main Day 9 Conclusion

Sprint 46's public eigensolver entry now has one clearer wrapper/composition
layer instead of keeping defaults, validation, result initialization, backend
selection, and backend delegation interleaved in one long `sparse_eigs_sym(...)`
body.

This batch was bounded to:

- one-shot wrapper normalization
- explicit wrapper-vs-backend ownership cleanup
- preservation of current public behavior

It did **not** widen into:

- eigensolver algorithm changes
- new workspace models
- benchmark work
- public API changes

## Landed Compatibility Scope

### 1. The one-shot public entry now uses small explicit wrapper helpers

Day 9 added small internal helpers for:

- default public option construction
- entry validation
- result initialization
- backend selection
- backend delegation

Interpretation:

- the public entry now reads more clearly as wrapper glue
- the compatibility boundary is easier to audit and reason about

### 2. The backend implementations remain the behavioral truth

After Day 9, the public wrapper explicitly composes over the existing internal
backend implementations for:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

Interpretation:

- the backend implementations still own the real eigensolver behavior
- the wrapper layer now owns only public-entry composition and compatibility
  concerns

### 3. AUTO and explicit dispatch are now one explicit compatibility seam

The current backend-choice rules now live behind one helper that preserves the
existing behavior for:

- explicit `SPARSE_EIGS_BACKEND_LOBPCG`
- explicit `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`
- AUTO LOBPCG routing with preconditioner / threshold / block-size gates
- AUTO thick-restart routing
- grow-m fallback

Interpretation:

- dispatch policy stayed unchanged
- the public wrapper now has one obvious backend-selection seam instead of an
  interleaved inline decision tree

## Preserved Boundaries

The Day 9 batch kept these responsibilities outside the new wrapper helpers:

- Lanczos iteration math
- thick-restart state/control
- LOBPCG block iteration math
- reusable workspace implementation
- repeated-run measurement policy

Interpretation:

- Day 9 normalized wrapper structure, not internal solver design
- Sprint 46 stayed away from reopening the already-landed workspace work

## Validation

Because `*.c` changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

Targeted wrapper-focused eigensolver follow-ons also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`

Interpretation:

- public defaults, AUTO/explicit dispatch, and one-shot example behavior all
  remained stable after the wrapper cleanup

## Sprint 46 Position After Day 9

The remaining sprint order is now clearer:

1. repeated-run benchmark design/evidence
2. maintainer-facing memory-behavior/documentation closeout

Interpretation:

- the primary workspace migration queue is already closed
- the one-shot wrapper/composition cleanup queue is now substantially reduced
- Sprint 46 is ready to pivot into measured repeated-run evidence

## Bottom Line

Day 9 delivered:

- a clearer compatibility-preserving one-shot wrapper structure for
  `sparse_eigs_sym(...)`
- explicit wrapper-vs-backend ownership separation
- preserved public defaults and backend routing behavior
- a green validation baseline plus direct wrapper-focused eigensolver reruns

That is the right bounded compatibility batch for Sprint 46.
