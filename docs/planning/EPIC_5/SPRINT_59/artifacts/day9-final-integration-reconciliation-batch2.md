# Sprint 59 Day 9 - final integration reconciliation batch 2

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Re-audit the landed Day 8 reconciliation surfaces and remove one last bounded
pocket of sprint-local caller-story phrasing from the top-level docs without
opening a broader docs or header cleanup pass.

## Touched surfaces

- `README.md`
- `docs/tutorial.md`

Untouched by design:

- public headers
- `examples/README.md`
- `benchmarks/README.md`
- proof surfaces under `tests/`

## Landed changes

### 1. README repeated-run section normalization

The repeated-run iterative-handle section now:

- states the supported handle set as a stable product rule:
  - `CG`
  - `GMRES`
  - `MINRES`
- states the explicit exclusions without sprint-local framing:
  - `BiCGSTAB`
  - block iterative workflows
- states that one-shot entries remain the compatibility path without referring
  to a specific sprint

### 2. Tutorial iterative wording normalization

The tutorial now refers to:

- `stable-dimension iterative-handle workflows`

instead of the looser phrase:

- `stable-dimension repeated iterative solves`

## Preserved invariants

The batch preserved:

- one-shot APIs as the default/front-door workflow
- repeated-run direct solves as the explicit analysis/factors lifecycle
- repeated-run iterative handles limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated-run eigensolver handle limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows as one-shot compatibility surfaces

## Alignment result

After the Day 9 patch, the top-level docs now read more consistently with the
stable lifecycle vocabulary already used by:

- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `examples/README.md`
- `benchmarks/README.md`

The remaining difference is mostly density:

- long-form history
- deep benchmark/performance context
- already-truthful lower-level detail that the top-level docs do not need to
  duplicate

## Sanity checks

Targeted checks after the patch:

- `git diff -- README.md docs/tutorial.md`
- `rg -n "Sprint 49|Sprint 54|iterative-handle workflows|public repeated-run iterative handles are intentionally limited|the library does not expose public repeated-run handles" README.md docs/tutorial.md`
- `wc -l README.md docs/tutorial.md`

This was a docs-only batch, so the `make format` / `make lint` / `make test`
code-day gate was not required.

## Conclusion

Day 9 lands one final bounded integration cleanup:

- the README repeated-run iterative section now uses stable product wording
- the tutorial points at the same iterative-handle lifecycle
- the remaining cross-surface queue is now mostly optional density rather than
  active caller-story drift
