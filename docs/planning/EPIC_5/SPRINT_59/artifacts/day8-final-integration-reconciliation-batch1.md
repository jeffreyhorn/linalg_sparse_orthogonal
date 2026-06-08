# Sprint 59 Day 8 - final integration reconciliation batch 1

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Land the first bounded final integration reconciliation patch by tightening the
top-level README/tutorial terminology and workflow positioning so the front-door
docs match the more precise stable vocabulary already used by the headers,
examples, and benchmark docs.

## Touched surfaces

- `README.md`
- `docs/tutorial.md`

Untouched by design:

- public headers
- `examples/README.md`
- `benchmarks/README.md`
- proof surfaces under `tests/`

## Landed changes

### 1. README terminology alignment

The touched README sections now:

- describe the direct repeated path as the explicit repeated-run direct
  lifecycle
- describe repeated iterative use as the explicit iterative-handle path
- describe repeated eigensolver use as the explicit eigensolver-handle path
- remove stale `Sprint 49` framing from the repeated-run handle section

### 2. Tutorial terminology alignment

The touched tutorial sections now:

- describe the opt-in reuse paths as the explicit iterative-handle or
  eigensolver-handle lifecycle
- describe the direct repeated path as the explicit repeated-run direct
  lifecycle

## Preserved invariants

The batch preserved:

- one-shot APIs as the default/front-door workflow
- repeated-run iterative handles limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated-run eigensolver handle limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows as one-shot compatibility surfaces
- the explicit analysis/factors lifecycle as the repeated direct path

## Alignment result

After the Day 8 patch, the top-level docs now line up more closely with the
stable lifecycle vocabulary already used by:

- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `examples/README.md`
- `benchmarks/README.md`

## Sanity checks

Targeted checks after the patch:

- `git diff -- README.md docs/tutorial.md`
- `rg -n "explicit repeated-run direct lifecycle|iterative handles|eigensolver handle|one-shot compatibility surfaces|BiCGSTAB|LOBPCG|grow-m|thick-restart" README.md docs/tutorial.md examples/README.md benchmarks/README.md include/sparse_analysis.h include/sparse_iterative.h include/sparse_eigs.h`
- `wc -l README.md docs/tutorial.md`

This was a docs-only batch, so the `make format` / `make lint` / `make test`
code-day gate was not required.

## Conclusion

Day 8 lands one bounded final integration patch:

- the README front door now uses more precise stable lifecycle vocabulary
- the tutorial reinforces the same explicit direct/iterative/eigensolver
  handle terminology
- the strongest remaining cross-surface drift is materially smaller
