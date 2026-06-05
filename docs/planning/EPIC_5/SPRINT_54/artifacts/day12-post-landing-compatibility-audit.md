# Sprint 54 Day 12 - post-landing compatibility audit

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Audit the landed Sprint 54 branch against the preserved public repeated-run
solver fence before the final validation sweep:

- one-shot APIs remain first-class
- repeated-run handles remain bounded opt-in paths
- excluded families still read as intentional exclusions
- the Day 13 validation checklist is fixed from the real landed state

## Audited surfaces

The audit checked the highest-signal live surfaces across:

- public headers
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
- caller-facing docs
  - `README.md`
  - `examples/README.md`
  - `docs/tutorial.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- direct proof surfaces
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
- benchmark proof surfaces
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- shipped examples
  - `example_iterative`
  - `example_ic_minres`
  - `example_eigs`

## Main findings

### 1. One-shot vs handle ownership remains consistent

The most important compatibility result is still intact:

- one-shot solver APIs remain first-class
- repeated-run handles remain opt-in for stable-dimension repeated runs
- shipped examples still read as intentionally one-shot-first

This stayed consistent across the top-level README, example README, public
headers, and benchmark descriptions.

### 2. The iterative repeated-run handle boundary remains honest

The landed iterative repeated-run handle set still reads consistently as:

- `CG`
- `GMRES`
- `MINRES`

And the intended exclusions still read as explicit exclusions:

- `BiCGSTAB`
- block iterative workflows

No audited surface overclaimed handle support for those excluded families.

### 3. The eigensolver repeated-run handle boundary remains honest

The landed eigensolver repeated-run handle set still reads consistently as:

- grow-m Lanczos
- thick-restart Lanczos
- explicit `LOBPCG`

The direct proof, benchmark proof, and user-facing docs now all agree on that
same three-backend support set.

### 4. Reuse semantics remain bounded and truthful

The landed wording still preserves the intended honesty boundary:

- reuse preserves allocation capacity
- reuse does not preserve old numerical iteration/search state
- one-shot APIs remain supported and are not demoted

That remains consistent across the README, examples, public headers, and the
measured benchmark/readback surfaces.

### 5. No blocker-level drift remains

The Day 12 audit did not surface a blocker-level mismatch between:

- code
- tests
- benchmarks
- examples
- caller-facing docs

The remaining queue is future-facing rather than corrective:

- larger tutorial/example modernization if a later sprint wants explicit
  repeated-run teaching code
- any later public-handle expansion beyond the bounded Sprint 54 surface

## Day 13 validation checklist

The validation checklist fixed from the landed branch is:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- targeted Sprint 54 follow-ons:
  - `./build/test_iterative`
  - `./build/test_minres`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

## Conclusion

Day 12 confirms that the landed Sprint 54 branch still matches the preserved
public repeated-run solver fence:

- one-shot APIs remain first-class
- repeated-run handles remain bounded opt-in paths
- excluded families still read as intentional exclusions
- the final validation sweep now has an explicit checklist from the real
  landed state
