# Sprint 58 Day 12 - post-landing compatibility audit

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Re-audit the landed Sprint 58 public-surface cleanup to confirm the simplified
docs/example/header/benchmark story still preserves the steady-state workflow
contract before Day 13 validation.

## Re-audited surfaces

- `README.md`
- `docs/tutorial.md`
- `include/sparse_eigs.h`
- `include/sparse_iterative.h`
- `examples/README.md`
- `examples/example_eigs.c`
- `benchmarks/README.md`

Supporting checks:

- `rg` scan over workflow/support-boundary phrasing
- `git diff --stat master...HEAD` for the touched Sprint 58 surfaces
- `wc -l` for the final touched file set

## Main findings

### 1. The public workflow fence is still explicit and coherent

The landed Sprint 58 surfaces still agree on the stable product story:

- one-shot APIs remain the default/front-door workflows
- repeated-run direct solves remain a separate analyze-once / factor-many path
- repeated-run iterative handles remain limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated-run eigensolver handles remain limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows still read as one-shot
  compatibility surfaces rather than hidden handle support

Conclusion:

- no blocker-level contract drift surfaced across the touched public surfaces

### 2. Remaining documentation density is residual, not a hidden Sprint 58 defect

The strongest residual finding is explicit:

- deeper historical sections in `README.md` still contain many sprint-stamped
  chronology references

That is intentionally retained residual density rather than a blocker for final
validation because:

- Sprint 58 targeted the highest-signal workflow framing, headers, example
  wording, and benchmark taxonomy first
- the touched top-level workflow story is now materially clearer even though
  the deeper long-form historical sections remain dense

Conclusion:

- the remaining queue is explicit and future-facing, not a contradiction in the
  landed Sprint 58 contract

### 3. The structural compatibility signal remains strong

Sprint 58 still reads as wording/story cleanup rather than API redesign:

- no public function signatures changed
- no public struct layouts changed
- the touched surface diff is concentrated in:
  - `README.md`
  - `docs/tutorial.md`
  - touched public header comments
  - example docs/source comments
  - benchmark taxonomy docs

Measured touched-surface line counts:

- `README.md`: `973`
- `docs/tutorial.md`: `453`
- `include/sparse_eigs.h`: `646`
- `include/sparse_iterative.h`: `765`
- `examples/README.md`: `134`
- `examples/example_eigs.c`: `287`
- `benchmarks/README.md`: `248`

## Day 13 validation checklist

Required baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Targeted Sprint 58 follow-ons:

- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

## Deferred queue

Explicitly deferred residuals after Day 12:

- deeper long-form `README.md` chronology/performance history cleanup
- any lower-priority public-header follow-through only if a real contradiction
  emerges during Day 13 validation
- broader docs-density reduction outside the bounded Sprint 58 target set

## Conclusion

Sprint 58 Day 12 confirms that:

- the landed public workflow fence is still coherent
- no blocker-level wording contradiction remains before Day 13
- the residual documentation density is now consciously deferred
- final validation scope is fixed from the landed tree
