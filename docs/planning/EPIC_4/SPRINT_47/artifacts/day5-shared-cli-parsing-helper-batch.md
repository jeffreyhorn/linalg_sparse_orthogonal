# Sprint 47 Day 5 Artifact: Shared CLI Parsing Helper Batch

## Purpose

Land the first real Sprint 47 code batch by introducing the bounded internal
benchmark CLI parsing helper seam and proving it in the main benchmark
modernization hotspot before widening into reorder-mode parity, peer benchmark
alignment, examples, or script-side cleanup.

## Main Day 5 Conclusion

Sprint 47 now has a real shared internal benchmark CLI parsing helper layer,
not just a design.

That first landing is intentionally narrow:

- new internal benchmark helper seam:
  - `benchmarks/bench_cli_parse_internal.h`
- first live consumer:
  - `benchmarks/bench_main.c`
- first migrated argument classes:
  - checked positive integer parsing
  - checked bounded integer parsing
  - checked enum-like choice parsing for one benchmark-local mode family

The batch stayed within the Day 3 and Day 4 boundaries:

- the helper layer is internal-only
- parse-plus-range-check is one helper contract
- benchmark-specific usage text and behavior remain local
- reorder-mode parsing is still deferred to the later parity batch

## Landed Helper Layer

### New internal helper seam

`benchmarks/bench_cli_parse_internal.h` now provides a small internal parser
surface for benchmark/example-style CLIs:

- checked integer parsing through `strtol`
- range-bounded `int` conversion helpers
- finite-double parsing support for later consumers
- caller-configured enum-like string choice parsing

The helper contract owns:

- missing-value rejection
- trailing-junk rejection
- overflow / underflow rejection
- range validation
- benchmark-local error emission with flag-aware messages

Interpretation:

- Sprint 47 does not need a public library API to modernize auxiliary CLIs
- the helper seam is broad enough for later bounded consumers without forcing
  Day 5 to rewrite every benchmark at once

## First Live Adoption

### `bench_main.c` now uses the shared helper seam for the first parser-drift set

`bench_main.c` now routes the touched arguments through the shared helper layer:

- `--spmv-iters`
- `--size`
- `--repeat`
- `--pivot`

This replaces the previous parser drift in the first three cases:

- raw `atoi(...)`
- manual positive-value checks
- local choice-string branching for pivot mode

The landed proof keeps benchmark-specific behavior local:

- `bench_main` still owns its usage/help text
- `bench_main` still owns runtime defaults
- `bench_main` still owns reorder-mode handling for the later parity batch

### Explicit non-goals for Day 5

This batch did **not** yet land:

- reorder-mode cleanup in `bench_main.c`
- peer benchmark helper alignment in `bench_eigs.c`
- repeated-run benchmark driver cleanup
- example CLI/helper cleanup
- script-side helper alignment

Interpretation:

- Day 5 proved the helper seam in the main hotspot without diluting the sprint
  into multi-surface churn
- Day 6 can now focus on wider `bench_main` modernization from a stable helper
  base

## Direct CLI Proof

The touched parse paths were exercised directly with small focused checks:

- valid generated-matrix SpMV path:
  - `./build/bench_main --spmv --size 8 --spmv-iters 5 --repeat 2 --pivot partial`
- valid solve-path pivot choice:
  - `./build/bench_main --size 8 --repeat 1 --pivot partial`
- malformed numeric rejection:
  - `./build/bench_main --spmv-iters nope`

Observed direct behavior:

- valid generated input completed successfully
- valid pivot-mode input completed successfully with `Pivot: partial`
- malformed `--spmv-iters` input failed cleanly with:
  - `bench_main: --spmv-iters: not a valid integer: 'nope'`

Interpretation:

- Day 5 proved both the successful checked parse path and the benchmark-local
  error-reporting contract

## Validation

Because `*.c` and `*.h` files changed, the required gate for the batch was:

```bash
make format
make lint
make test
```

Those all passed.

Because Day 4 set the shared-helper landing as a stronger reviewed-baseline
batch, the broader local wrapper validation was also run:

```bash
make quality-review-full
```

That passed as well.

The touched CLI surface was then verified with the direct `bench_main` reruns
listed above.

## Sprint 47 Position After Day 5

The next landing order is now clearer:

1. the shared parser helper seam exists and is live in `bench_main`
2. Day 6 can widen `bench_main` modernization from that seam
3. Day 7 can audit the residual parser and parity queue honestly
4. reorder-mode / emitted-label parity can then land on top of the helper-based
   benchmark surface

## Bottom Line

Day 5 delivered:

- the first real internal benchmark CLI parsing helper layer
- a live `bench_main` proof for checked integer and choice parsing
- direct malformed-input rejection evidence
- a fully green validation baseline for the touched helper/benchmark surface

That is the right bounded first code landing for Sprint 47.
