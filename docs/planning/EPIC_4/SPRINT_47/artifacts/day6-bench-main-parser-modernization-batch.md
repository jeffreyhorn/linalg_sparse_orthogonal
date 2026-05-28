# Sprint 47 Day 6 Artifact: `bench_main` Parser Modernization Batch

## Purpose

Widen the Day 5 helper landing into the main benchmark CLI by replacing the
remaining ad hoc parser paths in `bench_main.c`, tightening malformed-input and
unsupported-option handling, and proving the modernized CLI behavior directly
without broadening yet into reorder-mode parity or peer benchmark/example
cleanup.

## Main Day 6 Conclusion

`bench_main.c` now reads as a real shared-helper consumer instead of a mixed
old/new parser surface.

This batch stayed bounded to the main benchmark CLI:

- widened the shared helper seam usage inside `bench_main.c`
- added explicit help/usage handling
- tightened malformed-input and unsupported-option reporting
- added small invalid-combination checks for the existing benchmark modes

The important boundary held:

- no public CLI API was added
- no benchmark framework redesign landed
- no reorder-mode parity expansion landed beyond replacing the ad hoc parse
  branch with the shared helper contract
- peer benchmark, example, and script surfaces remain later Sprint 47 work

## Landed `bench_main` Modernization

### Shared-helper adoption widened

`bench_main.c` now routes the remaining mode-string parser drift through the
Day 5 helper seam:

- `--reorder`

That means the touched main CLI now consistently uses the shared helper layer
for:

- `--spmv-iters`
- `--size`
- `--repeat`
- `--pivot`
- `--reorder`

Interpretation:

- Day 6 removes the last high-signal ad hoc parse branch from the main benchmark
  CLI without jumping ahead to the later supported-mode parity policy batch

### Help and missing-value handling are now explicit

`bench_main.c` now provides:

- `--help`
- `-h`

and prints a dedicated usage block instead of falling through into benchmark
execution.

Flags that require values now fail explicitly and immediately when the value is
missing, including:

- `--spmv-iters`
- `--size`
- `--repeat`
- `--dir`
- `--pivot`
- `--reorder`

Interpretation:

- the main benchmark CLI now behaves like a real maintained command surface
  instead of relying on accidental fallthrough

### Unsupported-option and invalid-combination reporting tightened

`bench_main.c` now rejects:

- unknown options
- multiple positional matrix paths
- mixed `filename` plus `--dir`
- mixed `--spmv` plus `--iterative`

The user-facing error shape is now clearer and more consistent with the Sprint
47 design intent.

Interpretation:

- Day 6 improved benchmark CLI failure clarity without changing the benchmark
  capability set itself

## Direct CLI Proof

The touched CLI surface was exercised directly with focused checks:

- help path:
  - `./build/bench_main --help`
- valid reorder parse path:
  - `./build/bench_main --reorder amd --size 8 --repeat 1`
- missing value rejection:
  - `./build/bench_main --reorder`
- unknown option rejection:
  - `./build/bench_main --bogus`
- conflicting mode rejection:
  - `./build/bench_main --spmv --iterative`

Observed direct behavior:

- `--help` now prints usage and exits cleanly
- valid `--reorder amd` input completed successfully and reported `Reorder: amd`
- missing `--reorder` value failed cleanly with:
  - `bench_main: --reorder requires a value`
- unknown option failed cleanly with:
  - `bench_main: unknown option '--bogus' (try --help)`
- conflicting modes failed cleanly with:
  - `bench_main: choose only one mode: --spmv or --iterative`

Interpretation:

- Day 6 proved both the improved success-path behavior and the intended
  malformed/unsupported-input contracts

## Validation

Because `*.c` changed, the required gate was:

```bash
make format
make lint
make test
```

Those all passed.

Because Day 6 is a broader main-CLI modernization batch, the stronger reviewed
baseline also ran:

```bash
make quality-review-full
```

That passed too, including the reviewed CMake parity path:

- `53 / 53` tests passed

## Sprint 47 Position After Day 6

The next landing order is now clearer:

1. `bench_main` no longer depends on the older ad hoc parsing style
2. Day 7 can audit the residual queue honestly from a modernized main CLI
3. Day 8 can focus specifically on reorder-mode / emitted-label parity instead
   of still mixing in parser modernization work

## Bottom Line

Day 6 delivered:

- a modernized `bench_main` parser built around the Day 5 helper seam
- explicit help text and clearer error reporting
- bounded invalid-combination checks
- a fully green required gate plus stronger reviewed baseline

That is the right bounded second code landing for Sprint 47.
