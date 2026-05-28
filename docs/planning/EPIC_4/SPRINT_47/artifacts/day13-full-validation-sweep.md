# Sprint 47 Day 13: Full Validation Sweep

## Summary

Day 13 ran the authoritative full validation sweep for the Sprint 47 benchmark
CLI modernization, example safety cleanup, auxiliary tooling checks, and docs
refresh.

The result is clean:

- the mandatory code-change floor passed
- `make quality-review-full` passed
- reviewed CMake parity remained exact at `53`
- the direct benchmark CLI, example, tooling-build, and dead-code script
  follow-ons all passed

No Sprint-47-specific reconciliation queue surfaced.

## Validation Runs

### Mandatory floor

- `make format` → passed, `real 4.74`
- `make lint` → passed, `real 415.57`
- `make test` → passed, `real 180.79`

### Strong reviewed proof

- `make quality-review-full` → passed, `real 1619.23`

Included reviewed-path components:

- reviewed Makefile path
- `deadcode-check`
- reviewed clean CMake rebuild
- `ctest -N`
- full reviewed CMake `ctest`

### Direct benchmark / example / tooling reruns

- `make tooling-build` → passed, `real 0.29`
- `./build/bench_main --help` → passed
- `./build/bench_main --reorder nd --size 8 --repeat 1` → passed
- `./build/bench_main --reorder colamd` → rejected as intended, exit `1`
- `./build/example_eigs` → passed
- `python3 -m py_compile scripts/deadcode_report.py` → passed
- `bash -n scripts/deadcode_workflow.sh` → passed
- synthetic `deadcode_report.py` valid-artifact run → passed
- synthetic `deadcode_report.py --check` run → passed
- synthetic malformed coverage-note parse → rejected as intended

## Truthfulness Anchors

The preserved Sprint 40 validation/truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 186.24 sec`

Interpretation:

- Sprint 47 did not disturb the maintained local reviewed baseline
- the benchmark/example/tooling cleanup work remains aligned with the
  established Makefile/CMake parity contract

## Sprint-47-Specific Outcome

The direct reruns matter because Sprint 47 touched three different auxiliary
surface classes:

- benchmark CLI behavior in `bench_main`
- example-side helper adoption in `example_eigs`
- dead-code support tooling

The explicit Sprint 47 protection points stayed green:

- `bench_main --help`
  - printed the real usage block and exited cleanly
- `bench_main --reorder nd --size 8 --repeat 1`
  - completed successfully
  - reported `Reorder: nd`
- `bench_main --reorder colamd`
  - rejected unsupported input cleanly
  - pointed users to `bench_reorder` / `bench_colamd`
- `example_eigs`
  - nos4 largest-eigenvalue demo converged `5 / 5`
  - KKT nearest-sigma demo converged `3 / 3`
  - bcsstk04 LOBPCG + IC(0) demo converged `3 / 3`
- `deadcode_report.py`
  - handled a synthetic valid artifact cleanly
  - rejected malformed coverage-note metadata as intended
- `deadcode_workflow.sh`
  - passed shell syntax validation

Interpretation:

- the modernized benchmark CLI remains behaviorally honest on both success and
  malformed-input paths
- the touched example still demonstrates the intended eigensolver mix cleanly
- the auxiliary dead-code tooling remains strict on malformed inputs without
  regressing its normal support path

## Auxiliary Note

The reviewed clean CMake rebuild emitted a non-gating warning in
`bench_eigs_reuse.c`:

- `-Wdouble-promotion` on `NAN` fallback expressions

Interpretation:

- this did not fail the maintained reviewed gate
- it also did not come from a Sprint 47 touched file
- it should be treated as auxiliary follow-on context rather than a Day 13
  blocker

## Caveats

No new Sprint-47-specific caveats surfaced beyond the maintained contract:

- dead-code execution remains serialized
- reviewed CMake remains the strongest shared reviewed baseline
- Sprint 47 intentionally does not solve:
  - broad benchmark framework redesign
  - public CLI helper APIs in the core library
  - broad example/tutorial restructuring
  - dead-code workflow redesign

## Outcome

Sprint 47 now enters Day 14 closeout from a measured, validated state:

- benchmark CLI modernization validated
- reorder-mode / emitted-label parity validated
- example helper cleanup validated
- auxiliary tooling safety cleanup validated
- touched docs refresh validated against live behavior
- reviewed baseline and reviewed CMake parity preserved

That is the correct end-state before Sprint 47 closeout and handoff.
