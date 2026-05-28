# Sprint 47 Day 14: Closeout and Handoff

## Summary

Sprint 47 closes with a validated auxiliary-surface modernization package
across the benchmark CLI, example safety paths, bounded tooling hardening, and
touched benchmark/example docs.

The sprint now hands off:

- a small shared internal benchmark CLI parsing-helper seam
- a modernized `bench_main` CLI surface
- aligned reorder-mode / emitted-label behavior in `bench_main`
- bounded example-side safety/helper adoption in `example_eigs`
- bounded dead-code tooling hardening
- touched benchmark/example docs aligned with live runtime behavior
- a measured Day 13 validated baseline

This is a real auxiliary-surface safety and usability handoff, not just a set
of wording tweaks or isolated parser substitutions.

## What Sprint 47 Accomplished

### 1. Landed a shared internal benchmark CLI parsing-helper seam

Sprint 47 added:

- `benchmarks/bench_cli_parse_internal.h`

That helper seam now owns bounded parsing contracts for:

- checked integers
- bounded integers
- finite doubles
- enum-like choice parsing

This gives later auxiliary CLI work a small reusable internal parsing contract
instead of forcing each benchmark entrypoint to keep duplicating `atoi`-style
parsing and fragmented range checks.

### 2. Modernized `bench_main` without broadening into framework redesign

Sprint 47 modernized the main benchmark CLI in:

- `benchmarks/bench_main.c`

The landed behavior improvements are concrete:

- real `--help` / `-h`
- explicit missing-value failures
- explicit unknown-option failures
- explicit bounded invalid-combination rejection
- shared helper-backed parsing for key numeric and enum-like flags

This is the right scope for Sprint 47:

- better malformed-input behavior
- clearer usage reporting
- no broad benchmark framework redesign

### 3. Reconciled reorder-mode ownership and emitted-label parity

Sprint 47 also cleaned up internal drift around `--reorder` in `bench_main`:

- help text now matches the live supported modes
- accepted parser values now match emitted reporting
- unsupported `colamd` input now gives explicit handoff guidance to:
  - `bench_reorder`
  - `bench_colamd`

That leaves the benchmark surface clearer about which tool owns which reorder
comparison job.

### 4. Landed a bounded example-side safety cleanup

Sprint 47 made the intentionally small example-side cleanup in:

- `examples/example_eigs.c`

That batch:

- adopted `examples/example_alloc_helpers.h`
- routed touched allocations through checked helper calls
- tightened the multi-vector bundle count/size expressions

The result is a cleaner small-example safety surface without broad example
churn.

### 5. Hardened the touched auxiliary dead-code tooling paths

Sprint 47 landed bounded support-code hardening in:

- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

That work now gives the touched dead-code surfaces clearer failure behavior for:

- malformed coverage-note metadata
- invalid compile-database shape
- missing required compile-database fields

This is support-code alignment, not a dead-code workflow redesign.

### 6. Refreshed the touched benchmark/example docs to match live behavior

Sprint 47 refreshed:

- `benchmarks/README.md`
- `examples/README.md`

The docs now match the live auxiliary contract around:

- `bench_main --help`
- malformed-input behavior
- conflicting-mode rejection
- intentional `--reorder none|rcm|amd|nd` scope
- handoff to `bench_reorder` / `bench_colamd`
- the `example_alloc_helpers.h` convention
- the full three-part `example_eigs` demo story

This kept the docs aligned with the real runtime surface instead of relying on
implicit maintainer memory.

## Final Validated Baseline

Sprint 47 closes from the Day 13 measured baseline:

- `make format` → passed
- `make lint` → passed
- `make test` → passed
- `make quality-review-full` → passed

Truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`

Direct touched-surface reruns also passed:

- `make tooling-build`
- `./build/bench_main --help`
- `./build/bench_main --reorder nd --size 8 --repeat 1`
- `./build/bench_main --reorder colamd`
- `./build/example_eigs`
- `python3 -m py_compile scripts/deadcode_report.py`
- `bash -n scripts/deadcode_workflow.sh`
- synthetic valid and malformed `deadcode_report.py` checks

## Residual Queue for Later Epic 4 Work

Sprint 47 intentionally does **not** finish every remaining auxiliary-surface
cleanup thread.

The main later inherited queues are now:

- peer benchmark alignment work that stayed intentionally outside the first
  landing:
  - `bench_eigs.c`
  - `bench_iterative_reuse.c`
  - `bench_eigs_reuse.c`
- broader example cleanup surfaces that were not the right first bounded batch:
  - `example_ic_minres.c`
  - `example_analysis.c`
  - `example_condition.c`
- any future broader benchmark framework or CLI unification only if later work
  takes that wider scope on directly
- any future broader dead-code workflow redesign only if later work takes that
  wider support-tooling scope on directly

The main outward-facing non-goals left deliberately untouched are:

- public CLI helper APIs in the core library
- broad benchmark framework redesign
- large README/tutorial restructuring
- dead-code workflow redesign

These are deliberate Sprint 47 boundaries, not regressions or newly discovered
cleanup debt.

## `PROJECT_PLAN.md` Check

Sprint 47 did not surface any new deferred work beyond the benchmark CLI,
auxiliary safety, and example/tooling/doc cleanup queue already implied by the
Epic 4 roadmap.

No `PROJECT_PLAN.md` update was needed at closeout.

## Bottom Line

Sprint 47 leaves behind a cleaner and safer auxiliary surface package:

- shared internal benchmark CLI parsing helpers
- modernized `bench_main` behavior
- reconciled reorder-mode ownership and emitted-label parity
- bounded example safety/helper adoption
- bounded dead-code tooling hardening
- touched docs aligned with live runtime behavior
- validated local reviewed baseline preserved

That is the correct Sprint 47 handoff for later benchmark/example/tooling
cleanup without over-claiming broader framework or public API change.
