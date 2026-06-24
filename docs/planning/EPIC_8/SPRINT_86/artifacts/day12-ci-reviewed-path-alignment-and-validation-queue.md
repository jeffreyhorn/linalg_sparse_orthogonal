# Sprint 86 Day 12: CI / Reviewed-Path Alignment & Validation Queue Freeze

## Purpose

Reconcile the touched Sprint 86 reviewed-path, runtime-lane, and support
surfaces after the Day 6 ND runtime landing, Day 9 proof-owner rebalance, and
Day 11 benchmark follow-through, then freeze the exact Day 13 validation
queue.

## Main Result

No new support-only edit is needed before the full validation sweep.

The final Sprint 86 touched-surface truth map is now explicit:

- adopted runtime/scalability centers:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_nd_internal.h`
  - `src/sparse_graph.c`
  - `tests/test_reorder_nd.c`
  - `benchmarks/bench_reorder.c`
  - `Makefile`
  - `benchmarks/README.md`
- retained reviewed proof owners, not Sprint 86 adopted runtime centers:
  - `tests/test_reorder.c`
  - `tests/test_reorder_amd_qg.c`
  - `tests/test_graph.c`
- support-only surfaces that do not need new movement before Day 13:
  - `docs/maintainer_guide.md`
  - `README.md`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `scripts/bench_canonical_report.sh`

## Final Reviewed / Runtime Map

The final Sprint 86 reviewed-path and runtime-evidence split is now fixed:

- reviewed ND runtime/proof owner:
  - `tests/test_reorder_nd.c`
- adjacent reorder/graph reviewed proof owners:
  - `tests/test_reorder.c`
  - `tests/test_reorder_amd_qg.c`
  - `tests/test_graph.c`
- representative reviewed examples:
  - `example_analysis`
  - `example_basic_solve`
- branch-local runtime evidence owner:
  - `benchmarks/bench_reorder.c`
  - surfaced through:
    - `make bench-reorder-sprint86`
- canonical maintained benchmark/reporting owner:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`

## CI / Reviewed-Path Outcome

The CI and reviewed-path reading stayed fixed:

- the strongest local reviewed baseline remains:
  - `make quality-review-full`
- reviewed CMake parity remains the strongest explicit truth anchor:
  - `ctest -N --test-dir build/quality-review-cmake`
- `make bench-reorder-sprint86` is a bounded local/runtime evidence surface,
  not part of the maintained reviewed baseline and not a new CI timing gate
- no workflow file needs Sprint 86 wording movement before validation

## Support-Surface Boundary

The support-surface reading stayed fixed:

- `docs/maintainer_guide.md` and `README.md` already remain truthful about the
  reviewed baseline, the benchmark-owner split, and the retained package
  contract
- canonical maintained reporting remains command/script-owned through:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
- install/export proof remains script-owned through:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

Sprint 86 did not reopen install, export, package metadata, or consumer
mechanics, so those surfaces remain out of the Day 13 validation core.

## Frozen Day 13 Queue

The exact Day 13 queue is now fixed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- focused reviewed proof owners:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
- representative examples:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- benchmark/reporting follow-ons:
  - `make bench-reorder-sprint86`
  - `make bench-canonical-report`

## Strongest Clarification

The useful Day 12 clarification is explicit now:

- Sprint 86 does not need another proof-owner, workflow, or docs batch before
  validation
- the only authoritative correctness owners for the landed runtime package are
  the retained reviewed reorder/graph proof-owner tests and examples already
  fixed above
- the Day 11 Sprint 86 rerun slice remains runtime evidence only and does not
  redefine the maintained reviewed baseline or the canonical benchmark face

## Exit State

- no support-only drift remains before Day 13
- the reviewed-path and runtime-evidence split is explicit and unambiguous
- Day 13 can execute from a fixed touched-surface truth map rather than
  re-deciding Sprint 86 scope
