# Sprint 76 Day 4 Artifact: First Governance Boundary

Date: 2026-06-17
Branch: sprint-76

## Purpose

Freeze the first Sprint 76 reporting/governance fence so the next design pass
starts from one bounded longitudinal-reporting lane rather than from a mixed
benchmark-reporting, threshold, and documentation backlog.

## Main Result

Sprint 76 now has one explicit first landing boundary:

- required first landing:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
- support only if the first landing forces it:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `README.md`
- explicitly deferred:
  - canonical benchmark driver sources
  - reviewed proof-owner tests and examples
  - runtime-threshold surfaces:
    - `bench-fast`
    - `wall-check`
    - `bench_reorder`
    - `bench_amd_qg`

## Why This Is the Right First Fence

The canonical reporting workflow and schema lane remains the best first
landing because it already has the strongest bounded governance shape:

- one explicit threshold-free command surface:
  - `make bench-canonical-report`
- one real script owner:
  - `scripts/bench_canonical_report.sh`
- one already-small canonical maintained benchmark surface:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- one existing truthfulness fence against portability overclaim

That gives Sprint 76 the strongest combination of:

- reporting leverage
- low compatibility risk
- manageable proof cost
- bounded payoff without widening the benchmark claim surface

## Support Surface Reading

The support surfaces are bounded rather than assumed:

- `benchmarks/README.md`
  - move only if the first batch changes the benchmark-local interpretation of
    canonical reporting, role separation, or longitudinal comparison reading
- `docs/maintainer_guide.md`
  - move only if the first batch makes the canonical/runtime/exploratory policy
    or report-schema ownership clearer in a way the policy surface should
    capture
- `README.md`
  - move only if the compact front-door summary truly becomes inaccurate after
    the first batch

## Explicit Deferred Set

The Day 4 deferred set is now fixed:

- canonical benchmark driver churn:
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- broader runtime or exploratory benchmark work:
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
- threshold-policy lane:
  - `bench-fast`
  - `wall-check`
- proof-owner and adoption surfaces:
  - reviewed tests
  - examples
- broad docs/governance spill beyond support follow-through

## Non-Goal Fence

The first Sprint 76 batch explicitly does not include:

- widening the canonical maintained benchmark surface
- inventing a new timing-threshold gate under the name of longitudinal
  reporting
- broad benchmark-driver rewrites
- widening product, platform, or backend claims beyond maintained evidence
- reopening Sprint 75 backend architecture or Sprint 74 capability work

## Day 5 Implication

The Day 5 design pass should therefore start from:

- exact first implementation center:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
- support only if truly forced:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `README.md`
- explicitly not next:
  - canonical benchmark driver edits
  - threshold-policy widening
  - runtime-lane expansion
  - broad benchmark/docs cleanup
