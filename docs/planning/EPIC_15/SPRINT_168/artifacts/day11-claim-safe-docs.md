# Sprint 168 Day 11: Claim-Safe Documentation Update

## Purpose

Day 11 updates user-facing and maintainer-facing documentation for the selected
hosted performance lane added on Day 10. The documentation describes only the
freshness and methodology contract for the selected `bench_refactor_csc` row on
`nos4.mtx --repeat 1`.

The update does not claim portable performance, timing regression protection,
external-library parity, broad benchmark publication, broad platform support,
package/ABI support, release proof, or state-of-the-art sparse linear algebra
performance.

## Documentation Updates

### README

Updated `README.md` to:

- add `make bench-canonical-report-freshness` to the "when to widen beyond the
  first examples" guidance;
- add the target to the local command list;
- explain that the reviewed Linux hosted selected-performance lane runs only
  the selected `bench_refactor_csc` canonical row for `nos4.mtx --repeat 1`;
- state that the hosted lane checks artifact presence, selected row identity,
  methodology metadata, manifest agreement, and
  `hosted_selected_threshold_free` claim boundaries;
- explicitly reject timing comparisons, regression thresholds, portable speed,
  unselected canonical row promotion, external-library evidence, package/ABI
  evidence, broad platform evidence, release evidence, and state-of-the-art
  evidence.

### Benchmarks README

Updated `benchmarks/README.md` to:

- add `make bench-canonical-report-freshness` to the recommended handoff table;
- document the new canonical report metadata fields:
  - runner context;
  - build flags;
  - CPU model;
- define the selected freshness scope:
  - `artifact=bench_refactor_csc`;
  - `relative_path=bench_refactor_csc.csv`;
  - `command=tests/data/suitesparse/nos4.mtx --repeat 1`;
  - `fixture_or_workload=nos4.mtx`;
  - `repeat_semantics=configured_repeat_1`;
  - `baseline=n/a`;
  - `threshold=n/a`;
- explain hosted mode requirements for `hosted_selected`,
  `hosted_selected_threshold_free`, non-local runner context, recorded build
  flags, and non-`unlabeled` report label;
- keep `cpu_model=unknown` acceptable for hosted runners because CPU
  assignment can vary;
- add the freshness target to the report-index handoff table;
- clarify that selected performance freshness remains outside broad
  normalized-index promotion and does not make `bench_chol_csc`,
  `bench_iterative_reuse`, or `bench_eigs_reuse` reviewed hosted performance
  evidence.

### Maintainer Guide

Updated `docs/maintainer_guide.md` to:

- include runner context, build flags, and CPU model in canonical report
  metadata ownership;
- add `make bench-canonical-report-freshness` to the threshold-free reporting
  surface;
- document selected artifact, schema, metadata, threshold-free baseline and
  threshold, methodology-note, and manifest-agreement checks;
- add the target to the common focused checks list;
- explain the reviewed Linux hosted selected-performance lane and retained
  non-claims.

## Claim Scan

Ran a targeted claim scan across the updated docs and Sprint 168 artifacts:

```sh
rg -n "state[- ]of[- ]the[- ]art|portable performance|performance guarantee|superiority|external-library parity|broad benchmark|timing threshold|regression threshold|hosted selected-performance|bench-canonical-report-freshness|hosted_selected_threshold_free" \
  README.md benchmarks/README.md docs/maintainer_guide.md docs/planning/EPIC_15/SPRINT_168
```

Result: the new selected-performance references are present, and risky terms
appear in explicit non-claim or retained-boundary wording. No new broad
performance, superiority, external parity, release, or state-of-the-art claim
was introduced.

## Local Versus Hosted Distinction

The docs now distinguish:

| Surface | Meaning |
| --- | --- |
| `make bench-canonical-report` | Local threshold-free canonical report bundle. |
| `make bench-canonical-report-freshness` | Local selected-row freshness and methodology check. |
| Hosted selected-performance CI lane | Reviewed Linux hosted selected-row freshness check with hosted metadata. |
| Other canonical CSV rows | Uploaded as bundle context only; not promoted as reviewed hosted selected performance evidence. |

## Quality Gate

Day 11 changed documentation and planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate (`make format && make lint && make test`)
is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Docs describe only the selected hosted performance scope. | Complete | README, benchmarks docs, and maintainer guide name only `bench_refactor_csc` on `nos4.mtx --repeat 1` as selected hosted performance freshness evidence. |
| Local benchmark rows remain distinct from hosted evidence. | Complete | Docs keep `make bench-canonical-report` local/threshold-free and describe hosted mode as selected-row freshness only. |
| No broad performance or state-of-the-art claims are introduced. | Complete | Targeted claim scan found risky terms only in explicit non-claim or boundary language. |
