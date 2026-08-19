# Sprint 169 Day 3: Statistical Policy Design

## Purpose

Day 3 defines the statistical policy for the selected performance publication
lane before schema or implementation changes. The policy keeps Sprint 168's
selected hosted evidence narrow while giving every statistical field a clear
meaning for Days 4 through 6.

## Selected Lane Reminder

The policy applies only to the selected performance row:

| Field | Selected lane |
| --- | --- |
| Artifact | `bench_refactor_csc` |
| Command | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Fixture | `nos4.mtx` |
| Local target | `make bench-canonical-report-freshness` |
| Hosted job | `Linux reviewed hosted selected performance freshness` |
| Hosted claim boundary | `hosted_selected_threshold_free` |

Unselected canonical rows remain local/advisory and must not inherit selected
hosted support or claim metadata.

## Policy Decision Summary

| Topic | Decision | Rationale |
| --- | --- | --- |
| Repeat count | Preserve `--repeat 1` for the selected hosted publication lane during Sprint 169. | The Sprint 168 lane was selected because it is cheap and stable enough for hosted CI. Increasing repeats before schema hardening would mix runtime-risk work with methodology cleanup. |
| Sample count | Treat the selected publication row as a single configured sample. | Current benchmark output is one generated report row per command invocation, so variance cannot be inferred honestly. |
| Warmup | Replace ambiguous `not_recorded` with an explicit no-warmup policy in schema design. | The selected command does not run a separate warmup phase. Recording that directly is more interpretable than saying the state is unknown. |
| Variance | Replace ambiguous `not_recorded` with an explicit single-sample variance policy in schema design. | With one configured sample, variance is not computed. This should be explicit and should not imply confidence intervals. |
| Threshold | Keep selected publication rows threshold-free: `baseline=n/a`, `threshold=n/a`. | The selected report is methodology evidence, not a pass/fail timing gate. |
| Regression sentinel | Design separately on Days 7 and 8. | Any pass/fail timing behavior needs its own baseline provenance, runtime budget, machine-class caveat, and failure output. |
| Local versus hosted policy | Use the same statistical semantics locally and in hosted CI. | Local/hosted differ by support tier, claim boundary, runner context, build flags, and CPU metadata, not by statistical interpretation. |

## Repeat-Count Policy

The selected publication row remains:

```text
command=tests/data/suitesparse/nos4.mtx --repeat 1
repeat_semantics=configured_repeat_1
```

Interpretation:

- the row records one configured benchmark command invocation;
- the benchmark binary may perform its own internal timing as implemented by
  the benchmark source, but the publication schema treats the canonical row as
  one selected report observation;
- maintainers may compare selected rows across branches or runs, but a single
  row is not statistical proof of portable speed;
- any future increase in repeat count must update command identity,
  `repeat_semantics`, freshness checks, hosted runtime expectations, and docs
  together.

Day 4 should design an explicit field or value that makes this single-sample
policy machine-readable without changing the selected command unexpectedly.

## Warmup Policy

Current state:

```text
warmup=not_recorded
```

Sprint 169 policy:

- the selected publication row should record that no separate warmup phase is
  configured by the canonical report lane;
- the preferred normalized value for implementation is
  `warmup=none_configured`;
- docs should state that no warmup phase means the row is suitable for
  threshold-free comparison context only, not stable latency claims;
- if a future benchmark adds warmup iterations, the value must include the
  configured warmup count or policy name.

This policy avoids the ambiguity of `not_recorded`: the absence of warmup is
intentional and visible.

## Variance Policy

Current state:

```text
variance=not_recorded
```

Sprint 169 policy:

- the selected publication row should record that variance is not computed
  because the hosted publication lane uses one configured report observation;
- the preferred normalized value for implementation is
  `variance=not_computed_single_sample`;
- no documentation should describe the selected row as having confidence
  intervals, representative timing distribution, or statistical significance;
- if a future sprint adds repeated samples, variance must be computed from a
  documented sample set and the schema should name the statistic.

This keeps the row honest while making it easier for reviewers to distinguish
known single-sample limitations from missing metadata.

## Threshold Policy

Selected publication rows remain threshold-free:

```text
baseline=n/a
threshold=n/a
claim_boundary=hosted_selected_threshold_free
```

Allowed interpretation:

- selected hosted report freshness;
- selected command, fixture, row, and metadata currency;
- before/after inspection context for maintainers.

Disallowed interpretation:

- timing regression gate;
- portable performance superiority;
- external-library parity;
- broad benchmark-family publication;
- platform parity;
- state-of-the-art performance evidence.

## Regression Sentinel Boundary

Sprint 169 may still add or tighten a bounded regression sentinel, but it must
remain separate from the selected publication row.

Required sentinel properties if implemented:

- separate target or clearly separate output section;
- explicit baseline provenance;
- bounded fixture and command;
- machine-class or hosted-runner caveat;
- clear failure output;
- no external-library, portable-speed, or state-of-the-art wording;
- no mutation of the selected publication row's `baseline=n/a` and
  `threshold=n/a` values.

If those properties cannot be satisfied within Sprint 169, the sentinel should
be explicitly deferred rather than merged as ambiguous performance evidence.

## Local Versus Hosted Statistical Semantics

| Dimension | Local mode | Hosted mode |
| --- | --- | --- |
| selected command | `nos4.mtx --repeat 1` | `nos4.mtx --repeat 1` |
| repeat semantics | `configured_repeat_1` | `configured_repeat_1` |
| sample interpretation | single configured report observation | single configured report observation |
| warmup policy | no separate warmup configured | no separate warmup configured |
| variance policy | not computed for single sample | not computed for single sample |
| threshold policy | threshold-free | threshold-free |
| support tier | `local_only` or hosted-style dry run | `hosted_selected` |
| claim boundary | `local_threshold_free` or hosted-style dry run | `hosted_selected_threshold_free` |

The methodology does not become more statistical merely because it runs in
hosted CI. Hosted CI gives environment metadata, repeatable automation, and
artifact freshness; it does not create variance, confidence intervals, or
portable speed guarantees.

## Day 4 Schema Inputs

Day 4 should design schema changes that make these policy decisions explicit:

| Current field/value | Preferred normalized direction |
| --- | --- |
| `repeat_semantics=configured_repeat_1` | keep, and decide whether a separate `sample_count` value is needed |
| `warmup=not_recorded` | `warmup=none_configured` |
| `variance=not_recorded` | `variance=not_computed_single_sample` |
| `baseline=n/a` | keep |
| `threshold=n/a` | keep |
| `matrix_size=not_recorded` | decide on Day 4 after fixture/dimension review |

## Day 3 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every statistical field has a documented meaning. | Complete | Repeat, sample, warmup, variance, threshold, and sentinel semantics are defined above. |
| Threshold-free publication remains distinct from any regression sentinel. | Complete | Publication rows keep `baseline=n/a` and `threshold=n/a`; sentinel design is separate. |
| Policy avoids broad performance superiority claims. | Complete | Disallowed interpretations and retained non-claims are explicit. |

## Validation

Day 3 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Run after writing this artifact:

```sh
git diff --check
```
