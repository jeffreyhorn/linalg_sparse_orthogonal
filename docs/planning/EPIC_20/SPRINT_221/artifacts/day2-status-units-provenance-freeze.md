# Sprint 221 Day 2: Status, Units And Provenance Freeze

## V1 Record Rules

This is the accepted Sprint 215 Day 6-8 vocabulary made explicit for v1;
the declarative field shapes live in
`tests/corpus/schemas/external_baseline_evidence_v1.json`. One record is
one blueprint or one append-only generated attempt. Required envelope:
schema version, record/selection IDs, kind, origin, status/reason, claim
scope/non-claims, publication review, cell, methodology, environment (null
only with reason before execution), both correctness sides, timing and
artifacts. A complete `cell_id` has no reason; a null `cell_id` has a
nonempty reason. Generated runs require a nonempty `run_id`; blueprints
require `run_id=null`. Optional `block_gate`, `mismatch_field`, publication
review artifact, correctness consistency metrics and per-window timing
details are permitted only when their status needs them.

No unknown fields or duplicate JSON keys. Types are exact: boolean is not
integer; counts are nonnegative integers except observed thread counts,
which are positive; numeric metrics must be finite; hashes are exactly 64
lowercase hex; generated project Git, matrix-origin and baseline source
commits are full 40-character lowercase hex. The planning blueprint's
`aa1c04ea` matrix-origin abbreviation is not acceptable for a generated
record. UTC timestamps use a real calendar date/time
with `Z`, not a local time or an unparsed pattern match. Empty strings and
`unknown` masquerading as observed provenance are invalid. A null runtime,
metric or artifact field requires a named availability reason; `0` is a real
value, never a missing-data marker.

## Status Matrix

| Status | Positive v1 shape | Contradictory negative shape / rule code |
| --- | --- | --- |
| `planned` | Blueprint/planning, null run/cell IDs, both sides and timing `not_run`, no generated artifacts | Any solution, residual or sample: `status_contradiction` |
| `not_run` | Generated run ID and named reason, neither side executed, timing `not_run` | Prepared directory labeled a pass: `status_contradiction` |
| `blocked` | Named `block_gate=execution|publication` and missing prerequisite; publication block may retain separately validated local observations | Execution block with solver result or any valid timing/performance verdict: `blocked_cannot_claim_performance` |
| `non_comparable` | Named changed Day 3 identity field or inapplicable input; no paired verdict | Reused selected complete ID or ranked result: `cell_id_mismatch` or `unearned_claim` |
| `correctness_failed` | Complete eligible cell, at least one `fail` with raw status/metric evidence; timing `not_run|invalid` | No failure evidence or timing `measured`: `status_contradiction` |
| `correctness_observed` | Complete identity, two independently validated `pass` sides and hashed raw outputs; timing `not_run` | Missing pass, artifact or full provenance: `missing_provenance` or `status_contradiction` |
| `timing_invalid` | Complete identity and two initial correctness passes, named timing failure and preserved attempt evidence | Dropped failed trial or valid speed summary: `unit_or_sample_contract` |
| `measured` | Complete identity, two passes, verified host/threads/clock, complete paired raw samples and recomputed per-side statistics | One blocked/failed side or missing samples: `measured_requires_two_passes_and_raw_samples` |

The selected Sprint 215 pilot is **correctness-only**. The two timing states
above define a future v1 capability; they do not authorize a timing attempt
or a timing claim for this selection.

## Units, Raw Values And Sample Contract

`residual_inf=||A*x-b||_inf` is in RHS units;
`oracle_error_inf=||x-x_ref||_inf` is in solution units;
`backward_error=residual_inf/(||A||_inf*||x||_inf+||b||_inf)` is
dimensionless. Each side independently passes Day 4's predeclared limits
with `n=4`, `u=DBL_EPSILON`, `T=32*n*u`, `||A||_inf=17`,
`||b||_inf=17`, and condition bound `K=17/7`. Retain unrounded raw `x` and
metrics; recompute thresholds from these values, then check cross-solver
solution difference against the sum of the independent oracle limits.
Success exit status alone is not a pass.

Timing durations are positive integer `ns`. For each reported phase, retain
five paired warmups and 50 paired timed trials, alternating AB/BA order,
including per-trial statuses and both sides' raw durations. Per-side
minimum, maximum, median, Q1, Q3, IQR and MAD are finite nanosecond
statistics and may be fractional; record quartile convention and verify
statistics from raw samples. No trimming, post-hoc exclusions, milliseconds
under `_ns` keys or zero for unavailable timing. A median below 100 times
observed clock resolution is `timing_invalid`/resolution-limited, not a
performance observation.

## Provenance, Paths And Claims

Generated observations require actual matrix bytes; complete project
Git/worktree/patch/build/compiler/binary identity; actual loaded CHOLMOD
library, package/module/build, ABI and transitive dependency path/hash
inventory; verified requested and observed threads; host OS/build/arch/CPU,
command/cwd/UTC time; and all Day 3/4/5 methodology hashes. Hosted origin
also requires workflow/run/artifact IDs. An execution-blocked attempt may
leave unobserved runtime fields null with reasons, never fabricated.

Artifact references are safe relative paths beneath the declared repository
or bundle root. Reject absolute paths, traversal, symlink escapes, missing
files and SHA mismatches. Every raw input/log/solution/sample/report used by
an observed claim has a role, media type and hash. Local validation does not
authorize public claims: publication review is separately
`pending|approved|denied`, and `approved` requires a review artifact.
Forbidden v1 keys/claims include speedup, winner, hosted threshold, portable
speed, cross-platform timing parity and state-of-the-art status. The
selected pilot cannot assert any of these even if a future local record
passes correctness checks.

## Rule Codes And Gate

The schema's `x-v1-policy.rule_codes` is the stable namespace. Tests should
assert verdict and code, including `selection_mismatch`,
`missing_provenance`, `cell_id_mismatch`, `status_contradiction`,
`unit_or_sample_contract`, `artifact_path_or_hash`, `unearned_claim` and
`duplicate_or_unknown_field`. Error output must identify the offending
record and JSON pointer. Day 3 assigns canonical identity bytes and
schema authority; Day 4-8 must make these semantic checks executable.
Until then, the pilot validator self-test stays fail-closed.
