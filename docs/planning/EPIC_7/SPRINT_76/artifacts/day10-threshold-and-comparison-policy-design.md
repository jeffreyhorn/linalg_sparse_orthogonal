# Sprint 76 Day 10 Artifact: Threshold and Comparison Policy Design

Date: 2026-06-17
Branch: sprint-76

## Purpose

Define the narrow threshold and comparison policy Sprint 76 should actually
preserve after the Day 6 and Day 9 landings, rather than assuming the sprint
still needs a new threshold batch simply because it was originally queued.

## Main Result

The strongest Day 10 conclusion is now explicit:

- Sprint 76 does not need a new threshold or comparison-policy batch
- it needs the already-landed no-threshold and bounded-runtime reading
  preserved in writing

## Retained Policy

The current benchmark-governance policy is now fixed explicitly:

### Canonical maintained reporting

- `make bench-canonical-report`
  - threshold-free canonical reporting surface
  - comparison aid only
  - not a pass/fail timing gate
  - not a portability or machine-ranking claim

### Bounded runtime lane

- `bench-fast`
  - bounded runtime lane
  - useful current-branch signal
  - not canonical maintained proof

### Narrow thresholded regression gate

- `wall-check`
  - already-justified machine-class thresholded regression gate
  - intentionally narrower than the canonical report surface
  - not a portable performance claim

### Runtime/reporting context only

- `bench_reorder`
- `bench_amd_qg`

These remain runtime and reporting context only. They are not silently
promotable into canonical maintained truth.

## Why No New Policy Batch Is Needed

After the Day 6 and Day 9 landings, the strongest possible new threshold work
would be lower value and higher risk than preserving the current bounded
reading:

- canonical reporting is already clearer and more comparable
- support-surface drift is already closed
- the current thresholded lane is already intentionally narrow
- broader timing gates would increase portability overclaim risk without
  enough stability evidence

So the stronger design decision is:

- preserve the explicit no-threshold policy on the canonical surface
- preserve the existing narrow threshold only where already justified
- do not add historical-diff verdict logic, comparison bands, or new timing
  gates in Sprint 76

## Preserved Fence

Sprint 76 must preserve:

- no portable pass/fail timing claims
- no unstable thresholding on noisy surfaces
- no silent promotion of exploratory or runtime-lane surfaces into canonical
  maintained truth
- no widened state-of-the-art claim detached from retained measured evidence

## Day 11 Implication

No bounded Day 11 threshold-policy landing is currently required.

Day 11 should therefore act as a bounded recheck:

- verify that the current benchmark-local, maintainer-policy, top-level, and
  workflow wording still agree with the retained policy above
- avoid forcing an unnecessary policy batch just to satisfy the original
  backlog shape

## Exit State

Sprint 76 now has one explicit preserved threshold/comparison policy:

- canonical reporting stays threshold-free
- `bench-fast` stays the bounded runtime lane
- `wall-check` stays the narrow thresholded regression gate
- no new threshold machinery is required for this sprint
