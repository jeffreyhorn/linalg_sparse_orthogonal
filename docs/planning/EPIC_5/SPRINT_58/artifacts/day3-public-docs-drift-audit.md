# Sprint 58 Day 3 - public docs drift audit

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Reduce the strongest caller-facing docs to a concrete simplification map by
separating their live drift classes, ranking the real bounded cleanup targets,
and fixing the strongest first documentation landing boundary before any
permanent wording changes land.

## Audited files

Primary caller-facing docs surfaces:

- `README.md` = `987`
- `docs/tutorial.md` = `415`
- `examples/README.md` = `134`
- `benchmarks/README.md` = `235`

These were audited using:

- direct prose reads of the high-signal sections
- `rg` scans for sprint-history, workflow, repeated-run, factor-many,
  benchmark, and example-local terminology
- comparison against the validated Sprint 52-57 public workflow fence

## Drift shapes

### `README.md`

Live drift classes:

- stale sprint chronology
- repeated-run workflow ambiguity
- one-shot versus advanced-path imbalance
- example coverage mismatch
- benchmark taxonomy mismatch

Key fact:

- this is still the strongest public surface in the repo, but it remains the
  most crowded mix of stable workflow guidance and historical implementation
  narrative

### `docs/tutorial.md`

Live drift classes:

- repeated-run workflow ambiguity
- one-shot versus advanced-path imbalance
- mild terminology drift against the final repeated-run public story

Key fact:

- the tutorial is already comparatively stable and does not need a broad
  rewrite, only bounded reduction and terminology alignment

### `examples/README.md`

Live drift classes:

- stale sprint chronology
- repeated-run workflow ambiguity
- one-shot versus advanced-path support-boundary overexplanation

Key fact:

- the example inventory is already fairly aligned to the shipped surfaces, so
  the main cleanup need is product-level wording rather than structural change

### `benchmarks/README.md`

Live drift classes:

- stale sprint chronology
- benchmark taxonomy mismatch
- repeated-run workflow ambiguity
- one-shot versus advanced-path imbalance in a few sections

Key fact:

- stable workflow groups are already visible, but they still coexist with older
  sprint-stamped benchmark framing

## Ranked cleanup order

### Rank 1: `README.md`

Why it ranks first:

- highest caller visibility
- largest overlap of drift classes
- strongest opportunity to reduce sprint-local wording and make the final
  workflow story easier to scan

Likely first owned seam:

- top-level workflow guidance, one-shot versus repeated-run framing, and
  benchmark/example summary wording

### Rank 2: `docs/tutorial.md`

Why it ranks second:

- naturally pairs with the README cleanup
- comparatively low risk
- can absorb concise final workflow guidance without reopening product design

Why it does not rank first:

- lower visibility than README
- less sprint-history drift to remove directly

### Rank 3: `benchmarks/README.md`

Why it ranks third:

- high-value public taxonomy surface
- depends on the README/tutorial wording settling first
- needs workflow-group cleanup more than generic trimming

### Rank 4: `examples/README.md`

Why it ranks fourth:

- smaller and already closer to the intended one-shot-first posture
- still worthwhile, but easier to finish after the top-level language is fixed

## Rejected first moves

Rejected as the first Sprint 58 docs landing:

- benchmark taxonomy cleanup first
  - too dependent on final top-level terminology
- example README cleanup first
  - lower caller visibility and lower payoff than the README/tutorial pair
- broad tutorial rewrite
  - expansion-prone and out of scope for a simplification sprint

## Recommended landing order

1. Design and land a bounded `README.md` + `docs/tutorial.md` reduction
   boundary
2. Follow with benchmark taxonomy cleanup in `benchmarks/README.md`
3. Finish with example-doc alignment in `examples/README.md`

## Conclusion

The public-docs problem is now concrete:

- the first Sprint 58 target should be `README.md`
- the first boundary should pair README reduction with bounded tutorial
  alignment
- `benchmarks/README.md` is the strongest later taxonomy target
- `examples/README.md` is a smaller but still meaningful later cleanup target

That gives Day 4 a clear starting point for the first bounded docs-reduction
design.
