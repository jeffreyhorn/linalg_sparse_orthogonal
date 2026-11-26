# Sprint 228 Day 14: Retrospective Input

## What completed

- Promoted the previously optional per-cell sparse low-rank reconstruction to
  the maintained default without changing public ABI.
- Preserved an exact-`off` dense compatibility route and made malformed
  selector values fail closed.
- Closed numerical, threshold, ownership, allocation-failure, retry, width,
  sanitizer, leak, build-registration, and documentation gaps with named
  regression evidence.
- Added exact component counters and a reproducible raw packet proving the
  default adds no `m*n` reconstruction accumulator.
- Completed the required Make and CMake full-suite closeout with exact test
  inventory reconciliation.

## What worked well

- Freezing behavior and claim boundaries before promotion prevented memory
  optimization from silently changing threshold or error semantics.
- Independent analytical and long-double-factor oracles avoided circular
  route-to-route validation.
- Exhaustive fail-index sweeps exposed ownership transitions more clearly than
  a small set of hand-picked allocation failures.
- Component counters separated removed reconstruction scratch from retained
  dense factors, decomposition work, sparse metadata, and output nodes.
- Retaining failed timing evidence made the final claim smaller but auditable.

## What did not close

- The frozen timing gate failed for square and wide workloads, so no speed or
  promotion-level timing claim is available.
- The memory packet used a documented dirty-tree development override. Its raw
  memory result is useful for sprint completion, but a release-oriented claim
  would require clean exact-candidate recollection and review.
- Full SVD's dense factors and decomposition work remain by design; Sprint 228
  removed only the extra reconstruction accumulator.

## Follow-up guidance

1. Keep timing work separate from the accepted reconstruction-memory result.
2. If performance is revisited, profile per-cell arithmetic and insertion on a
   clean exact candidate without weakening the frozen correctness corpus.
3. Preserve component accounting and raw receipts in future memory work.
4. Bind the final review packet to the eventual sprint commit SHA.
5. Continue using claim tables that state both the accepted result and the
   explicitly unsupported broader interpretations.
