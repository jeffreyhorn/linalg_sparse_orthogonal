# Day 14 Retrospective Input

## Planned versus delivered

All six Sprint 227 items closed. The sprint selected one explicit original-state rejection policy,
replaced repeated output insertion in out-of-place addition, and treated failure behavior and
evidence boundaries as first-class implementation requirements rather than cleanup work.

## What worked

- Test-first nonidentity fixtures exposed misleading success before production changes.
- One shared guard kept all three arithmetic APIs coherent.
- Direct row/column-tail construction made structural invariants and complexity inspectable.
- Allocation injection plus exact topology snapshots made ownership and partial-failure behavior
  executable rather than descriptive.
- Pre-registering profiles before collection prevented selective reruns and claim drift.
- Separating raw evidence, derived samples, source hashes, and claim language made Day 13–14
  reconciliation straightforward.

## Friction and lessons

- In-place addition cannot inherit the out-of-place linear bound; the distinction must remain
  explicit in future documentation and reviews.
- Archive hashes are not sufficient binary-equivalence evidence when the archiver embeds member
  timestamps; object and executable comparisons were needed.
- Full lint and dual Make/CMake suites are materially slower than focused loops, so early focused
  checks and a dedicated final closeout day remain worthwhile.
- A benchmark can support local kernel-time evidence while offering no useful memory conclusion;
  metric boundaries should be frozen independently.

## Follow-up ownership

- Sprint 228 owns low-rank reconstruction output and memory, not further addition optimization.
- A future arithmetic-view project may add logical permuted-state support; until then rejection is
  the accepted contract.
- Independent timing replication and allocation-level memory accounting are optional future
  evidence work, not unfinished Sprint 227 acceptance criteria.
