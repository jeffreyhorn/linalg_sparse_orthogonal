# Sprint 35 Retrospective

**Sprint:** 35 — Public Docs, Header Examples & API-Usage Consistency  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 34 reviewed-quality baseline treated as the non-regression floor
- [x] installed-header example surface audited and reconciled
- [x] public example-style rule made explicit
- [x] README/tutorial ownership split defined before broad rewrites
- [x] README/tutorial snippets updated to current public API names and behavior
- [x] public precondition and matrix-state wording tightened where it mattered
- [x] INSTALL, examples, benchmark, and quality support docs reconciled
- [x] example/tooling validation passed
- [x] final reviewed Makefile/CMake validation passed
- [x] Sprint 36+ handoff inputs written

## What Went Well

1. **The sprint narrowed quickly to the real public drift instead of
   over-editing.** Day 2 and Day 6 both showed that the repo did not need a
   broad mechanical rewrite everywhere; the highest-value problems were stale
   type names, stale SVD wording, and implicit safety guidance.

2. **The ownership split prevented doc churn.** Deciding early that headers are
   the authoritative API contract, `README.md` is the concise entrypoint, and
   `docs/tutorial.md` is the fuller teaching surface kept the rewrite from
   turning into competing parallel documentation layers.

3. **The sprint preserved the Sprint 34 quality floor throughout.** Even
   though most of Sprint 35 was documentation-facing work, the final day still
   proved that the reviewed Makefile and CMake paths remained green after the
   public-doc changes.

4. **The public precondition cleanup was worth doing.** The biggest remaining
   user-facing ambiguity by mid-sprint was not initializer style anymore; it
   was whether users could tell which matrix state or precondition family each
   routine expected. Tightening that language materially improved the docs.

5. **The example-validation day paid for itself.** Running the concrete example
   binaries before the final full sweep prevented Day 13 from turning into a
   mixed validation/debug session.

## What Didn't Go Well

1. **The original Sprint 35 problem statement was broader than the real
   backlog.** The early plan still assumed there might be a larger leftover
   header-example conversion queue than actually existed. The audit days were
   necessary to collapse that assumption.

2. **Public-doc truthfulness depended on several layers staying aligned at
   once.** Headers, tutorial prose, README snippets, and support docs can drift
   independently. That makes this kind of sprint less about one big patch and
   more about holding a contract across multiple surfaces.

3. **Full validation remains expensive even for docs-heavy work.** Sprint 35
   correctly treated the reviewed wrappers as the authoritative floor, but the
   cost of that floor is still real, especially when the sprint mostly edits
   documentation.

## Final Metrics

### Example and tooling validation

| Metric | Final |
|---|---:|
| `make examples` | passed |
| `make tooling-build` | passed |
| targeted example binaries run | `5` passed |

### Direct maintained gates

| Metric | Day 13 final |
|---|---:|
| `make format` wall time | `5.16 s` |
| `make lint` wall time | `432.50 s` |
| `make test` wall time | `106.16 s` |

### Reviewed wrapper paths

| Metric | Day 13 final |
|---|---:|
| `make quality-review-compile` wall time | `381.24 s` |
| `make quality-review` wall time | `560.29 s` |
| `make quality-review-cmake-compile` wall time | `73.68 s` |
| full `ctest` real time on `build/quality-review-cmake` | `173.23 s` |

### Suite state

| Metric | Day 13 final |
|---|---:|
| `ctest -N` registered tests | `53` |
| full `ctest` result | `53 / 53` passed |
| `test_framework_optin` summary | `8` run / `0` failed / `3` skipped |

## Residual Deferred Debt

Sprint 35 closes without a new cleanup backlog.

Not carried forward as residual debt:

- stale public header example queue: none
- stale README/tutorial type-name queue: none
- known example-build mismatch after the rewrite: none
- Sprint 34 reviewed-quality regression triggered by public-doc updates: none

What remains for later sprints is regression prevention:

- preserve the public-doc ownership split
- keep support docs aligned if later sprints change quality commands or
  platform caveats
- avoid reintroducing stale option-struct, reorder, QR, SVD, or precondition
  guidance while cross-platform and quality-gate work continues

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [HANDOFF.md](./HANDOFF.md)
- [day3-public-initialization-standard.md](./artifacts/day3-public-initialization-standard.md)
- [day4-header-batch1.md](./artifacts/day4-header-batch1.md)
- [day5-header-batch2.md](./artifacts/day5-header-batch2.md)
- [day8-readme-tutorial-implementation.md](./artifacts/day8-readme-tutorial-implementation.md)
- [day10-api-precondition-implementation.md](./artifacts/day10-api-precondition-implementation.md)
- [day11-install-quality-docs-polish.md](./artifacts/day11-install-quality-docs-polish.md)
- [day12-example-build-validation.md](./artifacts/day12-example-build-validation.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)

## Bottom Line

Sprint 35 achieved its goal:

- public headers, README/tutorial docs, and support docs now teach the current
  stable API instead of stale names or implied older behavior
- public examples now follow a clear designated-initializer rule for non-
  default option-struct usage
- QR, SVD, and precondition guidance is materially more truthful than the
  pre-sprint state
- the Sprint 34 reviewed Makefile/CMake quality baseline stayed green
  throughout

Sprint 36 should treat Sprint 35 as a stable public-doc baseline and carry the
remaining cross-platform quality expansion forward without reopening a solved
documentation cleanup queue.
