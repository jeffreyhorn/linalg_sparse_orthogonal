# Sprint 33 Retrospective

**Sprint:** 33 — Dead-Code Detection Infrastructure & First Cleanup Pass  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Dead-code baseline captured from the Sprint 32 clean-state handoff
- [x] Dead-code policy and limitations documented before cleanup edits
- [x] `make deadcode` implemented
- [x] `make deadcode-report` and `make deadcode-check` implemented
- [x] Candidate public-surface findings audited separately from internal cleanup
- [x] First cleanup-ready definitely-unused internal candidate removed
- [x] Maintainer-facing dead-code docs written
- [x] Final validation passed from Makefile, CTest, and dead-code paths
- [x] Sprint 34+ handoff inputs written

## What Went Well

1. **The sprint did the policy work before the removals.** That prevented the
   new dead-code tooling from collapsing active code, historical evidence,
   exported APIs, and true internal cleanup candidates into one bucket.

2. **The reporting layer narrowed the queue quickly.** Once `deadcode-report`
   existed, the project stopped reasoning from raw scanner noise and started
   reasoning from explicit buckets: coverage gaps, internal candidates, public
   review items, secondary signals, and static-analysis noise.

3. **The public-surface audit prevented a bad cleanup instinct.** Sprint 33
   correctly kept `givens_apply_right` and the `sparse_print_*` helpers because
   installed-header reachability and maintained operator/docs surface matter
   more than absence of local call sites.

4. **The first cleanup pass stayed small and truthful.** The sprint removed one
   real internal candidate instead of manufacturing a larger deletion batch out
   of weak evidence.

5. **The sprint preserved the Sprint 32 truthfulness model.** `test_framework_optin`
   stayed live, `ctest -N` stayed auditable at `53`, and no dormant commented-out
   test queue was reintroduced while adding dead-code machinery.

## What Didn't Go Well

1. **The dead-code workflow still has real coverage gaps.** `xunused` is only
   as strong as the compilation database it sees, and Sprint 33 closed with
   `bench_svd` plus six examples still outside that surface.

2. **The workflow is not concurrency-safe yet.** The shared
   `build/deadcode-cmake` and `build/deadcode/` paths are fine for serialized
   local use, but they are not a clean foundation for careless parallel
   invocation.

3. **`cppcheck` remains high-noise for this purpose.** The report structure made
   that manageable, but the tool still produces a large supporting-signal set
   that is not deletion-ready by itself.

4. **The full validation path is expensive.** `make lint`, full CTest, and the
   dead-code reruns all pass, but they remain heavy enough that future quality
   enforcement work should be deliberate about cost and execution model.

## Final Metrics

### Dead-code report buckets

| Bucket | First classified report | Day 13 final | Delta |
|---|---:|---:|---:|
| `coverage-gap` | 7 | 7 | 0 |
| `definitely-unused-internal-candidate` | 1 | 0 | -1 |
| `public-surface-review` | 4 | 4 | 0 |
| `secondary-candidate-signal` | 35 | 35 | 0 |
| `non-deadcode-static-analysis-noise` | 6 | 6 | 0 |

### Cleanup-ready queue

| Metric | Day 7 first report | Day 13 final | Delta |
|---|---:|---:|---:|
| Approved internal cleanup candidates | 1 | 0 | -1 |
| Audited public keeps | 0 | 4 | +4 |

### Validation state

| Metric | Day 13 final |
|---|---:|
| `make lint` wall time | `473.25 s` |
| `make test` wall time | `132.22 s` |
| serialized Sprint 33 CMake rebuild wall time | `53.22 s` |
| `ctest -N` registered tests | `53` |
| full `ctest` real time | `178.31 s` |
| `make deadcode-report` wall time | `83.23 s` |
| `make deadcode-check` wall time | `104.00 s` |

## Residual Deferred Debt

Deferred Sprint 33 queue at close:

- compile-db coverage gap:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- dead-code workflow serialization/build-tree isolation work
- residual `cppcheck` supporting-signal review

Not carried forward as cleanup debt:

- definitely-unused internal candidate queue: none
- public-surface ambiguity on the four Day 8 audited symbols: none
- warning debt: none introduced by Sprint 33
- dormant test-scaffold debt: none reintroduced by Sprint 33

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [HANDOFF.md](./HANDOFF.md)
- [day4-tooling-integration-design.md](./artifacts/day4-tooling-integration-design.md)
- [day5-deadcode-target-implementation.md](./artifacts/day5-deadcode-target-implementation.md)
- [day7-report-wiring-and-first-report.md](./artifacts/day7-report-wiring-and-first-report.md)
- [day8-public-surface-audit.md](./artifacts/day8-public-surface-audit.md)
- [day10-cleanup-batch1.md](./artifacts/day10-cleanup-batch1.md)
- [day11-reconciliation.md](./artifacts/day11-reconciliation.md)
- [day12-documentation-refresh.md](./artifacts/day12-documentation-refresh.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)

## Bottom Line

Sprint 33 achieved its engineering goal:

- the repository now has a real dead-code workflow instead of an implicit
  scanner-and-grep habit
- the reporting model distinguishes cleanup-ready code from review/defer buckets
- the first definitely-unused internal cleanup pass is complete
- the Sprint 32 warning-clean and truthfulness baseline still holds

Sprint 34 should start by hardening and enforcing this workflow, not by
pretending Sprint 33 discovered a larger cleanup-ready code-removal backlog than
the evidence actually supports.
