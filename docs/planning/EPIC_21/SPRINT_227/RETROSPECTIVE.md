# Sprint 227 Retrospective

**Sprint:** 227 - Sorted Matrix Arithmetic and Permutation Safety

**Duration:** 14 planned days on branch `sprint-227`

**Status:** **Complete.** Out-of-place sparse addition now constructs sorted output without
repeated insertion scans; addition, in-place addition, and multiplication share an explicit
original-state guard; the independent oracle, structural, alias, and allocation-failure corpus
passes; and all final local quality, package, width, consumer, documentation, and evidence gates
are green. The retained timing result is local and workload-specific, and the memory evidence
supports no improvement claim.

## Source and Definition of Done

The [Epic 21 project plan](../PROJECT_PLAN.md) and [sprint plan](PLAN.md) allocate 140 estimated
hours across six items and fourteen days. These are planning estimates, not measured labor. The
[working notes](WORKING_NOTES.md), [closeout evidence index](artifacts/day14-closeout-evidence-index.md),
and [final quality decision](artifacts/day14-final-quality-and-handoff.md) reconcile every item,
gate, claim, and retained risk. Completion required direct sorted construction, one tested policy
for unsupported permutation states, independent numerical/failure evidence, and a fully passing
final-tree quality matrix.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 227.1 Contract Freeze | 18 h | Completed. Froze physical/original-state requirements, pointer/shape/state precedence, output ownership, alias behavior, cancellation threshold, and pre/post-validation failure semantics against deterministic dense-oracle fixtures. |
| 227.2 Permutation Guard | 30 h | Completed. Added one shared original-state guard across `sparse_add`, `sparse_add_inplace`, and `sparse_matmul`; both operands, forward/inverse maps, factor state, precedence, and unchanged-state rejection are covered. |
| 227.3 Sorted Output Builder | 28 h | Completed. Replaced repeated `sparse_insert` output construction in `sparse_add` with a two-list merge and direct row/column-tail append, with one finish/abort ownership boundary. |
| 227.4 Failure And Alias Matrix | 26 h | Completed. Closed shell/workspace/slab allocation boundaries, cancellation, rectangular/wide rows, randomized dense comparisons, caller-variable aliases, self-alias rejection, stale output, partial in-place failure, retry, copy, and destruction. |
| 227.5 Scaling Evidence | 20 h | Completed within a narrow claim. Retained 60 correctness-qualified measured processes and 300 hashed raw files for three pre-registered local workloads. Timing ranges separate; whole-process memory ranges overlap and authorize no memory claim. |
| 227.6 Quality Closure | 18 h | Completed. Format, lint, all Make/CMake tests, focused sanitizers, both index widths, installed consumers, static package mutations, API documentation, source inventory, links, and evidence integrity pass. |
| **Total** | **140 h** | **All six planned items and sprint completion criteria closed.** |

## What Went Well

1. **The unsupported-state bug became an explicit contract before implementation.** Day 1 probes
   demonstrated that physical arithmetic on row- or column-permuted storage could return success
   while disagreeing with logical dense arithmetic. Day 2 selected the repository-standard
   pre-mutation `SPARSE_ERR_BADARG` policy rather than adding partial logical-view support. The
   27-case cross-API matrix then made drift between add, in-place add, and matmul difficult.
2. **The builder design made complexity and ownership reviewable.** The output path now walks two
   sorted row lists and publishes each retained node once through a row tail and per-column tails.
   The builder has one active owner, one success transfer, and one abort cleanup path. This supports
   the documented `O(rows + cols + nnz(A) + nnz(B))` construction bound with `O(cols)` temporary
   workspace, excluding result storage.
3. **Failure behavior was tested as a data-structure contract.** Exact snapshots include metadata,
   maps, pool state, node identities, coordinates, values, and both link directions. Allocation
   sweeps cover the result shell, six arrays, workspace, first and later slabs. Every out-of-place
   failure preserves inputs and permits retry; the in-place post-validation failure retains the
   documented structurally valid scaled/prefix state and exact B.
4. **Independent oracles and adverse structures found more than happy-path tests could.** The
   corpus includes empty rows, disjoint and overlapping patterns, complete cancellation, zero
   coefficients, threshold boundaries, rectangular matrices, wide rows, fixed-seed randomized
   cases, downstream factorization, both index widths, and sanitizer runs. The final focused
   sequence passed 95 tests and 451,978 assertions.
5. **Measurement rules were frozen before collection.** Day 11 fixed three workload identities,
   exact source/build controls, AB/BA order, warmups, ten pairs, timeout limits, raw retention,
   invalidation rules, metric meaning, and claim limits. Day 12 completed one collection without
   selective reruns or discarded observations; the strict parser and SHA-256 manifests preserve
   the complete evidence chain.
6. **Closeout tested more than the modified unit.** The final gate covered all 62 Make test
   binaries, 63/63 CTests, package relocation/version checks, positive and negative 32/64-bit
   consumers, missing/swapped archive failures, static contract mutations, Doxygen coverage, and
   source-list parity. This caught interface and packaging regressions that focused arithmetic
   tests alone would not expose.

## What Did Not Go Well

1. **The original arithmetic behavior was misleading rather than fail-closed.** Valid nonidentity
   maps could produce a physically computed result with `SPARSE_OK`, and in-place self aliasing
   compounded coefficients instead of behaving like a snapshot. The sprint had to establish and
   enforce rejection semantics before optimizing construction.
2. **In-place addition could not share the out-of-place complexity or atomicity story.** B-only
   entries still use normal sorted insertion, so cost remains pattern dependent. A new-node
   allocation failure after validation can leave scaled A plus a completed B prefix. The state is
   valid and tested, but callers must understand that retrying the identical call can reapply work.
3. **Archive hashes obscured a comment-only equivalence check.** Apple `ar` refreshed member
   timestamps, changing the archive digest after Day 13 comments. Comparing the affected object
   and linked benchmark executable established byte identity; archive-container equality alone
   would have produced the wrong conclusion.
4. **The memory experiment was inconclusive.** RSS and footprint ranges overlapped for every
   fixture, and paired directions were mixed. The process measurements include setup and
   validation and do not isolate builder workspace. The sprint correctly retained no reduced-memory
   statement, but allocation instrumentation would be required for a stronger answer.
5. **The timing result is deliberately narrow.** One short local macOS collection lacks independent
   replication, frequency and thermal isolation, allocator controls, and confidence intervals.
   The large selected speed differences are useful local evidence for the construction change,
   not proof of portable speed, arbitrary sparsity, whole-library superiority, or state of the art.
6. **Full static analysis and dual build-system closure are expensive.** Focused loops were fast,
   while final lint, clean CMake rebuild, full CTest, package matrices, and long nested-dissection
   tests dominated closeout time. Reserving Day 14 for those gates was necessary and should remain
   standard practice.

## Final Metrics and Supported Claim

| Measure | Final audited result |
| --- | --- |
| Planned effort | 140 estimated hours; all 14 days and six items accounted for, actual hours not tracked |
| Contract | All three arithmetic APIs reject factored or nonidentity forward/inverse row/column maps after pointer/shape checks and before allocation or mutation |
| Construction | `sparse_add` uses sorted merge plus direct row/column-tail linking; no repeated output head scan per retained node |
| Focused regression | 49 arithmetic tests/79,666 assertions, 41 matmul tests/372,263 assertions, five integration tests/49 assertions; 95 tests and 451,978 assertions total |
| Full quality | `make format`, `make format-check`, `make lint`, all 62 Make test binaries, clean CMake registration parity, and 63/63 CTests pass |
| Sanitizer | Fresh ASan/UBSan affected lane passes 3/3 CTests with no finding |
| Package and width | Make width 28/28, Make install 26/26, CMake install 38/38, CMake installed width 20/20, pkg-config width/link 40/40, archive-swap 12/12, and static mutations 26/26 pass |
| Evidence bundle | 60 validated measured processes, 300 raw files, raw/sample/source SHA-256 manifests verified |
| Selected median time, baseline/candidate | Wide `13.4310/0.1515` ms; varied `2.1550/0.2470` ms; narrow `3.5605/0.2190` ms |
| Memory | RSS and footprint ranges overlap on all fixtures; no improvement claim |
| Documentation | All 18 public headers covered, 49 library sources registered, Sprint relative links and patch whitespace pass |

The supported sentence is: **“For the exact recorded source, host, compiler, and three selected
workloads, direct sorted row/column-tail construction reduced `sparse_add` kernel time relative to
the prior repeated-insertion path while preserving the frozen numerical and failure contract.”**
It does not extend to in-place addition, matmul, arbitrary sparsity, another platform or compiler,
peak memory, another library, release readiness, superiority, or state-of-the-art status.

## Daily Evidence

- [Day 1 arithmetic contract baseline](artifacts/day1-arithmetic-contract-baseline.md) and
  [dense-oracle fixtures](artifacts/day1-dense-oracle-fixtures.tsv)
- [Day 2 frozen arithmetic contract](artifacts/day2-arithmetic-contract.md) and
  [contract fixture matrix](artifacts/day2-contract-fixture-matrix.tsv)
- [Day 3 permutation fixtures](artifacts/day3-permutation-fixtures.md) and
  [regression corpus](artifacts/day3-permutation-regression-corpus.tsv)
- [Day 4 add permutation enforcement](artifacts/day4-add-permutation-enforcement.md)
- [Day 5 matmul enforcement and builder design](artifacts/day5-matmul-enforcement-and-builder-design.md)
- [Day 6 sorted addition builder](artifacts/day6-sorted-add-builder.md)
- [Day 7 builder invariants and edge corpus](artifacts/day7-builder-invariants-and-edge-corpus.md)
- [Day 8 allocation-failure sweep](artifacts/day8-add-allocation-failure-sweep.md)
- [Day 9 add failure, alias, and randomized closure](artifacts/day9-add-failure-alias-and-randomized-closure.md)
- [Day 10 in-place atomicity and alias closure](artifacts/day10-inplace-atomicity-and-alias-closure.md)
- [Day 11 integrated matrix and scaling protocol](artifacts/day11-integrated-failure-matrix-and-scaling-protocol.md)
- [Day 12 selected addition profile](artifacts/day12-add-profile.md),
  [samples](artifacts/day12-samples.csv), [raw hashes](artifacts/day12-raw-sha256.txt), and
  [collection source hashes](artifacts/day12-source-sha256.txt)
- [Day 13 claim ledger](artifacts/day13-claim-ledger.md),
  [quality packet](artifacts/day13-evidence-freeze-and-quality.md), and
  [current source hashes](artifacts/day13-source-sha256.txt)
- [Day 14 closeout evidence index](artifacts/day14-closeout-evidence-index.md),
  [final quality decision and handoff](artifacts/day14-final-quality-and-handoff.md), and
  [retrospective input](artifacts/day14-retrospective-input.md)

## Residual Handoff

Sprint 228 may reuse the original-state validation seam, ownership/failure test patterns,
independent dense-oracle helpers, and evidence-retention discipline. It must independently freeze
low-rank reconstruction output ordering, threshold behavior, allocation lifecycle, scratch-memory
accounting, and dense-reference tolerances. The file-local addition builder is not a generic public
builder, and Sprint 227 timing or memory observations do not transfer to reconstruction.

Arithmetic on logical permuted views remains a separate future design; until then rejection is the
accepted contract. Independent addition timing replication and allocation-level memory accounting
are optional future evidence work, not unfinished Sprint 227 criteria. After this retrospective is
committed and pushed, hosted Linux, macOS, and Windows CI must pass for the new branch head before
merge; local completion does not substitute for those platform results.
