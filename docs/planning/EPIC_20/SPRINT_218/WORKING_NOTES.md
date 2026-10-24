# Sprint 218 Working Notes

**State:** Day 14 completed the full Make and reviewed CMake quality gates;
all six Sprint 218 items are closed for the selected local matmul scope.
The sprint worktree remains uncommitted. **Plan:**
[Sprint 218 plan](PLAN.md). **Epic scope:**
[project plan](../PROJECT_PLAN.md).

## Day 1 Inventory

The [Day 1 contract and cost inventory](artifacts/day1-matmul-contract-cost-inventory.md)
traces `sparse_matmul` through `sparse_insert`, output-node and shell
allocation, row/column links, permutations, and cleanup. The result flush
sorts touched columns but scans each existing row and column chain again for
every retained value. A row with `r` retained entries therefore incurs
`Theta(r^2)` row-link comparisons, separate from accumulation and sorting.
The existing internal entry builder has append tails, but its duplicate and
zero semantics differ from matmul; Day 3 owns the reuse decision.

## Semantics Table

| Surface | Current observation | Evidence level / next owner |
| --- | --- | --- |
| Error and output pointer | NULL output pointer wins; otherwise `*C` is cleared before NULL-input and shape checks, and remains NULL on allocation failure. Old output storage is caller-owned, not freed. | Source and existing tests; Day 9-10 extend failure positions. |
| Output ordering | Touched columns are sorted; `sparse_insert` links rows by column and columns by row. Result permutations start as identity. | Source; Days 7-8 add structural traversal checks. |
| Duplicate paths and drop | Contributions sum in `A`-then-`B` traversal order; one touched column is flushed only when `fabs(value) >= 1e-15`. | Source plus Day 7 independent dense-value and explicit support-mask fixtures pin cancellation and threshold boundaries. |
| Input permutations | Public API declares physical-index multiplication and excludes non-identity permutations. | Header and source; do not silently change scope. |
| Non-finite arithmetic | NaN accumulator fails the threshold comparison; infinity is retained; no numeric error is returned for these cases. | Day 3 and Day 8 controls pin this observed behavior without broadening the API promise. |
| Result ownership | Output is a new pool-backed matrix, published only on success; `sparse_free` owns slabs and shell arrays. | Source and retry tests; Days 9-10 extend partial-node failure coverage. |

## Owner Map

| Owner | Current role | Sprint action |
| --- | --- | --- |
| `src/sparse_matrix.c` | Matmul accumulator, sorted flush, `sparse_insert`, shell/node lifecycle. | Days 3-6 builder; Days 9-10 rollback audit. |
| `src/sparse_matrix_build_internal.c` | Existing sorted-entry row/column-tail constructor. | Day 3 assess reuse, including materialization cost and semantic mismatch. |
| `src/sparse_alloc_internal.c` | Checked workspace/shell-array allocations and fault injection. | Days 9-10 map every new helper allocation. |
| `tests/test_matmul.c` | 18 registered cases, including three workspace failure sites and retry. | Days 6-10 oracle, ordering, and failure expansion. |
| `tests/test_matmul_allocation_failure_gate_registration.py`, `Makefile`, `CMakeLists.txt` | Gate and CTest registration. | Keep test ownership and registered execution intact. |
| Sprint artifacts | Raw fixtures, before/after measurements, gate ledger. | Day 2 baseline; Days 11-14 comparison and closeout. |

## Risk And Decision Log

| ID | Risk or decision | State / owner |
| --- | --- | --- |
| D1-1 | Preserve documented physical-index behavior; no implicit permutation support in this optimization. | Recorded; builder design Day 3. |
| D1-2 | Do not feed raw multiplication contributions to `sparse_matrix_build_from_entries`: it uses last-write duplicate resolution and exact-zero filtering. | Open design choice; Day 3. |
| D1-3 | The original fault injector missed raw output-shell and node-slab `malloc`. Matmul now routes both through the checked allocator without changing other callers' allocation sequence. | Site map closed on Day 9; rollback, retry, ASan/UBSan, and native leak audit passed on Day 10. |
| D1-4 | `fabs(NAN)` fails the drop predicate, while infinity passes; there is no dedicated matmul numeric status. Existing tests do not establish a public non-finite contract. | Observed behavior pinned by Day 3/8 controls; no new public contract claimed. |
| D1-5 | Current matmul tests use lookups and two cross-operation identities, not a small independent dense product oracle or traversal comparison. | Closed by Days 6-8 oracle, support masks, seeded fixtures, traversal, and consumer checks. |
| D1-6 | No speed or peak-memory conclusion follows from the focused test's framework time. | Measurement deferred; Days 2 and 11-12. |

## Test Matrix And Evidence Ledger

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; registration passed; 18 tests run, 0 failed, 0 skipped, 185 assertions. | Local Darwin 24.6.0 x86_64, Apple clang 17.0.0, normal Make build. Covers existing arithmetic and three workspace-failure positions, not wide-row timing or all allocation sites. |

No C/H file changed on Day 1, so the full C quality chain was not triggered.
CMake/CTest, sanitizer, repeated timing, and peak memory remain unrun. Day 2
must use pinned fixtures and retain raw samples rather than inferring speed
from this functional test run.

## Day 2 Baseline

The [Day 2 fixture and measurement record](artifacts/day2-matmul-baseline.md)
pins a narrow banded product, a 1024-entry wide output row, and a dense-result
rectangular product. `benchmarks/bench_matmul.c` creates inputs before the
timer, measures one kernel call, and validates expected input/output nnz after
the timer. Make and CMake both register the benchmark, but it is not part of
CTest or the fast benchmark suite. A fresh CMake `-O2` tree was used for the
five independent-process samples per fixture. The wide-row fixture produces
1024 sorted insertions into one row, exposing the existing quadratic scan.

The raw samples show substantial host variability, so Days 11-12 must retain
the same fixture, build mode, and multiple runs. Process peak RSS includes
setup and is not a kernel-only allocation delta. No before/after speedup or
memory reduction is claimed yet. The Day 1 semantics table still applies;
these positive finite fixtures establish no new NaN or infinity contract.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, `271bd3b704326296332cb397385ed458436ca0b7` | `cmake -S . -B build/s218-baseline -DCMAKE_C_FLAGS=-O2 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` then `cmake --build build/s218-baseline --target bench_matmul --parallel 4` | Both exit 0; fresh tree, Apple clang 17.0.0. | Dedicated build prevents stale Make objects from contaminating the benchmark. Existing unrelated `-Wdouble-promotion` warnings remain in library sources. |
| 2026-10-23, same revision | Five `/usr/bin/time -l build/s218-baseline/bench_matmul <fixture>` processes per fixture | 15/15 exit 0; times, max RSS, and peak footprint retained in Day 2 artifact. | Kernel-only time; whole-process memory peak. |
| 2026-10-23, same revision | `make matmul-allocation-failure-gate` | Exit 0; registration passed, 18 tests passed, 185 assertions. | Existing pre-change correctness and three workspace-failure positions. |
| 2026-10-23, same revision | `cmake --build build/s218-baseline --target test_matmul --parallel 4` then `ctest --test-dir build/s218-baseline -R '^test_matmul$' --output-on-failure` | Both exit 0; 1/1 focused CTest passed. | Confirms the selected CMake tree also registers and runs the existing matmul test. |
| 2026-10-23, same revision | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Includes newly registered benchmark in format list. |
| 2026-10-23, same revision | `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. | Repository-wide lint; existing library `-Wdouble-promotion` warnings during the separate CMake build did not fail the gate. |
| 2026-10-23, same revision | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite, including existing matmul cases. |

## Day 3 Builder Decision

The [Day 3 ownership design](artifacts/day3-builder-ownership-design.md)
selects direct ordered append into a private output matrix. One row-tail
pointer and a checked `B->cols`-length column-tail array replace per-entry
row and column head scans. The existing accumulator, contribution order,
sorted touched columns, and threshold predicate remain unchanged. This
avoids an O(output nnz) compressed staging copy and avoids the existing
sorted-entry builder's incompatible duplicate and zero rules.

The result owns every created node through its pool from allocation onward;
temporary tails never own nodes. On any failure, one cleanup path frees the
four work arrays and `sparse_free(out)`, while `*C` remains NULL. The design
requires an `IDX_MAX` guard before `nnz` increments and a checked tail-array
allocation. Day 4 may retain a column search temporarily while replacing row
searches, but every intermediate output must still have complete row and
column links. Day 5 completes the no-head-scan path.

The model separates `T` accumulation operations and per-row touched-column
sorts from O(output nnz) link work. For the pinned wide-row fixture, the
current row flush performs 523,776 row-head comparisons; direct linking
uses 1,024 appends plus the unchanged accumulation and sort. This is a
structural bound, not a measured speedup. No C/H file changed on Day 3, so
the Day 2 passing gates remain the latest executable evidence.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, `271bd3b704326296332cb397385ed458436ca0b7` | Source/design review of `sparse_matmul`, `sparse_matrix_build_from_entries`, pool allocation, and checked array helpers | Direct-tail design and single-owner rollback documented. | Design-only Day 3; implementation and fault-position tests begin Days 4-5 and 9-10. |

## Day 4 Row Append

The [Day 4 implementation record](artifacts/day4-row-append-implementation.md)
describes the private row-tail append helper, unchanged accumulation/drop
behavior, centralized partial-output cleanup, and three new focused tests.
Rows now append in constant link time after touched-column sorting; columns
still use a sorted head search and remain fully linked. Day 5 owns removal
of that remaining scan. The broad pre-change Day 2 timing sample must not be
presented as an after-change performance result.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; 21 tests passed, 6373 assertions. | Includes reverse-touch wide row, duplicate/cancellation/empty row, NaN/infinity tests; column-tail path not yet implemented. |
| 2026-10-23, same working tree | CMake `test_matmul` build and focused CTest | Both exit 0; 1/1 CTest passed. | Existing dedicated `-O2` tree rebuilt changed source. |
| 2026-10-23, same working tree | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Repository formatting and source-list checks. |
| 2026-10-23, same working tree | `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. | Repository-wide static checks after C changes. |
| 2026-10-23, same working tree | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite after matmul row-append change. |

## Day 5 Column Links

The [Day 5 integration record](artifacts/day5-column-tail-integration.md)
documents the checked, zero-initialized column-tail array and the finished
constant-time row/column append path. Sorted touched columns order row links;
increasing output rows order column links. The single node for each retained
coordinate is shared by both chains, and the result is not published until
all rows complete. All four temporary arrays and partial output have one
failure cleanup path. The earlier column-head-scan bound no longer applies
to this working tree; Day 2 measurements remain the pre-change baseline.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; 22 tests passed, 6487 assertions. | Includes direct row/column pointer identity, consumer conversions, and the new tail-allocation failure/retry. |
| 2026-10-23, same working tree | CMake `test_matmul` build and focused CTest | Both exit 0; 1/1 CTest passed. | Dedicated `-O2` tree rebuilt changed source. |
| 2026-10-23, same working tree | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Repository formatting and source-list checks. |
| 2026-10-23, same working tree | `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. | Repository-wide static checks after C changes. |
| 2026-10-23, same working tree | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite after matmul column-tail integration. |

## Day 6 Builder Review And Dense Oracle

The [Day 6 review and oracle record](artifacts/day6-builder-review-dense-oracle.md)
closes item 218.2's builder allocation: each retained output has one node
linked in O(1) in both directions, with no per-entry `sparse_insert` or head
scan. The checked workspaces and private output have a single failure cleanup
path. Five registered oracle cases independently compute dense products from
input arrays and check the exact output shape, values, nonzero count, ordered
row/column chains, and shared node identity. Inputs are finite small integers;
Day 7 covers contribution and cancellation behavior more broadly. The Day 2
benchmark is still a pre-change baseline only.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; 27 tests passed, 6846 assertions. | Adds independent dense rectangular, identity, empty, structural zero, and dense-result fixtures. |
| 2026-10-23, same working tree | CMake `test_matmul` build and focused CTest | Both exit 0; 1/1 CTest passed. | Dedicated `-O2` tree rebuilt changed source and tests. |
| 2026-10-23, same working tree | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Repository formatting and source-list checks. |
| 2026-10-23, same working tree | `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. | Repository-wide static checks after Day 6 C/test changes. |
| 2026-10-23, same working tree | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite with the independent dense oracle cases. |

## Day 7 Cancellation And Determinism

The [Day 7 fixture and verification record](artifacts/day7-cancellation-determinism.md)
extends the dense oracle with independently specified support masks. It pins
duplicate contributions to one output coordinate, exact cancellation,
positive and negative residuals on either side of `1e-15`, and equality at
the drop boundary. Binary-exact cases use zero value tolerance; the
three-term decimal case permits `8 * DBL_EPSILON` absolute error while
checking support separately. A 3x257 reverse-touch product is traversed in
both directions on three runs, checking 771 ordered entries each time. This
is correctness evidence only; no post-change speedup is claimed yet.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; 31 tests passed, 27275 assertions. | Four new registered Day 7 cases and existing allocation-failure coverage. |
| 2026-10-23, same working tree | CMake `test_matmul` build and focused CTest | Both exit 0; 1/1 CTest passed. | Focused CMake test parity. |
| 2026-10-23, same working tree | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Formatting and manifest checks. |
| 2026-10-23, same working tree | `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. | Repository-wide static checks after Day 7 test changes. |
| 2026-10-23, same working tree | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite after the Day 7 regressions. |

## Day 8 Seeded Shapes And Consumers

The [Day 8 correctness matrix](artifacts/day8-seeded-shape-consumer-matrix.md)
closes item 218.3's 28-hour allocation. Four fixed seeds cover tall and wide
products with sparse and fully dense results, plus an empty intermediate row,
output row, and output column. The oracle now checks source immutability,
`sparse_get`, matvec, CSR and CSC conversions, and ordered row/column chains
for all finite fixtures. Explicit support masks pin the four seeded shapes.
Large finite multiplication and non-finite accumulation controls match the
recorded pre-change behavior. No correctness discrepancy remains open in
this scope; allocation-site failure coverage and performance comparison
remain future Sprint 218 work.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; 37 tests passed, 29632 assertions. | Six new Day 8 cases plus prior regressions. |
| 2026-10-23, same working tree | CMake `test_matmul` and `test_csr` builds and focused CTest | Both builds exit 0; 2/2 CTests passed. | Matmul and conversion test paths. |
| 2026-10-23, same working tree | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Formatting and source manifest. |
| 2026-10-23, same working tree | `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. | Repository-wide static checks after Day 8 test changes. |
| 2026-10-23, same working tree | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite after Day 8. |

## Day 9 Allocation-Site Fault Coverage

The [Day 9 allocation-site map](artifacts/day9-allocation-site-failure-map.md)
enumerates the output shell, six shell arrays, four matmul workspaces, and
output-node slab allocations. Matmul alone routes shell and slab allocations
through the checked fault hook; other matrix constructors and pool callers
retain their prior allocation sequence. The focused gate covers each first
allocation site and a second-slab failure after nodes in the active row and
column have linked. Every injected case requires `SPARSE_ERR_ALLOC`, a NULL
output, unchanged inputs, caller ownership of a stale output, and a clean
uninjected retry. A three-node-slab CMake build verifies the test's dynamic
slab boundary. Day 10 still owns the independent rollback and instrumentation
audit; Day 9 does not establish a post-change performance result.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; registration passed, 39 tests passed, 48453 assertions. | All distinct matmul output allocation sites plus later slab after partial links. |
| 2026-10-23, same working tree | Default CMake `test_matmul` build and focused CTest | Both exit 0; 1/1 CTest passed. | Normal slab-size CMake registration and execution. |
| 2026-10-23, same working tree | CMake configure/build with `-DSPARSE_NODES_PER_SLAB=3`, then focused CTest | All exit 0; 1/1 CTest passed. | Small-slab fault position and rollback fixture. |
| 2026-10-23, same working tree | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Formatting and source manifest. |
| 2026-10-23, same working tree | `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. | Repository-wide static checks after Day 9 C/test changes. |
| 2026-10-23, same working tree | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite after Day 9. |

## Day 10 Rollback And Retry

The [Day 10 rollback and instrumentation record](artifacts/day10-rollback-retry-sanitizer.md)
extends the allocation gate with a three-row, wide-output fixture. On the
default build it injects failure at the second, third, and fourth slab
allocations; a one-node-slab build reaches each later node allocation across
row transitions and previously linked columns. Every failed call clears the
output without taking ownership of caller-held stale storage, leaves inputs
unchanged, and is followed immediately by a successful product checked
against an independent value formula and ordered row/column traversals.
The test frees each retried result before the next injection. NULL and shape
error precedence remain as on Day 1; zero dimensions are rejected by
`sparse_create` before matmul can receive a valid matrix.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; registration passed, 41 tests, 372262 assertions. | Includes three later-slab failures and immediate retries in the default build. |
| 2026-10-23, same working tree | CMake `-DSANITIZE=all -DCMAKE_C_FLAGS=-DSPARSE_NODES_PER_SLAB=1` configure/build and focused CTest | All exit 0; 1/1 CTest passed. | ASan/UBSan covers every later node allocation on this tiny fixture; macOS ASan leak detection is not used. |
| 2026-10-23, same working tree | Native one-node-slab CMake build, then `leaks --atExit -- build/s218-day10-leaks/test_matmul` | Build and test exit 0; 41 tests passed; 0 leaks, 0 leaked bytes. | Native macOS leak check of all eight later node failures. |
| 2026-10-23, same working tree | `leaks --atExit -- build/test_matmul` | Exit 0; 41 tests passed; 0 leaks, 0 leaked bytes. | Native leak check of default-slab wide-row/late-row failures. |
| 2026-10-23, same working tree | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Formatting and source manifest. |
| 2026-10-23, same working tree | `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. | Repository-wide static checks after Day 10 test changes. |
| 2026-10-23, same working tree | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite after the Day 10 regressions. |

## Day 11 Failure Closure And Measurement Setup

The [Day 11 signoff and measurement protocol](artifacts/day11-failure-closure-measurement-protocol.md)
closes item 218.4: the output shell, six shell arrays, four workspaces, first
node slab, and later slab allocations all have registered injected-failure
coverage. The gate passes again; Day 10's one-node-slab ASan/UBSan and native
leak runs provide cleanup evidence. No further explicit matmul allocation
site remains outside the checked hook. This does not claim control of
third-party allocator internals or synthetic `IDX_MAX` guard coverage.

A clean detached worktree at the pre-change commit and the current Sprint
working tree were built in separate CMake directories with the same Apple
Clang compiler, `-O2`, C11/project warnings, and no sanitizer or OpenMP.
Both binaries use the same current benchmark source. Post-timer validation
now checks the complete expected value grid and shape as well as `nnz`;
all three pinned shapes pass on both variants. The protocol freezes two
warmup processes and ten alternating-order paired sample rounds per shape,
raw stdout/stderr/status retention, whole-process memory fields, and median
plus spread reporting. The six smoke invocations are not a speedup result.
Source and executable hashes, host characteristics, confounders, and exact
rebuild commands are recorded in the Day 11 artifact.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make matmul-allocation-failure-gate` | Exit 0; registration passed, 41 tests passed, 372262 assertions. | Final item 218.4 allocation-site ledger check. |
| 2026-10-23, same working tree | Baseline/candidate fresh CMake configure and `sparse_lu_ortho` builds with `/usr/bin/clang`, `-O2` | All exit 0; cache and `compile_commands.json` parity checked. | Baseline at clean detached HEAD; candidate includes uncommitted Sprint 218 changes. |
| 2026-10-23, same working tree | Compile same `benchmarks/bench_matmul.c` against each archive and run `banded`, `wide-row`, `rectangular` once per variant | Both driver builds and all six smoke invocations exit 0; full post-timer value, shape, and `nnz` checks pass. | Setup only; not paired, warmed, or memory-instrumented. |
| 2026-10-23, same working tree | `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. | Formatting and source manifest after benchmark validation change. |
| 2026-10-23, same working tree | `make lint` | Exit 0; benchmark tooling build, strict compile, clang-tidy, and cppcheck completed. | Repository-wide static checks. |
| 2026-10-23, same working tree | `make test` | Exit 0; final output `All tests passed.` | Full normal Make suite after Day 11. |

## Day 12 Selected-Kernel Profile

The [Day 12 profile](artifacts/day12-matmul-profile.md) uses the matched
Day 11 baseline and candidate binaries, with source/executable hashes
rechecked before sampling. Two warmup processes per variant/fixture preceded
ten alternating-order paired samples per fixture. All 60 measured processes
exited zero after full output-value, shape, and `nnz` validation. The
[raw bundle](artifacts/day12-raw/) contains 180 original stdout, stderr,
and status files; the [per-process CSV](artifacts/day12-samples.csv) and
[SHA-256 manifest](artifacts/day12-raw-sha256.txt) permit reanalysis and
integrity checking. The structured parser validated the complete set.

Local median times were 0.0695 versus 0.0610 ms for `banded`, 1.314 versus
0.109 ms for `wide-row`, and 3.948 versus 1.166 ms for `rectangular`
(baseline versus candidate). Wide-row and rectangular ranges were separate;
banded ranges overlapped and candidate time was slower in three of ten
pairs. Whole-process median RSS was equal for wide-row and 8192 bytes higher
for the candidate on banded and rectangular. These are selected local
observations, not portable speed or memory claims. The profile attributes
the larger-shape timing difference to removal of repeated row/column head
scans while noting unchanged accumulation/sort work, the new column-tail
workspace, slab allocation, and linked-structure cost.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, baseline `271bd3b704326296332cb397385ed458436ca0b7`, candidate working tree on same HEAD | Day 11 protocol: two warmups per fixture/variant; ten alternating paired `/usr/bin/time -l` rounds on `banded`, `wide-row`, `rectangular` | 60/60 measured processes exit 0; all value/shape/`nnz` checks pass; 180 raw files retained. | Kernel timer excludes setup and validation; memory fields are whole-process peaks. |
| 2026-10-23, same builds | `python3 scripts/summarize_matmul_profile.py` on the raw directory | Exit 0; 60 successful samples validated; derived CSV and medians/spread reproduced. | Rejects incomplete or malformed sample records; not a statistical confidence interval. |
| 2026-10-23, same builds | `shasum -a 256 -c` on the raw-file manifest | Exit 0; 180/180 raw-file hashes match. | Integrity of retained local raw bundle. |

## Day 13 Measurement Freeze And Instrumented Checks

The [Day 13 freeze and quality record](artifacts/day13-measurement-freeze-quality.md)
pins the raw manifest, derived CSV, profile, driver, candidate source, and
both executable hashes. It also retains exact commands and the selected-host
claim boundary. No Day 12 sample or summary was revised. Item 218.5 is
closed: the wide-row and rectangular selected fixtures showed nonoverlapping
local timing ranges; banded ranges overlapped; whole-process memory does not
support a kernel memory-reduction claim.

A fresh normal Make build passed the registered matmul allocation-failure
gate. Separate CMake `SANITIZE=all` trees with default and one-node slabs
both passed focused CTest. Native macOS `leaks` reported zero leaked bytes on
the normal focused binary. ASan leak detection was disabled on this host, so
the native leak result is recorded separately. `make format-check`,
`make source-list-check`, and `git diff --check` passed. Diff review found no
public header, library source-list, or unrelated storage-format change.
The full Make/CMake suites remain Day 14 work.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make BUILDDIR=build/s218-day13-normal matmul-allocation-failure-gate` | Exit 0; registration passed, 41 tests, 372262 assertions. | Fresh normal Make build, default slabs. |
| 2026-10-23, same working tree | CMake `SANITIZE=all` configure/build and focused CTest in `build/s218-day13-sanitize` with `SPARSE_NODES_PER_SLAB=1` | All exit 0; 1/1 CTest passed; direct test 41 cases, 30800 assertions. | ASan/UBSan, one-node-slab fault sweep; ASan leak detection disabled. |
| 2026-10-23, same working tree | CMake `SANITIZE=all` configure/build and focused CTest in `build/s218-day13-sanitize-default` | All exit 0; 1/1 CTest passed. | ASan/UBSan, default-slab later-failure path. |
| 2026-10-23, same working tree | `leaks --atExit -- build/s218-day13-normal/test_matmul` | Exit 0; 0 leaks, 0 leaked bytes. | Native macOS leak observation; focused default-slab test. |
| 2026-10-23, same working tree | `make format-check`, `make source-list-check`, `git diff --check`, raw manifest `shasum -a 256 -c` | All exit 0; 49 library sources, 180 raw files verified. | Scope, formatting, source manifest, and evidence integrity. |

## Day 14 Full Quality And Closeout

The [Day 14 quality and claim closeout](artifacts/day14-quality-closeout.md)
records all six items and their evidence links. `make format` made no change
to the measured matmul source or benchmark driver. A fresh normal Make build
passed lint and the full 59-binary test suite; the reviewed CMake path
configured, serial clean-built, verified 59 Make tests plus one CTest-only
selector, and passed all 60 CTests. The labeled matmul CTest and final
normal-build allocation-failure gate also passed. The Day 13 default and
one-node-slab ASan/UBSan results and native leak check remain the focused
instrumented evidence. All 180 raw measurement files reverified against
their SHA-256 manifest.

The selected output-linking improvement is complete, but only local
`wide-row` and `rectangular` timings support a faster-median observation;
the banded timings overlap, and whole-process peaks do not support a
kernel-memory improvement claim. Opt-in/framework skips and CMake's
nonfatal conversion warnings remain disclosed in the closeout artifact.
There is no known failing required Sprint 218 gate on this host; broader
platform and performance claims remain out of scope.

| Date / revision | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-23, working tree on `271bd3b704326296332cb397385ed458436ca0b7` | `make format` | Exit 0; measured source and driver hashes unchanged. | Formatter over registered C/H files. |
| 2026-10-23, same working tree | `make BUILDDIR=build/s218-day14-normal lint` | Exit 0; strict compile, tooling builds, clang-tidy, cppcheck. | Fresh normal Make build; no sanitizer objects reused. |
| 2026-10-23, same working tree | `make BUILDDIR=build/s218-day14-normal test` | Exit 0; `All tests passed.` | Full 59-binary Make suite; opt-in skips remain skips. |
| 2026-10-23, same working tree | `make QUALITY_REVIEW_CMAKE_DIR=build/s218-day14-cmake quality-review-cmake` | Exit 0; 60/60 CTests passed; count parity passed. | Clean CMake rebuild and full reviewed CTest path on Apple Clang 17. |
| 2026-10-23, same working tree | `ctest --test-dir build/s218-day14-cmake -L '^matmul$' --output-on-failure` | Exit 0; 1/1 focused test passed. | Labeled matmul selector. |
| 2026-10-23, same working tree | `make BUILDDIR=build/s218-day14-normal matmul-allocation-failure-gate` | Exit 0; registration passed, 41 tests, 372262 assertions. | Final ownership, rollback, and retry check on normal build. |
| 2026-10-23, same working tree | `make format-check`, `make source-list-check`, `git diff --check`, raw manifest `shasum -a 256 -c` | All exit 0; 49 library sources, 180 raw files verified. | Post-gate hygiene and frozen evidence integrity. |

## Post-PR Linux Benchmark Build Follow-Up

On 2026-10-24, PR #242's first Linux reviewed CMake run failed compiling
`benchmarks/bench_matmul.c` under strict C11: `clock_gettime` and
`CLOCK_MONOTONIC` were hidden without a POSIX feature declaration. The
benchmark now defines `_POSIX_C_SOURCE=200809L` before its headers on POSIX.
Its fixture construction, timer call, post-timer validation, and library
code are unchanged. The new benchmark source SHA-256 is
`61d5bf2cd0896dc4b9f6b5507195781f5a72be20f4891370e3d683dc3806aa61`;
the [Day 11 provenance note](artifacts/day11-failure-closure-measurement-protocol.md)
pins the original measurement driver at `806b31ed`. The two frozen
benchmark executable hashes remain `fb8df0a3...` and `d65c68c5...`;
all 180 raw files still pass the SHA-256 manifest. No profile was rerun.

| Date | Command | Outcome | Scope / limit |
| --- | --- | --- | --- |
| 2026-10-24 | `make format`; `make BUILDDIR=build/s218-postci-normal lint`; `make BUILDDIR=build/s218-postci-normal test` | All exit 0; `All tests passed.` | Fresh normal build after feature-macro change. |
| 2026-10-24 | `make QUALITY_REVIEW_CMAKE_DIR=build/s218-postci-cmake quality-review-cmake` | Exit 0; `bench_matmul` built; parity passed; 60/60 CTests passed. | Clean reviewed CMake path on macOS Apple Clang 17; Linux confirmation awaits CI. |
| 2026-10-24 | `make format-check`; `make source-list-check`; `git diff --check`; raw manifest `shasum -a 256 -c` | All exit 0; 49 sources, 180 raw files verified. | Formatting, source list, whitespace, and frozen evidence integrity. |
