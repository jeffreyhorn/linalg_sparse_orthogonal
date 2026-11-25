# Sprint 227 Working Notes

**Branch:** `sprint-227`

**Base commit:** `fcdeaaee856821e8c801d805c63b5fd40eb89b2f` (merge of PR #251 on `master`)

**Scope:** Sorted matrix addition construction and one explicit permutation-safety contract for `sparse_add`, `sparse_add_inplace`, and `sparse_matmul`.

**Current decision:** **Sprint 227 complete.** Day 2 freezes pre-mutation rejection, now enforced across `sparse_add`, `sparse_add_inplace`, and `sparse_matmul`. Days 6–8 implement, harden, and exhaustively failure-test the sorted-output builder. Days 9–10 close out-of-place and in-place ownership. Day 11 integrates the adverse corpus and freezes three correctness-qualified `bench_add` profiles. Day 12 executes that protocol once: all 60 exact-source samples pass and show separated local kernel-time ranges, while overlapping whole-process memory ranges support no memory-improvement claim. Day 13 freezes those boundaries and passes focused gates. Day 14 passes the complete Make/CMake, sanitizer, package, width, consumer, documentation, and evidence-integrity matrix and hands only the bounded capability to Sprint 228.

## Day-by-day ledger

| Day | Work and decision | Evidence | Open condition |
| --- | --- | --- | --- |
| 1 | Traced public and internal arithmetic paths, recorded ownership/error behavior, ran focused baseline suites, demonstrated nonidentity logical mismatches and self-alias ambiguity, inventoried injectable allocations, and froze a deterministic dense-oracle fixture matrix. No contract choice or source repair was made. | [Day 1 arithmetic contract baseline](artifacts/day1-arithmetic-contract-baseline.md), [dense-oracle fixture matrix](artifacts/day1-dense-oracle-fixtures.tsv) | Day 2 must choose and document one compatible permutation policy, define alias/error precedence, and map each fixture to logical success or pre-mutation rejection. |
| 2 | Selected the repository-standard original-state guard and `SPARSE_ERR_BADARG` policy, froze API-specific null/shape/state precedence, rejected in-place self alias, bounded finite numerical/threshold semantics, retained post-validation partial-update behavior, and mapped 22 contract cases to exact outcomes. | [Frozen arithmetic contract and guard design](artifacts/day2-arithmetic-contract.md), [contract fixture matrix](artifacts/day2-contract-fixture-matrix.tsv) | Day 3 must encode valid row/column/inverse/factored fixtures and independent identity-state oracles before Day 4 changes production guards. |
| 3 | Added valid row-only, column-only, and row-plus-column fixture constructors; independent dense add/matmul oracles; map, snapshot, node-count, and row/column structural checks; and a 30-row rectangular corpus. Always-on helper tests pass, while the opt-in pending contract test proves all 27 unsupported-state combinations still fail against production. | [Day 3 fixture design and test-first evidence](artifacts/day3-permutation-fixtures.md), [permutation regression corpus](artifacts/day3-permutation-regression-corpus.tsv), `tests/arithmetic_contract_fixture.h` | Day 4/5 must add the shared guards, promote the experimental rejection corpus into the normal suite, and extend it with frozen inverse-only, factored, and alias precedence cases. |
| 4 | Applied the shared original-state seam to add and in-place add after pointer/shape checks and before allocation/mutation; rejected in-place self alias; promoted 18 permutation cases; added inverse-only, factored, shape-precedence, stale-output, alias, and allocation-sentinel regressions; and found no production callers requiring migration. | [Day 4 add permutation enforcement and caller audit](artifacts/day4-add-permutation-enforcement.md), `src/sparse_matrix.c`, `tests/test_sparse_arith.c` | Day 5 must enforce the same contract in matmul, promote its remaining nine cases, update public wording, and freeze the sorted-builder design. |
| 5 | Enforced the original-state seam in matmul before allocation; promoted all nine pending permutation cases; added inverse-only, factored, precedence, output, allocation, and unchanged-input regressions; migrated the Cholesky reconstruction fixture; updated all three public contracts; and froze a bounded sorted-builder design. | [Day 5 matmul enforcement and builder design](artifacts/day5-matmul-enforcement-and-builder-design.md), `include/sparse_matrix.h`, `src/sparse_matrix.c`, `tests/test_sparse_arith.c`, `tests/test_matmul.c` | Day 6 must implement the builder without broadening its scope or claims and prove representative structural invariants. |
| 6 | Implemented the file-local sorted-output builder; replaced all out-of-place add insertion calls with direct row/column tail linking; preserved validation, arithmetic, threshold, and ownership behavior; and added representative merge plus workspace/first-node failure evidence. | [Day 6 sorted addition append construction](artifacts/day6-sorted-add-builder.md), `src/sparse_matrix.c`, `tests/test_sparse_arith.c` | Day 7 must harden structural invariants and edge/oracle coverage; Day 8 owns the exhaustive allocation sweep. |
| 7 | Strengthened row/column cross-link identity validation; added empty, disjoint, cancellation, zero-scalar, rectangular, wide-row, copy/destruction, and 96 fixed-seed randomized oracle cases; passed focused 32/64-bit and sanitizer runs; and audited all builder size arithmetic and ownership transfers. | [Day 7 builder invariants and edge corpus](artifacts/day7-builder-invariants-and-edge-corpus.md), `tests/arithmetic_contract_fixture.h`, `tests/test_sparse_arith.c` | Day 8 must exhaustively sweep shell, workspace, first-slab, and later-slab failures and prove rollback/retry behavior at each boundary. |
| 8 | Enumerated all ten result allocations; swept failures across the object, six shell arrays, workspace, first slab, and second slab; proved success at and beyond the final allocation; retried after every failure; and verified zero leaks with platform memory tooling. | [Day 8 add allocation-failure sweep](artifacts/day8-add-allocation-failure-sweep.md), `tests/test_sparse_arith.c` | Day 9 must close remaining add failure/output-pointer and alias semantics without weakening this sweep. |
| 9 | Added exact topology/metadata snapshots; covered NULL, shape, state, allocation, stale-output, cancellation-only, caller-variable alias, repeated-call, and same-input behavior; documented output storage ownership; and added 48 fixed-seed wide rectangular oracle cases. | [Day 9 add failure, alias, and randomized closure](artifacts/day9-add-failure-alias-and-randomized-closure.md), `include/sparse_matrix.h`, `tests/arithmetic_contract_fixture.h`, `tests/test_sparse_arith.c` | Day 10 must close in-place partial-failure structural, retry, and alias behavior while retaining B's exact snapshot. |
| 10 | Made the existing in-place slab allocation injectable; proved exact pre-mutation rejections and self-alias behavior; exercised shared patterns, cancellation, scaling, empty rows, and free-list reuse; and froze the structurally valid scaled/prefix state after allocation failure with exact B preservation and fresh retry. | [Day 10 in-place atomicity and alias closure](artifacts/day10-inplace-atomicity-and-alias-closure.md), `include/sparse_matrix.h`, `src/sparse_matrix.c`, `tests/test_sparse_arith.c` | Day 11 must run the integrated cross-API failure matrix and pre-register selected scaling profiles without broadening claims. |
| 11 | Ran the integrated add/in-place/matmul adverse corpus; added a maintained correctness-qualified addition benchmark; pinned wide, varied, and narrow-row workload identities; and froze matched builds, AB/BA ordering, warmups, repetitions, host controls, time/memory meanings, invalidation rules, raw retention, and claim limits. | [Day 11 integrated matrix and selected scaling protocol](artifacts/day11-integrated-failure-matrix-and-scaling-protocol.md), `benchmarks/bench_add.c`, `Makefile`, `CMakeLists.txt` | Day 12 must collect the exact baseline/candidate pair under the frozen protocol; it may not selectively rerun or broaden the selected-workload claim. |
| 12 | Built isolated matched baseline/candidate static libraries, compiled one shared correctness driver, executed one capped 60-process alternating paired collection, retained 300 raw files plus exact receipts and hashes, validated every sample, and reported separated selected-kernel timing alongside overlapping whole-process memory. | [Day 12 selected addition profile](artifacts/day12-add-profile.md), [samples](artifacts/day12-samples.csv), [raw manifest](artifacts/day12-raw-sha256.txt), [source manifest](artifacts/day12-source-sha256.txt), `scripts/collect_add_profile.py`, `scripts/summarize_add_profile.py` | Day 13 must freeze each statement to this exact evidence, run quality/parity/sanitizer gates, and retain the local-only claim boundary. |
| 13 | Froze supported and prohibited statements; documented permutation, ownership, alias, failure, and complexity behavior; tied selected timing to raw evidence; proved the comment-only rebuild byte-identical at object/executable level; and passed focused Make, CMake parity, and sanitizer gates. | [Day 13 claim ledger](artifacts/day13-claim-ledger.md), [evidence freeze and quality results](artifacts/day13-evidence-freeze-and-quality.md), [current source manifest](artifacts/day13-source-sha256.txt) | Day 14 must run the complete required quality chain and close or explicitly block Sprint 227. |
| 14 | Passed the full final-tree quality chain, all 63 CTests, focused sanitizers, complete affected install/index-width/consumer matrix, documentation and evidence-integrity checks; reconciled all six Epic items and froze residual risks. | [Closeout evidence index](artifacts/day14-closeout-evidence-index.md), [final quality decision and handoff](artifacts/day14-final-quality-and-handoff.md), [retrospective input](artifacts/day14-retrospective-input.md) | Sprint 227 has no open acceptance blocker. Sprint 228 must independently freeze reconstruction-specific contracts and evidence. |

## Evidence ledger

| Evidence | Current value or state | Required next proof |
| --- | --- | --- |
| Baseline source | `fcdeaaee856821e8c801d805c63b5fd40eb89b2f`; no tracked C/header edits on Day 1 | Re-record exact source identity after implementation and final quality edits. |
| Existing focused suites | Dedicated Release CMake tree; `test_sparse_arith` 23/23 tests and 103 assertions pass; `test_matmul` 41/41 tests and 372,262 assertions pass | Add permutation, alias, and add-allocation coverage derived from the fixture matrix. |
| Nonidentity behavior | Baseline APIs return misleading success; frozen target is `SPARSE_ERR_BADARG` for factored state or any nonidentity forward/inverse map after pointer/shape checks | Days 3–5 executable regression, guard implementation, and public wording. |
| In-place self alias | Baseline computes `8*A` for coefficients 2 and 3; frozen target is `SPARSE_ERR_BADARG` before any cache/factor/value mutation | Day 3 unchanged-state fixture; Day 4–5 enforcement. |
| Add allocation injection | Fail-after indices 0–5 return `SPARSE_ERR_ALLOC` with `C_out=NULL`; indices 6–7 succeed | Make node/slab allocation injectable in the new builder and sweep all sites through success plus one boundary. |
| Current construction | Add and matmul now use distinct file-local direct row/column-tail append paths; add's merge no longer calls `sparse_insert` | Day 7 hardens the add builder's complete allocation and edge-case surface; consolidation is out of scope without separate evidence. |
| Selected process baseline | Local macOS: arithmetic test 0.00 s and 815,104-byte max RSS; matmul test 0.07 s and 3,526,656-byte max RSS | Day 11 pre-register isolated wide-row addition workloads; Day 12 collect matched before/after evidence. |
| Dense-oracle fixtures | 14 deterministic numerical rows plus a 22-case status/precedence matrix | Turn identity rows into independent C oracles and state/alias rows into unchanged-state regressions; preserve seeds and expected status. |
| Guard seam | `sparse_matrix_require_original_state` now guards both operands in all three APIs at the frozen precedence points; no public ABI changed | Retain through the builder refactor and later failure sweeps. |
| Day 3 executable corpus | All 27 unsupported-state combinations are always on and pass: left/right/both and row/column/both for add, in-place add, and rectangular matmul | Preserve as a cross-API consistency gate. |
| Day 5 enforcement | Matmul adds nine valid-pair permutation combinations plus inverse-only, factored, shape/state, output, allocation-precedence, and unchanged-input cases | Preserve through subsequent construction changes. |
| Sorted-builder implementation | One owned output, one column-tail workspace, one row tail, strict row/column sentinels, injectable shell/slabs, and single finish/abort transfer; all ten two-slab allocation boundaries, retries, 32/64-bit runs, sanitizers, and `leaks` pass | Preserve through the remaining sprint gates; Day 9 closed output-pointer and alias semantics. |
| Out-of-place add ownership | Exact pointer/topology/metadata snapshots cover all documented errors; cancellation and two-slab sweeps cover zero and large output; caller-variable aliases, stale outputs, repeated calls, and same-input success are executable | Preserve as closed in Day 11's combined cross-API adverse corpus. |
| In-place add mutation boundary | NULL/shape/state/alias failures are exact and pre-mutation; the one injectable post-validation slab failure retains scaled A plus the completed B prefix, leaves B exact, and remains copyable, readable, reusable, and destroyable; fresh retry succeeds | Preserve as closed and include in Day 11's combined cross-API adverse corpus. |
| Integrated adverse corpus | Add, in-place add, matmul, and downstream Sprint 4 integration pass 95 tests and 451,978 assertions in one normal 32-bit sequence | Preserve in the final sprint gate; investigate any later discrepancy before accepting evidence. |
| Selected scaling protocol | `bench_add` pins one 4096-wide row, 64 varied rows, and a 1024 narrow-row control; two warmups and ten alternating-order pairs use matched static builds and retain raw time plus whole-process memory context | Day 12 must freeze exact source/binary hashes and collect without changing profiles, ordering, sample policy, or claim limits. |
| Day 12 selected profile | 60/60 correctness-qualified samples validate; baseline/candidate median ms are 13.431/0.1515 wide, 2.155/0.247 varied, and 3.5605/0.219 narrow; candidate is faster in every pair and timing ranges are separated | Limit statements to this exact host/source/compiler/workload set; independently replicate before broader language. |
| Day 12 memory context | RSS and footprint ranges overlap on all fixtures; candidate pair directions are mixed | No reduced-memory claim; use allocation instrumentation if a later sprint needs one. |
| Day 13 evidence freeze | Contract/complexity wording and selected timing are reconciled; current source is hashed; Day 12 history is retained; comment-only rebuild reproduces the measured object and executable; focused quality, CMake parity, and sanitizer gates pass | Day 14 complete repository gates remain the only sprint-closeout condition. |
| Day 14 completion | All six Epic items map to retained artifacts; full Make/CMake tests, sanitizers, 32/64-bit package consumers, documentation, source-list, raw-profile, and hash gates pass | Closed. Preserve claim boundaries and use the bounded Sprint 228 handoff. |

## Day 1 verification

### Commands and outcomes

| Command | Outcome | Interpretation |
| --- | --- | --- |
| `cmake -S . -B build/s227-day1-cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` | Exit 0; AppleClang 17.0.0 | Dedicated baseline tree; existing nonfatal double-promotion warnings remain outside this sprint's Day 1 scope. |
| `cmake --build build/s227-day1-cmake --target test_sparse_arith test_matmul --parallel 4` | Exit 0 | Built the library and both focused test binaries. |
| `ctest --test-dir build/s227-day1-cmake --output-on-failure -R '^(test_sparse_arith\|test_matmul)$'` | 2/2 CTests pass in 0.83 s | Existing suites are green but contain no nonidentity arithmetic regressions. |
| Checked-in-baseline probe compiled against the dedicated static library | Exit 0 | All three APIs return success for a row-swapped operand and disagree with logical dense arithmetic; self-alias in-place scaling is not snapshot-stable. Probe source remains ignored under `build/`. |
| Add fail-after sweep, indices 0–7 | Six `SPARSE_ERR_ALLOC`/NULL outputs followed by two successes | Current injectable coverage reaches six result-shell buffers but not output-node slab allocation. |
| `/usr/bin/time -l build/s227-day1-cmake/test_sparse_arith` | Exit 0; 0.00 s real; 815,104-byte maximum resident set size; 569,344-byte peak footprint | Whole-process local baseline, not an isolated add benchmark. |
| `/usr/bin/time -l build/s227-day1-cmake/test_matmul` | Exit 0; 0.07 s real; 3,526,656-byte maximum resident set size; 3,215,360-byte peak footprint | Whole-process local baseline, retained only as context. |
| `make docs-check` after Day 1 artifacts | Exit 0; API coverage passed for all 18 checked-in public headers | Required because tracked Day 1 changes are documentation and fixture data only. |

### Day 1 completion state

- API and implementation trace: complete.
- Current identity/nonidentity, ownership, alias, drop-threshold, and failure inventory: complete for planning; ambiguous behavior is explicitly open for Day 2.
- Independent dense-oracle fixture definitions and deterministic seeds: complete.
- Focused test, allocation-count, time, and peak-memory baseline: complete.
- Production C/header change: none. The conditional `make format && make lint && make test` gate is therefore not triggered on Day 1.

## Day 2 verification

- Inspected existing original-state rejection conventions in ILU, IC, bidiagonalization, QR, SVD, analysis, and factorization paths; `SPARSE_ERR_BADARG` is the established result.
- Verified `sparse_matrix_require_original_state` checks factor state plus row, inverse-row, column, and inverse-column identity without adding a public symbol.
- Reconciled all 14 Day 1 fixtures and 22 Day 2 pointer/shape/state/alias/allocation cases with explicit status, output, and mutation expectations.
- Corrected the Day 1 baseline record: unlike matmul, `sparse_add` validates A/B/C_out together and does not clear output storage when an input pointer is NULL.
- Day 2 tracked changes are Markdown and TSV only; no C/header file changed, so the conditional `make format && make lint && make test` chain is not triggered. Artifact/link/TSV checks and `git diff --check` passed; `make docs-check` passed with all 18 checked-in public headers covered.

## Day 3 verification

- The always-on fixture tests construct and verify all three valid nonidentity permutation categories on a rectangular matrix, exercise independent dense add and matmul arithmetic, and prove snapshot mutation detection.
- The opt-in rejection test covers 27 combinations: each API, left/right/both operands, and row-only/column-only/row-plus-column maps. Against the pre-repair implementation it produces the required failure: 27 status mismatches, 18 unexpected outputs, and nine destination mutations.
- The experimental gate is temporary test-first scaffolding, not a waiver. It must be removed when the shared production validation is complete.
- `make format`, `make lint`, `make test`, `make docs-check`, `git diff --check`, and the 30-row/9-column TSV integrity check all pass.

## Day 4 verification

- The focused default arithmetic binary passes 32 tests and 388 assertions; only the explicit Day 5 matmul regression is skipped.
- With experimental tests enabled, every Day 4 test still passes and only matmul reports the expected nine status/output failures.
- The allocation sentinel remains armed across a rejected add, proving rejection occurs before result allocation; it then fires on a valid add and a reset retry succeeds.
- Repository-wide C/header call-site inspection found no production callers to migrate and no public ABI edit.
- `make format`, `make lint`, `make test`, `make docs-check`, strict standalone test compilation, and `git diff --check` all pass.

## Day 5 verification

- The complete 27-combination cross-API permutation matrix is always on; no experimental arithmetic contract test remains.
- Matmul rejects inverse-only and factor-state violations on either operand after shape checks and before allocation, clears valid output storage according to its existing pointer contract, and preserves both inputs.
- An armed allocation failure survives state rejection, fires at the first valid result allocation, and permits a successful retry with valid row/column structure.
- The focused arithmetic and matmul CTests pass 2/2. Both Cholesky reconstruction fixtures now copy physical factor values into fresh original-state matrices, while direct factor-state use is tested as rejection.
- Public comments state the precondition, `SPARSE_ERR_BADARG`, and alias behavior without changing declarations or layouts.
- `make format`, `make lint`, `make test`, `make docs-check`, strict standalone compilation of all changed tests, and `git diff --check` pass.

## Day 6 verification

- The add merge has no `sparse_insert` call; retained nodes are initialized once and linked in O(1) through row and column tails.
- A deterministic 4x5 case proves overlap, disjoint, cancellation, empty-row, and B-only output with strict row/column ordering, exact counts, identity maps, fresh metadata, dense-oracle values, and unchanged inputs.
- Column-tail workspace and first-node slab failures return `SPARSE_ERR_ALLOC`, leave output NULL, clean partial ownership, preserve inputs, and permit a successful retry.
- The focused arithmetic suite passes 39 tests and 595 assertions; strict standalone implementation/test compilation passes.
- `make format`, `make lint`, `make test`, strict standalone compilation, `make docs-check`, and `git diff --check` pass.

## Day 7 verification

- The edge corpus covers empty inputs and rows, disjoint and identical-column patterns, complete cancellation, zero coefficients, rectangular matrices, and a dense 64-column row.
- Every result and its copy match exact dense-oracle values, retain strict row/column ordering, have exact `nnz`, and use the same node pointer exactly once in each corresponding row and column chain.
- Ninety-six deterministic randomized cases use seed `0x2270add7`, dimensions 1–6, sparse integer inputs, coefficients -2 through 2, and the frozen drop threshold.
- The builder audit found checked shell/workspace byte calculations, bounded row and `nnz` increments, no fallible operation after node allocation and before publication, and a single finish/abort ownership boundary.
- Clean 32-bit and 64-bit focused suites each pass 41 tests and 4,447 assertions. The focused ASan/UBSan run passes without a sanitizer finding; local Apple LeakSanitizer is unavailable.
- `make format`, strict standalone test compilation, `make lint`, `make test`, `make docs-check`, and `git diff --check` pass.

## Day 8 verification

- The failure fixture emits `SPARSE_NODES_PER_SLAB + 1` nodes in one column, forcing exactly two output slabs without depending on the default slab size.
- Fail indices 0–9 cover the result object, six shell arrays, column-tail workspace, first slab, and second slab; each returns `SPARSE_ERR_ALLOC`, NULL output, unchanged inputs, and permits an immediate valid retry.
- Fail indices 10 and 11 both succeed, proving the final-allocation and one-beyond-final boundaries.
- Every successful result has exact values and `nnz`, identity maps, exactly two slabs, one node per row, a monotonic column chain, and identical node pointers from row and column traversal.
- Focused 32-bit, 64-bit, and ASan/UBSan suites pass. macOS `leaks` reports 0 leaks and 0 leaked bytes.
- Source inspection confirms `sparse_add` retains one two-pointer input merge and direct row/column-tail append with no output scan or `sparse_insert` call.
- `make format`, strict standalone test compilation, `make lint`, `make test`, `make docs-check`, and `git diff --check` pass.

## Day 9 verification

- Compact snapshots now compare shell storage, maps, factor/reorder pointers, pool state, node addresses, coordinates, values, and both link directions in addition to logical values and metadata.
- One ownership matrix proves exact input and stale-owner state for NULL output, NULL input, shape, original-state, and allocation errors with the frozen output-clearing precedence.
- A complete-cancellation 6x6 sweep fails cleanly at allocations 0–7, retries after each failure, and succeeds at indices 8 and 9 with no node slab.
- Caller variables holding A and stale prior outputs behave according to the documented non-owning output-storage contract; same-input out-of-place addition remains supported.
- Forty-eight wide rectangular cases use seed `0x22709add`; focused 32-bit, 64-bit, and ASan/UBSan runs each pass 46 tests and 79,486 assertions.
- `make format`, strict standalone test compilation, `make lint`, `make test`, `make docs-check`, and `git diff --check` pass.

## Day 10 verification

- The existing one-slab in-place allocation site is injectable without changing the public `sparse_insert` path or ABI; fail index 0 returns `SPARSE_ERR_ALLOC`, while indices 1 and 2 succeed.
- The failure state retains exact scaled/prefix values, existing node identities, one owned slab, coherent `nnz`, identity maps, cleared factor metadata, and valid row/column links; B matches its exact snapshot.
- Copy, lookup, out-of-place addition, a subsequent in-place call, destruction, and a clean-state retry all succeed after the injected failure.
- Shared-pattern, cancellation, scaling, B-only, A-only, empty-row, and `alpha=0` free-list cases match independent expected values.
- Focused 32-bit, 64-bit, and ASan/UBSan runs each pass 49 tests and 79,666 assertions.
- `make format`, `make lint`, `make test`, `make docs-check`, and `git diff --check` pass.

## Day 11 verification

- The normal 32-bit integrated sequence passes 49 arithmetic tests with 79,666 assertions, 41
  matmul tests with 372,263 assertions, and five Sprint 4 integration tests with 49 assertions:
  95 tests and 451,978 assertions total.
- `bench_add` builds through the maintained Make route and emits one validated CSV row for each
  pinned workload ID. Fixture construction and complete A/B/C checks remain outside the timer.
- The wide, varied, and narrow profiles freeze exact shape, input/output `nnz`, and maximum output
  row width. Day 11 records no accepted timing or memory comparison.
- The Day 12 protocol freezes matched static builds, two warmups, ten alternating-order pairs,
  host/environment controls, per-command and collection caps, raw retention, invalidation rules,
  metric interpretation, and selected-workload claim limits.
- `make format`, `make lint`, `make test`, `make docs-check`, CMake benchmark compilation,
  benchmark execution, and `git diff --check` pass.

## Day 12 verification

- The baseline is commit `fcdeaaee856821e8c801d805c63b5fd40eb89b2f`; the candidate is tied to
  the exact Sprint 227 source manifest, tracked diff, archives, generated headers, executables,
  compiler, flags, 32-bit index width, host, OS, power state, environment, and UTC window.
- Two warmups per fixture/variant and ten alternating-order pairs per fixture completed in one
  4.206-second collection. All 60 measured processes exited zero; no timeout, replacement,
  selective rerun, or discarded observation occurred.
- The strict parser validates exactly 300 raw stdout/stderr/time/status/meta files, every workload
  identity/count, pair order, invocation, timeout cap, status, elapsed value, RSS, and footprint.
  The raw SHA-256 manifest verifies all 300 files.
- Candidate timing is below baseline timing in all 30 pairs. Median baseline/candidate kernel
  times are 13.431/0.1515 ms wide, 2.155/0.247 ms varied, and 3.5605/0.219 ms narrow. The retained
  high observations do not create range overlap.
- Whole-process RSS and footprint ranges overlap and pair directions are mixed. No memory
  improvement, portable speed, whole-library, superiority, or state-of-the-art claim is made.
- `python3 -m py_compile`, the strict profile summarizer, raw and source manifest verification,
  `make docs-check`, and `git diff --check` pass. No C/header file changed after Day 11's passing
  `make format && make lint && make test` chain.

## Day 13 verification

- Public and internal documentation now states the original-state guard, exact ownership and
  alias boundaries, in-place partial-failure behavior, out-of-place linear construction bound,
  and pattern-dependent in-place cost without changing any public declaration or layout.
- The claim ledger traces every accepted timing statement to the 60-row sample table and
  300-file raw bundle, explicitly rejects a memory claim, and prohibits portable, broad-library,
  superiority, release, and state-of-the-art language.
- The historical Day 12 source manifest remains untouched. A clean exact-flag rebuild produced
  the same `sparse_matrix.o` SHA-256 and byte-identical candidate benchmark executable; the archive
  container changed only because Apple `ar` refreshed member timestamps.
- `make format` and `make lint` pass. The focused Make sequence passes 95 tests and 451,978
  assertions across arithmetic, matmul, and Sprint 4 integration.
- `quality-review-cmake-compile` passes a clean full build and registration audit: 63 CMake tests
  equal 62 Make tests plus one focused CTest-only selector.
- A fresh ASan/UBSan build passes all three focused CTests with no sanitizer finding.
- `make docs-check`, strict profile revalidation, raw/current-source hash-manifest checks, and
  `git diff --check` pass.
- Day 14 still owns `make format && make lint && make test`, documentation/integrity checks, and
  the final sprint-closeout decision.

## Day 14 verification

- `make format`, `make format-check`, `make lint`, and `make test` pass on the final source tree.
  The Make lane runs all 62 registered test binaries with no failure.
- A clean `quality-review-cmake` tree builds all library, test, benchmark, and example targets;
  registration parity is exact at 63 CMake tests versus 62 Make tests plus one focused selector;
  all 63 CTests pass in 136.28 seconds.
- A fresh ASan/UBSan tree passes arithmetic, matmul, and Sprint 4 integration, 3/3 CTests, with no
  sanitizer finding. The final non-instrumented focused rerun passes 95 tests and 451,978
  assertions.
- Make width (28), Make install (26), CMake install (38), CMake installed width (20), pkg-config
  installed width/link (40), installed archive-swap (12), and static contract mutation (26) checks
  all pass. The selected CMake 64-bit functional CTest also passes.
- `make source-list-check` confirms 49 library sources; `make docs-check` covers all 18 checked-in
  public headers. Both profile scripts compile, the strict parser revalidates 60 samples from 300
  raw files, raw/current-source SHA-256 manifests verify, and all Sprint 227 relative Markdown
  targets resolve.
- The [closeout evidence index](artifacts/day14-closeout-evidence-index.md) maps Items 227.1–227.6
  to exact evidence and retains the nonidentity rejection, in-place partial-update, local timing,
  inconclusive memory, hosted-platform, and pre-existing-warning boundaries.
- Sprint 227 is complete. No release, portable-performance, reduced-memory, superiority, or
  state-of-the-art conclusion is inferred.

