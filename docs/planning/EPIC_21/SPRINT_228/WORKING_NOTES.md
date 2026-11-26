# Sprint 228 Working Notes

**Branch:** `sprint-228`

**Base commit:** `363600ea27e854a4334cff5fa06ace6529045ccd` (merge of PR #252 on `master`)

**Scope:** Remove the additional `m×n` reconstruction accumulator from the default sparse low-rank output path while preserving accepted numerical, structural, ownership, and failure behavior.

**Current decision:** Sprint 228 is complete for its scoped reconstruction-memory objective. The per-cell outer route is the maintained default, exact `on` selects it explicitly, exact `off` retains dense compatibility behavior, and every other set value including an empty string fails closed before SVD work. Non-finite `drop_tol`, matrix data, factors, products, sums, and reconstructed values are rejected under the frozen status and output contract. The differential harness judges both explicit routes independently across 42 analytical or long-double-factor-oracle cells. Repeated mixed-route replay is deterministic and isolated across success, selector, allocation, numeric, high-threshold, and all-zero calls. Every economy-SVD, wide-transpose, all-zero, maintained-output, and compatibility allocation reached by the public sparse low-rank call is injectable. Exhaustive sweeps cross 26 default-square, 37 default-wide, 23 zero-shortcut, and 27 compatibility allocation boundaries through one-beyond and clean retry. The Day 11 exact counters prove that the default route adds no `m*n` reconstruction accumulator while retaining factor, decomposition, shell/map, and node-slab accounting. Day 12 publishes that narrow result while rejecting whole-SVD sparse-memory, lower-peak-memory, universal speed, and portable performance language. Day 13 passes focused Make/CMake execution, exact 63-versus-62-plus-one registration parity, both index widths, ASan/UBSan at both widths, Homebrew-Clang leak detection, strict lint, formatting, docs, and generated dependency checks. Day 14 passes the complete ordered Make quality chain, all 63 CTests in a fresh reviewed CMake tree, all 290 raw-evidence hashes, documentation/source checks, ABI/diff review, and debris audit. Square and wide timing remain blocked under the frozen gate. No public ABI changed.

## Day-by-day ledger

| Day | Work and decision | Evidence | Open condition |
| --- | --- | --- | --- |
| 1 | Traced both reconstruction routes through public API, SVD factorization, sparse output construction, tests, and selector parsing; separated factor, route-scratch, shell/map, and node-slab ownership; froze observed status/output/order/drop behavior; ran default and forced-outer focused baselines; and recorded source-derived allocation counts. | [Day 1 reconstruction path and allocation inventory](artifacts/day1-reconstruction-path-and-allocation-inventory.md) | Day 2 must decide accepted non-finite tolerance behavior, compatibility-selector status, independent oracle/tolerance rules, pinned fixtures, and exact memory/timing protocol. |
| 2 | Froze validation precedence, finite numerical behavior, default and compatibility selector semantics, output structure/ownership, independent analytical and long-double oracle rules, thirteen correctness/status fixtures, four measurement cells, component-level memory accounting, alternating paired execution, caps, invalidation rules, and bounded promotion thresholds. | [Day 2 reconstruction contract and measurement protocol](artifacts/day2-reconstruction-contract-and-measurement-protocol.md), [fixture matrix](artifacts/day2-reconstruction-fixture-matrix.tsv) | Day 3 must encode the core square/tall/wide route-to-oracle harness without weakening the frozen ambiguity band or using one production route as the other's oracle. |
| 3 | Added explicit dense/outer route execution with environment restoration; analytical square/tall/wide oracles at rank one, intermediate rank, and maximum accepted rank; exact status, shape, value, `nnz`, ordering, duplicate, metadata, and input-immutability checks; and deterministic replay evidence for all 18 route cells. | [Day 3 core differential harness](artifacts/day3-core-differential-harness.md), `tests/test_svd.c` | Day 4 extends this foundation to zero, rank-deficient, clustered, cancellation, threshold-equality, and tiny-scale fixtures. |
| 4 | Added zero and empty-pattern controls, clustered and rank-deficient spectra, an exact one-ULP threshold classifier, ordinary and `2^500`-scale signed cancellation, and `2^-900` tiny-scale coverage; both routes are independently checked for values, classification, finite output, and sparse invariants. | [Day 4 degenerate and numerical differential corpus](artifacts/day4-degenerate-and-numerical-differential-corpus.md), `tests/test_svd.c` | Day 5 adds seeded bounded random coverage and fixes the maintained implementation and future status-test seams before promotion work begins. |
| 5 | Added four xorshift64* replay fixtures spanning square/tall/wide, ranks 1–4, 25–85% density, mixed signs, and multiple tolerances; independently accumulated long-double factor oracles; passed the same corpus under 32- and 64-bit indices; audited allocation arithmetic; and fixed the checked selector, construction, ownership, cleanup, and future regression design. | [Day 5 differential acceptance and promotion design](artifacts/day5-differential-acceptance-and-promotion-design.md), `tests/test_svd.c` | Day 6 hardens finite arithmetic and cleanup in the outer route before any default change. |
| 6 | Hardened the outer helper with exact economy-factor metadata checks, checked sigma/U/Vt byte extents, finite singular/vector/product/sum validation, NULL-on-failure publication, partial-output destruction, and retry coverage; the frozen corpus remains unchanged. | [Day 6 outer-route hardening](artifacts/day6-outer-route-hardening.md), `src/sparse_svd.c`, `src/sparse_svd_internal.h`, `tests/test_svd.c` | Day 7 may promote the hardened route only with checked selector parsing, default-route proof, and caller/documentation audit. |
| 7 | Promoted the hardened per-cell route to default; retained exact `off` dense compatibility; rejected malformed selectors, non-finite tolerances, and non-finite matrix data under the frozen precedence; added default/parity/isolation regressions; and updated active API and maintainer documentation after auditing callers. | [Day 7 default promotion and selector contract](artifacts/day7-default-promotion-and-selector.md), `src/sparse_svd.c`, `include/sparse_svd.h`, `tests/test_svd.c`, `README.md`, `docs/maintainer_guide.md` | Day 8 supplements source-level route proof with dynamic allocation instrumentation and begins deterministic fail-index coverage. |
| 8 | Completed alternating valid/invalid selector replay; proved the ordinary route returns before exact-`off` dense scratch; added internal injectable matrix construction/insertion; swept all eight reconstruction output allocations through success and retry; and enumerated decomposition, factor, transpose, zero, compatibility, shell/map, and node allocations. | [Day 8 default route and initial allocation matrix](artifacts/day8-default-route-and-initial-allocation-matrix.md), `src/sparse_matrix.c`, `src/sparse_matrix_internal.h`, `src/sparse_svd.c`, `tests/test_svd.c` | Day 9 converts the remaining raw economy-SVD/wide/zero/compatibility allocations and completes exhaustive public-call failure sweeps. |
| 9 | Routed every public sparse low-rank economy allocation through the checked hook; added exhaustive square, wide, zero, and compatibility fail-index sweeps; froze negative/non-finite tolerance and rank behavior; retained dimension/factor/arithmetic numeric regressions; and passed focused ASan/LeakSanitizer/UBSan validation. | [Day 9 allocation and numeric failure evidence](artifacts/day9-allocation-and-numeric-failure-evidence.md), `src/sparse_bidiag.c`, `src/sparse_matrix.c`, `src/sparse_matrix_build_internal.c`, `src/sparse_svd.c`, `tests/test_svd.c` | Day 10 alternates successful and failed routes in one process and closes the final lifecycle/contract map. |
| 10 | Alternated default and compatibility successes with selector, allocation, and numeric failures in one process; proved immediate retry, destruction/reconstruction, unchanged input, and reusable high-threshold/all-zero empty output; and mapped every frozen precedence, ownership, cleanup, and structural clause to automated evidence. | [Day 10 lifecycle and contract closure matrix](artifacts/day10-lifecycle-and-contract-closure-matrix.md), `tests/test_svd.c` | Day 11 executes the frozen reconstruction-memory protocol without weakening these lifecycle gates. |
| 11 | Added source-bound exact allocation counters and a fresh-process collector; retained 16 warmups and 80 measured invocations with raw streams, receipts, metadata, hashes, and noisy trials; proved zero default reconstruction scratch versus one 2,097,152-byte dense accumulator on nonzero cells; and kept all retained memory categories visible. | [Day 11 reconstruction memory packet](artifacts/day11-reconstruction-memory-packet.md), [component accounting](artifacts/day11-memory-component-accounting.tsv), `benchmarks/bench_svd_lowrank_memory.c`, `scripts/run_sprint228_memory_packet.py` | Memory completion passes. Promotion timing remains blocked because square and wide ratios exceeded the frozen limits and the packet used a dirty-worktree development override. |
| 12 | Analyzed all retained timing pairs; found square and wide exceed the median threshold in 8/10 pairs and each has two pairs above the per-pair cap; documented exact selector, threshold, error, factor, decomposition, output, and RSS boundaries; and removed any basis for whole-SVD sparse-memory or speed language. | [Day 12 documentation and timing limitations](artifacts/day12-documentation-and-timing-limitations.md), `include/sparse_svd.h`, `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md` | Day 13 must run the full focused, cross-build, sanitizer, generated-doc, and dependency-surface gates. Timing promotion remains blocked. |
| 13 | Passed the complete focused SVD/partial-SVD corpus in fresh Make and CMake trees at both index widths; proved exact CTest registration parity; passed ASan/UBSan at both widths and a leak-enabled Homebrew-Clang run; and audited warnings, formatting, docs, collector syntax, and source registration. | [Day 13 focused cross-build and sanitizer validation](artifacts/day13-focused-cross-build-and-sanitizer-validation.md) | Day 14 must run the complete final-tree quality and evidence-integrity chain and issue the sprint closeout decision. Timing promotion remains blocked. |
| 14 | Passed the ordered final Make quality chain and complete reviewed CMake execution; reverified raw hashes, ABI, selector behavior, registration, generated docs, source lists, and workspace contents; and issued the narrow completion decision and no-claim handoff. | [Day 14 evidence index](artifacts/day14-closeout-evidence-index.md), [final quality and handoff](artifacts/day14-final-quality-and-handoff.md), [retrospective input](artifacts/day14-retrospective-input.md) | Sprint work is complete. Bind the packet to the eventual review commit SHA; timing remains blocked and carries no speed claim. |

## Evidence ledger

| Evidence | Current state | Required next proof |
| --- | --- | --- |
| Source identity | Branch work descends from `363600ea27e854a4334cff5fa06ace6529045ccd`; Days 6–9 modify SVD, bidiagonal, matrix-internal, test, and documentation sources | Re-record the final branch commit during sprint closeout. |
| Route selection | Unset and exact `on` use per-cell accumulation; exact `off` alone reaches dense compatibility; all other set values return `SPARSE_ERR_BADARG` before SVD work | Day 8 adds dynamic selection/allocation proof without changing this frozen behavior. |
| SVD factor storage | Economy SVD retains `sigma[k]`, `U[m×k]`, and `Vt[k×n]`; bidiagonalization also owns dense and reflector temporaries | Must remain separate from any claim about removed reconstruction scratch. |
| Default route scratch | Scalar per-cell accumulation only; source dispatch makes the checked `calloc(m*n, sizeof(double))` reachable only through exact `off` compatibility | Day 8 adds exact dynamic allocation evidence and starts fail-index sweeps. |
| Outer route scratch | Scalar per-cell accumulation only; sparse shell/maps and output slabs are still allocated | Differential, failure, and memory evidence must verify the route before promotion. |
| Output ownership | Success transfers one identity-permutation `SparseMatrix` to caller; failures after a valid output pointer leave `*result == NULL`; mixed-call replay proves destruction and reuse across both routes | Preserve during Day 11 measurement instrumentation. |
| Focused default baseline | 114 tests, 2,067 assertions, zero failures; 0.136 s harness-reported time | Preserve and extend with independent route/oracle cases. |
| Focused forced-outer baseline | 114 tests, 2,067 assertions, zero failures; 0.142 s harness-reported time | These are whole-suite observations, not accepted performance evidence. |
| Existing cross-route cells | Synthetic 32×32 plus `nos4`/`bcsstk04` at ranks 10 and 50 report relative Frobenius difference `0.000e+00` | Expand to pinned square/tall/wide, zero, rank-deficient, clustered, cancellation, and threshold-boundary cases. |
| Allocation injection | Every allocation reachable from the public economy sparse low-rank call is injectable; square/wide/zero/compatibility first-success indices are 26/37/23/27; one-shot failure followed immediately by success leaves no stale hook or result state | Preserve during measurement work. |
| Non-finite tolerance | Baseline accepts NaN or positive infinity and produces empty success; frozen target returns `SPARSE_ERR_BADARG` before SVD work | Days 6–8 implement fail-closed behavior and the enumerated precedence regressions. |
| Frozen selector contract | Target default is outer; unset or `on` selects outer, `off` selects dense compatibility, and any other set value (including empty) returns `SPARSE_ERR_BADARG` before SVD work | Days 6–8 implement the promotion and the Day 5 precedence/isolation test inventory without changing ABI. |
| Frozen numeric contract | NaN/infinite `drop_tol` returns `SPARSE_ERR_BADARG`; a non-finite reconstructed cell returns `SPARSE_ERR_NUMERIC`; finite `drop_tol <= 0` uses `DBL_EPSILON * sigma[0]` | Days 3–10 must encode argument, extreme-scale, non-finite, threshold, and cleanup cases. |
| Frozen fixture matrix | Thirteen pinned correctness/status rows cover square, tall, wide, zero, rank-deficient, clustered, exact cancellation, exact threshold equality, tiny finite scale, invalid tolerances, and non-finite matrix data | The correctness corpus is implemented; Days 6–10 must implement the remaining status, precedence, and cleanup rows with production hardening. |
| Measurement protocol | Four cells; same binary; two warmups; ten alternating pairs; 30 s per invocation; 15 min collection cap; exact scratch counters primary, RSS contextual | Day 11 executes once after the implementation and quality prerequisites are frozen. |
| Day 11 memory packet | All 40 default measured invocations report 0 reconstruction-scratch bytes/allocations; all 30 nonzero dense invocations report 2,097,152 bytes/1 allocation; route pairs retain identical factor, decomposition, shell/map, slab, checksum, and `nnz` evidence | Memory claim accepted for development. Clean exact-candidate recollection and passing square/wide timing limits remain required for promotion timing. |
| Day 12 claim boundary | Public and maintainer docs distinguish reconstruction scratch from dense economy factors, decomposition work, shell/maps, and retained node slabs; all ten timing ratios per workload remain published | No faster, whole-call peak-memory, portable performance, external ranking, or release claim; Day 13 owns cross-build and sanitizer validation. |
| Day 13 focused validation | Both widths pass 151 focused Make tests and 3/3 selected CTests; 63 CTests equal 62 Make tests plus one focused selector; ASan/UBSan and leak-enabled focused runs pass; format, lint, docs, source-list, and collector syntax gates pass | Day 14 owns complete final-tree test execution and sprint closeout; timing remains blocked. |
| Day 14 closeout | Ordered `make format && make lint && make test` passes; a fresh reviewed CMake tree passes 63/63 tests with exact 62+1 registration parity; 290/290 raw hashes and all supplemental checks pass | Closed for the narrow reconstruction-memory objective. Record the eventual commit SHA; do not promote timing or broader memory claims. |
| Core differential harness | Nine deterministic shape/rank replays × two explicit routes; both routes independently satisfy analytical values and complete sparse-structure invariants; all 18 cells have exact zero observed error | Day 4 adds degenerate, clustered, cancellation, threshold-boundary, and tiny-scale cases without replacing the independent oracle. |
| Numerical differential corpus | Eight deterministic fixtures × two routes cover zero/empty patterns, clustered and deficient rank, one-ULP threshold classification, ordinary/large cancellation, and tiny finite scale; all 16 cells pass analytical and structural gates | Day 5 adds deterministic randomized replay and enumerates the invalid-tolerance, selector, and non-finite-data status regressions for Days 6–10. |
| Seeded differential replay | Four fixed seeds × two routes add bounded shapes, ranks, densities, signs, and tolerances; maximum observed relative error is `1.146e-16`; ambiguity-band fixtures are rejected | Preserve exact seeds and generator while production hardening proceeds. |
| Index-width and overflow audit | Identical 123-test corpus passes in isolated 32- and 64-bit builds; 64-bit typedef/header/archive identity lane passes; factor, dense `m*n`, and output allocation arithmetic has named checked bounds | Re-run both widths after route hardening/default promotion; do not broaden this to universal 64-bit coverage. |
| Promotion design | Checked parser, outer default, exact `on`/`off`, invalid-value failure, common output ownership, and cleanup transitions are implemented internally with no ABI change | Preserve through Day 8 allocation instrumentation and failure injection. |
| Hardened outer route | Factor metadata and sigma/U/Vt byte extents are checked before indexing; every used factor, scaled product, term, and running sum must be finite; failures destroy partial output and leave NULL; retry succeeds | Preserve these boundaries through Day 8 allocation instrumentation and failure injection. |
| Promoted default | Ordinary calls and exact `on` use the hardened route and return before the exact-`off` dense accumulator; explicit `off` retains bounded rollback diagnostics; malformed values fail closed | Preserve this dispatch while Day 9 extends failure injection across the complete public call. |
| Initial allocation-failure matrix | Eight reconstruction-boundary allocations are injectable: shell, six map/header arrays, and first node slab; indices 0–7 fail cleanly, index 8 and a clean retry succeed | Day 9 extends injection through economy SVD, wide transpose, all-zero, and exact-`off` public-call paths. |
| Exhaustive public-call failure matrix | All indices below 26/37/23/27 fail with `SPARSE_ERR_ALLOC`, NULL output, unchanged input, and successful retry for default square/default wide/zero/compatibility; one index beyond each last allocation also succeeds | Closed and retained. |
| Mixed-call lifecycle | Two in-process cycles cover default success, invalid selector, one-shot allocation failure, immediate route-specific retry, numeric failure, successful retry, high-threshold empty output, and all-zero empty output | Closed by Day 10; preserve as a regression gate. |
| Contract closure | Frozen validation precedence, selector, threshold, structure, ownership, cleanup, unchanged-input, and retry clauses map to named automated tests | Closed by the Day 10 matrix; measurement claims remain separate. |

## Day 1 verification

### Environment

- Host: `Darwin yog-sothoth 24.6.0`, x86_64.
- Compiler: Apple Clang 17.0.0 (`clang-1700.6.4.2`).
- Make: GNU Make 3.81.
- CMake: 4.4.4.
- Make build flags observed: C11, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2`, `SPARSE_IDX_BITS=32`.

### Commands and outcomes

| Command | Outcome | Interpretation |
| --- | --- | --- |
| `make build/test_svd` | Exit 0 | Focused 32-bit Make binary built with the repository's maintained flags. |
| `env -u SPARSE_SVD_LOWRANK_OUTER ./build/test_svd` | Exit 0; 114 tests, 2,067 assertions, 0 failures; 0.136 s | Current default-route baseline. The binary also runs its explicit cross-route tests. |
| `SPARSE_SVD_LOWRANK_OUTER=on ./build/test_svd` | Exit 0; 114 tests, 2,067 assertions, 0 failures; 0.142 s | Forced-outer baseline for low-rank tests not internally overriding the selector. Not a controlled timing comparison. |
| Source allocation audit of `src/sparse_bidiag.c`, `src/sparse_svd.c`, and `src/sparse_matrix.c` | Complete | Provides formulas and the 32×32/rank-4 representative count in the Day 1 artifact. |

### Completion state

- Both route call graphs and owners: complete.
- Dense-factor versus route-specific scratch separation: complete.
- Current status, output, ordering, duplicate, zero, and threshold behavior: complete as observed baseline.
- Exact baseline commands and representative source-derived allocation counts: complete.
- Production C/header changes: none. The conditional full C-quality chain is not triggered by Day 1.

## Day 2 verification

- Reconciled the current public header, implementation, Sprint 29 design/sweep, Epic 21 review, and Day 1 ownership inventory.
- Rejected Sprint 29's historical factor-memory arithmetic as current evidence because it used requested `rank_k` where the implementation retains economy factors at `k = min(m,n)`; retained only its algorithmic context.
- Froze pointer/rank/state/tolerance/selector precedence and status/output guarantees before any production edit.
- Froze analytical expected matrices for diagonal, rectangular, zero, rank-deficient, cancellation, and threshold-equality fixtures; long-double reconstruction is permitted only as a separate reference, never one production route as oracle for the other.
- Froze exact memory categories, four measurement workloads, same-binary paired ordering, time caps, raw files, rejection rules, and promotion thresholds.
- Day 2 tracked changes are Markdown and TSV only. `make docs-check`, fixture-schema checks, and `git diff --check` pass; no C/header change triggers the full C-quality chain.

## Day 3 verification

- Added test-only helpers that preserve and restore `SPARSE_SVD_LOWRANK_OUTER`, invoke exact `off` and `on` routes, and require `SPARSE_OK` with a valid output.
- Replayed square 32×32, tall 9×5, and wide 5×9 diagonal fixtures at rank one, their pinned intermediate rank, and maximum accepted rank: nine deterministic replays and 18 route cells.
- Compared every logical cell to its analytical reconstruction, bounded relative Frobenius error, and checked exact dimensions and `nnz`.
- Walked internal row and column chains to prove sorted ordering, unique cross-linked nodes, finite nonzero storage, identity maps, fresh metadata, and count consistency; verified each input remained unchanged.
- All route cells observed `max_abs = 0.000e+00` and `rel_frob = 0.000e+00`.
- `make format`, `make lint`, and `make test` pass. The focused `build/test_svd` run reports 117 tests, 15,669 assertions, zero failures, and zero skips.
- Production source and public headers are unchanged; Day 3 adds only the maintained test harness and planning evidence.

## Day 4 verification

- Added an analytical dense-fixture helper for non-diagonal matrices while retaining the Day 3 route isolation, environment restoration, complete logical comparisons, and internal row/column-chain checks.
- Added eight deterministic fixture IDs and 16 explicit-route cells: zero input, empty pattern, clustered singular values, exact rank-two dependent columns, one-ULP threshold neighbors, ordinary and `2^500`-scale Hadamard cancellation, and `2^-900` tiny diagonal values.
- The threshold fixture proves `>=` behavior: immediately above and exactly at `0.5` are retained; immediately below is dropped.
- All zero, clustered, threshold, and tiny-scale outputs are exact. Rank-deficient reconstruction has `6.968e-16` relative error; ordinary and large cancellation each have `2.355e-16` relative error. All stored output is finite and structurally valid.
- `make format`, `make lint`, and `make test` pass. The focused `build/test_svd` run reports 122 tests, 16,581 assertions, zero failures, and zero skips.
- Production source and public headers remain unchanged; Day 4 adds only maintained tests and planning evidence.

## Day 5 verification

- Added four replayable xorshift64* fixtures covering 5×5 rank one, 7×4 rank three, 4×7 maximum rank four, and sparse 6×6 rank two inputs at 25–85% requested density with mixed signs and positive tolerances.
- Built expected matrices by independently accumulating accepted economy-SVD factors in `long double`; generated fixtures fail if any reference magnitude enters the frozen threshold ambiguity band.
- Both explicit routes pass every seeded value, `nnz`, normwise, ordering, duplicate, metadata, ownership, and input-immutability check. Maximum relative Frobenius error is `1.146e-16`.
- The focused SVD suite passes in isolated 32- and 64-bit builds with 123 tests and 17,333 assertions. The selected 64-bit identity lane also passes 5 tests and 115 assertions with typedef/header/archive all reporting 64 bits.
- Audited rank bounds, economy-factor products, dense `m*n` bytes, outer factor offsets, shell/maps, and insertion ownership. No unchecked new allocation product is needed by the promoted outer route.
- Fixed the checked parser, default/compatibility dispatch, output-construction, cleanup, and selector/status/environment test inventory. Those behavior-changing regressions land with Days 6–8, not as knowingly failing Day 5 tests.
- Production source and public headers remain unchanged; the pre-promotion corpus is complete, but default promotion remains blocked.

## Day 6 verification

- The frozen 42-cell differential corpus found no numerical, ordering, duplicate, drop-threshold, rectangular, or cancellation discrepancy requiring an algorithmic change.
- Hardened the lower-memory route at its internal boundary: matching economy dimensions and rank are mandatory; sigma, U, and Vt extents are byte-checked; all used factors and arithmetic stages must remain finite.
- Added direct regressions for malformed arguments, impossible 32/64-bit factor extents, non-finite singular/vector values, product overflow, sum overflow, failure after partial construction, NULL output, and successful retry.
- The helper now owns partial output until complete success and publishes no pointer on validation, allocation, insertion, or numerical failure. SVD factors remain borrowed from and freed by the public wrapper.
- The focused 32- and isolated 64-bit suites each pass 126 tests and 17,362 assertions with no failures or skips.
- `make format`, `make lint`, and `make test` pass after the final hardening and regression changes.
- `src/sparse_svd_internal.h` exposes only an internal proof surface; no installed public declaration or structure layout changes.
- Default selection remains unchanged on Day 6. Day 7 owns promotion, checked selector parsing, environment isolation, and public documentation updates.

## Day 7 verification

- Promoted the hardened per-cell route to the ordinary default; source control flow reaches the retained checked `calloc(m*n, sizeof(double))` accumulator only after exact `off` selection.
- Replaced permissive parsing with exact unset/`on`/`off` handling. Empty, case-varied, whitespace-padded, alias-like, and numeric values return `SPARSE_ERR_BADARG` and leave output NULL.
- Applied the frozen validation order before decomposition: pointer, rank, original state, finite tolerance, selector, then finite matrix data. Valid selectors report non-finite matrix storage as `SPARSE_ERR_NUMERIC`.
- Added default-to-explicit-outer equality, dense compatibility parity, all-zero precedence, non-finite tolerance/data, sequential selector, and exact environment-restoration regressions.
- Updated historical corpus tests to request exact `off` and `on`; they continue to report zero cross-route residual on the synthetic and SuiteSparse cells.
- Audited the active example, tutorial, cookbook, benchmark tree, README, maintainer guide, and public header. Callers do not depend on dense scratch; active API documentation now states the promoted behavior and retained-storage boundary.
- The focused 32- and isolated 64-bit suites each pass 129 tests and 17,485 assertions with no failures or skips.
- `make format`, `make lint`, `make test`, `make docs-check`, and `git diff --check` pass after the final Day 7 changes.
- No public declaration, option structure, or ABI layout changed. Day 8 owns dynamic allocation instrumentation and the first complete fail-index boundary.

## Day 8 verification

- Replayed unset baseline followed by `on`, `off`, `on`, `off`, `on`, and repeated invalid values in one process. Every valid output is exactly equal, malformed values fail with NULL output, and the prior environment is restored.
- Re-inspected the promoted control flow: unset/exact `on` enters the outer helper and returns before the checked dense compatibility accumulator; only exact `off` can reach that allocation. Dense bidiagonalization work remains separately documented and is not mislabeled as reconstruction scratch.
- Added internal-only injectable matrix construction and insertion surfaces that reuse the production implementation while routing the matrix object and node slab through the existing failure hook. Public API and ABI remain unchanged.
- Swept the direct maintained reconstruction boundary at indices 0–7: matrix shell, row headers, column headers, row permutation, inverse row, column permutation, inverse column, and first node slab. Every site returns `SPARSE_ERR_ALLOC`, leaves output NULL, preserves factor bytes, and destroys partial state.
- Index 8 is the first successful completion boundary for the pinned two-node fixture; a clean post-sweep retry also succeeds with the same sorted output.
- Enumerated the economy decomposition/factor, wide-transpose, all-zero, maintained output, and exact-`off` compatibility allocations. Raw decomposition and transpose allocations remain explicit Day 9 work, so Day 8 makes no exhaustive whole-call claim.
- The focused 32- and isolated 64-bit suites each pass 131 tests and 17,568 assertions with no failures or skips.
- `make format`, `make lint`, `make test`, `make docs-check`, and `git diff --check` pass after the final Day 8 changes.

## Day 9 verification

- Converted all raw economy low-rank decomposition/factor allocations to the internal checked hook, including bidiagonal work/reflectors, QR copies, economy U/V, Vt, and wide-factor extraction scratch.
- Added injectable transpose construction covering entries, shell/maps, tail arrays, and node slabs; the existing public transpose behavior remains unchanged.
- Swept every public-call allocation through first success for default square (26), default wide (37), all-zero (23), and exact-`off` compatibility (27) paths. Every lower index returns `SPARSE_ERR_ALLOC` with NULL output and unchanged input.
- One index beyond each final allocation and a clean retry both reproduce the uninjected baseline exactly. The compatibility sweep includes its dense accumulator; default sweeps do not.
- Added explicit public regressions for invalid ranks; finite negative tolerance defaulting; NaN and both infinities returning `SPARSE_ERR_BADARG`; and non-finite input returning `SPARSE_ERR_NUMERIC`. Existing direct regressions retain impossible-dimension, non-finite-factor, product/sum overflow, partial-output cleanup, and retry coverage.
- The focused 32-bit suite passes 133 tests and 19,219 assertions with no failures or skips.
- An isolated Homebrew Clang 22.1.5 ASan+UBSan build with leak detection passes the same 133 tests and 19,219 assertions without a sanitizer report.
- The isolated 64-bit focused suite passes the same 133 tests and 19,219 assertions.
- `make format`, `make lint`, `make test`, `make docs-check`, and `git diff --check` pass after the final Day 9 changes.

## Day 10 verification

- Added a two-cycle mixed-call replay that alternates unset default, exact `on`, exact `off`, malformed selector, allocation failure, non-finite input, high finite threshold, and all-zero calls in one process.
- The injected allocation failure is followed immediately by a successful call without an explicit allocator reset, proving the one-shot hook and partial ownership state do not leak into the retry.
- Every successful result is caller-destroyed before reuse. Both high-threshold and all-zero calls return fresh, identity-mapped, unfactored, structurally valid empty matrices under the two explicit routes.
- Added a dedicated precedence regression covering NULL publication, rank before state/tolerance/selector, state before tolerance/selector, tolerance before selector, and successful recovery after all rejected calls.
- The contract-to-test matrix records the exact automated owner for validation, selector, numerical, structure, ownership, failure cleanup, and retry clauses.
- The focused 32-bit suite passes 135 tests and 19,333 assertions with no failures or skips.
- The isolated 64-bit suite passes the same 135 tests and 19,333 assertions.
- An isolated Homebrew Clang 22.1.5 ASan+UBSan build with leak detection passes the same suite without a sanitizer report.
- `make format`, `make lint`, `make test`, `make docs-check`, and `git diff --check` pass after the final Day 10 changes.

## Day 11 verification

- Added internal thread-local, source-bound exact counters for decomposition peak extents, retained economy factors, route-specific reconstruction scratch, sparse shell/maps, and output node slabs. The counters do not change the public ABI.
- Added a four-workload fresh-process benchmark and bounded collector. It retains 16 warmups and 80 measured invocations, raw stdout/stderr, per-invocation receipts, commands, metadata, source/binary/library hashes, and a 290-entry SHA-256 manifest.
- The source snapshot remained unchanged during collection. Every invocation returned the expected status, values, checksum, `nnz`, and component accounting; `memory_pass=true` with no collector errors.
- Every default sample records 0 reconstruction-scratch bytes and 0 allocations. Every nonzero dense sample records exactly 2,097,152 bytes and 1 allocation. The all-zero shortcut records 0 for both routes.
- Bidiagonalization/decomposition, retained factors, shell/maps, node slabs, baseline RSS, and peak RSS are reported separately in the Day 11 table. No whole-SVD-memory reduction is inferred from process RSS.
- The square median/max timing ratios are 1.3923/1.5675 and the wide ratios are 1.3771/1.5764, exceeding the frozen 1.25/1.50 limits. The packet therefore records `timing_pass=false` and `protocol_accepted=false`; all failed/noisy samples and both superseded development packets are retained.
- The focused 32-bit suite passes 136 tests and 19,369 assertions with no failures or skips. `make format`, `make lint`, `make test`, `make docs-check`, CMake configuration and benchmark-target compilation, Python byte-compilation, dirty-tree fail-closed validation, all 290 raw-manifest hashes, and `git diff --check` pass on the final Day 11 tree.

## Day 12 verification

- Recomputed every paired timing ratio directly from the retained Day 11 receipts. Square and wide each exceed the 1.25 median threshold in eight of ten pairs and each contain two pairs above 1.50; tall passes and the zero cell remains a contextual control.
- Updated the public SVD header with strict-below drop behavior, equality retention, the finite nonpositive default tolerance, and explicit remaining-storage limits. No declaration or structure layout changed.
- Updated the README, benchmark guide, and maintainer guide with the maintained default, exact-`off` compatibility route, collector commands/caps, component meanings, output-density caveat, and forbidden whole-SVD/speed interpretations.
- Published the exact ten-ratio distributions, hashes, formulas, documentation owners, and claim boundary in the Day 12 limitation record. Every quantitative value points to the retained Day 11 packet or an implementation-owned storage formula.
- `make format`, `make lint`, `make test`, and `make docs-check` pass after the final Day 12 wording; `git diff --check` also reports a clean patch.

## Day 13 verification

- Fresh 32-bit and 64-bit Make trees each pass `test_svd` (136 tests, 19,369 assertions), `test_svd_partial_corpus` (10 tests, 247 assertions), and the selected width lane (5 tests; 111/115 assertions). Typedef, generated header, and archive identities agree at each width.
- One memory-probe smoke pair at each width preserves status, values, `nnz`, and checksum. Default reports zero reconstruction scratch; exact `off` reports one 2,097,152-byte accumulator. These executions are contract checks, not timing evidence.
- Fresh CMake trees at both widths pass 3/3 selected CTests. The maintained complete compile/parity wrapper builds all targets and proves 63 CTests equal 62 Make tests plus one intentional CTest-only selector.
- Apple-Clang ASan+UBSan trees pass the selected three CTests at both widths. A separate Homebrew Clang 22.1.5 ASan+UBSan run passes the SVD and partial-SVD suites with leak detection enabled and no sanitizer report.
- Existing CMake warning classes were inspected: SDK float-valued `NAN`/`INFINITY` macro promotion in unchanged status code and explicit non-finite fixtures, plus an unchanged test-only unused variable. The changed production route emits no new warning, while strict `make lint` passes.
- `make format-check`, `make source-list-check`, `make docs-check`, collector Python byte-compilation, and `git diff --check` pass. Day 14 retains full repository execution and the final closeout decision.

## Day 14 verification

- Ran the required final chain in order: `make format`, `make lint`, and `make test` all pass. Lint includes strict compilation, clang-tidy, and cppcheck over 112 files; the complete Make run executes all 62 registered test binaries.
- A fresh `build/s228-day14-cmake` reviewed-parity tree configures and clean-builds every target, proves 63 CTests equal 62 Make tests plus one intentional focused selector, and passes 63/63 CTests.
- Day 13's source-exact 32/64-bit and sanitizer/leak matrix remains valid because Day 14 adds only closeout documentation after the final C/header state.
- `make format-check`, `make source-list-check` (49 library sources), `make docs-check` (18/18 checked-in public headers), collector Python byte-compilation, and `git diff --check` pass.
- All 290 entries in `build/sprint228-day11-memory/manifest.sha256` reverify. The selected summary remains `memory_pass=true`, `timing_pass=false`, and `protocol_accepted=false` with 16 warmups and 80 measured invocations.
- Public-header review finds documentation changes only: no public declaration, structure, enum, typedef, or symbol changed. Selector and allocation behavior are covered by registered tests.
- `git status --short` contains only intended source, build-definition, benchmark, script, documentation, and Sprint 228 planning changes; generated build and API output remain ignored and no local-only debris is staged for review.
- The exact base commit is `363600ea27e854a4334cff5fa06ace6529045ccd`. Because the sprint diff is not yet committed, handoff explicitly requires recording the eventual review commit SHA rather than presenting the base SHA as final source identity.
- Sprint 228 closes only the additional default reconstruction-accumulator gap. Dense economy factors, decomposition work, output-density risk, and failed square/wide timing remain explicit limitations; no whole-call memory, speed, portable-performance, ranking, or release claim is accepted.
