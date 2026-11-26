# Sprint 228 Retrospective

**Sprint:** 228 - Default Low-Rank Reconstruction Memory

**Duration:** 14 planned days on branch `sprint-228`

**Status:** **Complete for the scoped reconstruction-memory objective.** The maintained default
for `sparse_svd_lowrank_sparse()` reconstructs each output cell directly and allocates no
additional `m*n` reconstruction accumulator; exact selector behavior, numerical and structural
results, ownership, failure cleanup, both index widths, build registration, sanitizers, and the
complete final quality matrix pass. The retained timing protocol failed for square and wide
workloads and used a dirty-tree development override, so it supports no speed or promotion-level
timing claim. Full SVD's remaining dense factor and decomposition storage is unchanged and
documented explicitly.

## Source and Definition of Done

The [Epic 21 project plan](../PROJECT_PLAN.md) and [sprint plan](PLAN.md) allocate 140 estimated
hours across six items and fourteen days. These are planning estimates, not measured labor. The
[working notes](WORKING_NOTES.md), [closeout evidence index](artifacts/day14-closeout-evidence-index.md),
and [final quality decision](artifacts/day14-final-quality-and-handoff.md) reconcile each item,
gate, evidence boundary, and residual risk. The branch began at
`363600ea27e854a4334cff5fa06ace6529045ccd`; the Day 11 development packet additionally binds its
source snapshot, binary, static library, receipts, and raw streams by SHA-256. The final review
commit created after this retrospective is the authoritative identity for the complete sprint
diff.

Completion required removal of the additional dense reconstruction accumulator from the default
route, independently verified accepted values and sparse structure, deterministic selector and
failure behavior, explicit accounting for memory that remains, and a fully passing final-tree
quality matrix. It did not require the per-cell route to beat the compatibility route on time.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 228.1 Baseline And Contract | 20 h | Completed. Froze pointer/rank/state/tolerance/selector precedence, exact drop behavior, ordering, ownership, independent oracle rules, four measurement cells, memory categories, caps, raw retention, and promotion thresholds before implementation changed. |
| 228.2 Differential Corpus | 26 h | Completed. Added analytical and long-double-factor references for square, tall, wide, zero, rank-deficient, clustered, cancellation, threshold-boundary, tiny-scale, and deterministic randomized cases; neither production route is the other's oracle. |
| 228.3 Default Path Promotion | 30 h | Completed. Hardened per-cell reconstruction and made unset/exact `on` select it; retained exact `off` dense compatibility; malformed values fail closed before decomposition; public ABI is unchanged. |
| 228.4 Failure Lifecycle | 24 h | Completed. Routed every reachable economy-SVD, transpose, compatibility, output, and zero-path allocation through the checked test seam; exhaustive sweeps, mixed-call replay, NULL publication, unchanged input, cleanup, destruction, and retry pass. |
| 228.5 Memory Accounting | 20 h | Completed for the narrow claim. Exact counters and 96 retained processes prove zero default reconstruction scratch versus one 2,097,152-byte compatibility accumulator for each nonzero measured cell, while factors, decomposition, shell/maps, slabs, and RSS remain separate. Timing failed and is not promoted. |
| 228.6 Quality Closure | 20 h | Completed. Format, strict lint, all Make/CMake tests, both widths, ASan/UBSan, leak detection, generated API docs, source inventory, evidence hashes, links, ABI review, and patch checks pass. |
| **Total** | **140 h** | **All six planned items and sprint completion criteria closed within the stated claim boundary.** |

## What Went Well

1. **Contract and claim boundaries preceded implementation.** Days 1 and 2 separated the
   reconstruction accumulator from economy factors, bidiagonalization work, sparse metadata, and
   output nodes before selecting a default. They also fixed threshold equality, non-finite input,
   selector, output, and ownership behavior. This prevented a memory-focused change from silently
   becoming a numerical or API change.
2. **Independent evidence avoided circular validation.** Analytical diagonal and structured
   fixtures cover exact expected matrices; nontrivial fixed-seed cases reconstruct accepted
   economy factors in `long double`. Both explicit routes must independently satisfy values,
   norms, dimensions, `nnz`, sorted row and column chains, uniqueness, finite storage, identity
   maps, metadata, and unchanged input. The completed 42-cell corpus does not treat either
   production route as truth for the other.
3. **The default route was hardened before promotion.** Factor dimensions and extents are checked
   before indexing, all used factors and arithmetic stages must be finite, partial outputs remain
   owned until publication, and every error destroys temporary state. Promotion then became a
   small checked dispatch change rather than a simultaneous algorithm-and-policy rewrite.
4. **Allocation sweeps made lifecycle evidence exhaustive.** First success occurs at allocation
   26 for default square, 37 for default wide, 23 for the zero shortcut, and 27 for exact-`off`
   compatibility. Every prior injected allocation fails with `SPARSE_ERR_ALLOC`, NULL output, an
   unchanged input, and a successful immediate retry. One-beyond and mixed-route replay establish
   that the failure hook and ownership state do not leak between calls.
5. **Component counters supported the exact memory claim.** The Day 11 probe records retained
   factor bytes, decomposition peak extents, reconstruction scratch, shell/maps, node slabs, and
   process RSS separately. All 40 measured default calls report zero reconstruction-scratch bytes
   and allocations; all 30 nonzero compatibility calls report one 2,097,152-byte allocation. The
   all-zero shortcut reports no reconstruction scratch on either route.
6. **Unfavorable timing was preserved rather than rationalized away.** The square and wide cells
   each exceeded the median threshold in eight of ten pairs and each contained two ratios above
   the individual-pair limit. No outlier deletion, selective rerun, superseded-packet pooling, or
   threshold change occurred. The result narrowed the accepted claim but strengthened its audit
   trail.
7. **Closeout covered the complete repository surface.** Both index widths, fresh Make and CMake
   trees, exact 63-versus-62-plus-one registration parity, Apple-Clang ASan/UBSan, a leak-enabled
   Homebrew-Clang run, strict compilation, clang-tidy, cppcheck, Doxygen coverage, source-list
   parity, raw hash verification, links, and the complete 62/63 test inventories passed.

## What Did Not Go Well

1. **The optional route initially lacked a promotion-grade error contract.** Selector parsing was
   permissive, non-finite tolerances and data could reach ambiguous behavior, factor metadata and
   arithmetic finiteness needed stronger checks, and partial construction needed a single explicit
   owner. Several days were required to turn an experimental route into a maintained default.
2. **Failure injection did not initially reach the full call graph.** Bidiagonalization, factor
   extraction, transpose construction, matrix shell/maps, node slabs, and compatibility scratch
   used mixed allocation paths. Internal checked seams had to be extended before whole-public-call
   allocation sweeps were possible.
3. **The timing objective did not pass.** Default/dense median ratios were 1.3923 for square and
   1.3771 for wide, with maxima of 1.5675 and 1.5764. Tall passed at 1.2297 median and the zero
   control was contextual, but the frozen gate requires every nonzero workload to satisfy both
   median and per-pair limits. The per-cell route therefore has no accepted speed claim.
4. **The selected raw packet is development evidence rather than release-candidate evidence.** It
   used the documented dirty-worktree override and reports `protocol_accepted=false`. The exact
   reconstruction counters are useful and internally consistent, but a future release-oriented
   timing statement requires clean exact-candidate collection and independent review.
5. **Removing reconstruction scratch does not remove the dominant dense SVD model.** Economy
   `sigma`, `U`, and `Vt`, bidiagonalization and reflector work, sparse matrix maps, and retained
   output nodes remain. A low threshold can retain nearly every output cell. The public wording
   needed repeated tightening to prevent “sparse output API” from being mistaken for end-to-end
   sparse-memory SVD.
6. **Complete validation remains expensive.** Focused SVD tests are quick, but dual-width builds,
   sanitizers, leak checks, full lint, clean CMake compilation, all CTests, and the long nested
   dissection test dominate closeout time. Reserving Days 13 and 14 for evidence and quality was
   necessary.

## Final Metrics and Supported Claim

| Measure | Final audited result |
| --- | --- |
| Planned effort | 140 estimated hours; all 14 days and six items accounted for, actual hours not tracked |
| Default selector | Unset and exact `on` use per-cell reconstruction; exact `off` uses dense compatibility; every other set value, including empty, returns `SPARSE_ERR_BADARG` before SVD work |
| Public ABI | No public declaration, structure, enum, typedef, or symbol changed; public-header edits document the existing function |
| Differential corpus | 42 independent analytical or long-double-factor-oracle route cells across deterministic core, adverse numerical, and seeded fixtures |
| Focused SVD suite | 136 tests, 19,369 assertions, zero failures or skips at both index widths |
| Partial-SVD corpus | 10 tests, 247 assertions at both index widths |
| Allocation lifecycle | First-success counts 26 default square, 37 default wide, 23 zero shortcut, and 27 compatibility; all preceding failures cleanly recover |
| Measured raw packet | 16 warmups plus 80 measured invocations; 290 manifest entries all verify |
| Reconstruction scratch | Default: 0 bytes/0 allocations in all 40 measured calls; nonzero compatibility: 2,097,152 bytes/1 allocation in all 30 measured calls; zero shortcut: 0 on both routes |
| Timing decision | `timing_pass=false`; square median/max 1.3923/1.5675, wide 1.3771/1.5764, tall 1.2297/1.4312; no speed claim |
| Protocol decision | `memory_pass=true`, `protocol_accepted=false`; development override was explicit |
| Full quality | `make format`, `make lint`, all 62 Make test binaries, clean CMake registration parity, and 63/63 CTests pass |
| Sanitizers | Apple-Clang ASan/UBSan passes at 32 and 64 bits; Homebrew-Clang ASan/UBSan with leak detection passes the affected corpus |
| Documentation and inventory | All 18 public headers covered, 49 library sources registered, Sprint Markdown links and patch whitespace pass |

The supported sentence is: **“For the recorded Sprint 228 implementation and measured fixtures,
the maintained default sparse low-rank reconstruction path allocates no additional `m*n`
reconstruction accumulator while preserving the frozen numerical, structural, ownership, and
failure contract.”**

It does not establish sparse-memory SVD end to end, lower whole-call peak RSS, universally sparse
output, faster execution, portable performance, external-library ranking, release readiness,
superiority, or state-of-the-art status.

## Daily Evidence

- [Day 1 reconstruction path and allocation inventory](artifacts/day1-reconstruction-path-and-allocation-inventory.md)
- [Day 2 reconstruction contract and measurement protocol](artifacts/day2-reconstruction-contract-and-measurement-protocol.md)
  and [fixture matrix](artifacts/day2-reconstruction-fixture-matrix.tsv)
- [Day 3 core differential harness](artifacts/day3-core-differential-harness.md)
- [Day 4 degenerate and numerical differential corpus](artifacts/day4-degenerate-and-numerical-differential-corpus.md)
- [Day 5 differential acceptance and promotion design](artifacts/day5-differential-acceptance-and-promotion-design.md)
- [Day 6 outer-route hardening](artifacts/day6-outer-route-hardening.md)
- [Day 7 default promotion and selector contract](artifacts/day7-default-promotion-and-selector.md)
- [Day 8 default-route and initial allocation matrix](artifacts/day8-default-route-and-initial-allocation-matrix.md)
- [Day 9 allocation and numeric-failure evidence](artifacts/day9-allocation-and-numeric-failure-evidence.md)
- [Day 10 lifecycle and contract closure matrix](artifacts/day10-lifecycle-and-contract-closure-matrix.md)
- [Day 11 reconstruction-memory packet](artifacts/day11-reconstruction-memory-packet.md)
  and [component accounting](artifacts/day11-memory-component-accounting.tsv)
- [Day 12 documentation and timing limitations](artifacts/day12-documentation-and-timing-limitations.md)
- [Day 13 focused cross-build and sanitizer validation](artifacts/day13-focused-cross-build-and-sanitizer-validation.md)
- [Day 14 closeout evidence index](artifacts/day14-closeout-evidence-index.md),
  [final quality decision and handoff](artifacts/day14-final-quality-and-handoff.md), and
  [retrospective input](artifacts/day14-retrospective-input.md)

## Residual Handoff

Sprint 229 may reuse the validation-precedence discipline, independent oracle patterns, complete
allocation-sweep method, mixed-call recovery checks, both-width matrix, and narrow claim ledger.
It must independently freeze the CG-family status, iteration, residual, preconditioner, and
reusable-handle contracts; SVD-specific selectors and memory evidence do not transfer.

The exact-`off` compatibility route remains intentionally available and should not be removed
without a separate migration decision. Future low-rank timing work must use a clean exact
candidate, preserve the frozen fixtures and thresholds, retain every pair, and receive independent
methodology and artifact review. Until then, timing remains blocked and no speed wording is
permitted. A stronger whole-call memory claim would require evidence that accounts for factors,
decomposition work, output density, allocator behavior, and process peak memory rather than only
the removed reconstruction accumulator.

After this retrospective is committed and pushed, hosted Linux, macOS, and Windows CI must pass
for the PR head before merge. Local completion and the Day 11 development packet do not substitute
for those platform results or independent review.
