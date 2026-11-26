# Sprint 228 Day 14: Final Quality Closure and Handoff

## Decision

Sprint 228 is complete for the scoped implementation objective. The maintained
default of `sparse_svd_lowrank_sparse()` reconstructs one output cell at a time
and does not allocate an additional `m*n` reconstruction accumulator. Exact
`SPARSE_SVD_LOWRANK_OUTER=off` retains the dense-accumulator compatibility
route; unset and exact `on` select the maintained route; every other set value
fails closed.

The decision is deliberately narrow. Full SVD still retains dense economy
factors and dense decomposition work, and a low-rank reconstruction can itself
be dense after thresholding. The retained Day 11 timing packet fails the
square and wide timing gates and was collected with the development override.
It therefore cannot support a speed, promotion-level timing, whole-call
peak-memory, or release claim.

## Final quality results

The required final chain ran in order on the complete sprint source state:

1. `make format` passed.
2. `make lint` passed strict compilation, clang-tidy, and cppcheck over all 112
   analyzed files.
3. `make test` passed all 62 registered Make test binaries. The long nested
   dissection suite and all SVD, partial-SVD, allocation-failure, and memory
   accounting cases completed successfully.
4. `make QUALITY_REVIEW_CMAKE_DIR=build/s228-day14-cmake quality-review-cmake`
   configured and clean-built every CMake target, proved that 63 CTests equal
   62 Make tests plus one intentional focused selector, and passed 63/63 CTests.

Supplemental closeout checks also passed: `make format-check`,
`make source-list-check` (49 library sources), `make docs-check` (18/18 public
headers), collector byte-compilation, all 290 Day 11 manifest hashes, and
`git diff --check`.

Day 13 supplied the final C/header state's fresh focused cross-build evidence:
32-bit and 64-bit Make/CMake tests, Apple-Clang ASan+UBSan at both widths, and a
Homebrew-Clang leak-enabled ASan+UBSan run. Day 14 changed only closeout
documentation after those runs, so that sanitizer record remains source-exact.

## ABI, selector, and registration audit

- No public function declaration, public structure, enum, typedef, or symbol
  was added, removed, or reordered. `include/sparse_svd.h` changes only the
  existing function's documentation.
- The selector is internal compatibility control, not a public ABI addition.
  Its accepted values and fail-closed behavior are covered by direct and
  repeated mixed-route tests.
- The memory benchmark is registered in both Make and CMake build surfaces.
  It is not a CTest because it is a measurement tool, while all behavioral
  regressions live in registered test binaries.
- The complete CTest inventory reconciles exactly with the maintained Make
  inventory and its one documented selector-only case.

## Accepted evidence

- All 42 analytical or long-double-factor-oracle route cells pass independently
  for the maintained and compatibility routes.
- Allocation sweeps reach first success at 26 default-square, 37 default-wide,
  23 zero-shortcut, and 27 compatibility allocations. Every preceding failure
  leaves NULL output, preserves input, and permits immediate clean retry.
- All 40 measured default invocations report zero reconstruction-scratch bytes
  and allocations. All 30 nonzero compatibility invocations report one
  2,097,152-byte accumulator. The all-zero shortcut allocates none on either
  route.
- Route pairs preserve status, dimensions, values, checksum, `nnz`, factor
  accounting, decomposition accounting, shell/maps, and node-slab accounting.
- The Day 11 raw packet contains 16 warmups and 80 measured invocations; every
  one of its 290 manifest entries reverified on Day 14.

## Residual risks and blocked evidence

1. The per-cell route performs scalar reconstruction and sparse insertion for
   each candidate cell. Its timing is workload-sensitive; square and wide
   ratios exceed the frozen limits.
2. Full SVD remains a dense-factor/decomposition operation. Removing one
   reconstruction accumulator does not make the whole operation sparse-memory
   or establish lower process peak RSS.
3. Output storage scales with retained entries and can approach dense storage.
4. The Day 11 packet is development memory evidence, not clean-candidate
   promotion evidence. A later timing claim requires a clean exact-candidate
   collection that passes the frozen protocol without override.
5. The branch is not yet committed. Review must bind this diff to the eventual
   commit SHA; the current base is
   `363600ea27e854a4334cff5fa06ace6529045ccd`.

## Handoff

- Preserve unset/`on`/`off` selector semantics and validation precedence.
- Keep both routes until a separately reviewed compatibility-removal decision.
- Run the complete quality chain after any review fix touching C or headers.
- Treat `day11-reconstruction-memory-packet.md` and its manifest as the memory
  evidence owner, and `day12-documentation-and-timing-limitations.md` as the
  claim-boundary owner.
- Do not describe the default as faster, universally lower-memory, or producing
  a sparse result in every case.
