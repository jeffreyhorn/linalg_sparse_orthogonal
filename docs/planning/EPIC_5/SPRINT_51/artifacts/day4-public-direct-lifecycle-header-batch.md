# Sprint 51 Day 4: Public Direct Lifecycle Header Batch

## Purpose

Day 4 lands the first bounded public direct-solver lifecycle header/API batch
for Sprint 51. The goal is to make the shared repeated-run direct contract
explicit in `sparse_analysis.h`, add relationship wording in the family-local
LU / Cholesky / LDL^T headers, preserve one-shot compatibility wording, and
validate the touched `*.h` surface through the full code-day gate plus the
stronger reviewed baseline.

## Main Day 4 Conclusion

The repo now has a live phase-1 public direct-lifecycle header contract:

- `sparse_analysis.h` is now explicitly the shared repeated-run direct path
- `sparse_lu.h` remains one-shot-first, but points stable-pattern repeated
  runs to the shared analysis/factor/refactor API
- `sparse_cholesky.h` does the same while preserving the visible in-place
  mutation truth
- `sparse_ldlt.h` now names its owned-factor role relative to the shared
  repeated-run direct path

The batch stayed inside the Sprint 50/51 scope fence:

- no new public generic direct handle
- no raw internal layout exposure
- no demotion of one-shot direct family APIs
- no promise that reuse preserves old numeric factor state

## Touched Public Headers

### `include/sparse_analysis.h`

Day 4 strengthened the shared repeated-run direct contract by making these
truths explicit in the header itself:

- this is the explicit public repeated-run direct-solver path
- the intended lifecycle is:
  - zero/init
  - analyze once
  - factor / solve
  - refactor / solve many
  - free explicitly
- `sparse_analysis_t` owns symbolic/permutation setup only
- `sparse_factors_t` owns numeric factor state only
- neither object owns the source matrix
- repeated-run reuse preserves symbolic/permutation setup rather than old
  numeric factor contents

### `include/sparse_lu.h`

Day 4 kept LU one-shot-first while adding the bounded repeated-run
relationship:

- file-level cross-reference to `sparse_analysis.h`
- one-shot copied-matrix story remains the simple/default path
- repeated same-pattern LU solves now point callers toward the shared
  analyze/factor/refactor direct path

### `include/sparse_cholesky.h`

Day 4 kept Cholesky one-shot-first and mutation-aware:

- file-level cross-reference to `sparse_analysis.h`
- repeated same-pattern SPD solves now point to the shared repeated-run path
- visible mutation guidance remains intact:
  - copy first if the original is needed later
  - lower triangle overwritten with `L`
  - upper triangle removed

### `include/sparse_ldlt.h`

Day 4 clarified the LDL^T role without redesigning it:

- file-level wording now identifies `sparse_ldlt.h` as the family-local
  owned-factor surface
- relationship wording now points repeated direct runs to the shared
  analysis/factor/refactor contract
- the `sparse_ldlt_t` owned-factor model stays intact and distinct

## Validation

Because `*.h` changed, the full required code-day gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this was a substantial public API batch, the stronger reviewed baseline
was also run:

- `make quality-review-full`

That also passed.

## Truthfulness Anchors Preserved

The maintained reviewed baseline stayed exact:

- `make quality-review-full` remained the strongest local reviewed baseline
- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 440.65 sec`

## Day 4 Operational Result

Sprint 51 now has a validated live public-header phase-1 direct lifecycle
contract. The next sprint slice is no longer header design; it is source-level
LU integration against that contract.

## Highest-Value Day 4 Conclusions

1. The shared repeated-run direct contract is now explicit in the public
   analysis/factor header.
2. The family-local LU / Cholesky / LDL^T headers now point to the shared path
   without duplicating its full prose.
3. One-shot compatibility wording remained intact and visible.
4. The full code-day gate and the stronger reviewed baseline both passed.
