# Sprint 83 Day 3: Capability Re-rank Audit

## Purpose

Reduce Sprint 83's broad capability problem to one ranked live contradiction
map so the sprint can choose one bounded scalar/index widening lane instead of
another generic “more numeric types” bucket.

## Main Result

Sprint 83's broad capability problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - shared public dense-scalar owner expansion on the highest-value matrix
    shell and one-shot solver seams
- strongest second target:
  - touched-path wider-index and package/ABI maturity on shared public paths
- strongest third target:
  - QR / SVD algorithm-surface widening after the shared scalar/index contract
    is explicit
- strongest fourth target:
  - true complex-scalar support
- strongest fifth target:
  - broad mixed-precision support
- strongest support-only but real target:
  - proof, docs, and package wording that still reflects the narrower current
    capability reading

## Strongest Current Contradiction

The strongest current contradiction is not the absence of any scalar/index
preparation seam:

- `include/sparse_types.h` already exposes:
  - `sparse_scalar_t`
  - `SPARSE_SCALAR_BITS`
  - `SPARSE_IDX_BITS`
- `include/sparse_iterative.h` and `include/sparse_eigs.h` already route
  their public dense-scalar contracts through `sparse_scalar_t`

The contradiction is that the highest-value shared and one-shot public seams
still do not:

- `include/sparse_matrix.h` still exposes shared dense-vector operations like
  matvec and norm helpers in raw `double`
- `include/sparse_qr.h` still exposes owned factor buffers, solve buffers,
  refinement buffers, and diagnostics in raw `double`
- `include/sparse_svd.h` still exposes result storage, extraction helpers,
  rank/pseudoinverse/low-rank interfaces, and condition estimation in raw
  `double`
- `include/sparse_cholesky.h` and `include/sparse_ldlt.h` still keep their
  one-shot solve and owned-factor numeric surfaces in raw `double`

That fixes the strongest first Sprint 83 move:

- widen the already-real scalar/index ownership story across the highest-value
  shared and one-shot public seams
- keep the shipped scalar contract real-only while doing it

## Second-Tier Contradictions

### Wider-Index / ABI Maturity

The strongest second contradiction is touched-path width maturity:

- `SPARSE_IDX_BITS` already makes width a compile-time contract
- the reviewed build still defaults to the 32-bit lane
- touched public structs, count-sensitive buffers, and package-visible width
  readings still need stronger consistency once the shared capability contract
  moves

This is real Sprint 83 work, but it reads as follow-through after the first
shared scalar/index contract is explicit.

### QR / SVD Algorithm Breadth

The strongest third contradiction is family-local algorithm breadth:

- `include/sparse_qr.h` and `include/sparse_svd.h` still publish result and
  helper surfaces in raw `double`
- their owned dense outputs make them the clearest next bounded
  algorithm-family widening lane once the shared contract exists

This means QR / SVD widening is real, but it should not lead Sprint 83 before
the shared scalar/index owner is fixed.

## Deferred Capability Claims

True complex-scalar support and broad mixed precision remain lower-value first
moves:

- both would force broader algorithm, proof, and package claims
- both would outrun the current bounded maintainer reading in
  `docs/maintainer_guide.md`
- both remain legitimate later capability lanes, but not the first credible
  Sprint 83 implementation center

## Interpretation

The useful Day 3 clarification is now explicit:

- the best first Sprint 83 move is not broad complex support
- it is one bounded widening of the already-real `sparse_scalar_t` / `idx_t`
  ownership story across the highest-value shared and one-shot public seams
- wider-index maturity follows next
- QR / SVD capability breadth follows after the shared contract
- proof and support surfaces stay support-only unless implementation truly
  moves the contract

## Exit State

- Sprint 83 now has one ranked live capability contradiction map grounded in
  the current tree.
- The first implementation center is fixed to shared scalar/index ownership on
  the highest-value public seams.
- Later complex and mixed-precision work is explicitly deferred behind more
  credible bounded lanes.
