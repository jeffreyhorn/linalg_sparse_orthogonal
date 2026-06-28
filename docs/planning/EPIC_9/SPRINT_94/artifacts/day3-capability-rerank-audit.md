# Sprint 94 Day 3: Capability Re-rank Audit

## Purpose

Reduce Sprint 94's broad capability-surface modernization problem to one
ranked live contradiction map centered on the strongest scalar-contract,
index-width maturity, and solver-family breadth seams.

## Main Result

Sprint 94's broad capability problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - the public scalar contract centered on `include/sparse_types.h` and the
    shared matrix-shell helper seam in `include/sparse_matrix.h`, where the
    repo now advertises one bounded scalar-preparation seam but still ships a
    real-only `double` contract
- strongest second target:
  - the touched 64-bit and ABI-maturity seam, where `SPARSE_IDX_BITS` and
    `idx_t` are real product features but touched consumer interpretation,
    formatting, and width-aware proof reading are still the main remaining
    maturity question rather than raw typedef availability
- strongest third target:
  - solver-family breadth concentrated in iterative, eigensolver, QR, SVD,
    and adjacent dense-helper owners, where caller-facing scalar naming has
    widened but implementation semantics still read as bounded real-only
    support
- strongest fourth target:
  - proof and benchmark follow-through so any widened scalar or index claim is
    anchored to maintained executable truth
- strongest support-only but real target:
  - public and maintainer wording that still needs to stay truthful about
    bounded capability breadth and explicit non-claims

## Strongest Current Contradiction

The strongest current contradiction is still the public scalar-contract
ceiling:

- `include/sparse_types.h` explicitly binds `sparse_scalar_t` to `double` and
  states that the current shipped contract remains real-only
- `include/sparse_matrix.h` routes shared matrix-shell helper paths through
  `sparse_scalar_t`, but explicitly says this does not imply broad generic
  numeric or complex support
- `README.md` and `docs/maintainer_guide.md` repeat the same bounded
  interpretation, which means the public capability story is intentionally
  prepared for widening but still narrower than a state-of-the-art
  competitive scalar surface

That fixes the strongest first Sprint 94 move:

- the project does not most urgently need another broad doc or workflow pass
- it needs one clearer scalar-contract widening seam on the highest-value
  public and shared implementation surface
- the current scalar aliasing work is real preparation, but it still reads as
  preparation more than as a landed widened capability claim

## Second-Tier Contradictions

### Index-Width Maturity

The strongest second contradiction is index-width maturity rather than missing
index configurability:

- `SPARSE_IDX_BITS` and `idx_t` are already first-class public contract
  surfaces
- Sprint 91-93 materially improved touched formatting and width-aware
  printing on some reviewed lanes, but the remaining challenge is now
  touched-path maturity and ABI clarity rather than "add 64-bit support"
- this is real Sprint 94 work because the wider-index story is only as strong
  as the touched public and consumer seams that actually respect it

### Solver-Family Breadth Concentration

The strongest third contradiction is solver-family breadth concentration:

- `include/sparse_iterative.h`, `include/sparse_eigs.h`, and
  `include/sparse_qr.h` already route many caller-owned buffers through
  `sparse_scalar_t`
- `include/sparse_dense.h` still exposes a `double`-backed dense matrix type
  and `double` dense kernels, which keeps the deeper implementation story
  materially real-only
- the strongest implementation owners tied to that bounded breadth remain:
  - `src/sparse_dense.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_eigs.c`
  - `src/sparse_iterative.c`
  - `src/sparse_matrix.c`

This is real Sprint 94 work, but it reads after the first scalar-contract
seam rather than before it.

### Proof and Evidence Follow-Through

The strongest fourth contradiction is proof and evidence follow-through:

- the repo already has reviewed scalar-sensitive proof owners:
  - `tests/test_dense.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_eigs.c`
  - `tests/test_iterative.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
- benchmark and evidence owners exist, but they remain second-tier relative
  to the reviewed executable proof surfaces:
  - `benchmarks/bench_svd.c`
  - `benchmarks/bench_eigs.c`
  - `benchmarks/bench_iterative_reuse.c`

This remains real Sprint 94 work, but it is explicitly later than the first
implementation seam.

## Fix-Now vs Deferred Split

The current tree now separates cleanly into:

### Contradictions that should drive Sprint 94 implementation

- one scalar-contract widening seam
- touched 64-bit and ABI maturity on that seam
- one bounded solver-family breadth follow-through
- proof/docs/package alignment for the widened claim

### Contradictions that remain later or bounded non-claims for now

- fake full-library complex support
- fake broad mixed-precision maturity
- generic templated rewrite across every family
- broad package or platform symmetry claims

### Contradictions already materially bounded entering Sprint 94

- compressed-first product entry and lifecycle clarity
- bounded portable dense-backend widening
- reviewed runtime and ND evidence follow-through

## Strongest Owner Surfaces

The highest-value owner surfaces tied to this audit are now explicit:

- public scalar/index contract owners:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `include/sparse_dense.h`
  - `include/sparse_qr.h`
  - `include/sparse_eigs.h`
  - `include/sparse_iterative.h`
- implementation owners:
  - `src/sparse_dense.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_eigs.c`
  - `src/sparse_iterative.c`
  - `src/sparse_matrix.c`
- proof-owner tests:
  - `tests/test_dense.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_eigs.c`
  - `tests/test_iterative.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`

## Interpretation

The useful Day 3 clarification is now explicit:

- Sprint 94 does not begin with generic feature expansion
- it begins with one ranked capability contradiction map
- the best first implementation center is the public scalar-contract seam
- index-width maturity and solver-family follow-through remain real Sprint 94
  work, but they are explicitly sequenced behind that first center

## Exit State

- Sprint 94 now has one ranked live capability contradiction map grounded in
  the current post-Sprint-93 tree.
- The first Sprint 94 implementation center is fixed to the public
  scalar-contract seam, with index-width maturity second and solver-family
  breadth third.
- Day 4 can freeze the scalar/index capability contract without reopening the
  ranked capability order.
