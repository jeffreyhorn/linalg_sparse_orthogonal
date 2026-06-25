# Sprint 92 Day 3: Dense Hotspot Profiling Audit

## Purpose

Reduce Sprint 92's broad portable-backend problem to one ranked live
contradiction map centered on the strongest builtin dense-kernel hotspots, the
highest-value direct-family consumers, and the narrow current optional-backend
story.

## Main Result

Sprint 92's broad backend problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - the shared dense-kernel owner in `src/sparse_dense.c`, where the builtin
    scalar dense primitives still define the broadest performance ceiling
- strongest second target:
  - the direct-family adoption seam concentrated in `src/sparse_chol_csc.c`
    and `src/sparse_ldlt_csc.c`, where backend dispatch is real but still
    narrow and family-local
- strongest third target:
  - QR and adjacent dense consumers that still read as builtin-only and do
    not yet share the strongest bounded backend seam
- strongest fourth target:
  - benchmark and observability follow-through so any widened backend path is
    measurable and fallback-visible
- strongest support-only but real target:
  - build/package/support wording that still truthfully reflects a builtin-
    first default plus bounded optional acceleration

## Strongest Current Contradiction

The strongest current contradiction is still the backend-maturity ceiling in
the shared dense owner:

- `src/sparse_dense.c` still owns the generic dense GEMM/GEMV and dense
  factor/solve primitives in self-contained scalar C
- the only current optional accelerated lane exposed there is the Apple-only
  Accelerate probe for the Cholesky supernodal dense-kernel descriptor
- that lane is environment-selected and bounded by backend-contract limits,
  not a broader portable backend story

That fixes the strongest first Sprint 92 move:

- the project does not most urgently need another public-story or
  package-wording pass
- it needs one clearer portable backend seam on the highest-value shared dense
  kernel surface
- the builtin scalar path remains a real strength for fallback truth and
  self-contained builds, but it still reads as the dominant ceiling rather
  than as a bounded default beneath a portable optional backend lane

## Second-Tier Contradictions

### Direct-Family Adoption Concentration

The strongest second contradiction is backend adoption concentration:

- `src/sparse_chol_csc.c` already consumes
  `chol_csc_supernodal_dense_kernels()` and therefore has the cleanest
  immediate backend-adoption seam
- `src/sparse_ldlt_csc.c` carries its own bounded optional Accelerate seam,
  but it is still family-local rather than converged with the shared dense
  owner
- `include/sparse_ldlt.h` still documents backend selection in family-local
  terms rather than as part of a broader dense-kernel maturity story

This is real Sprint 92 work because a bounded portable backend lane is not
valuable if it remains trapped inside one narrow family-local corner.

### QR and Later Dense Consumers

The strongest third contradiction is later-adopter breadth:

- `src/sparse_qr.c` remains a large dense consumer candidate
- `tests/test_qr.c` remains a major proof-owner surface
- the current tree does not yet show QR sharing the strongest bounded backend
  seam

This is real Sprint 92 work, but it reads after the first shared-kernel and
Cholesky/LDL^T adoption lane rather than before it.

### Observability and Support Follow-Through

The strongest fourth contradiction is observability:

- `benchmarks/bench_chol_csc.c` already acts as a backend comparison surface
- `benchmarks/bench_refactor_csc.c` and `benchmarks/bench_svd.c` remain likely
  measurement follow-through owners
- support surfaces still need to stay truthful about builtin-default fallback
  and bounded optional acceleration

This remains real Sprint 92 work, but it is explicitly later than the first
implementation seam.

## Fix-Now vs Deferred Split

The current tree now separates cleanly into:

### Contradictions that should drive Sprint 92 implementation

- shared dense-kernel backend seam
- strongest direct-family backend adopters
- backend observability and fallback proof

### Contradictions that remain later or bounded non-claims for now

- fake broad cross-platform backend symmetry
- generic runtime/threading widening
- full-family dense convergence everywhere at once
- capability-surface widening beyond backend maturity

### Contradictions already materially bounded entering Sprint 92

- compressed-first product entry and lifecycle clarity
- package/install/export contract sharpness
- front-door and support-surface layering

## Strongest Owner Surfaces

The highest-value owner surfaces tied to this audit are now explicit:

- backend implementation owners:
  - `src/sparse_dense.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
- benchmark and measurement owners:
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
- proof-owner tests:
  - `tests/test_dense.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`

## Interpretation

The useful Day 3 clarification is now explicit:

- Sprint 92 does not begin with another generic direct-family speed pass
- it begins with one ranked backend-hotspot contradiction map
- the best first implementation center is the shared dense-kernel owner and
  the strongest immediate Cholesky/LDL^T adoption seam
- QR follow-through, benchmark/reporting widening, and support-surface wording
  remain real Sprint 92 work, but they are explicitly sequenced behind that
  first center

## Exit State

- Sprint 92 now has one ranked live backend-hotspot contradiction map grounded
  in the current post-Sprint-91 tree.
- The first Sprint 92 implementation center is fixed to the shared
  dense-kernel owner and strongest immediate direct-family adopters.
- Day 4 can freeze the first implementation boundary without reopening the
  ranked backend-hotspot order.
