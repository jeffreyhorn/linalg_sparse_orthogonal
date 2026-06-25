# Sprint 90 Day 3: Product, Performance, and Capability Gap Audit

## Purpose

Reduce Sprint 90's broad post-Epic-8 structural problem to one ranked live
contradiction map across the public product model, the dense/backend
performance ceiling, and the bounded capability surface so Epic 9 can choose
real implementation centers instead of another generic modernization bucket.

## Main Result

Sprint 90's broad product/performance/capability problem is now reduced to
one ranked live contradiction map:

- strongest first target:
  - compressed-first product-model convergence centered on the public
    matrix-shell and direct-workflow ownership story
- strongest second target:
  - portable dense/backend maturity beyond the current builtin scalar core
    plus bounded optional Accelerate slices
- strongest third target:
  - capability-surface widening beyond the current real-only scalar and
    intentionally bounded solver/eigensolver breadth
- strongest fourth target:
  - runtime/threading and ABI/index maturity only where they materially
    sharpen the first three structural lanes
- strongest support-only but real target:
  - public/support wording that still truthfully reflects the narrower
    current product, backend, and capability reading

## Strongest Current Contradiction

The strongest current contradiction is still the public/core product model:

- `README.md` still introduces the library as an orthogonal linked-list sparse
  matrix library
- `include/sparse_matrix.h` still describes the matrix shell as the library's
  mutable sparse construction and one-shot direct-workflow compatibility shell
- the same header now explicitly says bounded compressed-first construction,
  import, and publication helpers may exist internally, but public ownership
  still stays with the compatibility shell
- `src/sparse_matrix.c` remains a major shell, mutation, and utility owner

That fixes the strongest first Epic 9 move:

- the project no longer most urgently needs another proof, package, or docs
  tightening pass
- it needs a clearer compressed-first product reading on the highest-value
  public direct and interop workflows
- the linked-list shell remains a real strength for pedagogy and mutation, but
  it still reads as the public conceptual center rather than as a bounded
  compatibility surface

## Second-Tier Contradictions

### Dense and Backend Maturity Ceiling

The strongest second contradiction is backend maturity:

- `src/sparse_dense.c` still owns the builtin dense GEMM/GEMV/factor/solve
  helpers in scalar C
- `src/sparse_ldlt_csc.c` still exposes only a bounded optional Accelerate
  lane on Darwin rather than a portable backend story
- `README.md` still presents acceleration as optional and bounded, not as a
  broad portable product lane

This is real Epic 9 work because a more competitive sparse numerical library
needs more than careful scalar fallback plus one narrow platform-specific
acceleration seam. The repo now has backend architecture, but it still does
not have backend maturity.

### Capability Breadth Ceiling

The strongest third contradiction is capability breadth:

- `include/sparse_types.h` still binds `sparse_scalar_t` to real-only
  `double`
- the same header explicitly says the widened scalar seam does not imply broad
  numeric genericity, complex support, or a wider precision product today
- `README.md` still keeps the iterative reusable-handle story and eigensolver
  breadth intentionally bounded
- the 64-bit index lane is now real, but still compile-time-selected rather
  than reading like a deeply matured always-supported product lane

This is real Epic 9 work, but it still reads after the product-model and
backend ceilings rather than before them. The repo is more disciplined than it
was pre-Epic-8, but it is still materially narrower than a state-of-the-art
capability claim would require.

### Runtime, Threading, and ABI Follow-Through

The strongest fourth contradiction is follow-through rather than first-center
work:

- `README.md` still presents OpenMP as optional and localized rather than as a
  product-wide runtime model
- reviewed runtime concentration still remains visible on the reorder/ND lane
- the widened scalar/index story is real, but the strongest remaining index
  and ABI work is still secondary to the larger product-model and backend
  questions

This remains real Epic 9 work, but it is now clearly behind the first three
structural lanes rather than ahead of them.

## Fix-Now vs Bounded Non-Claim Split

The current tree now separates cleanly into:

### Contradictions that should drive Epic 9 implementation

- linked-list-first public/product ownership
- bounded dense/backend maturity
- bounded capability breadth

### Contradictions that remain bounded non-claims for now

- broad complex-scalar support
- broad mixed-precision support
- symmetric cross-platform package and workflow parity
- broad "best-in-class runtime" or "best ordering choice on every matrix"
  claims

### Contradictions already materially improved by Epic 8

- sharper package/install/export truth
- materially cleaner front-door adoption and support split
- bounded external SPD comparison as a real maintained lane
- reviewed runtime concentration reduced from its earlier Sprint 85 shape
- public scalar/index ownership less inconsistent across the touched shared
  seams

## Strongest Owner Surfaces

The highest-value owner surfaces tied to this audit are now explicit:

- product-model owners:
  - `README.md`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- backend/performance owners:
  - `src/sparse_dense.c`
  - `src/sparse_ldlt_csc.c`
  - `README.md`
- capability-surface owners:
  - `include/sparse_types.h`
  - `README.md`
  - touched iterative/eigs/direct public headers where later widening may land

## Deferred Structural Claims

Broad claim widening remains lower-value first work:

- no fake "already compressed-first everywhere" story
- no fake backend-neutral acceleration maturity claim
- no fake broad complex or mixed-precision claim
- no fake platform symmetry claim
- no reopening package or usability lanes as if they were still the first-tier
  structural problem

## Interpretation

The useful Day 3 clarification is now explicit:

- Epic 9 does not begin with another generic modernization lane
- it begins with one ranked structural contradiction map
- the best first implementation center is the public/product reading around
  the linked-list shell and compressed-first workflows
- portable backend maturity follows next
- capability widening follows after that
- runtime/threading and wider ABI/index maturity remain real, but they are
  explicitly later than the first three structural lanes unless target-state
  design proves otherwise

## Exit State

- Sprint 90 now has one ranked live product/performance/capability
  contradiction map grounded in the current post-Epic-8 tree.
- The first structural Epic 9 center is fixed to compressed-first
  product-model convergence.
- Portable backend maturity, capability widening, and runtime/ABI follow-through
  are explicitly ordered behind that first center.
