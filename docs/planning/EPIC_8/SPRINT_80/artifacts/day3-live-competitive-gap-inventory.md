# Sprint 80 Day 3 - Live Competitive Gap Inventory

Date: 2026-06-18  
Branch: sprint-80

## Purpose
Re-rank the live structural, capability, performance, usability, and product
gaps against the current tree so Epic 8 starts from one ranked contradiction
map instead of one generic modernization bucket.

## Main Result
Epic 8's broad improvement problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - linked-list-first product/storage ceiling
- strongest second target:
  - builtin dense/backend performance ceiling
- strongest third target:
  - bounded capability surface
- strongest fourth target:
  - external differential / numerical-assurance ceiling
- strongest fifth target:
  - maintainability concentration in large source and giant-test hotspots
- strongest sixth target:
  - reviewed runtime concentration around reordering/ND proof
- strongest seventh target:
  - static-first and asymmetric package/platform maturity
- strongest support-only but real target:
  - front-door usability and policy density

## Strongest Current Contradiction
The strongest current contradiction center is still the public/core
storage-workflow model:

- `README.md` still opens by identifying the library as an orthogonal
  linked-list sparse matrix library
- `include/sparse_matrix.h` still keeps the mutable linked-list shell as the
  public conceptual center
- `src/sparse_matrix.c` still centers mutation around pointer-heavy row/column
  walks and slab-node allocation

That means the highest-value Epic 8 first move is still product/storage
modernization rather than package, workflow, or support-surface cleanup.

## Secondary Structural Ceilings

### Dense / Backend Ceiling
The strongest second contradiction remains the dense/backend surface:

- `src/sparse_dense.c` is still the clearest hard performance ceiling
- the repo now has backend-aware architecture from Epic 7
- but that architecture still terminates in a builtin scalar dense-kernel
  surface rather than backend maturity

### Capability Ceiling
The strongest third contradiction remains capability breadth:

- `include/sparse_types.h` still fixes the public scalar contract to real-only
- index width still reads as a compile-time-selected product lane
- iterative/eigs breadth remains materially narrower than a best-in-class
  scientific sparse library

### External Numerical Assurance Ceiling
The strongest assurance-side contradiction remains the lack of a maintained
external numerical-oracle lane strong enough to support a serious competitive
claim.

## Later But Still Real Gaps

### Maintainability Concentration
The strongest remaining maintainability hotspots still include:
- `src/sparse_iterative.c` = `1985`
- `src/sparse_chol_csc.c` = `1564`
- `src/sparse_qr.c` = `1563`
- `tests/test_chol_csc.c` = `4724`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_qr.c` = `3197`
- `tests/test_integration.c` = `2689`

These are real Epic 8 targets, but they now read as the fifth-ranked
contradiction rather than the first.

### Reviewed Runtime Concentration
The reviewed runtime long pole remains real:
- `tests/test_reorder_nd.c` = `2287`

But it is lower-order than the product, backend, capability, and oracle
ceilings.

### Package / Platform Asymmetry
Package/platform work remains important, but later:
- `CMakeLists.txt` still keeps the maintained package surface static-only
- Windows and macOS reviewed proof remain intentionally asymmetric

## Interpretation
The useful Day 3 clarification is now explicit:

- Sprint 81 should begin with the product/storage ceiling
- the dense/backend lane should remain the strongest second implementation lane
- capability expansion should remain third
- maintainability, runtime, and platform/package work remain real, but they
  are not the first contradiction center

## Exit State
- Epic 8 now has one ranked live contradiction map grounded in the tree.
- The first implementation center is fixed to the product/storage ceiling
  unless later Sprint 80 contract work uncovers a stronger contradiction.
