# Sprint 60 Day 4: Productization Gap Inventory, Part II

Date: 2026-06-08
Branch: `sprint-60`


## Purpose

Extend the Day 3 user-facing productization inventory into the strongest
remaining non-usability gaps, then merge the whole Epic 6 queue into one
ranked map suitable for the Day 5 target-definition batch.

## Additional Non-Usability Findings

### 1. Bounded backend/performance architecture is the strongest remaining structural gap

The repo has strong solver work and real benchmark evidence, but it still lacks
the architecture shape expected of a more state-of-the-art shipping library:

- no bounded dense-kernel backend abstraction
- no optional BLAS/LAPACK-style acceleration seam
- limited shared threading-policy layer
- static-library-first build story
- performance evidence is still more local/manual than governed

This is now the strongest non-usability Epic 6 gap.

### 2. Packaging/platform maturity remains important but secondary

The install/export and CI story is honest and credible, but still asymmetric:

- Linux is the enforced reviewed source-of-truth path
- macOS and Windows retain staged or reduced reviewed surfaces
- the distribution story is still static-first with a bounded ABI/release shape

This is important Epic 6 work, but it should follow the baseline/target and
architecture-contract decisions rather than define them.

### 3. Assurance depth is the strongest confidence gap after product-surface coherence

The repo already has:

- broad tests
- fuzz coverage
- random/stress checks
- selected oracle/reference comparisons

But the hardest workflows still lack a cleaner, more uniformly tiered
second-layer assurance story built around:

- differential checks
- property checks
- stronger oracle comparisons
- harder lifecycle/CSC/repeated-run stress

### 4. Remaining maintainability debt is real but now clearly bounded

The remaining large-file and giant-test seams still matter, especially in:

- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_qr.c`
- `tests/test_ldlt_csc.c`
- `tests/test_qr.c`
- `tests/test_graph.c`

But after Epic 5 they are no longer the whole story. They are bounded debt to
pay down where it helps future product and architecture work.

## Unified Ranked Epic 6 Gap Map

### Must-fix product gaps

1. **Direct-solver usability convergence**
   - reduce the split between compatibility-heavy one-shot direct solves and
     the explicit repeated-run direct lifecycle
2. **Typed configuration modernization**
   - replace the highest-value env-var-driven control surfaces with typed
     options and explicit precedence
3. **Bounded backend/performance architecture modernization**
   - add a real architecture seam for dense kernels, threading policy, and
     future acceleration

### Important quality/performance/platform gaps

4. **Benchmark/performance-governance consolidation**
   - define canonical benchmark tiers and stable product-performance evidence
5. **Packaging/platform/release-shape convergence**
   - improve distribution maturity and narrow the strongest platform asymmetries
6. **Second-layer assurance strengthening**
   - deepen oracle/property/differential coverage on the hardest workflows

### Bounded maintainability debt

7. **Residual large-source decomposition**
   - continue only where ownership seams are real
8. **Residual giant-test refactor**
   - continue only where proof readability improves materially
9. **Residual docs-density cleanup**
   - reduce remaining archaeology/reference overload after the higher-value
     product-surface work lands

### Explicit non-goals / stretch territory

10. **Distributed or cluster/HPC scope**
11. **Immediate vendor-backend parity**
12. **Broad universal shared-library/ABI guarantees without staged platform work**
13. **Major algorithm-family expansion as the defining Epic 6 theme**

## Relevant Carry-Forward Seams from Epic 5

Epic 5 deferred seams that still look relevant in Epic 6:

- later iterative decomposition:
  - `GMRES`
  - shared block-wrapper scaffolding
- later CSC residual cleanup if justified
- deferred giant-test seams:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - intentionally retained dense `tests/test_integration.c`
- broader docs-density cleanup
- staged macOS/Windows/platform residuals that remain honest but incomplete

## Preliminary Must-Fix vs Defer Split

**Must-fix in Epic 6**

- direct usability convergence
- typed configuration
- bounded backend/performance architecture
- performance-governance clarification
- at least some packaging/platform maturity improvement
- stronger assurance on the hardest workflows

**Defer or bound tightly**

- broad platform parity claims without fresh evidence
- open-ended backend ambition
- decomposition or giant-test cleanup without a clear seam
- docs-density cleanup that does not materially affect product adoption or
  truthfulness

## Day 4 Exit State

Sprint 60 now has one coherent ranked Epic 6 gap map. That is enough to define
what “state of the art” should mean for this project on Day 5 without drifting
into unrealistic target inflation.
