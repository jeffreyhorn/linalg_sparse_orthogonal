# Epic 10 Gap-Closure Todo - Codex - 2026-06-30

This todo turns the Epic 10 review into an execution sequence. The organizing
principle is simple: earn product claims with implementation, oracle evidence,
maintainable ownership, and user-facing clarity.

## Guiding Rules

1. Prefer compressed-first product maturity over adding unrelated solver
   surface area.
2. Add external evidence before broadening public claims.
3. Reduce large source and giant-test risk while touching each family.
4. Keep reviewed quality surfaces exact and explicit.
5. Treat packaging and platform support as product contracts, not incidental
   build details.
6. Preserve explicit non-claims when a state-of-the-art feature remains out of
   scope.

## Step-by-Step Plan

### 1. Freeze the Epic 10 Baseline and Claim Contract

- Re-run the strongest local quality baseline and capture current Make/CMake,
  install/export, source-list, coverage, and benchmark status.
- Convert the Epic 9 residual queue into an Epic 10 claim map.
- Define the exact meaning of "state of the art" for this repository:
  self-contained C library, optional backend hooks, compressed-first product
  workflows, and bounded external comparison evidence.
- List explicit non-goals: GPU, distributed memory, universal vendor backend
  parity, broad complex/mixed-precision maturity, and package-manager
  ecosystem ownership unless intentionally added later.
- Create evidence templates for solver comparisons, benchmark notes, and
  platform-tier claims.

### 2. Move the Product Center to Compressed CSR/CSC

- Audit all public matrix construction, import, mutation, publication, and
  solver entry paths.
- Identify where linked-list shell ownership still forces unnecessary
  conversions or unclear lifecycle rules.
- Add or refine compressed-first constructors, validators, examples, and
  solver adapters on the highest-value paths.
- Preserve compatibility for mutable matrix-shell users, but document it as a
  secondary compatibility path.
- Add regression coverage that proves ownership, lifetime, and error handling
  for compressed-first workflows.

### 3. Deepen Direct Solver Robustness

- Expand external dense-reference and known-matrix comparisons for Cholesky,
  LDLT, LU, QR, SVD, and dispatch paths.
- Stratify fixtures by symmetry, definiteness, singularity, rank, scale,
  ordering, and sparsity structure.
- Add explicit failure-mode tests where algorithms should reject unsupported
  inputs.
- Extract dense-reference helpers and solver-family fixture builders so large
  test owners shrink rather than grow.
- Document solver selection and solver-specific trust boundaries.

### 4. Expand Iterative and Eigensolver External Evidence

- Define oracle expectations for CG, MINRES, BiCGSTAB, eigen, thick-restart,
  and LOBPCG paths.
- Add external comparison artifacts where practical and deterministic fallback
  checks where external libraries are unavailable.
- Build matrix families for convergence, stagnation, preconditioning,
  tolerance, restart, and residual behavior.
- Add reporting that distinguishes correctness, convergence profile, and local
  timing.
- Preserve non-claims for broad ARPACK/Spectra/LOBPCG parity until evidence
  supports them.

### 5. Establish a Durable Backend and Runtime Contract

- Audit dense backend consumers, OpenMP paths, thread-local/global overrides,
  and runtime configuration.
- Define a small backend descriptor contract for builtin and optional
  accelerated dense kernels.
- Add backend observability that users and tests can inspect.
- Keep builtin fallback authoritative.
- Add bounded performance sentinels that catch obvious regressions without
  claiming portable benchmark superiority.

### 6. Improve Reordering, Graph, and Large-Matrix Scalability Evidence

- Consolidate reorder/fill measurement artifacts.
- Expand ordering comparisons on named matrices and generated graph families.
- Clarify nested-dissection, AMD/COLAMD, quotient graph, and fill metrics.
- Add large-matrix memory and runtime guardrails where deterministic enough
  for reviewed or supplemental checks.
- Remove remaining history-heavy comments from graph/reorder implementation
  surfaces as touched.

### 7. Continue Large-Source and Giant-Test Extraction

- Split `src/sparse_ldlt_csc.c`, `src/sparse_lu_csr.c`, `src/sparse_qr.c`,
  `src/sparse_eigs.c`, and `src/sparse_iterative.c` along real ownership
  boundaries.
- Split `tests/test_ldlt_csc.c`, `tests/test_integration.c`,
  `tests/test_qr.c`, `tests/test_ldlt.c`, `tests/test_etree.c`, and
  `tests/test_graph.c` into fixtures, helpers, and scenario owners.
- Keep source-list and CMake parity checks updated with every split.
- Prefer extraction that improves failure localization over cosmetic line-count
  churn.

### 8. Raise API, Documentation, and Example Usability

- Create a concise solver-selection and matrix-format guide.
- Make compressed-first examples the first examples users see.
- Add "minimal robust workflow" examples for direct solve, iterative solve,
  eigen solve, decomposition reuse, reorder/fill analysis, install/export, and
  benchmark interpretation.
- Tighten public header comments around ownership, errors, zero/default
  options, and thread/runtime behavior.
- Separate maintainer proof language from user adoption language.

### 9. Mature Packaging, ABI, and Platform Support

- Decide whether Epic 10 will add shared-library and ABI-versioning proof or
  preserve static-first as an explicit support tier.
- Strengthen CMake package, pkg-config, install/export, and downstream
  consumer tests.
- Make Linux, macOS, and Windows support tiers explicit.
- Reassess Windows exclusions and decide whether any can move into reviewed
  parity.
- Document exact-version package behavior and migration expectations.

### 10. Close with Competitive Calibration

- Re-run all reviewed quality, package, source-list, and CMake parity surfaces.
- Regenerate final external comparison, benchmark, coverage, and install
  artifacts.
- Compare final state against the Epic 10 state-of-the-art claim map.
- Remove or downgrade unsupported claims.
- Publish an Epic 10 retrospective with earned claims, non-claims, and
  carry-forward work for the next epic.

## Completion Definition

Epic 10 is complete only when:

- all sprint deliverables have evidence artifacts or committed docs;
- no sprint adds broad state-of-the-art claims without comparison proof;
- reviewed quality surfaces pass;
- documentation names the final platform/package/support tiers;
- large-source and giant-test ownership is measurably improved;
- final closeout identifies remaining gaps explicitly.
