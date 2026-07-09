# Epic 11 Gap-Closure Todo - Codex - 2026-07-09

This todo turns the 2026-07-09 review into an execution sequence for Epic 11.
It intentionally preserves Epic 10's truthfulness model: broaden claims only
after implementation, proof, validation, and public wording cleanup exist.

## Guiding Rules

1. Keep compressed-first sparse workflows as the product center.
2. Treat the orthogonal linked-list shell as supported compatibility, not the
   primary performance story.
3. Reduce giant source/test owners only along real behavior boundaries.
4. Add numerical oracles before expanding public correctness claims.
5. Treat benchmark output as local measurement unless a portable-performance
   contract is explicitly designed and validated.
6. Treat package, ABI, and platform support as product contracts.
7. Preserve explicit non-claims for GPU, distributed memory, broad ecosystem
   parity, package-manager support, and dynamic ABI until real proof exists.

## Step-by-Step Plan

### 1. Freeze the Post-Epic-10 Baseline

- Capture current file-size, source-list, CTest, package, benchmark, coverage,
  CI, and documentation metrics.
- Re-run or cite the strongest reviewed baseline and identify which lanes are
  reviewed, supplemental, staged, or local-only.
- Convert the Epic 10 residual queue into Epic 11 owners and proof gates.
- Create a compact current-product truth map: compressed-first support,
  mutable-shell compatibility, solver-family evidence, package/platform tiers,
  and explicit non-claims.

### 2. Reduce Remaining Proof-Owner and Source-Boundary Debt

- Start with the residual items from Epic 10:
  - eigensolver private owner movement;
  - `s20_select_indices`;
  - `s20_lift_ritz_vectors`;
  - shift-invert setup/conversion;
  - `lanczos_iterate_op`.
- For each movement, write exact old/new file plans, source-list/CMake impact,
  focused consumer tests, CTest count expectations, and rollback instructions.
- Move only one ownership boundary at a time.
- Keep broad source-split claims out of public docs until validated.

### 3. Split Giant Tests into Focused Proof Owners

- Prioritize `tests/test_ldlt_csc.c`, `tests/test_integration.c`,
  `tests/test_qr.c`, `tests/test_ldlt.c`, `tests/test_iterative.c`,
  `tests/test_svd.c`, `tests/test_graph.c`, and `tests/test_reorder_nd.c`.
- Extract reusable fixtures and helpers before adding new scenarios.
- Preserve CTest membership, Make/CMake parity, and failure localization.
- Track before/after line counts and test responsibility maps.

### 4. Build a Numerical Oracle and Corpus Architecture

- Define a matrix taxonomy for:
  - symmetry and definiteness;
  - rank and singularity;
  - conditioning and scaling;
  - sparsity pattern and fill behavior;
  - expected convergence/failure modes.
- Add deterministic dense-reference, external-process, or cross-solver oracle
  lanes where sustainable.
- Expand direct/iterative/eigensolver/SVD proof ownership without claiming
  broad ecosystem parity.
- Publish oracle interpretation docs for maintainers and concise public trust
  boundaries for users.

### 5. Strengthen Performance and Backend Governance

- Inventory hot compressed kernels, linked-list fallback paths, dense backend
  consumers, OpenMP behavior, and local sentinel coverage.
- Add local regression sentinels for high-value paths where deterministic
  enough to be useful.
- Add generated report indexes and manifests that are easier to compare across
  branches.
- Keep portable speed, universal reorder/fill superiority, and vendor backend
  parity as non-claims unless a separate proof campaign is funded.

### 6. Decide Package, ABI, and Platform Next Steps

- Decide whether Epic 11 will:
  - add shared-library support and a dynamic ABI policy; or
  - explicitly preserve static-first-only support for another epic.
- If adding ABI support, implement build rules, symbol/version policy,
  install/export proof, ABI tests, and platform loader checks.
- If deferring, make the deferral explicit in README, INSTALL, and maintainer
  docs.
- Evaluate Linux install CI, macOS CMake install/export parity, Windows
  install-validation, and Windows staged test portability as separate lanes.
- Do not infer support from local scripts without reviewed CI proof.

### 7. Simplify API, Documentation, and Examples

- Split or restructure `docs/algorithm.md` into:
  - concise current algorithm reference;
  - historical measurement appendix.
- Make compressed-first examples the first advanced workflow path.
- Add cookbook-style workflows for:
  - direct solve;
  - stable-pattern refactorization;
  - iterative solve with diagnostics;
  - eigensolver;
  - SVD;
  - Matrix Market load/save;
  - install and downstream consumption;
  - benchmark interpretation.
- Move maintainer-only proof detail out of public adoption text.

### 8. Improve Reportability and Coverage Architecture

- Convert benchmark, coverage, dead-code, source-list, and large-matrix
  guardrail outputs into a clearer recurring artifact index.
- Decide which report types are reviewed, supplemental, or local-only.
- Add explicit stale-report detection where practical.
- Use coverage as a targeted assurance tool for high-risk owners, not as a
  vanity percentage.

### 9. Recalibrate Claims and Non-Claims

- Before final Epic 11 closeout, rescan README, INSTALL, docs, examples,
  benchmarks, and maintainer guide.
- Remove, downgrade, or fence any claim not backed by implemented proof.
- Keep broad state-of-the-art replacement, portable performance superiority,
  ecosystem parity, dynamic ABI, package-manager, GPU, and distributed-memory
  support as explicit non-claims unless Epic 11 truly implements them.

### 10. Close Epic 11 with Evidence

- Run the strongest reviewed quality baseline.
- Run any package/platform/source-list/CMake/coverage/benchmark supplemental
  lanes required by touched surfaces.
- Publish earned claims, unearned claims, residuals, and post-Epic handoff.
- Ensure every sprint's residual deferred debt is either closed, carried
  forward, or explicitly rejected.

## Completion Definition

Epic 11 is complete only when:

- all Sprint 118-127 deliverables have evidence artifacts or committed docs;
- source/test ownership debt has measurable before/after improvement;
- numerical oracle/corpus evidence is broader and still honestly scoped;
- package/platform/ABI decisions are explicit and validated where claimed;
- adoption docs are simpler and less historical;
- reviewed quality surfaces pass;
- final closeout publishes earned claims, non-claims, and residuals.

