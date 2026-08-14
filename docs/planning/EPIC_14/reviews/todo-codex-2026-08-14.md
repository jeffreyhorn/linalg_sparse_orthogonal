# Epic 14 Gap Closure Todo - Codex - 2026-08-14

## Goal

Close the highest-value remaining gaps from Epic 13 with complete, reviewable
outcomes. Prefer narrow proof, explicit rejection, or source-controlled
documentation over broad partial progress.

## Step-By-Step Closure Plan

### 1. Freeze The Epic 14 Baseline

1. Capture source, header, test, script, benchmark, example, corpus, and docs
   inventory.
2. Record current CI support tiers and Windows CTest count.
3. Record package surfaces: Make install, CMake install/export, `pkg-config`,
   static-first deferral guard, and Windows CMake downstream validation.
4. Record generated surfaces: Doxygen HTML, corpus/oracle reports, comparison
   reports, coverage, dead-code, benchmark, sentinel, and large-matrix rows.
5. Publish an Epic 14 claim target register with accepted targets and explicit
   non-goals.

### 2. Close Generated API Reference Publication

1. Run `make docs` on a clean branch.
2. Capture Doxygen warnings.
3. Verify generated page coverage for every public header in `include/`.
4. Decide whether `docs/api/html/` should be checked in.
5. If committed, add review guidance that generated HTML changed with the
   source/header edits that explain it.
6. If not committed, add a guard and docs that make local-only generated HTML
   an explicit product decision.
7. Update `docs/api_reference.md`, `docs/maintainer_guide.md`, and public
   README links.

### 3. Promote Selected Generated Evidence To Hosted CI

1. Select only the claim-bearing generated families that are already mature:
   QR corpus/oracle, partial-SVD corpus/oracle, and selected comparison rows.
2. Define runtime budgets and artifact retention.
3. Add CI targets that run selected freshness gates.
4. Upload generated reports or emit deterministic summaries.
5. Teach `normalize_report_index.py` tests any stricter hosted semantics needed
   for claim-bearing rows.
6. Preserve local-only/advisory wording for coverage, dead-code, benchmark,
   large-matrix, and optional-data rows unless selected separately.

### 4. Expand One QR Comparison Family

1. Select one QR family from existing maintained corpus fixtures.
2. Define comparison metrics that do not depend on raw basis identity.
3. Extend `scripts/run_external_comparison.py` and its tests.
4. Add source-controlled expected/contract rows.
5. Add focused proof-owner C tests only if implementation behavior is touched.
6. Normalize comparison report rows and add freshness validation.
7. Update maintainer and public claim wording to remain fixture-local.

### 5. Publish One Partial-SVD Comparison Family

1. Select one partial-SVD fixture family with subspace-safe semantics.
2. Define value, projector, residual, orthogonality, convergence, and
   fail-closed comparison fields.
3. Extend the comparison harness and normalizer inputs.
4. Add source-controlled contract rows and generated report freshness.
5. Document skipped optional baselines and dependency provenance.
6. Preserve non-claims for broad SVD parity, raw vector identity, convergence
   rates, sparse-output optimality, performance, and state-of-art status.

### 6. Decide Windows Package Parity

1. Evaluate Windows `pkg-config` execution parity separately from Windows
   Makefile parity.
2. Select one narrow product decision:
   - promote Windows `pkg-config` proof with a chosen provider; or
   - explicitly retain the non-claim with stronger package metadata checks.
3. Keep Makefile parity separate unless it has a clear product owner.
4. Update Windows CI comments, `INSTALL.md`, README, and maintainer guide.
5. Ensure install validation does not imply package-manager or shared-library
   support.

### 7. Strengthen Performance Publication Without Overclaiming

1. Select a canonical benchmark/report subset with stable runtime budget.
2. Define which rows are hard gates, threshold-free reports, local-only
   comparisons, or advisory artifacts.
3. Add methodology fields where missing: platform, compiler, build mode,
   thread count, backend selection, fixture, repeats, and caveats.
4. Publish a methodology-bound report artifact.
5. Reject portable superiority wording unless recurring evidence exists.

### 8. Continue Public Header And API Coherence Cleanup

1. Select the next high-impact public header batch.
2. Capture declarations before cleanup.
3. Update comments, ownership, error contracts, output-buffer semantics,
   lifetime rules, and non-claims.
4. Re-capture declarations and require zero signature drift.
5. Refresh API reference links and generated docs policy.
6. Run full quality gates if C or header code changes.

### 9. Reconcile Claims And Product Boundaries

1. Scan public docs for unsupported claims about state-of-art status, external
   parity, Windows parity, package-manager support, shared libraries, ABI
   stability, runtime-loader behavior, and portable performance.
2. Map every positive public claim to a recurring local or hosted evidence
   owner.
3. Move unfunded work to the residual queue with owner, blocker,
   prerequisite, and promotion gate.
4. Publish the Epic 14 retrospective and updated project-plan status.

## Quality Gates

- Documentation-only changes: run `git diff --check`.
- Script changes: run targeted Python or shell tests plus `git diff --check`.
- C/header changes: run `make format && make lint && make test`.
- Build-system/package changes: run affected install/export scripts and CMake
  parity checks.
- CI changes: reconcile hosted lane names, support-tier docs, expected counts,
  and generated artifact semantics.

## Explicit Non-Goals Unless Re-Scoped

- package-manager distribution;
- full shared-library product support;
- dynamic ABI compatibility promise;
- broad external ecosystem parity;
- portable performance superiority;
- unqualified state-of-the-art sparse linear algebra claim;
- broad Windows Makefile parity unless selected as a product decision.
