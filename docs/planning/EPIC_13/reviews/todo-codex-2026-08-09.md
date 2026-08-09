# Codex Todo: Epic 13 Gap Closure Plan

**Date:** 2026-08-09
**Input Review:** `review-codex-2026-08-09.md`

## Goal

Close the highest-value remaining gaps from Epic 12 without diluting effort.
Epic 13 should prefer complete, evidence-backed closure of a few major gaps
over partial movement across every residual.

## Closure Principles

1. A gap is closed only when implementation, documentation, validation, CI or
   generated evidence, and claim wording agree.
2. Fixture-local evidence must remain fixture-local unless a broader corpus
   family and comparison method exists.
3. Platform claims require hosted platform proof.
4. Package and ABI claims require downstream consumer proof.
5. Performance and state-of-the-art claims require direct comparative
   methodology.
6. Source-controlled metadata is not observed pass evidence unless paired with
   generated or hosted proof.

## Step-by-Step Plan

### Step 1: Freeze Epic 13 Evidence Targets

- Reconcile Epic 12 residuals R1-R14 into Epic 13 selected and non-selected
  gaps.
- Define the exact claims Epic 13 will attempt to earn.
- Define explicit non-goals for claims that remain out of scope.
- Create evidence templates for Windows parity, corpus expansion, generated
  report freshness, shared-library ABI, and external comparison.
- Establish required checks by touched surface.

### Step 2: Close Windows Staged Test Portability

- Port or replace pthread-backed `test_threads` and
  `test_sprint4_integration` with Windows-compatible test paths.
- Port or replace POSIX temp-file behavior in `test_fuzz`.
- Add source-level portability helpers only where they reduce duplication.
- Update CMake gating so promoted tests register intentionally on Windows.
- Replace manual expected-count handling with a maintained checked list or
  generated expectation if feasible.
- Update CI comments, maintainer guidance, and support-tier docs.
- Run Linux, macOS, and Windows-relevant validation.

### Step 3: Decide Reviewed Windows Install-Validation Parity

- Audit Windows CMake install/downstream proof and compare it with Linux/macOS
  package proof.
- Decide whether Windows reviewed install-validation parity is ready to
  promote.
- If promoted, update workflow naming, report rows, INSTALL, README, and
  maintainer guide.
- If rejected, document why and what evidence is missing.
- Keep Makefile and `pkg-config` parity separate unless intentionally closed.

### Step 4: Expand Maintained QR Corpus Families

- Select two to three QR fixture families that can be closed completely:
  rank-deficient rectangular solve, minimum-norm underdetermined solve, and
  reorder/COLAMD-influenced QR are the strongest candidates.
- Add source-controlled fixture metadata and expected rows.
- Add or extend generated fixtures where deterministic generation is safer
  than checked-in matrices.
- Add focused proof-owner tests instead of expanding the largest monolithic QR
  test file.
- Add oracle/report rows and docs that keep claims bounded.
- Run corpus schema, focused QR tests, local oracle generation, and required
  C quality gates.

### Step 5: Expand Maintained Partial-SVD Corpus Families

- Select two to three partial-SVD fixture families that can be closed
  completely: repeated/clustered spectra, rank-deficient rectangular matrices,
  sparse low-rank output, and convergence/fail-closed behavior.
- Define subspace-safe comparisons and tolerance semantics before writing
  tests.
- Add expected rows, generated fixtures, and focused proof-owner tests.
- Avoid raw singular-vector identity claims.
- Update SVD docs, solver-selection docs, corpus docs, and report rows.

### Step 6: Publish Generated Report Freshness For Selected Families

- Choose generated families that support actual Epic 13 claims.
- Regenerate only those families with stable commands.
- Add freshness requirements using `normalize_report_index.py
  --require-generated <family>` where justified.
- Keep generated local rows out of source control unless the project
  intentionally promotes a source-controlled snapshot policy.
- Document branch/commit/platform/compiler boundaries.

### Step 7: Make Shared-Library ABI Product Decision

- Audit public symbols, public headers, static/global state, CMake export
  metadata, pkg-config metadata, Windows import/export requirements, and
  loader behavior.
- Decide whether Epic 13 implements shared-library support or publishes a
  stronger deferral.
- If implementing, add symbol visibility, shared build/install/export,
  downstream loader tests, version metadata, and platform CI proof.
- If deferring, document exact blockers and keep static-first tests strong.

### Step 8: Build External Comparison Harness

- Select a narrow comparison target, likely QR or partial-SVD, based on the
  expanded corpus work.
- Pin external tools/libraries and versions.
- Define metrics, tolerances, skip/defer semantics, and output schema.
- Generate comparison rows locally and optionally in hosted CI if dependencies
  are stable.
- Document why the results are narrow and what they do not prove.

### Step 9: Simplify Adoption And API Reference Surfaces

- Align the tutorial with the Sprint 145 first-use ladder.
- Clean additional public headers without changing signatures or ABI.
- Add a maintained API index or generated Doxygen publication plan.
- Reduce duplicated claim-boundary prose by linking to a single support-tier
  owner where practical.
- Validate docs and declaration preservation.

### Step 10: Final Claim Recalibration And Epic Closeout

- Run final local validation for all touched surfaces.
- Reconcile hosted Linux, macOS, and Windows CI.
- Audit public and maintainer wording for unsupported claims.
- Publish closed claims, non-claims, residuals, and next-epic handoff.
- Decide whether any narrow state-of-practice claim is earned.
- Keep broad state-of-the-art status rejected unless comparative evidence
  supports it.

## Acceptance Criteria

Epic 13 should be considered successful if it fully closes at least three of
these high-value gaps:

- Windows staged test portability and reviewed support-tier update;
- reviewed Windows install-validation parity decision;
- broader QR maintained corpus family;
- broader partial-SVD maintained corpus family;
- selected generated report freshness gate;
- shared-library ABI implementation or stronger product-level deferral;
- first direct external comparison harness;
- tutorial plus broader header cleanup.

The epic should not be considered successful if it merely adds more planning
docs without landing implementation, tests, package/CI changes, or generated
evidence for the selected gaps.
