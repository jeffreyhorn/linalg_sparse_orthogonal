# Epic 19 Gap-Closure Todo

**Date:** 2026-09-20  
**Source review:** `review-codex-2026-09-20.md`  
**Planning rule:** Close fewer gaps completely. Do not promote broad support
claims without exact evidence.

## Closure Track 1: Package Distribution Beyond Local Proof

1. Select exactly one user-facing provider path for promotion. Recommended:
   Homebrew tap/formula source install, not Homebrew/core bottles.
2. Define support tier vocabulary before implementation:
   - local developer proof;
   - public tap source install;
   - Homebrew/core readiness;
   - bottles;
   - Linuxbrew.
3. Decide whether Epic 19 will promote only a public tap or keep provider
   support deferred.
4. If promoting a public tap path, add formula ownership docs and exact proof
   command.
5. Add formula lint/render/install/test/uninstall validation that can run on a
   supported macOS runner or a documented local environment.
6. Add non-claim guards for Homebrew/core, bottles, Linuxbrew, vcpkg, Conan,
   pkgsrc, distro/system packages, and binary packages.
7. Update README, INSTALL, maintainer guide, and packaging docs with only the
   earned support tier.
8. Run package proof, static package guard, install tests, docs checks, and
   full C gate if source/header files change.

**Done when:** one provider path has exact proof and user-facing docs, or the
deferral is deliberately closed with stronger guard coverage and no public
support promotion.

## Closure Track 2: Selected Windows Cholesky Freshness Promotion

1. Fetch latest hosted Windows Cholesky selected comparison artifacts.
2. Verify target identity for `cholesky-spd-tridiag-5`.
3. Verify uploaded artifact membership, paths, row IDs, generated support tier,
   workflow metadata, and selected manifest contract.
4. Promote manifest metadata only if all evidence matches.
5. Add regression tests rejecting partial Windows metadata promotion.
6. Update README, INSTALL, corpus docs, maintainer guide, and selected target
   docs with promoted or re-deferred wording.
7. Run:
   - `python3 tests/test_selected_report_targets_manifest.py`;
   - `python3 tests/test_selected_comparison_workflow.py`;
   - `python3 tests/test_normalize_report_index.py`;
   - `python3 tests/test_run_external_comparison.py`;
   - `make windows-powershell-guard`;
   - selected freshness command.

**Done when:** the selected Windows Cholesky row is either fully promoted with
matching manifest/docs/guards or explicitly re-deferred with stronger proof of
absence.

## Closure Track 3: Windows QR Incompatible Promotion

1. Keep Sprint 203 local QR incompatible proof as the starting evidence.
2. Add hosted MSVC/CMake proof for `qr-incompatible-ls`.
3. Inspect generated artifacts and normalize Windows paths.
4. Add or update artifact membership tests and selected row filtering tests.
5. Promote selected metadata only if the hosted artifact proves the exact QR
   incompatible target.
6. Preserve broad QR, least-squares, external-library, package, ABI,
   performance, and state-of-the-art non-claims.
7. Run selected comparison, normalizer, manifest, workflow, PowerShell, and
   QR-focused tests.

**Done when:** Windows QR incompatible selected freshness is promoted with
hosted proof or re-deferred with current guards and docs explicitly aligned.

## Closure Track 4: Additional Allocation-Failure Owner

1. Rank candidate owners by risk and user impact:
   - matrix construction/import/export;
   - QR factorization workspace;
   - LDLT/Cholesky numeric factorization;
   - eigensolver workspace;
   - SVD workspace.
2. Select exactly one owner for Epic 19.
3. Record lifecycle invariants:
   - return status;
   - output cleanup;
   - stale-output suppression;
   - caller-owned input preservation;
   - retry after reset;
   - partial publication behavior.
4. Extend deterministic allocation-failure harness for the selected owner.
5. Add focused tests and a Makefile gate.
6. Add registration guard if the owner has a new focused binary/label.
7. Update claim docs to say selected owner only.
8. Run focused gate, relevant family tests, and full C quality gate.

**Done when:** one new allocation-heavy owner has deterministic failure,
cleanup, stale-output, and retry proof.

## Closure Track 5: Review-Surface Reduction

1. Rank large files by risk, churn, and test value. Candidate surfaces:
   - `tests/test_ldlt_csc.c`;
   - `tests/test_etree.c`;
   - `tests/test_integration.c`;
   - `tests/test_qr.c`;
   - `tests/test_iterative.c`;
   - `scripts/normalize_report_index.py`;
   - `scripts/run_external_comparison.py`.
2. Select one cluster, not a broad cleanup.
3. Define no-behavior-change invariants.
4. Extract helper ownership or split responsibilities where reviewability
   clearly improves.
5. Add guard coverage for registration, include ownership, and execution order
   where relevant.
6. Run focused tests plus full C quality gate if C/header files changed.
7. Document the remaining large surfaces as residuals.

**Done when:** one high-risk surface is smaller, better owned, guarded, and
behavior-preserving.

## Closure Track 6: Benchmark Methodology And Threshold Decision

1. Inventory current selected hosted benchmark lanes and report fields.
2. Decide whether Epic 19 will add thresholds or explicitly defer thresholds
   with stronger methodology docs.
3. If adding thresholds:
   - choose one selected benchmark row;
   - define runner class, compiler, repeats, warmup, variance rule, and
     allowed regression threshold;
   - add pass/fail logic and CI gating for that row only.
4. If deferring thresholds:
   - add explicit methodology proof that freshness is metadata-only;
   - strengthen docs/guards to prevent accidental speedup claims.
5. Add benchmark freshness/manifest regression tests.
6. Update README, benchmark docs, maintainer guide, and selected manifest.

**Done when:** one selected benchmark lane has a reviewed threshold policy, or
the threshold deferral is closed with stronger guard coverage and no
performance promotion.

## Closure Track 7: Generated API Publication

1. Decide whether generated API HTML remains local-only or gets published.
2. If staying local-only:
   - close the publication residual with stronger guard coverage;
   - keep `docs/api/` ignored;
   - document the exact reason for no hosted docs.
3. If publishing:
   - select hosted, artifact-retained, or committed generated HTML;
   - add workflow, retention, freshness, link, and staging validation;
   - update routing docs;
   - remove or adjust local-only guards deliberately.
4. Run `make docs-check`, `make api-docs-freshness`, API routing tests, and
   workflow publication validation.

**Done when:** the generated API docs policy is no longer ambiguous and has
matching automation.

## Closure Track 8: Shared Library, ABI, And Release Readiness

1. Decide whether Epic 19 implements shared-library proof or only a release/ABI
   design record.
2. If implementing shared library proof:
   - add symbol visibility/export policy;
   - add Linux SONAME behavior;
   - add macOS install-name/RPATH behavior;
   - add Windows DLL/import-library behavior if in scope;
   - add downstream shared consumer tests;
   - update package metadata and docs.
3. If deferring implementation:
   - create a complete ABI/release design with acceptance gates and guard
     coverage so future work is executable.
4. Define release readiness checklist:
   - versioning;
   - changelog;
   - artifacts;
   - tags;
   - package provenance;
   - source archive reproducibility;
   - CI evidence.
5. Run install/export tests, static package guard, docs checks, and full C gate
   if build files/source/header files change.

**Done when:** either shared-library proof exists for a narrow platform path or
ABI/release readiness has a complete reviewed design and deferral guard.

## Closure Track 9: External Baseline And State-Of-The-Art Evidence Program

1. Choose exact external baselines and versions:
   - SuiteSparse;
   - Eigen;
   - SciPy/NumPy/LAPACK;
   - ARPACK or comparable eigensolver stack if eigensolvers are included.
2. Choose exact matrix suites and fixture subsets.
3. Define correctness metrics and tolerances per solver family.
4. Define performance metrics separately from correctness.
5. Record platform/compiler/build metadata.
6. Add generated report schema fields for baseline identity and methodology.
7. Add one pilot comparison lane if feasible.
8. Keep docs explicit that this is a blueprint or selected pilot, not broad
   parity.

**Done when:** the project has an executable state-of-the-art evidence
blueprint and optionally one selected pilot, without making broad claims.

## Closure Track 10: Final Epic 19 Closeout

1. Reconcile all Sprint 207-215 outcomes.
2. Update README, INSTALL, maintainer guide, benchmark docs, API docs, and
   planning docs to match earned evidence.
3. Publish Epic 19 retrospective.
4. Publish Epic 19 residual queue.
5. Run final validation:
   - docs guards;
   - package/static guards;
   - selected manifest/report tests;
   - generated API checks;
   - full C gate if `.c` or `.h` changed;
   - relevant hosted evidence review.
6. State exactly whether any state-of-the-art claim is earned. Default answer
   remains no unless the evidence program proves otherwise.

**Done when:** Epic 19 closes with coherent earned claims, explicit residuals,
and no stale support-status contradictions.
