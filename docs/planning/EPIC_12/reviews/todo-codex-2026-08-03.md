# Epic 12 Gap-Closure Todo - Codex - 2026-08-03

This todo converts `review-codex-2026-08-03.md` into an execution sequence for
Epic 12. The principle is deliberate narrowing: close fewer gaps completely
instead of touching every state-of-the-art shortcoming superficially.

## Guiding Rules

1. Do not claim state-of-the-art status unless implementation, external
   comparison, reproducibility, packaging, platform, and documentation proof
   justify it.
2. Treat static-first packaging, platform tiers, report interpretation, and
   benchmark locality as product contracts, not incidental wording.
3. Close gaps with code, tests, docs, and validation together.
4. Prefer a maintained corpus lane over one-off fixtures that cannot be
   regenerated or interpreted.
5. Split giant tests only along real proof boundaries.
6. Promote platform support only after reviewed CI proof exists.
7. Keep deferred residuals explicit and dependency ordered.

## Step-by-Step Plan

### 1. Freeze the Post-Epic-11 Baseline

- Capture current metrics for source/test size, CMake test count, Windows
  reviewed subset count, package proof, report artifacts, and deferred queues.
- Reconcile Epic 11 residuals into Epic 12 owners:
  - QR residual and SuiteSparse/corpus expansion;
  - partial-SVD edge-case and convergence-budget expansion;
  - corpus/report normalized indexing;
  - runtime/backend sentinel expansion and stale-report automation;
  - shared-library, ABI, runtime-loader, and package-manager productization;
  - macOS/Windows package install/export promotion;
  - Windows staged pthread/POSIX portability.
- Define which gaps Epic 12 will close and which it will explicitly leave as
  non-goals.

### 2. Design the Maintained Numerical Corpus Contract

- Define fixture taxonomy for:
  - symmetry and definiteness;
  - rank and nullspace;
  - rectangularity and least-squares shape;
  - conditioning and scaling;
  - sparsity pattern, fill, graph shape, and ordering;
  - expected convergence, stagnation, singularity, and failure modes.
- Decide where fixtures live, how optional external data is downloaded or
  skipped, and how generated matrices are made deterministic.
- Define row schemas for corpus manifests, oracle outputs, skip reasons, and
  stale-report detection.
- Add one sustained corpus lane before adding many fixtures.

### 3. Close the QR Priority Residual

- Pick one high-value QR residual family from the Epic 11 residual queue.
- Add or strengthen deterministic fixtures for rank-deficient, rectangular,
  minimum-norm, nullspace, and least-squares behavior as applicable.
- Add dense or external oracle comparison for the chosen family.
- Split `tests/test_qr.c` only enough to give the new proof owner a stable
  home.
- Update solver-selection and algorithm docs with earned, bounded wording.

### 4. Close the Partial-SVD Priority Residual

- Pick one high-value partial-SVD residual family from the Epic 11 residual
  queue.
- Add deterministic edge-case fixtures and convergence-budget tests.
- Define comparison semantics for singular values, vectors, subspaces,
  ordering, tolerance, partial convergence, and skip behavior.
- Add sustained oracle/report output for the selected family.
- Update non-claims so full SVD, partial SVD, and external-library parity stay
  distinct.

### 5. Normalize Report Indexes and Freshness Gates

- Inventory current benchmark, performance sentinel, guardrail, coverage,
  dead-code, package, and oracle report outputs.
- Define a shared minimal metadata contract:
  - report family;
  - generator command;
  - source commit;
  - platform/compiler/configuration;
  - row meaning;
  - support tier;
  - freshness status;
  - skip/defer reason.
- Implement normalization only where row meaning can be preserved honestly.
- Add stale-report checks for maintained generated artifacts.

### 6. Strengthen Runtime and Backend Governance

- Audit runtime controls across OpenMP, backend selection, dense helpers,
  direct solver dispatch, eigensolver backends, environment variables, and
  typed options.
- Choose a precedence rule and document it as a maintained contract.
- Convert the highest-value remaining environment-only controls into typed
  options or explicitly keep them maintainer-only.
- Add sentinel rows and docs for backend/runtime behavior without claiming
  portable performance.

### 7. Decide and Implement the Package/ABI Path

- Make an explicit Epic 12 product decision:
  - implement shared-library ABI support; or
  - preserve static-first-only support as the maintained contract for another
    epic.
- If implementing ABI support:
  - add shared-library build rules;
  - define symbol visibility/versioning policy;
  - add install/export/pkg-config/CMake proof;
  - add loader and downstream tests on supported platforms;
  - update README, INSTALL, CMake, pkg-config, and maintainer docs.
- If deferring:
  - keep static-first deferral proof strict;
  - remove any ambiguous ABI/package-manager wording.

### 8. Promote One Platform Lane Completely

- Choose the highest-value platform promotion that can be fully closed:
  - macOS CMake install/export reviewed parity;
  - Windows CMake install/downstream reviewed parity;
  - Windows staged pthread/POSIX test portability;
  - Linux package/report/source-of-truth strengthening.
- Fix source, scripts, CI, docs, and support-tier language together.
- Do not promote a platform lane from supplemental to reviewed without CI
  proof and failure semantics.

### 9. Simplify the Adoption Surface After Evidence Lands

- Add or refine a high-level adoption front door only after QR/partial-SVD,
  corpus/report, runtime, package, and platform decisions are known.
- Keep the first-user path short:
  - install/build;
  - choose workflow;
  - run one solve;
  - inspect diagnostics;
  - know where deeper controls live.
- Move maintainer-only proof details out of first-use docs.
- Ensure examples, README, cookbook, solver-selection, INSTALL, benchmark
  docs, and maintainer guide agree on support tiers.

### 10. Close Epic 12 with Evidence

- Run the strongest reviewed quality gates for touched surfaces.
- Publish an Epic 12 retrospective with:
  - earned claims;
  - non-claims;
  - closed gaps;
  - residuals;
  - report freshness;
  - package/platform support tiers;
  - state-of-the-art assessment.
- If state-of-the-art status remains unearned, say so directly.

## Recommended Epic 12 Non-Goals

- GPU support.
- Distributed-memory support.
- Broad GraphBLAS/PETSc/Trilinos replacement claims.
- Universal external-library parity across all solver families.
- Portable performance superiority.
- Broad package-manager release across multiple ecosystems unless ABI and
  release mechanics are complete first.
- Full decomposition of every giant test file.

## Completion Definition

Epic 12 is complete when:

- the post-Epic-11 baseline is frozen and reconciled;
- at least one maintained numerical corpus/oracle lane exists with report
  semantics;
- the selected QR residual is closed with tests, docs, and claim updates;
- the selected partial-SVD residual is closed with tests, docs, and claim
  updates;
- report metadata/freshness is normalized for maintained report families;
- runtime/backend controls have a clearer maintained contract;
- the package/ABI path is either implemented with proof or explicitly deferred
  with enforced static-first checks;
- at least one platform support gap is fully promoted or explicitly rejected;
- adoption docs reflect the earned product truth;
- final validation passes for all touched surfaces.

