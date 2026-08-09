# Day 10 Residual Queue Design

## Scope

Day 10 designs the final Epic 12 residual queue from Sprint 137-145
retrospectives and Sprint 146 Days 2-9 inventories, validation results, CI
reconciliation, and claim audits. The queue favors complete closure of a few
high-value gaps over partial movement across many broad claims.

This is a design artifact. Day 11 will publish the final residual queue and
next-epic handoff with cross-links.

## Residual Groups

| Group | Residual Theme | Why It Remains Open |
| --- | --- | --- |
| Numerical coverage | Broad QR/SVD/partial-SVD correctness, external parity, and state-of-the-art numerical claims | Epic 12 closed selected fixture-local residuals, not broad solver-family or ecosystem parity. |
| Report and validation | Generated report refreshes, hosted CI reconciliation, coverage/dead-code/sentinel freshness | Source-controlled metadata exists, but many generated reports are local/advisory and branch-specific hosted Sprint 146 CI is unavailable. |
| Runtime/backend | Future typed-control promotion, additional backend sentinel rows, optional backend availability | Sprint 142 documented governance and local sentinels without expanding public API, backend portability, or portable performance claims. |
| Package/ABI | Shared-library support, dynamic ABI policy, loader proof, package-manager distribution | Sprint 143 deliberately chose static-first support and deferred shared ABI productization. |
| Platform | Windows Makefile parity, Windows `pkg-config`, reviewed Windows install-validation parity, staged POSIX/pthread tests | Windows remains reviewed CMake-first with supplemental CMake install/downstream confidence. |
| Adoption/docs | Tutorial alignment and broader public-header cleanup | Sprint 145 improved the front door and four headers; wider docs/header cleanup was intentionally bounded. |
| Competitive positioning | State-of-the-art assessment and external-library feature/performance comparison | No direct comparative evidence exists for a state-of-the-art sparse linear algebra claim. |

## Prioritized Residual Queue

| Priority | Residual | Group | Owner | Blocker | Prerequisites | Promotion Gate | Classification |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Branch-specific hosted CI reconciliation for Sprint 146 | Report and validation | CI maintainer | `sprint-146` has no hosted branch/PR run yet | Branch pushed and PR/workflow runs available | Linux, macOS, and Windows workflow conclusions recorded with run IDs and support-tier implications | Evidence refresh |
| 2 | Windows staged test portability closure | Platform | Platform maintainer | pthread APIs in `test_threads` and `test_sprint4_integration`; POSIX temp-file APIs in `test_fuzz` | Source-level portability design and Windows-native temp/thread strategy | Windows hosted CMake lane intentionally registers and executes promoted tests with updated expected count and docs/report rows | Product work |
| 3 | Windows reviewed install-validation parity decision | Platform/package maintainer | Current Windows install/downstream path is supplemental only | Product decision on Windows install-validation scope and proof ownership | Workflow, report rows, docs, and hosted proof explicitly promote or reject reviewed Windows install-validation parity | Product decision |
| 4 | Shared-library ABI productization | Package maintainer | No export/import macros, symbol allowlist, visibility policy, ABI versioning, loader proof, or shared package metadata | Static-first contract remains stable; ABI design approved | Shared build/install/export works on Linux/macOS/Windows with ABI policy, loader tests, symbol checks, package metadata, and docs | Product work |
| 5 | Broad QR residual expansion | QR owner | Current proof is fixture-local; broad rank threshold, solve, minimum-norm, reorder, and external corpus behavior remain unproved | Maintained fixture taxonomy and comparison semantics for QR expansion | Multiple reviewed fixture families with compiled proof owners, corpus/oracle rows, and claim audit updates | Numerical product work |
| 6 | Broad partial-SVD residual expansion | SVD owner | Current proof is fixture-local; repeated-spectrum, rank-deficient null-space, sparse-output, convergence-rate, and partial-result guarantees remain unproved | Maintained fixture taxonomy and subspace-safe comparisons | Multiple reviewed partial-SVD fixture families with proof owners and bounded public wording | Numerical product work |
| 7 | Generated benchmark, sentinel, coverage, dead-code, and guardrail refresh package | Report/benchmark maintainer | Day 5 did not refresh these generated families; local reports can be absent/stale | Decide which generated families are needed for the next claim audit | Maintained commands regenerate selected reports and `normalize_report_index.py --require-generated <family> --check-freshness` passes | Evidence refresh |
| 8 | Tutorial alignment with first-use ladder | Documentation maintainer | Sprint 145 prioritized README, INSTALL, examples, cookbook, solver-selection, and selected headers | Day 13 Sprint 145 adoption claim map | Tutorial routes through the same build, first solve, data input, solver choice, diagnostics, and install/downstream ladder; public claim scan passes | Documentation cleanup |
| 9 | Broader public-header cleanup | API maintainer | Sprint 145 intentionally cleaned only four high-impact headers | Header-specific cleanup design and declaration-preservation strategy | Remaining public headers updated without signature/typedef/enum/macro/struct-field drift; `make format && make lint && make test` passes | Documentation/API cleanup |
| 10 | Runtime/backend typed-control promotion review | Runtime/backend owner | Sprint 142 classified several env/build/report controls as maintainer-only or deferred | ABI/package implications understood | A selected control is promoted with typed API design, tests, docs, and ABI/package non-claim review, or remains explicitly non-API | Product decision |
| 11 | Additional runtime/backend sentinel rows | Runtime/backend and benchmark owners | Current sentinel set is intentionally narrow and local-only | Maintained command and row semantics for each proposed sentinel | New rows added with local-only/advisory or hard-gate classification and no portable performance wording | Evidence/product guardrail |
| 12 | External-library parity study | Numerical lead | No direct comparative evidence across external libraries and fixture families | Stable internal fixture corpus and comparison metrics | Comparative study against named libraries exists, with exact scope, tolerances, platforms, versions, and non-claims | Competitive evidence |
| 13 | State-of-the-art competitive decision | Epic owner | No direct comparative feature/performance/correctness package supports state-of-the-art language | External-library parity and performance studies, platform proof, package maturity proof | Either a narrow evidence-backed claim is approved or state-of-the-art remains an explicit non-claim | Competitive/product decision |
| 14 | Package-manager distribution | Package maintainer | No package recipes, release ownership, binary/source policy, or install lifecycle support | Static or shared package decision stable | Package-manager recipe(s), CI install proof, versioning policy, support docs, and uninstall/update validation exist | Product work |

## Owner And Blocker Map

| Owner Role | Residuals Owned | Primary Blockers |
| --- | --- | --- |
| CI maintainer | branch-specific hosted CI reconciliation; hosted log capture policy | Branch/PR not pushed yet; hosted logs external to source control. |
| Platform maintainer | Windows staged tests; Windows Makefile/`pkg-config`; Windows install-validation parity | pthread/POSIX source dependencies; missing Windows-specific proof lanes and support decisions. |
| Package maintainer | shared-library ABI; package-manager distribution; static/shared selectors | ABI design, symbol visibility, loader tests, package metadata, cross-platform proof, release ownership. |
| QR owner | broad QR residual expansion | fixture breadth, rank-threshold semantics, solve/minimum-norm/reorder proof, external corpus policy. |
| SVD owner | broad partial-SVD residual expansion | fixture breadth, subspace-safe comparisons, convergence and partial-result contracts. |
| Report/benchmark maintainer | generated report refreshes; benchmark/sentinel/coverage/dead-code freshness | generated local outputs are not source-controlled pass proof; measurement environment dependence. |
| Runtime/backend owner | typed-control promotion; additional sentinel rows | API/ABI implications, maintainer-only env controls, portable performance non-claims. |
| Documentation maintainer | tutorial alignment | avoiding duplicated support-tier detail and preserving claim boundaries. |
| API maintainer | broader public-header cleanup | declaration preservation and full C quality gate for header changes. |
| Epic owner | external-library parity study; state-of-the-art decision | direct comparative evidence absent. |

## Promotion-Gate Definitions

| Gate | Definition |
| --- | --- |
| Hosted CI gate | Hosted workflow run IDs, conclusions, job names, commit SHA, branch/PR context, and support-tier interpretation are recorded. |
| Numerical proof gate | Source-controlled fixture metadata, expected rows, compiled proof owner, validation command, generated-local classification, public wording, and non-claim scan all pass. |
| Report freshness gate | Relevant generated family is regenerated and `normalize_report_index.py --require-generated <family> --check-freshness` passes without reclassifying local rows as hosted proof. |
| Package product gate | Build/install/export metadata, downstream proof, unsupported-artifact checks, docs, report rows, and cross-platform CI lanes agree on the package contract. |
| ABI gate | Export/import policy, symbol visibility, ABI versioning, loader behavior, compatibility policy, and platform-specific proof are present. |
| Platform promotion gate | Workflow classification, hosted proof, expected test counts, docs/report rows, and staged blocker updates all agree. |
| Documentation coherence gate | Public/support docs map every claim to evidence or non-claim, and claim scans pass. |
| Header cleanup gate | Header diff preserves declarations and full C quality gate passes. |
| Competitive claim gate | Direct comparative evidence names baselines, versions, fixtures, metrics, platforms, and caveats before any competitive claim is promoted. |

## Priority Rationale

1. **Hosted CI reconciliation comes first** because it gates final Sprint 146
   closeout truth and is currently the only branch-specific evidence gap.
2. **Windows parity work is high value but should be closed as complete lanes**
   rather than partial wording changes.
3. **Shared-library ABI is a product decision, not a cleanup task**; it should
   only start when the project is ready to own ABI and loader proof.
4. **Numerical expansion should follow the successful fixture-local pattern**
   from Sprints 139-140 rather than widening public claims from existing
   narrow fixtures.
5. **Generated report refreshes are evidence-maintenance work** and should be
   requested only for families needed by a claim or review.
6. **Docs/header cleanup should remain bounded** so adoption improvements do
   not imply API, ABI, package, platform, or numerical proof.
7. **Competitive positioning is last** because it depends on multiple earlier
   product and evidence gates.

## Non-Claim Linkage

Each residual stays an explicit non-claim until its promotion gate passes:

- state-of-the-art status stays a non-claim until the competitive claim gate
  passes;
- broad QR/SVD/partial-SVD correctness stays a non-claim until numerical proof
  gates pass across broader fixture families;
- generated report freshness stays a non-claim until relevant generated
  families are refreshed and checked;
- portable performance stays a non-claim until a separate benchmark design and
  competitive evidence package exists;
- shared-library ABI, runtime-loader, package-manager, and static/shared
  selector support stay non-claims until package and ABI gates pass;
- Windows Makefile, `pkg-config`, install-validation parity, and staged test
  closure stay non-claims until platform promotion gates pass.

## Day 11 Handoff

Day 11 should convert this design into a published residual queue with stable
cross-links to the Day 2-9 evidence artifacts and next-epic handoff
candidates. The publication should keep the non-claim linkage visible so
future plans do not accidentally convert residuals into promises.
