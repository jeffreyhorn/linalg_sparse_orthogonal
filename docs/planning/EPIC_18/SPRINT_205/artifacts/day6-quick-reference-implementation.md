# Day 6: Quick Reference Implementation

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Theme:** Add the compact adoption quick reference to the selected
user-facing surface and route users to detailed docs.  
**Time estimate:** 12 hours  
**Branch:** `sprint-205`  
**Base commit:** `9a2b4ff6`

## Purpose

Day 6 implements the Day 4 quick-reference design and Day 5 support-truth
architecture in source-controlled user documentation. The implementation keeps
the quick reference compact and routes support/readiness questions to the
existing owner surfaces rather than creating new support claims.

## Changed Public Surfaces

| File | Change | Claim-boundary handling |
| --- | --- | --- |
| `docs/cookbook.md` | Added `## Problem-Shape Quick Reference` near the first-use ladder. | Each row has a first workflow, owner link, and short boundary. The lead-in points current support status to `INSTALL.md#support-readiness-matrix`. |
| `README.md` | Updated Start Here and Adoption Map links to route problem-shape decisions through the cookbook quick reference. | README remains the front door and still routes detailed support status to INSTALL. |
| `docs/tutorial.md` | Added the cookbook quick reference as the compact route and kept solver selection as the detailed decision tree. | Tutorial remains a learning path, not a support matrix. |
| `examples/README.md` | Added a Start Here bullet for choosing an example through the cookbook quick reference. | Examples remain runnable-owner docs; support/readiness still routes to INSTALL where relevant. |

## Implemented Quick-Reference Rows

| Row family | First workflow route | Retained boundary |
| --- | --- | --- |
| Local build and first solve | `examples/README.md#start-here`, `example_basic_solve` | Not install, package-manager, performance, or broad platform proof. |
| CSR/CSC and Matrix Market data | Cookbook data-first route and maintained examples | Storage/file format does not decide solver support or benchmark proof. |
| Direct solver shapes | Solver selection direct-solver section and examples | Selected fixture evidence is not broad parity or broad correctness proof. |
| Repeated-run direct lifecycle | README repeated-run workflow and `example_analysis` | Workflow-specific reuse, not package or ABI support. |
| Iterative and matrix-free workflows | Solver selection iterative section and examples | Iteration/residual behavior is local diagnostics, not portable performance. |
| Symmetric eigensolver | Solver selection eigensolver section and `example_eigs` | No nonsymmetric eigensolver or state-of-the-art parity claim. |
| SVD/rank/low-rank workflows | Solver selection SVD section and SVD examples | No broad SVD parity, raw singular-vector identity, or package/ABI claim. |
| Benchmarks/reports | Benchmark result interpretation docs | No portable performance, release, backend-superiority, or state-of-the-art claim. |
| Installed downstream consumer | INSTALL start path and installed consumer tutorial | Static-first only; no shared-library, dynamic ABI, or package-manager support. |
| API declarations/generated docs | API reference | Generated HTML remains local-only ignored output. |
| Threading/OpenMP controls | README runtime/backend controls and algorithm docs | No portable performance or broad platform proof. |

## Design Criteria Review

| Day 4 criterion | Day 6 result |
| --- | --- |
| Quick reference fits in one compact table or one short table plus lead-in. | Met. `docs/cookbook.md` now contains one compact table plus a short support-truth lead-in. |
| Every row links to source-controlled owner docs rather than planning artifacts. | Met. Links target cookbook, examples, solver selection, INSTALL, API reference, benchmarks, README, and algorithm docs. |
| Every support-sensitive row links or routes to the support/readiness matrix. | Met. The lead-in routes current support status to INSTALL, and installed-consumer rows link directly to INSTALL. |
| Solver rows link to solver selection instead of copying selected report caveats. | Met. Direct, iterative, eigensolver, QR, SVD, and low-rank rows link to solver selection. |
| Benchmark rows do not claim portable performance. | Met. The benchmark row states selected/local freshness does not prove portable performance. |
| API rows preserve local-only generated API policy. | Met. The API row states generated HTML is local-only ignored output, not hosted or release evidence. |
| Packaging rows avoid Homebrew/package-manager/shared-library/dynamic ABI claims. | Met. The installed-consumer row retains static-first and no shared-library/dynamic ABI/package-manager wording. |
| Windows selected freshness is not promoted. | Met. No row promotes Windows selected freshness; Cholesky/QR boundaries remain linked through solver selection and INSTALL. |
| Raw manifest/schema vocabulary stays out of the quick reference. | Met. The table avoids `support_tier`, `claim_boundary`, `freshness_policy`, selected IDs, and report row IDs. |

## Validation And Hygiene

| Check | Day 6 result |
| --- | --- |
| Public docs edited | Yes: README, tutorial, cookbook, examples. |
| Maintainer/report docs edited | No. |
| User-facing support claim changed | No new support status added; wording routes to existing owners. |
| `.c` or `.h` files changed | No. Full C gate not required for Day 6. |
| Generated output changed | No generated output intentionally created. |
| `git diff --check` | Planned after artifact creation. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 205.2 is implemented in source-controlled docs. | Met. `docs/cookbook.md` now contains the implemented quick reference. |
| Users can find the right workflow from common problem shapes. | Met. README, tutorial, and examples route to the quick reference; the table maps common shapes to first workflows and owner docs. |
| The quick reference does not broaden support, package, ABI, platform, performance, or state-of-the-art claims. | Met. Rows retain compact boundaries and route support status to INSTALL. |

## Day 6 Disposition

Day 6 is complete. Day 7 should perform the broader support-truth
consolidation pass, shortening repeated caveats where the new quick reference
and support/readiness routing make a link safer than duplicated prose.
