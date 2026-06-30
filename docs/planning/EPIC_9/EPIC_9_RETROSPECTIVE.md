# Epic 9 Retrospective

**Epic:** 9 - Product Surface, Maintainability, Assurance, Packaging, And
Closeout Modernization
**Sprint range:** 90-99
**Status:** Complete

## Epic Closeout Thesis

Epic 9 turned a broad contradiction set into a bounded, validated product
surface. It materially improved compressed-first entry paths, selected
backend/direct-family maturity, package/export proof, maintained comparison
lanes, source/proof-owner maintainability, and public claim hygiene.

It did not turn the project into a fully compressed-first, broad
complex/mixed-precision, backend-neutral, shared-library-first, symmetric
cross-platform, benchmark-superiority product. Those remain explicit
non-claims or post-Epic-9 residuals.

## What Epic 9 Resolved Or Improved

1. **Product direction became more explicit.**
   The library now has clearer compressed-first construction/export and direct
   workflow language while preserving the linked-list shell as the mutable
   compatibility owner.

2. **Backend and direct-family proof matured on named lanes.**
   Cholesky CSC and LDLT CSC gained stronger maintained evidence, including
   external dense-reference solve checks on named fixtures.

3. **Capability surfaces widened without hiding limits.**
   Scalar/index seams, direct solver paths, eigensolver/SVD/QR-related
   surfaces, and workflow examples improved, while broad complex and
   mixed-precision maturity stayed outside the claim set.

4. **Build and package proof became sharper.**
   Static-first install/export and CMake consumer proof are now validated by
   scripts that assert both positive installed artifacts and negative
   shared-library non-claims.

5. **Workflow and platform language became more precise.**
   Linux remains the strongest reviewed source of truth. macOS and Windows
   have narrower reviewed or supplemental roles, with Windows intentionally
   scoped to a reviewed CMake-first subset.

6. **Benchmark and comparison language became safer.**
   Reorder/fill and canonical benchmark reporting now support bounded local
   calibration, not portable timing thresholds or universal superiority
   claims.

7. **Maintainability improved in selected places.**
   Sprint 96 and adjacent sprints reduced selected source/proof-owner
   concentration and clarified ownership, even though large files and giant
   tests remain.

8. **Epic 9 ended with validation rather than only documentation.**
   Sprint 99 closed with `make quality-review-full`, install/export proof,
   CMake consumer proof, example execution, and benchmark/reporting generation
   all passing.

## What Epic 9 Did Not Resolve

1. **The linked-list shell remains part of the public identity.**
   Compressed-first entry paths improved, but the shell is still the mutable
   compatibility owner.

2. **Large implementation and proof owners remain.**
   Several large source and test files remain active maintainability debt.

3. **External comparison depth is still narrow.**
   Cholesky CSC and LDLT CSC have maintained lanes. Iterative, eigensolver,
   QR, SVD, broader LDLT corpus, and ecosystem comparisons require future
   architecture.

4. **Cross-platform proof is intentionally asymmetric.**
   The project did not become symmetric across Linux, macOS, and Windows.

5. **The package story remains static-first.**
   Shared-library-first packaging, package-manager integration, and dynamic
   ABI guarantees were not implemented or claimed.

6. **Benchmark evidence remains local calibration.**
   No portable timing or universal reorder/fill superiority claim is supported.

## Epic Metrics

| Metric | Epic 9 close state |
|---|---:|
| sprint range | 90-99 |
| final reviewed local baseline | `make quality-review-full` passed |
| local Make/CMake test-count parity | 54/54 |
| full CTest result in closeout | 54 passed, 0 failed |
| maintained install/export proof scripts | 2 |
| Day 11 package proof results | Make 14/0, CMake 16/0/0 |
| representative examples executed in closeout | 4 |
| canonical benchmark report files regenerated | 6 |
| final unsupported claims to remove | 0 |
| post-Epic-9 carry-forward queue items | 8 |

## Epic Lessons

- A contradiction map is more useful than a broad theme when an epic spans
  product, implementation, packaging, CI, and docs.
- Evidence lanes should be selected before validation; otherwise closeout can
  drift into opportunistic proof expansion.
- Non-claims need named owners. Saying "not supported" is weaker than tying
  the guardrail to README, INSTALL, workflows, benchmark docs, maintainer
  guidance, and tests.
- Static-first package proof is stronger when scripts assert both what should
  install and what must not install.
- Benchmarks need manifest language and report notes that prevent local
  timings from becoming product claims.
- Make/CMake parity checks are valuable because they catch registration drift
  without pretending every platform has identical reviewed scope.
- Source/test extraction should stay family-local and validation-backed; broad
  extraction campaigns are too risky for closeout.

## Evidence Package

Authoritative closeout evidence:

- [Sprint 99 closeout evidence package](./SPRINT_99/artifacts/day12-closeout-evidence-package.md)
- [Sprint 99 final residual queue](./SPRINT_99/artifacts/day9-final-residual-queue.md)
- [Sprint 99 reviewed validation](./SPRINT_99/artifacts/day10-reviewed-validation.md)
- [Sprint 99 surface validation](./SPRINT_99/artifacts/day11-surface-validation.md)
- [Post-Epic-9 handoff](./POST_EPIC_9_HANDOFF.md)

## Handoff

Post-Epic-9 planning should start from the Day 9 residual queue and the Day 12
closeout evidence package. Future work should begin with boundary artifacts
before implementation when it touches external comparison, benchmark/reporting,
platform proof, package maturity, or large source/test extraction.
