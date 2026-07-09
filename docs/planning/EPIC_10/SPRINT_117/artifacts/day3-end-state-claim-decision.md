# Sprint 117 Day 3 End-State Claim Decision

## Purpose

Day 3 turns the Day 2 claim inventory into explicit closeout decisions. These
decisions determine which Epic 10 claims may remain public after final
validation, which claims must stay bounded or downgraded, which claims are
deferred, and which claims are explicit non-claims.

This artifact does not edit public documentation. It is the decision source for
Day 8 unsupported-claim cleanup and the Day 13-14 Sprint/Epic retrospectives.

## Decision Rules

| Rule | Application |
|---|---|
| Claims must point to implementation, docs/artifacts, and validation evidence. | No claim is accepted from aspiration or sprint title alone. |
| Bounded evidence earns bounded wording only. | Family-local solver evidence, local benchmarks, and tiered platform proof cannot become universal parity claims. |
| Sprint 114-116 residuals remain guardrails. | Proof-owner, package/platform, ABI, Windows, adoption, and performance residuals are not silently promoted. |
| Final Sprint 117 validation is still required. | Claims marked earned here are earned pending the Day 4-6 validation package and Day 8 cleanup check. |
| Non-claims must remain visible. | Unsupported support, performance, platform, ABI, package-manager, and ecosystem claims stay explicit. |

## Earned Claim Decision Table

| Claim | Decision | Evidence basis | Public wording allowed | Validation still required |
|---|---|---|---|---|
| Compressed-first CSR/CSC workflows are a primary product path. | Earned pending final validation. | Sprint 101 constructor/API work; Sprint 111 docs/examples; README, tutorial, solver guide, and example references. | The library supports compressed-first CSR/CSC construction/export paths for callers whose data already lives in compressed storage. | Confirm current docs/examples/tests remain aligned in final validation. |
| Selected direct-solver external oracle evidence is deeper. | Earned pending final validation, selected lanes only. | Sprint 102 artifacts; maintainer-guide owner table; Cholesky CSC, LDLT CSC, and linked-list LU dense-reference lanes. | Selected direct-solver families have named external dense-reference evidence and tolerance records. | Day 7-8 final comparison package must name the exact solver families and fixtures. |
| API usability and adoption docs are clearer. | Earned pending final validation. | Sprint 111 solver guide, tutorial, Matrix Market docs, examples, and header/tutorial coherence; Sprint 116 adoption QA. | Users have clearer public workflow guidance, examples, and support-surface routing. | Documentation hygiene and Day 8 public wording recheck. |
| Matrix Market load/save behavior is documented. | Earned pending final validation, public function surface only. | Sprint 110 Matrix Market source split; Sprint 111 Matrix Market docs; Sprint 116 non-claim checklist. | Matrix Market load/save behavior is documented with supported and unsupported formats. | Keep no-public-module and no-public-builder non-claims visible. |
| Static-first package support is the current package truth. | Earned pending final validation. | Sprint 112 install/export proof; Sprint 115 package/platform decisions; README and INSTALL wording. | Static archive install/export with `pkg-config` and `find_package(Sparse)` is the maintained package story. | Run or cite relevant package validation if package/install surfaces change. |
| Platform support tiers are explicit. | Earned pending final validation, tiered only. | Sprint 112 platform-tier contract; Sprint 115 platform deferrals; Sprint 116 adoption QA. | Linux is the strongest reviewed source of truth; macOS and Windows have narrower reviewed/supplemental scopes. | Day 4-6 validation package should preserve expected-count and staged-exclusion boundaries. |

## Bounded Or Partially Earned Claim Decisions

| Claim | Decision | Required wording boundary | Day 8 cleanup trigger |
|---|---|---|---|
| Product-grade self-contained C sparse library maturity | Partially earned. | Say Epic 10 improved productization, evidence, support tiers, docs, and claim discipline; do not say the project is an unqualified state-of-the-art replacement. | Any unqualified state-of-the-art, replacement, or ecosystem-parity wording. |
| Mutable matrix shell remains supported but secondary | Bounded earned. | Describe the shell as supported compatibility/public workflow, with compressed-first paths preferred when callers already have CSR/CSC data. | Any wording implying the mutable shell was replaced or deprecated without evidence. |
| Iterative/eigensolver/SVD comparison architecture | Partially earned. | Describe fixture-local residual, convergence, reconstruction, rank, and orthogonality evidence; avoid ARPACK/LAPACK/SciPy/PETSc/Trilinos parity. | Any broad external parity or every-family validation wording. |
| Backend/runtime observability | Partially earned. | Describe clearer runtime/backend observability and bounded fallback/reporting; avoid vendor backend parity. | Any wording implying universal vendor backend or acceleration parity. |
| Local performance sentinels | Partially earned. | Describe local regression/sentinel evidence and benchmark context; avoid portable performance guarantees. | Any universal speed, portable timing, or cross-platform max-RSS threshold claim. |
| Reorder/fill and graph evidence | Partially earned. | Describe named fixture, fill metric, and report-contract evidence; avoid universal reorder/fill superiority. | Any claim that one ordering is universally best or portable memory/runtime superior. |
| Maintainability/source ownership improvement | Partially earned. | Describe source/test ownership improvements where artifacts and validation exist; keep unresolved source movement residuals explicit. | Any broad claim that all large-source, proof-owner, or giant-test risk is closed. |

## Explicit Non-Claims

These claims are rejected for Epic 10 closeout unless a later sprint replans
them with implementation and validation proof:

- unqualified state-of-the-art sparse linear algebra replacement;
- SuiteSparse, PETSc, Trilinos, ARPACK, LAPACK, SciPy, or vendor backend
  parity;
- every solver family externally validated;
- portable performance superiority;
- universal reorder/fill superiority;
- cross-platform max-RSS thresholds;
- shared-library package support;
- dynamic ABI compatibility guarantee;
- package-manager support for Homebrew, vcpkg, distro packages, Windows
  package managers, or similar recipe ecosystems;
- Windows Makefile parity;
- Windows separate install-validation parity;
- full macOS install/export parity;
- symmetric Linux/macOS/Windows reviewed parity;
- public Matrix I/O module or public builder API;
- public proof-owner/internal-helper contract;
- broad complex-number or mixed-precision maturity;
- GPU sparse kernels;
- distributed-memory sparse solvers;
- full replacement of the mutable matrix shell.

## Public Surface Decision Cross-Check

| Surface | Day 3 decision |
|---|---|
| `README.md` | Keep compressed-first, static package, CI-support split, local benchmark, and real-only scalar wording if Day 8 recheck remains clean. |
| `INSTALL.md` | Keep static-first package, reviewed platform split, Windows CMake-first story, and dynamic-ABI non-claim. |
| `docs/tutorial.md` | Keep workflow/adoption guidance bounded to public APIs. |
| `docs/solver_selection.md` | Keep compressed-first recommendation and state-of-the-art parity non-claim. |
| `docs/matrix_market.md` | Keep load/save behavior and unsupported complex/Hermitian/skew-symmetric format boundaries. |
| `docs/algorithm.md` | Keep technical-background positioning and local-performance caveats. |
| `benchmarks/README.md` | Keep local measurement and test-owner separation. |
| `examples/README.md` | Keep examples as runnable workflow references, not proof of ecosystem parity. |
| `docs/maintainer_guide.md` | Keep detailed oracle/package/platform/proof-owner interpretation as maintainer-facing evidence, not adoption marketing. |

## Unsupported-Claim Cleanup Candidates

Day 3 did not find a known required public-doc edit. Day 8 should still perform
a focused cleanup pass because final validation and comparison packaging may
surface stale wording.

| Candidate cleanup area | Current decision | Day 8 action |
|---|---|---|
| Epic 10 final maturity wording | Bounded only. | Add or keep language that says productization/evidence improved, not ecosystem replacement. |
| Solver comparison wording | Bounded by family and fixture. | Remove or fence any every-family external-validation wording. |
| Benchmark/performance wording | Local evidence only. | Remove or fence any portable performance or universal speed wording. |
| Package/platform wording | Static-first and tiered only. | Remove or fence any shared-library, dynamic ABI, package-manager, Windows install parity, or symmetric platform wording. |
| Matrix Market wording | Public load/save functions only. | Remove or fence any separate public Matrix I/O module or builder API wording. |
| Maintainability wording | Touched-owner improvements only. | Remove or fence any claim that all proof-owner/source-boundary debt is closed. |

## Day 8 Cleanup Checklist

- Re-scan public and maintainer-facing docs for the explicit non-claims above.
- Compare any Sprint 117 final validation failures or skips against public
  wording before keeping the claim.
- Keep broad maturity language qualified as productization/evidence/support
  tier progress.
- Preserve local benchmark caveats in README, benchmark docs, solver guide, and
  algorithm notes.
- Preserve static-first package and no-dynamic-ABI wording in README, INSTALL,
  and maintainer guide.
- Preserve Windows reviewed CMake subset, staged exclusions, and no separate
  install-validation claim.
- Preserve Matrix Market public-surface wording as load/save functions, not a
  separate public module or builder API.
- Record any no-edit result explicitly if the recheck stays clean.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 1 is complete. | Complete. |
| Day 8 can apply cleanup without rediscovering claim evidence. | Complete. |
| Final public claims are either earned, downgraded, or explicitly non-claims. | Complete pending Day 4-6 validation and Day 8 final cleanup recheck. |
| Day 3 remains documentation-only. | Complete. |
