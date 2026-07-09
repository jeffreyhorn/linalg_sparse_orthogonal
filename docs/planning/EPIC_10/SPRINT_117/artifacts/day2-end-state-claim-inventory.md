# Sprint 117 Day 2 End-State Claim Inventory

## Purpose

Day 2 converts the Sprint 100 state-of-the-art target and claim register into a
Sprint 117 closeout inventory. This is not the final claim decision. It records
which claims have evidence, which claims still require final validation, and
which claims remain deferred or explicit non-claims before Day 3 makes the
earned/downgrade/non-claim decision.

## Claim-State Definitions

| State | Meaning for this inventory |
|---|---|
| Earned pending final validation | Evidence exists in implementation, docs, artifacts, and prior validation; Sprint 117 still needs final closeout validation before public claims are frozen. |
| Partially earned | A bounded version appears supported, but broader wording would overclaim or needs additional evidence. |
| Deferred | Useful work remains explicitly outside Sprint 117 unless replanned with matching implementation and validation. |
| Non-claim | The project should not claim this as an Epic 10 outcome. |
| Needs Day 3 decision | The claim needs an explicit keep, downgrade, fence, or remove decision before cleanup work. |

## Sprint 100 Target Claims Extracted

| Target claim area | Sprint 100 expectation | Evidence required before claim is earned |
|---|---|---|
| Product-grade self-contained C sparse library maturity | The project becomes more usable and evidence-backed without claiming ecosystem replacement. | Public docs, examples, validation artifacts, support tiers, and explicit non-claims. |
| Compressed-first workflows | CSR/CSC workflows are obvious product-center paths while the mutable matrix shell remains supported. | Public constructors/imports, lifecycle tests, examples/docs, and compatibility wording. |
| Direct solver external oracle evidence | Direct solvers have deeper external-reference evidence on named families. | Fixture taxonomy, oracle helpers, tolerance model, and focused validation. |
| Iterative/eigensolver/SVD comparison architecture | Non-direct solver evidence is structured by convergence, residual, rank, and unsupported cases. | Family-local artifacts, test owners, residual criteria, validation commands, and non-claims. |
| Backend/runtime behavior | Optional acceleration and fallback/runtime behavior are observable and bounded. | Descriptor/observability tests, benchmark fields, fallback docs, and performance caveats. |
| Benchmark/reorder/fill evidence | Benchmark/reporting is decision-grade but not portable superiority evidence. | Report metadata, fixture contracts, local timing caveats, and bounded sentinels. |
| Maintainability and source/test ownership | Large source and giant-test risk decreases in touched families. | Before/after metrics, source-list parity, extraction artifacts, focused tests, and full C validation when applicable. |
| API usability and docs | Users can choose public workflows and examples without reading maintainer proof artifacts first. | Solver guide, Matrix Market docs, examples, README routing, and public header coherence. |
| Package/platform support tiers | Install/export, CMake, pkg-config, versioning, and support tiers are explicit. | Install scripts, downstream consumer proof, platform-tier table, expected counts, and staged exclusions. |
| Final competitive calibration | Public claims are truthful relative to mature ecosystems. | Final validation, unsupported-claim cleanup, residual queue, and explicit non-claims. |

## Evidence Mapping

| Claim area | Evidence found | Public surfaces | Inventory state | Final validation required |
|---|---|---|---|---|
| Product-grade self-contained maturity | Sprint 100-116 retrospectives; Sprint 111 docs/examples; Sprint 112 package proof; Sprint 116 adoption QA | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `examples/README.md`, `INSTALL.md` | Partially earned: productization and evidence are stronger, but broad ecosystem replacement remains a non-claim. | Day 3 claim decision; Day 4-6 final validation package; Day 8 public wording cleanup if needed. |
| Compressed-first CSR/CSC workflows | Sprint 101 compressed-first API/design/constructor work; Sprint 111 example/docs alignment | `README.md`, `docs/solver_selection.md`, `docs/tutorial.md`, `examples/README.md`, `include/sparse_csr.h` references | Earned pending final validation for bounded compressed-first construction/export wording. | Confirm examples/docs and current tests remain aligned during final validation. |
| Mutable matrix shell remains supported but secondary | Sprint 101 compatibility wording; README workflow routing | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md` | Partially earned: supported compatibility shell is documented; not replaced. | Day 3 should keep wording that avoids implying full replacement of the mutable shell. |
| Direct solver external oracle evidence | Sprint 102 artifacts; maintainer guide direct-family owner table; Cholesky CSC, LDLT CSC, and linked-list LU external dense-reference lanes | `docs/maintainer_guide.md`, tests referenced from artifacts | Earned pending final validation for selected direct-family lanes only. | Focused final comparison package should preserve named-family and named-fixture boundaries. |
| Every direct solver externally validated | Sprint 102 explicitly leaves QR/SVD broader external lanes bounded or absent | `docs/maintainer_guide.md` | Non-claim. | Day 3 should keep as non-claim; Day 8 should remove any broader wording if found. |
| Iterative comparison evidence | Sprint 103 iterative and eigensolver comparison architecture; Sprint 113-114 behavior/proof-owner tests | `docs/maintainer_guide.md`, `docs/solver_selection.md`, examples | Partially earned: fixture-local convergence and residual evidence exists; external ecosystem parity is not earned. | Day 3 should decide exact bounded wording; Day 7-8 final comparison should list unsupported broader parity. |
| Eigensolver/SVD comparison evidence | Sprint 103, Sprint 113, and Sprint 114 artifacts; SVD deterministic reconstruction/rank evidence | `docs/maintainer_guide.md`, `README.md`, examples | Partially earned: bounded residual/reconstruction evidence, not ARPACK/LAPACK/SciPy parity. | Final comparison package should separate family-local evidence from non-claims. |
| Backend/runtime observability | Sprint 104 runtime contract, descriptor, threading, and benchmark reporting artifacts | `benchmarks/README.md`, `docs/maintainer_guide.md`, README benchmark section | Partially earned: clearer runtime/benchmark evidence, not universal backend parity. | Day 4-6 validation should decide which runtime checks are current; Day 8 should keep backend wording bounded. |
| Local performance sentinels | Sprint 104 performance sentinel batch; README and benchmark docs mention local sentinel bundle | `README.md`, `benchmarks/README.md` | Partially earned: local regression evidence only. | Confirm no portable timing claim remains; rerun or document relevant benchmark/report checks if touched. |
| Reorder/fill and graph evidence | Sprint 105 fill/fixture contract and named-matrix evidence; benchmark docs scanability cleanup | `benchmarks/README.md`, `docs/algorithm.md`, `docs/maintainer_guide.md` | Partially earned: named fixture/reporting evidence, not universal reorder/fill superiority. | Day 7-8 should include final comparison summary and local-timing caveats. |
| Maintainability/source ownership | Sprint 106-110 extraction/source-list work; Sprints 113-114 proof-owner cleanup; residual source movement decisions | `docs/maintainer_guide.md`, sprint artifacts | Partially earned: touched owners improved; several large-source/proof-owner movements remain deferred. | Day 9-10 residual queue must keep unresolved source movement and proof-owner debt explicit. |
| API usability and examples | Sprint 111 solver guide, Matrix Market docs, compressed-first examples, tutorial/header coherence | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `docs/matrix_market.md`, `examples/README.md` | Earned pending final validation for bounded adoption workflow clarity. | Day 3 and Day 8 should ensure no docs imply unsupported APIs or package/platform maturity. |
| Matrix Market public behavior | Sprint 110 internal source split; Sprint 111 behavior docs; Sprint 116 non-claim checklist | `docs/matrix_market.md`, `docs/solver_selection.md`, `examples/README.md` | Earned pending final validation for load/save behavior; no separate public Matrix I/O module or builder API claim. | Keep module/builder non-claims explicit. |
| Static-first package support | Sprint 112 install/export proof; Sprint 115 package/platform decisions | `INSTALL.md`, `README.md`, `docs/maintainer_guide.md` | Earned pending final validation for static archive, pkg-config, and CMake package shape. | Day 4-6 should select package/install validation if package surfaces are touched. |
| Platform support tiers | Sprint 112 platform-tier contract; Sprint 115 deferrals; Sprint 116 adoption QA | `INSTALL.md`, `README.md`, `docs/maintainer_guide.md` | Earned pending final validation for tiered support wording; symmetric parity remains non-claim. | Day 3 should confirm wording stays tiered; Day 4-6 should include reviewed/supplemental lane inventory. |
| Shared-library and dynamic ABI support | Sprint 112 and Sprint 115 selected static-first contract and deferred dynamic ABI | `INSTALL.md`, `docs/maintainer_guide.md` | Non-claim. | Keep as explicit non-claim; do not add ABI wording without product contract proof. |
| Package-manager support | Sprint 115 deferred; Sprint 116 removed ambiguous wording | `README.md`, `INSTALL.md` | Non-claim. | Confirm no Homebrew/vcpkg/distro/Windows package-manager support claim appears. |
| Windows install-validation or Makefile parity | Sprint 112 platform-tier contract; Sprint 115 Windows deferrals | `INSTALL.md`, workflow comments, `docs/maintainer_guide.md` | Non-claim beyond reviewed MSVC CMake-first subset. | Keep expected Windows CTest count and exclusions bounded if referenced. |
| Broad state-of-the-art, SuiteSparse/PETSc/Trilinos parity | Sprint 100 disallowed broad claims; Sprint 116 guardrail recheck | public docs | Non-claim. | Day 3 should reject any unqualified broad claim; Day 8 should clean up if a surface contains one. |
| Complex and mixed precision maturity | Sprint 100 non-goal; README scalar wording | `README.md`, `docs/matrix_market.md`, `docs/maintainer_guide.md` | Non-claim. | Keep real-only/double wording and Matrix Market complex unsupported wording. |

## Public Claim Scan Summary

The current adoption-facing scan found bounded public wording rather than
obvious broad overclaims:

- `README.md` advertises compressed-first construction, static package usage,
  CI support split, local benchmark/sentinel boundaries, real-only scalar
  scope, and links to deeper support docs.
- `INSTALL.md` states the static-first package shape, reviewed platform split,
  Windows CMake-first consumer story, and dynamic-ABI non-claim.
- `benchmarks/README.md` repeatedly fences benchmark rows as local measurement
  artifacts rather than portable performance guarantees.
- `docs/solver_selection.md` recommends compressed-first workflows when data
  is already CSR/CSC and explicitly rejects portable state-of-the-art parity.
- `docs/matrix_market.md` documents supported Matrix Market load/save behavior
  and unsupported complex/Hermitian/skew-symmetric features.
- `docs/algorithm.md` positions itself as technical background, not an install,
  support, ABI, or portable-performance reference.
- `examples/README.md` demonstrates supported workflows without promoting
  SuiteSparse or ecosystem parity claims.
- `docs/maintainer_guide.md` owns detailed package/platform, oracle,
  benchmark, and proof-owner interpretation for maintainers.

Day 3 should still make an explicit claim decision because the final validation
package has not yet been run in Sprint 117.

## Unsupported Or Partially Supported Claim Candidates

These candidates require a Day 3 disposition before any cleanup work:

| Candidate | Current inventory disposition | Day 3 question |
|---|---|---|
| Product-grade self-contained maturity | Partially earned | What exact public wording is allowed without implying ecosystem replacement? |
| Compressed-first product model | Earned pending final validation | Which files are the canonical evidence sources for the final claim? |
| Direct solver oracle evidence | Earned pending final validation for selected lanes | How should the final package name the Cholesky CSC, LDLT CSC, and linked-list LU boundaries? |
| Iterative/eigensolver/SVD comparisons | Partially earned | Which solver-family claims stay fixture-local and which become residuals? |
| Backend/runtime observability | Partially earned | Which runtime claims require validation commands in Days 4-6? |
| Performance sentinels and benchmark reports | Partially earned | Are existing local-sentinel and benchmark caveats sufficient, or does wording need cleanup? |
| Reorder/fill evidence | Partially earned | How should named-matrix evidence be summarized without universal superiority wording? |
| Maintainability improvement | Partially earned | Which source/test ownership improvements can be claimed, and which residuals stay deferred? |
| Package/platform support tiers | Earned pending final validation | Which validation lanes must be rerun or referenced before final closeout? |
| Adoption docs and examples | Earned pending final validation | Does any public wording require Day 8 cleanup after the claim decision? |

## Day 3 Audit Checklist

- Decide the exact allowed wording for the final Epic 10 maturity claim.
- Mark each target claim as earned, partially earned, unsupported, deferred, or
  explicit non-claim.
- Keep broad replacement, ecosystem parity, portable superiority, dynamic ABI,
  package-manager, and symmetric platform parity claims as non-claims unless
  new evidence exists.
- Decide which public surfaces need Day 8 cleanup:
  - `README.md`
  - `INSTALL.md`
  - `docs/tutorial.md`
  - `docs/solver_selection.md`
  - `docs/matrix_market.md`
  - `docs/algorithm.md`
  - `benchmarks/README.md`
  - `examples/README.md`
  - `docs/maintainer_guide.md`
- Decide which validation lanes Days 4-6 must run or explicitly document:
  - documentation hygiene;
  - full C quality chain if `.c` or `.h` files change;
  - CMake/CTest parity if CMake or registration surfaces change;
  - install/export proof if package surfaces change;
  - source-list parity if source ownership or build metadata changes;
  - benchmark/report regeneration if benchmark semantics or reports change.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Every Sprint 100 target claim has an owner and evidence disposition. | Complete. |
| Unsupported or partially-supported claims are visible before cleanup. | Complete. |
| No public claim is accepted without an evidence source or explicit non-claim. | Complete for inventory; final acceptance is Day 3. |
| Day 2 remains documentation-only. | Complete. |
