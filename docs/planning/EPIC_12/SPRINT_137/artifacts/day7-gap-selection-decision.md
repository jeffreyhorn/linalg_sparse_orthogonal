# Sprint 137 Day 7 - Epic 12 Gap-Selection Decision

## Purpose

Day 7 applies the Day 6 complete-closure rubric to the Day 5 residual owner
map and selects the specific gaps Sprint 138-146 will close, defer, or reject.

The selected scope keeps Epic 12 focused on complete closure: each later sprint
gets one primary target with implementation, validation, documentation, support
tier, claim gates, and residual disposition.

## Selection Summary

| Sprint | Selected gap target | Primary owner | Closure result expected |
| --- | --- | --- | --- |
| 138 | Maintained numerical corpus/oracle contract with one durable deterministic fixture lane and explicit skip/defer semantics. | Corpus/oracle owner | Corpus fixtures, manifests, oracle rows, validation command, docs, and report handoff exist before solver claims expand. |
| 139 | QR rank-deficient nullspace/subspace residual closure backed by the Sprint 138 corpus lane. | QR owner | Focused QR fixtures, projector/subspace comparison semantics, tolerance/rank/nullity metadata, tests, docs, and QR non-claims close one high-value residual. |
| 140 | Partial-SVD repeated/clustered-spectrum residual closure with convergence-budget semantics for deterministic fixtures. | Partial-SVD owner | Focused SVD fixtures, singular-value and subspace comparison semantics, budget/failure diagnostics, tests, docs, and SVD non-claims close one high-value residual. |
| 141 | Row-meaning-preserving report index normalization plus stale-report checking for maintained evidence families. | Report-index owner | Normalized metadata and freshness gates cover selected corpus/oracle, benchmark/sentinel, package, coverage, and dead-code outputs without overclaiming. |
| 142 | Runtime/backend precedence contract plus one normalized local sentinel lane. | Runtime/backend owner | Runtime/backend control precedence is documented or typed, one sentinel lane is maintained, and portable-performance/backend-parity claims remain blocked. |
| 143 | Static-first package/ABI product decision with stricter static-first follow-through and optional-mode install/downstream proof. | Package/ABI owner | Shared-library and dynamic ABI support are explicitly deferred; static-first install/export/pkg-config/CMake proof and optional mode boundaries are strengthened. |
| 144 | Windows CMake install/downstream reviewed-parity lane, or explicit rejection with source-level blockers if hosted proof fails. | Platform owner | One Windows package/install lane is promoted only if hosted CI proof, failure semantics, expected counts, and support-tier docs are complete. |
| 145 | Adoption front door for earned build/install, solver selection, diagnostics, corpus/report, runtime, package, and platform behavior. | Adoption/docs owner | Public docs and examples become easier to follow without widening solver, package, platform, performance, or state-of-the-art claims. |
| 146 | Epic 12 final evidence inventory, claim recalibration, residual queue, and state-of-the-art assessment. | Closeout owner | Closed gaps, non-claims, validation evidence, report freshness, platform/package tiers, and future residuals are published. |

## Scored Candidate Families

Scores use the Day 6 rubric: user value, state-of-the-art relevance,
dependency readiness, testability/proofability, platform/package risk,
documentation/adoption impact, and complete-closure feasibility. Maximum score
is 21.

| Candidate family | Score | Decision | Reason |
| --- | ---: | --- | --- |
| Numerical corpus index | 20 | Select for Sprint 138 | Foundational dependency for QR, partial-SVD, report freshness, optional-data interpretation, and future state-of-the-art reassessment. |
| External-reference helper generated index | 17 | Select inside Sprint 138 where bounded | Provides oracle row semantics and fixture-key discipline if kept as a helper-specific evidence class. |
| SuiteSparse rank-deficient QR corpus evidence | 15 | Defer to Sprint 139 as optional extension | Valuable, but only after Sprint 138 defines optional-data provenance, skip behavior, runtime budget, and support tier. |
| SuiteSparse and optional-large minimum-norm expansion | 12 | Defer | Too broad for early Epic 12 unless narrowed to a single selected QR/SVD proof lane. |
| QR compatible zero-residual lane | 14 | Defer | Useful but lower value than rank-deficient subspace behavior and can remain fixture-local residual work. |
| QR wide residual-only behavior | 16 | Defer unless required by selected QR fixtures | Closeable, but rank-deficient subspace semantics provide higher proof value for solver trust. |
| QR rank-deficient nullspace/subspace expansion | 19 | Select for Sprint 139 | High user and competitive value, corpus-compatible, proofable with projector/two-way projection semantics, and closeable as one residual family. |
| Additional QR-vs-SVD minimum-norm cross-checks | 13 | Defer | Depends on both QR and partial-SVD comparison semantics; use only as supporting evidence if it falls out naturally. |
| Partial-SVD repeated/clustered spectra | 19 | Select for Sprint 140 | High solver-trust value, bounded fixture scope, and directly addresses ordering/tolerance/subspace ambiguity. |
| Partial-SVD rank-deficient subspace expansion | 16 | Defer | Valuable but overlaps selected subspace semantics; keep as residual unless Sprint 140 can close it without broadening scope. |
| Partial-SVD low-rank optimality expansion | 14 | Defer | Useful proof family, but lower priority than repeated/clustered spectra plus convergence-budget clarity. |
| Partial-SVD convergence reporting semantics | 18 | Select inside Sprint 140 where tied to selected fixtures | Required to make repeated/clustered-spectrum evidence interpretable without masking non-convergence. |
| Cross-report normalized index | 19 | Select for Sprint 141 | Enables report freshness and traceability across maintained evidence without implying broad correctness or release proof. |
| Automated stale-report scanner | 18 | Select for Sprint 141 | Completes report governance by detecting stale generated evidence after row metadata exists. |
| Runtime/backend sentinel expansion | 17 | Select for Sprint 142 | Useful if narrowed to one local sentinel lane with variance and support-tier semantics. |
| LDLT report-only sentinel | 14 | Defer | Lower priority than a general runtime/backend precedence contract and one selected sentinel lane. |
| Iterative convergence or BiCGSTAB sentinel | 15 | Defer unless selected by Sprint 142 audit | Viable as the Sprint 142 sentinel only if it best exercises the chosen precedence rule. |
| Eigensolver backend/preconditioner sentinel | 15 | Defer unless selected by Sprint 142 audit | Viable but depends on backend vocabulary and may be higher risk than iterative/direct sentinel work. |
| SVD/bidiag report rows | 13 | Defer to Sprint 140/141 byproduct only | Should not become a standalone runtime claim unless selected SVD work needs report rows. |
| Optional backend availability rows | 16 | Select as supporting Sprint 142/143 scope | Useful to explain unsupported/unavailable/probed/fallback states, but not a separate closure target. |
| Static-first optional package mode matrix | 18 | Select for Sprint 143 | Strengthens the maintained package contract without taking on full ABI/release infrastructure. |
| Shared-library packaging | 11 | Reject for Epic 12 implementation, keep as future residual | Complete closure requires ABI, loader, symbol, install, downstream, and platform proof beyond the chosen Epic 12 budget. |
| Dynamic ABI compatibility | 8 | Reject for Epic 12 implementation, keep as future residual | Depends on shared-library productization and public ABI policy not selected for this epic. |
| Package-manager support | 7 | Reject for Epic 12 implementation, keep as future residual | Requires release mechanics and ecosystem-specific recipes beyond the selected closure path. |
| macOS reviewed install/export parity | 15 | Defer | Valuable but Windows install/downstream parity has higher risk-reduction value after Sprint 143 package follow-through. |
| Windows reviewed install-validation parity | 18 | Select for Sprint 144 | High adoption value, clear CMake-first scope, hosted-runner proof path, and direct package-contract payoff. |
| Windows staged pthread/POSIX test promotion | 12 | Defer | Source portability work is valuable but higher risk and less directly tied to the selected package/adoption path. |
| Documentation-link automation | 13 | Defer unless needed by Sprint 145 validation | Helpful but not a primary product gap closure. |
| Algorithm reference continued slimming | 14 | Select only as supporting Sprint 145 scope | Useful for adoption after evidence lands, but should not drive solver claims. |
| Cookbook and adoption navigation maintenance | 18 | Select for Sprint 145 | High user value once evidence-bearing sprints settle package, platform, solver, and report wording. |
| Generic QR/SVD/minimum-norm helper consolidation | 12 | Defer unless required by Sprint 139 or Sprint 140 | Maintainability work should follow selected proof-owner needs, not become broad refactoring scope. |

## Dependency-Ordered Handoff

1. **Sprint 138 corpus/oracle first:** Define fixture taxonomy, manifest rows,
   deterministic generation, optional-data skip/defer semantics, expected
   result metadata, oracle rows, and validation commands.
2. **Sprint 139 QR residual closure:** Use Sprint 138 corpus semantics to close
   rank-deficient nullspace/subspace QR behavior with projector or two-way
   projection metrics and no raw-basis parity claim.
3. **Sprint 140 partial-SVD residual closure:** Use Sprint 138 corpus semantics
   and Sprint 139 comparison lessons to close repeated/clustered-spectrum
   behavior with convergence-budget interpretation.
4. **Sprint 141 report governance:** Normalize report metadata only after
   corpus/oracle, QR, and partial-SVD evidence rows have concrete meanings.
5. **Sprint 142 runtime/backend governance:** Use normalized report fields to
   publish one local sentinel lane and runtime/backend precedence semantics.
6. **Sprint 143 package/ABI follow-through:** Preserve static-first as the
   maintained product contract, strengthen optional-mode install/downstream
   proof, and explicitly defer shared ABI/package-manager work.
7. **Sprint 144 Windows platform lane:** Attempt Windows CMake
   install/downstream reviewed parity only after Sprint 143 package semantics
   are stable.
8. **Sprint 145 adoption simplification:** Rewrite first-use and cookbook
   paths from earned behavior after solver, report, runtime, package, and
   platform decisions are known.
9. **Sprint 146 closeout:** Reconcile evidence, support tiers, residuals, and
   state-of-the-art wording after all selected gap closures land.

## Selected Claim Boundaries

| Selected target | Claims allowed after closure | Claims still blocked |
| --- | --- | --- |
| Sprint 138 corpus/oracle | Maintained fixture taxonomy, deterministic row semantics, explicit skip/defer interpretation, and fixture-local oracle evidence. | Broad corpus completeness, SuiteSparse parity, external-library parity, and state-of-the-art numerical coverage. |
| Sprint 139 QR | Selected rank-deficient nullspace/subspace QR behavior for named fixtures with documented tolerance and comparison semantics. | General QR superiority, all rank-deficient behavior, raw basis equality, global minimum-norm guarantees, and SuiteSparse parity. |
| Sprint 140 partial-SVD | Selected repeated/clustered-spectrum partial-SVD behavior and convergence-budget diagnostics for named fixtures. | ARPACK/SciPy parity, global singular-vector ordering, broad sparse-output behavior, and unbounded convergence guarantees. |
| Sprint 141 reports | Freshness and traceability for maintained report families with preserved row meanings. | Release certification, broad correctness proof, coverage completeness, and portable performance proof. |
| Sprint 142 runtime/backend | Documented runtime/backend precedence and one local sentinel lane. | Backend parity, portable speedup, scalability, and memory superiority. |
| Sprint 143 package/ABI | Static-first install/export/downstream contract and selected optional static modes. | Shared-library ABI, dynamic loader compatibility, package-manager support, and broad platform install parity. |
| Sprint 144 platform | Windows CMake install/downstream reviewed parity only if hosted proof and support-tier docs pass. | General Windows parity, POSIX/pthread staged-test promotion, macOS reviewed promotion, and all-platform support parity. |
| Sprint 145 adoption | Simpler first-use flow and examples for earned behavior. | Any new solver, package, platform, runtime, performance, or state-of-the-art claim not backed by prior sprint evidence. |
| Sprint 146 closeout | Epic 12 closed-gap inventory and honest state-of-the-art assessment. | Unqualified state-of-the-art status unless final evidence independently earns it. |

## Deferrals and Rejections

| Residual | Disposition | Reason |
| --- | --- | --- |
| SuiteSparse rank-deficient QR corpus evidence | Deferred | Reconsider inside Sprint 139 only after Sprint 138 optional-data semantics and runtime budget exist. |
| SuiteSparse and optional-large minimum-norm expansion | Deferred | Too broad for complete closure unless narrowed to selected QR/SVD fixtures. |
| QR compatible zero-residual lane | Deferred | Lower priority than rank-deficient subspace semantics. |
| QR wide residual-only behavior | Deferred | Can become supporting evidence if selected QR fixtures require it. |
| Additional QR-vs-SVD minimum-norm cross-checks | Deferred | Depends on both QR and partial-SVD comparison semantics. |
| Partial-SVD rank-deficient subspace expansion | Deferred | Overlaps selected subspace semantics but is not the primary Sprint 140 closure target. |
| Partial-SVD low-rank optimality expansion | Deferred | Lower priority than repeated/clustered spectra and convergence-budget interpretation. |
| LDLT report-only sentinel | Deferred | Sprint 142 should select one sentinel after runtime/backend audit. |
| Iterative convergence or BiCGSTAB sentinel | Deferred by default | Candidate for Sprint 142 sentinel selection, not preselected on Day 7. |
| Eigensolver backend/preconditioner sentinel | Deferred by default | Candidate for Sprint 142 sentinel selection, not preselected on Day 7. |
| SVD/bidiag report rows | Deferred | Include only if Sprint 140 or Sprint 141 needs them for selected evidence. |
| macOS reviewed install/export parity | Deferred | Windows CMake install/downstream parity is selected as the one platform promotion lane. |
| Windows staged pthread/POSIX test promotion | Deferred | Higher source-portability risk and less directly tied to the package/adoption closure path. |
| Documentation-link automation | Deferred | May support Sprint 145 validation but is not selected as a primary gap. |
| Generic QR/SVD/minimum-norm helper consolidation | Deferred | Only permitted when required by selected QR or partial-SVD proof ownership. |
| Shared-library packaging | Rejected for Epic 12 implementation | Complete closure would require ABI, loader, symbol, install/export, downstream, platform, and documentation proof beyond the selected budget. |
| Dynamic ABI compatibility | Rejected for Epic 12 implementation | Depends on shared-library support and public ABI policy not selected for this epic. |
| Package-manager support | Rejected for Epic 12 implementation | Requires release mechanics and manager-specific proof outside the selected closure targets. |
| Unqualified state-of-the-art claim | Rejected unless Sprint 146 evidence unexpectedly earns it | Current plan closes prerequisites and selected gaps, not broad ecosystem superiority. |
| GPU or distributed-memory support | Rejected | Outside project architecture, evidence base, and Epic 12 selected scope. |
| Broad external-library parity | Rejected | Requires reproducible external comparisons across solver families beyond the selected gap closures. |

## Day 8-10 Template Requirements

Day 7 selections require the remaining Sprint 137 template days to prepare the
following handoff contracts:

- Day 8 must define corpus fixture, deterministic generated-matrix,
  optional-data skip/defer, oracle row, and oracle failure templates for the
  Sprint 138 corpus lane.
- Day 9 must define report-index and stale-report templates that preserve
  row meanings across corpus/oracle, benchmark/sentinel, package, coverage,
  and dead-code outputs.
- Day 10 must define package/ABI decision, platform promotion, downstream
  proof, and public claim templates that support the static-first Sprint 143
  path and Windows Sprint 144 lane.

## Day 7 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every later Epic 12 sprint has a selected gap target. | Complete | Selection summary assigns Sprint 138-146 one primary target each. |
| Deferred and rejected gaps have reasons. | Complete | Deferrals and rejections table records disposition and rationale for non-selected residuals. |
| No selected gap depends on unearned or unavailable evidence. | Complete | Dependency-ordered handoff places corpus/oracle before solver, report, runtime, package, platform, adoption, and closeout work, and claim boundaries block unsupported expansion. |
