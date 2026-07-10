# Sprint 118 Day 6 Residual Owner and Dependency Map

## Purpose

Day 6 turns the deduplicated residual intake from Day 5 into an owner,
dependency, and proof-gate map for Epic 11. It assigns each residual to a
Sprint 119-127 owner or an explicit future-epic deferral bucket and records the
conditions that must be met before implementation or claim expansion.

## Residual Owner Table

| Residual owner candidate | Primary owner | Dependencies | Proof gates |
|---|---|---|---|
| Eigensolver private owner movement | Sprint 119 | Sprint 118 baseline, residual map, source-list/CMake expectations | Exact old/new file plan, internal header contract, source-list update, CMake update, focused consumer proof, CTest count evidence, rollback instructions. |
| `s20_select_indices` and `s20_lift_ritz_vectors` movement | Sprint 119 | Movement feasibility audit and grow-m/thick-restart/LOBPCG proof inventory | Public-result invariants, partial-publication behavior proof, compile-unit proof, CMake/source-list parity, explicit move/defer decision. |
| Shift-invert setup/conversion movement | Sprint 119 | LDLT lifecycle proof, operator selection proof, public error propagation proof, cleanup ownership proof | Focused shift-invert tests, failure cleanup proof, source-list/CMake parity, no broad eigensolver parity claim. |
| `lanczos_iterate_op` movement | Sprint 119 | Consumer inventory across basic Lanczos, thick restart, and LOBPCG-adjacent paths | Compile-unit proof for all consumers, focused behavior tests, rollback instructions. |
| Direct/iterative generated-RHS oracle | Sprint 120 | Sprint 119 source-boundary lessons and Sprint 118 templates | Shared fixture design that keeps solver-specific tolerances visible, pilot proof, focused tests, reviewed quality if C changes. |
| Direct/iterative giant-test split | Sprint 120 | Oracle ownership audit and helper design | Before/after responsibility map, CTest membership preservation, Make/CMake parity, failure localization proof. |
| SVD/QR/rank-deficient proof helpers | Sprint 121 | Sprint 120 fixture/oracle patterns | Helper extraction that preserves storage, leading-dimension, rank, reconstruction, orthogonality, and tolerance evidence at assertions. |
| SVD/QR dense or external reference pilot | Sprint 121 | Matrix taxonomy and current SVD/QR proof-gap audit | Deterministic fixture, bounded reference source, trust-boundary docs, no LAPACK/SciPy parity claim. |
| Numerical corpus taxonomy | Sprint 122 | Sprint 120-121 oracle taxonomy decisions | Tags for symmetry, definiteness, rank, conditioning, scaling, sparsity pattern, ordering, expected failures, and solver family. |
| Report index architecture | Sprint 122 | Corpus inventory and coverage/report ownership decisions | Generated or documented index contract, stale-report decision, reviewed/supplemental/local classification. |
| Coverage architecture | Sprint 122 | Current coverage threshold and risk-ranked owner list | Risk-based coverage interpretation, no vanity-percentage claim, affected report checks. |
| Performance/backend governance | Sprint 123 | Report index model and current benchmark/backend non-claims | Hot-path inventory, bounded local sentinel design, explicit non-portable interpretation, focused benchmark/report validation. |
| Package/ABI product decision | Sprint 124 | Day 4 static-first package truth and Sprint 118 residual owner map | Decide shared-library support versus static-first continuation; add build/install/symbol/ABI proof if support is added or explicit deferral checks if not. |
| Package-manager support | Sprint 124 closeout or future epic | Package/ABI product decision | Real recipes and install/consumer proof for each claimed manager/platform; otherwise publish as residual/non-claim. |
| Linux install CI lane | Sprint 125 | Sprint 124 package/ABI decision | Promote or explicitly defer with CI runtime ownership, install-script proof, support wording, and validation. |
| macOS CMake install/export parity | Sprint 125 | Sprint 124 package/ABI decision and macOS reviewed lane constraints | Add, strengthen, or explicitly defer reviewed install/export parity with CI proof and support wording. |
| Windows install validation | Sprint 125 | Sprint 124 package/ABI decision and Windows CMake subset truth | MSVC install/downstream consumer proof, expected CTest count impact, non-claim wording, or explicit deferral. |
| Windows thread/fuzz/property staged tests | Sprint 125 | Windows test membership and native behavior audit | Native Windows behavior proof, CTest count updates, staged-exclusion update, or explicit deferral. |
| Algorithm doc split | Sprint 126 | Sprint 122 report decisions and Sprint 124-125 package/platform truth | Public/current reference versus historical appendix design, redirects/link checks, claim-boundary scan. |
| Compressed-first cookbook and docs simplification | Sprint 126 | Product truth map, package/platform truth, oracle/performance outcomes | Cookbook examples/docs with current evidence boundaries and no unsupported claims. |
| Final claim recalibration and residual publication | Sprint 127 | Sprint 118-126 artifacts and validation outcomes | Full validation design/execution, unsupported-claim cleanup, earned/non-earned claim table, post-Epic-11 residual queue. |

## Dependency Order

1. Sprint 118 establishes baseline validation, platform truth, residual intake,
   product truth, hotspot metrics, templates, and claim-audit inputs.
2. Sprint 119 handles eigensolver source-boundary decisions first because later
   oracle, report, and claim work should not depend on unstable private-owner
   boundaries.
3. Sprint 120 uses Sprint 119 lessons to split direct/iterative proof owners
   and pilot shared oracle fixtures.
4. Sprint 121 builds on Sprint 120 fixture discipline for SVD, QR, rank,
   pseudoinverse, low-rank, and dense/external reference proof.
5. Sprint 122 consolidates corpus, coverage, and report architecture after the
   first oracle patterns are known.
6. Sprint 123 uses the report architecture to improve local performance and
   backend governance without portable speed claims.
7. Sprint 124 decides package/ABI product direction before platform install
   parity work so Sprint 125 does not validate the wrong package contract.
8. Sprint 125 applies package/platform decisions to Linux, macOS, and Windows
   install and staged-lane follow-through.
9. Sprint 126 simplifies adoption docs after package/platform and report truth
   are stable enough to present to users.
10. Sprint 127 performs final validation, claim recalibration, unsupported
    wording cleanup, residual publication, and Epic 11 closeout.

## Proof-Gate Checklist

| Work type | Required proof gate |
|---|---|
| Source movement | Exact old/new files, internal headers, source-list check, CMake build, focused consumer tests, CTest count evidence, rollback instructions. |
| Private owner extraction | Behavior proof before movement, compile-unit proof after movement, no hidden public API change. |
| Shared oracle fixture | Solver-specific tolerances and expected failures remain visible; at least one bounded pilot proves the design. |
| Giant-test split | CTest membership unchanged unless explicitly justified; before/after responsibility map; focused reruns plus required full quality if C changes. |
| Corpus taxonomy | Deterministic tags, fixture ownership, expected-failure classification, and trust-boundary docs. |
| Report index | Reviewed/supplemental/local classification, stale-report handling, reproducibility command, and no benchmark semantic drift unless validated. |
| Performance sentinel | Local machine/fixture interpretation, baseline source, tolerance/risk note, and explicit non-portable wording. |
| Package/ABI decision | Static-first deferral proof or shared-library build/install/symbol/version/loader/ABI proof. |
| Platform support change | Workflow-equivalent proof, expected CTest count impact, staged-exclusion update, and support-doc wording. |
| Adoption docs change | Link/path checks, claim-boundary scan, and alignment with current product truth map. |
| Public claim expansion | Implementation evidence, validation evidence, public wording cleanup, and explicit remaining non-claims. |

## Future-Epic Deferral Notes

These items are not unscheduled Sprint 118 blockers. They should remain
explicit non-claims unless an owner sprint implements and validates them:

| Candidate | Current disposition | Rationale |
|---|---|---|
| Package-manager support | Sprint 124 residual or future epic | No real recipes or manager-specific install/consumer proof exist yet. |
| Windows Makefile parity | Sprint 125 audit or future epic | Current Windows reviewed lane is CMake-first consumer subset only. |
| GPU support | Future epic/non-claim | No implementation, runtime, tests, docs, or package proof. |
| Distributed-memory support | Future epic/non-claim | No implementation, runtime, tests, docs, or package proof. |
| Broad ecosystem parity | Future epic/non-claim | No SuiteSparse/PETSc/Trilinos/ARPACK/LAPACK/SciPy-wide parity suite exists. |
| Broad complex or mixed-precision maturity | Future epic/non-claim | Current product truth is not a broad precision campaign. |
| Portable performance superiority | Future epic/non-claim | Current performance evidence is local/sentinel/report context only. |

## Sprint 119-127 Handoff Candidates

| Sprint | Handoff from Sprint 118 | First decision expected |
|---|---|---|
| 119 | Eigensolver residual owner list, source-list/CMake expectations, source-movement proof gates. | Which private owner is lowest risk to move, and which candidates must defer. |
| 120 | Direct/iterative oracle candidates, giant-test split candidates, evidence-template requirements. | Whether one shared generated-RHS/dense-reference pilot is safe. |
| 121 | SVD/QR helper candidates and oracle/template requirements. | Which helper extraction improves proof ownership without hiding evidence. |
| 122 | Corpus/report residuals and current coverage/report boundaries. | Which index/report artifact becomes the first recurring architecture proof. |
| 123 | Benchmark/performance non-claims and report-index expectations. | Which sentinels are deterministic enough for useful local regression proof. |
| 124 | Static-first package truth and ABI/package-manager residuals. | Add shared-library ABI support or explicitly continue static-first support. |
| 125 | Platform staged-exclusion register and support-tier boundaries. | Which Linux/macOS/Windows lane is promoted, strengthened, or deferred. |
| 126 | Optional scanability work, adoption candidates, claim-boundary rules. | Whether algorithm doc split and cookbook updates are first-phase or full batch. |
| 127 | Non-claim register and owner-sprint residual queue expectations. | Which Epic 11 claims are earned, downgraded, or carried forward. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Residual owner table is complete. | Complete. |
| Dependency order is documented. | Complete. |
| Proof-gate checklist is documented. | Complete. |
| Future-epic deferral notes are explicit. | Complete. |
| Sprint 119-127 handoff candidates are visible. | Complete. |
| Residual work is assigned or explicitly deferred. | Complete. |
| No residual depends on work scheduled after it without a documented prerequisite. | Complete. |
