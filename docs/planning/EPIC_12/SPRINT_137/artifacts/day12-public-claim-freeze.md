# Sprint 137 Day 12 - Public Claim Freeze

## Purpose

Day 12 audits the current public and maintainer-facing wording before Epic 12
implementation sprints begin. The goal is to freeze current claim boundaries,
separate unsupported cleanup from future candidate claims, and keep
state-of-the-art wording blocked unless later evidence earns it.

This artifact does not change public docs. It records the current claim state
and the gates later sprints must pass before widening README, INSTALL,
solver-selection, cookbook, tutorial, algorithm, benchmark, example, public
header, or maintainer-guide wording.

## Audited Surfaces

| Surface | Role in claim freeze |
| --- | --- |
| `README.md` | Public front door for current capabilities, workflow selection, CI support tiers, benchmark interpretation, package summary, coverage, and quality commands. |
| `INSTALL.md` | Authoritative install, downstream consumer, static-first package, CMake/pkg-config, platform support, coverage, and unsupported shared/ABI/package-manager wording. |
| `docs/solver_selection.md` | Solver workflow guidance and performance/state-of-the-art non-claims. |
| `docs/cookbook.md` | Adoption workflows and local sentinel command references. |
| `docs/tutorial.md` | First-use and repeated-run workflow documentation. |
| `docs/matrix_market.md` | Matrix Market format, SuiteSparse acquisition, and unsupported format behavior. |
| `docs/algorithm.md` | Current algorithm reference with warnings that it is not an adoption, package, ABI, or portable performance surface. |
| `docs/algorithm_history.md` | Historical measurement and report notes with non-claim boundaries. |
| `benchmarks/README.md` | Benchmark/report interpretation, local performance boundaries, guardrail report meanings, and generated-report non-claims. |
| `docs/maintainer_guide.md` | Maintainer authority for reviewed baselines, platform tiers, package/ABI contract, solver-family evidence, report meaning, coverage, and dead-code interpretation. |
| `examples/` | Adoption examples; no Day 12 source changes were made. |
| `include/` public headers | Public API comments can imply ABI, solver behavior, callback, or support claims; no Day 12 header changes were made. |

## Public Claim Inventory

| Claim family | Current public wording state | Freeze decision |
| --- | --- | --- |
| State-of-the-art status | Public docs avoid unqualified state-of-the-art status and frame current work as bounded evidence. | Frozen: no state-of-the-art claim unless Sprint 146 evidence inventory explicitly earns it. |
| External parity | Solver docs and maintainer guide use bounded fixture and helper evidence while blocking broad LAPACK, NumPy, SciPy, PETSc, Trilinos, ARPACK, SuiteSparse, and ecosystem parity. | Frozen: later solver docs may widen only for selected fixture-local QR/SVD evidence. |
| Static-first package contract | README, INSTALL, and maintainer guide state that install/export is real, maintained, and static-first. | Frozen until Sprint 143 strengthens or revises the static-first package decision. |
| Shared-library support | INSTALL and maintainer guide say `BUILD_SHARED_LIBS=ON` is intentionally rejected and shared-library support is deferred. | Frozen: no shared-library support wording in Epic 12 unless the selected path changes by approved decision. |
| Dynamic ABI compatibility | INSTALL and maintainer guide explicitly avoid dynamic ABI promises. | Frozen: no ABI stability or loader compatibility claim without shared-library ABI product work. |
| Package-manager support | Current docs do not claim package-manager recipes or distribution. | Frozen: remain unsupported/deferred in Epic 12. |
| Platform support tiers | README, INSTALL, and maintainer guide identify Linux as strongest reviewed source of truth, macOS Apple Clang as reviewed source/build with supplemental package lanes, and Windows as reviewed CMake subset with supplemental CMake install/downstream confidence. | Frozen until Sprint 144 attempts exactly one Windows CMake install/downstream promotion lane. |
| Windows staged tests | INSTALL and maintainer guide keep pthread/POSIX-backed staged tests outside the reviewed Windows subset. | Frozen: no staged test promotion without source portability, CTest count update, hosted proof, and docs update. |
| Runtime/backend behavior | README and maintainer guide describe selected backend/runtime behavior and observability while avoiding backend parity and portable speedup claims. | Frozen until Sprint 142 defines runtime/backend precedence and one sentinel lane. |
| Performance and benchmarks | README, solver-selection, algorithm, algorithm-history, and benchmarks docs state benchmark/sentinel rows are local measurement context, not portable performance guarantees. | Frozen: no portable timing, speedup, memory, scalability, or backend superiority claim without new cross-platform evidence. |
| Generated reports | README, benchmarks docs, algorithm-history, and maintainer guide treat generated reports as freshness, traceability, row interpretation, and local evidence. | Frozen until Sprint 141 adds normalized report/freshness gates. |
| Coverage | README, INSTALL, benchmarks docs, and maintainer guide treat coverage as supplemental and tree-mutating. | Frozen: no coverage-completeness claim. |
| Dead-code | README and maintainer guide treat dead-code output as report completeness and triage evidence, not removal-ready proof. | Frozen: no removal-ready wording without owner/API review. |
| Corpus/oracle evidence | Current public docs describe existing bounded fixtures and SuiteSparse usage, but do not claim maintained corpus completeness. | Frozen until Sprint 138 implements corpus/oracle contract. |
| QR behavior | Maintainer guide lists bounded QR evidence and extensive QR non-claims. | Frozen until Sprint 139 closes the selected rank-deficient nullspace/subspace residual. |
| Partial-SVD behavior | Maintainer guide lists bounded SVD/partial-SVD evidence and SVD non-claims. | Frozen until Sprint 140 closes the selected repeated/clustered-spectrum residual. |
| Adoption surface | README, cookbook, tutorial, and solver-selection provide the current adoption path. | Frozen until Sprint 145 rewrites from evidence earned in Sprints 138-144. |

## Frozen-Claim Register

| Frozen claim | Current allowed wording | Required evidence before widening |
| --- | --- | --- |
| Linux remains strongest reviewed source of truth. | Linux carries reviewed Makefile quality, reviewed CMake parity, dead-code, and reviewed static-first package contract. | Hosted Linux CI plus updated report/package evidence if scope changes. |
| macOS install/export remains supplemental. | macOS has reviewed Apple Clang source/build plus supplemental package confidence. | Sprint 144 or later platform promotion with hosted proof and docs alignment. |
| Windows remains reviewed CMake subset plus supplemental install/downstream. | Windows supports the maintained CMake-first consumer story but not separate reviewed install-validation parity. | Sprint 144 Windows CMake install/downstream lane proof, expected counts, support-tier docs, and fallback semantics. |
| Static-first is maintained. | Make/CMake/pkg-config install/export describe the installed static archive. | Sprint 143 package/ABI decision and proof before changing package contract wording. |
| Shared libraries are unsupported. | Shared-library requests are rejected/deferred and not silently treated as supported. | Shared build rules, artifact naming, symbol/export policy, loader behavior, downstream tests, platform proof, and docs. |
| Dynamic ABI is unsupported. | Version metadata is package metadata, not ABI stability. | ABI epoch, symbol inventory, layout policy, compatibility tests, loader policy, platform proof, and docs. |
| Package-manager support is unsupported. | No package-manager recipe support is claimed. | Manager recipes, dependency metadata, install roots, upgrade/uninstall proof, and downstream tests. |
| Benchmark evidence is local. | Benchmark/sentinel rows are local measurement or bounded sentinel evidence. | Reproducible cross-platform benchmark design, variance policy, thresholds, and report freshness before portable claims. |
| Generated reports are not release proof. | Reports provide row interpretation, freshness, artifacts, and traceability. | Release criteria, maintained gates, and explicit release proof process. |
| Coverage is supplemental. | Coverage is a line-coverage signal, not behavioral completeness. | Separate behavioral completeness model and reviewed quality decision. |
| Dead-code output is triage evidence. | Dead-code report generation/completeness is enforced, but findings need review. | Owner/API review and removal decision for specific symbols. |
| External solver parity is blocked. | Current evidence is fixture-local and family-specific. | External corpus/oracle design, reproducible comparisons, tolerance policy, support tier, and docs. |
| State-of-the-art status is blocked. | Current docs may discuss state-of-the-art gaps but must not claim unqualified status. | Sprint 146 final evidence inventory must prove implementation, external comparison, reproducibility, packaging, platform, and documentation claims. |

## Unsupported Wording Cleanup List

No immediate public-doc cleanup was identified during the Day 12 audit. The
current live surfaces already contain explicit boundaries for the high-risk
claim families:

- static-first package support;
- shared-library and dynamic ABI deferral;
- Linux, macOS, and Windows tier differences;
- Windows staged pthread/POSIX tests;
- local benchmark and sentinel interpretation;
- generated-report limits;
- supplemental coverage;
- dead-code triage meaning;
- fixture-local solver evidence;
- broad external parity and state-of-the-art non-claims.

Future implementation sprints must still rerun a claim-boundary scan whenever
they edit public documentation, public headers, examples, package metadata,
workflows, report schemas, or generated report interpretation.

## Non-Claim Register

| Non-claim | Applies to | Gate before promotion |
| --- | --- | --- |
| Unqualified state-of-the-art sparse linear algebra library | Whole project | Sprint 146 final evidence inventory plus external comparison, reproducibility, package/platform support, and docs alignment. |
| Broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, ARPACK, GraphBLAS, oneMKL, or vendor parity | Solver docs, benchmarks, maintainer guide | Maintained corpus/oracle rows, external comparison policy, tolerance policy, support tiers, and generated reports. |
| Broad QR correctness for all rank-deficient, nullspace, minimum-norm, economy, reorder, sparse-mode, or SuiteSparse cases | QR docs and maintainer guide | Sprint 139 selected residual closure plus explicit fixture-local boundaries. |
| Broad partial-SVD vector/subspace, ordering, convergence, sparse-output, drop-tolerance, or platform parity | SVD docs and maintainer guide | Sprint 140 selected residual closure plus comparison semantics and convergence-budget evidence. |
| Portable speedup, backend parity, scalability, or memory superiority | README, solver-selection, algorithm, benchmarks | Sprint 142 runtime/backend governance plus cross-platform benchmark proof if ever selected. |
| Report index as release, broad correctness, coverage, or performance proof | README, benchmarks, algorithm-history, maintainer guide | Sprint 141 report normalization with explicit row meanings and non-claims. |
| Coverage percentage as behavioral completeness | README, INSTALL, maintainer guide, CI | Separate coverage architecture and behavioral proof decision. |
| Dead-code findings as removal-ready proof | README, maintainer guide, dead-code reports | Owner/API review and focused removal validation. |
| Shared-library packaging | README, INSTALL, CMake, maintainer guide | Shared build/install/export metadata, downstream proof, loader behavior, platform proof, and docs. |
| Dynamic ABI compatibility | README, INSTALL, public headers, CMake, maintainer guide | ABI epoch, symbol inventory, public layout policy, compatibility tests, loader proof, and docs. |
| Package-manager support | README, INSTALL, maintainer guide | Package-manager recipes, dependency metadata, install roots, upgrade/uninstall proof, and downstream tests. |
| macOS reviewed install/export parity | INSTALL, README, workflows, maintainer guide | Hosted macOS package promotion lane with failure semantics and support-tier docs. |
| Windows general parity or reviewed install-validation parity | INSTALL, README, workflows, maintainer guide | Sprint 144 selected Windows CMake install/downstream lane or later platform promotion proof. |
| Windows POSIX/pthread staged test support | INSTALL, workflows, maintainer guide | Windows-native portability changes, expected CTest count update, hosted proof, and docs. |

## Claim Gate for Later Public Wording

Any later public wording change must record:

1. The exact wording being added or changed.
2. The evidence rows or validation commands that earned it.
3. The support tier and platform boundary.
4. The documentation surfaces updated together.
5. The non-claims that remain blocked.
6. The owner and rollback or demotion condition.

If any of those fields are missing, the wording remains frozen.

## Day 12 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Current public wording is reconciled before implementation sprints begin. | Complete | Audited surface table, public claim inventory, and frozen-claim register record current package, platform, solver, report, benchmark, coverage, dead-code, and state-of-the-art boundaries. |
| Unsupported claim cleanup is separated from future candidate claims. | Complete | Cleanup list records no immediate public-doc cleanup, while future promotions remain gated by the non-claim register and claim gate. |
| State-of-the-art wording remains blocked unless proof exists. | Complete | Frozen-claim and non-claim registers keep unqualified state-of-the-art status blocked until Sprint 146 final evidence independently earns it. |
