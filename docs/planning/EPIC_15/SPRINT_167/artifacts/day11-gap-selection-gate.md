# Sprint 167 Day 11: Gap Selection Gate

## Purpose

Day 11 turns the reviewed evidence ledger into the finite Epic 15 gap list.
The selected gaps are either closeable implementation gaps or
decision-closeable product/documentation gaps. Broad claims remain deferred
unless the Epic 15 sprint sequence can close them with specific evidence,
validation commands, and claim-safe documentation.

## Selection Rules

| Rule | Effect |
| --- | --- |
| Prefer complete closure over broad partial progress. | Select bounded performance, package, API, comparison, report, and failure-path work with one clear owner each. |
| Keep evidence scopes narrow. | A hosted lane, selected fixture, local report, or static-first install proof cannot become a broad support claim. |
| Treat unsupported product surfaces as decisions. | Shared libraries, dynamic ABI, package-manager distribution, and generated API publication can be closed by explicit supported-path or retained-deferral decisions. |
| Preserve non-claims where Epic 15 cannot prove them. | State-of-the-art status, broad external parity, portable performance superiority, broad platform parity, and broad solver correctness are not selected as closure targets. |
| Assign exactly one future owner. | Each selected gap maps to Sprint 168 through Sprint 176 or stays explicitly deferred. |

## Selected Epic 15 Closure Targets

| Selection ID | Selected gap | Ledger rows | Future owner | Closure mode | Rationale |
| --- | --- | --- | --- | --- | --- |
| G167-01 | Hosted methodology-bound performance publication | E15-014, E15-017, NC-003 | Sprint 168 and Sprint 169 | Promote one selected performance report into hosted freshness-checked evidence, then harden methodology and indexing. | Performance wording is a high-risk credibility surface. A single hosted, method-bound lane is complete and useful without implying portable superiority. |
| G167-02 | Shared-library ABI product decision | E15-006, E15-007, E15-005, NC-004, NC-005 | Sprint 170 | Decide static-first continuation or staged shared-library track; enforce docs/build/package guards. | ABI ambiguity blocks package-manager readiness and can easily drift into unsupported binary-compatibility claims. |
| G167-03 | Package-manager readiness or formal deferral | E15-008, E15-005, NC-006 | Sprint 171 | Prove one provider path or publish an enforceable deferral. | Source install is already supported, but users need clear guidance on whether provider packaging exists. |
| G167-04 | Public-header coherence batch | E15-010, E15-009 | Sprint 172 | Normalize one high-impact public header family and add lightweight drift guardrails. | Public headers are the adoption-facing API contract and improve the inputs for any generated API publication decision. |
| G167-05 | Generated API HTML publication status | E15-009, NC-009 | Sprint 173 | Implement hosted/committed/artifact-only publication or reaffirm local-only status with freshness checks. | Current generated API HTML is local-only; adoption docs need one explicit supported answer. |
| G167-06 | Additional bounded external comparison family | E15-012, E15-013, E15-018, NC-002 | Sprint 174 | Add one solver-family comparison with fixtures, tolerances, report rows, and freshness checks. | One more complete comparison family improves numerical credibility while keeping broad ecosystem parity unsupported. |
| G167-07 | Cross-platform report freshness promotion or deferral | E15-002, E15-003, E15-004, E15-015, NC-007, NC-010 | Sprint 175 | Promote one selected report freshness path beyond Linux or formally close the deferral with exact blockers. | Report freshness currently has selected Linux coverage; broad platform/report parity remains too broad, but one platform decision is closeable. |
| G167-08 | Deterministic allocation-failure evidence for one subsystem | E15-016, NC-011 | Sprint 176 | Select one allocation-heavy subsystem and add deterministic failure-path proof plus cleanup invariants. | Functional tests do not prove allocation failure behavior. One selected subsystem is high-value and bounded. |
| G167-09 | Final claim recalibration and Epic closeout | E15-001 through E15-018, NC-001 through NC-012 | Sprint 176 | Reconcile README, docs indexes, evidence ledger, non-claims, and retrospective against completed work. | The final sprint must prevent selected evidence from expanding into broad state-of-the-art, parity, package, ABI, or platform claims. |

## Deferred Or Retained Non-Claim Residuals

| Deferred residual | Related ledger rows | Why deferred | Required future evidence before support claim |
| --- | --- | --- | --- |
| Unqualified state-of-the-art sparse linear algebra status | E15-017, NC-001 | Non-closeable in Epic 15. It would require broad competitive correctness, performance, package, ABI, platform, documentation, and reliability evidence. | Multi-family external comparisons, robust performance methodology across platforms, ABI/package maturity, broader corpus coverage, and release-quality adoption proof. |
| Broad external-library ecosystem parity | E15-018, NC-002 | Epic 15 can add one bounded family, not parity with SuiteSparse, Eigen, SciPy, PETSc, Trilinos, LAPACK, or vendor libraries. | Systematic matrix-family corpus, comparator-specific policies, broad solver-family tolerances, and maintained cross-platform comparison reports. |
| Portable performance superiority | E15-014, E15-017, NC-003 | A hosted performance lane can publish scoped evidence but cannot prove superiority across hardware, compilers, backends, or matrix families. | Repeated cross-platform benchmark campaigns, external baselines, variance controls, hardware disclosure, and release-bound methodology. |
| Broad platform parity | E15-002, E15-003, E15-004, NC-007 | Current platform support is tiered by workflow and command surface. Broad parity would overstate Windows and generated-report/package parity. | Equivalent Make/CMake/package/report/test surfaces across Linux, macOS, and Windows, with hosted evidence for each. |
| Broad package-manager ecosystem distribution | E15-008, NC-006 | Sprint 171 can select one provider or formal deferral; ecosystem-wide distribution is larger than one epic. | Provider recipes, installation tests, version/provenance policy, update workflow, and support policy per provider. |
| Broad dynamic ABI stability | E15-006, E15-007, NC-005 | Sprint 170 can make a product decision, but binary compatibility guarantees require a larger ABI program. | Exported-symbol controls, ABI checker, compatibility policy, release cadence, loader behavior, and platform matrix. |
| Broad all-family generated-report freshness | E15-015, NC-010 | Sprint 175 can promote one path or formalize a deferral. All-family freshness would be too wide for this epic. | Complete report-family matrix, hosted freshness lanes for each supported family, and generated-output publication policy. |
| Broad solver correctness beyond maintained fixtures | E15-011, NC-012 | Existing tests are substantial but remain fixture, tolerance, and family scoped. | Larger corpus strategy, independent references, randomized property tests with oracle policy, and failure-mode coverage by solver family. |
| Broad allocation-failure guarantee across all solvers | E15-016, NC-011 | Sprint 176 can close one selected subsystem, not all allocation-heavy paths. | Shared failure-injection harness, subsystem-by-subsystem cleanup invariants, and broad CI integration. |
| Windows Makefile and Windows `pkg-config` execution parity | E15-004, NC-008 | Current Windows support is CMake-first. This is not required for the selected Epic 15 package-manager path unless Sprint 171 explicitly chooses it. | Windows shell/tooling decision, pkg-config availability policy, Makefile portability work, and hosted validation. |

## Sprint Ownership Map

| Sprint | Selected gap IDs | Expected Day 11 output consumed |
| --- | --- | --- |
| Sprint 168 | G167-01 | Pick one performance family/platform/command and establish hosted publication without superiority wording. |
| Sprint 169 | G167-01 | Harden the selected performance publication with methodology policy, stable schema, sentinel, and caveats. |
| Sprint 170 | G167-02 | Close the shared-library and ABI question with a decision record and enforcement guards. |
| Sprint 171 | G167-03 | Close package-manager readiness by one provider proof or formal deferral. |
| Sprint 172 | G167-04 | Select and clean one public header family with guardrails. |
| Sprint 173 | G167-05 | Decide generated API HTML publication and implement matching freshness/navigation behavior. |
| Sprint 174 | G167-06 | Select and add one bounded external comparison family. |
| Sprint 175 | G167-07 | Promote one report freshness path beyond Linux or formalize the deferral with blockers. |
| Sprint 176 | G167-08, G167-09 | Add one deterministic allocation-failure proof and recalibrate final claims/non-claims. |

## Selection Dependencies

| Dependency | Required ordering |
| --- | --- |
| Performance selection before methodology hardening | Sprint 168 must pick and host one scope before Sprint 169 can harden schema, policy, and sentinels. |
| ABI decision before package-manager readiness | Sprint 170 must settle static-first/shared posture before Sprint 171 chooses provider support or deferral language. |
| Header cleanup before generated API publication | Sprint 172 improves API source inputs before Sprint 173 chooses publication or local-only enforcement. |
| Corpus/report evidence before comparison expansion | Sprint 174 must use maintained corpus/report conventions identified in Sprint 167. |
| Platform matrix before report freshness promotion | Sprint 175 must avoid treating one promoted lane as broad platform parity. |
| Final claim recalibration after all selected gaps | Sprint 176 should update claims only after Sprints 168-175 evidence is known. |

## Day 12 Handoff

Day 12 should convert G167-01 through G167-09 into acceptance criteria, concrete
validation commands, and stop conditions. The acceptance criteria should be
written so future sprints can prove completion without widening selected
evidence into retained non-claims.

## Validation Notes

Day 11 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Epic 15 closure targets are explicit and finite. | Complete | G167-01 through G167-09 define the selected gap set. |
| Each selected gap has a future sprint owner. | Complete | Ownership map assigns Sprints 168 through 176. |
| Deferred gaps are documented rather than hidden. | Complete | Deferred residual table records retained non-claims and required future evidence. |
