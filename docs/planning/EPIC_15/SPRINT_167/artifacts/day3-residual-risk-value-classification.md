# Sprint 167 Day 3: Residual Risk And Value Classification

## Purpose

Day 3 ranks the residual IDs from Day 2 by claim risk, user value, closure
feasibility, dependencies, and recommended Epic 15 handling. The output is a
ranked candidate queue for Day 11 gap selection and a dependency map for Days
4-10 evidence inventory work.

## Classification Scale

| Dimension | Values | Meaning |
| --- | --- | --- |
| Claim risk | Critical, High, Medium, Low | Risk that unclear wording or missing evidence would cause an unsupported public claim. |
| User value | High, Medium, Low | Expected value to downstream users, maintainers, reviewers, or adopters if the residual is closed. |
| Closure feasibility | High, Medium, Low, Non-closeable in Epic 15 | Likelihood that the residual can be completely closed or explicitly resolved within Epic 15. |
| Recommended handling | Close, Decide, Promote, Guard, Defer, Non-claim | Day 3 recommendation for the Day 11 selection gate. |

## Residual Classification Table

| Residual ID | Candidate gap | Claim risk | User value | Closure feasibility | Dependencies | Recommended handling |
| --- | --- | --- | --- | --- | --- | --- |
| R167-01 | PR-hosted CI confirmation | Medium | Medium | High | Access to merged PR #184 hosted results or current master workflow evidence | Guard as evidence-confirmation work, not a standalone Epic 15 product gap. |
| R167-02 | Hosted performance publication decision | High | High | Medium | Benchmark/report inventory, CI lane design, methodology metadata, report freshness | Close or decide early because performance language is a major state-of-the-art risk. |
| R167-03 | Shared-library ABI product design | High | High | Medium | Package inventory, public header surface audit, build-system behavior, package docs | Decide explicitly; either retain static-first with stronger guardrails or select a staged ABI track. |
| R167-04 | Package-manager distribution readiness | High | High | Medium | ABI/package decision, install proof, provider scope, versioning/provenance docs | Decide and close one provider proof or formal deferral after R167-03. |
| R167-05 | Broader public-header cleanup batch | Medium | High | High | Source/header inventory, generated API docs policy, declaration-preservation checks | Close with one selected high-impact header family. |
| R167-06 | Additional bounded comparison family | High | High | Medium | Test/corpus inventory, comparator availability, report normalizer, freshness checks | Close one selected family; keep solver and fixture scope explicit. |
| R167-07 | Broad generated-report platform parity | Medium | Medium | Medium | CI inventory, report-family inventory, platform shell/path behavior | Promote one selected report-family platform lane or explicitly defer broad parity. |
| R167-08 | Hosted generated API HTML publication | Medium | Medium | Medium | Header cleanup status, docs generator behavior, artifact/site publication policy | Decide after API/header inventory; close by publication or reaffirm local-only policy. |
| R167-09 | Allocation/failure-path evidence | Medium | High | Medium | Source/header inventory, allocation-heavy subsystem selection, test harness design | Close for one selected subsystem late in Epic 15 after inventories identify the best target. |
| R167-10 | Broad state-of-the-art/external parity | Critical | High | Non-closeable in Epic 15 | Comprehensive competitive evidence, broad matrix corpus, performance methodology, package/ABI/platform maturity | Keep as explicit non-claim; do not select as a direct Epic 15 closure target. |

## Ranked Queue For Epic 15 Selection

| Rank | Residual ID | Candidate gap | Why it ranks here |
| ---: | --- | --- | --- |
| 1 | R167-02 | Hosted performance publication decision | Performance is one of the easiest surfaces to overclaim, and Epic 14 left it local-only. A hosted lane or explicit retained non-claim materially improves credibility. |
| 2 | R167-03 | Shared-library ABI product design | ABI and shared-library wording affects package, user adoption, and downstream integration. The project needs a clear product decision before package-manager work. |
| 3 | R167-04 | Package-manager distribution readiness | Package-manager support is adoption-critical but depends on the ABI/static-first decision. Closing one provider path or a formal deferral is valuable and bounded. |
| 4 | R167-05 | Broader public-header cleanup batch | Public headers are the API contract. One selected cleanup batch is high-value, feasible, and lowers generated-doc and adoption risk. |
| 5 | R167-06 | Additional bounded comparison family | One more complete comparison family improves numerical credibility without pretending to prove ecosystem parity. |
| 6 | R167-07 | Broad generated-report platform parity | Broad parity is too large, but promoting or explicitly deferring one selected report family can close a real cross-platform ambiguity. |
| 7 | R167-08 | Hosted generated API HTML publication | Useful for adoption and documentation, but less urgent than package/ABI/performance evidence because Epic 14 already closed ambiguity with a local-only policy. |
| 8 | R167-09 | Allocation/failure-path evidence | High engineering value, but best selected after source/header inventory identifies an allocation-heavy subsystem with bounded test scope. |
| 9 | R167-01 | PR-hosted CI confirmation | Important if citing PR-specific hosted evidence, but PR #184 has merged and this is operational confirmation rather than a product gap. |
| 10 | R167-10 | Broad state-of-the-art/external parity | Highest claim risk, but not realistically closeable in Epic 15. It should remain a guarded non-claim and final recalibration target. |

## Closeable Candidate Set

These residuals are closeable or decision-closeable within Epic 15:

| Candidate | Closure mode | Likely sprint owner from Epic 15 plan |
| --- | --- | --- |
| Hosted performance publication | Add a hosted lane or explicitly retain local-only status with stricter wording. | Sprint 168 and Sprint 169 |
| Shared-library ABI product design | Publish a product decision and enforce package/build/docs behavior. | Sprint 170 |
| Package-manager readiness | Prove one provider path or publish a formal deferral with claim guards. | Sprint 171 |
| Public-header cleanup batch | Clean one selected header family with declaration-preservation evidence. | Sprint 172 |
| Generated API HTML publication status | Publish hosted/committed/artifact-only docs or reaffirm local-only status with freshness checks. | Sprint 173 |
| Additional bounded comparison family | Add one fixture-local family with metrics, tolerances, generated rows, and freshness checks. | Sprint 174 |
| Cross-platform report freshness | Promote one selected report family or formally defer broad parity. | Sprint 175 |
| Allocation-failure evidence | Add deterministic failure-path proof for one selected subsystem. | Sprint 176 |

## Dependency Map

| Dependency | Downstream effect |
| --- | --- |
| Shared-library ABI product design before package-manager readiness | Package-manager support must know whether it packages static-first only or a shared ABI surface. |
| Package inventory before ABI decision | Static-first guards, CMake package metadata, `sparse.pc`, install tests, and docs define the current supported boundary. |
| Header inventory before generated API publication | Generated API publication should use cleaned and authoritative public-header inputs. |
| Test/corpus inventory before comparison-family selection | The next comparison family should be chosen from a solver/test area where fixture coverage and reference metrics can be bounded. |
| CI inventory before hosted performance or report promotion | Hosted lanes need clear runtime, artifact, platform, and support-tier boundaries. |
| Evidence ledger before public claim edits | Claim wording should not change until supported, partial, local-only, hosted-only, deferred, and unsupported rows have owners. |
| Source/header inventory before allocation-failure selection | Failure-path tests should target a subsystem with high allocation risk and bounded cleanup semantics. |

## Non-Claim Queue

These are not Day 3 closure targets. They should remain explicit non-claims
unless future evidence substantially changes:

| Non-claim | Reason |
| --- | --- |
| Unqualified state-of-the-art sparse linear algebra status | Requires broad competitive correctness, performance, package, ABI, platform, and failure-behavior evidence. |
| Broad external-library parity | Existing comparison evidence is fixture-local and selected-family-only. |
| Portable performance superiority | Hosted performance publication is not yet selected or proven, and portable superiority is broader than one hosted lane. |
| Broad package-manager ecosystem distribution | One provider proof would not prove ecosystem-wide package-manager support. |
| Broad dynamic ABI compatibility | Requires an ABI policy, exported-symbol controls, binary compatibility checks, and platform loader validation. |
| Broad cross-platform parity | Platform evidence remains tiered by workflow, generator, package, and command support. |

## Day 4 Handoff

Day 4 should inventory source and public-header surfaces with special attention
to:

- remaining high-risk public headers for R167-05;
- allocation-heavy implementation families for R167-09;
- source/header surfaces that affect ABI readiness for R167-03;
- implementation/test clusters that may affect comparison-family selection for
  R167-06;
- any `.c` or `.h` changes that would require
  `make format && make lint && make test` in later sprints.

## Validation Notes

Day 3 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every open residual has risk, value, and feasibility labels. | Complete | Residual classification table covers R167-01 through R167-10. |
| Dependency relationships are explicit. | Complete | Dependency map links ABI/package, header/API docs, comparison/report, CI/performance, and allocation-failure work. |
| Highest-value closeable gaps are visible. | Complete | Ranked queue and closeable candidate set identify the primary Epic 15 candidates for Day 11 selection. |
