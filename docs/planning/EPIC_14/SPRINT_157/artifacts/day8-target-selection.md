# Day 8 Epic 14 Target Selection

## Scope

Day 8 selects the Epic 14 targets that can close completely inside Sprints
158-166. The selection uses the Day 7 residual register, the Epic 14 review,
and the project plan. It rejects broad goals that would require a larger
product, release, platform, or ecosystem program.

## Selection Criteria

| Criterion | Selection rule |
| --- | --- |
| User value | Prefer work that makes the project easier to adopt, review, install, or trust. |
| Proof cost | Prefer targets with a concrete command, artifact, hosted lane, or binary product decision. |
| Runtime cost | Prefer bounded checks that can run locally or in CI without destabilizing the normal quality path. |
| Risk | Prefer declaration-preserving, fixture-local, static-first, or methodology-bound work over broad rewrites. |
| Claim impact | Prefer targets that convert current residuals into evidence-backed claims or explicit retained non-claims. |

## Selected Epic 14 Target Register

| Target ID | Selected target | Source residual | Sprint | Closure form | Evidence owner | Claim impact |
| --- | --- | --- | --- | --- | --- | --- |
| T157-01 | Generated API reference publication decision | E14-R01 | 158 | Binary publication decision plus docs/warning/page-coverage evidence. | Documentation/API owner | Closes the generated API HTML residual or records a guarded local-only policy. |
| T157-02 | Hosted selected generated oracle/comparison freshness | E14-R02 | 159 | Reviewed hosted lane plus artifact upload or deterministic summary for selected families. | CI, corpus, comparison, and report owners | Converts selected local-only generated evidence into hosted review evidence without widening solver claims. |
| T157-03 | One bounded QR comparison family | E14-R03 | 160 | One fixture-family comparison expansion with metric contract, normalized rows, freshness, and docs. | QR and comparison owners | Widens comparison breadth narrowly while rejecting broad QR/ecosystem parity. |
| T157-04 | One bounded partial-SVD comparison family | E14-R04 | 161 | One subspace-safe comparison publication with selected rows, freshness, and docs. | SVD and comparison owners | Adds first partial-SVD comparison publication without raw-vector or broad SVD parity claims. |
| T157-05 | Windows package parity decision | E14-R05 | 162 | Product decision: implement selected proof or strengthen explicit retained non-claim. | Platform/package owner | Removes ambiguity around Windows Makefile and Windows `pkg-config` parity. |
| T157-06 | Methodology-bound performance publication | E14-R06 | 163 | Bounded report artifact with methodology fields, row classification, and non-superiority wording. | Benchmark and report owners | Improves performance-report usefulness while rejecting portable superiority. |
| T157-07 | Public header/API coherence batch | E14-R07 | 164 | Finite declaration-preserving header cleanup plus cross-link alignment and generated-doc policy application. | Header owners | Improves usability without changing API signatures unless explicitly reviewed. |
| T157-08 | Static-first package boundary hardening | E14-R08 | 165 | Metadata/docs/guard hardening plus refreshed static package proof. | Package/ABI owner | Strengthens static-first support and keeps shared-library/dynamic ABI deferrals explicit. |
| T157-09 | Final claim recalibration and residual publication | E14-R13 plus all selected targets | 166 | Final evidence inventory, validation, public claim audit, retrospective, and residual queue. | Epic/product owner | Ensures earned claims map to recurring evidence and unsupported broad claims remain rejected. |

## Target Scores

Scores are relative planning values from 1 to 5. Higher user value and claim
impact are better. Lower proof/runtime/risk scores indicate cheaper or lower
risk work.

| Target | User value | Proof cost | Runtime cost | Risk | Claim impact | Day 8 decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T157-01 generated API reference | 5 | 3 | 2 | 2 | 4 | Select. |
| T157-02 hosted generated evidence | 5 | 4 | 3 | 3 | 5 | Select with strict family scope. |
| T157-03 QR comparison family | 4 | 4 | 3 | 3 | 4 | Select one family only. |
| T157-04 partial-SVD comparison family | 4 | 4 | 3 | 4 | 4 | Select one subspace-safe family only. |
| T157-05 Windows package decision | 4 | 3 | 3 | 3 | 4 | Select as decision closure, not guaranteed promotion. |
| T157-06 performance publication | 4 | 3 | 4 | 3 | 3 | Select methodology-bound report only. |
| T157-07 header/API cleanup | 4 | 3 | 2 | 3 | 3 | Select finite declaration-preserving batch. |
| T157-08 static-first boundary hardening | 4 | 3 | 3 | 2 | 4 | Select static-first hardening, not shared-library support. |
| T157-09 final claim recalibration | 5 | 3 | 2 | 2 | 5 | Select as required closeout. |

## Explicit Non-Goal Register

| Non-goal | Related residual | Reason rejected for Epic 14 | Required future promotion gate |
| --- | --- | --- | --- |
| Unqualified state-of-the-art sparse linear algebra claim | E14-R13 | Current evidence remains bounded by selected fixtures, local/hosted lanes, static-first packaging, and narrow comparisons. | Broad recurring evidence across correctness, external parity, performance, platform, package, and ABI dimensions. |
| Broad external ecosystem parity | E14-R09 | Epic 14 selects one QR and one partial-SVD comparison family, not LAPACK/SuiteSparse/Eigen/PETSc/Trilinos parity. | Dependency/provenance policy, multiple selected families, recurring hosted comparison evidence, and public claim audit. |
| Package-manager distribution | E14-R10 | No release-channel owners, recipes, update/uninstall policy, or package-manager CI validation are selected. | Product release policy, recipe ownership, channel validation, install/uninstall checks, and support docs. |
| Full shared-library product support | E14-R11 | Shared-library build/install, symbol visibility, loader metadata, and shared consumer validation exceed Epic 14 scope. | Cross-platform shared-library design, symbol allowlist, loader metadata, installed shared consumers, and CI proof. |
| Dynamic ABI compatibility promise | E14-R12 | Public structs and exported declarations lack a reviewed ABI stability policy. | ABI policy, compatibility window, binary compatibility fixtures, versioning rules, and release checks. |
| Portable performance superiority | E14-R06/E14-R13 | Sprint 163 can publish methodology-bound local/selected evidence, not cross-machine superiority. | Recurring benchmark matrix, variance policy, competitive baselines, threshold policy, and hosted publication. |
| Broad Windows Makefile parity | E14-R05 | Windows support is CMake-first; Sprint 162 may choose to retain this non-claim. | Explicit product decision, Windows Makefile implementation, install/uninstall behavior, and hosted validation. |
| Windows `pkg-config` execution parity unless selected in Sprint 162 | E14-R05 | Current Windows lane checks `sparse.pc` metadata but does not execute `pkg-config`. | Selected provider, shell/path/link policy, hosted compile/link/run proof, and synchronized docs. |
| Runtime/backend API promotion | E14-R14 | No typed runtime/backend API or ABI scope is selected. | API design, ABI review, metrics, tests, docs, and support-tier policy. |
| Generated coverage/dead-code/advisory rows as pass evidence | E14-R16 | These rows remain advisory or supplemental unless explicitly selected. | Row-level selection, freshness semantics, hosted proof or publication policy, and docs update. |

## Target-To-Sprint Map

| Sprint | Selected target | Primary artifacts expected | Required decision shape |
| --- | --- | --- | --- |
| 158 | T157-01 generated API reference publication | Doxygen baseline, warning triage, page coverage check, publication decision, docs alignment, closeout handoff. | Commit generated HTML, publish via another explicit route, or retain local-only policy with guard. |
| 159 | T157-02 hosted selected generated evidence | Family selection, runtime budget, CI lane, artifact policy, normalizer semantics, docs alignment. | Selected hosted freshness lane passes or selected scope is explicitly narrowed. |
| 160 | T157-03 QR comparison family | Fixture selection, metric contract, harness changes, focused tests, normalized report rows, docs. | One QR comparison family passes freshness and stays fixture-local. |
| 161 | T157-04 partial-SVD comparison family | Subspace-safe fixture selection, metric contract, harness/report rows, focused tests, docs. | One partial-SVD comparison family passes freshness and rejects broad SVD parity. |
| 162 | T157-05 Windows package parity decision | Windows package audit, product decision, selected proof or rejection guard, CI/docs alignment. | Implement selected proof or retain non-claim with stronger evidence and wording. |
| 163 | T157-06 methodology-bound performance publication | Surface selection, methodology contract, report enhancements, gate classification, docs, validation. | Bounded report artifact exists and portable superiority remains rejected. |
| 164 | T157-07 public header/API coherence batch | Header selection, declaration baseline, comment/cross-link cleanup, declaration-preservation proof, validation. | Zero signature drift or explicit API review. |
| 165 | T157-08 static-first package boundary hardening | Package metadata audit, deferral guard hardening, ABI non-claim audit, downstream proof refresh, docs. | Static-first package boundary is stronger; shared-library/dynamic ABI residuals stay explicit. |
| 166 | T157-09 final claim recalibration | Final evidence inventory, validation baseline, hosted CI reconciliation, claim audit, plan reconciliation, retrospective, residual queue. | Every earned claim maps to evidence; every remaining unsupported claim is residualized or rejected. |

## Coherence Rules For Selected Work

- Generated local rows do not become public pass evidence until a selected
  hosted or publication gate promotes them.
- Comparison work remains fixture-local and metric-bound.
- Package work remains static-first unless a future explicit product decision
  funds shared-library, dynamic ABI, or package-manager support.
- Windows package work must separate CMake install/downstream support from
  Makefile and `pkg-config` parity.
- Header cleanup must preserve declarations unless an API change is explicitly
  reviewed.
- Performance publication must state hardware, compiler, build mode, thread
  count, fixture, repeat policy, thresholds, and caveats before any row is
  interpreted.
- Final closeout must reject state-of-the-art wording unless every dimension is
  backed by recurring evidence.

## Day 9 Inputs

Day 9 should create evidence templates for the selected target families:

- API documentation publication;
- hosted generated report freshness;
- QR comparison rows;
- partial-SVD comparison rows;
- Windows package parity decision;
- methodology-bound performance publication;
- declaration-preserving header cleanup;
- static-first package boundary hardening;
- final claim audit and residual publication.

## Completion Check

- Every selected target can end with a binary proof, artifact, or product
  decision.
- Long-horizon work is captured in the explicit non-goal register.
- Sprints 158-166 each have one coherent target lane.
- The target register does not create broad state-of-the-art, ecosystem parity,
  package-manager, shared-library, dynamic ABI, broad Windows, or portable
  performance claims.
