# Day 11 Published Residual Queue

## Scope

Day 11 publishes the final Sprint 146 residual queue as future planning input.
The queue is based on the Day 10 design and cross-links each residual to the
evidence inventory, validation log, CI intake, platform reconciliation, and
claim audits from Days 2-9.

Residuals are not promises. Each item remains a non-claim until its promotion
gate passes.

## Published Residual Queue

| ID | Priority | Residual | Owner | Status | Promotion Gate |
| --- | ---: | --- | --- | --- | --- |
| R1 | 1 | Branch-specific hosted CI reconciliation for Sprint 146 | CI maintainer | open | Hosted Linux, macOS, and Windows workflow run IDs, conclusions, job names, commit SHA, branch/PR context, and support-tier implications are recorded. |
| R2 | 2 | Windows staged test portability closure | Platform maintainer | open | Windows hosted CMake lane intentionally registers and executes promoted `test_threads`, `test_sprint4_integration`, and/or `test_fuzz` coverage with updated expected count and docs/report rows. |
| R3 | 3 | Windows reviewed install-validation parity decision | Platform/package maintainer | open | Product decision explicitly promotes or rejects reviewed Windows install-validation parity, with workflow, report rows, docs, and hosted proof aligned. |
| R4 | 4 | Shared-library ABI productization | Package maintainer | open | Shared build/install/export works on Linux, macOS, and Windows with ABI policy, loader tests, symbol checks, package metadata, and docs. |
| R5 | 5 | Broad QR residual expansion | QR owner | open | Multiple reviewed QR fixture families have source-controlled metadata, expected rows, compiled proof owners, generated-local classification, validation commands, and bounded public wording. |
| R6 | 6 | Broad partial-SVD residual expansion | SVD owner | open | Multiple reviewed partial-SVD fixture families have source-controlled metadata, expected rows, compiled proof owners, subspace-safe comparisons, validation commands, and bounded public wording. |
| R7 | 7 | Generated benchmark, sentinel, coverage, dead-code, and guardrail refresh package | Report/benchmark maintainer | open | Selected generated families are regenerated and `normalize_report_index.py --require-generated <family> --check-freshness` passes without reclassifying local rows as hosted proof. |
| R8 | 8 | Tutorial alignment with first-use ladder | Documentation maintainer | open | Tutorial routes through the same build, first solve, data input, solver choice, diagnostics, and install/downstream ladder; public claim scan passes. |
| R9 | 9 | Broader public-header cleanup | API maintainer | open | Remaining public headers are updated without signature, typedef, enum, macro, or struct-field drift; declaration-preservation scan and `make format && make lint && make test` pass. |
| R10 | 10 | Runtime/backend typed-control promotion review | Runtime/backend owner | open | A selected control is promoted with typed API design, tests, docs, and ABI/package non-claim review, or remains explicitly non-API. |
| R11 | 11 | Additional runtime/backend sentinel rows | Runtime/backend and benchmark owners | open | New sentinel rows have maintained commands, row semantics, local-only/advisory or hard-gate classification, and no portable performance wording. |
| R12 | 12 | External-library parity study | Numerical lead | open | Comparative study names libraries, versions, fixtures, metrics, tolerances, platforms, and caveats before any parity language is promoted. |
| R13 | 13 | State-of-the-art competitive decision | Epic owner | open | Either a narrow evidence-backed claim is approved from direct comparative evidence, or state-of-the-art remains an explicit non-claim. |
| R14 | 14 | Package-manager distribution | Package maintainer | open | Package-manager recipe, CI install proof, versioning policy, support docs, and uninstall/update validation exist. |

## Next-Epic Handoff Candidates

| Candidate | Residuals Covered | Why It Should Be A Complete Gap Closure |
| --- | --- | --- |
| Windows platform closure epic | R2, R3 | Closing Windows staged tests and install-validation parity together would convert the largest platform support gap into explicit hosted proof rather than scattered workflow wording. |
| Shared-library and ABI productization epic | R4, R14 | Shared-library support and package-manager distribution need one coherent product contract: ABI policy, loader behavior, metadata, packaging, and cross-platform validation. |
| Numerical corpus expansion epic | R5, R6, R12 | Broad QR and partial-SVD claims require a larger maintained corpus plus external comparisons. Treating this as one evidence program avoids overclaiming isolated fixtures. |
| Report evidence refresh epic | R1, R7 | Branch-specific hosted evidence and generated report freshness need a repeatable evidence publication model before they are cited in claims. |
| Adoption/documentation completion epic | R8, R9 | Tutorial and remaining header cleanup should close the front-door documentation gap without changing product support language. |
| Runtime/backend governance follow-through epic | R10, R11 | Typed-control promotion and sentinel expansion should be handled together so API, ABI, report rows, and performance non-claims stay aligned. |
| Competitive positioning epic | R12, R13 | State-of-the-art status should only be revisited after direct comparative correctness, feature, package, platform, and performance evidence exists. |

## Residual-To-Non-Claim Map

| Residual | Non-Claim Until Gate Passes |
| --- | --- |
| R1 | No branch-specific hosted Sprint 146 CI pass. |
| R2 | No Windows staged pthread/POSIX test closure. |
| R3 | No reviewed Windows install-validation parity. |
| R4 | No shared-library support, dynamic ABI compatibility, runtime-loader compatibility, or static/shared selector support. |
| R5 | No broad QR correctness, global rank-threshold policy, broad rank-deficient solve, minimum-norm, reorder, SuiteSparse, platform, performance, or state-of-the-art claim. |
| R6 | No broad partial-SVD correctness, repeated-spectrum generality, rank-deficient null-space, sparse-output, convergence-rate, partial-result, platform, performance, or state-of-the-art claim. |
| R7 | No generated benchmark, sentinel, coverage, dead-code, guardrail, or report freshness claim from source-controlled rows alone. |
| R8 | No claim that the tutorial fully matches the new first-use ladder. |
| R9 | No claim that all public headers have received the Sprint 145 cleanup treatment. |
| R10 | No new typed runtime/backend API or ABI promise for maintainer-only controls. |
| R11 | No expanded runtime/backend sentinel coverage or portable performance proof. |
| R12 | No external-library parity claim against LAPACK, NumPy, SciPy, SuiteSparse, ARPACK, PETSc, Trilinos, or other ecosystems. |
| R13 | No state-of-the-art sparse linear algebra claim. |
| R14 | No package-manager distribution or package-manager support claim. |

## Cross-Reference Checklist

| Evidence Area | Source Artifact |
| --- | --- |
| Evidence families and closeout criteria | [day1-closeout-intake-evidence-map.md](./day1-closeout-intake-evidence-map.md) |
| Corpus, QR, and partial-SVD evidence | [day2-corpus-solver-evidence-inventory.md](./day2-corpus-solver-evidence-inventory.md) |
| Report, runtime/backend, package, platform, adoption evidence | [day3-support-evidence-inventory.md](./day3-support-evidence-inventory.md) |
| Final validation command design | [day4-final-validation-baseline-design.md](./day4-final-validation-baseline-design.md) |
| Local validation pass log | [day5-final-local-validation-command-log.md](./day5-final-local-validation-command-log.md) |
| CI lane inventory and hosted master baseline | [day6-ci-evidence-intake.md](./day6-ci-evidence-intake.md) |
| Platform support-tier reconciliation | [day7-cross-platform-reconciliation.md](./day7-cross-platform-reconciliation.md) |
| Public claim audit | [day8-public-claim-audit.md](./day8-public-claim-audit.md) |
| Support/maintainer claim audit | [day9-support-maintainer-claim-audit.md](./day9-support-maintainer-claim-audit.md) |
| Residual queue design | [day10-residual-queue-design.md](./day10-residual-queue-design.md) |

## Residual Validation Summary

The residual queue publication is documentation-only. It relies on already
completed Sprint 146 evidence:

- Day 5 local validation passed for corpus schema, report normalization,
  report freshness, package deferral, Make install, CMake install, examples,
  QR proof, partial-SVD proof, and local oracle/report refresh.
- Day 6 found no hosted `sprint-146` run; latest inspected `master` hosted
  baseline at `daac9a85` passed Linux, macOS, and Windows workflows.
- Day 7 found no platform support-tier wording mismatch.
- Day 8 found no public wording fix required.
- Day 9 found no support/maintainer wording fix required.
- Day 10 assigned residual owners, blockers, prerequisites, and promotion
  gates.

## Publication Notes

- R1 must be revisited after branch/PR CI exists.
- R2 and R3 should not be split into small wording-only changes; they need
  hosted Windows proof.
- R4 should remain a product decision, not a CMake option toggle.
- R5 and R6 should follow the maintained corpus/proof-owner pattern from
  Sprints 138-140.
- R7 should only require generated families that are needed for a concrete
  claim or review.
- R8 and R9 are documentation/API cleanup work and should not widen support
  claims.
- R12 and R13 are deliberately late because competitive claims require direct
  comparative evidence.

## Day 12 Handoff

Day 12 should draft the Epic 12 retrospective using the evidence inventory,
validation log, claim audits, and this published residual queue. The
retrospective should state that state-of-the-art status remains a non-claim
unless direct comparative evidence is added before final closeout.
