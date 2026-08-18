# Sprint 164 Retrospective

**Sprint:** 164 - Public Header And API Coherence Batch
**Duration:** 14 days (Days 1-14 landed on branch `sprint-164`)
**Status:** Complete

## Source Artifact Note

Sprint 164 was executed from the Epic 14 project-plan section for Sprint 164
and lives under `docs/planning/EPIC_14/SPRINT_164/` with its plan, working
notes, artifacts, and retrospective in one package. The original sprint prompt
referenced an older Epic 12 project-plan path; `WORKING_NOTES.md` records that
path mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 164 plan, working notes, daily artifacts, closeout
      artifact, and retrospective.
- [x] Inventoried public headers, generated API reference inputs, README,
      tutorial, cookbook, solver-selection docs, maintainer guide, and docs
      validation targets.
- [x] Selected a bounded public-header cleanup batch:
      `include/sparse_iterative.h`, `include/sparse_eigs.h`, and
      `include/sparse_matrix.h`.
- [x] Designed and captured a normalized declaration baseline before public
      header edits.
- [x] Cleaned selected header ownership, lifetime, output-buffer, error,
      option, backend, and workflow-navigation wording without changing public
      declarations.
- [x] Updated `docs/solver_selection.md` to align eigensolver AUTO routing and
      workflow links with the selected headers.
- [x] Updated README and tutorial API wording where stale public-doc names or
      backend summaries contradicted the selected headers.
- [x] Applied the Sprint 158 generated-reference policy to the selected header
      batch with `make docs-check`.
- [x] Re-captured declarations after cleanup and after formatting; all
      checksums matched the baseline.
- [x] Ran the required public-header quality gate:
      `make format && make lint && make test`.
- [x] Preserved package, ABI, shared-library, runtime-loader, hosted-docs,
      backend-superiority, portable-performance, external-parity, and
      state-of-the-art non-claims.
- [x] Published the Sprint 165 static-first package/API handoff.

## What Went Well

1. **The header batch stayed deliberately bounded.** The sprint selected three
   high-impact public headers and resisted widening into every public header,
   which kept declaration preservation reviewable.

2. **Declaration preservation was concrete.** The sprint created a repeatable
   normalized declaration capture, recorded a SHA-256 baseline, and proved the
   final selected-header state matched exactly.

3. **Header cleanup focused on call-site clarity.** Ownership, borrowing,
   output-buffer, result-state, progress callback, matrix-free, repeated-run
   handle, and eigensolver backend wording became clearer without moving
   maintainer-only history into public headers.

4. **Backend wording was recalibrated.** Eigensolver AUTO dispatch,
   `backend_used`, `peak_basis_size`, preconditioner, block-size, and LOBPCG
   wording now describe routing and telemetry instead of backend superiority
   or portable performance evidence.

5. **Public docs now match the selected headers.** The README no longer
   describes `sparse_eigs_sym(...)` as only grow-m Lanczos, and the tutorial
   now uses the actual public result type `sparse_eigs_t`.

6. **Generated reference policy stayed clean.** `make docs-check` passed with
   full checked-in public-header coverage, while generated HTML remained local
   ignored output rather than a hosted or source-controlled publication claim.

## What Didn't Go Well

1. **The sprint prompt path was stale.** The request referenced Epic 12 while
   the active Sprint 164 plan lives under Epic 14. The sprint recorded the
   mismatch and proceeded from the current Epic 14 plan.

2. **No maintained declaration helper exists yet.** The normalization command
   was repeatable and recorded, but it remains local evidence instead of a
   first-class repository script or make target.

3. **Generated API HTML remains local-only.** This matches the Sprint 158
   policy, but reviewers still need to regenerate local Doxygen output rather
   than inspect committed generated pages.

4. **Several public headers remain outside the cleanup batch.** Direct-solver,
   QR, SVD, preconditioner, reorder, dense, bidiag, vector, and shared-type
   headers still have separate cleanup opportunities.

5. **Full validation is expensive.** Public-header edits correctly required
   `make format && make lint && make test`; the gate passed, but the lint/test
   path is long enough that later artifact-only days reused the recorded full
   gate and ran targeted closeout checks.

## Final Metrics

### Validation

| Metric | Sprint 164 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | yes: 3 selected public headers |
| full C quality gate required | yes |
| full C quality gate | passed: `make format && make lint && make test` |
| declaration baseline checksum | `513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41` |
| final declaration checksum | `513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41` |
| declaration diff | no output |
| generated API docs | passed: `make docs-check` |
| checked-in public headers covered by Doxygen | 18 |
| generated reference pages | 18 |
| generated source pages | 18 |
| stale eigensolver type/backend scan | passed |
| unsupported-claim scan | passed; hits are disclaimers or policy boundaries |
| generated/local evidence committed | 0 files |
| final `git diff --check` | passed |

### Selected Header Surface

| Metric | Sprint 164 close state |
| --- | ---: |
| public headers inventoried | 18 checked-in headers plus version template |
| selected cleanup headers | 3 |
| normalized selected-header lines | 346 |
| selected header declaration drift | 0 |
| selected installed header name changes | 0 |
| public struct layout changes | 0 |
| function declaration changes | 0 |
| macro or enum changes | 0 |

### Artifact Package

| Metric | Sprint 164 close state |
| --- | ---: |
| daily artifacts | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| public headers changed | 3 |
| public docs changed | 3 |
| generated Doxygen HTML committed | 0 |
| local declaration evidence committed | 0 |

## Closed Claim

Sprint 164 closes this public-header/API coherence claim:

The project now has a declaration-preserving cleanup batch for
`sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h`. The selected
headers provide clearer ownership, lifetime, output-buffer, result-state,
callback, repeated-run handle, and backend-routing contracts without changing
public declarations. README, tutorial, solver-selection, generated-reference
policy, and maintainer guidance are coherent with the selected headers. The
generated API reference remains local-only and validated by `make docs-check`.
No package, ABI, shared-library, runtime-loader, backend-superiority, portable
performance, external-parity, hosted-docs, or state-of-the-art claim was added.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-header-selection.md](./artifacts/day2-header-selection.md);
- [day3-declaration-baseline-design.md](./artifacts/day3-declaration-baseline-design.md);
- [day4-declaration-baseline.md](./artifacts/day4-declaration-baseline.md);
- [day5-ownership-cleanup.md](./artifacts/day5-ownership-cleanup.md);
- [day6-error-output-cleanup.md](./artifacts/day6-error-output-cleanup.md);
- [day7-options-backend-cleanup.md](./artifacts/day7-options-backend-cleanup.md);
- [day8-cross-link-cleanup.md](./artifacts/day8-cross-link-cleanup.md);
- [day9-generated-reference-check.md](./artifacts/day9-generated-reference-check.md);
- [day10-declaration-preservation.md](./artifacts/day10-declaration-preservation.md);
- [day11-documentation-coherence.md](./artifacts/day11-documentation-coherence.md);
- [day12-focused-validation.md](./artifacts/day12-focused-validation.md);
- [day13-evidence-review.md](./artifacts/day13-evidence-review.md);
- [day14-closeout.md](./artifacts/day14-closeout.md).

## Sprint 165 Readiness

Sprint 165 should begin from these settled Sprint 164 boundaries:

| Starting item | Required posture |
| --- | --- |
| Public declarations | Treat selected public declarations as unchanged unless future work explicitly approves API drift. |
| Public-header/API docs | Use Sprint 164 cleaned wording as the current call-site contract reference for the selected headers. |
| Generated API HTML | Keep local-only and ignored; use `make docs-check` for coverage/freshness. |
| Static-first package boundary | Audit package metadata and install surfaces without reopening Sprint 164 header declarations. |
| Package/ABI wording | Keep shared-library, dynamic ABI, runtime-loader, package-manager, and Windows parity wording as explicit non-claims unless new product proof exists. |

Recommended Sprint 165 first step:

Audit CMake package files, `sparse.pc`, install scripts, CI checks, version
docs, README/INSTALL/package comments, and maintainer guidance for accidental
shared-library, dynamic ABI, runtime-loader, package-manager, or broad Windows
parity claims.

## Residual Deferred Debt

Still explicitly unresolved at Sprint 164 close:

- broader non-selected-header public-comment cleanup:
  `sparse_ldlt.h`, `sparse_qr.h`, `sparse_svd.h`, `sparse_ilu.h`,
  `sparse_ic.h`, and other lower-risk public headers;
- table-wide README/API index reshaping;
- generated API HTML publication beyond local ignored output;
- package/ABI product changes or shared-library support;
- backend threshold retuning or new performance claims;
- exhaustive tutorial expansion for every option/result field;
- maintained helper script or make target for declaration-preservation capture.

Still consciously constrained rather than silently solved:

- no public declaration changes;
- no dynamic ABI compatibility claim;
- no shared-library support claim;
- no runtime-loader behavior claim;
- no package-manager distribution claim;
- no broad Windows Makefile or Windows `pkg-config` parity claim;
- no backend superiority claim;
- no portable performance claim;
- no hosted generated API HTML publication claim;
- no state-of-the-art coverage claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-header-selection.md](./artifacts/day2-header-selection.md)
- [day3-declaration-baseline-design.md](./artifacts/day3-declaration-baseline-design.md)
- [day4-declaration-baseline.md](./artifacts/day4-declaration-baseline.md)
- [day5-ownership-cleanup.md](./artifacts/day5-ownership-cleanup.md)
- [day6-error-output-cleanup.md](./artifacts/day6-error-output-cleanup.md)
- [day7-options-backend-cleanup.md](./artifacts/day7-options-backend-cleanup.md)
- [day8-cross-link-cleanup.md](./artifacts/day8-cross-link-cleanup.md)
- [day9-generated-reference-check.md](./artifacts/day9-generated-reference-check.md)
- [day10-declaration-preservation.md](./artifacts/day10-declaration-preservation.md)
- [day11-documentation-coherence.md](./artifacts/day11-documentation-coherence.md)
- [day12-focused-validation.md](./artifacts/day12-focused-validation.md)
- [day13-evidence-review.md](./artifacts/day13-evidence-review.md)
- [day14-closeout.md](./artifacts/day14-closeout.md)
