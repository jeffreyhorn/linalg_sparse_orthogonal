# Sprint 172 Retrospective

**Sprint:** 172 - Public Header Coherence Batch 2
**Duration:** 14 days (Days 1-14 landed on branch `sprint-172`)
**Status:** Complete

## Source Artifact Note

Sprint 172 was executed from the active Epic 15 project-plan section for
Sprint 172 and lives under `docs/planning/EPIC_15/SPRINT_172/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 172 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Inventoried checked-in public headers and selected exactly one cleanup
      family for this sprint.
- [x] Selected `include/sparse_lu.h` as the next high-impact public header
      family after rechecking prior Epic 14 cleanup targets.
- [x] Normalized LU ownership, lifecycle, error, tolerance,
      workspace/allocation, progress/cancellation, solve, condition-estimate,
      transpose, helper, and refinement comments.
- [x] Organized the LU public declarations under concise workflow headings
      without changing the declaration-like surface.
- [x] Captured before/after normalized LU declaration evidence for the Day 5
      and Day 7 header edit points.
- [x] Updated `docs/tutorial.md` so the LU refinement snippet uses the public
      six-argument `sparse_lu_refine(...)` signature.
- [x] Added `scripts/check_lu_header_docs_guard.sh` as a focused guard for LU
      header section drift, declaration-name presence, tutorial signature
      drift, stale tutorial signature rejection, and scoped unsupported-claim
      absence.
- [x] Ran required full C quality gates after public header edits.
- [x] Ran focused LU guard, package-manager deferral guard, static package
      deferral guard, claim scans, generated-output staging scans, and
      diff-hygiene checks before closeout.

## What Went Well

1. **The sprint picked one header family and stayed scoped.** Day 3 selected
   `include/sparse_lu.h` after inventorying the public headers and rechecking
   prior cleanup history. This avoided broad public-header churn.

2. **The LU header became easier to read without changing behavior.** The
   sprint cleaned contract comments and added workflow headings for options,
   factorization, solves, conditioning/transpose solves, advanced solver
   phases, and refinement while preserving declaration order and signatures.

3. **Declaration-preservation evidence was explicit.** Day 5 and Day 7
   captured before/after LU declaration-like surfaces and normalized captures,
   giving reviewers concrete evidence that the header cleanup was
   comment/organization work rather than API behavior work.

4. **The tutorial now matches the public API.** Day 9 fixed the stale
   five-argument `sparse_lu_refine(A, LU, b, x, 3)` snippet so the tutorial
   shows the six-argument signature with tolerance.

5. **Documentation drift now has a local guard.** The new
   `scripts/check_lu_header_docs_guard.sh` guard protects the LU section
   headings, selected declaration names, tutorial refinement call shape, stale
   tutorial call absence, and scoped unsupported-claim boundary.

6. **Claim boundaries stayed disciplined.** Sprint 172 did not turn header
   cleanup into package-manager, shared-library, dynamic ABI, runtime-loader,
   platform parity, performance, external-library parity, LU CSR parity,
   generated API HTML freshness, or state-of-the-art claims.

7. **Validation matched the changed surface.** The sprint ran the full C gate
   after public header edits and final focused guards before closeout.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 172 plan belongs to Epic 15. The sprint handled this by
   recording the mismatch and proceeding from
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.

2. **The LU guard is intentionally shallow.** It checks section headings and
   declaration-name presence, not exact C signatures. Exact signature proof
   remains covered by compiler/test gates and normalized declaration captures.

3. **The guard is not wired into a Makefile target.** Keeping it standalone
   made review easier, but maintainers must invoke it directly until a future
   sprint decides whether to add it to `docs-check`, `quality-review`, or
   another target.

4. **Generated API HTML publication is still pending.** Sprint 172 improved
   public header inputs, but it did not regenerate, stage, publish, or claim
   fresh generated API HTML.

5. **Only one header family was cleaned.** That was the correct scope for this
   sprint, but remaining public headers still require future candidate
   selection and cleanup passes.

6. **Claim scans still require interpretation.** Valid non-claim language and
   guard regex text intentionally include package, ABI, platform, performance,
   and state-of-the-art terms. The scans are useful, but not zero-match gates
   over all planning text.

## Final Metrics

### Validation

| Metric | Sprint 172 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | yes: `include/sparse_lu.h` |
| full C quality gate required by changed files | yes |
| full C quality gate | passed: `make format && make lint && make test` |
| LU header/docs guard | passed |
| package-manager deferral guard | passed |
| static package/shared ABI deferral guard | passed |
| LU declaration preservation captures | passed: normalized Day 5 and Day 7 captures matched |
| LU refinement signature scan | passed: direct references use the six-argument public signature |
| targeted claim scan | passed by inspection; matches were non-claims, planning boundaries, or guard regex text |
| generated-output staging check | passed: no generated API HTML, Doxygen, build, coverage, package, or binary artifacts found |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 172 close state |
| --- | ---: |
| C source files changed | 0 |
| public header files changed | 1 |
| user documentation files changed | 1 |
| shell guard scripts added | 1 |
| Makefile behavior changed | 0 |
| CMake behavior changed | 0 |
| workflow files changed | 0 |
| daily artifacts under `SPRINT_172/artifacts/` | 14 |
| declaration capture files | 8 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |

### Claim Governance

| Metric | Sprint 172 close state |
| --- | ---: |
| selected public header families cleaned | 1 |
| package-manager support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI compatibility claims added | 0 |
| runtime-loader support claims added | 0 |
| Windows Makefile parity claims added | 0 |
| Windows `pkg-config` execution parity claims added | 0 |
| broad platform parity claims added | 0 |
| portable performance claims added | 0 |
| external-library parity claims added | 0 |
| LU CSR parity claims added | 0 |
| generated API HTML freshness claims added | 0 |
| state-of-the-art sparse linear algebra claims added | 0 |
| new focused guards for selected-header drift | 1 |

## Closed Claim

Sprint 172 closes this Epic 15 public-header coherence claim:

The project has one additional cleaned public header family,
`include/sparse_lu.h`, with clearer local API contract comments, coherent
workflow headings, preserved declaration evidence, aligned tutorial usage, and
a focused guard against LU header/docs drift.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-header-intake.md](./artifacts/day1-header-intake.md);
- [day2-header-candidate-inventory.md](./artifacts/day2-header-candidate-inventory.md);
- [day3-header-selection.md](./artifacts/day3-header-selection.md);
- [day4-contract-cleanup-design.md](./artifacts/day4-contract-cleanup-design.md);
- [day5-contract-cleanup-implementation.md](./artifacts/day5-contract-cleanup-implementation.md);
- [day6-declaration-organization-design.md](./artifacts/day6-declaration-organization-design.md);
- [day7-declaration-organization-implementation.md](./artifacts/day7-declaration-organization-implementation.md);
- [day8-usage-documentation-design.md](./artifacts/day8-usage-documentation-design.md);
- [day9-usage-documentation-update.md](./artifacts/day9-usage-documentation-update.md);
- [day10-mechanical-guard-design.md](./artifacts/day10-mechanical-guard-design.md);
- [day11-mechanical-guard-implementation.md](./artifacts/day11-mechanical-guard-implementation.md);
- [day12-integrated-header-validation.md](./artifacts/day12-integrated-header-validation.md);
- [day13-integrated-claim-review.md](./artifacts/day13-integrated-claim-review.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No package-manager provider availability, shared-library build/install
support, dynamic ABI stability, runtime-loader behavior, Windows Makefile
parity, Windows `pkg-config` execution parity, broad platform parity, portable
performance guarantee, external-library parity, LU CSR parity, generated API
HTML freshness, or state-of-the-art sparse linear algebra claim was added.

## Sprint 173 Readiness

| Future need | Sprint 172 handoff |
| --- | --- |
| Generated API HTML publication | Treat the cleaned LU header as improved input, but do not claim fresh generated API HTML until Sprint 173 generation, freshness, staging, and publication gates pass. |
| LU header/docs drift | Run `bash scripts/check_lu_header_docs_guard.sh` before citing LU header cleanup or tutorial alignment. |
| Guard integration | Decide whether the LU guard belongs in `docs-check`, `quality-review`, or another focused target after review. |
| Exact signature guard scope | Use compiler/test gates or a structured parser if future work needs exact public signature checks. |
| Package-manager wording changes | Run `bash scripts/package_manager_deferral_check.sh`. |
| Static package/shared ABI wording changes | Run `bash scripts/static_package_deferral_check.sh`. |
| Future public-header cleanup | Repeat candidate selection before touching another header family. |
