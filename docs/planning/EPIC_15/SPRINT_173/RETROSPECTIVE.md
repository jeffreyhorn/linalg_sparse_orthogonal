# Sprint 173 Retrospective

**Sprint:** 173 - Generated API HTML Publication Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-173`)
**Status:** Complete

## Source Artifact Note

Sprint 173 was executed from the active Epic 15 project-plan section for
Sprint 173 and lives under `docs/planning/EPIC_15/SPRINT_173/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 173 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited the generated API configuration, Doxygen input set, output path,
      local commands, ignore policy, and existing page coverage guard.
- [x] Compared hosted, committed, CI artifact-only, and guarded local-only
      generated API HTML publication modes.
- [x] Selected guarded local-only generated API HTML as the Sprint 173
      publication decision.
- [x] Added `scripts/check_api_docs_local_only.sh` to enforce ignored,
      untracked, unstaged, and non-ignored-untracked generated API output
      boundaries.
- [x] Added `make api-docs-local-only`, `make api-docs-validate`, and
      `make api-docs-freshness` while preserving existing `make docs` and
      `make docs-check` behavior.
- [x] Updated README, `docs/api_reference.md`, and `docs/maintainer_guide.md`
      so users and maintainers know that generated API HTML is local-only and
      current only after `make api-docs-freshness` passes.
- [x] Ran generated API freshness, local-only staging, claim-scan, static
      package deferral, package-manager deferral, and diff-hygiene checks.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required for Sprint 173 edits.

## What Went Well

1. **The generated API publication question got a concrete decision.** Sprint
   173 selected guarded local-only generated API HTML instead of leaving
   hosted, committed, or artifact-only generated documentation ambiguous.

2. **The selected path is executable.** The new
   `scripts/check_api_docs_local_only.sh` guard and
   `make api-docs-freshness` target give maintainers a single command that
   regenerates local Doxygen HTML, checks expected pages, and enforces
   generated-output staging boundaries.

3. **Existing generation behavior stayed stable.** The sprint preserved
   `make docs`, `make docs-check`, `Doxyfile`, `.gitignore`, and the existing
   `scripts/check_api_docs_coverage.py` coverage model while adding a clearer
   freshness command on top.

4. **Documentation now matches the maintained path.** README lists the new
   freshness target, `docs/api_reference.md` explains when generated HTML is
   current, and `docs/maintainer_guide.md` records the local-only maintenance
   policy and non-claims.

5. **The guard covers the most likely generated-output mistakes.** It fails if
   `docs/api/` stops being ignored, generated API HTML becomes tracked or
   staged, or ignored generated output becomes visible as non-ignored
   untracked output.

6. **Claim boundaries stayed narrow.** Sprint 173 did not turn local generated
   API HTML into hosted documentation, committed generated output, artifact
   publication, release evidence, package-manager support, shared-library
   support, dynamic ABI support, platform parity, performance evidence,
   external-library parity, or a state-of-the-art claim.

7. **Validation matched the changed surface.** The sprint ran the API docs
   freshness/local-only checks, deferral guards, claim scans, ignored-output
   staging checks, and diff hygiene after the docs and Makefile/script changes.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 173 plan belongs to Epic 15. The sprint handled this by
   recording the mismatch and proceeding from
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.

2. **No hosted generated API site was delivered.** The sprint chose the right
   low-risk path for the current evidence, but users still do not have a
   hosted generated API HTML URL.

3. **CI does not yet run the generated API freshness target.** The selected
   command is local and source-controlled. A future sprint can add a CI
   docs-check lane, but Sprint 173 did not promote generated API HTML into
   hosted or artifact evidence.

4. **The freshness model is command-regenerated, not timestamp metadata.**
   `make api-docs-freshness` is reliable when run, but the repository still
   does not persist generated-output freshness metadata for ignored HTML.

5. **`sparse_version.h` remains outside Doxygen page coverage.** This is
   correct under the current generated installed-header policy, but it remains
   a point that users and maintainers must understand.

6. **Claim scans still require interpretation.** Valid local-only, hosted
   report, release-evidence, package, ABI, and artifact terms appear in
   non-claims, unrelated report surfaces, and guard failure messages.

## Final Metrics

### Validation

| Metric | Sprint 173 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required by changed files | no |
| generated API freshness target | passed: `make api-docs-freshness` |
| direct local-only generated-output guard | passed: `make api-docs-local-only` |
| generated API page coverage | passed: 18 checked-in public headers, 18 reference pages, 18 source pages |
| generated `sparse_version.h` policy | separate installed-header policy row; not an expected Doxygen page |
| generated API output staging state | passed: `docs/api/` remains ignored local output |
| static package/shared ABI deferral guard | passed |
| package-manager deferral guard | passed |
| targeted generated API claim scan | passed by inspection; matches were selected local-only wording, guard text, or unrelated bounded report surfaces |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 173 close state |
| --- | ---: |
| C source files changed | 0 |
| public header files changed | 0 |
| shell guard scripts added | 1 |
| Makefile targets added | 3 |
| public/maintainer docs changed | 3 |
| workflow files changed | 0 |
| Doxygen config files changed | 0 |
| generated API HTML files tracked | 0 |
| daily artifacts under `SPRINT_173/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |

### Claim Governance

| Metric | Sprint 173 close state |
| --- | ---: |
| generated API publication decisions recorded | 1 |
| selected generated API publication path | guarded local-only |
| hosted generated API HTML claims added | 0 |
| committed generated API HTML claims added | 0 |
| artifact-only generated API HTML claims added | 0 |
| release-evidence generated API claims added | 0 |
| package-manager support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI compatibility claims added | 0 |
| runtime-loader support claims added | 0 |
| broad platform parity claims added | 0 |
| portable performance claims added | 0 |
| external-library parity claims added | 0 |
| state-of-the-art sparse linear algebra claims added | 0 |
| new generated-output local-only guards | 1 |

## Closed Claim

Sprint 173 closes this Epic 15 generated API publication claim:

The project has a maintained local generated API HTML freshness path. Generated
Doxygen HTML remains ignored local output under `docs/api/`, and maintainers
can run `make api-docs-freshness` to regenerate it, check expected public
header pages, and prove the generated API tree is ignored, untracked,
unstaged, and not visible as non-ignored untracked output.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-api-docs-intake.md](./artifacts/day1-api-docs-intake.md);
- [day2-generator-inventory.md](./artifacts/day2-generator-inventory.md);
- [day3-publication-options.md](./artifacts/day3-publication-options.md);
- [day4-publication-decision.md](./artifacts/day4-publication-decision.md);
- [day5-generator-design.md](./artifacts/day5-generator-design.md);
- [day6-generator-implementation.md](./artifacts/day6-generator-implementation.md);
- [day7-freshness-design.md](./artifacts/day7-freshness-design.md);
- [day8-freshness-implementation.md](./artifacts/day8-freshness-implementation.md);
- [day9-navigation-design.md](./artifacts/day9-navigation-design.md);
- [day10-navigation-update.md](./artifacts/day10-navigation-update.md);
- [day11-generator-validation.md](./artifacts/day11-generator-validation.md);
- [day12-maintenance-review.md](./artifacts/day12-maintenance-review.md);
- [day13-claim-review.md](./artifacts/day13-claim-review.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No hosted generated API HTML publication, committed generated API HTML,
CI artifact-only generated API HTML, generated API release evidence,
generated installed-header Doxygen coverage for `sparse_version.h`,
package-manager provider availability, shared-library build/install support,
dynamic ABI stability, runtime-loader behavior, Windows Makefile parity,
Windows `pkg-config` execution parity, broad platform parity, portable
performance guarantee, external-library parity claim, or state-of-the-art
sparse linear algebra claim was added.

## Sprint 174 Readiness

| Future need | Sprint 173 handoff |
| --- | --- |
| Local generated API freshness | Run `make api-docs-freshness` before relying on local generated Doxygen HTML. |
| Generated API output staging | Do not stage files under `docs/api/`; run `make api-docs-local-only` for direct boundary proof. |
| Hosted generated API HTML | Create a new publication decision with URL ownership, deployment permissions, retention policy, freshness semantics, and support wording before implementation. |
| CI artifact-only generated API HTML | Define upload/retention/reviewer-access policy and claim status before adding an artifact lane. |
| Committed generated API HTML | Reverse the local-only decision explicitly before changing `.gitignore` or staging generated files. |
| CI docs-check lane | It may run `make api-docs-freshness` as a check without changing publication status. |
| `sparse_version.h` Doxygen coverage | Treat it as install/version-validation owned unless Doxygen input policy changes. |
| Package-manager wording changes | Run `bash scripts/package_manager_deferral_check.sh`. |
| Static package/shared ABI wording changes | Run `bash scripts/static_package_deferral_check.sh`. |
