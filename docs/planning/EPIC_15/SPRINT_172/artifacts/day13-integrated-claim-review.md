# Sprint 172 Day 13: Integrated Claim Review

## Purpose

Reconcile the Sprint 172 LU header cleanup, tutorial signature update, guard
coverage, and validation evidence into one coherent public-API claim boundary.

## Reviewed Inputs

- `docs/planning/EPIC_15/SPRINT_172/WORKING_NOTES.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day1-header-intake.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day2-header-candidate-inventory.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day3-header-selection.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day4-contract-cleanup-design.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day5-contract-cleanup-implementation.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day6-declaration-organization-design.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day7-declaration-organization-implementation.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day8-usage-documentation-design.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day9-usage-documentation-update.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day10-mechanical-guard-design.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day11-mechanical-guard-implementation.md`
- `docs/planning/EPIC_15/SPRINT_172/artifacts/day12-integrated-header-validation.md`

## Integrated Claim Boundary

Sprint 172 supports these narrow claims:

- `include/sparse_lu.h` has clearer LU ownership, lifecycle, tolerance,
  workspace/allocation, progress/cancellation, solve, condition-estimate,
  transpose, helper, and refinement comments.
- `include/sparse_lu.h` has concise workflow headings for options,
  factorization, solves, conditioning/transpose solves, advanced solver
  phases, and refinement.
- The Day 5 and Day 7 normalized declaration captures show the LU declaration
  surface was preserved across the header cleanup.
- `docs/tutorial.md` now matches the public six-argument
  `sparse_lu_refine(...)` signature.
- `scripts/check_lu_header_docs_guard.sh` provides a lightweight guard for the
  selected LU header headings, selected declaration-name presence, tutorial
  refinement signature drift, and scoped unsupported-claim drift.
- Day 12 `make format && make lint && make test` passed after the public
  header edits.

Sprint 172 does not support these claims:

- package-manager provider availability;
- shared-library build or install support;
- dynamic ABI stability;
- runtime-loader behavior;
- Windows Makefile or Windows `pkg-config` parity;
- broad platform parity;
- portable performance or external-library parity;
- LU CSR parity;
- generated API HTML freshness;
- state-of-the-art sparse linear algebra coverage.

## Docs, Examples, And Guard Reconciliation

The cleaned LU header and tutorial now agree on the refinement call shape:

```c
sparse_lu_refine(A, LU, b, x, 3, 1e-15);
```

The focused guard passed and checks the same LU header and tutorial surfaces
that Sprint 172 changed. No examples, tests, package metadata, generated API
HTML, CMake package files, Makefile install rules, or provider recipes were
changed by the LU usage documentation update.

The guard intentionally remains standalone. It is narrow, deterministic, and
reviewable, but Sprint 172 did not promote it into a broad Makefile quality
target.

## Generated-Output Staging Check

Ran:

```sh
git status --short
git diff --name-only
git ls-files --others --exclude-standard | sort
git ls-files --others --exclude-standard | rg '(^build/|/build/|docs/api/html|doxygen|\.o$|\.a$|\.so$|\.dylib$|\.dll$|\.exe$|compile_commands\.json|coverage|\.gcda$|\.gcno$)' || true
```

Result:

- No files are staged.
- Tracked source diffs are limited to `docs/tutorial.md` and
  `include/sparse_lu.h`.
- Untracked Sprint 172 files are the planned Sprint 172 artifacts, working
  notes, plan, and `scripts/check_lu_header_docs_guard.sh`.
- No generated API HTML, Doxygen output, build output, object/archive/shared
  library artifact, executable, coverage file, or compile database is visible
  in the untracked-file generated-output scan.

## Claim Scans

Ran:

```sh
rg -n "state-of-the-art|external-library parity|portable performance|performance guarantee|package-manager support|shared-library support|dynamic ABI|runtime-loader|broad Windows parity|Windows Makefile parity|Windows pkg-config parity|LU CSR parity|Homebrew|vcpkg|Conan|pkgsrc|distro package|binary package|symbol allowlist|export macro|import macro|SONAME|DLL|dylib" include/sparse_lu.h docs/tutorial.md scripts/check_lu_header_docs_guard.sh docs/planning/EPIC_15/SPRINT_172
bash scripts/check_lu_header_docs_guard.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
```

Results:

- `include/sparse_lu.h` has no unsupported-claim matches.
- `docs/tutorial.md` matches are existing whole-file non-claim boundary
  statements that explicitly say what is not being claimed.
- `scripts/check_lu_header_docs_guard.sh` contains the unsupported-claim regex
  used to reject scoped claim drift.
- Sprint 172 planning artifacts contain the expected non-claim and stop-condition
  wording required by the sprint.
- The LU header/docs guard passed.
- The package-manager deferral guard passed.
- The static-package deferral guard passed.

## Residuals

- The LU header guard checks declaration-name presence, not exact C signatures.
  That is intentional for Sprint 172; exact signature proof remains covered by
  compiler/test gates and the Day 5/Day 7 normalized declaration captures.
- The guard is not wired into a Makefile target yet. A future sprint can decide
  whether it belongs in `docs-check`, `quality-review`, or another focused
  quality target after review.
- Generated API HTML was not regenerated or published in Sprint 172.

## Sprint 173 Handoff

- Reuse the Sprint 172 public-header cleanup pattern for the next selected
  header family only after a fresh candidate selection.
- If the LU guard is promoted into a Makefile target, preserve its scoped
  tutorial LU-section claim check to avoid false failures on valid whole-file
  non-claim boundary text.
- Continue using package-manager and static-package deferral guards whenever
  adoption, package, ABI, runtime-loader, or platform wording changes.
- Do not use Sprint 172 as evidence for shared-library, dynamic ABI,
  package-manager, broad platform, portable performance, external-library
  parity, LU CSR parity, generated API HTML publication, or state-of-the-art
  claims.

## Completion Status

Day 13 is complete. The selected LU header claim boundary is coherent with the
tutorial update, focused guard, Day 12 validation, and inherited package/ABI
deferral records. No generated artifacts are staged or unintentionally visible
in the branch artifact set.
