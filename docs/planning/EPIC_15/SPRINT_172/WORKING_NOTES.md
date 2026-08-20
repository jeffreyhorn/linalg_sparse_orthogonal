# Sprint 172 Working Notes

## Sprint Goal

Normalize one additional high-impact public header family and add lightweight
guardrails against documentation drift.

## Source Artifact Note

The Sprint 172 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`,
but the active merged Sprint 172 planning source is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section
"Sprint 172: Public Header Coherence Batch 2".

## Branch Baseline

- Branch: `sprint-172`
- Starting point: current `master` after PR #190 merge.
- Sprint 171 status: complete and merged, with formal package-manager
  deferral recorded and guarded.
- Sprint 172 plan status: day-by-day plan exists at
  `docs/planning/EPIC_15/SPRINT_172/PLAN.md`.

## Prior Evidence Carried Forward

| Input | Source | Sprint 172 use |
| --- | --- | --- |
| Epic 14 header cleanup candidate selection | `docs/planning/EPIC_14/SPRINT_164/artifacts/day2-header-selection.md` | Provides prior header-risk criteria and deferred header queue for selecting the next family. |
| Generated public-header coverage map | `docs/planning/EPIC_14/SPRINT_158/artifacts/day3-public-header-coverage-map.md` | Confirms checked-in `include/*.h` coverage expectations and generated-version-header treatment. |
| Epic 15 evidence ledger baseline | `docs/planning/EPIC_15/SPRINT_167/RETROSPECTIVE.md` | Carries claim-gate conventions and broad non-claim boundaries. |
| Static package ABI product decision | `docs/planning/EPIC_15/SPRINT_170/RETROSPECTIVE.md`, `scripts/static_package_deferral_check.sh` | Keeps shared-library, dynamic ABI, runtime-loader, and Windows package parity claims out of header cleanup. |
| Package-manager deferral decision | `docs/planning/EPIC_15/SPRINT_171/RETROSPECTIVE.md`, `scripts/package_manager_deferral_check.sh` | Keeps provider package-manager availability out of public headers and API docs. |
| Current public headers | `include/*.h` | Candidate set for Day 2 inventory and Day 3 selection. |

## Public Header Candidate Set

Sprint 172 starts with these checked-in public headers:

- `include/sparse_analysis.h`
- `include/sparse_bidiag.h`
- `include/sparse_cholesky.h`
- `include/sparse_csr.h`
- `include/sparse_dense.h`
- `include/sparse_eigs.h`
- `include/sparse_ic.h`
- `include/sparse_ilu.h`
- `include/sparse_iterative.h`
- `include/sparse_ldlt.h`
- `include/sparse_lu.h`
- `include/sparse_lu_csr.h`
- `include/sparse_matrix.h`
- `include/sparse_qr.h`
- `include/sparse_reorder.h`
- `include/sparse_svd.h`
- `include/sparse_types.h`
- `include/sparse_vector.h`

The generated installed `sparse_version.h` remains package/install owned and
should not be treated as a checked-in header cleanup target unless a later
artifact explicitly changes that policy.

## Retained Header And Claim Non-Claims

Sprint 172 starts with no support claim for:

- dynamic ABI stability from public header comments;
- shared-library builds, installs, export macros, import macros, symbol
  allowlists, or runtime-loader behavior;
- package-manager support through vcpkg, Homebrew, Conan, pkgsrc,
  distro/system packages, provider registries, taps, recipes, or binary
  packages;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- portable performance superiority;
- external-library parity;
- state-of-the-art sparse linear algebra coverage.

Header cleanup may improve API readability, ownership/lifecycle clarity,
declaration organization, and generated-doc inputs. It must not widen package,
ABI, runtime-loader, platform, performance, or ecosystem claims.

## Sprint 172 Stop Conditions

Stop and revise before proceeding if a change:

- edits public header declarations, struct layout, enum values, typedef names,
  macro definitions, include guards, or installed header names without an
  explicit Day 4/Day 6 behavior-preservation decision;
- changes `.c` or `.h` files without running
  `make format && make lint && make test`;
- treats comment cleanup as an ABI, package, runtime-loader, platform, or
  state-of-the-art support claim;
- implies package-manager provider availability contrary to Sprint 171
  deferral;
- weakens Sprint 170 shared-library/dynamic ABI deferral wording;
- adds generated API HTML, Doxygen outputs, build outputs, report artifacts,
  provider recipes, or package archives unintentionally;
- updates adoption/package/ABI/platform wording without running targeted claim
  scans and the relevant deferral guard.

## Working Assumptions

- Sprint 172 should select exactly one public header family before editing.
- Header edits should be declaration-preserving unless a later artifact
  explicitly records a safe non-behavioral reorder.
- Public header comments should describe local API contracts, not tutorials,
  product roadmaps, ABI policy, or package support.
- Prior Epic 14 header cleanup patterns should guide selection and
  declaration-preserving cleanup style.
- If only planning files change on a given day, `git diff --check` is
  sufficient for that day.
- If `.c` or `.h` files change, run the full C quality gate.
- If package/adoption/ABI wording changes, run
  `bash scripts/package_manager_deferral_check.sh` and/or
  `bash scripts/static_package_deferral_check.sh` as appropriate.

## Daily Log

### Day 1: Sprint Intake And Header Baseline

- Re-read the Sprint 172 section of
  `docs/planning/EPIC_15/PROJECT_PLAN.md`.
- Confirmed the prompt path points at an older Epic 12 report-index section,
  while the active Sprint 172 section lives in Epic 15.
- Reviewed prior Epic 14 public-header cleanup artifacts and generated
  public-header coverage mapping.
- Reviewed Sprint 167 evidence-ledger claim-gate conventions, Sprint 170
  static package/shared ABI decision boundaries, and Sprint 171
  package-manager deferral handoff.
- Created Sprint 172 working notes and artifact directory structure.
- Recorded the starting public-header candidate set under `include/*.h`.
- Defined stop conditions for public header declaration drift, package/ABI
  overclaims, generated-output staging, and quality-gate requirements.
- Day 1 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day1-header-intake.md`.

### Day 2: Public Header Candidate Inventory

- Inventoried the 18 checked-in public headers under `include/*.h`.
- Reused prior public-header cleanup selection and generated API coverage
  artifacts to keep the candidate set aligned with existing policy.
- Confirmed the generated installed `sparse_version.h` remains outside the
  checked-in header cleanup target set.
- Captured current header line counts and a coarse API-risk-term signal across
  ownership, lifecycle, error, tolerance, workspace, threading, callback,
  allocation, backend, package, ABI, and performance wording.
- Ranked the Day 3 candidate pool as:
  `include/sparse_analysis.h`, `include/sparse_ldlt.h`,
  `include/sparse_lu.h`, `include/sparse_qr.h`,
  `include/sparse_types.h`, and `include/sparse_lu_csr.h`.
- Marked `include/sparse_iterative.h`, `include/sparse_eigs.h`, and
  `include/sparse_matrix.h` as recheck-only because prior public-header
  cleanup already selected that high-impact batch.
- Recorded documentation-drift and declaration-organization risks before any
  public header edit.
- Day 2 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day2-header-candidate-inventory.md`.

### Day 3: Header Family Selection Decision

- Compared the Day 2 candidate ranking against prior cleanup history.
- Confirmed `include/sparse_analysis.h` and `include/sparse_ldlt.h` were
  already part of the Sprint 155 selected public-header cleanup batch, so they
  should be recheck-only rather than Sprint 172's primary target.
- Selected exactly one Sprint 172 cleanup family:
  `include/sparse_lu.h`.
- Recorded why LU is the next highest-value uncleaned target: it is the
  general one-shot direct-solver first-use surface and appears across README,
  examples, cookbook, API summaries, maintainer proof notes, and tests.
- Defined in-scope contract-language cleanup areas for LU ownership,
  lifecycle, errors, tolerance, allocation/workspace, threading, callbacks,
  and possible declaration sectioning.
- Defined unsupported claims for ABI stability, shared libraries,
  package-manager availability, runtime loader behavior, broad platform
  parity, portable performance, external-library parity, LU CSR parity,
  state-of-the-art status, and generated API HTML freshness.
- Identified likely impacted docs/examples/tests for Day 4 design without
  authorizing broad edits.
- Day 3 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day3-header-selection.md`.

### Day 4: Contract Cleanup Design

- Read the selected public header `include/sparse_lu.h`.
- Checked LU implementation evidence in `src/sparse_lu.c` for factored-state
  checks, error returns, cancellation behavior, in-place mutation, workspace
  allocation, and reordered one-shot publishing behavior.
- Checked focused LU tests and examples for public contract evidence:
  `tests/test_sparse_lu.c`, `tests/test_edge_cases.c`,
  `tests/test_lu_csr.c`, `examples/example_basic_solve.c`,
  `examples/example_condition.c`, `examples/example_colamd.c`, and
  `examples/example_matrix_market.c`.
- Mapped the selected LU declarations and preserved the current workflow order:
  options/factor, solve, condition/transpose, advanced helpers, refinement.
- Drafted normalized language targets for ownership, lifecycle, errors,
  tolerance, workspace/allocation, threading, callbacks, and advanced helper
  positioning.
- Explicitly disallowed Day 5 declaration, struct layout, macro, include,
  implementation, test, generated API, package, shared-library, ABI,
  runtime-loader, platform, performance, external-parity, LU CSR parity, and
  state-of-the-art claim changes.
- Defined the Day 5 declaration-preservation command plan and required full C
  quality gate because Day 5 is expected to edit a public `.h` file.
- Day 4 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day4-contract-cleanup-design.md`.

### Day 5: Contract Cleanup Implementation

- Captured the LU declaration-like surface before header edits in
  `artifacts/day5-lu-declarations-before.txt`.
- Edited only comments in `include/sparse_lu.h`; no declarations, typedefs,
  struct field order, enum values, includes, include guards, macros, or
  implementation files were intentionally changed.
- Normalized LU ownership, one-shot lifecycle, stable-pattern handoff,
  option-field, progress/cancellation, tolerance, allocation/workspace,
  solve, condition-estimate, transpose, helper, and refinement contract
  language.
- Ran `make format`; the tracked code diff remained scoped to
  `include/sparse_lu.h`.
- Captured the LU declaration-like surface after header edits in
  `artifacts/day5-lu-declarations-after.txt`.
- Created normalized before/after declaration captures and confirmed
  `diff -u` over the normalized declaration text was empty.
- Ran the unsupported-claim scan for ABI, package, runtime-loader, platform,
  performance, external-parity, LU CSR parity, and state-of-the-art wording;
  no matches were found in `include/sparse_lu.h`.
- Ran `git diff --check`; it passed.
- Ran the required full C quality gate because a public `.h` file changed:
  `make format && make lint && make test`; all passed.
- Created `artifacts/day5-contract-cleanup-implementation.md`.

### Day 6: Declaration Organization Design

- Reviewed the post-Day 5 declaration order in `include/sparse_lu.h`.
- Confirmed the current LU declaration sequence is already workflow-oriented:
  options, factorization, solves, condition/transpose support, advanced phase
  helpers, and refinement.
- Designed a Day 7 organization pass that preserves declaration order and
  normalizes section headings for generated API readability.
- Defined the recommended section groups: Options, Factorization, Solves,
  Conditioning and Transpose Solves, Advanced Solver Phases, and Refinement.
- Recorded ordering rules that keep option types before consumers, keep
  one-shot factorization APIs adjacent, keep solve APIs adjacent, keep
  condition and transpose declarations adjacent, keep phase helpers together,
  and keep refinement last.
- Recorded non-move exceptions for each LU declaration block so any Day 7 diff
  is easy to review as non-behavioral.
- Reconfirmed Day 7 must avoid declaration, struct layout, enum, macro,
  include, include-guard, implementation, test, ABI, package, runtime-loader,
  platform, performance, external-parity, LU CSR parity, and state-of-the-art
  support changes.
- Day 6 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day6-declaration-organization-design.md`.

### Day 7: Declaration Organization Implementation

- Captured the LU declaration-like surface before heading edits in
  `artifacts/day7-lu-declarations-before.txt`.
- Implemented the Day 6 declaration organization design in
  `include/sparse_lu.h`.
- Preserved declaration order and added or normalized short workflow headings:
  Options, Factorization, Solves, Conditioning and Transpose Solves, Advanced
  Solver Phases, and Refinement.
- Replaced the prior long decorative separators with concise ASCII headings.
- Ran `make format`; the tracked code diff remained scoped to
  `include/sparse_lu.h`.
- Captured the LU declaration-like surface after heading edits in
  `artifacts/day7-lu-declarations-after.txt`.
- Created normalized before/after declaration captures and confirmed
  `diff -u` over the normalized declaration text was empty.
- Ran the unsupported-claim scan for ABI, package, runtime-loader, platform,
  performance, external-parity, LU CSR parity, and state-of-the-art wording;
  no matches were found in `include/sparse_lu.h`.
- Ran the required full C quality gate because a public `.h` file changed:
  `make format && make lint && make test`; all passed.
- Ran `git diff --check`; it passed.
- Created `artifacts/day7-declaration-organization-implementation.md`.

### Day 8: Usage Documentation Design

- Reviewed user-facing LU usage references in `README.md`,
  `docs/tutorial.md`, `docs/cookbook.md`, `docs/api_reference.md`,
  `docs/maintainer_guide.md`, and LU-adjacent examples.
- Selected the Day 9 recommended workflow: one-shot LU solve on a fresh
  `sparse_copy(A)`, with optional iterative refinement using the original
  matrix and LU factor copy.
- Confirmed `README.md` already shows the intended copy/factor/solve/refine
  shape with the six-argument `sparse_lu_refine(...)` call and keeps
  one-shot vs repeated-run routing separate.
- Found one concrete usage mismatch in `docs/tutorial.md`: the LU section
  calls `sparse_lu_refine(A, LU, b, x, 3)` without the required tolerance
  argument from `include/sparse_lu.h`.
- Confirmed `docs/cookbook.md`, `docs/api_reference.md`,
  `docs/maintainer_guide.md`, `examples/example_basic_solve.c`,
  `examples/example_condition.c`, `examples/example_matrix_market.c`,
  `examples/example_colamd.c`, and `examples/cmake_example/main.c` can remain
  recheck-only for Day 9.
- Defined Day 9 edit scope as `docs/tutorial.md` only, unless the pre-edit
  recheck finds another direct LU signature mismatch.
- Defined Day 9 signature and unsupported-claim scans.
- Day 8 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day8-usage-documentation-design.md`.

### Day 9: Usage Documentation Update

- Implemented the Day 8 documentation scope in `docs/tutorial.md`.
- Updated the LU refinement snippet from the stale five-argument
  `sparse_lu_refine(A, LU, b, x, 3)` call to the public six-argument
  `sparse_lu_refine(A, LU, b, x, 3, 1e-15)` signature.
- Kept the edit scoped to the tutorial; no examples, tests, public headers,
  generated API output, package metadata, or build files were changed for
  Day 9.
- Ran the targeted LU refinement signature scan across `README.md`,
  `docs/tutorial.md`, `include/sparse_lu.h`, `examples`, and `tests`; all
  direct calls now use the six-argument signature.
- Ran the targeted unsupported-claim scan on `docs/tutorial.md`; matches were
  existing non-claim boundary language only.
- Ran `git diff --check`; it passed.
- Day 9 changed documentation only. No `.c` or `.h` files were modified for
  this day, so the full C quality gate is not required.
- Created `artifacts/day9-usage-documentation-update.md`.

### Day 10: Mechanical Guard Design

- Reviewed existing guard patterns in `scripts/package_manager_deferral_check.sh`,
  `scripts/static_package_deferral_check.sh`, report/index Python checks, and
  Makefile quality/documentation targets.
- Selected a focused shell guard design for Day 11 because the required checks
  are deterministic text-presence and scoped text-absence checks.
- Designed `scripts/check_lu_header_docs_guard.sh` as the Day 11 guard.
- Defined positive checks for the Day 7 LU header section headings:
  Options, Factorization, Solves, Conditioning and Transpose Solves, Advanced
  Solver Phases, and Refinement.
- Defined positive checks for the selected LU declaration names so the guard
  detects accidental removal from `include/sparse_lu.h` without pretending to
  prove ABI or behavior.
- Defined a positive check for the Day 9 six-argument tutorial refinement
  snippet and a negative check against the stale five-argument snippet.
- Defined scoped negative checks for unsupported package-manager,
  shared-library, dynamic ABI, runtime-loader, platform, performance,
  external-parity, LU CSR parity, and state-of-the-art claim wording.
- Deferred Makefile wiring until Day 11 implementation proves the script is
  narrow, fast, deterministic, and independent of generated files.
- Day 10 changed planning artifacts only. No `.c` or `.h` files were modified
  for this day, so the full C quality gate is not required.
- Created `artifacts/day10-mechanical-guard-design.md`.

### Day 11: Mechanical Guard Implementation

- Implemented `scripts/check_lu_header_docs_guard.sh` as the focused Day 11
  shell guard.
- Added positive checks for the Day 7 LU header section headings and selected
  LU declaration names in `include/sparse_lu.h`.
- Added a positive check for the Day 9 six-argument tutorial refinement
  snippet in `docs/tutorial.md`.
- Added a negative check that rejects the stale five-argument tutorial
  refinement snippet.
- Added scoped unsupported-claim absence checks for `include/sparse_lu.h` and
  the LU section of `docs/tutorial.md`.
- Kept the guard standalone and did not wire it into the Makefile on Day 11 so
  the new failure behavior remains directly reviewable before broader quality
  integration.
- Ran `bash scripts/check_lu_header_docs_guard.sh`; it passed with clear
  section, declaration, tutorial signature, and unsupported-claim output.
- Ran `git diff --check`; it passed.
- Day 11 did not modify `.c` or `.h` files, so the full C quality gate is not
  required for this day.
- Created `artifacts/day11-mechanical-guard-implementation.md`.

### Day 12: Integrated Header Validation

- Ran the required full C quality gate because Sprint 172 modified a public
  `.h` file: `make format && make lint && make test`; it passed.
- Ran `bash scripts/check_lu_header_docs_guard.sh`; it passed with section,
  declaration, tutorial signature, and unsupported-claim checks.
- Ran a focused LU header declaration smoke scan; expected include guard,
  option struct, and LU public declaration names remain present.
- Ran the targeted `sparse_lu_refine(...)` signature scan across `README.md`,
  `docs/tutorial.md`, `include/sparse_lu.h`, `examples`, and `tests`; direct
  references use the six-argument public signature.
- Ran the targeted unsupported-claim scan across `include/sparse_lu.h` and
  `docs/tutorial.md`; the LU header had no matches and tutorial matches were
  existing non-claim boundary wording only.
- Did not run package-manager or static-package deferral guards because Day 12
  did not touch adoption, package, ABI, runtime-loader, or platform-boundary
  wording.
- Ran `git diff --check`; it passed.
- Created `artifacts/day12-integrated-header-validation.md`.

### Day 13: Integrated Claim Review

- Reviewed Sprint 172 working notes and Day 1 through Day 12 artifacts.
- Reconciled the supported Sprint 172 claim boundary to LU header readability,
  declaration-preserving organization, tutorial refinement signature
  alignment, focused guard coverage, and Day 12 full-gate validation.
- Reconfirmed Sprint 172 does not claim package-manager provider availability,
  shared-library support, dynamic ABI stability, runtime-loader behavior,
  Windows Makefile or Windows `pkg-config` parity, broad platform parity,
  portable performance, external-library parity, LU CSR parity, generated API
  HTML freshness, or state-of-the-art sparse linear algebra coverage.
- Checked branch inventory with `git status --short`, `git diff --name-only`,
  and untracked-file scans; no generated API HTML, Doxygen output, build
  output, object/archive/shared-library artifact, executable, coverage file, or
  compile database is visible as an unintended branch artifact.
- Ran the broad targeted claim scan across `include/sparse_lu.h`,
  `docs/tutorial.md`, `scripts/check_lu_header_docs_guard.sh`, and Sprint 172
  planning artifacts; hits were expected non-claim boundary text or guard regex
  text, with no unsupported claim in the selected LU header.
- Ran `bash scripts/check_lu_header_docs_guard.sh`; it passed.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Recorded residuals for future exact-signature guard scope, possible Makefile
  guard wiring, and generated API HTML publication.
- Created `artifacts/day13-integrated-claim-review.md`.

### Day 14: Sprint Closeout And Sprint 173 Handoff

- Reconciled Sprint 172 project-plan items 172.1 through 172.6 against the
  daily artifacts and source-controlled deliverables.
- Confirmed Sprint 172 selected and cleaned one public header family:
  `include/sparse_lu.h`.
- Confirmed the tutorial LU refinement snippet is aligned with the public
  six-argument `sparse_lu_refine(...)` signature.
- Confirmed `scripts/check_lu_header_docs_guard.sh` remains the focused guard
  for LU header headings, selected declaration-name presence, tutorial
  signature drift, stale five-argument tutorial call rejection, and scoped
  unsupported-claim absence.
- Ran `bash scripts/check_lu_header_docs_guard.sh`; it passed.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `git diff --check`; it passed.
- Checked branch inventory and generated-output staging; no generated API HTML,
  Doxygen output, build output, object/archive/shared-library artifact,
  executable, coverage file, or compile database was found as an unintended
  branch artifact.
- Recorded that Day 12 remains the final full C gate for Sprint 172 after
  public header edits: `make format && make lint && make test` passed.
- Prepared Sprint 173 handoff notes for generated API HTML publication
  boundaries.
- Created `artifacts/day14-sprint-closeout.md`.
