# Sprint 139 Working Notes

## Sprint Goal

Completely close the selected QR residual with corpus-backed fixtures, oracle
evidence, focused proof ownership, and updated claim wording.

Sprint 139 consumes the Sprint 138 maintained corpus lane and turns it into
solver-backed QR evidence. The sprint must close one bounded QR residual fully
instead of widening broad QR, SuiteSparse, minimum-norm, raw-basis, platform,
or state-of-the-art claims.

## Starting Constraints

- Use `qr_rank_deficient_6x4_nullspace_v1` as the first QR closure lane unless
  Day 2 finds a blocking contradiction.
- Preserve the Sprint 138 corpus boundary: generated fixture facts and oracle
  rows are fixture-local evidence only.
- Close solver-backed QR behavior by comparing normalized residual evidence
  for the fixed null-vector direction `[-1, -1, 0, 1]`.
- Do not require raw QR basis equality; valid QR nullspace bases may differ by
  sign, scale normalization, orientation, or equivalent subspace basis.
- Keep optional-data and SuiteSparse rows separate from QR pass evidence until
  reviewed hosted evidence exists.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check`, a trailing
  whitespace scan under the sprint directory, and focused Markdown link/path
  validation.

## Input Artifact Inventory

| Input | Role in Sprint 139 |
| --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 139 | Defines Sprint 139 items, deliverables, prerequisites, and 168-hour estimate. |
| `docs/planning/EPIC_12/SPRINT_139/PLAN.md` | Provides day-level execution order and validation expectations. |
| `docs/planning/EPIC_12/SPRINT_137/RETROSPECTIVE.md` | Confirms Epic 12 selected targets: corpus first, QR residual closure second, partial-SVD third. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day7-gap-selection-decision.md` | Establishes the QR residual as a selected Epic 12 gap to close after corpus architecture. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day8-corpus-oracle-evidence-templates.md` | Defines fixture, expected-result, oracle, report, and failure semantics inherited by QR rows. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day11-quality-surface-map.md` | Defines validation by touched surface, including full quality gates for `.c` and `.h` edits. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day12-public-claim-freeze.md` | Freezes unsupported public claims until evidence exists. |
| `docs/planning/EPIC_12/SPRINT_138/RETROSPECTIVE.md` | Publishes Sprint 139 readiness fields for the first QR corpus lane. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day13-documentation-sprint139-handoff.md` | Provides fixture key, generator key, row IDs, null-vector direction, tolerance, and validation prerequisites. |
| `tests/corpus/README.md` | Documents the maintained corpus layout, QR handoff, and pass-evidence boundaries. |
| `tests/corpus/manifests/fixtures.tsv` | Owns `qr_rank_deficient_6x4_nullspace_v1` fixture metadata. |
| `tests/corpus/manifests/generators.tsv` | Owns `qr_rank_deficient_6x4_nullspace_generator_v1` deterministic generation metadata and hashes. |
| `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` | Owns expected rank, nullity, and normalized residual rows. |
| `scripts/validate_corpus_schema.py` | Validates corpus shape, generator hashes, expected rows, and false-pass guardrails. |
| `scripts/run_corpus_oracle.py` | Emits local generated oracle/report rows for the current corpus lane. |
| `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_qr_helpers.h` | Current QR proof surfaces that may receive or donate a focused proof owner. |
| `tests/qr_external_dense_reference.py` | Existing external dense-reference helper for bounded QR fixture checks. |
| `include/sparse_qr.h`, `src/sparse_qr.c`, `src/sparse_qr_householder.c`, `src/sparse_qr_internal.h` | QR public and implementation surfaces; changes here trigger full C quality gates. |
| `README.md`, `docs/algorithm.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md`, `examples/README.md` | Public and maintainer wording surfaces that may need earned QR claim updates after evidence exists. |

## Current QR Evidence Inventory

| Surface | Current evidence | Sprint 139 interpretation |
| --- | --- | --- |
| `include/sparse_qr.h` | Public QR API for factorization, solve, rank, nullspace, condition estimate, and minimum-norm refinement. | Public contract surface; do not widen behavior claims without matching tests/docs. |
| `src/sparse_qr.c` and `src/sparse_qr_householder.c` | QR implementation and private Householder kernel ownership. | Candidate implementation surfaces if closure reveals a solver defect. |
| `tests/test_qr.c` | Existing QR factorization, rank, nullspace, projector, rank-threshold, and scalar-boundary coverage. | Candidate source of a focused nullspace proof owner. |
| `tests/test_qr_solve.c` | QR solve, least-squares, rank-deficient residual, minimum-norm, and external-reference coverage. | Candidate integration surface if selected residual needs solve-side evidence. |
| `tests/test_qr_helpers.h` | QR fixture builders, insertion helpers, reconstruction and residual helpers. | Likely place to add reusable corpus fixture builder if C proof work starts. |
| `tests/qr_external_dense_reference.py` | Dense Python references for bounded QR least-squares, rank, residual, and minimum-norm fixtures. | Potential oracle helper pattern; not a broad LAPACK/NumPy/SciPy parity claim. |
| `tests/corpus/*` | Maintained first corpus lane with rank, nullity, and normalized null-vector residual expected rows. | Sprint 139 proof should consume this lane rather than creating an unrelated QR fixture. |
| `docs/maintainer_guide.md` | Lists QR evidence owners and extensive QR non-claims. | Wording must be tightened only after the focused proof owner lands. |
| `README.md`, `docs/algorithm.md`, `docs/cookbook.md`, `docs/solver_selection.md` | Public QR usage and algorithm guidance. | Earned wording may mention the closed fixture-local residual only after validation. |

## Sprint 139 Closure Candidate

| Field | Value |
| --- | --- |
| fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| generator key | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| shape | 6 rows by 4 columns |
| nonzeros | 14 |
| expected rank | 3 |
| expected nullity | 1 |
| null vector direction | `[-1, -1, 0, 1]` |
| rank row ID | `qr_rank_deficient_6x4_nullspace_v1_rank` |
| nullity row ID | `qr_rank_deficient_6x4_nullspace_v1_nullity` |
| residual row ID | `qr_rank_deficient_6x4_nullspace_v1_projector_residual` |
| initial tolerance | normalized null-vector residual `<= 1e-10` |

Initial closure criteria:

- QR factorization of the selected fixture reports rank `3`.
- QR nullspace extraction reports nullity `1`.
- A solver-backed nullspace vector satisfies the maintained normalized
  residual threshold against the fixture.
- The proof owner is focused and discoverable, either as an extracted lane from
  `tests/test_qr.c` or as a dedicated helper/test owner.
- Documentation wording remains fixture-local unless reviewed evidence supports
  a broader support tier.

## Day-Level Ownership

| Day | Owner focus | Project-plan items |
| --- | --- | --- |
| 1 | QR residual intake, inherited evidence inventory, closure criteria, validation expectations | Items 1-7 |
| 2 | Re-rank QR residuals and select the priority residual | Item 1 |
| 3 | Define selected residual closure design, fixture class, oracle semantics, and proof boundary | Items 1, 2, 3, 4 |
| 4 | Specify deterministic QR fixture batch and expected-result rows | Item 2 |
| 5 | Implement QR fixture manifest/generator/expected-result updates | Item 2 |
| 6 | Design dense/external-reference QR oracle comparison semantics | Item 3 |
| 7 | Implement QR oracle comparison/report rows | Item 3 |
| 8 | Design focused QR proof owner and build/test integration | Item 4 |
| 9 | Implement focused QR proof owner | Item 4 |
| 10 | Update solver, algorithm, cookbook, and public QR wording | Item 5 |
| 11 | Update maintainer guidance and remaining residual queue | Items 5, 7 |
| 12 | Run focused QR/corpus validation and required quality gates | Item 6 |
| 13 | Publish closed QR claim, remaining non-claims, and Sprint 140 handoff | Item 7 |
| 14 | Final closeout validation summary and retrospective input | Item 7 |

## Initial Validation Expectations

| Change type | Required validation |
| --- | --- |
| Sprint 139 planning artifacts only | `git diff --check`, trailing-whitespace scan under `docs/planning/EPIC_12/SPRINT_139`, and focused Markdown link/path validation under `docs/planning/EPIC_12`. |
| Corpus manifests, expected rows, generator rows, schemas, or oracle outputs | `python3 scripts/validate_corpus_schema.py`, `python3 scripts/run_corpus_oracle.py`, TSV width checks, and report non-claim review. |
| Python oracle/reference scripts | `python3 -m py_compile <script>` plus focused command validation for touched code paths. |
| QR C tests or helpers | Focused QR test target plus source-list/CMake parity checks for new files. |
| QR `.c` or `.h` implementation/API files | Focused QR tests followed by `make format && make lint && make test`. |
| Build-system files | Relevant Make/CMake source-list or target validation plus full gates if C/H behavior changed. |
| Public or maintainer documentation | `git diff --check`, focused Markdown link/path validation, and claim-boundary scan against QR non-claims. |
| Generated reports | Capture command, source commit, platform, compiler/configuration, support tier, row meaning, freshness, and skip/defer status; keep generated outputs uncommitted unless explicitly promoted. |

## Sprint-Level Non-Claim Register

| Non-claim | Sprint 139 boundary |
| --- | --- |
| Broad QR correctness | Sprint 139 closes one selected fixture-local residual, not all QR behavior. |
| Raw QR basis parity | Closure uses residual/subspace-safe evidence, not exact basis vector equality. |
| Global rank-threshold policy | The selected lane may validate one threshold behavior but not all numerical rank choices. |
| Broad least-squares or minimum-norm behavior | The first closure lane is rank-deficient nullspace behavior unless later days explicitly add solve evidence. |
| SuiteSparse or external corpus parity | Optional external-data rows remain disabled/skipped until reviewed evidence exists. |
| LAPACK, NumPy, or SciPy parity | Existing dense helper evidence is bounded and does not imply broad external-library parity. |
| Partial-SVD correctness | Sprint 140 owns selected partial-SVD residual closure. |
| Report freshness normalization | Sprint 141 owns normalized freshness/stale diagnostics. |
| Package, ABI, platform, or performance support | QR residual closure does not promote package, platform, ABI, loader, performance, or state-of-the-art claims. |

## Day 1 Notes

- Created the Sprint 139 working-notes baseline and artifact directory.
- Re-read the Sprint 139 section of `docs/planning/EPIC_12/PROJECT_PLAN.md`.
- Reviewed Sprint 137 selected-gap context and Sprint 138 QR handoff artifacts.
- Inventoried current QR tests, helpers, external reference script, corpus
  rows, implementation surfaces, and public/maintainer documentation surfaces.
- Confirmed `qr_rank_deficient_6x4_nullspace_v1` as the initial closure
  candidate unless Day 2 finds a blocker.
- Mapped Sprint 139 Items 1-7 to day-level owners across Days 1-14.
- Recorded initial closure criteria, validation expectations, and QR
  non-claims before fixture, source, or documentation behavior changes.
- Day 1 changed planning documentation only. No `.c` or `.h` files changed, so
  the full C quality gate was not required.

## Day 2 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day2-qr-residual-reaudit.md`.
- Re-ranked QR residual candidates across rank-deficient nullspace behavior,
  duplicate-column projector behavior, rank-threshold policy, rectangular
  least-squares, minimum-norm, COLAMD/reordered QR, SuiteSparse optional data,
  and broad external-library parity.
- Selected `qr_rank_deficient_6x4_nullspace_v1` as the priority residual for
  Sprint 139 closure.
- Recorded `qr_rankdef_duplicate_5x4_nullspace_projector` as the backup
  candidate because it has existing bounded projector evidence, but it is not
  the Sprint 138 maintained corpus lane.
- Deferred rank-threshold policy, rectangular least-squares, minimum-norm,
  COLAMD/reorder behavior, SuiteSparse optional data, broad external-library
  parity, and partial-SVD subspace behavior with explicit reasons.
- Defined Day 3 design focus: choose the focused QR proof owner, mirror or
  consume the maintained fixture entries, define solver-backed observed rows,
  choose residual/subspace comparison semantics, and preserve `local_only`
  support tier unless reviewed evidence exists.
- Day 2 changed planning documentation only. No `.c` or `.h` files changed, so
  the full C quality gate was not required.

## Day 3 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day3-closure-design.md`.
- Defined the selected residual closure as solver-backed rank, nullity, and
  normalized nullspace residual behavior for
  `qr_rank_deficient_6x4_nullspace_v1`.
- Confirmed the fixture class: generated 6x4 structured sparse QR fixture,
  14 nonzeros, rank 3, nullity 1, with dependency `c3 = c0 + c1` and
  reference null vector `[-1, -1, 0, 1]`.
- Chose normalized solver-produced nullspace residual
  `||A*v_solver||_2 / ||v_solver||_2 <= 1e-10` as the primary comparison,
  avoiding raw QR basis equality.
- Selected a dedicated focused proof owner, `tests/test_qr_corpus.c`, for the
  maintained corpus lane instead of adding more evidence to the already-large
  `tests/test_qr.c`.
- Recorded the likely C helper addition
  `tf_qr_make_rankdef_6x4_nullspace_v1()` in `tests/test_qr_helpers.h` and
  the Make/CMake registration work required if the dedicated test lands.
- Recorded touched-surface validation requirements for planning docs, corpus
  rows, Python oracle scripts, QR C tests/helpers, Make/CMake files, and QR
  documentation.
- Day 3 changed planning documentation only. No `.c` or `.h` files changed, so
  the full C quality gate was not required.

## Day 4 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day4-fixture-batch-design.md`.
- Confirmed the Sprint 139 first-class fixture batch is intentionally narrow:
  `qr_rank_deficient_6x4_nullspace_v1` remains the only source-controlled QR
  fixture planned for closure.
- Preserved existing fixture and generator identifiers:
  `qr_rank_deficient_6x4_nullspace_v1` and
  `qr_rank_deficient_6x4_nullspace_generator_v1`.
- Defined expected-result row short forms for rank, nullity, and normalized
  residual without introducing ambiguous row IDs.
- Mapped success, diagnostic failure, tolerance-boundary, raw-basis variation,
  and optional SuiteSparse behavior to pass/fail/skip interpretations.
- Staged `qr_rankdef_duplicate_5x4_nullspace_projector` as a backup/pattern
  only, and deferred rank-threshold, least-squares, minimum-norm,
  COLAMD/reorder, SuiteSparse, and broad external-parity fixture candidates.
- Directed Day 5 to confirm the existing corpus rows against the design and to
  avoid adding new source-controlled QR fixture rows unless a mismatch is
  documented.
- Day 4 changed planning documentation only. No `.c` or `.h` files changed, so
  the full C quality gate was not required.

## Day 5 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day5-fixture-batch-implementation.md`.
- Confirmed the existing corpus fixture, generator, and expected-result rows
  already implement the Day 4 first-class fixture batch.
- Made no source-controlled corpus row changes because
  `qr_rank_deficient_6x4_nullspace_v1` already has matching fixture metadata,
  generator metadata, expected rows, non-claims, and validation command.
- Reproduced the generator hashes:
  `81496065f83410049f2c32556a3cb705375fe1e076112149a750489b4854f505`
  for structure and
  `2c6e0846a8a8bbe2c67786c25c029237acfccc891817ed3038b0b0e3646c36e2`
  for values.
- Ran corpus schema validation, corpus TSV width validation, and the maintained
  oracle/report command.
- Kept duplicate-column projector, near-dependent rank-threshold,
  least-squares, minimum-norm, COLAMD/reorder, SuiteSparse, and broad
  external-parity fixtures deferred.
- Day 5 changed planning documentation only. No `.c` or `.h` files changed, so
  the full C quality gate was not required.

## Day 6 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day6-oracle-comparison-design.md`.
- Chose a solver-backed QR observed-value approach: build the maintained
  fixture, run `sparse_qr_factor()`, record `sparse_qr_rank()`, record
  nullity from `sparse_qr_nullspace()`, extract one solver-produced nullspace
  vector, and compare `||A*v_solver||_2 / ||v_solver||_2 <= 1e-10`.
- Kept generated-reference rows distinct from solver-backed QR evidence:
  generated metadata remains `solver_family=unknown`, while Sprint 139 QR
  evidence should use `solver_family=qr`.
- Defined tolerance and failure semantics for factorization failure, rank
  mismatch, nullity mismatch, basis extraction failure, zero-norm basis,
  residual mismatch, malformed rows, generator mismatch, stale reports,
  optional-data skips, and unsupported platform rows.
- Confirmed optional SuiteSparse data is not required for the selected closure
  and must remain non-pass skip/defer evidence when unavailable.
- Defined Day 7 command ownership options: extend `scripts/run_corpus_oracle.py`
  behind an explicit solver-QR flag or add a dedicated QR corpus oracle command
  if compiled proof integration needs stronger separation.
- Recorded solver-backed provenance requirements for compiler, configuration,
  proof owner, fixture hash, command, support tier, claim scope, and non-claims.
- Day 6 changed planning documentation only. No `.c` or `.h` files changed, so
  the full C quality gate was not required.

## Day 7 Notes

- Updated `scripts/run_corpus_oracle.py` with an explicit
  `--include-solver-qr` option.
- Preserved default behavior: without the flag, the runner emits only the
  original generated-reference rows with `solver_family=unknown`.
- Added a temporary static-library QR probe for the opt-in path. The probe
  builds `qr_rank_deficient_6x4_nullspace_v1`, links against
  `build/libsparse_lu_ortho.a`, runs `sparse_qr_factor()`, records rank,
  nullity, and normalized residual, and then emits solver-backed QR rows.
- Added separate solver-backed row IDs:
  `qr_rank_deficient_6x4_nullspace_v1_qr_rank`,
  `qr_rank_deficient_6x4_nullspace_v1_qr_nullity`, and
  `qr_rank_deficient_6x4_nullspace_v1_qr_nullspace_residual`.
- Added compiler/proof-owner provenance and manifest row-count metadata for
  mixed generated-reference and solver-backed oracle output.
- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day7-oracle-comparison-implementation.md`.
- Validation passed for Python compile, corpus schema validation, default
  oracle generation, opt-in solver QR generation, non-repo-CWD solver QR smoke
  test, and solver QR oracle/report metadata checks.
- Day 7 changed Python and planning documentation only. No `.c` or `.h` files
  changed, so the full C quality gate was not required.

## Day 8 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day8-proof-owner-design.md`.
- Confirmed the focused proof owner should be a new dedicated executable,
  `tests/test_qr_corpus.c`, instead of adding the Sprint 139 corpus proof to
  the already-large `tests/test_qr.c`.
- Defined retained ownership: `tests/test_qr.c` keeps broad QR factorization,
  Q, rank, nullspace, economy, sparse-mode, reorder, threshold, and
  external-reference checks; `tests/test_qr_solve.c` keeps solve,
  least-squares, rank-deficient residual, and minimum-norm ownership.
- Defined Day 9 focused tests for fixture shape, rank/nullity, solver-produced
  nullspace residual, and the deterministic reference null-vector direction.
- Defined helper additions for Day 9:
  `tf_qr_make_rankdef_6x4_nullspace_v1()` and a metric-only normalized
  matvec residual helper in `tests/test_qr_helpers.h`.
- Mapped build-system touch points: add `test_qr_corpus` to `Makefile` near
  `test_qr.c`/`test_qr_solve.c` and add `add_sparse_test(test_qr_corpus)` to
  `CMakeLists.txt` near `test_qr`/`test_qr_solve`.
- Recorded Day 9 validation requirements: focused Make/CMake
  `test_qr_corpus` runs, the opt-in corpus oracle command, and
  `make format && make lint && make test` because Day 9 is expected to modify
  `.c` and `.h` files.
- Day 8 changed planning documentation only. No `.c` or `.h` files changed, so
  the full C quality gate was not required.

## Day 9 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day9-proof-owner-implementation.md`.
- Added the focused proof owner `tests/test_qr_corpus.c`.
- Added `tf_qr_make_rankdef_6x4_nullspace_v1()` and
  `tf_qr_normalized_matvec_residual()` to `tests/test_qr_helpers.h`.
- Registered `test_qr_corpus` in both `Makefile` and `CMakeLists.txt`.
- Implemented four focused tests: fixture shape/nnz, rank/nullity,
  solver-produced nullspace residual, and deterministic reference direction.
- The first focused run caught a C fixture copy mismatch from the Day 8 sketch:
  the helper produced 15 nonzeros instead of the canonical 14. Corrected the
  helper and Day 8 artifact to mirror the canonical
  `scripts/validate_corpus_schema.py` generator entries exactly.
- Focused Make proof passed:
  `make build/test_qr_corpus && ./build/test_qr_corpus`, with 4 tests, 0
  failures, 0 skips, and 83 assertions.
- Focused CMake proof passed:
  `cmake -S . -B build/qr-corpus-proof && cmake --build build/qr-corpus-proof --target test_qr_corpus && ./build/qr-corpus-proof/test_qr_corpus`.
- Opt-in solver QR oracle/report generation passed with
  `--include-solver-qr`.
- Required full gate passed: `make format && make lint && make test`,
  including the registered `test_qr_corpus`.

## Day 10 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day10-solver-documentation-update.md`.
- Updated `README.md`, `docs/solver_selection.md`, `docs/algorithm.md`,
  `docs/cookbook.md`, `examples/README.md`, `tests/corpus/README.md`, and
  `docs/maintainer_guide.md` with earned QR wording for
  `qr_rank_deficient_6x4_nullspace_v1`.
- Public wording now points to the focused proof owner
  `tests/test_qr_corpus.c` and the opt-in oracle command
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`.
- Preserved non-claims for broad QR correctness, raw basis/sign/orientation
  parity, global rank-threshold policy, broad rank-deficient solve,
  least-squares/minimum-norm broadening, SuiteSparse/LAPACK/NumPy/SciPy parity,
  platform, performance, corpus completeness, and state-of-the-art claims.
- Updated `tests/corpus/README.md` from a Sprint 139 handoff posture to an
  implemented Sprint 139 QR lane while keeping generated outputs ignored.
- Day 10 changed documentation only. The Day 9 full C gate already passed
  after C/helper/build-system edits, so no additional full C gate was required
  for Day 10.

## Day 11 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day11-maintainer-guidance-residual-queue.md`.
- Added a Sprint 139 QR corpus maintenance section to
  `docs/maintainer_guide.md` with the regeneration commands, expected
  generated outputs, stale-report signals, support-tier interpretation, and
  remaining QR residual queue.
- Expanded `tests/corpus/README.md` with the opt-in solver-backed QR output
  shape, stale or unsupported report signals, and remaining residuals that are
  not closed by Sprint 139.
- Recorded that the QR lane remains `local_only`; optional-data skip/defer rows
  are not solver pass evidence.
- Identified Sprint 140 dependencies for partial-SVD clustered/repeated
  singular-value and rank-deficient range-projector follow-through.
- Day 11 changed documentation only. No additional C/H edits were made.

## Day 12 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day12-focused-validation.md`.
- Ran corpus schema validation:
  `python3 scripts/validate_corpus_schema.py`.
- Ran focused Make QR proof:
  `make build/test_qr_corpus && ./build/test_qr_corpus`; it passed with 4
  tests, 0 failures, 0 skips, 83 assertions, solver-produced normalized
  residual about `2.220e-16`, and reference-direction residual `0.000e+00`.
- Ran opt-in solver-backed QR oracle/report generation:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`.
- Confirmed generated report metadata: 6 oracle rows,
  `solver_families=qr,unknown`, `solver_qr_row_count=3`, and all six oracle
  rows passing.
- Ran CMake target parity:
  `cmake -S . -B build/qr-corpus-proof && cmake --build build/qr-corpus-proof --target test_qr_corpus && ./build/qr-corpus-proof/test_qr_corpus`.
- Ran script compile, source-list reference, whitespace, trailing-whitespace,
  Markdown link, ignored-build-artifact, and tracked-generated-artifact checks.
- Removed the local `scripts/__pycache__/` bytecode cache produced by
  `py_compile`.
- Ran the required full C gate because Sprint 139 modified `.c` and `.h`
  files: `make format && make lint && make test`; it passed with final output
  `All tests passed.`
- Two report-inspection commands were rerun with corrected TSV field names.
  These were inspection-command mistakes, not oracle-generation or validation
  failures.

## Day 13 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day13-claim-closure-handoff.md`.
- Published the closed Sprint 139 QR claim as fixture-local evidence for
  `qr_rank_deficient_6x4_nullspace_v1`: rank `3`, nullity `1`, and
  solver-produced normalized nullspace residual `<= 1e-10`.
- Added validation-to-claim traceability from fixture identity, rank, nullity,
  residual, oracle/report metadata, Make/CMake target ownership, generated
  artifact hygiene, and the full `make format && make lint && make test` gate.
- Restated remaining QR non-claims for broad QR correctness, raw-basis parity,
  global rank-threshold policy, broad solve/least-squares/minimum-norm
  behavior, COLAMD/reordered QR, external-library parity, optional-data pass
  evidence, platform, performance, package, ABI, corpus completeness, and
  state-of-the-art claims.
- Defined Sprint 140 handoff requirements for partial-SVD: use the corpus and
  oracle/report pattern, but define partial-SVD-specific fixtures, expected
  rows, proof owner, tolerances, basis/subspace ambiguity rules, and non-claims.
- Day 13 changed planning documentation only. No additional C/H edits were
  made, and the Day 12 full quality gate remains the current code-validation
  evidence.

## Day 14 Notes

- Wrote
  `docs/planning/EPIC_12/SPRINT_139/artifacts/day14-closeout-validation-summary.md`.
- Reviewed Sprint 139 requirements against artifacts and marked Items 1-7
  complete: residual selection, fixture implementation, oracle rows, focused
  proof owner, documentation wording, validation, and claim/handoff
  publication.
- Recorded final artifact inventory, code/documentation inventory, validation
  summary, deferred QR work, Sprint 140 dependency notes, and retrospective
  inputs.
- Confirmed Day 13 and Day 14 changed planning documentation only. The latest
  code/build validation remains Day 12: `make format && make lint && make test`
  passed with final output `All tests passed.`
- Confirmed generated oracle/report outputs remain ignored build artifacts and
  should not be promoted without a later reviewed support-tier gate.
- Sprint 139 is ready for retrospective creation and pull-request packaging.
