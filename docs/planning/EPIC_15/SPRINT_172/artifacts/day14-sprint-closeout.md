# Sprint 172 Day 14: Sprint Closeout And Sprint 173 Handoff

## Purpose

Day 14 closes Sprint 172 by reconciling project-plan items 172.1 through
172.6, confirming final validation, checking generated-output staging hygiene,
and preparing the Sprint 173 generated API HTML publication handoff.

## Project-Plan Item Reconciliation

| Item | Name | Result |
| --- | --- | --- |
| 172.1 | Header Family Selection | Complete. Day 3 selected `include/sparse_lu.h` as the next high-impact public header family after Day 2 inventory and prior-cleanup recheck. |
| 172.2 | Contract Cleanup | Complete. Day 5 normalized LU ownership, lifecycle, error, tolerance, workspace/allocation, progress/cancellation, solve, condition-estimate, transpose, helper, and refinement comments without intentional declaration changes. |
| 172.3 | Declaration Organization | Complete. Day 7 added concise LU workflow headings for options, factorization, solves, conditioning/transpose solves, advanced solver phases, and refinement while preserving declaration order and normalized declaration captures. |
| 172.4 | Example Alignment | Complete. Day 9 aligned `docs/tutorial.md` with the public six-argument `sparse_lu_refine(...)` signature; examples and other direct references were rechecked and did not require edits. |
| 172.5 | Mechanical Guard | Complete. Day 11 added `scripts/check_lu_header_docs_guard.sh` for LU section-heading presence, selected declaration-name presence, tutorial refinement signature drift, stale five-argument tutorial call rejection, and scoped unsupported-claim absence. |
| 172.6 | Validation | Complete. Days 5, 7, 12, 13, and 14 recorded declaration-preservation checks, full C gates, focused guard checks, claim scans, deferral guards, generated-output scans, and diff hygiene. |

## Final Sprint 172 State

Sprint 172 leaves one cleaned public header family:

- selected family: `include/sparse_lu.h`;
- declaration behavior: preserved by normalized declaration captures and full
  C quality gates;
- usage documentation: `docs/tutorial.md` now uses the public six-argument
  LU refinement signature;
- guard coverage: `scripts/check_lu_header_docs_guard.sh` protects the LU
  header headings, selected declaration names, tutorial refinement signature,
  stale tutorial signature absence, and scoped unsupported-claim boundary;
- non-claims: package-manager provider availability, shared-library support,
  dynamic ABI stability, runtime-loader behavior, Windows Makefile parity,
  Windows `pkg-config` parity, broad platform parity, portable performance,
  external-library parity, LU CSR parity, generated API HTML freshness, and
  state-of-the-art coverage remain unsupported by Sprint 172.

## Source-Controlled Deliverables

| Deliverable | Path |
| --- | --- |
| Sprint 172 plan | `docs/planning/EPIC_15/SPRINT_172/PLAN.md` |
| Sprint 172 working notes | `docs/planning/EPIC_15/SPRINT_172/WORKING_NOTES.md` |
| Daily artifacts | `docs/planning/EPIC_15/SPRINT_172/artifacts/day1-header-intake.md` through `day14-sprint-closeout.md` |
| LU declaration preservation captures | `docs/planning/EPIC_15/SPRINT_172/artifacts/day5-lu-declarations-*.txt` and `day7-lu-declarations-*.txt` |
| Cleaned LU public header | `include/sparse_lu.h` |
| Tutorial LU refinement signature update | `docs/tutorial.md` |
| LU header/docs guard | `scripts/check_lu_header_docs_guard.sh` |

## Final Validation Record

Day 14 reran final lightweight validation:

```sh
bash scripts/check_lu_header_docs_guard.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
git status --short
git diff --name-only
git ls-files --others --exclude-standard | rg '(^build/|/build/|docs/api/html|doxygen|\.o$|\.a$|\.so$|\.dylib$|\.dll$|\.exe$|compile_commands\.json|coverage|\.gcda$|\.gcno$)' || true
```

Results:

- LU header/docs guard passed;
- package-manager deferral guard passed;
- static package/shared ABI deferral guard passed;
- diff hygiene passed;
- tracked source diffs are limited to `docs/tutorial.md` and
  `include/sparse_lu.h`;
- untracked files are the planned Sprint 172 artifacts and
  `scripts/check_lu_header_docs_guard.sh`;
- no generated API HTML, Doxygen output, build output, object/archive/shared
  library artifact, executable, coverage file, or compile database was found
  by the generated-output scan.

Day 12 remains the final full C quality gate for the sprint after public header
edits:

```sh
make format && make lint && make test
```

Result: passed.

## C Quality-Gate Decision

Sprint 172 modified `include/sparse_lu.h`, so the full C quality gate was
required and was run on Day 5, Day 7, and Day 12 after the public header
changes. Day 14 changed planning documentation only, so the final closeout did
not rerun the full C gate.

## Generated-Output Staging Check

No generated API HTML, Doxygen output, build output, object/archive/shared
library artifact, executable, coverage file, compile database, provider recipe,
or package archive is part of Sprint 172. The final staging check found no
unintended generated-output artifacts outside the planned source, documentation,
script, and Sprint 172 planning files.

## Residuals

| Residual | Status |
| --- | --- |
| Exact C signature parsing in LU guard | Deferred. The guard intentionally checks declaration-name presence; exact signature proof remains covered by compiler/test gates and Day 5/Day 7 normalized declaration captures. |
| Makefile wiring for LU header/docs guard | Deferred. The standalone guard is source-controlled and reviewable; a future sprint can decide whether to wire it into `docs-check`, `quality-review`, or another focused target. |
| Generated API HTML publication | Deferred to Sprint 173. Sprint 172 improved public header inputs but did not regenerate or publish generated API HTML. |
| Broader public-header cleanup | Deferred. Sprint 172 intentionally cleaned one selected header family; future batches should repeat candidate selection before touching another family. |

## Sprint 173 Handoff

Sprint 173 generated API HTML publication work should begin from these
boundaries:

1. Treat `include/sparse_lu.h` as cleaned Sprint 172 input for generated API
   docs.
2. Re-run `bash scripts/check_lu_header_docs_guard.sh` before citing LU header
   cleanup or tutorial alignment in generated-doc publication notes.
3. Do not publish generated API HTML as fresh unless the selected Sprint 173
   freshness, generation, staging, and publication gates pass.
4. Generated API HTML work must not infer package-manager provider
   availability, shared-library support, dynamic ABI stability,
   runtime-loader support, Windows Makefile parity, Windows `pkg-config`
   parity, broad platform parity, portable performance, external-library
   parity, LU CSR parity, or state-of-the-art coverage from Sprint 172.
5. If generated API publication touches adoption, package, ABI,
   runtime-loader, or platform wording, run
   `bash scripts/package_manager_deferral_check.sh` and
   `bash scripts/static_package_deferral_check.sh`.

## Day 14 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Final Sprint 172 validation record | Complete | Day 14 lightweight checks and Day 12 full C gate are summarized above. |
| Project-plan item reconciliation | Complete | Items 172.1 through 172.6 are reconciled. |
| Generated-output staging check | Complete | No unintended generated outputs were found. |
| Sprint 173 handoff | Complete | Generated API HTML publication boundaries are listed above. |
| Day 14 sprint-closeout artifact | Complete | This file. |

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| One high-impact public header family is cleaned and validated. | Complete | `include/sparse_lu.h` was selected, cleaned, organized, declaration-checked, and covered by full C gates. |
| Documentation and guards match the selected header contract. | Complete | `docs/tutorial.md` uses the six-argument refinement call and `scripts/check_lu_header_docs_guard.sh` validates the selected LU header/docs surface. |
| Sprint 173 can begin from a clearer public API documentation boundary. | Complete | Sprint 173 handoff separates generated API HTML publication from unsupported package, ABI, platform, performance, external-parity, and state-of-the-art claims. |
