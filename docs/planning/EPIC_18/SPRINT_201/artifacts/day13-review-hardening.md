# Sprint 201 Day 13 Review-Surface Hardening

## Scope

Day 13 audited the selected SVD helper extraction for unnecessary breadth,
evidence consistency, and claim safety before closeout. The reviewed surface is
limited to:

- `tests/test_svd.c`
- `tests/test_svd_helpers.h`
- `tests/test_svd_selected_helpers.h`
- `scripts/check_svd_helper_guard.sh`
- `tests/test_svd_helper_guard.py`
- `Makefile`
- Sprint 201 planning and maintainer evidence docs

No public headers, library implementation files, CMake test registration, source
manifest entries, workflow files, benchmark methodology, package metadata, or
runtime behavior contracts are included in the selected Day 13 surface.

## Diff Scope Review

| Surface | Hardening Finding | Action |
| --- | --- | --- |
| SVD proof owner | `tests/test_svd.c` keeps the selected `RUN_TEST(...)` registrations and delegates moved bodies to helper-owned wrappers. | No code change needed. |
| SVD helper header | `tests/test_svd_helpers.h` owns only shared SVD fixtures; `tests/test_svd_selected_helpers.h` owns selected rank, pseudoinverse, and dense low-rank helper implementations. | No code change needed. |
| Guard script | `scripts/check_svd_helper_guard.sh` checks required files, proof-owner registration, shared/selected helper include boundaries, selected marker movement, frozen `RUN_TEST(...)` order, selected-helper dependency includes, explicit Makefile prerequisites, and header-only non-registration. | PR #223 review follow-up strengthened guard coverage. |
| Guard regression test | `tests/test_svd_helper_guard.py` includes fixture-positive and drift-negative cases for missing includes, missing QR/SVD/vector dependency includes, duplicate moved ownership in either source/helper surface, missing and reordered registrations, stale-binary prerequisite drift, and accidental helper registration. | PR #223 review follow-up strengthened negative coverage. |
| Build registration | `Makefile` exposes `svd-helper-guard` and lists both SVD helper headers as `build/test_svd` prerequisites; `CMakeLists.txt` and `build-metadata/library_sources.txt` intentionally do not register helper headers. | PR #223 review follow-up added stale-binary prevention. |
| Maintainer/planning docs | Sprint 201 status language is selected-cluster scoped and keeps broad review-surface cleanup as future work. | Updated evidence links through Day 13. |

## Final Invariant-To-Regression Traceability

| Invariant | Evidence | Day 13 Status |
| --- | --- | --- |
| Selected moved bodies remain owned by `tests/test_svd_selected_helpers.h`, not `tests/test_svd.c` or the shared `tests/test_svd_helpers.h`. | `make svd-helper-guard`; `python3 tests/test_svd_helper_guard.py`; moved marker checks in `scripts/check_svd_helper_guard.sh`. | Covered. |
| Selected `RUN_TEST(...)` registrations stay in the `tests/test_svd.c` proof-owner binary exactly once and in frozen order. | `make svd-helper-guard`; missing-registration and reordered-registration negative fixtures in `tests/test_svd_helper_guard.py`; focused `./build/test_svd`. | Covered. |
| Selected-helper dependency includes stay explicit and local to the selected helper header. | `make svd-helper-guard`; missing `sparse_qr.h`, `sparse_svd.h`, and `sparse_vector.h` negative fixtures. | Covered. |
| Helper headers remain header-only and are not registered as standalone CMake tests or library sources. | `make svd-helper-guard`; helper CMake and library-manifest negative fixtures; Day 12 `make source-list-check`; Day 12 CMake configure check. | Covered. |
| Helper-only edits rebuild the proof-owner binary. | `Makefile` lists `tests/test_svd_helpers.h` and `tests/test_svd_selected_helpers.h` as explicit `build/test_svd` prerequisites; guard regression covers missing selected-helper prerequisite. | Covered. |
| Selected SVD behavior is preserved after extraction. | Day 10 focused `./build/test_svd`; Day 12 focused `./build/test_svd`; Day 12 full `make test`. | Covered. |
| Documentation and planning describe a no-behavior-change review-surface reduction only. | Day 11 maintainer alignment; Day 12 risk register; Day 13 claim-scope review. | Covered. |
| Public API, ABI, solver behavior, numerical tolerances, package support, platform support, performance, and state-of-the-art claims are not promoted by this sprint. | `git diff --name-only -- include src CMakeLists.txt build-metadata/library_sources.txt .github`; maintainer/project-plan/residual-queue non-claims. | Covered as non-claims. |

## Claim-Safety Review

Allowed Sprint 201 claim:

- One selected SVD review surface was reduced by moving the rank,
  pseudoinverse, and dense low-rank test bodies into
  `tests/test_svd_selected_helpers.h`, while retaining shared fixture helpers
  in `tests/test_svd_helpers.h` and retaining `tests/test_svd.c` as the
  proof-owner binary.

Explicit non-claims:

- no new SVD algorithm capability;
- no broad SVD correctness claim;
- no partial-SVD ownership change;
- no public API or ABI change;
- no library implementation behavior change;
- no numerical tolerance change;
- no performance improvement;
- no platform or package-manager support change;
- no repository-wide review-surface cleanup.

## Closeout Checklist Draft

- [x] Selected large-surface intake and ranking recorded.
- [x] Selected SVD rank/pseudoinverse/dense-low-rank cluster frozen.
- [x] Behavior-preservation invariants recorded.
- [x] Extraction design recorded.
- [x] First extraction and cohesion passes completed.
- [x] Build/registration alignment recorded.
- [x] Guard and guard regression tests added.
- [x] Focused regression evidence recorded.
- [x] Maintainer/planning docs aligned.
- [x] Integrated validation evidence recorded.
- [x] Review-hardening evidence recorded.
- [ ] Day 14 closeout and retrospective inputs prepared.

## Residual Review Surfaces

The following remain outside Sprint 201 completion claims:

- remaining large `tests/test_svd.c` clusters outside the selected rank,
  pseudoinverse, and dense low-rank group;
- `tests/test_svd_partial_corpus.c` ownership;
- broader solver, graph, direct-solver, and allocation-failure review-surface
  reductions;
- shared helper dependency tracking beyond the selected SVD helper guard;
- public documentation, package, platform, performance, release, or
  state-of-the-art support promotion.

## Day 13 Completion

Day 13 completes review-surface hardening for Sprint 201. No further code
change was needed after the audit; the remaining sprint work is Day 14 closeout
and retrospective input preparation.
