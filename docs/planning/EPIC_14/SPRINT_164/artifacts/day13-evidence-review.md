# Sprint 164 Day 13: Evidence Review And Sprint 165 Handoff

## Purpose

Day 13 reviewed the Sprint 164 evidence chain end to end so the public-header
cleanup remains reviewable and Sprint 165 can start from a clear static-first
package/API boundary.

## Changed Surface Trace

Tracked source and public-documentation changes currently cover:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `README.md`
- `docs/solver_selection.md`
- `docs/tutorial.md`

Sprint planning evidence is under:

- `docs/planning/EPIC_14/SPRINT_164/WORKING_NOTES.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/`

## Declaration-To-Cleanup Trace

| Cleanup Area | Evidence |
| --- | --- |
| Header selection | Day 2 selected `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h` as the bounded cleanup batch. |
| Declaration baseline | Day 4 captured the before-state normalized declaration checksum. |
| Ownership/error/backend cleanup | Days 5-8 edited comments and related docs while repeatedly checking declaration preservation. |
| Re-capture | Day 10 re-ran the normalized declaration capture and produced an identical checksum. |
| Post-format validation | Day 12 re-ran declaration capture after `make format`; the checksum still matched. |

Stable declaration checksum:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

The before/after and before/post-format declaration diffs produced no output.
Result: Sprint 164 header cleanup remains declaration-preserving.

## Documentation Coherence Trace

| Surface | Day 13 Review Result |
| --- | --- |
| `README.md` | Eigensolver summary now describes the public AUTO/Lanczos/thick-restart/LOBPCG surface and `result.backend_used` telemetry. |
| `docs/tutorial.md` | Eigensolver example now uses the actual public result type `sparse_eigs_t`. |
| `docs/solver_selection.md` | Eigensolver AUTO routing is framed as policy rather than backend superiority and links to tutorial/cookbook workflow sections. |
| `docs/cookbook.md` | Already starts symmetric eigensolver users with AUTO and routes exact details to `sparse_eigs.h`. |
| `docs/api_reference.md` | Already keeps checked-in headers as exact declaration source of truth and generated HTML as local-only output. |
| `docs/maintainer_guide.md` | Already preserves public-header cleanup and generated-reference policy boundaries. |

Day 13 stale-reference scan found no remaining hits for:

- `sparse_eigs_result_t`
- `sparse_eigs_sym(A, k, &opts, &result)` described as only grow-m Lanczos
- stale "via Lanczos (growing-m)" README/API wording

## Generated Reference Trace

Day 9 and Day 12 both ran `make docs-check`.

Latest generated-reference result:

```text
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
```

Day 13 confirmed:

- generated API HTML remains local-only validation output;
- `docs/api/html/` has no tracked churn;
- local declaration evidence under `build/sprint164/declarations/` has no
  tracked churn;
- generated `sparse_version.h` remains an install/version policy row rather
  than a Doxygen coverage requirement.

## Non-Claim Trace

Day 13 reviewed public docs, selected headers, API reference policy, and
maintainer guidance for unsupported wording.

Retained non-claims remain explicit for:

- dynamic ABI compatibility;
- shared-library support;
- runtime-loader behavior;
- package-manager distribution;
- broad Windows Makefile or Windows `pkg-config` parity;
- external-library parity;
- portable runtime or performance guarantees;
- hosted generated documentation publication;
- source-controlled generated HTML;
- backend superiority;
- state-of-the-art coverage.

Scan hits were limited to explicit disclaimers, local dispatch-policy
boundaries, and maintainer policy guidance.

## Sprint 165 Static-First Package/API Handoff

Sprint 165 is planned as **Static-First Package Boundary Hardening**. Its goal
is to harden the static-first package boundary so shared-library, dynamic ABI,
runtime-loader, and package-manager non-claims cannot drift.

Sprint 164 hands off the following ready inputs:

- public-header/API cleanup status is known and declaration-preserving;
- selected headers no longer contain stale backend-superiority or portable
  performance wording from the cleanup batch;
- API reference generated HTML policy is confirmed local-only and complete for
  checked-in headers;
- user-facing docs route exact declarations to checked-in headers;
- package, ABI, shared-library, runtime-loader, and package-manager wording
  remains bounded to explicit non-claims.

Recommended Sprint 165 starting points:

- audit CMake package metadata, `sparse.pc`, install scripts, and CI checks for
  unsupported package or ABI wording;
- review public structs, version docs, install docs, and package metadata for
  accidental ABI promises;
- align README, INSTALL, maintainer guide, CMake docs, and package comments
  with the static-first contract;
- preserve Sprint 164's declaration-preserving header cleanup as a prerequisite
  rather than reopening public signatures.

## Residual Queue

Out of scope for Sprint 164 and suitable for later work:

- broader non-selected-header public-comment cleanup (`sparse_ldlt.h`,
  `sparse_qr.h`, `sparse_svd.h`, preconditioner headers);
- table-wide README/API index reshaping;
- generated API HTML publication beyond local ignored output;
- package/ABI product changes or shared-library support;
- backend threshold retuning or new performance claims;
- exhaustive tutorial expansion for every option/result field;
- maintained helper script for declaration-preservation capture.

## Validation

- reviewed current diff and changed-file list;
- reviewed Day 9, Day 10, Day 11, and Day 12 evidence artifacts;
- scanned for stale eigensolver type/backend wording;
- scanned for unsupported package, ABI, runtime-loader, platform, external
  parity, portable-performance, hosted-docs, backend-superiority, and
  state-of-the-art wording;
- checked generated/local evidence status:
  `git status --short -- docs/api/html build/sprint164/declarations`;
- `git diff --check`.

## Outcome

Sprint 164 is reviewable end to end, positive API wording is bounded by
unchanged declarations and local validation evidence, and the Sprint 165
static-first package/API handoff is ready.
