# Day 1: Generated API Intake

**Sprint:** 204 - Generated API Publication Decision  
**Theme:** Establish Sprint 204 scope, inherited generated API policy, and
current validation surfaces.  
**Time estimate:** 12 hours  
**Branch:** `sprint-204`  
**Base commit:** `cb4371b3`

## Scope

Day 1 establishes the generated API publication decision surface without
changing generated output policy. The current repository posture is:

- `docs/api_reference.md` plus checked-in public headers under `include/` are
  the source-controlled API reference path.
- `docs/api/html/` is generated Doxygen output under `docs/api/`.
- `docs/api/` is ignored by `.gitignore`.
- `make docs-check` runs Doxygen and generated page coverage.
- `make api-docs-freshness` runs generation, coverage, and local-only staging
  enforcement.
- The current policy does not publish generated API HTML as hosted docs,
  retained CI artifacts, committed output, or release evidence.

## Item Traceability

| Item | Day 1 mapping | Planned evidence |
| --- | --- | --- |
| 204.1 Product Decision | Intake keeps all policy options open and captures prior local-only decisions. | Day 3 option matrix, Day 4 acceptance gate, Day 5 product-decision artifact. |
| 204.2 Publication Or Guard Implementation | Identified current implementation surfaces: `.gitignore`, `Doxyfile`, Makefile, local-only guard, workflows. | Day 6 design, Day 7 implementation batch. |
| 204.3 Freshness And Link Checks | Identified current freshness owners and link-routing gap for later evaluation. | Day 8 freshness/coverage checks, Day 9 routing/link validation. |
| 204.4 API Routing Docs | Identified `docs/api_reference.md`, README, INSTALL, and maintainer guide as routing authorities. | Day 10 user-facing docs update. |
| 204.5 Claim Boundary Guard | Identified existing non-claim surfaces and current local-only guard wording checks. | Day 11 claim-boundary guard update. |
| 204.6 Validation | Identified required docs/API checks and escalation rules for `.c` or `.h` edits. | Day 12 integrated validation, Day 13 hardening, Day 14 closeout. |

## Current Policy Inventory

| Surface | Observed Day 1 policy |
| --- | --- |
| README | Lists `make docs`, `make docs-check`, and `make api-docs-freshness`; generated API HTML is local-only and not hosted, retained, source-controlled, or release evidence. |
| INSTALL | Support/readiness matrix marks `Local generated API HTML` as `local-only`; retained non-claim says no hosted API publication or completeness beyond checked-in public headers selected by `Doxyfile`. |
| `docs/api_reference.md` | States checked-in public headers are the source of truth; generated HTML is local-only generated output and is current only after `make api-docs-freshness`. |
| `docs/maintainer_guide.md` | Maintainer guidance preserves Sprint 179 local-only generated API decision and keeps `R186-HOSTED-API` open until a later product decision selects hosted, retained, or committed output. |
| `.gitignore` | Ignores `docs/api/`; `git check-ignore -v docs/api docs/api/html docs/api/html/index.html` maps all three paths to `.gitignore:44:docs/api/`. |
| `Doxyfile` | Generates output below `docs/api`. |
| Makefile | Exposes `docs`, `api-docs-coverage`, `api-docs-local-only`, `docs-check`, `api-docs-validate`, and `api-docs-freshness`. |
| `scripts/check_api_docs_local_only.sh` | Guards ignore rules, no tracked/staged/non-ignored generated API files, Doxyfile local-only settings, product wording, and absence of workflow references to generated API output paths. |
| `scripts/check_api_docs_coverage.py` | Checks generated Doxygen HTML page coverage for checked-in public headers. |
| Workflows | Current workflow surface is `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and `.github/workflows/windows-ci.yml`; the local-only guard rejects generated API path references while local-only status is active. |

## Public Header Source Set

The current Doxygen source set is the 18 checked-in public headers under
`include/`:

| Header |
| --- |
| `include/sparse_analysis.h` |
| `include/sparse_bidiag.h` |
| `include/sparse_cholesky.h` |
| `include/sparse_csr.h` |
| `include/sparse_dense.h` |
| `include/sparse_eigs.h` |
| `include/sparse_ic.h` |
| `include/sparse_ilu.h` |
| `include/sparse_iterative.h` |
| `include/sparse_ldlt.h` |
| `include/sparse_lu.h` |
| `include/sparse_lu_csr.h` |
| `include/sparse_matrix.h` |
| `include/sparse_qr.h` |
| `include/sparse_reorder.h` |
| `include/sparse_svd.h` |
| `include/sparse_types.h` |
| `include/sparse_vector.h` |

`sparse_version.h` remains a generated installed header owned by `VERSION`,
`include/sparse_version.h.in`, and install/package validation. It is not part
of the current checked-in Doxygen input set.

## Prior Generated API Decisions

| Prior sprint | Relevant evidence |
| --- | --- |
| Sprint 158 | Closed generated API HTML as local-only, ignored, validated by `make docs-check`, and not committed or hosted. |
| Sprint 179 | Selected strengthened local-only generated API HTML status and made `make api-docs-freshness` the combined generation, coverage, and local-only staging proof. |
| Sprint 186 | Preserved generated API evidence as local freshness/staging proof and kept `R186-HOSTED-API` open for any future hosted, retained, or committed publication decision. |
| Epic 18 closeout | Retained generated API publication policy as a future closure candidate while confirming no hosted API publication claim was present. |

## Initial Risk Register

| Risk | Why it matters | Day 1 mitigation |
| --- | --- | --- |
| Accidental generated HTML staging | Generated files could become stale review noise or look like source-controlled API truth. | Keep `docs/api/` ignore and staging guard in the validation matrix. |
| Hosted docs without freshness | A URL or retained artifact could outlive the branch evidence that generated it. | Require Day 4 acceptance criteria before any workflow or artifact change. |
| Claim drift | Generated docs can be misread as ABI, package, platform, or completeness proof. | Keep non-goals and claim-boundary surfaces explicit from Day 1. |
| Link rot | User-facing docs may point to a generated path that is not present or current. | Reserve Day 9 for routing and link validation. |
| Weak workflow publication guard | Current local-only workflow check is string-based. | Evaluate structured workflow validation if the selected policy depends on workflow semantics. |

## Validation Matrix Seed

| Command or check | Purpose | Day 1 disposition |
| --- | --- | --- |
| `make docs-check` | Generate Doxygen HTML and verify public-header page coverage. | Planned for Day 2 baseline. |
| `make api-docs-freshness` | Run Doxygen, page coverage, and local-only staging guard. | Planned for Day 2 baseline. |
| `bash scripts/check_api_docs_local_only.sh` | Verify local-only generated-output policy directly. | Planned after guard or wording changes. |
| `python3 scripts/check_api_docs_coverage.py` | Verify generated page coverage logic directly. | Planned if coverage logic changes. |
| Link validation | Verify API routing surfaces do not point to unavailable generated output. | Candidate for Day 9. |
| `make format && make lint && make test` | Full C quality gate if `.c` or `.h` files change. | Not required on Day 1; no `.c` or `.h` files changed. |

## Non-Goals Recorded

Sprint 204 does not claim broad API completeness, dynamic ABI compatibility,
shared-library support, package-manager distribution, Homebrew/core readiness,
bottles, Linuxbrew, public tap support, broad platform parity, external-library
parity, portable performance, solver behavior changes, release evidence from
generated API HTML, hosted generated API publication, or committed generated
HTML unless the later Sprint 204 product decision explicitly selects and
validates those paths.

## Completion Criteria Review

| Day 1 criterion | Status |
| --- | --- |
| Every Sprint 204 item has an initial evidence path or artifact category. | Complete; see item traceability above and `WORKING_NOTES.md`. |
| Current local-only generated API semantics are understood before any policy change. | Complete; README, INSTALL, API reference, maintainer guide, `.gitignore`, Doxyfile, Makefile, and guard scripts were reviewed. |
| Unsupported API, ABI, package, and publication claims remain explicitly out of scope. | Complete; non-goals and claim-boundary risks are recorded. |

## Changed Surfaces

- Added `docs/planning/EPIC_18/SPRINT_204/WORKING_NOTES.md`.
- Added `docs/planning/EPIC_18/SPRINT_204/artifacts/day1-generated-api-intake.md`.

No source, public header, workflow, Makefile, Doxyfile, generated output, or
user-facing documentation behavior changed on Day 1.
