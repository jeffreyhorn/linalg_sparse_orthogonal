# Sprint 145 Day 13 Adoption Claim Map And Residual Debt

## Purpose

Day 13 publishes the Sprint 145 adoption claim map, residual documentation
debt, public-header cleanup status, support-tier consistency check, and Sprint
146 closeout handoff draft. It uses the Day 12 validation gate as the latest
proof owner for cumulative checks.

## Adoption Claim Map

| Adoption-facing claim | Public surface | Source evidence | Validation owner |
| --- | --- | --- | --- |
| First-use readers have a concrete route from local build to first solve, diagnostics, install/downstream consumption, and advanced controls. | `README.md`, `examples/README.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `INSTALL.md` | Day 3 workflow design, Day 5 example/cookbook batch, Day 6 README restructure, Day 7 INSTALL restructure, Day 8 solver front door | Day 12 `make examples-build`, install/downstream checks, docs scans, and `git diff --check` |
| Maintained examples are a runnable teaching ladder, not broad numerical or platform proof. | `examples/README.md`, `docs/cookbook.md`, `README.md` | Day 4 example/cookbook design, Day 5 batch, Day 12 example build | `make examples-build`; install scripts execute maintained downstream examples |
| Static-first install/export is the maintained package contract. | `INSTALL.md`, `README.md`, `examples/README.md`, `tests/corpus/manifests/report_families.tsv` | Sprint 143 package/ABI decision, Sprint 144 platform support tiers, Day 7 INSTALL restructure | `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, package report-family checks |
| Shared-library packaging, dynamic ABI compatibility, runtime-loader behavior, package-manager distribution, and static/shared selectors remain non-claims. | `INSTALL.md`, `README.md`, `examples/README.md`, report-family metadata | Sprint 143 static-first decision, Day 7 INSTALL restructure, Day 12 validation gate | install/downstream checks plus unsupported-package-claim scans from Days 7 and 12 |
| Linux, macOS, and Windows support tiers are distinct and should not be read as broad cross-platform parity. | `INSTALL.md`, `README.md`, report-family metadata | Sprint 144 platform support-tier closure, Day 7 INSTALL restructure, Day 11 coherence pass | Day 11 support-tier scan, Day 12 local validation with hosted-CI skip rationale |
| Solver selection starts from matrix/problem shape, then escalates through diagnostics before changing backend, preconditioner, tolerance, or benchmark settings. | `docs/solver_selection.md`, `README.md`, `docs/cookbook.md`, `examples/README.md` | Day 8 solver front door, Day 11 cross-surface coherence pass | Day 8 diagnostics scans, Day 11 routing scans, Day 12 full validation |
| QR adoption wording is bounded to maintained fixture-local evidence and API-local diagnostics. | `docs/solver_selection.md`, `docs/cookbook.md`, `README.md`, `include/sparse_qr.h`, `examples/README.md` | Sprint 139 QR evidence, Day 8 solver front door, Day 10 public-header cleanup | Day 10 header quality gate, Day 12 full C/header gate |
| Partial-SVD adoption wording is bounded to maintained fixture-local residual/convergence evidence and does not imply broad SVD parity. | `docs/cookbook.md`, `README.md`, `include/sparse_svd.h`, `examples/README.md` | Sprint 140 partial-SVD evidence, Day 10 public-header cleanup | Day 10 header quality gate, Day 12 full C/header gate |
| Report indexes and report-family rows are navigation, ownership, and freshness aids; source-controlled rows do not become generated pass evidence. | `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, `tests/corpus/schemas/report_index_fields.md`, `tests/corpus/manifests/report_families.tsv` | Sprint 141 report normalization, Day 11 runtime/backend row repair | schema validation, normalization checks, freshness checks from Days 11 and 12 |
| Runtime/backend controls and benchmark/sentinel rows are local diagnostic and maintainer controls, not portable performance or state-of-the-art claims. | `README.md`, `docs/solver_selection.md`, `docs/cookbook.md`, `benchmarks/README.md`, report-family metadata | Sprint 142 backend governance, Day 8 solver front door, Day 11 coherence pass | Day 11 unsupported-claim scan, Day 12 report and full validation gates |
| Public headers now emphasize API-local contracts instead of maintainer-history prose. | `include/sparse_matrix.h`, `include/sparse_iterative.h`, `include/sparse_qr.h`, `include/sparse_svd.h` | Day 9 header-cleanup design, Day 10 header cleanup batch | declaration-preservation scan plus `make format && make lint && make test` on Days 10 and 12 |

## Residual Documentation Debt Ledger

| Residual debt | Owner | Why outside Sprint 145 | Promotion gate |
| --- | --- | --- | --- |
| Full tutorial alignment with the new first-use ladder. | Documentation maintainer | Sprint 145 prioritized README, INSTALL, examples, cookbook, solver-selection, and selected public headers. | Tutorial updated and scanned against the Day 13 claim map. |
| Broader public-header cleanup beyond the four selected adoption headers. | API maintainer | Day 10 intentionally touched only matrix construction, iterative diagnostics, QR, and SVD headers to avoid unrelated API-contract churn. | Header-specific design review plus declaration-preservation scan and full C quality gate. |
| Hosted CI reconciliation after final PR runs. | CI maintainer | Day 12 was local validation only. Hosted Linux, macOS, and Windows evidence lives in CI logs. | Sprint 146 final evidence inventory records latest hosted run ids, statuses, and support-tier implications. |
| Generated benchmark, coverage, dead-code, and sentinel report refresh. | Report and benchmark maintainers | Sprint 145 changed adoption wording and source-controlled report metadata, not measurement rows. | Regenerate the relevant reports under maintained commands and normalize/freshness-check the resulting indexes. |
| Windows staged parity closure. | Platform maintainer | Sprint 145 preserved Sprint 144 support tiers and did not implement Windows Makefile, `pkg-config`, POSIX temp-file, or pthread parity. | Windows staged blockers removed with reviewed CMake/Make/install validation lanes promoted explicitly. |
| Shared-library and dynamic ABI productization. | Package maintainer | Sprint 145 deliberately preserved the Sprint 143 static-first contract. | Build rules, package metadata, installed-consumer proof, runtime-loader validation, and ABI policy exist before public claims change. |
| State-of-the-art competitive positioning. | Epic 12 closeout owner | Adoption simplification does not prove competitive numerical breadth, portable performance, or external-library parity. | Sprint 146 claim audit compares earned evidence against the original Epic 12 review gaps and records non-claims. |

## Public Header Cleanup Status

Completed headers:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`

Preserved contract boundaries:

- no signature, typedef, enum, macro, or struct-field changes were intended;
- caller ownership and NULL/error semantics remain public-header owned;
- QR and partial-SVD evidence remains bounded;
- backend/runtime controls remain local workflow controls, not ABI, package,
  platform, or portable performance claims.

Deferred header surfaces:

- `include/sparse_analysis.h`
- `include/sparse_cholesky.h`
- `include/sparse_dense.h`
- `include/sparse_eigs.h`
- `include/sparse_ldlt.h`
- `include/sparse_lu.h`
- `include/sparse_reorder.h`
- other headers not selected by the Day 9 adoption-friction ranking

Each deferred header needs its own cleanup design before comment changes, then
the same declaration-preservation scan and full C quality gate used in Sprint
145.

## Support-Tier Consistency Check

The current public surfaces agree on these boundaries:

- README owns the shortest first-use route and links to deeper evidence
  owners.
- INSTALL owns static-first install/downstream behavior and platform support
  tiers.
- `examples/README.md` owns runnable teaching workflows.
- `docs/cookbook.md` owns data-first recipes and diagnostic routing.
- `docs/solver_selection.md` owns solver choice, QR/partial-SVD evidence
  boundaries, and advanced-control escalation.
- benchmark/report docs and report-family metadata own report interpretation,
  freshness, and non-claim boundaries.
- public headers own API-local contracts.

No public surface should be read as claiming:

- broad QR, SVD, partial-SVD, LAPACK, NumPy, SciPy, or SuiteSparse parity;
- portable performance, benchmark superiority, or state-of-the-art status;
- shared-library packaging or dynamic ABI compatibility;
- package-manager distribution;
- Windows Makefile or Windows `pkg-config` parity;
- source-controlled report rows as generated pass evidence.

## Sprint 146 Closeout Handoff Draft

Sprint 146 should start from this adoption map and:

1. Build the final Epic 12 evidence inventory from Sprints 137-145.
2. Reconcile final hosted CI results with the support tiers preserved in
   Sprint 145 public docs.
3. Run the final public claim and non-claim audit against README, INSTALL,
   examples, cookbook, solver-selection, benchmark/report docs, public
   headers, and report-family metadata.
4. Publish the remaining residual queue with explicit owners, blockers, and
   promotion gates.
5. Decide whether any state-of-the-art claim has actually been earned; absent
   direct comparative evidence, keep the claim as a non-claim.

## Validation

- Day 12 `make format && make lint && make test` remains the latest full
  C/header quality gate for the cumulative public-header diff.
- Day 12 report schema, normalization, freshness, example build, install, and
  CMake install checks all passed.
- `git diff --check` passed after the Day 13 planning-doc edits.
- Day 13 trailing-whitespace scan passed for the working notes and claim-map
  artifact.
- Day 13 non-claim scan found only explicit non-claims, residual-debt entries,
  or Sprint 146 audit handoff language.
- Day 13 adds planning documentation only and does not require another C/header
  quality gate.
