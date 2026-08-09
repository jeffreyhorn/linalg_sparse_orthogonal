# Day 8 Public Claim Audit

## Scope

Day 8 audits public-facing documentation and selected public headers for
unsupported state-of-the-art, external-library parity, package-manager,
shared-library, dynamic ABI, Windows parity, portable performance,
generated-report freshness, and platform-overreach wording. The audit uses the
Day 2 numerical evidence inventory, Day 3 support evidence inventory, Day 5
local validation log, and Day 7 platform reconciliation as the evidence base.

Audited surfaces:

- `README.md`
- `INSTALL.md`
- `examples/README.md`
- `docs/cookbook.md`
- `docs/tutorial.md`
- `docs/solver_selection.md`
- `benchmarks/README.md`
- selected public headers:
  - `include/sparse_matrix.h`
  - `include/sparse_iterative.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`

`docs/maintainer_guide.md` was included in the scan to catch public/support
drift, but its detailed support-surface audit is assigned to Day 9.

## Unsupported-Claim Scan

Command used:

```sh
rg -n -i "state[- ]of[- ]the[- ]art|best[- ]in[- ]class|world[- ]class|outperform|faster than|superior|parity|equivalent to|compatible with (lapack|numpy|scipy|suitesparse|arpack)|shared[- ]library|dynamic ABI|ABI compatibility|package-manager|homebrew|apt|dnf|pacman|vcpkg|conan|portable performance|performance guarantee|coverage completeness|zero dead|generated report freshness|fresh generated" README.md INSTALL.md examples/README.md docs/cookbook.md docs/tutorial.md docs/solver_selection.md benchmarks/README.md docs/maintainer_guide.md include/sparse_matrix.h include/sparse_iterative.h include/sparse_qr.h include/sparse_svd.h
```

Result summary:

| Match Class | Audit Result | Action |
| --- | --- | --- |
| `state-of-the-art` | Matches appear as explicit non-claims or warnings not to infer state-of-the-art behavior. | Preserve. |
| `parity` | Matches are either reviewed CMake/platform-lane terms, internal fixture-local parity wording, or explicit non-claims. | Preserve, with Day 9 support audit to recheck maintainer-only nuance. |
| `shared-library`, `dynamic ABI`, `ABI compatibility` | Matches preserve static-first deferral and explicit non-claims. | Preserve. |
| `package-manager` and package names | Matches are install prerequisites or explicit package-manager non-claims. | Preserve. |
| `portable performance` and `performance guarantee` | Matches are explicit non-claims or benchmark/report interpretation boundaries. | Preserve. |
| `coverage completeness`, `zero dead`, generated report freshness | No public claim requiring a Day 8 fix was found; report/coverage/dead-code boundaries remain support-audit items for Day 9. | Defer detailed support-row audit to Day 9. |
| `equivalent to` | Matches in public headers describe local API equivalence, not external-library parity. | Preserve. |

No public wording fix was required on Day 8.

## Public Claim Inventory

| Public Claim | Owner Surface | Evidence | Status |
| --- | --- | --- | --- |
| The project has a short first-use route from local build to first solve, data input, solver choice, diagnostics, and static-first install/downstream use. | `README.md`; `INSTALL.md`; `examples/README.md`; `docs/cookbook.md`; `docs/solver_selection.md` | Sprint 145 adoption work; Day 5 `make examples-build`; Day 5 install checks. | Supported as adoption/documentation workflow guidance. |
| QR has a maintained fixture-local corpus proof for `qr_rank_deficient_6x4_nullspace_v1`. | `README.md`; `docs/solver_selection.md`; `docs/tutorial.md`; `include/sparse_qr.h` | Day 2 QR evidence inventory; Day 5 focused QR proof: 4 tests, 0 failures, 83 assertions, residual `2.220e-16`. | Supported only for the named fixture. |
| Partial-SVD has a maintained fixture-local corpus proof for `partial_svd_clustered_repeated_diag8x6_k3_v1`. | `README.md`; `docs/solver_selection.md`; `docs/cookbook.md`; `docs/tutorial.md`; `include/sparse_svd.h` | Day 2 partial-SVD evidence inventory; Day 5 focused partial-SVD proof: 6 tests, 0 failures, 140 assertions. | Supported only for the named fixture and checks. |
| Report indexes normalize heterogeneous report-family rows and expose freshness diagnostics. | `README.md`; `benchmarks/README.md`; `docs/cookbook.md`; `INSTALL.md` | Sprint 141 report index work; Day 5 report tests and normalization/freshness checks. | Supported as report navigation and diagnostics, not generated pass proof. |
| Runtime/backend governance and sentinels are local control and measurement surfaces. | `README.md`; `benchmarks/README.md`; `docs/cookbook.md`; `docs/solver_selection.md` | Sprint 142 governance; Day 3 support inventory; Day 7 support-tier reconciliation. | Supported as local governance/sentinel interpretation only. |
| Install/package support is static-first. | `README.md`; `INSTALL.md`; `examples/README.md` | Sprint 143 package decision; Day 5 static package deferral, Make install, and CMake install checks. | Supported for static archive package metadata and downstream proof. |
| Linux is the strongest reviewed source-of-truth platform. | `README.md`; `INSTALL.md` | Day 7 reconciliation; latest inspected green master CI run `31335415785`. | Supported for latest inspected master baseline; Sprint 146 branch-hosted proof is pending. |
| macOS has reviewed static-first install/export proof. | `README.md`; `INSTALL.md` | Day 7 reconciliation; latest inspected green master macOS run `31335415782`. | Supported for latest inspected master baseline; Sprint 146 branch-hosted proof is pending. |
| Windows is CMake-first with staged pthread/POSIX tests outside the reviewed subset. | `README.md`; `INSTALL.md` | Day 7 reconciliation; latest inspected green master Windows run `31335415791`; workflow expects `56` CTest registrations. | Supported for latest inspected master baseline; Sprint 146 branch-hosted proof is pending. |
| Selected public headers document ownership, NULL/error, QR, SVD, and iterative behavior more clearly. | `include/sparse_matrix.h`; `include/sparse_iterative.h`; `include/sparse_qr.h`; `include/sparse_svd.h` | Sprint 145 header cleanup; Day 5 found no new header changes in Sprint 146. | Supported as documentation cleanup; no new API or ABI claim. |

## Claim-To-Evidence Map

| Claim Family | Evidence Source | Non-Claim Boundary |
| --- | --- | --- |
| QR fixture-local proof | Day 2 inventory; Day 5 QR test proof; `tests/test_qr_corpus.c`; `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` | No broad QR correctness, raw QR basis parity, global rank-threshold policy, SuiteSparse parity, hosted-platform parity, performance, or state-of-the-art claim. |
| Partial-SVD fixture-local proof | Day 2 inventory; Day 5 partial-SVD test proof; `tests/test_svd_partial_corpus.c`; `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv` | No broad SVD/partial-SVD correctness, raw singular-vector identity, repeated-spectrum generality, convergence-rate, partial-result, external-library parity, performance, or state-of-the-art claim. |
| Report index/freshness | Day 3 support inventory; Day 5 report commands | No generated report freshness from source-controlled rows; no hosted CI proof from local normalized indexes; no coverage completeness or zero-dead-code claim. |
| Runtime/backend governance | Day 3 support inventory; Sprint 142 artifacts | No backend portability, optional-backend availability, portable timing, package/ABI closure, or state-of-the-art claim. |
| Static-first package | Day 3 support inventory; Day 5 package checks | No shared-library support, dynamic ABI compatibility, runtime-loader behavior, package-manager distribution, or static/shared selector support. |
| Platform tiers | Day 6 CI intake; Day 7 reconciliation | No branch-specific Sprint 146 hosted pass yet; no broad platform parity; Windows staged exclusions remain staged. |
| Adoption front door | Sprint 145 adoption work; Day 5 examples/install checks | No tutorial completion, all-header cleanup, package-manager support, Windows parity, portable performance, or state-of-the-art claim. |

## Wording Fix List

No Day 8 wording fixes were needed. All scanned public matches were either:

- explicit non-claims;
- bounded fixture-local claims tied to Day 2 and Day 5 evidence;
- static-first package boundaries tied to Day 5 evidence;
- platform support-tier statements tied to Day 7 reconciliation;
- benchmark/report guidance that treats generated rows and timing as local
  context;
- local API equivalence language unrelated to external-library parity.

## Non-Claim Preservation Summary

The following explicit non-claims remain intentionally visible:

- no unqualified state-of-the-art sparse linear algebra claim;
- no broad external-library parity claim against LAPACK, NumPy, SciPy,
  SuiteSparse, ARPACK, PETSc, Trilinos, or other ecosystems;
- no broad QR, SVD, or partial-SVD correctness claim beyond bounded fixtures;
- no raw QR basis or singular-vector identity parity claim;
- no portable performance or benchmark-superiority claim;
- no generated report freshness proof from source-controlled rows;
- no shared-library support, dynamic ABI compatibility, runtime-loader
  compatibility, package-manager support, or static/shared selector support;
- no Windows Makefile parity, Windows `pkg-config` parity, reviewed Windows
  install-validation parity, or Windows staged pthread/POSIX test closure;
- no branch-specific hosted Sprint 146 CI pass until branch/PR workflows run.

## Day 9 Handoff

Day 9 should repeat this audit from the support/maintainer side: maintainer
guide details, benchmark/report guidance, report schemas, report-family rows,
CI comments, install validation docs, and recent sprint artifacts. It should
pay special attention to maintainer-only uses of "parity" so internal evidence
language does not leak into unsupported public claims.
