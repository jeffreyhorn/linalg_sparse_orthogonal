# Sprint 172 Day 2: Public Header Candidate Inventory

## Purpose

Day 2 inventories the remaining public header families before Sprint 172
selects a cleanup target. This artifact is intentionally decision-preparatory:
it ranks candidate families and documents drift risk, but it does not select a
header or edit any `.h` file.

## Inventory Inputs

| Input | Use |
| --- | --- |
| `include/*.h` | Current checked-in public header source set. |
| `docs/planning/EPIC_14/SPRINT_164/artifacts/day2-header-selection.md` | Prior cleanup selection and deferred queue. |
| `docs/planning/EPIC_14/SPRINT_158/artifacts/day3-public-header-coverage-map.md` | Confirms the 18 checked-in public headers and generated `sparse_version.h` treatment. |
| `docs/planning/EPIC_15/SPRINT_167/RETROSPECTIVE.md` | Claim-gate and evidence-ledger boundaries. |
| `docs/planning/EPIC_15/SPRINT_171/RETROSPECTIVE.md` | Package-manager deferral and adoption-boundary handoff. |
| README/API/tutorial/cookbook/maintainer references | Adoption visibility and documentation-drift signal. |

## Current Public Header Metrics

The risk-term count is a coarse local signal from ownership, lifecycle, error,
tolerance, workspace, threading, callback, allocation, backend, package, ABI,
performance, and related wording. It is not a correctness metric.

| Header | Lines | Risk-term hits | Day 2 disposition |
| --- | ---: | ---: | --- |
| `include/sparse_iterative.h` | 828 | 320 | Recheck only. High complexity, but it was already part of the prior public-header cleanup batch. |
| `include/sparse_eigs.h` | 648 | 213 | Recheck only. High complexity, but it was already part of the prior public-header cleanup batch. |
| `include/sparse_matrix.h` | 609 | 107 | Recheck only. Core lifecycle surface, but it was already part of the prior public-header cleanup batch. |
| `include/sparse_analysis.h` | 488 | 54 | Top candidate. Direct repeated-run lifecycle, analysis/refactor ownership, reorder controls, and docs visibility make it a strong Batch 2 family. |
| `include/sparse_qr.h` | 373 | 63 | Top candidate. QR rank, least-squares, minimum-norm, residual, and corpus/comparison wording are adoption-sensitive. |
| `include/sparse_lu.h` | 360 | 58 | Top candidate. General direct-solver first-use surface with factor/solve/refine/error semantics. |
| `include/sparse_types.h` | 324 | 56 | Candidate with high caution. Shared scalar/index/error/macro surface has ABI and declaration-drift risk. |
| `include/sparse_lu_csr.h` | 322 | 64 | Candidate. Specialized direct/CSR working-format API with historical Doxygen warning ownership. |
| `include/sparse_ldlt.h` | 315 | 86 | Top candidate. Symmetric-indefinite lifecycle, backend selection, telemetry, and solve/refine semantics remain high value. |
| `include/sparse_svd.h` | 243 | 66 | Candidate. Partial-SVD and low-rank wording is evidence-sensitive, but recent corpus work raises overclaim risk. |
| `include/sparse_cholesky.h` | 227 | 66 | Candidate/defer. Direct SPD surface is important, but it has prior cleanup history and narrower current drift than analysis/LU/LDLT. |
| `include/sparse_ilu.h` | 200 | 42 | Medium candidate. Preconditioner setup/failure/lifecycle language matters but is narrower. |
| `include/sparse_dense.h` | 197 | 16 | Lower candidate. Dense helper surface has lower current adoption-drift risk. |
| `include/sparse_reorder.h` | 186 | 31 | Medium candidate. Reorder language should likely align with `sparse_analysis.h` rather than stand alone. |
| `include/sparse_csr.h` | 161 | 31 | Lower candidate. Compressed-storage helper cleanup should follow matrix/direct decisions. |
| `include/sparse_ic.h` | 121 | 21 | Lower candidate. Smaller preconditioner surface and recent header cleanup history. |
| `include/sparse_bidiag.h` | 72 | 9 | Lower candidate. Narrow helper surface. |
| `include/sparse_vector.h` | 70 | 11 | Lower candidate. Narrow helper surface. |

The generated installed `sparse_version.h` remains outside the checked-in
header cleanup target set. Its template, `include/sparse_version.h.in`, is
package/install owned and should not become a Day 3 cleanup target unless the
sprint explicitly changes the generated-header policy.

## Candidate Ranking For Day 3

| Rank | Candidate family | Rationale | Main risk to control |
| ---: | --- | --- | --- |
| 1 | Direct analysis family centered on `include/sparse_analysis.h` | It is a large repeated-run direct-solver lifecycle surface, appears in adoption docs, and was explicitly deferred for a direct-solver batch. | Avoid implying broader direct-solver superiority, package support, ABI stability, or platform parity. |
| 2 | LDLT family centered on `include/sparse_ldlt.h` | It has the highest remaining risk-term density among likely targets and exposes backend selection, telemetry, symmetric-indefinite semantics, and lifecycle behavior. | Keep backend wording descriptive, not a performance or portability claim. |
| 3 | General LU family centered on `include/sparse_lu.h` | It is the most familiar direct-solver first-use API and carries factor/solve/refine/tolerance/error semantics. | Preserve in-place factorization semantics and avoid declaration or behavior drift. |
| 4 | QR family centered on `include/sparse_qr.h` | It has strong docs visibility and sensitive least-squares, rank, nullspace, and minimum-norm language. | Keep corpus/comparison and residual evidence local; do not widen claims. |
| 5 | Shared type family centered on `include/sparse_types.h` | It affects error codes, scalar/index contracts, version macros, callbacks, and cross-header vocabulary. | High macro/typedef/enum ABI risk; likely requires a very narrow comment-only scope. |
| 6 | LU CSR family centered on `include/sparse_lu_csr.h` | It owns specialized CSR direct working-format language and prior Doxygen warning ownership. | Avoid exposing private implementation details as public support promises. |

## Documentation Drift Notes

- README and first-use docs continue to surface `sparse_lu.h`,
  `sparse_analysis.h`, `sparse_ldlt.h`, `sparse_qr.h`, `sparse_svd.h`,
  `sparse_lu_csr.h`, and `sparse_types.h` prominently.
- `docs/maintainer_guide.md` carries direct-family ownership and private
  implementation-boundary notes that should not migrate into public headers.
- Prior generated API coverage confirmed all 18 checked-in public headers have
  generated reference/source pages; Day 2 therefore treats missing publication
  as out of scope and focuses on source comment coherence.
- Prior warning ownership included `include/sparse_lu_csr.h` and
  `include/sparse_types.h`; selecting either family should include a Day 4
  warning triage pass.
- Recent tutorial/API coherence work already touched several public header
  narratives. Day 3 should avoid selecting a family only because it is visible;
  the selected family should have a clear bounded cleanup value.

## Declaration-Organization Risk List

| Risk area | Candidate headers | Day 3 control |
| --- | --- | --- |
| Repeated-run lifecycle and factor ownership | `sparse_analysis.h`, `sparse_ldlt.h`, `sparse_lu.h` | Require a declaration baseline before comment cleanup. |
| Backend selector and telemetry wording | `sparse_ldlt.h`, `sparse_analysis.h`, `sparse_qr.h` | Describe observed API semantics only; avoid performance superiority language. |
| Tolerance/residual/rank interpretation | `sparse_qr.h`, `sparse_svd.h`, `sparse_lu.h`, `sparse_ldlt.h` | Keep evidence local to maintained tests and docs. |
| Shared typedef/macro/error-code vocabulary | `sparse_types.h` | Avoid macro, enum-value, typedef, include-guard, or installed-header changes. |
| Specialized working-format ownership | `sparse_lu_csr.h`, `sparse_csr.h` | Separate public API contract from private storage implementation details. |
| Generated installed version metadata | `sparse_version.h.in` | Keep generated-header policy separate from checked-in header cleanup. |

## Day 3 Handoff

Day 3 should select exactly one public header family from the ranked candidate
pool. The strongest evidence-supported pool is:

1. `include/sparse_analysis.h`
2. `include/sparse_ldlt.h`
3. `include/sparse_lu.h`
4. `include/sparse_qr.h`
5. `include/sparse_types.h`
6. `include/sparse_lu_csr.h`

The prior cleanup batch already covered `include/sparse_iterative.h`,
`include/sparse_eigs.h`, and `include/sparse_matrix.h`; Sprint 172 should not
reselect those unless Day 3 finds a regression that outweighs the remaining
direct-solver candidates.

## Validation Notes

Day 2 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 2.

## Completion Check

- All checked-in public headers are visible in a comparable matrix.
- Candidate families are ranked by adoption risk, API complexity, prior cleanup
  history, and documentation-drift value.
- Declaration and claim risks are explicit before Day 3 selection.
- No public header edits were made before selecting a cleanup target.
