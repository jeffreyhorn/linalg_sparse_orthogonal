# Sprint 222 Day 1: Selected Fixture And Evidence Inventory

## Scope and authority

This is the Day 1 input freeze for item 222.1. The selected cell is the Sprint 215 correctness-only, one-RHS SPD solve. The source revision inspected was `dab7943c218e997cda6898c9cdd8d3572a6029c9` on `sprint-222`. The Sprint 222 plan is currently untracked; no solver execution or observed comparison is represented here.

The frozen contracts are Sprint 215's `day3-methodology-inputs.md`, `day3-baseline-selection-decision.md`, `day4-correctness-and-tolerance-contract.md`, and `day5-external-comparison-methodology.md`, plus Sprint 221's `day14-integrated-validation-and-sprint222-handoff.md`. The machine-readable record authority is `tests/corpus/schemas/external_baseline_evidence_v1.json` (`$id=external_baseline_evidence_v1`).

## Selected problem and byte inventory

| Component | Frozen value | Verification |
| --- | --- | --- |
| Matrix source | `tests/data/symmetric_4.mtx`; 114 bytes; SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335` | Independently read raw bytes and hashed them; matches Sprint 215 and v1 schema. No copy or rewrite was made. |
| Matrix Market declaration | `matrix coordinate real symmetric`; 4 rows, 4 columns, 7 stored entries | Parsed the header and all seven one-based, finite, unique lower-triangle coordinates. |
| Expanded matrix | `[[10,1,2,0],[1,12,1,0],[2,1,14,0],[0,0,0,16]]`; 10 nonzeros | Mirrored each off-diagonal once, retained each diagonal once, checked symmetry and strict positive diagonal dominance (minimum margin 7). |
| Oracle and RHS | `x_ref=[1,1,1,1]`; `b=A*x_ref=[13,14,17,16]`, float64, one RHS | Recomputed from the expanded matrix. The canonical values are the authority; these optional diagnostic hashes encode four little-endian IEEE-754 binary64 values in vector order: RHS `c7f4c5dc3a5a222d2810c5b77d0ccac1dc2db388bd1100cf618d4203fbe331ef`, oracle `c914e8188e43fff1c96e25283e15b252af0d9f39b469f2d1518915802c756d18`. |
| Lower CSC for CHOLMOD | `stype=-1`, `p=[0,3,5,6,7]`, `i=[0,1,2,1,2,2,3]`, `x=[10,1,2,12,1,14,16]` | Independently sorted lower entries by column and row. Arrays match `tests/test_cholmod_pilot_csc.c`; `cholmod_pilot_csc_matches_input` reconstructs and checks the full matrix. This is a source-level conversion check, not an external solver run. |

The original matrix remains the residual and RHS reference. The project loader expands symmetry, then the project factor path must use a fresh copy. The CHOLMOD side uses only the lower CSC view and must compare its reconstruction against the untouched expanded matrix before factoring. Any changed matrix byte, duplicate, nonfinite coefficient, shape or conversion mismatch is `invalid-input`, not a repaired version of this cell.

## Solver and numerical controls

| Side | Required path and controls | Still to prove at runtime |
| --- | --- | --- |
| Project | `sparse_load_mm`; `sparse_cholesky_factor_opts` with `SPARSE_REORDER_NONE` and `SPARSE_CHOL_BACKEND_LINKED_LIST`; `sparse_cholesky_solve`; float64, `SPARSE_IDX_BITS=32`, `used_csc_path=0` | Exact build, binary/archive hash, all statuses, effective backend and finite solution. |
| Baseline | SuiteSparse `v7.12.2`, source commit `42151688813c45846a597edcb601435a0e38f3dd`, CHOLMOD 5.3.4; `cholmod_analyze`, `cholmod_factorize`, `cholmod_solve(CHOLMOD_A)`; real double, int32, natural ordering, postorder off, simplicial factor | Loaded library and dependency hashes, build/module options, actual ABI, effective method/factor controls and all phase statuses. The old Homebrew discovery hash is only a candidate identity. |

The Day 4 rules apply independently to each solver: all phase statuses must succeed; outputs and intermediates must be finite; residual `R=||Ax-b||_inf` must meet `R <= T*max(1,||b||_inf)+T*(||A||_inf*||x||_inf+||b||_inf)`; oracle error `E=||x-x_ref||_inf` must meet `E <= T*K*max(1,||x_ref||_inf)+T*K*||x_ref||_inf`, with `T=32*4*2^-52` and `K=17/7`. The cross-solver difference is checked only after both independent gates. No timing is part of this Day 1 work.

## V1 field-to-evidence map and promotion gates

| V1 field group | Required source before `correctness_observed` | Day 1 status / owner |
| --- | --- | --- |
| `schema_version`, `selection_id`, `record_id`, `run_id`, `record_kind`, `evidence_origin`, `evidence_status`, `status_reason`, `cell_id`, `cell_id_reason` | V1 schema and a unique generated attempt; canonical cell ID from complete runtime identity | Selection frozen; attempt and cell ID pending, owned by execution and validation days. |
| `cell.matrix`, `cell.problem`, `cell.rhs` | Exact matrix bytes, source commit/attribution review, 4x4/7/10 conversion, one float64 RHS and oracle | Numerical input checked here; attribution/rights review pending Day 2. |
| `cell.project` | Full Git SHA and dirty patch if needed; compiler/flags/build mode; exact executable and archive bytes; API/backend/order/ABI | Build and runtime evidence pending Day 3. |
| `cell.baseline` | SuiteSparse source/package, actual loaded CHOLMOD path/hash, build/module options, dependency graph, int32 ABI, effective controls | Loaded runtime evidence pending Day 4. |
| `methodology.identity`, `.correctness`, `.measurement` | Hashed Day 3/4/5 contract files and fixed units/policy | Source hashes captured below; record references pending. |
| `environment` | Host OS/build, architecture/CPU, compiler runtime, BLAS/OpenMP identities, requested and observed threads, full command/cwd/time/fingerprint | Actual run environment pending Day 5. |
| `correctness.project`, `.baseline`, cross-solver fields | Separate raw solver streams, per-phase statuses, factor diagnostics, solutions, recomputed residual/oracle metrics and consistency limit | Both solvers unrun; pending Days 6-11. |
| `artifacts` and `publication_review` | Byte-backed manifest/log/solver outputs, exact executable/library, dirty patch if applicable, separate rights decision artifact | Generated bundle absent; rights decision pending Day 2 and publication review later. |
| `timing`, `claim_scope`, `non_claims` | `timing.state=not_run`, no samples or ranking claim; local correctness-only wording if earned | No timing or correctness claim earned. |

Observed status additionally requires all referenced byte hashes, complete cell identity, both independently passing sides, and a valid `--input --bundle-root` validator verdict. A pilot `attempt.json`, a successful process exit, the validator self-test, or a synthetic fixture does not meet this gate. Publication approval is separate from numerical validation. Generated attempts remain under ignored `build/external-baseline/`; no selected report or public API-doc output is an evidence destination.

## Source/control hashes captured on Day 1

| File | SHA-256 |
| --- | --- |
| `docs/planning/EPIC_19/SPRINT_215/artifacts/day3-methodology-inputs.md` | `5bd0708e36816135ed99a95a8b28fd23b7b43876a8956ce64c830f3139eb9943` |
| `docs/planning/EPIC_19/SPRINT_215/artifacts/day4-correctness-and-tolerance-contract.md` | `7ca1249219cb2842a9a4827ecdeb57024a5a03b2b0762400988051505c91483e` |
| `docs/planning/EPIC_19/SPRINT_215/artifacts/day5-external-comparison-methodology.md` | `bcee23be60b8612f91951f6afa94ab7d48f7c85085b43c97319ff8d614c49235` |
| `tests/corpus/schemas/external_baseline_evidence_v1.json` | `7a0c61ef6ea47a68da3caf34a378e73f5305736c518ee76731fbed2d8865bfe4` |
| `scripts/cholmod_pilot.c` | `28e736f57087782459ca4afe9c8f00232b279764cdf85c4c6660062ede1fa50a` |
| `scripts/cholmod_pilot_csc.h` | `50cd5925b74af82c3049bb76ebd3b59802b9224f0b07dad34301de8308dbe466` |
| `scripts/run_cholmod_pilot.py` | `fe151b9265de2ef4f1d5509fc11410a30e1ebafd00afb0bb59c8e56add0277c7` |
| `scripts/validate_external_baseline_evidence.py` | `9969cb3b32d61e118d33a73e321f12900e2cd44fdbd87bf6e7628b68a52a69fe` |

Verification commands and outcomes are in `../WORKING_NOTES.md`. This artifact is an input and evidence inventory, not an observed baseline result.
