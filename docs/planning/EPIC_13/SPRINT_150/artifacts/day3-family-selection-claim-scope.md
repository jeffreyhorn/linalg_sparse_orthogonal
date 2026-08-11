# Sprint 150 Day 3: Family Selection And Claim Scope

## Purpose

Select the QR fixture families Sprint 150 will close, define the precise claim
scope and non-claims for each family, and map implementation ownership before
metadata, oracle, test, or documentation edits begin.

## Decision

Sprint 150 will close **two** maintained QR corpus families:

1. **Rank-deficient rectangular QR family**
2. **Underdetermined minimum-norm QR family**

Sprint 150 will **not** promote reorder/COLAMD QR into the maintained corpus
family set. Reorder/COLAMD remains owner-local evidence for this sprint because
the candidate mixes residual, status, fill, ordering, optional SuiteSparse, and
performance-adjacent semantics. Closing it completely would either require a
separate sprint or would risk weakening the closure quality for the two
higher-scoring families.

## Selected Family 1: Rank-Deficient Rectangular QR

### Claim Scope

For selected source-controlled rank-deficient rectangular QR fixtures, the
project QR implementation may claim fixture-family-local evidence for:

- expected matrix shape, `nnz`, rank, and nullity;
- QR factorization success;
- rank/nullity agreement with expected rows;
- normalized residual for solver-produced nullspace vectors;
- subspace-safe projector or residual comparisons where expected rows define
  them;
- least-squares residual for selected rank-deficient rectangular solve rows.

The claim is local-only and generated-local until a later sprint promotes
hosted evidence. Expected rows define targets; solver pass evidence requires
focused proof-owner tests and generated oracle/report rows.

### Candidate Fixtures For Metadata Design

Day 4 should design rows for these likely fixtures, subject to schema and
generator feasibility:

| Fixture Candidate | Source Evidence | Planned Semantics |
| --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1` | Existing Sprint 139 maintained corpus fixture | Keep as the seed row and ensure family rows remain consistent. |
| `qr_rankdef_duplicate_5x4_*` | `tf_qr_make_rankdef_duplicate_5x4`, dense-reference rank/residual/projector tests | Rank `3`, nullity `1`, residual, and projector-safe nullspace evidence. |
| `qr_rankdef_dependent_row_4x3_*` | `tf_qr_make_dependent_row_4x3`, dense-reference residual/projector tests | Rank `2`, nullity `1`, residual, and projector-safe nullspace evidence. |
| `qr_rankdef_wide_3x5_nullspace_subspace` | `make_rankdef_wide_3x5` and external projector/subspace test | Rank `2`, nullity `3`, and subspace-safe nullspace evidence if metadata remains tractable. |

Day 4 may narrow the fixture set if the wide-subspace row would make the family
too large for complete Sprint 150 closure.

### Non-Claims

- no raw Q/R basis equality;
- no Q-sign, orientation, scale, or column-order parity;
- no global rank-threshold policy;
- no broad rank-deficient QR correctness;
- no broad least-squares residual guarantee;
- no SuiteSparse, LAPACK, NumPy, SciPy, or external-library parity;
- no platform, package, ABI, performance, or state-of-the-art claim.

## Selected Family 2: Underdetermined Minimum-Norm QR

### Claim Scope

For selected source-controlled underdetermined minimum-norm QR fixtures, the
project QR implementation may claim fixture-family-local evidence for:

- expected matrix shape, `nnz`, rank/nullity when relevant, and RHS policy;
- `sparse_qr_solve_minnorm()` success for selected consistent fixtures;
- residual `||Ax-b||` within the fixture tolerance;
- solution norm agreement with the expected minimum-norm row;
- selected exact solution entries only where the generator makes them
  analytically stable and the expected row names that exact-value claim;
- optional refinement behavior only if an expected row and proof-owner test
  explicitly select it.

The claim is local-only and generated-local until a later sprint promotes
hosted evidence. Exact solution rows are allowed only for deterministic small
fixtures and must not become a global algorithmic identity claim.

### Candidate Fixtures For Metadata Design

Day 4 should design rows for these likely fixtures, subject to schema and
generator feasibility:

| Fixture Candidate | Source Evidence | Planned Semantics |
| --- | --- | --- |
| `qr_underdetermined_minnorm_2x4` | `tests/test_qr_solve.c` and `tests/test_colamd.c` exact 2x4 checks | Residual, solution norm `1.0`, and exact entries `[0.5, 0.5, 0.5, 0.5]`. |
| `qr_minnorm_3x6_exact_values` | `tests/test_colamd.c::test_minnorm_3x6` | Residual, solution norm `sqrt(8.4)`, and exact expected entries. |
| `qr_minnorm_5x10_exact_values` | `tests/test_colamd.c::test_minnorm_5x10` | Residual, solution norm `sqrt(11.0)`, and exact expected entries. |
| `qr_minnorm_rankdef_2x4` | `tests/test_colamd.c::test_minnorm_rank_deficient` | Residual and norm for a rank-deficient consistent system if Day 4 keeps scope manageable. |
| `qr_minnorm_zero_row_2x4` | `tests/test_colamd.c::test_minnorm_zero_row` | Residual and norm for a zero-row consistent system if Day 4 keeps scope manageable. |

Day 4 may select only the exact 2x4, 3x6, and 5x10 fixtures if rank-deficient
and zero-row variants would prevent complete closure.

### Non-Claims

- no global minimum-norm guarantee beyond selected fixtures and tolerances;
- no SVD-pseudoinverse-as-global-oracle claim;
- no broad rank-deficient recovery claim;
- no broad inconsistent-system behavior claim;
- no exact-vector identity for fixtures that do not explicitly own exact rows;
- no LAPACK, NumPy, SciPy, SuiteSparse, or broad external-library parity;
- no platform, package, ABI, performance, or state-of-the-art claim.

## Deferred Family: Reorder/COLAMD QR

### Deferral Rationale

Reorder/COLAMD QR scored lower on Day 2 because its current evidence mixes
several different claim types:

- residual/status behavior for QR+COLAMD solves;
- permutation validity;
- fill diagnostics;
- natural/AMD/COLAMD comparisons;
- sparse-mode behavior;
- optional SuiteSparse context.

Promoting this family in Sprint 150 would require substantial non-claim and
report design to avoid implying reorder optimality, fill improvement,
performance, SuiteSparse parity, or broad COLAMD parity. That is viable future
work, but it is not selected for Sprint 150 because the sprint is better spent
fully closing rank-deficient rectangular and underdetermined minimum-norm
families.

### Deferred Non-Claims

- no broad reorder/COLAMD QR family claim;
- no fill-reduction guarantee;
- no ordering optimality claim;
- no COLAMD parity claim;
- no SuiteSparse corpus completeness claim;
- no performance claim.

## Implementation Map

| Workstream | Rank-Deficient Rectangular | Underdetermined Minimum-Norm | Owner Days |
| --- | --- | --- | --- |
| Fixture rows | Duplicate, dependent-row, wide-subspace candidates plus existing 6x4 seed | 2x4, 3x6, 5x10, optional rank-deficient/zero-row candidates | Days 4-5 |
| Generator rows | Deterministic helper-derived matrix structures and hashes | Deterministic exact small systems and RHS policies | Days 4-5 |
| Expected rows | Rank, nullity, residual, projector/subspace metrics | Residual, solution norm, selected exact values, status | Days 4-7 |
| Oracle semantics | Rank/nullity exact rows; residual/projector/subspace-safe rows | Residual and norm rows; exact-value rows only where stable | Days 6-7 |
| Proof-owner tests | Extend focused QR corpus proof owner and helpers | Extend focused QR corpus proof owner and helpers | Days 8-9 |
| Report integration | Add generated-local QR family rows with command/commit/platform/compiler/configuration/support tier | Add generated-local QR minimum-norm rows with the same evidence fields | Days 10-11 |
| Documentation | Corpus, solver-selection, README, tutorial/cookbook, maintainer guide | Same surfaces with minimum-norm wording and non-claims | Day 12 |
| Validation | Corpus schema, focused QR corpus tests, oracle/report checks, full C gate if `.c`/`.h` changes | Same | Days 13-14 |

## Rollback Rules

| Trigger | Rollback |
| --- | --- |
| A fixture row cannot be generated deterministically with stable structure/value hashes. | Drop that fixture from Sprint 150 selection before implementation, or keep it as owner-local evidence only. |
| Expected rows require raw basis identity, sign, orientation, or column order. | Replace the expected row with residual/projector/subspace-safe semantics or drop the fixture. |
| Rank/nullity semantics depend on a broad global threshold policy. | Narrow to named fixture/tolerance rows or defer the fixture. |
| Minimum-norm exact entries prove brittle while residual/norm semantics remain stable. | Keep residual/norm rows and remove exact-value rows for that fixture. |
| Focused QR corpus tests become too large or duplicate monolithic `test_qr.c`. | Split helper logic but keep proof-owner tests focused on selected families. |
| Oracle/report rows omit required command, commit, platform, compiler, configuration, support tier, claim scope, or non-claims. | Do not use generated rows as Sprint 150 evidence until fixed. |
| Documentation implies broad QR, external-library, platform, package, performance, or state-of-the-art claims. | Revert or narrow documentation before closeout. |
| Any required validation fails. | Fix the failure or mark the affected family unclosed before retrospective. |

## Day 4 Handoff

Day 4 should design concrete source-controlled metadata rows for the selected
families. It should prefer a fixture set that can close completely:

1. Keep `qr_rank_deficient_6x4_nullspace_v1` as the existing QR corpus seed.
2. Add two or three rank-deficient rectangular rows from duplicate,
   dependent-row, and wide-subspace candidates.
3. Add two or three underdetermined minimum-norm rows from exact 2x4, 3x6, and
   5x10 candidates.
4. Treat rank-deficient/zero-row minimum-norm as optional expansion only if the
   first rows remain manageable.
5. Do not add reorder/COLAMD metadata rows in Sprint 150 unless Day 4 discovers
   the selected families cannot close and the replacement claim is narrower.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Selected families can be fully closed within Sprint 150. | Complete | Two-family decision selects the highest-scoring, clearest families and defers reorder/COLAMD. |
| Every claim has a matching proof owner and report/update owner. | Complete | Implementation map assigns fixture, generator, expected, oracle, proof-owner, report, docs, and validation owners. |
| Unsupported QR claims are explicit before metadata work starts. | Complete | Family non-claims and deferred-family non-claims list raw-basis, broad QR, external parity, platform, package, performance, and state-of-the-art exclusions. |
