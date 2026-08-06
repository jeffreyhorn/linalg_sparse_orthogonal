# Sprint 139 Day 6: Oracle Comparison Design

## Purpose

Day 6 defines the solver-backed QR oracle comparison for the selected Sprint
139 fixture. The design keeps generated-reference evidence from Sprint 138
separate from QR implementation pass evidence and defines exact observed
values, tolerance semantics, failure classes, optional-data handling,
provenance, command ownership, and report freshness expectations.

This is a design artifact. It does not change QR source, tests, corpus rows,
oracle commands, generated outputs, public documentation, or support tiers.

## Oracle Reference Approach

Chosen approach:

Use the library's QR implementation as the observed side and the maintained
corpus expected rows as the target side.

The observed solver-backed command should:

1. Build `qr_rank_deficient_6x4_nullspace_v1` from the maintained deterministic
   fixture entries.
2. Run `sparse_qr_factor()`.
3. Record `sparse_qr_rank(&qr, 0.0)`.
4. Record nullity from `sparse_qr_nullspace(&qr, 0.0, NULL, &ndim)`.
5. Extract the single solver-produced nullspace vector.
6. Record `||A*v_solver||_2 / ||v_solver||_2`.
7. Compare those observations against the existing expected-result rows.

No external dense reference is required for the primary Day 6 design because
the fixture has exact rank, nullity, and a residual check that is insensitive
to raw basis orientation. External projector/subspace comparison remains a
future enhancement, not a prerequisite for closing the first selected lane.

## Generated Reference vs Solver-Backed Evidence

| Evidence kind | Current or planned owner | `solver_family` | Meaning |
| --- | --- | --- | --- |
| Generated reference metadata | `scripts/run_corpus_oracle.py` current first-lane rows | `unknown` | Regenerates fixture metadata and reference facts without running the QR implementation. |
| Solver-backed QR evidence | Sprint 139 focused QR proof/oracle command | `qr` | Runs the QR implementation on the maintained fixture and records rank, nullity, and residual observations. |

Day 7 should avoid overwriting row meaning silently. If the same oracle row IDs
are reused for solver-backed rows, the command, `solver_family`, compiler,
configuration, claim scope, and non-claims must make the evidence type clear.
If separate row IDs are added, they should include the operation detail, for
example:

- `qr_rank_deficient_6x4_nullspace_v1_qr_rank`
- `qr_rank_deficient_6x4_nullspace_v1_qr_nullity`
- `qr_rank_deficient_6x4_nullspace_v1_qr_nullspace_residual`

The Day 7 implementation should choose one naming strategy and document it in
the generated report and maintainer notes.

## Observed Values

| Observation | Source | Expected value | Tolerance | Pass condition |
| --- | --- | --- | --- | --- |
| rank | `sparse_qr_rank(&qr, 0.0)` after `sparse_qr_factor()` | `3` | `exact=0` | observed integer equals `3` |
| nullity | `sparse_qr_nullspace(&qr, 0.0, NULL, &ndim)` | `1` | `exact=0` | observed integer equals `1` |
| normalized residual | extracted nullspace vector `v_solver` | `normalized_null_vector_residual<=1e-10` | `absolute=1e-10` | `||A*v_solver||_2 / ||v_solver||_2 <= 1e-10` |

Residual computation rules:

- Use the matrix entries from `qr_rank_deficient_6x4_nullspace_v1`.
- Use the solver-produced basis vector, not the reference vector
  `[-1, -1, 0, 1]`.
- Fail closed if the nullspace vector norm is zero.
- Do not compare raw vector entries to the reference direction.
- Do not compute a least-squares or minimum-norm residual for this lane.

## Tolerance and Failure-Class Table

| Condition | Status | Failure class | Interpretation |
| --- | --- | --- | --- |
| factorization succeeds, rank is `3`, nullity is `1`, residual `<= 1e-10` | `pass` | empty | Fixture-local QR residual is closed for this row. |
| `sparse_qr_factor()` fails | `fail` | `fail_oracle_mismatch` | Solver did not produce the required QR factorization for the selected fixture. |
| rank observation differs from `3` | `fail` | `fail_oracle_mismatch` | Rank comparison failed. |
| nullity query fails or observed nullity differs from `1` | `fail` | `fail_oracle_mismatch` | Nullity comparison failed. |
| basis extraction fails | `fail` | `fail_oracle_mismatch` | Nullspace observation could not be produced. |
| extracted vector has zero norm | `fail` | `fail_oracle_mismatch` | Residual comparison is invalid. |
| normalized residual exceeds `1e-10` | `fail` | `fail_oracle_mismatch` | Solver-produced nullspace vector does not satisfy the fixture residual tolerance. |
| expected row is missing, duplicate, malformed, or has unsupported tolerance semantics | `fail` | `fail_malformed_row` | Corpus/oracle metadata integrity failure. |
| generated fixture structure or values mismatch manifest hashes | `fail` | `fail_generator_mismatch` | Fixture cannot be trusted for comparison. |
| stale command, commit, compiler, configuration, or support-tier metadata is detected | `fail` | `fail_report_stale` | Report is not current enough to support the claim. |
| optional SuiteSparse data is disabled or unavailable | `skip` | `skip_optional_unavailable` | Optional-data row remains non-pass evidence. |
| platform lacks the focused proof command or compiler requirement | `unsupported` | `unsupported_platform` | Platform row cannot support a solver pass claim. |

## Optional-Data and External Reference Policy

The selected Sprint 139 closure does not require optional external data.

Rules:

- `suitesparse_rank_deficient_qr_subset_v1` remains disabled by default.
- Optional-data skip/defer rows are policy evidence only.
- A skipped optional-data row must never increase QR pass counts.
- External dense-reference helpers may be used as local cross-check patterns,
  but Day 6 does not require broad LAPACK, NumPy, SciPy, or SuiteSparse parity.
- If a later day adds external projector/subspace rows, those rows must carry
  explicit non-claims and support-tier metadata.

## Command Ownership

Preferred Day 7 direction:

Add a focused solver-backed QR oracle path that can be invoked from the
repository root and from other working directories.

Candidate command shape:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

Alternative command shape if separation is cleaner:

```sh
python3 scripts/run_qr_corpus_oracle.py
```

Decision criteria for Day 7:

- If implementation can reuse the existing corpus validation, generator, TSV,
  report, and skip/defer plumbing cleanly, extend `scripts/run_corpus_oracle.py`
  with an explicit opt-in flag.
- If calling compiled QR proof code or parsing a focused test binary makes the
  command too different, use a dedicated QR oracle command and keep row/report
  semantics aligned with the existing schema.
- Either path must record the exact command in generated oracle rows and report
  rows.

## Provenance Fields

Solver-backed QR rows should record:

| Field | Required value |
| --- | --- |
| `solver_family` | `qr` |
| `compiler` | detected compiler/version or the compiler used to build the focused test command |
| `configuration` | semicolon-separated `key=value` text, including build profile, optional-data policy, fixture hash, and proof owner |
| `support_tier` | `local_only` unless reviewed hosted evidence promotes it |
| `command` | exact command used to generate observed rows |
| `source_commit` | current commit SHA |
| `source_branch` | current branch, tag, or `detached` |
| `platform` | OS and architecture |
| `claim_scope` | fixture-local solver-backed QR rank/nullity/nullspace residual evidence |
| `non_claims` | no broad QR correctness; no raw-basis parity; no global rank-threshold policy; no broad rank-deficient solve; no minimum-norm or least-squares claim; no SuiteSparse or external-library parity; no platform/performance/state-of-the-art claim |

Example solver-backed configuration components:

```text
build_profile=static_default;optional_data_policy=disabled;proof_owner=test_qr_corpus;fixture_hash=<value>;qr_tolerance=1e-10
```

## Report Freshness Expectations

Solver-backed QR report rows become stale when any of these inputs change:

- `src/sparse_qr.c`, `src/sparse_qr_householder.c`, `src/sparse_qr_internal.h`,
  or `include/sparse_qr.h`;
- `tests/test_qr_corpus.c` or any QR helper used by the proof owner;
- fixture, generator, expected-result, or oracle schema rows;
- `scripts/run_corpus_oracle.py` or a future QR-specific oracle command;
- Make/CMake build configuration or compiler;
- tolerance, support tier, claim scope, or non-claim wording;
- source commit or branch.

Fresh report rows must preserve row meaning and support tier. A local pass row
does not become reviewed Linux, macOS, Windows, package, performance, or
state-of-the-art evidence by being listed in a report index.

## Day 7 Implementation Checklist

Day 7 should:

1. Choose whether to extend `scripts/run_corpus_oracle.py` or add a dedicated
   QR oracle command.
2. Keep generated-reference `solver_family=unknown` rows distinct from
   solver-backed `solver_family=qr` rows.
3. Emit rank, nullity, and normalized residual observations with exact
   tolerance semantics.
4. Record compiler/configuration/proof-owner provenance for solver-backed rows.
5. Preserve optional-data skip semantics.
6. Run corpus schema validation, oracle generation, report metadata checks, and
   Python compile checks for touched scripts.

## Day 6 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Comparison semantics are explicit before runner changes. | Complete | Observed values, residual computation rules, tolerance table, and failure classes are defined. |
| Optional or external references cannot be counted as pass evidence when unavailable. | Complete | Optional-data and external-reference policy keeps skip/defer rows separate from QR pass evidence. |
| The oracle design supports the selected residual without broad claims. | Complete | Provenance fields, claim scope, non-claims, and report freshness expectations keep evidence fixture-local. |
