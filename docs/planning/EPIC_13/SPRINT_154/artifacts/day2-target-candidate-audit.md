# Sprint 154 Day 2 Target Candidate Audit

## Purpose

Day 2 compares the maintained QR and partial-SVD corpus families as candidate
targets for the first narrow external comparison study. The goal is to choose a
target that can be implemented completely without turning one fixture-local
study into broad ecosystem parity.

## Candidate Inputs

| Candidate | Maintained Fixtures | Source-Controlled Expected Rows | Current Proof Owner | Current Report Path |
| --- | ---: | ---: | --- | --- |
| QR corpus family | 6 | 23 | `tests/test_qr_corpus.c` plus `scripts/run_corpus_oracle.py --include-solver-qr` | generated-local oracle rows under `build/corpus/oracle/*.tsv` |
| Partial-SVD corpus family | 4 | 26 | `tests/test_svd_partial_corpus.c` plus `scripts/run_corpus_oracle.py --include-partial-svd` | generated-local oracle rows under `build/corpus/oracle/*.tsv` |

Both candidates are local-only fixture evidence. Neither candidate currently
claims broad external-library parity, hosted CI proof, package proof, ABI
proof, performance proof, or state-of-the-art behavior.

## QR Candidate Assessment

### Fixture Shape

The maintained QR family contains:

- one Sprint 139 seed fixture for rank, nullity, and normalized null-vector
  residual;
- two Sprint 150 rank-deficient rectangular fixtures with rank, nullity,
  normalized nullspace residual, and subspace distance rows;
- three Sprint 150 underdetermined minimum-norm fixtures with rank, nullity,
  residual, solution norm, and selected exact-value rows.

### Existing Reference Surface

`tests/qr_external_dense_reference.py` already owns bounded dense-reference
logic for several QR-related fixtures:

- overdetermined compatible and incompatible least-squares systems;
- rank-only duplicate-column fixture;
- rank-threshold families;
- residual-only rank-deficient systems;
- rank-deficient wide system helpers;
- `qr_underdetermined_minnorm_2x4`.

This helper is useful prior art, but it is a Python stdlib dense-reference
process, not an external package or ecosystem baseline. A Sprint 154 QR study
would need an explicit dependency decision if it intends to compare against an
external library such as NumPy/SciPy. If it stays with the existing helper, the
study should be described as an external-process dense-reference comparison,
not external-library parity.

### Metric Strength

QR has strong metric clarity for:

- rank and nullity;
- residual norms;
- solution norms for minimum-norm fixtures;
- selected exact solution values;
- projector/subspace distance for nullspace comparisons.

QR has higher overclaim risk for:

- raw Q/R basis values;
- sign, orientation, and column-order parity;
- global rank-threshold policy;
- broad minimum-norm behavior;
- LAPACK/NumPy/SciPy parity wording.

### Feasibility

QR is feasible for a first narrow study if the selected scope is one of:

1. `qr_underdetermined_minnorm_2x4` against an external-process dense
   minimum-norm reference;
2. the three underdetermined minimum-norm fixtures against a pinned optional
   dense baseline, if dependency policy stays clean;
3. one rank-deficient rectangular fixture using only rank/nullity/residual and
   subspace-safe metrics.

The smallest complete target is
`qr_underdetermined_minnorm_2x4`, because existing reference construction is
already available and the metric set is easy to explain.

## Partial-SVD Candidate Assessment

### Fixture Shape

The maintained partial-SVD family contains:

- clustered/repeated diagonal top-k behavior;
- rank-deficient rectangular range projectors;
- sparse low-rank output behavior at `drop_tol=0`;
- tight-budget fail-closed behavior and default-budget recovery.

### Existing Reference Surface

`tests/svd_external_dense_reference.py` already owns bounded dense singular
value references for several SVD and partial-SVD fixtures, including earlier
partial-SVD diagonal and nonsymmetric fixtures.

The Sprint 151 corpus fixtures extend beyond simple singular values:

- projector distances;
- triplet residuals;
- orthogonality residuals;
- sparse low-rank output shape, selected values, and dense/sparse agreement;
- fail-closed status and no-partial-array diagnostics.

These are good local correctness metrics, but an external-library comparison
would require more parsing and more careful semantics than QR. In particular,
raw singular-vector parity is invalid for repeated or clustered singular
values, and sparse-output behavior does not map cleanly onto many external
dense libraries.

### Metric Strength

Partial-SVD has strong metric clarity for:

- sorted top-k singular values;
- rank;
- selected subspace projector distances;
- triplet residuals;
- U/V orthogonality residuals;
- exact fail-closed status and recovery behavior within the project API.

Partial-SVD has higher overclaim risk for:

- raw singular-vector identity;
- sign, phase, orientation, or basis-order parity;
- convergence-rate and iteration-count behavior;
- useful partial-result guarantees after non-convergence;
- broad low-rank sparse-output/drop-tolerance policy;
- LAPACK/NumPy/SciPy parity wording.

### Feasibility

Partial-SVD is feasible if the first study is narrowed to singular values on a
diagonal fixture. It becomes much more complex if the study includes
projectors, sparse low-rank output, or fail-closed status, because those fields
need either a library-specific comparison contract or a project-only caveat.

The smallest complete target is
`partial_svd_rankdef_diag6x4_k2_range_projector_v1` restricted to top-2
singular values and rank. A richer study using projector distances is possible
but riskier because repeated, sign, and basis-order semantics must be designed
carefully.

## External Baseline Candidate List

| Baseline Candidate | Applicability | Dependency Posture | Day 2 Risk |
| --- | --- | --- | --- |
| Existing Python stdlib dense-reference helpers | QR and SVD bounded reference calculations | Already source-controlled; no optional package install | Not an external library; should be named external-process dense reference only. |
| NumPy | Dense QR/SVD/least-squares and singular values | Optional local Python package; version must be captured if used | Missing dependency must skip/defer, not fail or pass; no package-manager claim. |
| SciPy | Dense/sparse linear algebra and potentially sparse SVD routines | Optional local Python package; version and backend details must be captured | Larger dependency and broader ecosystem-parity wording risk. |
| LAPACK via system tooling | Dense QR/SVD reference semantics | Not directly available through current repo scripts | High discovery/parsing cost for a first harness. |
| SuiteSparse/CHOLMOD | Sparse direct solver ecosystem | Not aligned with QR/partial-SVD first target | Wrong target family; high overclaim risk. |
| Eigen command-line helper | Possible if a helper is written and compiler/toolchain exists | Requires new helper build and version/provenance policy | Too much implementation surface for first narrow study. |

## Candidate Scorecard

Scores use `1` for weak/high-risk and `5` for strong/low-risk.

| Criterion | QR Candidate | Partial-SVD Candidate | Notes |
| --- | ---: | ---: | --- |
| Maintained fixture maturity | 5 | 5 | Both have recent maintained corpus families. |
| External baseline availability | 4 | 3 | QR has an existing minimum-norm helper fixture; partial-SVD has singular-value helpers but less coverage for Sprint 151 rows. |
| Metric clarity | 4 | 3 | QR residual/rank/norm rows are simpler than subspace/sparse-output/fail-closed SVD rows. |
| Tolerance stability | 4 | 3 | QR exact/small residual rows are stable; partial-SVD projector and residual tolerances require more care. |
| Parsing complexity | 4 | 3 | QR can start with scalar/vector summaries; partial-SVD may need vectors, projector metrics, and sparse-output fields. |
| Optional dependency risk | 3 | 3 | Both need explicit skip/defer rules if NumPy/SciPy is selected. |
| Report integration cost | 4 | 3 | QR first-study rows can mirror existing oracle row shapes more easily. |
| Documentation overclaim risk | 3 | 2 | Partial-SVD wording has more sign/phase/convergence/drop-tolerance traps. |
| Complete-in-sprint confidence | 4 | 3 | QR offers a smaller first complete study. |

## Recommended Day 3 Bias

Day 3 should prefer a QR first study unless a dependency-policy requirement
rules it out.

The recommended narrow target is:

- `qr_underdetermined_minnorm_2x4`;
- metric family: residual norm, solution norm, and selected solution entries;
- baseline posture: external-process dense-reference first, with optional
  NumPy/SciPy dependency considered only if Day 4 can make skip/defer and
  version capture airtight;
- non-claim: no broad QR, LAPACK, NumPy, SciPy, minimum-norm, performance,
  package, ABI, platform, or state-of-the-art parity.

The recommended deferred target is:

- partial-SVD subspace/projector comparison;
- reason: richer scientific value, but higher schema, dependency, and wording
  risk for the first narrow comparison harness.

## Candidate-Specific Blockers

### QR Blockers

- Need an explicit decision on whether the baseline is the existing
  source-controlled dense-reference helper or an optional external package.
- Need dependency version capture if NumPy/SciPy is used.
- Need to avoid raw QR basis, sign, orientation, and ordering claims.
- Need to keep QR-only generated output separate from the Sprint 152 combined
  oracle freshness gate unless report integration explicitly selects otherwise.

### Partial-SVD Blockers

- Need raw singular-vector identity to remain out of scope.
- Need subspace/projector comparison semantics before using vector outputs.
- Need sparse-output comparison caveats if the low-rank fixture is selected.
- Need fail-closed status caveats because external libraries may not expose the
  same error or partial-output behavior.
- Need stricter wording controls around convergence rate, iteration count,
  repeated-spectrum behavior, and drop-tolerance policy.

## Day 3 Handoff

Day 3 should select exactly one target and freeze:

- fixture key or keys;
- baseline type: external-process helper, optional NumPy/SciPy, or deferred;
- accepted metrics;
- tolerance policy;
- skip/defer status semantics;
- non-claims;
- report integration expectation for Days 10-11.

The smallest complete first study remains QR minimum-norm on
`qr_underdetermined_minnorm_2x4`.
