# Sprint 154 Day 3 Comparison Target Selection

## Purpose

Day 3 selects the first narrow external comparison target and freezes the
fixture, baseline type, accepted metrics, tolerances, caveats, failure
statuses, deferred targets, and Day 4 dependency-policy handoff.

## Decision

Sprint 154 will implement the first comparison study around the maintained QR
minimum-norm fixture:

- selected fixture: `qr_underdetermined_minnorm_2x4`;
- fixture family: `qr_minnorm_underdetermined`;
- matrix shape: `2 x 4`;
- nonzeros: `4`;
- rank/nullity: full row rank `2`, nullity `2`;
- RHS policy: explicit RHS;
- expected solution: `[0.5, 0.5, 0.5, 0.5]`;
- expected solution norm: `1.0`;
- expected residual: `<= 1e-10`.

This target is selected because it is the smallest maintained QR candidate
with clear scalar/vector metrics, existing source-controlled expected rows,
existing external-process dense-reference prior art, and low wording risk when
kept fixture-local.

## Selected Baseline Type

The initial baseline type is:

- external-process dense reference;
- source-controlled helper first;
- optional external package comparison only if Day 4 can define clean
  dependency discovery, version capture, skip/defer behavior, and
  non-package-manager wording.

The existing `tests/qr_external_dense_reference.py` helper already contains
the fixture construction for `qr_underdetermined_minnorm_2x4`. It may be reused
as prior art, but the Sprint 154 harness should record the baseline as an
external-process dense reference unless a later day explicitly adds a pinned
optional package baseline.

## Accepted Metrics

The first study may emit these metrics:

| Metric | Expected Value | Tolerance | Status Meaning |
| --- | --- | --- | --- |
| project status | `SPARSE_SUCCESS` | status-only | Project QR minimum-norm solve completed for this fixture. |
| baseline status | `success` | status-only | External-process dense reference completed for this fixture. |
| residual norm | `<= 1e-10` | absolute `1e-10` | Project solution satisfies the fixture equations within the maintained QR tolerance. |
| solution norm | `1.0` | absolute `1e-10` | Project solution norm matches the expected minimum-norm value for this fixture. |
| solution values | `0.5,0.5,0.5,0.5` | absolute `1e-10` per component | Project solution values match the fixture-local dense reference. |
| project-vs-baseline max abs delta | `<= 1e-10` | absolute `1e-10` | Project solution matches the external-process dense reference for the selected vector values. |

The first study must not emit raw QR basis, Q/R matrix-entry, pivot-order, or
timing metrics.

## Tolerance Policy

Use the existing source-controlled expected-row tolerances from
`tests/corpus/expected/qr_underdetermined_minnorm_2x4.tsv`:

- residual norm: absolute `1e-10`;
- solution norm: absolute `1e-10`;
- solution values: absolute `1e-10` per component;
- status: exact status match.

Any optional external package baseline must either meet the same tolerance for
the selected metrics or produce a non-pass status with a caveat. The harness
must not silently relax tolerances to make a baseline pass.

## Status Semantics

The first study should use explicit statuses:

| Status | Meaning | Counts As Proof |
| --- | --- | --- |
| `pass` | Project and selected baseline produced comparable selected metrics within tolerance. | Yes, fixture-local only. |
| `fail` | Project or baseline ran, but selected metrics missed tolerance or status expectations. | No. |
| `skip` | Optional dependency is missing or intentionally disabled according to Day 4 policy. | No. |
| `defer` | Candidate target or baseline was intentionally not selected for this study. | No. |
| `error` | Harness output was malformed, command execution failed unexpectedly, or provenance was incomplete. | No. |

Skipped and deferred rows must not be counted as comparison proof.

## Caveats And Non-Claims

The study is allowed to claim only:

The project solution for `qr_underdetermined_minnorm_2x4` matches the selected
external-process dense reference on fixture-local status, residual norm,
solution norm, and selected solution values under the recorded command,
version, platform, compiler, configuration, and tolerance policy.

The study must not claim:

- broad QR correctness;
- broad minimum-norm QR behavior;
- broad rank-deficient recovery;
- residual-only least-squares behavior;
- raw QR basis, sign, orientation, or pivot-order parity;
- SVD pseudoinverse global-oracle behavior;
- LAPACK, NumPy, SciPy, SuiteSparse, Eigen, CHOLMOD, PETSc, Trilinos, or
  ecosystem parity;
- performance, timing, throughput, or memory superiority;
- hosted CI proof;
- package-manager support;
- shared-library support, dynamic ABI compatibility, or runtime-loader
  behavior;
- platform portability beyond the recorded local run;
- state-of-the-art sparse linear algebra status.

## Deferred Targets

| Deferred Target | Reason |
| --- | --- |
| All three QR minimum-norm fixtures | Useful later, but too wide for the first harness; start with one fixture and expand only after schema and dependency policy prove stable. |
| QR rank-deficient nullspace/subspace comparison | Requires projector/subspace semantics and stricter raw-basis non-claim handling. |
| QR reorder/COLAMD comparisons | Mixes ordering, fill, residual, optional SuiteSparse, and performance-adjacent semantics. |
| Partial-SVD singular-value-only comparison | Feasible, but lower first-study priority than QR because selected Sprint 151 rows involve richer semantics. |
| Partial-SVD subspace/projector comparison | Requires sign/phase/basis-order-safe subspace comparison and careful repeated-spectrum wording. |
| Partial-SVD sparse low-rank output comparison | External dense libraries do not naturally map to project sparse-output/drop-tolerance behavior. |
| Partial-SVD fail-closed comparison | External libraries may not expose equivalent convergence status or partial-array behavior. |
| NumPy/SciPy package baseline | Deferred until Day 4 determines whether optional package discovery and version capture are clean enough for the first study. |
| Timing/performance comparison | Requires benchmark methodology outside Sprint 154 Day 3 scope. |

## Report Integration Expectation

Day 10 should expect a narrow generated-local comparison artifact. Report-index
integration should be added only if the row schema can preserve:

- library/baseline name and version;
- command;
- platform;
- compiler/configuration;
- fixture key;
- metric;
- tolerance;
- status;
- caveat;
- artifact path;
- local-only support tier.

If any of those fields remain unsettled, Day 10 should choose artifact-only
publication and explicitly defer normalized comparison rows.

## Day 4 Handoff

Day 4 should define dependency policy for the selected target:

- canonical baseline command name;
- whether the baseline is source-controlled helper only or optional
  NumPy/SciPy is allowed;
- baseline version capture;
- executable/interpreter discovery;
- skip/defer behavior for missing optional dependencies;
- provenance fields for command, platform, compiler, configuration, fixture,
  and artifact path;
- security/reproducibility boundaries for external packages.

The selected target is intentionally small enough that Day 4 can choose a
source-controlled external-process dense reference and still produce a complete
first study.
