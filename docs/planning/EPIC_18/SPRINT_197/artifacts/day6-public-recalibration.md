# Sprint 197 Day 6 Public Documentation Recalibration

## Purpose

Day 6 performs the public documentation recalibration step for final-validation
item 206.2. It checks whether user-facing documentation should be edited now or
whether the evidence supports retaining the current public wording.

## Decision

No public documentation edits were made on Day 6.

The Day 2 outcome ledger and Day 3 evidence conflict review show that the
implementation sprints expected to create new Epic 18 support evidence are not
available on this branch yet. The Day 4 public claim audit also found that the
current public docs already route users to the support/readiness source of truth
and retain non-claims for package managers, Windows report freshness,
benchmarks, generated API HTML, shared-library ABI support, release readiness,
and broad state-of-the-art parity.

Editing public docs before those later evidence artifacts exist would risk
turning planning intent into a user-facing support claim. Day 6 therefore
records a deliberate no-promotion decision instead of changing public wording.

## Public Claim Surface Review

| Surface | Day 6 finding | Day 6 action | Future edit trigger |
| --- | --- | --- | --- |
| `README.md` | Top-level package, Windows, comparison, benchmark, generated API, ABI, release, and state-of-the-art caveats remain aligned with current evidence. | Retained as-is. | Later sprint evidence promotes a specific support tier or removes a residual. |
| `INSTALL.md` | The support readiness matrix remains the public source of truth and explicitly separates supported, validated, hosted-evidence, guarded-workflow, local-only, deferred, and unclaimed surfaces. | Retained as-is. | Package, Windows, benchmark, API publication, shared-library, or ABI evidence changes. |
| `benchmarks/README.md` | Benchmark wording remains methodology-bound and does not turn local or selected hosted rows into portable performance claims. | Retained as-is. | Sprint 202 or later adds a reviewed hosted platform/row with updated methodology metadata. |
| `docs/api_reference.md` | Generated API HTML remains local-only and checked-in public headers remain the source of truth. | Retained as-is. | Sprint 204 or later changes publication, artifact, hosting, or freshness policy. |
| `docs/solver_selection.md` | Solver guidance keeps comparison/oracle examples scoped to selected fixtures and avoids package, ABI, platform, performance, or state-of-the-art claims. | Retained as-is. | Adoption simplification work changes support routing or selected solver guidance. |
| `tests/corpus/README.md` | Corpus docs retain selected target, report freshness, Windows, package, ABI, performance, release, and state-of-the-art boundaries. | Retained as-is. | Selected target manifest or report-index semantics change. |
| `tests/corpus/schemas/report_index_fields.md` | Schema documentation continues to require exact metadata before selected target promotion. | Retained as-is. | Manifest fields, support tiers, workflow platforms, freshness policy, or selected artifact paths change. |

## Retained Public Non-Claims

- Package-manager support remains unclaimed, including Homebrew/core,
  Linuxbrew, bottles, taps, vcpkg, Conan, pkgsrc, and distro/system packages.
- Shared-library packaging and dynamic ABI support remain deferred.
- Windows support remains bounded to the reviewed MSVC/CMake static-first path;
  Windows Makefile parity, Windows `pkg-config` execution parity, runtime
  loader behavior, and broad Windows parity are not claimed.
- The Windows selected Cholesky comparison path remains a guarded workflow path
  until hosted evidence, manifest metadata, support tier, and claim contract are
  reviewed together.
- Broad Windows report freshness, Windows selected oracle freshness, Windows
  selected benchmark freshness, and unselected Windows comparison families
  remain unclaimed.
- Benchmark rows remain methodology-bound evidence and do not create portable
  timing thresholds, speedup guarantees, release benchmark claims, or
  state-of-the-art performance claims.
- Generated API HTML remains local-only and not hosted, artifact-published, or
  source-controlled as release evidence.
- Allocation-failure proof remains selected and local-only, not broad OOM or
  platform reliability proof.
- Broad ecosystem parity with SuiteSparse, PETSc, Trilinos, Eigen, SciPy, NumPy,
  LAPACK, or other external libraries is not claimed.
- Release readiness and unqualified state-of-the-art sparse linear algebra
  status remain unclaimed.

## Support and Readiness Routing

| User need | Public route retained |
| --- | --- |
| Install and support status | `INSTALL.md#support-readiness-matrix` |
| First-contact capability overview | `README.md` |
| Benchmark interpretation | `benchmarks/README.md` |
| API source of truth and generated HTML policy | `docs/api_reference.md` |
| Solver choice and selected comparison/oracle interpretation | `docs/solver_selection.md` |
| Report target and corpus row semantics | `tests/corpus/README.md` and `tests/corpus/schemas/report_index_fields.md` |

## Item 206.2 Evidence

Day 6 closes the public-documentation portion of item 206.2 for this branch by
confirming that no stronger public claim is supported by current evidence and
that existing public docs already carry the required caveats. Maintainer/API
recalibration remains a separate Day 7 task.

## Validation Notes

- Reviewed the public claim audit from Day 4 before deciding not to edit public
  docs.
- Re-scanned README, INSTALL, benchmark, API, solver-selection, corpus, and
  report-index schema docs for support/readiness, Windows, PowerShell,
  Homebrew, benchmark, generated API, ABI, package, and state-of-the-art
  wording.
- No production code, public headers, public docs, generated reports, or schema
  files were edited on Day 6.
