# Sprint 103 Retrospective

**Sprint:** 103 - Iterative, Eigensolver & SVD External Comparisons
**Duration:** 14 days (Days 1-14 landed on branch `sprint-103`)
**Status:** Complete

> Note: this retrospective is stored at the requested path
> `docs/planning/EPIC_10/SPRINT_99/RETROSPECTIVE.md`. The sprint artifacts it
> summarizes live under `docs/planning/EPIC_10/SPRINT_103/`.

## Definition Of Done Checklist

- [x] Sprint 103 started from the Epic 10 project-plan scope and Sprint 102
      direct-solver evidence boundary.
- [x] iterative, eigensolver, thick-restart, LOBPCG, and SVD comparison gaps
      were audited before implementation.
- [x] solver-family evidence was ranked by user impact, comparison weakness,
      numerical risk, and validation cost.
- [x] convergence fixture taxonomy and helper/reporting boundaries were
      frozen before code changes.
- [x] BiCGSTAB received bounded deterministic and internal-consistency
      comparison evidence.
- [x] LOBPCG received closed-form Laplacian residual/orthogonality evidence
      and a fixture-local preconditioner comparison on `bcsstk04`.
- [x] thick-restart eigensolver evidence gained an exact diagonal fixture with
      residual, orthogonality, and bounded peak-basis checks.
- [x] SVD received a deterministic full-UV, reconstruction, orthogonality, and
      rank-threshold fixture.
- [x] maintainer documentation now distinguishes deterministic fixture
      evidence, direct-solver cross-checks, internal consistency, residual and
      orthogonality quality evidence, external dense-reference lanes, and
      absent external package parity.
- [x] every implemented comparison has a maintained test owner, artifact trail,
      validation record, and explicit non-claim boundary.
- [x] final validation passed:
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan across touched Sprint 103 code and documentation
- [x] Sprint 104 prerequisites, deferred work, and external-helper risks were
      recorded explicitly.

## What Went Well

1. **The sprint followed the Day 2 evidence ranking.**
   BiCGSTAB was the highest-risk iterative gap and landed first. LOBPCG and
   thick-restart followed as the highest-value spectral lanes, and SVD waited
   until the spectral residual and orthogonality pattern was established.

2. **The implementation stayed bounded to existing test owners.**
   The new comparison evidence landed in `tests/test_bicgstab.c`,
   `tests/test_eigs_lobpcg.c`, `tests/test_eigs_thick_restart.c`, and
   `tests/test_svd.c`. No public headers, library sources, Makefiles, CMake
   files, or external helpers were widened for Sprint 103.

3. **BiCGSTAB gained a useful nonsymmetric evidence set.**
   Sprint 103 added a deterministic known-solution fixture checked against LU,
   a `steam1` BiCGSTAB+ILU versus GMRES(30)+ILU internal comparison, and an
   expected non-convergence budget boundary.

4. **Spectral evidence now separates value, residual, and basis quality.**
   LOBPCG and thick-restart tests check eigenvalue agreement separately from
   Ritz residuals and orthogonality. The thick-restart diagonal fixture also
   avoids treating grow-`m` parity as independent proof.

5. **SVD evidence inherited the same discipline.**
   The SVD follow-through checks exact singular values, full-mode
   reconstruction residual, U/Vt orthogonality, and rank thresholds as separate
   criteria on one deterministic fixture.

6. **Claim boundaries were documented before closeout.**
   `docs/maintainer_guide.md` now gives future public-documentation edits a
   source of truth for residual interpretation and non-claims around PETSc,
   SciPy, Trilinos, ARPACK, LAPACK, NumPy, and broad ecosystem parity.

7. **The full code-touch gate passed twice late in the sprint.**
   Day 13 reconciled evidence after `make format && make lint && make test`,
   and Day 14 reran the same full gate before closeout. Both runs ended with
   `All tests passed.`

## What Didn't Go Well

1. **No external helper-backed package parity landed.**
   Sprint 103 intentionally stayed with deterministic fixtures, direct-solver
   cross-checks, and internal consistency. That keeps the evidence reliable,
   but external PETSc/SciPy/Trilinos/ARPACK/LAPACK/NumPy comparison remains
   future work.

2. **Several solver families remain documentation or taxonomy candidates.**
   MINRES and CG were ranked lower because existing coverage is stronger, but
   they still need consolidated convergence-profile artifacts if later public
   wording depends on them.

3. **The grow-`m` eigensolver still lacks its own close comparison artifact.**
   Day 12 documented general residual rules, but no grow-`m`-specific external
   or deterministic follow-through was added in this sprint.

4. **Iteration counts need repeated guardrails.**
   LOBPCG and BiCGSTAB tests report useful local iteration counts, but Sprint
   103 artifacts repeatedly need to state that these are fixture-local
   diagnostics rather than portable performance claims.

5. **The requested retrospective path is historical, not semantic.**
   The retrospective is stored under `SPRINT_99` by request, while all Sprint
   103 artifacts live under `SPRINT_103`. The note at the top of this file is
   necessary to avoid future navigation confusion.

## Final Metrics

### Validation

| Metric | Sprint 103 close state |
|---|---:|
| full branch-level gate | `make format && make lint && make test` passed |
| final test summary | `All tests passed.` |
| focused BiCGSTAB binary after implementation | `61` tests, `0` failures, `466` assertions |
| focused LOBPCG binary after implementation | `27` tests, `0` failures, `247` assertions |
| focused thick-restart binary after implementation | `21` tests, `0` failures, `285` assertions |
| focused SVD binary after implementation | `98` tests, `0` failures, `1093` assertions |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scans | passed on touched Sprint 103 code and documentation |

### Sprint 103 Artifact Package

| Metric | Sprint 103 close state |
|---|---:|
| total artifact files under `SPRINT_103/artifacts/` | `15` |
| baseline/audit/taxonomy artifacts | `4` |
| iterative design/implementation/closeout artifacts | `3` |
| eigensolver design/implementation/closeout artifacts | `3` |
| SVD/documentation/reconciliation/closeout artifacts | `5` |
| working-notes line count at closeout | `907` |

Notes:

- baseline, audit, taxonomy, and helper-boundary artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-scope-baseline.md`
  - `day2-solver-family-comparison-audit.md`
  - `day3-convergence-fixture-taxonomy.md`
  - `day4-helper-reporting-boundary.md`
- iterative artifacts:
  - `day5-iterative-oracle-batch-design.md`
  - `day6-iterative-oracle-batch-implementation.md`
  - `day7-iterative-closeout-and-rerank.md`
- spectral and SVD artifacts:
  - `day8-eigensolver-oracle-batch-design.md`
  - `day9-eigensolver-oracle-batch-implementation.md`
  - `day10-spectral-closeout-and-svd-scope.md`
  - `day11-svd-comparison-follow-through.md`
- documentation, reconciliation, and closeout artifacts:
  - `day12-reporting-and-documentation-update.md`
  - `day13-validation-and-evidence-reconciliation.md`
  - `day14-closeout-and-handoff.md`

### Landed Evidence Surface

| Metric | Sprint 103 close state |
|---|---:|
| focused C test files updated | `4` |
| maintainer documentation files updated | `1` |
| new BiCGSTAB comparison lanes | `3` |
| new or strengthened LOBPCG comparison lanes | `2` |
| new thick-restart deterministic fixture lanes | `1` |
| new SVD deterministic rank/full-UV fixture lanes | `1` |
| external package helper lanes added | `0` |

## Residual Deferred Debt

Most important carry-forward work:

- external-helper policy for iterative, eigensolver, and SVD oracle lanes
- BiCGSTAB external helper-backed comparison after helper policy is frozen
- LOBPCG or thick-restart external spectral oracle after ARPACK/SciPy-style
  scope is bounded
- SVD external LAPACK/NumPy/SciPy comparison, starting with singular values and
  rank before full vector parity
- MINRES consolidated comparison artifact
- CG convergence-profile consumer lane
- grow-`m` eigensolver residual narrative and comparison artifact
- benchmark or iteration-count claims only after a future competitive
  calibration sprint owns them

Still consciously constrained rather than silently solved:

- no broad PETSc, SciPy, Trilinos, ARPACK, LAPACK, NumPy, or package-wide
  parity
- no claim that internal solver comparisons are independent external oracles
- no portable iteration-count or runtime superiority claim
- no broad nonsymmetric iterative proof from one BiCGSTAB batch
- no broad eigensolver proof from one LOBPCG and one thick-restart batch
- no broad SVD correctness proof from one deterministic full-UV fixture
- no external helper availability, skip, version, or CI-role contract
- no public wording beyond maintained test-owner and artifact evidence

Not carried forward as unresolved Sprint 103 debt:

- solver-family comparison audit
- fixture taxonomy
- helper/reporting boundary
- BiCGSTAB deterministic and internal-consistency comparison batch
- LOBPCG residual and orthogonality comparison batch
- thick-restart exact diagonal comparison batch
- SVD diagonal rank/full-UV comparison
- maintainer evidence-boundary documentation
- validation and evidence reconciliation
- closeout and Sprint 104 handoff

## Key Deliverables

- [PLAN.md](../SPRINT_103/PLAN.md)
- [WORKING_NOTES.md](../SPRINT_103/WORKING_NOTES.md)
- [day1-scope-baseline.md](../SPRINT_103/artifacts/day1-scope-baseline.md)
- [day2-solver-family-comparison-audit.md](../SPRINT_103/artifacts/day2-solver-family-comparison-audit.md)
- [day3-convergence-fixture-taxonomy.md](../SPRINT_103/artifacts/day3-convergence-fixture-taxonomy.md)
- [day4-helper-reporting-boundary.md](../SPRINT_103/artifacts/day4-helper-reporting-boundary.md)
- [day5-iterative-oracle-batch-design.md](../SPRINT_103/artifacts/day5-iterative-oracle-batch-design.md)
- [day6-iterative-oracle-batch-implementation.md](../SPRINT_103/artifacts/day6-iterative-oracle-batch-implementation.md)
- [day7-iterative-closeout-and-rerank.md](../SPRINT_103/artifacts/day7-iterative-closeout-and-rerank.md)
- [day8-eigensolver-oracle-batch-design.md](../SPRINT_103/artifacts/day8-eigensolver-oracle-batch-design.md)
- [day9-eigensolver-oracle-batch-implementation.md](../SPRINT_103/artifacts/day9-eigensolver-oracle-batch-implementation.md)
- [day10-spectral-closeout-and-svd-scope.md](../SPRINT_103/artifacts/day10-spectral-closeout-and-svd-scope.md)
- [day11-svd-comparison-follow-through.md](../SPRINT_103/artifacts/day11-svd-comparison-follow-through.md)
- [day12-reporting-and-documentation-update.md](../SPRINT_103/artifacts/day12-reporting-and-documentation-update.md)
- [day13-validation-and-evidence-reconciliation.md](../SPRINT_103/artifacts/day13-validation-and-evidence-reconciliation.md)
- [day14-closeout-and-handoff.md](../SPRINT_103/artifacts/day14-closeout-and-handoff.md)
- [Sprint 103 maintainer evidence boundary](../../../maintainer_guide.md)

## Bottom Line

Sprint 103 achieved its goal:

- iterative, spectral, and SVD comparison gaps were audited and ranked before
  implementation
- BiCGSTAB, LOBPCG, thick-restart, and SVD received bounded maintained
  comparison evidence
- residual, orthogonality, reconstruction, rank, and expected non-convergence
  criteria are documented by fixture and owner
- maintainer documentation now prevents broad external-parity overclaims
- final validation passed before closeout
- Sprint 104 receives external-helper policy, oracle independence, and
  package-parity comparison work as explicit future prerequisites, not implied
  Sprint 103 claims
