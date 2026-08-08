# Sprint 142 Retrospective

**Sprint:** 142 - Runtime Backend Governance & Sentinel Expansion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-142`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 142 day-by-day plan, working notes, artifact directory,
      and closeout artifact.
- [x] Consumed the Sprint 141 `runtime_backend` defer row as a governance
      handoff rather than unfinished report-index work.
- [x] Audited runtime/backend controls across typed options, compatibility
      environment variables, build flags, runtime context, benchmark/report
      commands, and documentation.
- [x] Documented Cholesky, LDLT, eigensolver, analysis/reorder, dense-helper,
      OpenMP, graph/ND/FM, direct-solver analysis, sentinel, and normalized
      report-index control surfaces.
- [x] Defined the maintained precedence contract for explicit typed options,
      AUTO/default semantics, compatibility environment overrides,
      compile-time flags, fallback behavior, and maintainer/report context.
- [x] Completed the typed-control batch as an explicit non-expansion:
      existing public typed controls remain the public caller-facing surface,
      while dense-helper selectors, SVD low-rank env selection, FM/debug
      variables, OpenMP runtime context, package/link controls, and
      benchmark/test opt-ins remain maintainer-only or deferred.
- [x] Designed runtime/backend sentinel expansion without portable timing,
      package, ABI, platform, optional-backend, or state-of-the-art claims.
- [x] Added advisory `S3` LDLT KKT backend context rows to
      `make performance-sentinels` through `bench_refactor_csc
      --indefinite-kkt --repeat 1`.
- [x] Preserved `S5` as the only hard local sentinel gate and preserved `S2`
      as threshold-free Cholesky CSC advisory context.
- [x] Added normalized report-index synthetic coverage for `S3` rows,
      including advisory status and backend request/selected/fallback field
      preservation.
- [x] Updated README, benchmark docs, algorithm docs, cookbook, maintainer
      guide, and report-index field docs with runtime/backend boundaries and
      sentinel interpretation.
- [x] Published earned runtime/backend claims, remaining non-claims, and
      concrete Sprint 143 package/ABI prerequisites.
- [x] Published final validation evidence and generated-output hygiene.
- [x] Ran Sprint 142 validation:
  - focused backend/precedence builds:
    `make build/test_chol_csc build/test_ldlt_backend_dispatch
    build/test_eigs_thick_restart build/test_eigs_lobpcg
    build/test_reorder_nd build/test_ldlt`;
  - focused C tests:
    `./build/test_chol_csc`, `./build/test_ldlt_backend_dispatch`,
    `./build/test_eigs_thick_restart`, `./build/test_eigs_lobpcg`,
    `./build/test_reorder_nd`, and `./build/test_ldlt`;
  - `make performance-sentinels`;
  - `bash -n scripts/performance_sentinels.sh`;
  - `python3 -m py_compile tests/test_normalize_report_index.py
    scripts/normalize_report_index.py scripts/validate_corpus_schema.py`;
  - `python3 tests/test_normalize_report_index.py`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `python3 scripts/normalize_report_index.py --family sentinel --output
    build/report-index/normalized-index.tsv`;
  - `python3 scripts/normalize_report_index.py --family sentinel
    --check-freshness`;
  - `python3 scripts/normalize_report_index.py --family benchmark --family
    sentinel --family guardrail --check-freshness`;
  - `make format && make lint`;
  - `git diff --check`;
  - trailing-whitespace scans.
- [x] No C or header files changed in the final diff, so the sprint did not
      require a full `make test` run.

## What Went Well

1. **The sprint turned runtime/backend behavior into an explicit contract.**
   The audit and precedence artifacts now describe which controls are public
   typed options, which controls are compatibility or maintainer-only, and how
   fallback/default behavior should be interpreted.

2. **The typed-control decision stayed conservative.** Instead of adding API
   surface prematurely, Sprint 142 recorded a deliberate non-expansion. That
   keeps ABI and package questions out of runtime governance and gives Sprint
   143 a cleaner starting point.

3. **The sentinel expansion was narrow and useful.** `S3` adds LDLT KKT
   backend visibility to the existing `performance-sentinels` bundle without
   creating a new report family, a hard timing gate, or a portable performance
   claim.

4. **The docs now distinguish control surfaces clearly.** README,
   maintainer, benchmark, cookbook, algorithm, and report-index docs all use
   the same split between public typed controls, maintainer/build/report
   controls, and local-only sentinel evidence.

5. **Validation stayed proportional to the touched surface.** Focused C tests
   proved the existing backend/precedence owners, while script/report/docs
   changes received Python, shell, report-index, freshness, lint, and hygiene
   coverage.

## What Didn't Go Well

1. **The runtime/backend vocabulary is still subtle.** LDLT selected-backend
   wording has to distinguish top-level dispatch from internal CSC completion
   fallback. The Day 3 and Day 4 artifacts are important references for that
   nuance.

2. **Environment controls remain tempting to over-document.** Dense-helper,
   SVD low-rank, FM/debug, OpenMP runtime, and benchmark/test opt-in variables
   are useful, but publishing them as stable API would overclaim the evidence.
   The sprint had to make several explicit deferrals.

3. **Sentinel freshness can still look noisier than the actual risk.** `S2`
   and `S3` advisory rows compare fresh against the local manifest, while
   existing `S5` hard-gate rows remain `generated_present_unchecked` in the
   normalizer output. The diagnostics pass, but maintainers need to read the
   row meanings.

4. **A validation command was initially malformed on Day 12.** Attempting to
   run Python bytecode compilation on `scripts/performance_sentinels.sh`
   failed because it is a shell script. The corrected shell syntax and Python
   compile checks passed and the invalid attempt was recorded.

## Final Metrics

### Validation

| Metric | Sprint 142 close state |
| --- | --- |
| tracked `.c`/`.h` changes | no |
| full C `make test` required | no |
| focused backend/precedence C tests | passed |
| `make performance-sentinels` | passed |
| `make format && make lint` | passed |
| shell syntax check | passed |
| Python compile checks | passed |
| `python3 tests/test_normalize_report_index.py` | passed |
| `python3 scripts/validate_corpus_schema.py` | passed |
| sentinel normalized index | passed: `21` rows |
| sentinel freshness check | passed: `21` rows |
| benchmark/sentinel/guardrail freshness check | passed: `25` rows |
| generated normalized-index file tracked | no |
| generated normalized-index file ignored | yes: `build/report-index/normalized-index.tsv` |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |

### Artifact Package

| Metric | Sprint 142 close state |
| --- | ---: |
| daily artifacts under `SPRINT_142/artifacts/` | 14 |
| final retrospective files | 1 |
| shell scripts changed | 1 |
| Python test files changed | 1 |
| report-index schema docs changed | 1 |
| public/maintainer documentation surfaces changed | 5 |
| build files changed | 1 |
| source-controlled generated report files | 0 |

## Closed Claim

Sprint 142 closes this claim:

The project now has a maintained runtime/backend governance contract for the
current public typed controls, explicit maintainer-only/deferred control
classification, and normalized local sentinel visibility for selected
runtime/backend rows without widening package, ABI, platform, backend
portability, portable performance, or state-of-the-art claims.

This claim is supported by:

- `docs/planning/EPIC_12/SPRINT_142/artifacts/day2-runtime-control-inventory.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day3-backend-dispatch-audit.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day4-precedence-contract-design.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day5-precedence-contract-implementation.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day6-typed-control-selection.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day7-typed-control-batch.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day8-runtime-sentinel-design.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day9-sentinel-implementation.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day10-runtime-docs-alignment.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day11-focused-runtime-validation.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day12-quality-gate.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day13-claim-closure-and-sprint143-handoff.md`;
- `docs/planning/EPIC_12/SPRINT_142/artifacts/day14-closeout-validation-summary.md`;
- `scripts/performance_sentinels.sh`;
- `tests/test_normalize_report_index.py`;
- `README.md`;
- `benchmarks/README.md`;
- `docs/algorithm.md`;
- `docs/cookbook.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/schemas/report_index_fields.md`.

## Sprint 143 Readiness

Sprint 143 should consume the Day 13 handoff as its starting point:

| Handoff field | Sprint 143 requirement |
| --- | --- |
| runtime/backend public-control boundary | Treat existing typed controls as caller-facing behavior and env/build/report controls as non-API unless Sprint 143 explicitly promotes a package-relevant control. |
| static-first install baseline | Audit static archive install, installed headers, `pkg-config`, CMake export, exact-version config, and shared-library rejection before the product decision. |
| shared-library decision gate | Choose one path: implement shared-library ABI support with symbol/export/loader proof, or strengthen static-first-only deferral wording and guards. |
| downstream consumer proof | Revalidate `tests/test_install.sh`, `tests/test_cmake_install.sh`, CMake package config, `pkg-config`, version constraints, and unsupported-artifact checks. |
| sentinel non-claim boundary | Do not use `S2` or `S3` timing rows as package, ABI, platform, or portable performance proof. |
| platform tier separation | Keep macOS/Windows support-tier promotion routed to Sprint 144 rather than folding it into package/ABI claims. |

## Residual Deferred Debt

Most important carry-forward work:

- Sprint 143 shared-library ABI versus stricter static-first product decision;
- package/install/export downstream consumer proof refresh;
- CMake/pkg-config metadata review under the selected package decision;
- Sprint 144 platform promotion lane selection and proof;
- possible future typed-control promotion for currently maintainer-only
  controls, if a later sprint earns API and ABI evidence;
- possible future runtime/backend sentinel rows for eigensolver AUTO,
  shift-invert LDLT, or OpenMP context, if a maintained command and non-claim
  boundary are defined first.

Still consciously constrained rather than silently solved:

- no shared-library ABI or dynamic-loader claim;
- no package-manager availability claim;
- no macOS/Windows platform parity claim;
- no portable performance claim;
- no optional dense-kernel availability claim;
- no broad backend portability claim;
- no broad solver correctness or corpus-completeness claim;
- no state-of-the-art claim;
- no hosted CI proof from local generated rows.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-runtime-governance-intake.md](./artifacts/day1-runtime-governance-intake.md)
- [day2-runtime-control-inventory.md](./artifacts/day2-runtime-control-inventory.md)
- [day3-backend-dispatch-audit.md](./artifacts/day3-backend-dispatch-audit.md)
- [day4-precedence-contract-design.md](./artifacts/day4-precedence-contract-design.md)
- [day5-precedence-contract-implementation.md](./artifacts/day5-precedence-contract-implementation.md)
- [day6-typed-control-selection.md](./artifacts/day6-typed-control-selection.md)
- [day7-typed-control-batch.md](./artifacts/day7-typed-control-batch.md)
- [day8-runtime-sentinel-design.md](./artifacts/day8-runtime-sentinel-design.md)
- [day9-sentinel-implementation.md](./artifacts/day9-sentinel-implementation.md)
- [day10-runtime-docs-alignment.md](./artifacts/day10-runtime-docs-alignment.md)
- [day11-focused-runtime-validation.md](./artifacts/day11-focused-runtime-validation.md)
- [day12-quality-gate.md](./artifacts/day12-quality-gate.md)
- [day13-claim-closure-and-sprint143-handoff.md](./artifacts/day13-claim-closure-and-sprint143-handoff.md)
- [day14-closeout-validation-summary.md](./artifacts/day14-closeout-validation-summary.md)

## Closeout

Sprint 142 is complete. It closes runtime/backend governance with an explicit
control inventory, precedence contract, typed-control deferral decision,
advisory LDLT KKT sentinel rows, aligned documentation, validation evidence,
and a concrete Sprint 143 package/ABI handoff. It does not promote local
sentinel timing rows into portable performance evidence or widen public claims
beyond the validated runtime/backend governance surface.
