# Sprint 104 Retrospective

**Sprint:** 104 - Performance Backend & Parallel Runtime Modernization
**Duration:** 14 days (Days 1-14 landed on branch `sprint-104`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 104 started from the Epic 10 project-plan scope and the Sprint
      100 benchmark/sentinel evidence rules.
- [x] backend consumers, optional dense backend seams, OpenMP controls,
      benchmark fields, and reporting surfaces were audited before changes.
- [x] builtin dense kernels were preserved as the portable baseline.
- [x] invalid optional dense backend requests now have focused fallback tests:
  - Cholesky CSC supernodal dense kernels
  - LDLT CSC dense factorization
- [x] OpenMP runtime ownership is documented beside the relevant source
      parallel regions and in maintainer/user docs.
- [x] `make performance-sentinels` landed as a bounded local sentinel bundle.
- [x] benchmark reporting language now distinguishes:
  - local measurements
  - hard regression gates
  - optional backend context
  - OpenMP runtime context
  - reviewed CI/platform scope
- [x] canonical benchmark report metadata now uses `category=measurement`.
- [x] cross-platform runtime review documented the POSIX 54-test CMake surface
      and the Windows 51-test reviewed subset.
- [x] final validation passed:
  - `bash -n scripts/performance_sentinels.sh && bash -n scripts/bench_canonical_report.sh`
  - focused affected tests
  - `make bench-canonical-report`
  - `make performance-sentinels`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scans
- [x] Sprint 105 handoff items are explicit and bounded.

## What Went Well

1. **The sprint started with claim boundaries.**
   Day 1 carried forward the Sprint 100 evidence templates and Sprint 102-103
   comparison discipline. That kept the sprint focused on backend/runtime
   truth rather than performance-superiority language.

2. **The backend audit found the right seams.**
   Day 2 separated Cholesky CSC dense kernels, LDLT CSC dense factorization,
   public backend selectors, eigensolver telemetry, OpenMP regions, and
   benchmark-only diagnostics before code changes started.

3. **Builtin fallback became test-owned behavior.**
   Day 5 added focused tests for invalid `SPARSE_CHOL_DENSE_BACKEND` and
   `SPARSE_LDLT_DENSE_BACKEND` values. The tests make the portable fallback
   contract executable instead of relying on prose alone.

4. **OpenMP ownership is now visible near the implementation.**
   Day 7 added source comments beside SpMV/block-SpMV and eigensolver MGS
   parallel regions. The comments match the docs: the library does not own a
   public thread-count API or translate `SPARSE_*` settings into OpenMP teams.

5. **The sentinel work stayed conservative.**
   Day 8 designed a broader sentinel set, but Day 9 implemented only the
   smallest useful batch: S5 `wall-check` as the existing hard gate and S2
   Cholesky CSC rows as threshold-free report context.

6. **Benchmark reporting was corrected before closeout.**
   Day 10 identified wording risks, and Day 11 aligned README, benchmark docs,
   maintainer docs, algorithm notes, and canonical report metadata.

7. **The platform review kept CI scope honest.**
   Day 12 documented Linux, macOS, Windows, local Make, local CMake, OpenMP,
   and sentinel roles without turning supplemental lanes into reviewed parity
   claims.

8. **The final code-touch gate passed.**
   Day 13 ran the full `make format && make lint && make test` chain after
   focused backend/runtime/report checks. That gives Sprint 104 a clean
   validation anchor.

## What Didn't Go Well

1. **Performance sentinel expansion remains incomplete by design.**
   S1, S3, and S4 were selected as useful future lanes, but Sprint 104 did not
   add hard thresholds or local baselines for them. That restraint is correct,
   but the sentinel suite is still a first batch rather than a full program.

2. **Optional backend observability is still mostly benchmark/test-local.**
   Cholesky and LDLT dense backend context is clearer, but Sprint 104 did not
   create a public selected/fallback telemetry API. Public wording must still
   stay narrow.

3. **OpenMP proof remains intentionally asymmetric.**
   The code and docs are clearer, but runtime evidence still depends on local
   or CI-specific OpenMP behavior. Nested parallelism, runtime affinity, and
   optional BLAS/OpenMP interactions remain non-claims.

4. **Benchmark language needs continual maintenance.**
   The `category=proof` label in the canonical report showed how easily
   generated metadata can drift into overclaiming. Future report fields need
   the same review discipline.

5. **Windows scope still requires manual count discipline.**
   The Windows reviewed subset is clear, but any future test additions must
   update the expected CTest count only with an explicit staged-scope decision.

## Final Metrics

### Validation

| Metric | Sprint 104 close state |
|---|---:|
| full branch-level gate | `make format && make lint && make test` passed |
| focused Cholesky CSC supernodal test | 62 tests, 0 failures, 8170 assertions |
| focused LDLT test | 89 tests, 0 failures, 912 assertions |
| focused OpenMP status test | 12 tests, 0 failures, 831 assertions |
| focused eigensolver test | 31 tests, 0 failures, 310 assertions |
| POSIX CMake registered tests | 54 |
| Windows reviewed CTest count | 51 |
| canonical report metadata | `category=measurement` |
| sentinel hard-gate rows | S5 `wall-check` pass rows |
| sentinel report-only rows | S2 Cholesky CSC threshold-free rows |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scans | passed on touched source, docs, scripts, and Sprint 104 artifacts |

### Sprint 104 Artifact Package

| Metric | Sprint 104 close state |
|---|---:|
| total artifact files under `SPRINT_104/artifacts/` | 15 |
| baseline/audit/contract artifacts | 4 |
| implementation/design/review artifacts | 6 |
| reporting/platform/validation/closeout artifacts | 5 |

Notes:

- baseline, audit, and contract artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-runtime-baseline.md`
  - `day2-backend-consumer-audit.md`
  - `day3-runtime-contract-design.md`
- implementation, design, and review artifacts:
  - `day4-descriptor-surface-boundary.md`
  - `day5-backend-descriptor-batch.md`
  - `day6-openmp-threading-audit.md`
  - `day7-threading-cleanup.md`
  - `day8-performance-sentinel-design.md`
  - `day9-performance-sentinel-batch.md`
- reporting, platform, validation, and closeout artifacts:
  - `day10-benchmark-reporting-audit.md`
  - `day11-benchmark-reporting-alignment.md`
  - `day12-cross-platform-runtime-review.md`
  - `day13-validation-reconciliation.md`
  - `day14-closeout-and-handoff.md`

### Landed Surface

| Metric | Sprint 104 close state |
|---|---:|
| library source files touched | 2 |
| test files touched | 2 |
| Make/script files touched | 3 |
| public/maintainer docs touched | 4 |
| new Make targets | 1 |
| new helper scripts | 1 |
| new focused fallback tests | 2 |

## Residual Deferred Debt

Most important carry-forward work:

- same-worktree or local-baseline design before hard thresholds for S1/S3/S4
- optional backend telemetry only if future public consumers need it
- OpenMP runtime and nested-parallelism evidence only with fresh validation
- benchmark field/metadata wording review whenever report schemas change
- Windows expected CTest count updates tied to explicit reviewed-scope changes

Still consciously constrained rather than silently solved:

- no portable timing superiority claim
- no broad vendor-backend parity
- no required optional dense acceleration
- no public thread-pool or library-owned OpenMP team-size API
- no tuned nested-parallelism claim
- no Windows Makefile, benchmark, fuzz/property, or install-validation parity
- no hard performance thresholds for S1/S2/S3/S4 beyond existing S5
- no benchmark residual/agreement fields as correctness substitutes

Not carried forward as unresolved Sprint 104 debt:

- backend consumer audit
- runtime contract design
- descriptor boundary decision
- invalid Cholesky/LDLT dense backend fallback tests
- OpenMP/threading audit and comment cleanup
- first sentinel bundle implementation
- benchmark reporting audit and alignment
- cross-platform runtime review
- final validation reconciliation
- closeout and Sprint 105 handoff

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-runtime-baseline.md](./artifacts/day1-runtime-baseline.md)
- [day2-backend-consumer-audit.md](./artifacts/day2-backend-consumer-audit.md)
- [day3-runtime-contract-design.md](./artifacts/day3-runtime-contract-design.md)
- [day4-descriptor-surface-boundary.md](./artifacts/day4-descriptor-surface-boundary.md)
- [day5-backend-descriptor-batch.md](./artifacts/day5-backend-descriptor-batch.md)
- [day6-openmp-threading-audit.md](./artifacts/day6-openmp-threading-audit.md)
- [day7-threading-cleanup.md](./artifacts/day7-threading-cleanup.md)
- [day8-performance-sentinel-design.md](./artifacts/day8-performance-sentinel-design.md)
- [day9-performance-sentinel-batch.md](./artifacts/day9-performance-sentinel-batch.md)
- [day10-benchmark-reporting-audit.md](./artifacts/day10-benchmark-reporting-audit.md)
- [day11-benchmark-reporting-alignment.md](./artifacts/day11-benchmark-reporting-alignment.md)
- [day12-cross-platform-runtime-review.md](./artifacts/day12-cross-platform-runtime-review.md)
- [day13-validation-reconciliation.md](./artifacts/day13-validation-reconciliation.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)
- [Benchmark guide](../../../../benchmarks/README.md)
- [Maintainer guide](../../../maintainer_guide.md)

## Bottom Line

Sprint 104 achieved its goal:

- backend/runtime surfaces were audited before implementation;
- builtin fallback remains the portable product truth;
- invalid optional Cholesky and LDLT dense backend requests are covered by
  tests;
- OpenMP runtime ownership is clearer in source and docs;
- bounded local performance sentinels now exist without overclaiming timing
  portability;
- benchmark reporting and canonical metadata now use measurement language;
- cross-platform CI and CTest scope are documented accurately;
- final validation passed before closeout;
- Sprint 105 receives explicit guardrails for benchmark, reordering, graph, and
  large-matrix scalability work.
