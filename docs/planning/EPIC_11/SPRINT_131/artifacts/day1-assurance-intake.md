# Sprint 131 Day 1 - Assurance Intake

## Purpose

Day 1 establishes Sprint 131 scope, source-area intake, day-level owners,
artifact structure, validation expectations, and duplicate fences for the
numerical corpus, coverage, benchmark, dead-code, large-matrix, oracle, and
guardrail surfaces.

Sprint 131 starts from the Sprint 120-130 oracle taxonomy and residual claim
gates. It is not a test-expansion day. The main outcome is a baseline that
lets later days inventory and index existing evidence without promoting smoke,
timing, optional, or supplemental outputs into broader reviewed claims.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 131 project-plan section | Defines seven items for corpus inventory, taxonomy, report-index design, coverage architecture, generated index work, validation, and closeout. |
| Sprint 131 plan | Defines the 14-day sequence and 168-hour budget. |
| Sprint 118 oracle and coverage templates | Provide reusable artifact shapes for evidence, coverage interpretation, validation commands, and non-claims. |
| Sprint 120-130 artifacts and retrospectives | Preserve oracle taxonomy, external-reference trust boundaries, residual lanes, optional-corpus support-tier decisions, helper ownership, and solver-selection claim gates. |
| `docs/maintainer_guide.md` | Current maintainer policy for benchmark reports, coverage, dead-code, large-matrix guardrails, and non-claim interpretation. |
| `benchmarks/README.md` | Benchmark command groups, CSV fields, canonical report outputs, performance sentinel outputs, and large-matrix guardrail report semantics. |
| `docs/matrix_market.md` | Matrix Market user-facing behavior documentation. |
| `Makefile` | Owns benchmark, report, quality, dead-code, coverage, and large-matrix guardrail targets. |
| `tests/data/` | Checked-in Matrix Market and SuiteSparse-derived fixture source area. |
| `tests/*_external_dense_reference.py` | External dense-reference helper protocols and bounded oracle source area. |
| `tests/test_*.c` and helper headers | Generated fixtures, known matrices, expected failures, skips, optional gates, and current test ownership source area. |
| `benchmarks/` | Benchmark driver and CSV schema implementation source area. |
| `scripts/` report helpers | Report generation source area for canonical benchmarks, performance sentinels, large-matrix guardrails, dead-code, CI, and related tooling. |

## Artifact Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_131/PLAN.md` | Day-by-day Sprint 131 execution plan. |
| `docs/planning/EPIC_11/SPRINT_131/WORKING_NOTES.md` | Rolling sprint baseline, decisions, validation expectations, and day notes. |
| `docs/planning/EPIC_11/SPRINT_131/artifacts/` | Day-specific artifacts for corpus inventory, taxonomy, report-index design, coverage architecture, validation, closeout, and handoff. |
| `docs/planning/EPIC_11/SPRINT_131/artifacts/day1-assurance-intake.md` | Day 1 source-area intake, owner map, duplicate fences, and completion status. |

## Project-Plan Owner Map

| Item | Sprint 131 owner days | Likely touched files | Required validation |
| --- | --- | --- | --- |
| 1. Corpus Inventory | Days 1-3 | Sprint 131 artifacts, `tests/data/`, `tests/test_*.c`, helper headers, `tests/*_external_dense_reference.py` | Documentation hygiene for inventory-only work; focused helper/test validation if fixture or helper behavior changes. |
| 2. Corpus Taxonomy | Days 4-5 | Sprint 131 artifacts, possible maintainer guide taxonomy references | Documentation hygiene; claim-boundary scan if maintainer wording changes. |
| 3. Report Index Design | Days 6-7 | Sprint 131 artifacts, `benchmarks/README.md`, `docs/maintainer_guide.md`, possible report scripts | Documentation hygiene; focused script dry run if generation behavior changes. |
| 4. Coverage Architecture | Days 8-9 and 12 | Sprint 131 artifacts, `Makefile`, `scripts/deadcode_report.py`, coverage/dead-code/guardrail docs | Documentation hygiene for architecture-only work; coverage/dead-code/guardrail command validation if targets or scripts change. |
| 5. Generated Index Batch | Days 10-11 | Report-index script or explicit deferral artifact, generated index artifact, Sprint 131 artifacts | Script syntax check, focused dry run, generated artifact freshness check, and full quality gate if `.c` or `.h` files change. |
| 6. Validation | Days 11-13 | Sprint 131 artifacts, generated report/index outputs, docs, affected scripts/tests | Command matrix based on touched surfaces; docs hygiene for documentation-only validation package. |
| 7. Closeout | Day 14 | Sprint 131 closeout artifact, working notes, retrospective candidates, maintainer docs only if claim wording changes | Evidence-to-claim traceability, residual queue completeness, docs hygiene, and no-claim scan. |

## Source-Area Intake List

| Source area | Candidate evidence | Current default interpretation | Later owner |
| --- | --- | --- | --- |
| `tests/data/*.mtx` | Checked-in Matrix Market fixtures such as bad header, identity, diagonal, symmetric, unsymmetric, pattern, and tridiagonal inputs. | Fixture data only until owner, metadata, solver family, oracle, support tier, and expected behavior are recorded. | Days 2, 4, and 5 |
| `tests/data/suitesparse/*.mtx` | Checked-in SuiteSparse-derived matrices such as `nos4`, `west0067`, `bcsstk04`, `bcsstk14`, and related corpus files. | Checked-in corpus smoke or bounded fixture evidence, not broad SuiteSparse parity. | Days 2-5 |
| Generated matrix families in `tests/test_*.c` | Analytic, structural, sparse, graph, solver, SVD, QR, eigenvalue, integration, and stress inputs. | Local regression coverage unless the taxonomy records independent corpus status. | Days 2, 4, and 5 |
| `tests/test_known_matrices.c` | Known matrix behavior coverage. | Named-matrix regression surface; support level depends on documented owner and metadata. | Days 2 and 5 |
| `tests/*_external_dense_reference.py` | Cholesky, LDLT, LU, QR, and SVD external dense-reference helpers. | Bounded helper protocol and fixture-specific oracle surface. | Day 3 |
| Expected failures, skips, optional gates | Unsupported behavior and optional dependency paths. | Failure/skip semantics, not positive evidence. | Day 3 |
| `benchmarks/` | Benchmark binaries, CSV fields, local timing/report outputs. | Measurement and report surface, not correctness oracle or portable performance proof. | Days 6-7 and 10-11 |
| `scripts/bench_canonical_report.sh` | Canonical benchmark report bundle with manifest and `index.tsv`. | Threshold-free artifact index for maintained benchmark surface. | Days 6-7 and 10-11 |
| `scripts/performance_sentinels.sh` | Bounded local sentinel reports. | Local sentinel context; only existing wall-check lane is a hard timing gate. | Days 6-7 and 10-11 |
| `scripts/large_matrix_guardrails.sh` | Reviewed structural guardrails and supplemental large-matrix reports. | Structural guardrail and report context, not broad large-matrix scalability proof. | Days 6-7 and 10-11 |
| `Makefile` coverage targets | `coverage`, `coverage-lcov`, and `coverage-gcovr` reports. | Tree-mutating supplemental coverage signal with aggregate threshold behavior. | Days 8 and 11-13 |
| `scripts/deadcode_workflow.sh` and `scripts/deadcode_report.py` | Raw static-analysis inputs, classified `report.md`, `report.tsv`, and coverage-gap notes. | Report-completeness and residual queue surface, not automatic removal proof. | Days 9 and 11-13 |
| `docs/maintainer_guide.md` | Evidence interpretation, ownership, non-claims, and stable repo norms. | Authoritative maintainer policy surface. | Days 6-14 |
| `benchmarks/README.md` | Benchmark-local usage and schema details. | Local benchmark truth surface, not public performance guarantee. | Days 6-14 |

## Duplicate Fence

Later Sprint 131 work may promote or index a source only when all of the
following are explicit before implementation or wording changes:

1. The source has a stable key, path, owner, and artifact class.
2. The source is classified as fixture, generated family, external-reference
   helper, expected failure, skip, benchmark report, coverage report,
   dead-code report, guardrail report, or documentation policy.
3. The reviewed versus supplemental status is recorded.
4. The source declares matrix metadata, solver family, support tier, optional
   availability, oracle/provenance, tolerance or threshold policy, runtime
   expectation, and failure interpretation when applicable.
5. The validation command and freshness rule are recorded.
6. The artifact states the exact claim it supports and the broader claims it
   does not support.
7. Existing Sprint 120-130 evidence is linked as prior context instead of
   rebranded as new Sprint 131 proof.

If any condition is missing, the source remains inventory or supplemental
context with blocker, owner, and promotion criteria recorded.

## Non-Claim Boundary

Day 1 does not claim:

- full numerical corpus coverage;
- broad SuiteSparse, Matrix Market, LAPACK, NumPy, SciPy, PETSc, Trilinos,
  Eigen, ARPACK, or vendor-backend parity;
- solver-selection wording readiness beyond current bounded maintainer
  evidence;
- generated-family independence from product implementation;
- benchmark timing portability or performance regression proof;
- coverage percentage as behavior completeness;
- dead-code findings as removal-ready proof;
- large-matrix guardrails as scalability or memory-bound guarantees;
- optional corpus availability across platforms;
- new public API, package, ABI, CMake, CI, install-header, or documentation
  guarantees.

## Validation Boundary

| Change class | Day 1 rule |
| --- | --- |
| Documentation-only Sprint 131 artifacts | `git diff --check` and trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_131`. |
| Corpus inventory only | Documentation hygiene; no test run required unless files or behavior change. |
| Report-index generation changes | Script syntax check, focused dry run, generated artifact inspection, and documentation hygiene. |
| Coverage changes | Selected coverage command, tree-mutation note, artifact inspection, and `make clean` before normal validation. |
| Dead-code report changes | `make deadcode-report` and `make deadcode-check`, run serially. |
| Benchmark or guardrail report changes | Focused report target and artifact inspection; no timing claim unless a later gate defines one. |
| `.c` or `.h` edits | `make format && make lint && make test`. |

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 131 project-plan item has a day-level owner. | Complete | Owner map ties Items 1-7 to Days 1-14 with likely touched files and validation. |
| Existing Sprint 120-130 evidence boundaries are preserved. | Complete | Duplicate fence and non-claim boundary preserve oracle, residual, helper, optional-corpus, report, and solver-selection limits. |
| Corpus, report, coverage, and validation surfaces are visible before design or implementation begins. | Complete | Source-area intake list covers `tests/data/`, generated tests, external helpers, benchmarks, report scripts, coverage targets, dead-code workflow, large-matrix guardrails, and maintainer docs. |

## Day 2 Handoff

Day 2 should convert the source-area intake into a checked-in fixture and
generated-family inventory. It should classify every visible Matrix Market
fixture and generated numerical family by owner, solver family, metadata
completeness, support tier, and missing promotion criteria without assigning
new reviewed evidence claims.

