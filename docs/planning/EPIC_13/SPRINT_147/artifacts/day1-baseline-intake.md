# Day 1 Baseline Intake

## Scope

Day 1 establishes the Sprint 147 planning baseline. It does not select final
Epic 13 gaps yet; it defines the evidence surfaces, source inputs, artifact
structure, day-level owners, capture format, and stop conditions that later
Sprint 147 days must use.

## Source Inputs

| Input | Purpose |
| --- | --- |
| `docs/planning/EPIC_13/PROJECT_PLAN.md` | Authoritative Sprint 147 item list, duration, goal, prerequisites, and deliverables. |
| `docs/planning/EPIC_13/reviews/review-codex-2026-08-09.md` | Full-code-review findings and initial gap assessment for Epic 13. |
| `docs/planning/EPIC_13/reviews/todo-codex-2026-08-09.md` | Step-by-step remediation plan that turns the review into implementation candidates. |
| `docs/planning/EPIC_12/EPIC_12_RETROSPECTIVE.md` | Epic 12 earned claims, non-claims, validation snapshot, and residual/future-epic candidates. |
| `docs/planning/EPIC_12/SPRINT_146/artifacts/day11-published-residual-queue.md` | Stable residual IDs R1-R14, owners, blockers, and promotion gates. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `benchmarks/README.md` | Public and maintainer support-tier wording to freeze before Epic 13 implementation. |
| `Makefile`, `CMakeLists.txt`, `.github/workflows/*.yml` | Build, package, CI, and platform support evidence. |
| `tests/corpus/**`, `scripts/validate_corpus_schema.py`, `scripts/run_corpus_oracle.py`, `scripts/normalize_report_index.py` | Corpus, generated evidence, and report/freshness ownership. |

## Baseline Categories

| Category | Day Owner | What To Capture | Evidence Boundary |
| --- | --- | --- | --- |
| Source and test size | Day 2 | File counts, line counts, largest tests, largest implementation files, source-list duplication. | Size and maintainability evidence only; not quality proof by itself. |
| Build and package | Day 2 | Make/CMake source ownership, package scripts, install proof, static-first deferral guard. | Static-first package proof only unless later ABI gates pass. |
| CI and platform | Days 2 and 7 | Linux, macOS, Windows reviewed/supplemental/staged lanes, Windows CTest count, hosted evidence requirements. | Hosted platform claims require current workflow evidence. |
| Corpus | Day 3 | Fixture rows, generator rows, expected rows, optional-data policy, QR/partial-SVD proof owners. | Fixture-local evidence only until broader family gates pass. |
| Report | Days 3 and 9 | Report-family rows, freshness policy, generated artifact paths, missing/stale/advisory semantics. | Source-controlled rows are not generated pass proof. |
| Residuals | Days 4-5 | Epic 12 residuals R1-R14, owners, blockers, dependencies, duplicate fences, selected gaps. | Residuals remain non-claims until promotion gates pass. |
| Claims | Days 6 and 13 | Candidate earned claims, rejected broad claims, wording boundaries, rollback rules. | State-of-the-art requires direct comparative evidence. |
| Quality | Day 12 | Required checks by touched surface and stop conditions. | C/header changes require full C quality gate. |

## Day-Level Owner Map

| Day | Owner Area | Primary Output |
| ---: | --- | --- |
| 1 | Baseline intake | Working notes, artifact structure, baseline categories, item owner map, stop conditions. |
| 2 | Technical baseline | Source/test/build/package/CI baseline and largest-file risks. |
| 3 | Evidence baseline | Corpus/report rows, fixture-local proof owners, generated-local boundaries. |
| 4 | Residual intake | Epic 12 residual grouping, dependency map, owner/prerequisite classification. |
| 5 | Gap selection | Selected Epic 13 gaps, explicit non-goals, duplicate fences, sprint-to-gap map. |
| 6 | Claim target register | Candidate claims, required evidence, rejected claims, promotion/rollback rules. |
| 7 | Windows evidence gate | Windows staged-test and install-validation parity gate. |
| 8 | Corpus-family gates | QR and partial-SVD corpus-family proof gates. |
| 9 | Generated freshness gate | Required-generated and advisory report freshness policy. |
| 10 | ABI/package gate | Shared-library implementation or stronger static-first deferral gate. |
| 11 | External comparison gate | Narrow comparison target, dependency pinning, row schema, and claim rules. |
| 12 | Quality surface map | Validation matrix, full C gate triggers, supplemental checks, stop conditions. |
| 13 | Public claim freeze | Public/support claim scan, fix list or no-fix rationale, wording baseline. |
| 14 | Closeout handoff | Final artifact index, Sprint 148 Windows prerequisites, validation summary. |

## Evidence Capture Format

Later Sprint 147 artifacts should record evidence in this format:

| Field | Requirement |
| --- | --- |
| Evidence source | File, command, CI job, report row, generated artifact, or retrospective source. |
| Claim supported | Narrow claim, support-tier statement, or non-claim boundary. |
| Status | `supported`, `selected`, `candidate`, `deferred`, `rejected`, `residual`, or `blocked`. |
| Validation required | Exact local command, hosted CI requirement, generated freshness check, or manual audit. |
| Boundary | What the evidence does not prove. |
| Owner | Maintainer surface responsible for future changes. |
| Handoff | Next day or sprint that consumes the evidence. |

## Stop Conditions

| Stop Condition | Required Action |
| --- | --- |
| Candidate claim lacks concrete evidence. | Keep it as residual/non-claim and do not add public wording. |
| State-of-the-art or external parity language appears without direct comparative evidence. | Reject the claim and record it in the non-claim register. |
| Platform support wording lacks hosted platform proof. | Keep the lane staged, supplemental, local-only, or deferred. |
| Package/ABI wording lacks downstream consumer proof. | Preserve static-first or deferred wording. |
| Generated local rows are treated as source-controlled pass evidence. | Correct the artifact wording and re-run report/freshness checks if needed. |
| C/header changes occur. | Require `make format && make lint && make test` before closeout. |
| Documentation claim audit finds unsupported wording. | Fix wording or stop for clarification. |
| Review feedback or validation failure is unclear. | Stop and ask before committing a claim or fix. |

## Sprint 148 Handoff Seed

Sprint 148 should begin from the Day 7 Windows evidence gate, but Day 1
already identifies the critical prerequisite surfaces:

- `.github/workflows/windows-ci.yml`
- `CMakeLists.txt`
- `tests/test_threads.c`
- `tests/test_sprint4_integration.c`
- `tests/test_fuzz.c`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- `tests/corpus/manifests/report_families.tsv`

Until Sprint 148 promotes or replaces the staged coverage with hosted Windows
proof, Windows pthread/POSIX staged test closure remains a non-claim.
