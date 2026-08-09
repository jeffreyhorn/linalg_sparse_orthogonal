# Sprint 145 Day 12 Validation Gate

## Purpose

Day 12 runs the formal validation gate for the cumulative Sprint 145 adoption
surface. The gate covers documentation routing, maintained examples,
static-first install/downstream consumption, report-index normalization, and
the public-header quality gate required by the current `.h` diff.

## Changed Surface Matrix

| Surface | Representative paths | Required Day 12 gate |
| --- | --- | --- |
| Public adoption docs | `README.md`, `INSTALL.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `examples/README.md` | reference/whitespace checks plus focused example and install/downstream proof |
| Public headers | `include/sparse_matrix.h`, `include/sparse_iterative.h`, `include/sparse_qr.h`, `include/sparse_svd.h` | declaration-preservation scan plus `make format && make lint && make test` |
| Report metadata and schema | `tests/corpus/manifests/report_families.tsv`, `tests/corpus/schemas/report_index_fields.md`, `scripts/validate_corpus_schema.py`, `tests/test_normalize_report_index.py` | corpus schema, normalization, and freshness checks |
| Planning evidence | `docs/planning/EPIC_12/SPRINT_145/WORKING_NOTES.md`, Sprint 145 artifacts | whitespace and `git diff --check` |

## Command Log

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Passed | `validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok` |
| `python3 tests/test_normalize_report_index.py` | Passed | `test-normalize-report-index: ok` |
| `python3 scripts/normalize_report_index.py --family documentation --family package --family ci --family runtime_backend --check` | Passed | `normalize-report-index: 9 rows ok` |
| `python3 scripts/normalize_report_index.py --family documentation --family package --family ci --family runtime_backend --check-freshness` | Passed | Freshness advisory rows stayed source-controlled and the selected families ended with `normalize-report-index: freshness ok (9 rows)`. |
| `make examples-build` | Passed | Built 14 maintained example binaries without execution failures. |
| `bash tests/test_install.sh` | Passed | Install validation summary: Passed 23, Failed 0, `ALL INSTALL TESTS PASSED`. |
| `bash tests/test_cmake_install.sh` | Passed | CMake install validation summary: Passed 26, Failed 0, Skipped 0, `ALL CMAKE INSTALL TESTS PASSED`. |
| `make format && make lint && make test` | Passed | Formatting, lint, and the full local test suite completed with `All tests passed.` |

## Skipped Or Deferred Checks

| Check | Rationale |
| --- | --- |
| Hosted Linux, macOS, and Windows CI reruns | Day 12 ran local gates only. Hosted platform lanes remain the PR/CI proof owner because the local environment is macOS and cannot reproduce all hosted runner images or Windows MSVC behavior. |
| Regenerating benchmark, coverage, dead-code, or sentinel measurement reports | Sprint 145 changed adoption docs, public-header comments, and source-controlled report metadata. It did not add new performance measurements or coverage/dead-code report rows. Report normalization and freshness checks validate the touched source-controlled report-family metadata. |
| Executing every maintained example binary individually | `make examples-build` compiled all 14 maintained examples, while install/downstream tests executed the maintained installed-consumer paths. No example source changed on Day 12. |

## Environment Constraints

- Local validation ran from `/Users/jeff/experiments/linalg_sparse_orthogonal`
  on a macOS developer environment.
- Install and CMake install checks used temporary prefixes managed by the
  validation scripts.
- OpenMP remains environment-dependent; the local test output reported the
  serial OpenMP path when OpenMP was unavailable.
- Generated build outputs remain ignored local artifacts and are not sprint
  deliverables.
- Hosted platform parity, Windows Makefile parity, Windows `pkg-config`
  parity, shared-library ABI support, package-manager distribution, and
  portable performance parity remain non-claims.

## Failure Fix List

No Day 12 validation failures required code or documentation fixes. The Day 11
runtime/backend report-family coherence repair was validated by schema,
normalization, and freshness checks.

## Completion Criteria

- Required report checks passed.
- Required install/downstream checks passed.
- Required maintained example build check passed.
- Required C/header quality gate passed because public headers remain modified
  in the cumulative branch diff.
- Skipped checks have explicit rationale and ownership.
- No unresolved Day 12 validation failure remains.

## Day 13 Handoff

Day 13 can use this artifact as the latest validation proof for the Sprint 145
claim map and residual-debt review. The remaining closeout work should focus on
checking that public docs, headers, examples, report rows, and planning
artifacts make the same bounded adoption claims without adding unsupported
state-of-the-art, package, ABI, platform, or performance language.
