# Day 14: Integrated Validation And Sprint 222 Handoff

## Validation Ledger

Run from the repository root on branch `sprint-221` with Python 3.14.5,
Doxygen 1.16.1 and PyYAML 6.0.3 in the ignored
`build/api-docs-venv` environment. Schema `$id` is
`external_baseline_evidence_v1`. This is a local format/consistency and
preflight validation, not an external solver run.

| Command | Outcome |
| --- | --- |
| `python3 scripts/validate_external_baseline_evidence.py --schema tests/corpus/schemas/external_baseline_evidence_v1.json --self-test` | PASS, including negative self-test gates. |
| `python3 scripts/validate_external_baseline_evidence.py --schema tests/corpus/schemas/external_baseline_evidence_v1.json --input tests/corpus/fixtures/external_baseline_planned_v1.json` | PASS as `evidence_status=planned`, not observed. |
| `python3 tests/test_validate_external_baseline_evidence.py` | PASS, 45 tests; positive/negative statuses, byte-backed bundles and mutation diagnostics. |
| `python3 tests/test_run_cholmod_pilot.py` | PASS, 16 tests; no real CHOLMOD execution. |
| `python3 scripts/validate_corpus_schema.py` | PASS. |
| `python3 tests/test_selected_comparison_workflow.py` | PASS. |
| `python3 tests/test_selected_report_targets_manifest.py` | PASS. |
| `python3 tests/test_normalize_report_index.py` | PASS. |
| `python3 tests/test_selected_performance_docs.py` | PASS. |
| `make docs-check` | PASS; 18 public headers, 18 reference pages and 18 source pages. |
| `make api-docs-routing API_DOCS_PYTHON=build/api-docs-venv/bin/python` | PASS; local-only guard checked 111 job/step records, routing checked seven documents, no generated-API publication links. |
| `python3 -m py_compile scripts/validate_external_baseline_evidence.py scripts/run_cholmod_pilot.py tests/test_validate_external_baseline_evidence.py tests/test_run_cholmod_pilot.py` | PASS. |
| `git diff --check` | PASS. |

The optional Python CI step added this sprint runs the same validator
`--self-test`, validator suite and pilot suite shown above. No `.c` or `.h`
file changed during Sprint 221, so the conditional `make format && make lint
&& make test` gate was not applicable. No required Day 14 check failed or
was skipped. The Doxygen and API-doc routing commands generated only ignored
local output.

## Source And Publication Audit

`git status --short --untracked-files=all` shows only the sprint's validator,
schema, synthetic fixtures, pilot/CI integration, tests, README and planning
artifacts. No selected corpus manifest, selected-report target, normalized
report output or canonical benchmark row changed. The selected comparison
workflow and report-index regression tests passed. `git check-ignore -v`
confirms `build/external-baseline/.../attempt.json` is covered by the
`build/` ignore rule and generated `docs/api/html/index.html` by `docs/api/`.
No `build/external-baseline/` directory exists from these checks. No new
publication route, external-result row or performance claim was added.

## Deliverable Reconciliation

| Item | Closed deliverable | Evidence |
| --- | --- | --- |
| 221.1 Schema Freeze | Frozen v1 selection, status, units, provenance and canonical identity contract. | Schema, planning fixture and Days 1-3 artifacts. |
| 221.2 Validator Implementation | Strict JSON, exact shape, semantic, canonical-ID and byte-backed artifact checks with stable code/pointer diagnostics. | Validator, self-test and Days 4-6 artifacts. |
| 221.3 Fixture Matrix | Positive planning/non-execution and synthetic observed/measured/failed cases with negative neighbors. | Test suite and Days 7-9 artifacts. |
| 221.4 Pilot Preflight | Real negative self-test before dependency discovery, build or either solver; immutable attempt-path refusal and CI registration. | Pilot tests and Days 10-11 artifacts. |
| 221.5 Mutation Proof | 16 Day 12 and five Day 13 syntax-valid CLI mutations, plus Day 11 parser seeds; exact diagnostics and restored positives. | Days 11-13 verdict ledgers and `CliMutationTests`. |
| 221.6 Docs And Validation | Maintainer guide, README link and passing Day 14 validation ledger. | Day 13 guide and this artifact. |

The package is ready as a **validator prerequisite**, not as accepted
CHOLMOD evidence. A successful `--input` verdict proves the supplied v1
record is internally consistent with the referenced bytes. It cannot prove
that a synthetic record came from an independent solver or that a review
artifact contains an approved rights decision.

## Sprint 222 Handoff And Open Gates

1. Verify the selected matrix/RHS/conversion bytes and attribution scope;
   independently review CHOLMOD module licenses and retention rights before
   any publication decision.
2. Establish an actual SuiteSparse v7.12.2 / CHOLMOD 5.3.4 runtime and
   verify the loaded library, transitive dependency hashes, module/build
   options, int32 ABI, natural/simplicial/no-postorder controls, project
   binary/build flags, host, clock-independent correctness environment and
   requested/observed threads. Candidate library presence alone is not
   loaded-library provenance.
3. Execute the optional pilot only in a new ignored attempt directory.
   Preserve raw stdout/stderr bytes, exit and factor statuses, solution
   vectors and failed attempts verbatim. The pilot currently emits an
   `unvalidated_attempt` envelope; even
   `two_sided_correctness_candidate` is not a v1 accepted result.
4. Materialize a generated v1 record and bundle from genuine two-sided
   execution, verify independent Day 4 residual/backward-error/oracle gates,
   canonical cell ID and all artifact hashes with `--input --bundle-root`,
   then repeat from frozen inputs. Do not reuse the synthetic fixture IDs or
   promote a one-sided/blocked attempt.
5. Keep publication review separate. A rights denial keeps the bundle local;
   an approval needs an exact-bundle review artifact and a narrowly worded
   local correctness claim. The correctness-only selection does not permit
   timing, speedup, broad parity, portable performance or state-of-the-art
   claims.

The [maintainer guide](day13-validator-maintainer-guide.md) contains the v1
CLI and diagnostic contract. Sprint 222's project-plan items 222.1-222.6
own these remaining execution, provenance, repeat and publication gates.
