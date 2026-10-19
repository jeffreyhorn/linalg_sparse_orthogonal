# Sprint 215 Day 13: Integrated Validation

## Scope And Verdict

Run on 2026-10-19, local Darwin 24.6.0 x86-64 with Apple clang 17.0.0, on
branch `sprint-215` at Git HEAD `07e482875d35c51d4f50de922508fabd5963be15`
plus uncommitted Sprint 215 work. **All executable quality and compatibility
checks below passed.** This is local validation only; no hosted workflow ran.
The separate external v1 schema and semantic validator are still absent, so
the selected CHOLMOD pilot remains blocked before solver execution. No
`correctness_observed`, `measured`, portable-speed, parity or state-of-the-art
evidence was produced.

The [Day 12 validation matrix](day12-claim-documentation-and-validation-plan.md)
defines expected outcomes. The [Day 10 attempt](day10-selected-pilot-outcome.md)
is the retained raw preflight bundle inspected below. Generated comparison
and benchmark output is ignored local `build/` material, not checked-in or
hosted evidence.

## Command Results

| Command | Exit | Observed result and scope |
| --- | ---: | --- |
| `make format && make lint && make test` | 0 | Full required C gate passed after the Sprint's opt-in C helper addition. `clang-tidy` completed with configured suppressed diagnostics, `cppcheck` reached 109/109 files, and `make test` ended `All tests passed.` Some test-framework/ND cases reported intentional skips; they were not treated as numerical pilot results. |
| `clang-format --dry-run --Werror scripts/cholmod_pilot.c` | 0 | The opt-in helper is outside the regular Makefile source list; its own formatting check passed. |
| `cc -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror -O2 -Iinclude -Ibuild/include $(pkg-config --cflags cholmod) scripts/cholmod_pilot.c -Lbuild -lsparse_lu_ortho $(pkg-config --libs cholmod) -lm -o build/cholmod_pilot_day13` | 0 | Strict local compile/link passed against the discovered CHOLMOD package. The resulting binary was **not executed** or accepted as runtime-provenance evidence. |
| `python3 scripts/validate_corpus_schema.py` | 0 | Source-controlled corpus schema valid; this is not the proposed external v1 schema. |
| `python3 tests/test_selected_report_targets_manifest.py` | 0 | Existing selected target metadata remained valid. |
| `python3 -m json.tool docs/planning/EPIC_19/SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json` | 0 | Planning JSON parses; no v1 semantic validator exists to accept it as an observed run. |
| `test -f tests/corpus/schemas/external_baseline_evidence_v1.json` | 1 (expected blocker) | Proposed external v1 schema is absent. |
| `test -f scripts/validate_external_baseline_evidence.py` | 1 (expected blocker) | Proposed external v1 semantic validator is absent. These two nonzero probes are not passing checks or solver failures. |
| `python3 tests/test_normalize_report_index.py` | 0 | Existing normalized family/status behavior passed its regression suite. No external-baseline family was added. |
| `python3 tests/test_run_external_comparison.py` | 0 | Existing source-controlled-helper comparison regression passed. |
| `make report-index-comparison-freshness` | 0 | Regenerated selected local comparison output and reported freshness for 46 rows. The generated Cholesky `study.tsv` names `external-process-source-controlled-helper`; it is not CHOLMOD evidence. |
| `make bench-canonical-report-freshness-tests` | 0 | Canonical selected-row positive and negative regressions passed, including threshold-free and hosted-metadata boundaries. |
| `make bench-canonical-report-freshness` | 0 | Regenerated the canonical bundle and passed the **local** `bench_refactor_csc` freshness check. `index.tsv` still records `status=measurement`, `support_tier=local_only`, `claim_boundary=local_threshold_free`, `warmup=none_configured`, `variance=not_computed_single_sample`, `baseline=n/a`, `threshold=n/a`. The selected workload remains `nos4.mtx --repeat 1`, not `symmetric_4.mtx`. |
| `python3 tests/test_external_baseline_claim_docs.py` | 0 | Four test methods passed, including broad-claim mutations on all four guarded docs. |
| `python3 tests/test_selected_performance_docs.py` | 0 | Existing selected-performance claim guard passed. |
| `make docs-check` | 0 | Doxygen and API coverage passed for 18 checked-in public headers, 18 generated reference pages and 18 source pages. |
| `python3 tests/test_run_cholmod_pilot.py` | 0 | Nine synthetic parser/status/numerical/fail-closed tests passed; no external solver ran. |

`make format` did not create unrelated tracked C/header changes. The full C
gate validates maintained Makefile sources; the separate format and strict
compile/link commands above cover the opt-in helper specifically. Python
synthetic checks cover its runner's parser and blocked paths, not actual
loaded-library, module, ABI, thread or numerical runtime provenance.

## Retained Attempt Inspection

The ignored Day 10 bundle is
`build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-preflight-20261019/`.
The command below exited 0 and returned `true`:

```sh
jq -e '.provisional_status == "blocked" and .status_reason == "schema_validation_unavailable" and .schema_validation.status == "not_run" and .timing == "not_run" and .sides == {} and (.commands | has("project") | not) and (.commands | has("baseline") | not)' build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-preflight-20261019/attempt.json
```

`shasum -a 256` returned the previously recorded attempt digest
`ef5ba0af8d878141a993cc4348efa9c5f2ac9fd5b99f1bef6da3d45b3dc0568a`.
Every SHA-256-referenced command stdout/stderr file in that attempt was
rechecked with `jq -r '.commands[].files[] | "\(.sha256)  build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-preflight-20261019/\(.path)"' ... | shasum -a 256 -c`:
16/16 files reported `OK`, exit 0. The logs are discovery/preflight logs
only. There is no `project` or `baseline` solver command, no side result, no
timing sample and no `evidence.json` in the bundle. A file-existence probe
for that `evidence.json` returned 1, as expected. This proves the recorded
blocker and raw-log integrity, not numerical correctness.

## Remaining Evidence Gate

Item 215.3 remains a **design**, not an implemented external v1 validator.
Implement the isolated schema, semantic validator, negative fixtures and
complete cell-ID test vector before another solver attempt. Then recheck
actual loaded CHOLMOD and transitive dependencies, build/module options, ABI,
effective ordering/backend/threads, exact matrix/RHS conversion, retained
raw commands and both frozen Day 4 numerical gates. An optional later timing
run additionally needs the Day 5 paired sampling, clock and host-provenance
contract. Fixture attribution and module-license review remain publication
gates. Do not promote the blocked `attempt.json` to `evidence.json` or map
provisional states into existing selected report rows.

Day 14 may close the **blueprint and blocker record** with this limitation;
it cannot close the external numerical, hosted or performance evidence gate
as passed. No external pilot was rerun on Day 13 because the required v1
precondition still fails.
