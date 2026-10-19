# Sprint 215 Day 8: Schema Acceptance And Pilot Feasibility

## Decisions

**Accept the Day 6-7 separate v1 evidence design for the selected pilot.**
`day7-external-evidence-blueprint.example.json` remains a planning record,
not a generated result. A generated attempt belongs under
`build/external-baseline/<selection_id>/<run_id>/`; no selected manifest,
normalizer, hosted workflow or canonical benchmark row changes. The schema
and semantic validator are still **designs**, not implemented authorities.
Before any observed record is accepted, Day 9 must implement the bounded
validator and its negative fixtures, or Day 10 must report `blocked`.

**GO for one opt-in local correctness-only pilot preparation and gated
attempt.** The selected cell is SuiteSparse `v7.12.2` (commit
`42151688813c45846a597edcb601435a0e38f3dd`), CHOLMOD 5.3.4, against
the project no-reorder linked-list Cholesky solver on
`tests/data/symmetric_4.mtx`. The operation is real-float64 SPD factorization
plus one `A*x=b` solve with `x_ref=[1,1,1,1]` and `b=[13,14,17,16]` on
**local macOS x86-64 only**. This is authorization to build the opt-in
harness and check execution gates, not a claim that the solver has run.
No timing, speed ratio, winner, hosted threshold, cross-platform parity or
third-party endorsement is authorized. Publication remains **pending** until
fixture attribution and CHOLMOD module/license review are recorded.

## Schema Acceptance Boundary (215.3)

The Day 7 example is the positive blueprint fixture. `record_kind=blueprint`,
`evidence_origin=planning`, `evidence_status=planned`, `run_id=null`,
`cell_id=null`, two `not_run` correctness sides, `timing.state=not_run` and
`artifacts=[]` are required together. The same v1 envelope admits a generated
`not_run` or execution-`blocked` attempt with a run ID, named reason and null
unavailable runtime fields; it admits `correctness_observed` only with two
independent passes, complete cell identity and hashed raw outputs. A
publication-only block may retain locally observed data but cannot be
published or become `measured`. The Day 7 status matrix, units, unknown-field
rejection and mutation cases are accepted as the pilot's validation contract.

| Owner | Accepted responsibility | Gate before an observed record |
| --- | --- | --- |
| Corpus owner | Freeze fixture bytes, expanded matrix/RHS checks and source attribution. | Verify SHA-256 and 4 x 4/seven stored/ten expanded coefficients; publication attribution remains separate. |
| Solver owner | Build the opt-in two-sided harness and retain raw phase statuses, vectors and Day 4 metrics. | Verify effective project backend and CHOLMOD ordering/mode, ABI and identical inputs before classification. |
| Report owner | Implement the isolated v1 schema/validator and append-only run bundle; preserve `planned`, `blocked`, `fail` and `pass` distinctions. | Reject duplicate/unknown fields, missing hashes, mismatched selection/cell identity, invalid units and unsupported claims. |
| Benchmark owner | Keep the Day 5 timing contract separate. | No timing collection in this selected pilot; no reuse of the selected canonical benchmark row. |
| Publication reviewer | Review fixture authorship/terms and linked CHOLMOD module terms. | Explicit approval artifact before any public external-result claim. |

Freeze `selection_id=s215-cholmod-v7_12_2-symmetric4-spdsolve-v1`, the exact
Day 3 matrix/RHS/solver controls and v1 status vocabulary. For a complete
`cell_id`, use the Day 7 sorted-key, compact ASCII JSON/SHA-256 rule over the
explicit identity fields: matrix SHA; problem class, operation, RHS and
precision; both index ABIs; project commit, worktree/patch identity, binary
SHA, compiler/build flags, API, backend and ordering; baseline source commit,
loaded library SHA, dependency path/hash list, module/build identity, API,
mode and ordering; requested/observed threads; and host OS/build,
architecture and CPU identity. Dependency entries sort by role and absolute
loaded path. No null is allowed in this complete identity. The Day 9
validator must freeze the exact JSON object layout and a canonical byte/hash
test vector **before assigning any production cell ID**. A blueprint or
execution block retains `cell_id=null` with a reason; never hash a partial
tuple. Methodology file hashes and raw artifacts are run provenance, not
substitutes for cell identity. Changing a selected identity field requires
a new selection decision, not a silent edit of the example.

Compatibility acceptance is deliberately narrow: keep
`tests/corpus/manifests/selected_report_targets.tsv`,
`tests/corpus/manifests/report_families.tsv`,
`scripts/normalize_report_index.py`, and
`scripts/run_external_comparison.py` unchanged. The latter's
`external-process-source-controlled-helper` rows do not become CHOLMOD
observations. The canonical `nos4` row remains repeat-one, `baseline=n/a`,
`threshold=n/a`. Positive blueprint and synthetic blocked/observed fixtures,
plus Day 7 negative mutations, are required before the new validator can
accept pilot output. Existing corpus, manifest, normalizer, comparison and
benchmark tests are regression checks, not proof of an external run.

## Feasibility Evidence And Open Gates (215.4)

Day 8 rechecked the local candidate on Darwin 24.6.0 x86-64:

| Gate | Day 8 observation | Decision |
| --- | --- | --- |
| Immutable fixture | `shasum -a 256 tests/data/symmetric_4.mtx` returned `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`. The Day 3/4 expansion and oracle are predeclared, not newly executed here. | Pin exact bytes. |
| External package | `pkg-config --modversion cholmod` returned `5.3.4`; prefix was `/usr/local/Cellar/suite-sparse/7.12.2`; package is a poured Homebrew bottle. The candidate `libcholmod.5.3.4.dylib` hash was `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`, and `file` reported an x86-64 Mach-O dylib. | Available for an opt-in local harness, not yet runtime-verified. |
| Linkage/provenance | `otool -L` listed AMD/COLAMD/CAMD/CCOLAMD, SuiteSparseConfig, Accelerate, libomp and libSystem links. The exact loaded paths/hashes, transitive dependencies, bottle formula revision, build/module options, BLAS thread behavior and effective CHOLMOD controls have not been captured. | Hard stop before `correctness_observed` or any complete `cell_id` if these cannot be verified. |
| Operation and oracle | Day 3 fixes the same SPD solve, natural/no-reorder controls, float64, one RHS and conversion invariants. Day 4 predeclares independent residual/oracle limits. | Comparable by design; the harness must verify actual paths and statuses. |
| Local use and publication | The installed package contains a bundled multi-license `LICENSE.txt` (SHA-256 `c07f7bb69aa870c774bd01f040dabd179b7cc8c60021c2c56db9d63d37deabcb`); Homebrew reports mixed licenses. The fixture first appears in repo commit `aa1c04eab3536031b9a92ee92587260520ce1913`, with no file-level attribution. | Local evaluation may proceed without redistribution; do not publish a result or bundle the third-party binary until module and fixture terms are reviewed. |
| Time and dependency scope | One 4 x 4 solve, correctness only; no download or dependency added to default build/test. | One optional local cell fits the Day 9-10 pilot allocation. |

The Day 8 discovery does not establish the full runtime-loaded library graph,
effective controls, numerical correctness or license approval for a public
comparison. The bottle's mixed license expression is not a module-level
review. If a missing build or dependency identity cannot be recovered, emit
a named `blocked` attempt and use the executable-design fallback rather than
filling a provenance field with `unknown` or claiming a pass.

## Exact Pilot Path And Stop Conditions

The following discovery commands are executable now. They must be repeated
when the harness runs because the installed package can change:

```sh
shasum -a 256 tests/data/symmetric_4.mtx
pkg-config --modversion cholmod
pkg-config --variable=prefix cholmod
pkg-config --cflags --libs cholmod
shasum -a 256 /usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib
otool -L /usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib
python3 -m json.tool docs/planning/EPIC_19/SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json
```

Day 9 owns the proposed opt-in interface below; **these commands are not
executable until the named files exist**:

```sh
RUN_ID=local-correctness-001
python3 scripts/run_cholmod_pilot.py --fixture tests/data/symmetric_4.mtx --output "build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/$RUN_ID" --correctness-only --execute
python3 scripts/validate_external_baseline_evidence.py --schema tests/corpus/schemas/external_baseline_evidence_v1.json --input "build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/$RUN_ID/evidence.json" --bundle-root "build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/$RUN_ID"
```

Expected files are immutable `evidence.json`, input/binary/dependency
fingerprints, compiler/build/environment manifest, raw per-phase status log,
both full-precision solution/metric files and SHA-256 references for each.
Expected outcomes are either a validated one-cell
`correctness_observed` record (both Day 4 passes), an honest
`correctness_failed`/`non_comparable` record, or a named `blocked` record.
Timing stays `not_run` in every selected-pilot outcome. A missing optional
package must not make maintained CI fail or produce a synthetic pass.

Stop before solving if fixture bytes/shape/RHS differ, the package version
or binary hash changes, actual linked library/dependency identity cannot be
recorded, index ABI or solver controls differ, inputs cannot be shown equal,
or the validator cannot reject the Day 7 negative fixtures. Stop numerical
acceptance if either side returns a failing status, nonfinite output or fails
Day 4 thresholds; retain raw diagnostics. Stop publication until attribution
and module-license review approves it. No silent substitute baseline, matrix,
host or solver family is allowed.

## Bounded Day 9-10 Worklist

Allowed pilot changes: `scripts/run_cholmod_pilot.py`, an opt-in C helper
`scripts/cholmod_pilot.c` if the C API requires one,
`scripts/validate_external_baseline_evidence.py`,
`tests/corpus/schemas/external_baseline_evidence_v1.json`, focused
`tests/test_run_cholmod_pilot.py` and
`tests/test_validate_external_baseline_evidence.py`, and generated files only
under `build/external-baseline/`. The Day 7 planning JSON is a fixture,
not an observed report; Sprint 215 artifacts/notes may record design and
outcomes. Do not edit unrelated matrix data, selected manifests, report
normalization, canonical benchmark output, workflows, or public speed claims.

Focused checks: positive blueprint and blocked-record validation; duplicate
key, missing hash, altered matrix, mismatched cell ID, unsupported enum,
nonfinite metric, missing raw file, and `measured`-without-two-passes
mutations; opt-in harness missing-package and solver-status failures;
`python3 scripts/validate_corpus_schema.py`,
`python3 tests/test_normalize_report_index.py`,
`python3 tests/test_selected_report_targets_manifest.py`,
`python3 tests/test_selected_comparison_workflow.py`, and
`python3 tests/test_bench_canonical_freshness.py`. Run `make docs-check`
for changed docs. If a `.c` or `.h` file changes, run
`make format && make lint && make test` before accepting the work. No
external dependency enters the maintained default path.

## Day 8 Validation And Handoff

The blueprint parses and the selected fixture/library hashes match the Day 3
discovery. No external solve, timing run, schema implementation or generated
pilot record was produced on Day 8. Day 9 must first implement the opt-in
harness/validator and its negative tests. Day 10 may execute only after the
stop conditions above pass; otherwise it must preserve a reproducible
blocker and report `blocked`, not a skipped run as evidence.
