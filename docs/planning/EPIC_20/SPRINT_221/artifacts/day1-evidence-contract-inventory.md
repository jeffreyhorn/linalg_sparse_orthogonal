# Sprint 221 Day 1: Evidence Contract Inventory

## Source And Blocker

Inventory taken against source revision
`50dab57aa8cc6cbdeacd70357d0453bb5170993e` on the `sprint-221`
branch. The accepted contract is Sprint 215 Day 6-8's separate v1 external
evidence design, Day 3 input identity, Day 4 correctness gates, and Day 5
measurement policy. The Day 7 blueprint is parseable **planning**, not an
observation. Sprint 215 Day 10's local attempt stopped before either solver:
`schema_validation_unavailable`, `sides={}`, `timing=not_run`, no
`evidence.json`. The historical missing-file gate is now replaced by a
fail-closed Day 3 validator interface; that does not turn the attempt into
external evidence.

The selected fixture hash was rechecked locally with
`shasum -a 256 tests/data/symmetric_4.mtx`:
`ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`.
The selection fixes a real symmetric 4 x 4 SPD matrix with seven stored and
ten expanded entries, one float64 `A*x=b` solve, RHS `[13,14,17,16]` and
oracle `[1,1,1,1]`. The project path is int32 no-reorder linked-list
Cholesky; the baseline is SuiteSparse v7.12.2 commit
`42151688813c45846a597edcb601435a0e38f3dd`, CHOLMOD 5.3.4,
natural/simplicial `CHOLMOD_A`. The previously discovered Homebrew dylib
hash is a candidate, **not** proof of the runtime-loaded library.

## Field And Owner Map

| Contract group | Source | Validation owner and required check | Fixture owner |
| --- | --- | --- | --- |
| Envelope, record/run/selection IDs, status and origin | Sprint 215 Day 6-8; blueprint | Report: exact v1 schema, unique IDs, generated-vs-planning relationship and append-only runs | Report |
| Matrix bytes, shape, field, symmetry and attribution | Day 3/4, Day 8; `symmetric_4.mtx` | Corpus: actual SHA-256, expansion, coefficient/RHS identity; publication attribution separate | Corpus |
| Problem, RHS/oracle, precision and both index ABIs | Day 3/8; blueprint | Solver: same inputs and effective ABI on both sides; Report: immutable selection controls | Solver and Report |
| Project API, backend, ordering and build identity | Day 3/4/8 | Solver: effective path/status; Report: full Git/patch/compiler/flags/binary SHA provenance | Solver and Report |
| CHOLMOD source, loaded binary, modules, build and dependencies | Day 2/3/8 | Solver: loaded library and control capture; Report: source commit, absolute dependency paths and byte hashes | Solver and Report |
| Host, UTC time, command/cwd, requested/observed threads | Day 3/5/7 | Report: complete environment and hosted IDs where applicable; Benchmark: clock/thread verification | Report and Benchmark |
| Methodology paths and bytes | Day 3/4/5/7 | Report: SHA-256 all referenced contracts for generated runs; do not put them into `cell_id` | Report |
| Per-side factor/solve status, finite solution and metrics | Day 4/7 | Solver: raw outputs; Report: recompute independent residual/oracle/backward-error gates and cross-solver consistency | Solver and Report |
| Timing pairs, raw ordered samples and summaries | Day 5/7 | Benchmark: five warmup and 50 alternating timed pairs per window; Report: hashes and recomputed statistics | Benchmark and Report |
| Artifacts and publication/claim scope | Day 6-8 | Report: safe paths, hashes, status/claim consistency; Publication reviewer: attribution and module terms | Report and Publication reviewer |

## Boundary And Risk Ledger

Generated attempts belong only under
`build/external-baseline/<selection_id>/<run_id>/`; the pilot currently
writes `attempt.json`, not validated `evidence.json`. The external record is
not a selected comparison or benchmark family. Do not edit
`selected_report_targets.tsv`, `report_families.tsv`,
`normalize_report_index.py`, `run_external_comparison.py`, or the canonical
`nos4` benchmark to accommodate it. Existing comparison rows use a
source-controlled dense helper, not an independently packaged CHOLMOD
baseline.

Cross-field risks requiring semantic checks beyond JSON shape: partial
`cell_id` promotion; missing/short Git or loaded-binary identity; mismatched
selection despite equal labels; null metrics without reason; status
contradictions; a single-side pass becoming paired evidence; nanosecond
unit/sample drift; missing or escaping artifacts; and publication/portable
speed claims inferred from a local cell. The report owner must test these
with positive and negative fixtures before pilot preflight can pass.

This inventory closes no external correctness or timing claim. Publication
review remains pending. Day 2 freezes the status/unit/provenance decisions;
Day 3 freezes canonical identity and creates the schema authority.
