# Sprint 221 Retrospective

**Sprint:** 221 - External Evidence V1 Validator

**Duration:** 14 planned days on branch `sprint-221`

**Status:** Six planned deliverables closed locally; hosted CI and the first real external cell remain pending

## Source And Definition Of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) allocates 140 estimated hours
across six items. The [sprint plan](PLAN.md), [working notes](WORKING_NOTES.md)
and [Day 14 validation and handoff](artifacts/day14-integrated-validation-and-sprint222-handoff.md)
retain the scope and evidence. Hours are planning estimates, not measured
effort. The objective was to make the optional CHOLMOD pilot pass a real,
fail-closed v1 format/consistency preflight, not to produce an external
correctness or performance result. Local checks passed; the branch's hosted
jobs have not yet run and are not counted as passing evidence.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 221.1 Schema Freeze | 24 h | Closed locally: selected cell, eight statuses, units, provenance, claims and canonical identity frozen. |
| 221.2 Validator Implementation | 30 h | Closed locally: strict JSON, exact shape, cross-field, identity and byte-backed artifact checks with stable diagnostics. |
| 221.3 Fixture Matrix | 28 h | Closed locally: frozen planning input and synthetic non-execution, observed, failed and measured positives with negative neighbors. |
| 221.4 Pilot Preflight | 22 h | Closed locally: real negative self-test before dependency discovery, build and solve; optional CI step registered. |
| 221.5 Mutation Proof | 18 h | Closed locally: 21 syntax-valid CLI mutations plus parser seeds, exact code/pointer verdicts and restored positives. |
| 221.6 Docs And Validation | 18 h | Closed locally: maintainer guide, README entry, corpus/selected-report/API-doc checks and handoff ledger. |
| **Total** | **140 h** | **Local deliverables complete; hosted result and real CHOLMOD cell not established.** |

## What Went Well

1. **The contract was frozen before promotion.** The v1 schema pins one
   matrix, RHS, solver configuration and baseline version. The planning
   blueprint and synthetic canonical byte vector distinguish selected
   controls from per-run provenance, so a different cell cannot borrow the
   same `selection_id` or `cell_id`.
2. **Validation goes beyond JSON shape.** The stdlib validator rejects
   duplicate keys, nonfinite numbers, unknown fields and status conflicts;
   it hashes matrix, methodology and bundle bytes through normalized,
   descriptor-relative paths. Raw solver statuses, controls and solutions
   must match the reported values; paired timing statistics are recomputed
   from hashed samples.
3. **The pilot gate moved ahead of risky work.** The real negative self-test
   runs after the selected matrix hash and before `pkg-config`, archive
   build, helper compilation or either solver. Missing, unreadable,
   malformed or policy-relaxed inputs stop with an empty-sided,
   timing-`not_run` attempt. Raw preflight streams retain byte-exact hashes,
   and an existing attempt directory is not overwritten.
4. **Mutation proof reached intended semantic gates.** Day 12 tested matrix,
   artifact, path, identity, status, unit, timing and claim changes. Day 13
   added method-byte, rehashed raw-output, blocked-performance, duplicate
   dependency and unlisted-review cases. Each CLI case asserts its code and
   pointer, then revalidates the restored positive fixture.
5. **The claim boundary remained explicit.** The optional CI lane runs
   fixture and preflight checks without CHOLMOD. The maintainer guide says
   that a passing validator proves only format and internal consistency,
   not solver provenance, rights approval, portable speed or broad parity.
   Selected reports, benchmark rows and generated API publication routing
   were not changed.

## What Didn't Go Well

1. **The first two day artifacts were reconstructed on Day 3.** They record
   the accepted Sprint 215 contract and owner map, but do not represent two
   separate external executions or independently timed days. The working
   notes make that catch-up explicit.
2. **The early validator was intentionally unusable as a passing gate.**
   Days 3-9 returned `implementation_pending` while shape, semantic and
   artifact checks were built. That prevented premature promotion but meant
   pilot integration could only be completed after the Day 10 negative
   self-test existed.
3. **A mutation expectation initially stopped too early.** The matrix digest
   and fixed sample-count fields are schema-pinned, so a changed value fails
   selection before deeper byte or timing checks. The sample-count proof was
   corrected to remove a raw pair and update its hash, reaching the intended
   semantic timing gate. The planning fixture was also kept free of staged
   execution hashes. The final 45-test validator suite passes.
4. **No actual CHOLMOD result was available.** The positive observed and
   measured fixtures are synthetic; the passing pilot-integration test mocks
   downstream probes and solvers. The pilot still emits an
   `unvalidated_attempt` envelope, not a generated v1 evidence record. A
   successful `two_sided_correctness_candidate` is not accepted parity.
5. **Hosted and publication gates are open.** Local Python, corpus,
   selected-report and API-doc routing checks passed, but the branch had no
   hosted CI result at closeout. The validator can verify the presence and
   digest of a review artifact; it cannot authenticate the human rights
   decision or the origin of a synthetic solver record.

## Final Metrics And Claim

| Metric | Sprint 221 local state |
| --- | --- |
| Planned effort | 140 h across six items and 14 days; actual hours not tracked |
| Daily evidence | 14 primary day artifacts, working notes and this retrospective |
| Schema and identity | One frozen v1 selection, eight statuses, exact canonical byte/hash test vector |
| Validator suite | 45/45 tests passed, including all eight synthetic status classes and 21 syntax-valid CLI mutations |
| Pilot suite | 16/16 tests passed; real self-test, mocked downstream integration and fail-closed paths |
| Corpus and selected reports | Corpus schema, selected comparison workflow, selected target manifest, report-index regression and selected performance docs passed |
| Documentation routing | `make docs-check` and `make api-docs-routing` passed; generated API HTML remains ignored/local-only |
| Hosted acceptance | New optional Python CI step registered; changed-branch run pending at local closeout |
| External evidence | No real two-sided CHOLMOD result, timing sample or publication approval accepted |

The earned claim is a **locally validated v1 evidence format and pilot
preflight** for one selected cell. It is not an independent CHOLMOD solve,
validated external correctness, performance comparison, broad parity,
publication approval or state-of-the-art claim. No `.c` or `.h` files
changed; the conditional C `make format`, `make lint` and `make test` gates
were not applicable.

## Daily Evidence

- [Day 1 evidence contract inventory](artifacts/day1-evidence-contract-inventory.md)
- [Day 2 status, units and provenance freeze](artifacts/day2-status-units-provenance-freeze.md)
- [Day 3 canonical identity and schema authority](artifacts/day3-canonical-identity-and-schema-authority.md)
- [Day 4 strict JSON and structural validation](artifacts/day4-strict-json-and-structural-validation.md)
- [Day 5 semantic status and cell validation](artifacts/day5-semantic-status-and-cell-validation.md)
- [Day 6 safe artifacts and fixture harness](artifacts/day6-safe-artifacts-and-fixture-harness.md)
- [Day 7 planning and blocked fixtures](artifacts/day7-planning-and-blocked-fixtures.md)
- [Day 8 synthetic observed and measured fixtures](artifacts/day8-synthetic-observed-and-measured-fixtures.md)
- [Day 9 fixture closure and pilot gate](artifacts/day9-fixture-closure-and-pilot-gate.md)
- [Day 10 fail-closed pilot preflight](artifacts/day10-fail-closed-pilot-preflight.md)
- [Day 11 pilot integration and mutation seeds](artifacts/day11-pilot-integration-and-mutation-seeds.md)
- [Day 12 semantic mutation proof](artifacts/day12-semantic-mutation-proof.md)
- [Day 13 mutation closure ledger](artifacts/day13-mutation-closure-ledger.md) and [maintainer guide](artifacts/day13-validator-maintainer-guide.md)
- [Day 14 integrated validation and Sprint 222 handoff](artifacts/day14-integrated-validation-and-sprint222-handoff.md)

## Residual Handoff

After pushing, record the exact commit and hosted workflow conclusions;
do not convert local passes into a hosted claim. Sprint 222 must verify
rights and the actual loaded CHOLMOD/module/dependency/ABI identities,
execute both solvers on the frozen input, preserve raw bytes and failed
attempts, materialize and validate a genuine v1 record, repeat the cell,
and obtain an independent exact-bundle publication decision. Rights denial
keeps results private. The correctness-only selection does not authorize
timing, winner, speedup or portable-performance language.
