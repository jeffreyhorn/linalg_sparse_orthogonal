# Sprint 222 Retrospective

**Sprint:** 222 - One Two-Sided CHOLMOD Correctness Cell

**Duration:** 14 planned days on branch `sprint-222`

**Status:** Two local numerical pairs passed; complete observed v1 evidence and publication remain blocked

## Source And Definition Of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) allocates 140 estimated hours across six items. The [sprint plan](PLAN.md), [working notes](WORKING_NOTES.md), [Day 14 closure ledger](artifacts/day14-final-quality-and-closure.md) and [Sprint 223 handoff](artifacts/day14-sprint223-handoff.md) retain the scope, results and limits. Hours are planning estimates, not measured effort or solver timings. The intended deliverable was a validator-accepted, reproducible two-sided CHOLMOD correctness cell plus a separate exact-bundle rights and claim decision. Both solvers ran and passed local numerical gates, but missing provenance prevented an accepted `correctness_observed` cell. No independent attribution or rights approval was available, so the written disposition is no publication with `publication_review=pending`.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 222.1 Fixture And Rights | 20 h | Frozen 114-byte matrix, RHS, oracle and CSC conversion checked; fixture origin and license inventory recorded. Independent attribution and binary-rights decision pending. |
| 222.2 Runtime Provenance | 28 h | Project binary, loaded CHOLMOD and ten file-backed dependencies, host and requested thread controls captured. Complete bottle/module options, Apple shared-cache BLAS identity and effective BLAS threads remain unverified. |
| 222.3 Two-Sided Execution | 34 h | Fresh project and CHOLMOD processes, separate input/state, raw commands/streams and independent numerical metrics retained. Local two-sided numerical gate passed. |
| 222.4 Validation And Repeat | 24 h | Append-only repeat reproduced solutions and metrics; failed preparations and validator rejections retained. First and repeat v1 records validate only as `blocked`; a common canonical cell ID was not earned. |
| 222.5 Claim Review | 18 h | Exact-bundle review packet and hashed no-publication disposition written. Reviewer identities and approval remain unset; no external claim or artifact was approved. |
| 222.6 Test Closure | 16 h | Applicable local pilot, validator, CSC, install, focused CMake and documentation checks passed. Full hosted CI and platform-wide CTest are not claimed. |
| **Total** | **140 h** | **Local execution and test closure; observed-cell and publication deliverables blocked.** |

## What Went Well

1. **The problem stayed fixed.** The original `tests/data/symmetric_4.mtx` bytes (SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`) matched both attempt copies. The independently checked 4×4 symmetric, seven-stored/ten-expanded matrix, RHS `[13,14,17,16]`, oracle `[1,1,1,1]` and lower CSC conversion matched the Sprint 215 contract.
2. **Both sides actually executed.** The project and CHOLMOD attempts used fresh processes and distinct directories. Saved command manifests, raw stdout/stderr, status fields, solution vectors, loader traces and source snapshots make the local result auditable. The baseline reported sorted packed lower `CHOLMOD_INT` CSC, natural ordering and successful simplicial factorization/solve controls; the project reported its expected linked-list path and successful statuses.
3. **The numerical check was independent of solver reporting.** The Day 8 checker reconstructed the matrix and RHS from the frozen source bytes, computed both true residuals, backward errors and oracle distances, then compared the solutions only after each side passed. The first pair passed 148 checks. The Day 10 fresh pair passed 219 checks and reproduced both stdout streams and numerical metrics. Each paired solution difference was `2.220446049250313e-16`, below the fixed `2.760966058953532e-13` cross-solver limit.
4. **Provenance checks prevented premature promotion.** The Day 9 and Day 10 observed candidates failed the v1 validator at missing `/environment`; the honest blocked records passed. Day 11 mutation checks rejected missing raw output, changed loaded-library bytes, bad ABI, nonfinite metrics, a stale canonical ID and unsupported claim scope. The original and repeat source-patch hashes differed, so identical solutions were not presented as a same-cell v1 repeat.
5. **Rights and publication stayed separate.** The exact 386-file local source index, 39-payload review manifest and hashed Day 12 disposition were retained without publishing binary or raw trace bytes. The disposition has no reviewer signature, approved claim or artifact allowlist. Selected reports and generated API HTML stayed outside the external comparison; Doxygen output remained ignored and local.

## What Didn't Go Well

1. **The selected installed CHOLMOD package lacked enough build provenance.** The loaded dylib and ten file-backed dependencies were hashed, but exact Homebrew bottle build flags and compiled module composition were not established. Apple BLAS/LAPACK images came from the shared cache without an approved equivalent per-image identity. Requested single-thread environment variables and an OpenMP probe did not reveal effective Accelerate BLAS threading. These are material v1 identity gaps, not harmless missing prose.
2. **The first and repeat were not one canonical cell.** Sprint documents accumulated as untracked files between solves, changing dirty-worktree patch digests. Day 6 also lacked an execution-time host fingerprint. The Day 10 pair had a stable local snapshot and matching host fingerprints on both sides, but could not retroactively repair the first pair's identity. Both accepted records remain `cell_id=null`, `evidence_status=blocked`.
3. **No independent rights or attribution decision was obtained.** The fixture-introducing Git commit identifies Jeffrey Horn, but independent matrix authorship and exact wording were not confirmed. Actual CHOLMOD module and transitive notice obligations, plus 12 raw stderr files containing home-directory paths, require reviewer and privacy decisions. The Day 11 proposed one-cell sentence remains conditional; the Day 12 record is a no-publication hold, not a legal denial or approval.
4. **Two local test/setup issues needed correction.** Day 9 record preparation had two Python helper-name errors before validation, retained as failed drafts. A Day 13 parallel runner test initially let its 0.2-second timeout expire before a child flushed output; the test fixture deadline was widened to 1 second while retaining its under-2.5-second completion check, and the 19-test suite passed serially. An API-docs check first ran without pinned PyYAML; the temporary environment with `PyYAML==6.0.3` passed the guard. No solver or validator rule was changed to hide these outcomes.
5. **The raw proof is local by design.** The binary and raw evidence is ignored under `build/external-baseline/` and is not part of this branch's reviewable Git diff. The committed ledgers and hashes describe what was checked locally; a PR reviewer cannot independently inspect the raw bundle from the PR alone. Sharing a raw or redacted derivative requires a separate exact-bundle rights and privacy decision. Hosted CI results for this branch were not available at retrospective drafting.

## Final Metrics And Claim

| Metric | Sprint 222 local state |
| --- | --- |
| Planned effort | 140 h across six items and 14 days; actual hours not tracked |
| Daily evidence | 14 day ledgers, working notes, this retrospective and an ignored append-only local evidence tree |
| First pair | Project and CHOLMOD exits/statuses and CSC passed; 148/148 reconciliation checks |
| Repeat | New two-process pair; 219/219 checks; same solution stdout hashes and numerical metrics as first pair |
| Project / CHOLMOD true residual infinity norms | `3.552713678800501e-15` / `0`; fixed limit `1.4495071809506044e-12` |
| Paired solution difference | `2.220446049250313e-16` in each pair; fixed limit `2.760966058953532e-13` |
| V1 acceptance | Two blocked records valid; two attempted observed records rejected at `/environment`; canonical IDs null |
| Claim and rights | No timing samples, approved external claim or artifact allowlist; attribution and publication review pending |
| Focused local tests | Validator 67, pilot runner 19, claim docs 5, install 26, installed-width 12, focused CMake CSC 1; Make CSC and API documentation checks passed |
| Documentation boundary | 18 generated reference and source pages stayed ignored/local; workflow guard checked 111 records; normalized report index remained 114 rows |

The earned statement is **two locally observed numerical passes on the one frozen 4×4 problem**, with a matching numerical repeat and preserved raw evidence. It is **not** a complete v1 `correctness_observed` cell, an approved external comparison, a timing result, portable parity, a winner or state-of-the-art evidence. The [Day 12 disposition](artifacts/day12-independent-rights-and-claim-decision.md) and its [hashed JSON](artifacts/day12-publication-disposition.json) keep the public claim blocked. No `.c` or `.h` files changed in Sprint 222, so the conditional `make format && make lint && make test` requirement was not triggered. The only tracked code edit widened one Python test fixture's startup margin; its full suite passed. The local focused CMake CSC route passed, but full CTest and hosted Windows acceptance were not claimed.

## Daily Evidence

- [Day 1 selected fixture and evidence inventory](artifacts/day1-selected-fixture-and-evidence-inventory.md)
- [Day 2 rights, retention and attempt layout](artifacts/day2-rights-retention-and-attempt-layout.md)
- [Day 3 project build identity](artifacts/day3-project-build-identity.md)
- [Day 4 CHOLMOD runtime identity](artifacts/day4-cholmod-runtime-identity.md)
- [Day 5 host and execution setup](artifacts/day5-host-execution-setup.md)
- [Day 6 project-side solve](artifacts/day6-project-side-solve.md)
- [Day 7 baseline-side solve](artifacts/day7-baseline-side-solve.md)
- [Day 8 two-sided numerical reconciliation](artifacts/day8-two-sided-numerical-reconciliation.md)
- [Day 9 v1 record and validation](artifacts/day9-v1-record-and-validation.md)
- [Day 10 repeat and identity comparison](artifacts/day10-repeat-and-identity.md)
- [Day 11 validation and retention ledger](artifacts/day11-validation-retention-ledger.md) and [rights review packet](artifacts/day11-rights-and-wording-review-packet.md)
- [Day 12 rights and claim disposition](artifacts/day12-independent-rights-and-claim-decision.md)
- [Day 13 claim scope and focused checks](artifacts/day13-claim-scope-and-focused-checks.md)
- [Day 14 quality closure](artifacts/day14-final-quality-and-closure.md) and [Sprint 223 handoff](artifacts/day14-sprint223-handoff.md)

## Residual Handoff

Keep Sprint 222's raw attempts and rejected candidates local and append-only. A future owner needs one immutable source state for both solvers, execution-time host and effective BLAS threading observations, exact CHOLMOD bottle/module and shared-cache BLAS identities, a complete accepted v1 cell, and named independent attribution/rights decisions for an exact artifact allowlist and sentence. Any redacted disclosure is a new hashed artifact. These are Sprint 222 follow-ups, **not** an implicit dependency or evidence source for Sprint 223.

Sprint 223 separately targets a selected Windows Cholesky freshness promotion. It must obtain exact-commit hosted MSVC run and six-file artifact IDs, align manifest/normalizer/public wording atomically, reject stale or contradictory artifacts, and recheck the final commit. Sprint 222's Mac CHOLMOD local numerical work cannot satisfy that hosted Windows gate or justify broader solver claims. After this branch is pushed, record its actual hosted CI outcomes separately; do not convert a local pass into a hosted pass by inference.
