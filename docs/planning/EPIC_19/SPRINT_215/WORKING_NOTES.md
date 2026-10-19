# Sprint 215 Working Notes: External Baseline Evidence Blueprint

## Goal And Boundary

Produce an executable external-baseline methodology and report contract, with
one selected pilot only if its inputs and validation are feasible. Sprint 212's
selected canonical benchmark remains threshold-free. Existing selected
comparison rows use source-controlled dense reference helpers; they are not
versioned third-party solver results or state-of-the-art evidence.

Day 1 is evidence intake. Day 2 surveys version-pinned candidates and fixture
bytes. Day 3 selects a correctness-only CHOLMOD blueprint cell and freezes its
comparison inputs; no pilot has run. Day 4 fixes numerical acceptance rules;
Day 5 fixes the optional measurement and provenance methodology. Day 6
selects a separate external-evidence schema home; Day 7 specifies its
validation and compatibility rules. Day 8 accepts that isolated design and
authorizes one gated, opt-in local correctness-only pilot attempt. No external
solver or timing run has been made through Day 10, and no selected report
target is promoted.
Day 9 adds an opt-in C/Python harness and synthetic parser tests. The helper
was compiled but not executed; the v1 validator and complete runtime
provenance remain gates before any observed pilot record.
Day 10 reproduced the absent-schema/validator blocker through a fail-closed
wrapper preflight. It recorded package/fixture discovery only, with no solver
command or timing result.
Day 11 audits item 215.4 as blocked, not executed; the claim matrix keeps
external pilot, cross-matrix, cross-solver, hosted, cross-platform and
state-of-the-art claims unearned. Benchmark and maintainer documentation now
distinguish internal selected freshness from this external design.
Day 12 completes the public/maintainer claim boundary and adds focused
documentation mutations. It reconciles the planning JSON, proposed v1
contract, and actual provisional runner fields without promoting an attempt.
Day 13 passes the final-branch local C, schema/corpus, report, benchmark,
docs and synthetic pilot checks. The retained Day 10 raw preflight logs
rehash successfully; the absent v1 schema/validator still blocks execution.
Day 14 closes the blueprint, item audit, residual handoff and retrospective
as a design/blocked-pilot outcome. No external solver, timing or hosted result
was added by closeout.

## Item Ledger

Owners below are provisional functional reviewers, not assigned individuals.
An acceptance check identifies what later work must demonstrate; listing a
command does not assert that it has been run.

| Item | Provisional owner | Planned evidence artifact | Provisional acceptance check | Current state |
| --- | --- | --- | --- | --- |
| 215.1 Baseline Selection | Benchmark and solver maintainers; corpus maintainer for matrices | Days 2-3 candidate survey and baseline decision | Exact third-party project, immutable version/acquisition, solver/operation, matrix identity/provenance and exclusions recorded; candidate can be reproduced or is explicitly blocked. | Day 3 decision recorded: CHOLMOD 5.3.4 / SuiteSparse `v7.12.2` on hash-pinned `symmetric_4.mtx`; execution and publication gates remain. |
| 215.2 Methodology Specification | Numerical solver and benchmark maintainers | Days 3-5 comparison-cell, correctness, tolerance, measurement and provenance contracts | Correctness gate precedes timing; named phase boundaries, repeat/warmup/dispersion and environment fields are predeclared. | Specification complete through Day 5: identity/conversion, numerical gates, optional timing phases/repeats/dispersion and provenance are predeclared. Pilot execution remains separate. |
| 215.3 Report Schema Design | Report and corpus maintainers | Days 6-8 schema design, validator/fixture evidence | External identity and evidence status cannot be confused with source-controlled helper or canonical benchmark freshness; `python3 scripts/validate_corpus_schema.py` and focused report tests pass if affected. | Design accepted; Day 7 planning fixture parses. Schema, validator and production cell-ID test vector remain absent and are residual gate 1. |
| 215.4 Selected Pilot | Selected solver owner with report/benchmark maintainers | Days 8-11 pilot or executable blocked design | One pinned comparison cell has raw inputs, correctness outcomes, provenance and rerun instructions, or a named blocker and exact future checks; no unsupported timing verdict. | Closed for Sprint 215 as `blocked`: absent v1 schema/validator stopped the Day 10 preflight before either solver ran. The fallback harness and exact retry gates are documented; no numerical or timing result. |
| 215.5 Claim Documentation | Documentation and report maintainers | Days 11-12 public/maintainer claim-boundary updates | Blueprint, local pilot and any hosted evidence are distinguished; no broad external parity, portable speed or state-of-the-art claim. Run `python3 tests/test_selected_performance_docs.py` and `make docs-check` if related docs change. | Complete on Day 12: README, solver guide, benchmark and maintainer wording plus a four-surface claim regression guard. No external result is claimed. |
| 215.6 Validation And Closeout | Sprint owner with affected surface owners | Days 12-14 validation record, closeout and residual queue | Record actual schema/report/comparison/docs/pilot outcomes; run `make format && make lint && make test` if any `.c` or `.h` changes. | Day 13 integrated local validation passed all executable checks. Day 14 closes blueprint, queue, audit and retrospective; external v1 absence remains an explicit pilot blocker. |

## Day 1 Contract Ledger

| Surface | Current authority and observed boundary |
| --- | --- |
| Product scope | `../PROJECT_PLAN.md` Sprint 215 specifies six items and an optional narrow pilot. It does not authorize a broad parity claim. |
| Benchmark decision | `../SPRINT_212/artifacts/day5-product-policy-decision.md` selects threshold-free freshness for `SRT-BENCH-REFACTOR-CSC-NOS4`; one repeat, no warmup, no variance, no baseline or threshold. |
| Benchmark guide | `benchmarks/README.md` documents canonical CSV, `manifest.txt`, `index.tsv`, selected Linux/macOS lanes and local sentinel separation. `benchmarks/bench_refactor_csc.c` records internal refactor/solve timings and residuals, not an external solver result. |
| Selected report authority | `tests/corpus/manifests/selected_report_targets.tsv` lists six selected comparison targets and one selected benchmark target; `tests/corpus/manifests/report_families.tsv` distinguishes `comparison` generated-local rows from `benchmark` measurements. |
| Comparison harness | `scripts/run_external_comparison.py` emits fixture-local project/helper observations, dependency status, study rows, summary and manifest under `build/comparison/*/`. Its baseline type is `external-process-source-controlled-helper`, not a third-party package. |
| Corpus | `tests/corpus/manifests/fixtures.tsv`, `generators.tsv`, `optional_data.tsv` and their schemas own maintained fixture provenance; `tests/data/suitesparse/` contains checked-in Matrix Market inputs. The optional SuiteSparse subset is disabled by default. |
| Generated evidence | `build/bench-reports/canonical/`, `build/comparison/*/`, and normalized report indexes are generated output, not source-controlled external-baseline proof. |
| Hosted lanes | Linux/macOS selected comparison and selected performance workflows own named freshness artifacts. Their workflow definitions do not prove a new Day 1 run, external-library parity or comparable timing across runners. |

The detailed inventory and source pointers are in
`artifacts/day1-external-baseline-evidence-intake.md`.

## Candidate Register

| Candidate | Version/acquisition | Solver operation | Matrix subset | Status and next evidence |
| --- | --- | --- | --- | --- |
| SuiteSparse CHOLMOD | SuiteSparse `v7.12.2`, commit `42151688813c45846a597edcb601435a0e38f3dd`; local Homebrew `suite-sparse 7.12.2` and `cholmod` 5.3.4 detected | Sparse SPD Cholesky factor and one solve | Hash-pinned `symmetric_4.mtx` and fixed RHS only; `tridiagonal_20.mtx` and `HB/nos4` are later cells | **Selected for correctness-only blueprint/pilot cell**, not executed. Verify binary/dependency/build provenance, effective controls and publication terms before claiming evidence. |
| SciPy SuperLU wrapper | SciPy `v1.17.0`, commit `8c75ae75176236f233824e9a0483c26a69e6dfec`; pinned PyPI wheel route in Day 2 artifact | Sparse square LU factor via `splu`, then one solve | Inline `lu_nonsym_square_5` pending canonical materialization/hash; `unsymm_5.mtx` excluded as a nonsymmetric representative | Deferred, not selected. SciPy/NumPy are absent locally; dependency lock and pivot/ordering alignment remain gates. |

Potential **internal operation families for screening**, not baseline choices:
QR minimum-norm/least-squares, partial-SVD top-k, nonsymmetric LU solve, and
SPD Cholesky solve from the selected comparison targets; CSC refactor/solve
from the selected benchmark row. These are different problem cells and must
not be pooled into one speed or correctness ranking.

## Decision Log

| Day | Decision | Reason / follow-up |
| --- | --- | --- |
| 1 | Preserve Sprint 212 threshold-free canonical policy. | Selected freshness has no statistical or external-baseline contract; external methodology cannot silently promote it. |
| 1 | Treat current dense helpers as internal source-controlled references. | The comparison harness names helper files as `baseline_version`; that is not a versioned external solver package. |
| 1 | Defer baseline, matrix subset and pilot selection. | Day 2 provenance/feasibility survey and Day 3 selection are required first. |
| 2 | Keep CHOLMOD and SciPy as version-pinned candidates, with CHOLMOD first in feasibility order. | CHOLMOD matching version is discoverable locally; SciPy is not installed. Availability does not prove a valid comparison. Day 3 must select or block one cell. |
| 2 | Exclude `unsymm_5.mtx` as a nonsymmetric-LU representative. | Its Matrix Market header says `general`, but the listed coefficients are symmetric. Use a truly nonsymmetric fixture with canonical bytes instead. |
| 2 | Preserve source/license attribution gates for `HB/nos4` and module-level CHOLMOD licensing. | Matrix Collection terms and CHOLMOD module licenses differ from the repository's MIT license; do not treat a path/name as redistribution approval. |
| 3 | Select CHOLMOD 5.3.4 / SuiteSparse `v7.12.2` against `symmetric_4.mtx` for one correctness-only SPD factor-and-solve pilot cell. | Local package is discoverable, exact fixture bytes and RHS are fixed, and ordering/backend controls can be pinned. No run or timing claim follows from selection. |
| 3 | Defer SciPy/SuperLU and all secondary matrix cells. | Missing local NumPy/SciPy, canonical nonsymmetric fixture bytes, dependency/pivot evidence, or separate matrix provenance prevent silent substitution. |
| 3 | Require full comparison identity and input-conversion checks. | A matrix name or version string alone cannot distinguish solver path, binary, backend, build, thread or input changes. Day 4 and Day 5 must resolve numerical and measurement rules separately. |
| 4 | Freeze independent solver-status, residual and oracle-error gates using float64 precision and a cell-specific condition bound. | The selected 4 x 4 SPD input has known scale and `K<=17/7`; thresholds are specified before observing solver output, not fitted to a pilot. |
| 4 | Treat ineligible, mismatched and failed eligible solves separately. | `not_comparable` is not a passing numerical result; either side failing its gate invalidates any timing comparison. Direct solvers have no iteration or convergence count to invent. |
| 5 | Define a separate optional local timing protocol but keep the selected pilot correctness-only. | A tiny 4 x 4 cell and unequal internal analysis/conversion paths support only descriptive call windows after both numerical gates; no winner, ratio or threshold follows. |
| 5 | Require raw paired samples and reproducible binary/host provenance for any later timing row. | Version strings, environment variables, one sample or unmatched hosts cannot establish comparable performance evidence. |
| 6 | Keep external-baseline evidence in a separate versioned JSON schema and local run bundle. | Current selected comparison rows are source-controlled-helper comparisons, and the selected benchmark is threshold-free freshness on a different operation/matrix. Neither can carry a third-party observation without changing its meaning. |
| 6 | Separate the selection ID from a complete cell ID. | A blueprint or blocked attempt may lack runtime host, binary or build identity; a hash over an incomplete tuple would falsely imply reproducible evidence. |
| 7 | Keep the proposed v1 validator stdlib-only and fail closed on missing provenance, unit/status drift and unsupported claims. | A parseable blueprint is not a generated run; observed or measured rows require full cell identity, both numerical sides and hashed raw artifacts. |
| 7 | Preserve existing manifests and normalizer semantics while designing external validation. | New external records remain separate; any future selected-family migration needs explicit tests and approval, not an implicit row projection. |
| 8 | Accept the separate v1 schema design and planning-only blueprint, but not an observed record without an implemented validator and complete cell-ID test vector. | The design represents planned, blocked and observed states without changing selected report or canonical benchmark semantics. |
| 8 | GO for one opt-in local correctness-only CHOLMOD pilot preparation and gated attempt. | Fixture and installed package hashes match the Day 3 selection; execution provenance and effective controls must still be captured, and publication needs separate license/attribution approval. |
| 9 | Keep the opt-in harness output provisional in `attempt.json`, not `evidence.json`. | A compiled helper and parsed side outputs are not v1-validated observations; raw logs and hashes must precede any promotion. |
| 10 | Stop before solving when the v1 schema or validator is unavailable. | Day 8 requires negative-fixture validation before execution; local package availability alone is not an accepted external result. |
| 10 | Classify the selected attempt as blocked, with no numerical or timing claim. | The generated preflight bundle has `sides={}`, `timing=not_run` and no solver commands; a baseline-version mutation also blocks before execution. |
| 11 | Close item 215.4 as a named blocker, not a comparison result. | Missing v1 validation halted both solvers; the Day 10 attempt establishes only preflight identity and guard behavior. |
| 11 | Restrict claim wording to blueprint and blocked-preflight facts. | The claim matrix leaves selected numerical evidence, cross-matrix, cross-solver, hosted external, cross-platform, speed and state-of-the-art claims unearned. |
| 12 | Finish item 215.5 with explicit public routing and a bounded wording guard. | README and solver selection distinguish the CHOLMOD design from existing selected helper/benchmark rows; synthetic claim mutations reject broad promotions while allowing explicit non-claims. |
| 12 | Keep proposed v1 fields separate from actual provisional attempt fields. | Day 7 `evidence_status=planned` and null `cell_id` describe planning JSON; Day 10 `provisional_status=blocked`, `sides={}` and scalar `timing=not_run` describe an unvalidated generated attempt. |
| 13 | Accept local branch quality and compatibility checks without promoting external evidence. | Full C gate, focused helper compile/link, corpus/schema, report/comparison, benchmark, docs and synthetic pilot checks passed; 16/16 retained preflight raw logs rehashed. External v1 schema/validator absence still blocks solver execution and accepted rows. |
| 14 | Close Sprint 215 as blueprint plus blocked pilot, not an external comparison. | Six item outcomes and 166 estimated hours are reconciled without invented actual hours. The residual queue names owners and gates for v1 implementation, runtime provenance, publication, expanded cells, hosted evidence and future claims. |

## Risk Register

| Risk | Consequence | Mitigation owner / trigger |
| --- | --- | --- |
| Helper described as third-party baseline | False external parity claim | Report owner: preserve explicit baseline type and non-claim in schema and docs. |
| Mismatched solve phases or semantics | Misleading speed/correctness result | Solver and benchmark owners: compare exact operation, precision, ordering, backend, thread count and phase boundaries before accepting a cell. |
| Single-sample or unmatched host timing | Unsupported speed ranking | Benchmark owner: retain threshold-free status; specify warmup, repeats, dispersion and machine provenance before timing interpretation. |
| Matrix provenance or license unclear | Pilot cannot be reproduced/published | Corpus owner: identify file hash, source, license/terms and exact subset on Day 2. |
| Optional external data unavailable | Missing rows counted as pass | Corpus/report owners: use explicit blocked, skipped or not-run status. |
| Selected report schema drift | Existing freshness gates weakened | Report owner: inspect normalizer/manifest compatibility and mutation tests before extending a selected row. |
| External dependency drift | Reproduction changes with package or transitive versions | Pilot owner: pin source/wheel hash, NumPy/BLAS or CHOLMOD dependencies, build flags and actual binary provenance before execution. |
| Selected CHOLMOD package has unknown build/module composition | Different backend, ABI, thread behavior or redistribution terms invalidate the chosen cell or publication | Pilot owner: inspect actual loaded binary/dependencies, effective CHOLMOD controls, build mode and module terms; block execution or publication until the relevant gate is resolved. |
| Successful process exit misreported as numerical correctness | A failed factorization, nonfinite solution or large residual becomes false external evidence | Solver/report owners: require per-phase statuses, factor diagnostics, independent residual and oracle-error limits, and explicit failure classification from Day 4 contract. |
| Microbenchmark noise or unmatched phase boundaries | Misleading one-cell speed ranking | Benchmark/report owners: use Day 5 prepared-input windows, paired raw samples, clock-resolution checks and explicit `not-evidence`; do not turn the correctness-only pilot into a speedup claim. |
| External evidence accidentally inherits selected freshness | An optional missing package appears as a required CI failure or a source-controlled helper `pass` | Report/corpus owners: keep the separate Day 6 schema and status vocabulary out of selected manifests until a reviewed, tested promotion. |
| Permissive JSON or incomplete cell IDs admit false evidence | Missing package/build identity, NaN, unit drift or altered raw artifacts appear measured | Report owner: implement Day 7 duplicate-key/type/hash/status checks and mutation fixtures before accepting pilot output. |
| Day 8 local candidate differs at execution | A discovered library hash or package version is mistaken for the loaded runtime | Pilot owner: recheck package and fixture hashes, loaded dependencies, build/module/ABI identity and effective controls; emit `blocked` if incomplete. |
| Provisional harness result mistaken for accepted evidence | A two-sided output is reported without v1 schema checks or complete cell identity | Report owner: keep `attempt.json` separate; implement validator, negative fixtures and provenance review before any `evidence.json`. |

## Changed-Surface Tracker

| Surface | Change through Day 10 | Later candidate work |
| --- | --- | --- |
| `SPRINT_215/PLAN.md` | Existing uncommitted sprint plan; unchanged on Days 1-10. | Governs day-by-day scope. |
| `SPRINT_215/WORKING_NOTES.md` | Created Day 1; updated selection, correctness, measurement, schema, pilot and claim decisions through Day 12. | Update validation and residuals during Days 13-14. |
| `SPRINT_215/artifacts/day1-external-baseline-evidence-intake.md` | Created intake inventory. | Day 2 uses its unknowns. |
| `SPRINT_215/artifacts/day2-candidate-baselines-and-matrices.md` | Created versioned survey, fixture hashes, exclusions and provisional pilot shortlist. | Day 3 chooses or records a blocker. |
| `SPRINT_215/artifacts/day3-baseline-selection-decision.md` | Selected one versioned CHOLMOD correctness-only cell and documented deferred choices and execution/publication gates. | Day 4 numerical criteria and later pilot evidence. |
| `SPRINT_215/artifacts/day3-methodology-inputs.md` | Frozen comparison identity, exact RHS, conversion invariants and input eligibility. | Day 4 tolerances; Day 5 timing protocol. |
| `SPRINT_215/artifacts/day4-correctness-and-tolerance-contract.md` | Predeclared precision/scale/conditioning-based numerical limits, statuses, negative fixtures and non-comparability rules. | Day 5 timing protocol; later pilot/validator must apply these gates unchanged. |
| `SPRINT_215/artifacts/day5-external-comparison-methodology.md` | Defined optional call-level timing windows, warmups, paired raw samples, dispersion, failure handling, provenance and rerun acceptance. | Day 6 report-schema design; later pilot remains correctness-only unless a separate timing run satisfies this contract. |
| `SPRINT_215/artifacts/day6-external-report-schema-design.md` | Inventoried current report authorities and selected a separate external-evidence JSON schema with field/status dictionary and example row shapes. | Day 7 executable validation design and Day 8 schema acceptance. |
| `SPRINT_215/artifacts/day7-schema-validation-design.md` | Specified stdlib validator interface, identity hashing, status/field rules, mutation fixtures and compatibility checks. | Day 8 schema acceptance and bounded implementation decision. |
| `SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json` | Added complete parseable planning-only v1 record with null unverified runtime fields and no observed results. | Positive fixture for later validator; never publish as pilot evidence. |
| `SPRINT_215/artifacts/day8-pilot-feasibility-decision.md` | Accepted isolated v1 schema design, recorded local candidate rechecks and one gated pilot GO decision. | Day 9 implements opt-in harness and validator or records an executable fallback. |
| `SPRINT_215/artifacts/day9-selected-pilot-harness.md` | Recorded opt-in C/Python helper, raw attempt bundle, synthetic tests and unaccepted-evidence boundary. | Day 10 execution or named blocked outcome. |
| `SPRINT_215/artifacts/day10-selected-pilot-outcome.md` | Recorded reproduced schema/validator blocker, preflight command/results, negative baseline mutation and fallback requirements. | Day 11 claim audit; future validator/provenance implementation remains separate. |
| `SPRINT_215/artifacts/day11-evidence-and-claim-matrix.md` | Audited blocked item 215.4 and recorded approved/rejected wording, claim scopes, source links and remaining gates. | Day 12 public wording and regression review. |
| `benchmarks/README.md`, `docs/maintainer_guide.md` | Day 11 distinguishes internal selected freshness from the blocked external CHOLMOD cell and names separate promotion gates. | Day 12 public/maintainer wording completion. |
| `SPRINT_215/artifacts/day12-claim-documentation-and-validation-plan.md` | Closed claim wording, reconciled planned versus actual field names, and prepared exact Day 13 validation commands/expectations. | Day 13 executes and records the acceptance matrix. |
| `SPRINT_215/artifacts/day13-integrated-validation.md` | Recorded exact local commands/exits, generated-row identity, preflight raw-log integrity and unresolved external v1 blocker. | Day 14 closeout and residual handoff. |
| `SPRINT_215/STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md`, `RESIDUAL_QUEUE.md` | Day 14 selection/method/schema/conditional-rerun contract and prioritized functional-owner handoff. | Future v1 implementation and solver execution require a new gated decision. |
| `SPRINT_215/artifacts/day14-closeout-review.md`, `RETROSPECTIVE.md` | Day 14 six-item audit, planned-versus-observed pilot reconciliation and sprint assessment. | Sprint 216 Epic 19 residual integration. |
| `README.md`, `docs/solver_selection.md`, `tests/test_external_baseline_claim_docs.py` | Day 12 public routing and four-surface wording guard with mutation/non-claim cases. | Run focused docs guard in Day 13 validation; avoid interpreting it as general semantic proof. |
| `scripts/cholmod_pilot.c`, `scripts/run_cholmod_pilot.py`, `tests/test_run_cholmod_pilot.py` | Day 9 opt-in helper/wrapper/tests; Day 10 wrapper now gates before solving and test covers a changed baseline version. | Implement v1 validator and full runtime provenance before any accepted solver result. |
| Manifests, workflows, normalizer | No Sprint 215 changes through Day 12. | Keep current row semantics; Day 13 validates compatibility without promoting an external row. |

## Validation Matrix

| Scope | Candidate command or check | Status through Day 13 |
| --- | --- | --- |
| Corpus schema | `python3 scripts/validate_corpus_schema.py` | Passed on Day 13; source-controlled corpus only, not external v1 validation. |
| Selected comparison freshness | `make report-index-comparison-freshness`; `python3 tests/test_run_external_comparison.py`; `python3 tests/test_selected_report_targets_manifest.py` | All passed on Day 13; 46 generated local selected rows use source-controlled helpers. No CHOLMOD row was promoted. |
| Selected benchmark freshness | `make bench-canonical-report-freshness-tests`; `make bench-canonical-report-freshness` | Both passed on Day 13; selected `bench_refactor_csc` is local, repeat-one and threshold-free. |
| Normalizer and selected manifest | `python3 tests/test_normalize_report_index.py`; `python3 tests/test_selected_report_targets_manifest.py` | Both passed on Day 13; no external family or selected-manifest edit. |
| External blueprint syntax | `python3 -m json.tool docs/planning/EPIC_19/SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json` | Passed on Day 13; semantic validator is designed but not implemented. |
| External v1 implementation | `test -f tests/corpus/schemas/external_baseline_evidence_v1.json`; `test -f scripts/validate_external_baseline_evidence.py` | Both returned 1 on Day 13 as expected: schema and validator are absent. This blocks an accepted pilot result. |
| Documentation | `make docs-check`; `python3 tests/test_selected_performance_docs.py`; `python3 tests/test_external_baseline_claim_docs.py` | All passed on Day 13; `docs-check` reported 18 checked-in public headers and matching generated reference/source pages. |
| Day 12 contract syntax and compatibility | `python3 -m json.tool docs/planning/EPIC_19/SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json`; `python3 scripts/validate_corpus_schema.py`; `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_normalize_report_index.py` | All passed on Day 12. JSON parse is syntax only; corpus/manifest/normalizer checks do not implement external v1 acceptance. |
| Opt-in pilot synthetic tests | `python3 tests/test_run_cholmod_pilot.py`; direct strict compile/link of `scripts/cholmod_pilot.c` | Nine synthetic tests, helper clang-format and strict compile/link passed on Day 13. No external solver execution. |
| Day 10 preflight | `python3 scripts/run_cholmod_pilot.py --fixture tests/data/symmetric_4.mtx --output build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-preflight-20261019 --correctness-only --execute` | Day 10 exit 2 was the named blocker. Day 13 rechecked the retained `attempt.json` invariant/digest and 16/16 raw-log hashes; no solver command, side, timing or `evidence.json` exists. |
| C quality gate | `make format && make lint && make test` if `.c` or `.h` changes | Full chained gate passed on Day 13 for the final branch state; the opt-in helper also passed its own focused format and strict compile/link checks. |
| Day 14 closeout | Relative Markdown link review; generated-file exclusions; `git diff --check`; focused docs/claim checks | Passed: all new closeout relative links resolved; `git check-ignore -v` confirmed `build/` and `docs/api/` exclusions; tracked diff and edited-Markdown trailing-whitespace checks were clean; `make docs-check`, `python3 tests/test_external_baseline_claim_docs.py` (4 methods), and `python3 tests/test_selected_performance_docs.py` all exited 0. Docs-only closeout did not rerun the Day 13 C gate. |

Day 1 source-pointer resolution and trailing-whitespace checks passed for the
new Markdown. No benchmark, comparison, schema, hosted, or C quality result is
inferred from this documentation check.

## Day 2 Survey And Day 3 Handoff

The versioned candidate, operation, platform, license and fixture survey is in
`artifacts/day2-candidate-baselines-and-matrices.md`. The narrowest apparent
pilot cell is a local SPD factor-and-solve comparison on `symmetric_4.mtx`
against CHOLMOD `v7.12.2`, but it is **not selected**. Day 3 must confirm
matrix load/expansion, RHS, float64 configuration, natural-ordering and
backend controls, package/module provenance, and a correctness-only initial
claim, or record a named blocker. SciPy/SuperLU LU is a second candidate with
an absent local dependency and a still-unmaterialized nonsymmetric fixture.
Neither direction inherits Sprint 212's selected benchmark timing policy.
`make docs-check` passed locally on Day 2; trailing-whitespace inspection
found no matches in the edited Markdown. No numerical or C gate was run.

## Day 3 Decision And Day 4 Handoff

`artifacts/day3-baseline-selection-decision.md` selects SuiteSparse CHOLMOD
5.3.4 (`v7.12.2`) on exact `symmetric_4.mtx` bytes with one fixed RHS and an
explicit no-reorder/project linked-list versus CHOLMOD natural/simplicial
solve path. `artifacts/day3-methodology-inputs.md` freezes the comparison-cell
identity, matrix conversion invariants and inapplicable-input handling. The
local Homebrew library was discovered, not exercised; its loaded binary,
transitive dependencies, build/module settings, ABI and effective thread
controls need verification before the pilot. Fixture authorship and CHOLMOD
module terms remain publication gates. Day 4 must predeclare numerical status,
residual/error formulas and tolerances before any result is classified. Day 5
must separately define timing; no speed evidence is implied here.
`make docs-check` passed locally on Day 3; Markdown trailing-whitespace
inspection found no matches. No solver, benchmark, comparison, or C gate was
run.

## Day 4 Correctness Contract And Day 5 Handoff

`artifacts/day4-correctness-and-tolerance-contract.md` fixes the selected
cell's status checks, finite-value requirement, infinity-norm residual and
oracle-solution-error formulas, scale/conditioning-based float64 limits and
negative fixtures. Each side must pass independently on the same verified
input before a pair receives `correctness-pass`; changed semantics,
inapplicable matrices and missing evidence are not silently accepted. A
failed numerical gate makes any observed runtime `not-evidence`. Day 5 must
define phases, warmups, repeats, dispersion, thread controls and machine/
binary provenance separately. No solver was run or timing verdict made on
Day 4. `make docs-check` passed; trailing-whitespace inspection of the edited
Markdown found no matches. No C, corpus, comparison or hosted gate was run.

## Day 5 Methodology And Day 6 Handoff

`artifacts/day5-external-comparison-methodology.md` completes the item 215.2
method contract: prepared-input setup, factor-ready, one-solve and combined
windows are distinguished; reuse is inapplicable to the selected cell. An
optional later local timing run requires five warmup pairs, 50 alternating-
order timed pairs, raw samples, median/dispersion, clock-resolution checks,
per-trial numerical validation and complete host/build/library provenance.
Unavailable time, memory and fill are explicit states, never zero. The
selected pilot remains correctness-only. Day 6 can define a separate external
evidence schema using these fields without recasting Sprint 212 canonical
freshness or current source-controlled helper rows as third-party timing
results. No solver or benchmark was run on Day 5. `make docs-check` passed;
trailing-whitespace inspection of the edited Markdown found no matches.

## Day 6 Schema Inventory And Day 7 Handoff

`artifacts/day6-external-report-schema-design.md` inventories report-family
contracts, selected targets, normalized-index field/status behavior,
source-controlled-helper comparison rows and the repeat-one canonical
benchmark bundle. It chooses a separate versioned JSON evidence schema and
local run bundle. A stable selection ID can exist before execution; a full
cell ID requires actual binary/build/host provenance. The field dictionary
distinguishes a planning blueprint, blocked or non-comparable attempt,
observed one-cell correctness, and a separately valid local timing run.
Examples are explicitly synthetic, not pilot evidence. Day 7 must define
parseable fixtures, cross-field validation and compatibility checks; no
existing selected manifest, normalizer or freshness contract changed here.
Day 6 checks passed: `python3 scripts/validate_corpus_schema.py`,
`python3 tests/test_normalize_report_index.py`,
`python3 tests/test_selected_report_targets_manifest.py`, and
`make docs-check`. Edited Markdown had no trailing-whitespace matches.

## Day 7 Validation Design And Day 8 Handoff

`artifacts/day7-schema-validation-design.md` specifies a stdlib-only
validator, complete-cell hashing, field/enum/unit checks, cross-field
status rules, raw artifact integrity and positive/negative mutation cases.
`artifacts/day7-external-evidence-blueprint.example.json` is a complete
parseable **planned** record; its null cell ID, runtime provenance and
results prevent it from being mistaken for generated proof. The proposed
schema and validator are not yet implemented; Day 8 must accept the schema
home and pilot go/no-go before any external observation is reported. Existing
selected manifest and normalizer row meanings remain untouched. Day 7
compatibility checks passed for corpus schema, report normalizer, selected
manifest, selected comparison workflow, canonical benchmark regression, JSON
syntax and `make docs-check`. The selected generating freshness targets were
not run. The fixture SHA-256 matched the example, and edited Markdown/JSON
had no trailing-whitespace matches.

## Day 8 Schema Acceptance And Day 9 Handoff

`artifacts/day8-pilot-feasibility-decision.md` accepts the separate v1
planning/attempt/observation design and keeps the Day 7 JSON as a blueprint,
not evidence. It records a **GO for opt-in local correctness-only pilot
preparation and gated attempt** on the single pinned `symmetric_4.mtx` /
CHOLMOD 5.3.4 cell. The fixture hash, package version/prefix and candidate
dylib hash matched Day 3 discovery; actual loaded dependencies, bottle build
options, effective solver controls and publication terms remain gates.
Day 9 must implement the opt-in harness, complete cell-ID encoding, schema
validator and negative fixtures before any observed result is accepted. If
those gates cannot be met, Day 10 reports a named `blocked` outcome with a
reproducible fallback; it does not invent a pass or timing sample. No
external solver or timing run took place on Day 8. JSON syntax, corpus schema,
normalizer, selected manifest, selected comparison workflow, canonical
benchmark regression and `make docs-check` passed; generating freshness
targets were not run because their surfaces were unchanged.

## Day 9 Harness And Day 10 Handoff

`artifacts/day9-selected-pilot-harness.md` records the opt-in helper,
wrapper, raw-bundle contract and eight synthetic parser/failure fixtures.
The helper linked against the installed local CHOLMOD but was not executed;
the selected pilot still has no observed numerical or timing result. The
runner's `attempt.json` is expressly provisional, even if both Day 4
numerical checks would pass. Day 10 must either implement and pass the v1
validator/cell-ID/provenance gates before accepting an observed record, or
retain a named blocked outcome and reproducible raw attempt. No selected
manifest, report normalizer, workflow or canonical benchmark semantics
changed. `make format`, `make lint`, `make test`, eight focused synthetic
tests, strict helper compile/link and `make docs-check` passed.

## Day 10 Blocker Proof And Day 11 Handoff

`artifacts/day10-selected-pilot-outcome.md` records the selected cell as
**blocked before execution**. The Day 8-required v1 schema and validator are
absent. The opt-in wrapper now checks this gate after hash/package/toolchain
discovery but before build or either solver. The generated local
`attempt.json` captured exact commands and SHA-256-referenced logs; it has
`schema_validation.status=not_run`, `sides={}`, `timing=not_run`, and no
`evidence.json`. A synthetic changed CHOLMOD version also blocks before
solving; the focused test suite now has nine passing cases. Day 11 should
audit only this blocked outcome and the claim boundary. No numerical,
performance, hosted or publication claim follows from the preflight.
`python3 scripts/validate_corpus_schema.py`, JSON parsing of the attempt,
the blocked/no-solver `jq` invariant and `make docs-check` passed on Day 10.

## Day 11 Claim Audit And Day 12 Handoff

`artifacts/day11-evidence-and-claim-matrix.md` closes item 215.4 for this
sprint as a reproducible blocked preflight, not an executed or non-comparable
solver result. The exact project/CHOLMOD `symmetric_4.mtx` cell remains
correctness-only and has no accepted `evidence.json`, numerical pass or timing
sample. The matrix approves blueprint and blocker wording, while selected
external observation, cross-matrix, cross-solver, hosted, cross-platform,
portable-speed and state-of-the-art claims remain unearned. Initial updates
to `benchmarks/README.md` and `docs/maintainer_guide.md` keep this track
separate from internal selected benchmark freshness and source-controlled
comparison helpers. Day 12 should finish public routing and consider focused
claim-regression coverage without altering selected report rows.
`python3 tests/test_selected_performance_docs.py`, `make docs-check`, and
`git diff --check` passed. Edited Markdown had no trailing-whitespace
matches. No `.c` or `.h` files changed on Day 11; no C quality gate or
external solver run is inferred from these documentation checks.

## Day 12 Documentation Closure And Day 13 Handoff

`artifacts/day12-claim-documentation-and-validation-plan.md` completes item
215.5 and gives Day 13 exact commands and expected outcomes for corpus/schema,
selected manifest, normalizer, comparison, benchmark, docs, pilot synthetic
checks and the retained blocked preflight. README and solver-selection text
now identify the `symmetric_4.mtx`/CHOLMOD cell as a separate blueprint, not
the selected Cholesky helper comparison or canonical benchmark row. The
maintainer guide distinguishes Day 7 planning `evidence_status` from the Day
10 runner's `provisional_status` and lists the raw-artifact, provenance,
correctness, optional timing and publication gates. The new four-surface
docs test rejects explicit blueprint/pilot/single-measurement promotions to
state-of-the-art, platform parity, portable speed or cross-platform
performance.

`python3 tests/test_external_baseline_claim_docs.py` passed four tests with
mutation subcases; `python3 tests/test_selected_performance_docs.py`,
`make docs-check`, planning JSON syntax, `python3 scripts/validate_corpus_schema.py`,
`python3 tests/test_selected_report_targets_manifest.py`,
`python3 tests/test_normalize_report_index.py`, and nine synthetic pilot tests
passed. Inspection of the retained Day 10 `attempt.json` confirmed
`record_kind=unvalidated_attempt`, `provisional_status=blocked`,
`status_reason=schema_validation_unavailable`, empty `sides` and no solver
command. Day 12 did not execute either solver or alter any C/header,
external v1 schema, report contract, manifest, normalizer or workflow file.
The Sprint's Day 9 C helper means Day 13 should rerun the final branch's
`make format && make lint && make test` before closeout; no new C gate is
attributed to Day 12 documentation work.

## Day 13 Integrated Validation And Day 14 Handoff

`artifacts/day13-integrated-validation.md` records the local Darwin x86-64
command matrix and exit statuses. The full chained `make format && make lint
&& make test` gate passed on the final branch state. The opt-in C helper also
passed direct clang-format and strict compile/link checks without execution.
Corpus schema, selected-target manifest, normalizer, existing
source-controlled-helper comparison tests and freshness (46 local rows),
canonical benchmark regression and threshold-free local freshness, docs/API
checks, planning JSON syntax and nine synthetic pilot tests all passed.

The retained Day 10 `attempt.json` still has the recorded SHA-256, a blocked
schema-validation status, empty sides and no project/baseline solver commands;
all 16 referenced raw preflight stdout/stderr files rehashed. The proposed
external v1 schema and validator both remain absent, so the external pilot
was **not** rerun or accepted as numerical/timing evidence. Day 14 may close
the blueprint and named blocker with this residual, but cannot claim an
executed CHOLMOD comparison or broader parity/performance outcome. No hosted
workflow ran on Day 13.

## Day 14 Blueprint And Sprint Handoff

`STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md` packages the selected CHOLMOD cell,
Day 3-5 method, Day 6-7 proposed v1 contract, the actual blocked preflight,
conditional rerun order and expansion gates. `RESIDUAL_QUEUE.md` assigns
functional decisions for validator/cell ID, complete runtime provenance,
fixture/module publication review, additional matrices and solver families,
hosted evidence and claim approval. `artifacts/day14-closeout-review.md`
reconciles all six Epic items and planned versus observed pilot evidence;
`RETROSPECTIVE.md` records the sprint assessment. The Day 10 bundle remains
provisional and no external solver or timing run was added on Day 14.
All new closeout relative links resolved, and generated `build/` and
`docs/api/` paths remained ignored. `git diff --check` and a trailing-
whitespace scan of edited Markdown passed. `make docs-check` passed with
18 checked-in headers and matching generated reference/source pages; both
the four-method external-claim guard and selected-performance docs guard
passed. No `.c` or `.h` file changed on Day 14, so the Day 13 full C gate
remains the sprint's final source-quality run.
