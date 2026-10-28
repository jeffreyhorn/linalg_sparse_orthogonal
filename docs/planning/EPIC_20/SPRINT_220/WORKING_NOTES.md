# Sprint 220 Working Notes

**State:** Days 1-14 local closeout complete; hosted validation pending.
**Plan:** [Sprint 220 plan](PLAN.md). **Epic scope:**
[project plan](../PROJECT_PLAN.md).

## Day 1 Inventory

The [guard and workflow inventory](artifacts/day1-guard-and-workflow-inventory.md)
maps the current local-only Make route, four embedded Python normalization
stages, shared helper, 190-case fixture suite, three real workflows, nine
artifact-upload steps, and non-workflow publication checks. It is an input
map, not a change to the local-only policy. Day 2 will freeze allowed and
forbidden fixture classifications before the parser is replaced.

## Day 2 Classification Freeze

The [classification and baseline artifact](artifacts/day2-classification-and-baseline.md)
and [190-function fixture manifest](artifacts/day2-fixture-classification.tsv)
freeze the expected allow/reject verdict and rejection diagnostic for every
registered test function. The manifest contains 151 rejection-only, 35
allowance-only, and 4 mixed functions. The mixed functions expand to 26
variant-level cases with 17 rejections and 9 allowances, covering PowerShell
commands, backtick continuations, Windows separators, and folded/literal
paragraph boundaries. These expectations are the fixture oracle for the
future structured parser, not a blanket endorsement of the current text
normalizer.

The matrix separates direct `docs/api` references from broader publication
semantics; it also records false-positive controls for comments, inert
anchors, safe `build` artifacts, unrelated actions, and non-PowerShell
backticks. No parser choice or guard behavior change was made on Day 2.

## Day 3 Edge Freeze And Parser Contract

The [edge fixture matrix](artifacts/day3-edge-fixture-matrix.tsv) adds ten
complete workflow fixtures under `tests/fixtures/api_docs_yaml_edges/`:
eight expected rejections (duplicates, malformed flow, unknown alias, merge
key, custom tag, dynamic publication path, dynamic action identity) and two
expected allowances (static anchor and static path). The
[parser/extraction contract](artifacts/day3-parser-and-extraction-contract.md)
chooses PyYAML node composition with duplicate and unsupported-syntax checks,
explicit provisioning, and per-workflow/job/step records. Item 220.1's
22-hour planned fixture freeze is complete; parser implementation remains
Day 4 onward. The existing guard is unchanged.

The legacy probe allowed `unknown_alias.yml`, `custom_tag.yml`, and
`dynamic_action_identity.yml` despite the new oracle's required rejection.
Other rejected edge cases were generally caught by path-text detection, not
the required parse/unsupported-syntax stage. These discrepancies are inputs
to the Day 9 differential, not waived failures or a change to local-only
policy.

## Day 4 Structured Reader

The [reader implementation and evidence](artifacts/day4-structured-reader.md)
records `scripts/api_docs_workflow_reader.py`, its eleven-test parser suite,
`requirements-api-docs.txt` pin, and isolated Linux/macOS CI test setup.
All three maintained workflows parse, including their nine upload steps;
the Day 3 malformed/ambiguous corpus rejects at its expected reader stage.
The new module is not yet invoked by the production shell guard, so the
legacy guard remains the active local-only enforcement path. Day 5 owns
deeper alias-site and scalar/shell fidelity work; Days 6-9 own policy
integration and differential regression.

## Day 5 Scalar, Alias, And Shell Fidelity

The [Day 5 evidence](artifacts/day5-scalar-alias-shell-evidence.md) records
rendered `run` scalar behavior, alias-use and definition marks, and explicit
shell source/family on each step. Twenty reader tests pass, including folded
versus literal shell comments, flow and alias publishing fields, merge/flow
ambiguity rejection, and unchanged Windows paths/backticks. No policy rules
were moved into the reader; Day 6 still owns the guard handoff.

## Day 6 Parser-To-Policy Handoff

The [handoff and discrepancy log](artifacts/day6-parser-policy-handoff.md)
records the structured preflight in the production guard, source-located
policy inputs, conservative path entries, and maintained workflow comparison.
All 21 jobs, 110 steps, and nine upload actions match the Day 1 inventory;
PowerShell and `cmd` identities remain distinct. The old publication matcher
continues to make verdicts until Days 7-8 port the policy. Five legacy
fixture functions needed either valid YAML repairs or explicit parse-stage
expectations; their rejection verdicts did not change.

## Day 7 Direct Publication Policy

The [Day 7 policy and regression matrix](artifacts/day7-direct-publication-policy.md)
records source-located direct generated-path and broad docs/root publication
checks on structured step records. Dynamic action identities and dynamic or
typed publication paths fail closed; unrelated checkout actions and safe
literal build paths remain allowed. The production guard runs this policy
after its legacy checks, which still own staging behavior until Day 8.

## Day 8 Shell Staging Policy

The [Day 8 staging policy and paired fixtures](artifacts/day8-shell-staging-policy.md)
link shell staging outputs to explicit later upload, deploy, and release
inputs within the same job. POSIX and PowerShell command boundaries are
preserved; the origin step is reported for rejected routes. The staging-only
legacy regex branches were removed because they rejected unrelated uploads.
Direct legacy path/dynamic checks remain pending the Day 9 differential.

## Day 9 Policy Edges And Differential

The [Day 9 comparison ledger](artifacts/day9-policy-differential-ledger.md)
records the pinned-baseline, structured-policy, and current-guard comparison.
The frozen Day 2 corpus produced 225 actual invocation rows, with no workflow
verdict disagreement and no fixture error. The ten Day 3 edge fixtures exposed
three legacy missed bypasses (unknown alias, custom tag, dynamic action
identity); structured and current guards rejected each as required. The
machine-readable [frozen corpus](artifacts/day9-differential-report.jsonl)
and [edge corpus](artifacts/day9-edge-differential-report.jsonl) retain all
verdicts, diagnostics, and locations. Item 220.3's 28-hour policy allocation
is closed; Day 10 retains adversarial mutation and reconciliation work.

## Day 10 Differential Reconciliation

The [resolved classification](artifacts/day10-differential-reconciliation.md)
records a fresh 225-invocation frozen-corpus run and ten-edge run, both
byte-for-byte identical to the Day 9 reports. All frozen allow/reject
verdicts match; the three edge old/new differences are intentional
fail-closed rejections. Of 89 changed rejection diagnostics, 87 preserve
their required substring with better source location; the remaining two
invalid-YAML fixtures now reject at parsing and have explicit rationale.
The differential runner now fails for unexpected current/structured verdicts
and edge-stage mismatches. Its four-test harness suite includes a Doxyfile
mutation proving that the non-workflow output contract remains active.

The [maintained-workflow validation](artifacts/day10-real-workflow-validation.md)
independently counts 21 jobs and 110 steps across the three workflow files,
matching all 110 structured policy records and nine upload actions. The
combined guard passes, with no newly rejected real step. Day 11 owns the
broader mutation gate; no frozen expectation was altered on Day 10.

## Day 11 Mutation Gate And Guide Outline

The [mutation gate and discrepancy ledger](artifacts/day11-mutation-gate-and-discrepancy-ledger.md)
records 13 named cases: three rejected publication routes, five fail-closed
grammar/dynamic cases, and five benign allowances. Every case is asserted
against both the standalone structured policy and combined guard in a
temporary worktree. The three-test suite passes and is now called from the
API-docs Make target and Linux/macOS structured-reader CI steps. An
additional probe demonstrates one legacy false positive: a checkout into
`docs/` followed by a separate `build/output` upload is allowed by the
structured policy but rejected by old whole-workflow matching. Its
investigation is closed, with remediation assigned to Day 13 retirement of
the legacy matcher; it is not a publication exception.

The [maintainer-guide outline](artifacts/day11-maintainer-guide-outline.md)
covers the PyYAML 6.0.3 dependency, supported syntax and shells, fail-closed
forms, reproduction commands, diagnostics, and escalation to an explicit
publication decision. Day 12 will integrate it into the existing guide.
Item 220.4's 20-hour differential allocation is closed; no frozen oracle
expectation changed.

## Day 12 Maintainer Guide And Diagnostics

The [Day 12 guide and diagnostic review](artifacts/day12-maintainer-guide-and-diagnostics.md)
integrates the outline into `docs/maintainer_guide.md`. It documents the
pinned local setup, supported YAML and shell boundaries, source-located
reader/policy failures, direct reproduction commands, and local-only repair
path. README and API-reference wording were reviewed and did not need edits;
both already exclude hosted, committed, and artifact-published generated HTML.
The combined guard's remaining legacy diagnostic precedence is identified
for Day 13 cleanup rather than presented as structured-parser behavior.

## Day 13 Escalation And Rollout Preparation

The [Day 13 retirement and rollout record](artifacts/day13-rollout-and-retirement.md)
documents removal of the legacy whole-workflow YAML scans. The shell guard
retains the non-workflow local-only checks and calls the structured policy
for discovered workflows. The frozen suite, mutation matrix, ten YAML edge
fixtures, and 225-variant differential pass; the known unrelated-checkout
false positive is now an allow assertion. Intentional message changes are
limited to source-located generated-path, dynamic-path, and parse diagnostics.
Make's Doxygen-backed local-only and routing targets pass, and Linux and
Apple-Clang CI now invoke the frozen fixture suite. Item 220.5's 18-hour
guide allocation is closed; Day 14 owns final integration and hosted result
review.

## Day 14 Integrated Gate And Closeout

The [integrated gate and closeout record](artifacts/day14-integrated-gate-and-closeout.md)
reports both Make targets passing, Doxygen coverage 18/18/18, the 190-function
frozen suite, the mutation suite, and routing checks. The full 225-variant
differential and ten YAML edges pass with no current/structured workflow
verdict mismatch. The direct guard checks 110 job/step records; generated
HTML remains ignored, untracked, unstaged, and absent from upload paths.
The input-adapter suite exposed one obsolete wrapper-message assertion; only
that test was corrected, then its suite and the differential were rerun.
No hosted result is claimed for the unpushed branch. Item 220.6's planned
18-hour rollout allocation and the 140-hour sprint deliverable scope are
closed; residual shell/action coverage and hosted validation have owners in
the closeout artifact.

## Owner Map

| Owner | Current role | Next action |
| --- | --- | --- |
| `Makefile`, `docs`, `api-docs-coverage`, `api-docs-local-only`, `api-docs-routing` | Generate/check API docs and run local-only and routing validation in order. | Preserve the passing local route; do not add publication semantics without a new decision. |
| `scripts/check_api_docs_local_only.sh` | Git-state, ignore, Doxyfile, wording, and structured workflow policy invocation. | Keep non-workflow checks and the structured handoff together. |
| `scripts/api_docs_workflow_policy_inputs.py` | Source-located job/step inputs, path entries, effective shell, and explicit staging-link type. | Add fixtures before supporting new workflow fields or shell forms. |
| `scripts/api_docs_workflow_policy.py`, `scripts/api_docs_workflow_staging.py` | Structured direct and shell-staging publication checks with source-located diagnostics and same-job ordering. | Review new publisher actions and command families with targeted fixtures. |
| `scripts/api_docs_workflow_yaml_common.py` | Stateful shell-comment helper shared by structured direct and staging policies. | Preserve while those consumers need it. |
| `tests/test_api_docs_local_only_guard.py` | Frozen temporary-worktree allow/reject corpus. | Keep the verdict manifest stable; reconcile any future diagnostic changes explicitly. |
| `.github/workflows/ci.yml`, `macos-ci.yml`, `windows-ci.yml` | Actual Linux/macOS/Windows workflow inputs; nine `upload-artifact` steps total. | Inspect hosted Linux and Apple-Clang results after push; continue parsing all current workflows. |
| `.gitignore`, `Doxyfile`, `README.md`, `docs/api_reference.md`, `docs/maintainer_guide.md` | Local generated-output and product decision contract. | Preserve local-only wording and ignored output. |
| `scripts/check_api_docs_routing.py`, `tests/test_api_docs_routing.py` | Routing/documentation guard. | Keep source-controlled API-reference routing enforced. |

## Risk And Decision Log

| ID | Risk or decision | State / next owner |
| --- | --- | --- |
| D1-1 | Whole-workflow text normalization can lose field or step boundaries and overjoin staging/upload evidence. | Open; freeze examples Day 2, structured records Days 3-6. |
| D1-2 | YAML anchors, flow collections, folded/quoted scalars, comments, and dynamic values have accumulated specialized handling. | Open; fixture classification Days 2-3, parser tests Days 4-5. |
| D1-3 | Windows `pwsh` and `cmd` differ from POSIX shell escaping; PowerShell backslashes/backticks must survive extraction. | Open; effective-shell contract Day 3, tests Day 5 and policy Day 8. |
| D1-4 | Path staging in one step may feed a later upload, while unrelated steps must not become fictitious single commands. | Open; policy relationship Days 6-9. |
| D1-5 | A YAML parser dependency is not currently declared in the repository. | Decision: pinned PyYAML node composition; Day 4 must provision it explicitly and fail closed if missing, Day 12 documents it. |
| D1-6 | Ignored/untracked, Doxyfile, and product-wording checks are outside YAML parsing and must not be dropped. | Invariant; owner map above, integrated gate Day 14. |
| D1-7 | A local pass does not establish hosted CI behavior or authorize generated HTML publication. | Invariant; hosted evidence at rollout, no policy expansion. |
| D2-1 | Four functions have both pass and fail variants; assigning a single function-level verdict would erase the shell/scalar distinction. | Closed for the frozen oracle by the variant matrix; Day 9 differential runner must preserve variant granularity. |
| D2-2 | Existing tests assert specific diagnostic substrings as well as rejection. | Preserve verdict first; document intentional diagnostic changes before editing the test suite. |
| D3-1 | Legacy guard accepts an unknown alias, custom tag, and dynamic action identity. | Open for structured reader/policy implementation; new oracle requires rejection with the right failure stage. |
| D3-2 | Duplicate and malformed cases can be rejected by incidental path text while YAML ambiguity remains unrecognized. | Open; node-level duplicate, syntax, and unsupported-feature errors on Day 4. |
| D3-3 | YAML 1.1 `on` typing and alias cycles can distort workflow structure. | Contract uses source scalar key lexemes and rejects cycles; verify against real workflows on Days 4-5. |
| D4-1 | `retention-days` is an integer scalar in a maintained workflow; rejecting all non-string `with` inputs would reject valid real input. | Reader retains typed action inputs with tags; policy must reject unsafe typed publication paths rather than discarding values. |
| D4-2 | PyYAML composition resolves aliases to the anchor node, so alias-site marks are not directly retained. | Closed on Day 5: the reader loader records both alias-use marks and anchor-definition marks. |
| D4-3 | A standalone reader test does not prove policy parity. | Old guard stays active until later policy and differential stages; CI reader step is parser coverage only. |
| D6-1 | Invalid YAML fixtures can reject at parsing before the old publication diagnostic. | Five functions identified; three fixtures repaired to valid YAML and two now assert parse-stage rejection. The full 190-function Make gate passes. |
| D6-2 | The handoff validates structured inputs but legacy publication errors still lack job/step attribution. | Open for Days 7-8 policy port; the Day 6 adapter carries location on each candidate and forbids inferred cross-step links. |
| D7-1 | A dynamic action identity with a safe-looking literal path escaped the legacy publication matcher. | Closed by the structured direct policy; full-guard focused regression rejects with workflow/job/step location. |
| D7-2 | Legacy scans still own shell staging and may issue file-scoped diagnostics before the structured checker. | Open through Days 8-9; frozen 190-function suite remains green under the combined guard. |
| D8-1 | Legacy staging regex rejects docs copies even when only unrelated outputs are uploaded. | Closed by structured path relationship and full-guard allowance fixture; other legacy direct checks remain for Day 9. |
| D8-2 | Implicit shell publisher inputs and unsupported command forms need edge-case comparison. | Day 10 reconciled the frozen corpus and inspected maintained `cmd` steps; broader command mutations remain Day 11 work. |
| D9-1 | A function-level pass can hide mixed-variant failures. | Closed by per-`run_guard` capture and explicit 26-variant oracle; runner validates 190 registered functions. |
| D9-2 | Legacy guard misses three Day 3 YAML edge cases. | Closed by Day 10 reconciliation: all three are intended fail-closed rejections with expected failure stages. |
| D10-1 | A verdict-only differential can hide a changed failure stage. | Closed: edge-stage expectations are gated; the two historical invalid-YAML diagnostic changes are explicitly reconciled without changing the frozen verdict oracle. |
| D10-2 | Current `cmd` steps are not shell-staging parsed. | Reviewed: all six maintained `cmd` steps are validation/comparison commands, while both Windows uploads are distinct parsed action steps. Arbitrary `cmd` publication remains outside the current shell-staging recognizer. |
| D11-1 | Legacy whole-workflow matching overrejects an unrelated checkout `path: docs/` when another step uploads only `build/output`. | Closed on Day 13: the superseded scan was removed and the combined-guard probe now asserts allowance; frozen and mutation gates pass. |
| D14-1 | The input-adapter regression still expected the retired extraction wrapper. | Closed: assert the source-located duplicate-key parse error and current policy wrapper; all seven tests pass. |
| D14-2 | Local checks cannot establish hosted Linux/Apple-Clang results. | Open for PR/CI owner after push; inspect the pinned dependency and all structured-reader job steps before claiming hosted validation. |
| D14-3 | New `cmd`, unknown-shell, or unclassified action publication behavior is outside the demonstrated command set. | Workflow guard maintainer owns targeted fixtures and action-input review before expanding claims. |

## Evidence Ledger

| Revision / environment | Command | Result | Scope / limit |
| --- | --- | --- | --- |
| `05dba39abaa7c0acb38ff064690cdf5f67acf352`; local macOS, Doxygen `/usr/local/bin/doxygen`, Python 3.14.5 | `make api-docs-local-only` | Exit 0. Coverage and direct guard passed; 18 public headers, 18 reference pages, 18 source pages; the 190-case Python suite completed without an assertion failure. | Generates ignored local HTML. Not a hosted result or authorization to publish. |
| Same source revision and local host | `python3 tests/test_api_docs_local_only_guard.py` | Exit 0; all 190 registered functions completed without assertion output. | Temporary Git fixture trees; Python test entry point emits no per-case progress. |
| Same source revision and local host | `bash scripts/check_api_docs_local_only.sh` | Exit 0. Git-state, ignore, Doxyfile, wording, and real-workflow checks passed. | Direct guard does not execute the fixture mutations. |
| Same source revision and local host; ten temporary Git fixture trees | `run_guard()` on each Day 3 edge fixture | Five rejected, five allowed. The three unexpected allowances were unknown alias, custom tag, and dynamic action identity; the two safe controls also passed. | Observed legacy behavior only. The matrix freezes eight required rejections and two allowances; failure stage is not established by legacy exit code. |
| Same source revision; local Python 3.14.5, isolated PyYAML 6.0.3 under `/tmp` | `yaml.compose_all(..., Loader=yaml.SafeLoader)` with key/value-node, duplicate, tag, and cycle checks | All three maintained workflows parsed. The six syntax/unsupported edge fixtures were rejected at the expected stage; two dynamic-policy and two safe controls remained structured for policy classification. | One-off extraction feasibility probe, not production guard integration or a full replacement regression suite. |
| Same source revision and local host | `python3 tests/test_api_docs_local_only_guard.py` | Exit 0; all 190 registered legacy fixture functions passed. | Existing guard suite only; the new edge corpus is an oracle for Day 4 wiring and is intentionally not counted as legacy pass evidence. |
| Same branch, local Python 3.14.5; isolated PyYAML 6.0.3 venv | `tests/test_api_docs_workflow_reader.py` | Exit 0; eleven tests passed, including real workflows and Day 3 parser-stage cases. | Parser only, not a replacement policy verdict. |
| Same branch and host | `bash scripts/check_api_docs_local_only.sh` | Exit 0 after CI workflow edits. | Existing guard remains authoritative until the new reader is integrated. |
| Same branch and host | `python3 tests/test_api_docs_routing.py` | Exit 0. | Existing documentation routing suite; no C/H code changed. |
| Same branch and host, site packages disabled | `python3 -S scripts/api_docs_workflow_reader.py` | Exit 1 with pinned-dependency diagnostic. | Expected fail-closed setup error, not a workflow policy decision. |
| Same branch, isolated PyYAML 6.0.3 venv | `tests/test_api_docs_workflow_reader.py` | Exit 0; 20 reader tests passed. | Extraction and source/shell fidelity only; production guard not switched. |
| Same branch, isolated PyYAML 6.0.3 venv | `tests/test_api_docs_workflow_policy_inputs.py` | Exit 0; seven adapter tests passed. | Real-workflow inventory, canonical path entries, dynamic values, provenance, and staging boundary. |
| Same branch and venv | One-pass 190-function legacy fixture sweep | Five diagnostic-stage mismatches, no changed allow/reject verdicts; targeted rerun of all five passed after fixture repair/expectation update. | Function-level comparison only; variant-level differential remains Day 9. |
| Same branch and venv | `make api-docs-local-only API_DOCS_PYTHON=<venv-python>` | Exit 0; Doxygen coverage, direct guard, and all 190 fixture functions passed. | Local-only generated HTML remains ignored. Hosted CI has not yet run on this change. |
| Same branch and local host | `python3 tests/test_api_docs_routing.py` | Exit 0. | Routing regression after the Day 6 guard handoff. |
| Same branch and pinned venv | `tests/test_api_docs_workflow_policy.py` | Exit 0; 11 focused Day 7 tests passed. | Direct paths/actions; shell staging awaits Day 8. |
| Same branch and pinned venv | One-pass frozen guard sweep | 190 functions, zero mismatches. | Existing guard classifications and diagnostics under combined policy. |
| Same branch and pinned venv | `make api-docs-local-only API_DOCS_PYTHON=<venv-python>` | Exit 0; Doxygen coverage, direct guard, and all 190 fixture functions passed. | Maintained local-only gate; no hosted CI result claimed. |
| Same branch and local host | Reader, policy-input, and routing test suites | Reader 20, adapter 7, and routing tests passed. | Regression after Day 7 wiring. |
| Same branch, pinned PyYAML 6.0.3 venv | Day 8 focused structured policy suite | Exit 0; 16 tests passed, including staging chains, POSIX/PowerShell continuations, matched upload/Pages/release paths, and unrelated-output controls. | Local policy and full-guard fixtures; hosted CI remains unverified. |
| Same branch and venv | `make api-docs-local-only API_DOCS_PYTHON=<venv-python>` | Exit 0 after final Day 8 parser changes; Doxygen coverage, maintained-workflow guard, and 190 historical fixture functions passed. | Generated HTML remains ignored and local. |
| Same branch and local host | Reader, adapter, routing, `bash -n`, `git diff --check` | Passed; reader 20 and adapter 7 tests. | Supporting regressions and syntax/whitespace checks. |
| Same branch and pinned venv | Full frozen differential runner | Exit 0; 225 variant rows, zero fixture errors or old/new workflow verdict differences. | Baseline pinned to `05dba39...`; 6 non-workflow cases compare legacy/current only. |
| Same branch and pinned venv | Day 3 YAML edge differential | Exit 0; 10 rows, 3 legacy missed bypasses, zero structured/current mismatches. | Initial discrepancy ledger; not a claim of exhaustive mutation coverage. |
| Same branch and pinned venv | Focused policy and differential harness suites | Exit 0; 17 and 3 tests passed respectively. | Day 9 edge matrix and reproducible report smoke coverage. |
| Same branch and local host | Real-workflow local-only guard and routing suite | Exit 0; 110 structured job/step records checked; routing suite passed. | Current checkout only, not hosted CI. |
| Same branch and pinned venv, Day 10 | Full frozen and YAML-edge differential reruns | Exit 0; 225 and 10 rows, respectively, each byte-identical to its Day 9 JSONL report. | Frozen corpus only; full rerun used temporary output, not a new baseline. |
| Same branch and pinned venv, Day 10 | Reader, policy-input, structured-policy, and differential-harness suites | Exit 0; 20, 7, 17, and 4 tests passed. | Differential harness now gates unexpected verdict and edge-stage labels; Doxyfile mutation test passes. |
| Same branch and local host, Day 10 | Independent workflow step count, direct local-only guard, and routing suite | Exit 0; 3 workflows, 21 jobs, 110 steps, 9 upload actions; guard and routing passed. | `cmd` steps inspected; local evidence only, not hosted CI. |
| Same branch and pinned venv, Day 11 | `tests/test_api_docs_workflow_mutations.py` | Exit 0; 3 methods, 13 matrix subcases and one isolated legacy false-positive probe. | Matrix checks both structured and combined guards; hosted execution awaits CI. |
| Same branch and pinned venv, Day 11 | `make api-docs-local-only API_DOCS_PYTHON=<venv-python>` | Exit 0; Doxygen coverage 18/18/18, current-workflow guard 110 records, frozen fixture command, and new three-method mutation suite passed. | Integrated local route only; generated HTML remains ignored. |
| Same branch and local host, Day 11 | `tests/test_api_docs_routing.py` | Exit 0. | Documentation routing regression after Make/CI mutation-suite wiring. |
| Same branch and pinned venv, Day 12 | Reader, structured-policy, and mutation suites | Exit 0; 20, 17, and 3 tests passed respectively. | Focused parser/policy/fixture coverage; does not replace Doxygen. |
| Same branch and pinned venv, Day 12 | Direct reader and policy CLIs | Exit 0; 3 workflows parsed and 110 job/step records checked. | Confirms the documented direct reproduction route on maintained workflows. |
| Same branch and local host, Day 12 | Direct combined local-only guard and routing suite | Exit 0; Git-state, Doxyfile, wording, workflow, and routing checks passed. | Local checkout only; full Make/Doxygen gate last passed on Day 11. |
| Same branch and pinned venv, Day 13 | Frozen 225-variant differential and ten YAML edge fixtures | Exit 0; zero fixture errors, zero current/structured workflow verdict differences; edge stages passed. | Frozen verdict manifest unchanged; PR #244 review later invalidated the apparent legacy false positives caused by a mismatched helper. |
| Same branch and pinned venv, Day 13 | Frozen guard suite and mutation suite | Exit 0; 190 registered functions and three mutation methods passed. | Includes repaired unrelated-checkout allowance. |
| Same branch and local host, Day 13 | `make api-docs-routing API_DOCS_PYTHON=<venv-python>` | Exit 0; Doxygen coverage 18/18/18, 110 current job/step records, guard/frozen/mutation/routing checks passed. | Local only; hosted CI for this change has not run. |
| Same branch, local macOS with Doxygen 1.16.1 and pinned PyYAML 6.0.3 venv, Day 14 | `make api-docs-local-only API_DOCS_PYTHON=<venv-python>` and `make api-docs-routing API_DOCS_PYTHON=<venv-python>` | Both exit 0; Doxygen 18/18/18, guard 110 records, frozen/mutation suites, and routing checks passed. | Local generated HTML only; no hosted pass. |
| Same branch and pinned venv, Day 14 | Reader, adapter, policy, differential-harness suites | Reader 20, adapter 7 after one stale diagnostic assertion fix, policy 17, harness 4 tests passed. | Test-only assertion update; no policy verdict changed. |
| Same branch and pinned venv, Day 14 | Ten-edge and full frozen differential runners | Exit 0; ten edge fixtures and 225 variants, zero fixture errors or current/structured workflow verdict mismatches. | PR #244 review regenerated both reports with the matching historical helper: no frozen legacy discrepancies and three edge missed bypasses. |
| Same branch and local host, Day 14 | Direct guard, `git check-ignore`, Git-state inspection, `bash -n`, `git diff --check` | Passed; generated API output ignored, untracked, unstaged; nine upload actions have no generated API path. | Direct local checks do not establish hosted behavior. |

No C/H files were changed on Days 1-14. `make format && make lint && make
test` is therefore not required for this Python/fixture/documentation work.
The Day 2 and Day 3 matrices record expected semantics rather than merely
repeating the current guard's verdict.
