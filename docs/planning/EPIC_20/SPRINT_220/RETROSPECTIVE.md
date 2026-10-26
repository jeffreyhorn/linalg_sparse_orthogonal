# Sprint 220 Retrospective

**Sprint:** 220 - Structured Local-Only Workflow Guard

**Duration:** 14 planned days on branch `sprint-220`

**Status:** Six planned deliverables closed locally; hosted validation pending

## Source And Definition Of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) and [sprint plan](PLAN.md)
allocate 140 hours across six items. The [working notes](WORKING_NOTES.md) and
[Day 14 closeout](artifacts/day14-integrated-gate-and-closeout.md) retain the
evidence and limits. Hours are estimates, not measured effort. The intended
result is a structured, fail-closed check that keeps generated API HTML local
while preserving the existing Git-state, Doxyfile, and documentation checks.
The local deliverables are complete; the changed hosted Linux and Apple-Clang
jobs had not run at closeout and are not counted as passing evidence.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 220.1 Fixture Freeze | 22 h | Closed locally: classified existing cases and froze YAML edge expectations before replacing extraction. |
| 220.2 Structured Parser | 34 h | Closed locally: pinned PyYAML reader preserves job/step boundaries, source locations, scalar rendering, aliases, and effective shell. |
| 220.3 Policy Engine | 28 h | Closed locally for the declared action, path, POSIX, and PowerShell surfaces; unknown shell forms and unclassified actions remain explicit limits. |
| 220.4 Differential Regression | 20 h | Closed locally: frozen variant comparison, edge matrix, and mutation gate distinguish intended changes from regressions. |
| 220.5 Maintainer Guide | 18 h | Closed locally: dependency, direct reproduction, supported forms, diagnostics, and coverage limits documented. |
| 220.6 Gate And Rollout | 18 h | Local gate and CI wiring closed; hosted results and final rollout acceptance pending. |
| **Total** | **140 h** | **Local deliverables complete; hosted result not yet established.** |

## What Went Well

1. **The fixture oracle preceded the implementation.** The baseline classified
   accepted cases as carefully as rejections, including folded shell comments,
   PowerShell paths, flow collections, aliases, duplicate keys, tags, and
   dynamic publication fields. The 225-variant differential report preserved
   verdicts per variant rather than collapsing mixed cases into one test name.
2. **The parser kept workflow structure intact.** The pinned PyYAML reader
   produces source-located job and step records, renders YAML scalars before
   shell analysis, and carries effective shell identity. Unsupported or
   ambiguous publication-relevant syntax rejects with a useful location.
3. **Policy moved to the right boundary.** Action and path checks now use
   structured fields; staging checks connect commands to consumers within the
   relevant job and step. Removing the legacy whole-workflow text scan fixed
   the unrelated `checkout path: docs/` false positive without weakening the
   frozen rejection cases.
4. **Differential evidence exposed both old failures and new behavior.** The
   historical guard had three missed YAML-edge bypasses and 43 false-positive
   verdicts against the frozen oracle. The final current and structured
   workflow verdicts agreed across all 225 variants, with zero fixture errors.
   The ten-edge matrix and mutation gate remained green after retirement.
5. **The product decision stayed narrow.** Local Doxygen output remained
   ignored, untracked, and unstaged. All nine maintained upload-artifact steps
   were inspected, and none selected generated API HTML. No Pages, release, or
   retained-HTML route was introduced.

## What Didn't Go Well

1. **The legacy guard was difficult to retire safely.** Its YAML text
   normalization and whole-file matching coupled unrelated steps. Earlier
   fixture comparisons found diagnostic-stage differences, and Day 13 needed
   a targeted benign checkout regression before removing the superseded scan.
2. **One test still expected a retired diagnostic wrapper.** The first Day 14
   adapter run failed on its expected message even though the guard correctly
   reported a source-located duplicate `run` key. The assertion was updated;
   adapter, edge, and full differential suites then passed. No policy verdict
   was changed to satisfy the test.
3. **Hosted results were not available at closeout.** Linux and Apple-Clang
   jobs now install the pinned dependency and call the focused suites, but an
   unpushed branch cannot provide a hosted pass. The local evidence cannot
   close that acceptance gate.
4. **The guard is not a general shell or action analyzer.** Maintained `cmd`
   steps were inspected, but `cmd` and unknown-shell staging lack general
   command coverage. A syntactically valid third-party action not classified
   by policy also needs human input review before it can be considered safe.

## Final Metrics And Claim

| Metric | Sprint 220 local state |
| --- | --- |
| Planned effort | 140 h across six items and 14 days; actual hours not tracked |
| Daily evidence | 14 primary day artifacts, working notes, this retrospective |
| Generated API docs | 18 public headers, 18 reference pages, 18 source pages; output ignored and unstaged |
| Maintained workflows | Three workflows, 21 jobs, 110 job/step records, nine upload-artifact steps |
| Focused suites | Reader 20, adapter 7, policy 17, differential harness 4 tests passed |
| Frozen and mutation gates | 190 registered guard functions and three mutation methods passed |
| Differential | 225 variants and ten YAML edge fixtures passed; zero fixture errors or current/structured workflow verdict differences |
| Integrated local route | `make api-docs-local-only` and `make api-docs-routing` passed with pinned PyYAML environment |
| Hosted acceptance | No changed-branch Linux or Apple-Clang run at Day 14 closeout; pending |

The earned claim is a **locally validated structured guard for the declared
workflow publication surfaces**, with generated API HTML still local-only.
It is not a claim of exhaustive shell or third-party-action analysis, a hosted
CI pass, or approval to publish generated HTML. No `.c` or `.h` files changed;
the C `make format`, `make lint`, and `make test` gates were not applicable to
this Python, shell, workflow, fixture, and documentation change set.

## Daily Evidence

- [Day 1 guard and workflow inventory](artifacts/day1-guard-and-workflow-inventory.md)
- [Day 2 classification and baseline](artifacts/day2-classification-and-baseline.md)
- [Day 3 parser contract and edge fixtures](artifacts/day3-parser-and-extraction-contract.md)
- [Day 4 structured reader](artifacts/day4-structured-reader.md)
- [Day 5 scalar, alias, and shell evidence](artifacts/day5-scalar-alias-shell-evidence.md)
- [Day 6 parser-policy handoff](artifacts/day6-parser-policy-handoff.md)
- [Day 7 direct publication policy](artifacts/day7-direct-publication-policy.md)
- [Day 8 shell staging policy](artifacts/day8-shell-staging-policy.md)
- [Day 9 policy differential ledger](artifacts/day9-policy-differential-ledger.md)
- [Day 10 real-workflow validation](artifacts/day10-real-workflow-validation.md)
- [Day 11 mutation gate and discrepancy ledger](artifacts/day11-mutation-gate-and-discrepancy-ledger.md)
- [Day 12 maintainer guide and diagnostics](artifacts/day12-maintainer-guide-and-diagnostics.md)
- [Day 13 rollout and retirement](artifacts/day13-rollout-and-retirement.md)
- [Day 14 integrated gate and closeout](artifacts/day14-integrated-gate-and-closeout.md)

## Residual Handoff

After pushing, record the exact commit SHA, hosted Linux and Apple-Clang job
URLs, failed or skipped steps, and rerun conclusions. Do not convert local
passes into a hosted claim. For any new `cmd`, unknown-shell, or third-party
publication path, add both rejecting and benign fixtures and review the action
inputs before relying on the guard. Keep generated HTML ignored and local;
any hosted or retained route requires a separate explicit product decision
covering hosting, freshness, retention, routing, rollback, and security.
