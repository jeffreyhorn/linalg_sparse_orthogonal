# Sprint 147 Day 14 Closeout And Windows Handoff

## Purpose

Day 14 closes Sprint 147 and packages the evidence that Sprints 148-156 should
use without reopening baseline decisions. The closeout confirms that Sprint 147
selected Epic 13 closure gaps, froze the public claim baseline, defined
evidence gates, and prepared the Sprint 148 Windows staged-test portability
handoff.

## Sprint 147 Deliverable Status

| Deliverable | Status | Evidence |
| --- | --- | --- |
| Epic 13 selected-gap register | Complete | `artifacts/day5-selected-gap-register.md` selects Windows, QR, partial-SVD, report freshness, ABI/package, external comparison, adoption, and final validation work while deferring lower-priority residuals. |
| Claim target and non-goal register | Complete | `artifacts/day6-claim-target-register.md` defines candidate earned claims C1-C9 plus rejected/deferred broad claims. |
| Evidence gate templates | Complete | Days 7-11 define Windows, corpus, generated freshness, ABI/package, and external comparison gates. |
| Quality surface map | Complete | `artifacts/day12-quality-surface-map.md` maps touched surfaces to required validation and stop conditions. |
| Public claim freeze audit | Complete | `artifacts/day13-public-claim-freeze-audit.md` records no unsupported public wording fix was needed and freezes the bounded claim baseline. |
| Sprint 148 Windows portability handoff | Complete | This artifact publishes the Sprint 148 prerequisite checklist and links it back to the Day 7 Windows evidence gate. |

No Sprint 147 implementation item remains partially complete. Lower-priority
Epic 12 residuals remain explicit non-goals or deferred items rather than
unfinished Sprint 147 work.

## Artifact Index And Handoff Map

| Day | Artifact | Primary output | Consumed by |
| --- | --- | --- | --- |
| 1 | `day1-baseline-intake.md` | Baseline evidence schema, stop conditions, and Sprint 148 handoff seed. | All later Sprint 147 artifacts. |
| 2 | `day2-technical-baseline.md` | Source/test/build/package/CI baseline and Windows CTest snapshot. | Sprint 148 and Sprint 149. |
| 3 | `day3-corpus-report-evidence-baseline.md` | Corpus, oracle, report-family, and freshness baseline. | Sprints 150-152 and Sprint 154. |
| 4 | `day4-epic12-residual-intake.md` | Epic 12 residual intake with owner surfaces and dispositions. | Day 5 selection and Sprint 156 residual closeout. |
| 5 | `day5-selected-gap-register.md` | Selected Epic 13 gaps and deferred non-goals. | Sprints 148-156 sequencing. |
| 6 | `day6-claim-target-register.md` | Candidate earned claims, evidence gates, and rejected broad claims. | Public/support docs and final claim recalibration. |
| 7 | `day7-windows-evidence-gate.md` | Windows staged-test and install-validation evidence gates. | Sprints 148-149. |
| 8 | `day8-corpus-family-evidence-gate.md` | QR and partial-SVD corpus-family row, proof-owner, oracle, and comparison semantics. | Sprints 150-151. |
| 9 | `day9-generated-report-freshness-gate.md` | Required-generated versus advisory freshness policy. | Sprint 152 and Sprint 156. |
| 10 | `day10-abi-package-evidence-gate.md` | Static-first package and shared-library ABI decision gate. | Sprint 153. |
| 11 | `day11-external-comparison-evidence-gate.md` | First narrow external comparison target shape and wording boundary. | Sprint 154. |
| 12 | `day12-quality-surface-map.md` | Touched-surface validation map and Sprint 156 validation-package seed. | Every implementation sprint. |
| 13 | `day13-public-claim-freeze-audit.md` | Public claim freeze and wording baseline. | Sprints 148-156 docs and claim updates. |
| 14 | `day14-closeout-and-windows-handoff.md` | Sprint closeout, Windows prerequisites, validation summary, and retrospective notes. | Sprint 148 and Sprint 147 retrospective. |

## Sprint 148 Windows Prerequisite Checklist

Sprint 148 starts with Windows staged-test portability closure. It should not
begin by simply changing the expected Windows CTest count. The count can change
only after a reviewed promotion, replacement, or explicit removal decision is
recorded.

Required starting facts:

- reviewed Windows workflow: `.github/workflows/windows-ci.yml`
- reviewed job: `Windows enforced reviewed CMake consumer subset (MSVC)`
- runner: `windows-2022`
- generator: `Visual Studio 17 2022`
- architecture: `x64`
- build type: `Release`
- current enforced registered CTest count: `EXPECTED_WINDOWS_CTEST_COUNT=56`
- staged test surfaces:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`
- current blockers:
  - direct pthread API usage in `test_threads`
  - direct pthread API usage in `test_sprint4_integration`
  - POSIX temp-file assumptions in `test_fuzz`

Required Sprint 148 intake actions:

1. Audit each staged test source and CMake gate before editing.
2. Choose per-test disposition: direct port, Windows-native equivalent,
   platform-specific proof split, retained staged status, or explicit removal
   from the promotion target.
3. Preserve Linux/macOS/POSIX proof while adding any Windows-compatible path.
4. Update CMake registration only after the source can configure, build, and
   execute on MSVC.
5. Update `EXPECTED_WINDOWS_CTEST_COUNT` only with a before/after CTest
   enumeration and explanation.
6. Record hosted Windows run ID, commit SHA, job name, conclusion, CTest count,
   and promoted test list before making reviewed Windows coverage claims.
7. Keep Windows install-validation parity out of Sprint 148 except as a handoff
   dependency for Sprint 149.
8. Run `make format && make lint && make test` for any `.c` or `.h` change.

Sprint 148 must preserve these non-claims:

- no Windows Makefile parity;
- no Windows `pkg-config` parity;
- no separate reviewed Windows install-validation parity until Sprint 149
  promotes or rejects it;
- no shared-library support;
- no dynamic ABI compatibility;
- no runtime-loader compatibility;
- no package-manager distribution;
- no broad Windows platform parity.

## Final Selected-Gap Index

| Sprint | Selected gap | Closure target |
| --- | --- | --- |
| 148 | Windows staged test portability | Promote or replace staged pthread/POSIX test surfaces in the reviewed Windows CMake lane. |
| 149 | Windows install-validation parity decision | Promote, keep supplemental, or reject reviewed Windows CMake install/downstream parity without implying Unix Make/`pkg-config` parity. |
| 150 | QR maintained corpus expansion | Add broader but bounded QR fixture families with metadata, expected rows, proof-owner tests, oracle/report rows, and narrow wording. |
| 151 | Partial-SVD maintained corpus expansion | Add broader but bounded partial-SVD fixture families with subspace-safe semantics and fail-closed evidence. |
| 152 | Generated report freshness publication | Require generated freshness only for selected claim-bearing families. |
| 153 | Shared-library ABI/static-first decision | Implement shared support with proof or strengthen the static-first deferral. |
| 154 | External comparison harness and narrow study | Add one named dependency/version/fixture/metric comparison without broad ecosystem parity wording. |
| 155 | Adoption-surface simplification | Improve front-door docs, tutorial/header handoffs, and support-surface navigation without widening claims. |
| 156 | Final validation and closeout | Reconcile local validation, hosted CI, reports, docs, residuals, and final claims. |

## Residuals And Non-Goals

The following remain explicit non-goals at Sprint 147 close:

| Area | Disposition |
| --- | --- |
| Broad state-of-the-art sparse linear algebra status | Rejected without direct comparative evidence across named libraries, versions, fixture sets, metrics, platforms, and tolerances. |
| Broad external-library parity | Rejected; Sprint 154 may earn only one narrow comparison study. |
| Runtime/backend typed-control promotion outside selected work | Deferred unless a later sprint swaps it into scope with API, ABI, tests, docs, and package review. |
| Additional standalone sentinel expansion | Deferred unless tied to a selected report/freshness or performance claim. |
| Package-manager distribution | Deferred behind ABI/release mechanics, recipe ownership, and install proof. |
| Shared-library support | Deferred unless Sprint 153 explicitly implements and validates it. |

## Validation Summary

Sprint 147 was planning/documentation work only. No `.c` or `.h` files were
modified by the sprint artifacts, so the full C gate is not required for Day 14.

Required lightweight validation:

```sh
git diff --check -- docs/planning/EPIC_13/SPRINT_147
rg -n "[[:blank:]]$" docs/planning/EPIC_13/SPRINT_147 || true
test -f docs/planning/EPIC_13/SPRINT_147/artifacts/day14-closeout-and-windows-handoff.md
git diff --name-only -- '*.c' '*.h'
```

Completion evidence should record that whitespace checks passed and the C/H
diff check returned no files.

## Retrospective Input Notes

What worked:

- Sprint 147 converted a broad Epic 13 review into selected closure tracks
  with bounded evidence gates.
- Windows work was split into staged-test portability and install-validation
  parity so one support claim cannot mask the other.
- Corpus, generated freshness, package/ABI, and external comparison work now
  have explicit proof requirements before public wording can widen.
- The public claim freeze found no immediate unsupported wording fix, which
  gives implementation sprints a stable baseline.

Risks to carry forward:

- Hosted Windows evidence is mandatory for Sprint 148, so local checks alone
  cannot close the sprint.
- CTest expected-count drift has caused prior CI failures; Sprint 148 must
  treat the count as a reviewed policy value, not a mechanical bump.
- Corpus and generated report rows can look like evidence before executable
  proof runs; Sprints 150-152 must keep metadata and pass evidence separate.
- Shared-library or package-manager wording can easily overstate the current
  static-first product contract.

Recommended retrospective framing:

- Sprint 147 was successful if it prevented unsupported implementation-epoch
  claims and made future validation choices mechanical.
- The main follow-through risk is evidence discipline under implementation
  pressure, especially on Windows, generated reports, and ABI/package wording.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 147 deliverables are complete. | Complete | All Day 1-14 artifacts exist and map to the Sprint 147 project-plan deliverables. |
| Sprint 148 can begin without reopening baseline decisions. | Complete | The Windows prerequisite checklist names staged tests, blockers, current count, promotion rules, hosted evidence, and non-claims. |
| Documentation validation passes. | Complete | `git diff --check`, trailing-whitespace scan, artifact existence check, and C/H diff check passed. |
| Residuals and non-goals remain explicit. | Complete | The residual/non-goal table preserves rejected and deferred claims. |
