# Sprint 156 Working Notes

## Goal

Sprint 156 closes Epic 13 by validating the final evidence state, recalibrating
public claims and non-claims, publishing residuals, writing the Epic 13
retrospective, and preparing a next-epic handoff grounded in reviewed evidence
rather than aspiration.

## Starting Evidence

- Sprint 147 established Epic 13 baseline gaps, claim targets, and evidence
  gates.
- Sprint 148 promoted Windows staged test portability work and recorded staged
  blockers.
- Sprint 149 made the Windows install-validation parity decision and promoted
  reviewed CMake install/downstream validation while preserving Windows
  Makefile and `pkg-config` non-claims.
- Sprint 150 expanded the maintained QR corpus family and report evidence.
- Sprint 151 expanded the maintained partial-SVD corpus family and report
  evidence.
- Sprint 152 published generated report freshness policy and selected
  freshness gates.
- Sprint 153 preserved the static-first package contract and deferred
  shared-library ABI support as a product decision.
- Sprint 154 created the first narrow external comparison harness and
  `qr-minnorm` study without broad ecosystem parity claims.
- Sprint 155 aligned the tutorial, selected public headers, and API reference
  entry point while preserving declaration and claim boundaries.

## Item-To-Day Owner Map

| Sprint 156 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Final Evidence Inventory | Days 1-2 | Day 1 inventories source structure and stop conditions; Day 2 builds the final evidence matrix. |
| Item 2: Full Quality Baseline | Days 3-5 | Day 3 designs validation scope; Day 4 runs local baseline; Day 5 validates install/package surfaces. |
| Item 3: Cross-Platform/CI Reconciliation | Day 6 | Day 6 reconciles Linux, macOS, Windows, reviewed, supplemental, staged, local-only, and deferred evidence. |
| Item 4: Claim And Non-Claim Audit | Days 7-10 | Day 7 validates corpus/report evidence; Day 8 validates comparison evidence; Day 9 validates adoption/API evidence; Day 10 audits public claims. |
| Item 5: Residual Queue Publication | Day 11 | Day 11 consolidates residuals with owners, blockers, prerequisites, and promotion gates. |
| Item 6: Epic 13 Retrospective | Day 12 | Day 12 drafts the Epic 13 retrospective from evidence, claims, non-claims, validation, and residuals. |
| Item 7: Final Project Plan Reconciliation | Days 13-14 | Day 13 reconciles plan status; Day 14 finalizes closeout and next-epic handoff. |

## Stop Conditions

- Any public documentation presents the project as an unqualified
  state-of-the-art sparse linear algebra library.
- Narrow QR, partial-SVD, direct-solver, eigensolver, comparison, benchmark, or
  corpus evidence is converted into broad external-library or ecosystem parity.
- Local benchmark, report, sentinel, or comparison rows are treated as portable
  performance proof.
- Static-first package metadata is interpreted as shared-library support,
  dynamic ABI compatibility, runtime-loader compatibility, package-manager
  distribution, or static/shared selector support.
- Windows reviewed CMake-first support is widened into Windows Makefile parity,
  Windows `pkg-config` execution parity, or broad Windows platform parity.
- Generated report rows or generated API HTML are treated as fresh evidence
  without the selected freshness command or publication policy.
- Optional external dependencies, external-service outages, skipped CI lanes,
  staged blockers, or local-only checks are counted as passing support evidence.
- `.c` or public `.h` edits land without `make format && make lint && make
  test`.
- Public header edits land without declaration-preservation evidence.

## Validation Surfaces

- local formatting, lint, and test gates;
- Make install, uninstall, and `pkg-config` package proof;
- CMake install/export and downstream consumer proof;
- Windows reviewed CMake-first CTest and install/downstream proof;
- macOS and Linux reviewed/supplemental package lanes;
- QR and partial-SVD maintained corpus tests and report rows;
- generated report freshness normalization commands;
- external comparison harness and `qr-minnorm` report family;
- README, INSTALL, tutorial, cookbook, solver-selection, benchmark/report,
  maintainer, support-tier, API reference, and selected public-header wording;
- Sprint 147-155 plans, working notes, artifacts, retrospectives, and handoffs.

## Daily Log

### Day 1: Sprint Intake And Closeout Baseline

- Re-read the Sprint 156 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`.
- Confirmed Sprint 156 is the Epic 13 final validation and claim
  recalibration sprint, even though the initiating prompt referenced the stale
  Epic 12 project-plan path.
- Created the Sprint 156 artifact directory.
- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day1-closeout-baseline.md`.
- Inventoried Sprint 147 through Sprint 155 evidence sources:
  - all nine prior Epic 13 sprint directories have a plan, working notes,
    retrospective, and day artifacts;
  - Sprint 152 and Sprint 154 include additional cross-sprint/report study
    artifacts;
  - Sprint 155 includes declaration-preservation scans and the explicit Sprint
    156 handoff.
- Recorded closeout stop conditions for state-of-the-art, parity, ABI,
  package-manager, platform, performance, report freshness, and generated API
  reference claims.
- Identified validation surfaces touched during Epic 13.
- Day 2 handoff: build the final evidence inventory matrix across platform,
  corpus, report, ABI/package, comparison, adoption, and validation surfaces.

### Day 2: Final Evidence Inventory

- Re-read the Day 2 plan tasks from
  `docs/planning/EPIC_13/SPRINT_156/PLAN.md`.
- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day2-evidence-inventory.md`.
- Inventoried the final Epic 13 evidence by deliverable family:
  - Sprint 147 baseline, gap selection, claim targets, and evidence gates;
  - Sprint 148 Windows staged portability and promoted CMake test count;
  - Sprint 149 Windows reviewed CMake install/downstream validation;
  - Sprint 150 QR maintained corpus and generated-local report rows;
  - Sprint 151 partial-SVD maintained corpus and generated-local report rows;
  - Sprint 152 selected oracle generated report freshness policy;
  - Sprint 153 static-first package contract and shared-library ABI deferral;
  - Sprint 154 first narrow `qr-minnorm` comparison study;
  - Sprint 155 adoption/API/header documentation coherence.
- Classified evidence as reviewed, supplemental, local, generated-local,
  source-controlled, staged, or deferred so later claim audit work can avoid
  treating all evidence as equivalent.
- Recorded missing or stale evidence for Day 3 through Day 10 follow-up:
  hosted CI reconciliation, final local quality baseline, package validation,
  corpus/report freshness checks, comparison freshness, adoption/API scan,
  generated API HTML refresh deferral, remaining public-header cleanup, and
  final public claim audit.
- Day 3 handoff: turn the evidence inventory into a concrete validation matrix
  with commands, required gates, skip/defer semantics, and escalation rules.

### Day 3: Validation Matrix Design

- Re-read the Sprint 147 Day 12 quality surface map and the current Makefile,
  install, CMake, report-index, corpus, and comparison validation targets.
- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day3-validation-matrix.md`.
- Defined the final validation matrix by surface:
  - documentation-only closeout changes require whitespace and claim-evidence
    checks;
  - `.c` or public `.h` changes require `make format && make lint && make
    test`;
  - strongest local code/build baseline is `make quality-review-full`;
  - static-first package proof uses `bash tests/test_install.sh`,
    `bash tests/test_cmake_install.sh`, and
    `bash scripts/static_package_deferral_check.sh`;
  - corpus/report proof uses schema validation, selected oracle generation,
    and `make report-index-oracle-freshness`;
  - comparison proof uses `python3 scripts/run_external_comparison.py
    --self-check` and `make report-index-comparison-freshness`;
  - platform proof remains a hosted CI reconciliation task, not something local
    commands can substitute for.
- Recorded skip/defer semantics for unavailable hosted CI, optional external
  dependencies, generated API HTML refresh, staged Windows lanes, and
  source-controlled report rows.
- Day 4 handoff: run the selected local baseline from the matrix and record
  exact command results, environment details, failures, skips, and deferrals.

### Day 4: Full Local Quality Baseline

- Created
  `docs/planning/EPIC_13/SPRINT_156/artifacts/day4-local-baseline.md`.
- Ran Day 4 intake checks:
  - `git status --short --branch`;
  - `git diff --check`;
  - `git diff --name-only master...HEAD`;
  - local environment capture with `uname`, `cc --version`,
    `cmake --version`, and `make --version`.
- Confirmed the Sprint 156 branch delta before Day 4 artifact edits was
  documentation-only under `docs/planning/EPIC_13/SPRINT_156/`.
- Ran the strongest local reviewed baseline:
  - `make quality-review-full`.
- Result: passed.
  - Makefile reviewed path passed: format-check, lint, full `make test`, and
    deadcode-check.
  - CMake reviewed path passed: configure, clean serial build, `ctest -N`,
    Makefile/CMake test-count parity, and full CTest execution.
  - CMake registered `59` tests, matching the Makefile test count.
  - CTest passed `59/59` tests with `0` failures.
- Recorded expected/known limitations:
  - evidence is local macOS evidence only, not hosted Linux/macOS/Windows proof;
  - full C quality gate was not required by the Sprint 156 file delta because
    no `.c` or `.h` files changed, but the stronger local reviewed baseline
    was run for final closeout confidence;
  - CMake build emitted one non-fatal AppleClang warning in
    `tests/test_svd_partial_corpus.c` for `INFINITY` double promotion; the
    reviewed target still passed.
- Day 5 handoff: validate package/install proof with Make install,
  CMake install/export, downstream consumer checks, and static package
  deferral guard.
