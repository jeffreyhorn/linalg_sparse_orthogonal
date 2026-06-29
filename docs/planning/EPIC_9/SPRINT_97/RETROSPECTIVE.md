# Sprint 97 Retrospective

**Sprint:** 97 - Build, Packaging & Cross-Platform Product Convergence Phase 4
**Duration:** 14 days (Days 1-14 landed on this branch)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 97 started from the Epic 9 project-plan section and the live
      post-Sprint-96 build/package/platform surface
- [x] duplicated Make, CMake, workflow, install/export, benchmark, example,
      and package surfaces were audited before build topology changed
- [x] the highest-value convergence target was selected from evidence:
      library-source membership drift between Make and CMake
- [x] a bounded source-list manifest and checker were designed before landing
      implementation changes
- [x] `build-metadata/library_sources.txt` now records the reviewed library
      source list
- [x] `scripts/check_library_sources.py` validates manifest, Makefile
      `LIB_SRCS`, and CMake `add_library(...)` membership and order
- [x] `make source-list-check` is available as a direct reviewed target
- [x] `make quality-review-compile` now runs
      `format-check + source-list-check + lint`
- [x] the package-surface decision was made from live evidence: preserve the
      maintained static-first package contract
- [x] shared-library packaging remains an explicit deferred non-claim
- [x] install/export proof scripts now assert no shared-library artifacts are
      installed
- [x] README, INSTALL, maintainer-guide, and workflow wording were aligned with
      the new proof ownership
- [x] Windows and macOS platform limits remain explicit without claiming fake
      parity
- [x] Sprint 97 validation passed for the touched build, CMake, install/export,
      workflow, and documentation surfaces
- [x] Sprint 97 closed with a ranked Sprint 98 handoff queue

## What Went Well

1. **The sprint ranked duplication before reducing it.**
   Day 1 and Day 2 separated costly build-topology duplication from repeated
   proof assertions. That kept the implementation focused on library-source
   drift instead of flattening every repeated Make, CMake, CI, and install
   surface.

2. **The source-list solution preserved reviewability.**
   Sprint 97 did not hide source membership behind generation. The manifest,
   Makefile, and CMake source list remain directly reviewable, while
   `source-list-check` now turns drift into an explicit failure.

3. **The first reduction landed with a clear proof owner.**
   `source-list-check` is wired into `quality-review-compile`, so Linux and
   macOS reviewed compile-quality workflows inherit the guard through the
   existing target. CMake configure-time enforcement stayed deferred instead
   of adding a new dependency surface.

4. **The package decision stayed evidence-based.**
   Day 7 framed static-first versus shared-library support as a proof question.
   Day 8 preserved the static-first contract and made shared-library packaging
   a deliberate deferred non-claim rather than a vague future possibility.

5. **Install/export proof became sharper.**
   `tests/test_install.sh` and `tests/test_cmake_install.sh` now prove both the
   positive static artifact and the negative shared-artifact claim. That makes
   the package scripts match the public product contract more directly.

6. **Workflow and platform language stayed close to proof.**
   README, INSTALL, maintainer-guide, and workflow comments now agree on the
   reviewed target behavior, expected Windows CTest count, macOS supplemental
   install confidence, and Windows CMake-first consumer scope.

7. **The sprint closed from strong targeted validation.**
   Day 13 ran the direct source-list checker, reviewed Make compile-quality
   wrapper, reviewed CMake compile/parity wrapper, both install/export proof
   scripts, diff hygiene, and whitespace scans.

## What Didn't Go Well

1. **The highest-cost duplication was guarded, not removed.**
   Make and CMake still both list the library sources. This was the right
   first step for reviewability, but it means future work may still consider
   whether generation or shared fragments are worth the added indirection.

2. **Test registration remains a harder problem.**
   The test list is still duplicated across Make and CMake, and Windows still
   owns a smaller CTest subset. Centralizing that safely requires preserving
   platform exclusions and expected-count assertions.

3. **The validation path was expensive for a non-`.c`/`.h` sprint.**
   No implementation/header files changed, but the sprint touched Makefile,
   CMake comments, install scripts, workflow YAML, and support docs. Running
   the reviewed Make/CMake and install/export checks was necessary, but not
   cheap.

4. **Historical wording still exists in older planning artifacts.**
   Sprint 97 removed stale history from active workflow and maintainer
   surfaces, but did not rewrite older planning history. That is intentional,
   but searches still find historical references in archival context.

5. **Platform parity remains intentionally asymmetric.**
   Linux, macOS, and Windows are clearer after Sprint 97, but not equivalent.
   The sprint correctly preserved those differences rather than trying to
   promise unsupported parity.

## Final Metrics

### Validation

| Metric | Sprint 97 close state |
|---|---:|
| source-list checker | `source-list-check: PASS (42 library sources)` |
| reviewed Make compile-quality path | `make quality-review-compile` passed |
| reviewed CMake compile/parity path | `make quality-review-cmake-compile` passed |
| local CMake tests registered | `54` |
| local Makefile tests registered | `54` |
| Make/CMake test-count parity | passed |
| Make install/`pkg-config` proof | `14` passed, `0` failed |
| CMake install/export proof | `16` passed, `0` failed, `0` skipped |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on touched workflow/docs/scripts/build metadata/build files |

No `.c` or `.h` files were modified, so the full
`make format && make lint && make test` chain was not required by the sprint
validation rule. The reviewed compile-quality and CMake compile/parity wrappers
were still run because the branch changed build, script, package, and workflow
surfaces.

### Sprint 97 Artifact Package

| Metric | Sprint 97 close state |
|---|---:|
| total artifact files under `SPRINT_97/artifacts/` | `15` |
| baseline/audit/design artifacts | `4` |
| source-list reduction artifacts | `3` |
| package/consumer/workflow/platform artifacts | `5` |
| validation/closeout artifacts | `3` |

Notes:

- baseline/audit/design artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-build-topology-baseline.md`
  - `day2-build-topology-duplication-audit.md`
  - `day3-convergence-architecture-design.md`
- source-list reduction artifacts:
  - `day4-source-list-boundary-freeze.md`
  - `day5-source-list-reduction-batch1.md`
  - `day6-source-list-reduction-closeout.md`
- package/consumer/workflow/platform artifacts:
  - `day7-package-surface-decision-audit.md`
  - `day8-package-surface-decision.md`
  - `day9-consumer-proof-follow-through.md`
  - `day10-workflow-coherence-follow-through.md`
  - `day11-cross-platform-calibration.md`
- validation/closeout artifacts:
  - `day12-cross-platform-product-follow-through.md`
  - `day13-validation-and-residual-queue.md`
  - `day14-sprint-closeout-and-handoff.md`

### Landed Convergence Package

| Metric | Sprint 97 close state |
|---|---:|
| new build metadata files | `1` |
| new checker scripts | `1` |
| Makefile reviewed targets added | `1` |
| install/export proof scripts strengthened | `2` |
| active workflow files touched | `1` |
| public/support docs touched | `4` |

Notes:

- build metadata:
  - `build-metadata/library_sources.txt`
- checker script:
  - `scripts/check_library_sources.py`
- Makefile target:
  - `source-list-check`
- proof scripts:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- workflow:
  - `.github/workflows/windows-ci.yml`
- public/support docs:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `CMakeLists.txt` package-surface comment

## Residual Deferred Debt

Sprint 97 deliberately stopped after the first source-list guard, static-first
package decision, consumer-proof follow-through, workflow coherence pass, and
platform calibration.

Most important carry-forward work:

- test registration convergence or stronger parity guard that preserves
  platform exclusions and Windows expected-count assertions
- benchmark registration reduction only if the manifest/checker pattern stays
  simple and keeps benchmark subsets visible
- example registration reduction only if CMake target names stay clear
- optional checker self-test fixture if `scripts/check_library_sources.py`
  grows beyond the current narrow parser
- possible Windows CMake install-validation lane only after explicit design and
  CI cost review

Still consciously constrained rather than silently solved:

- no shared-library package contract
- no dynamic ABI guarantee
- no package-manager claim
- no reviewed Windows install-validation lane
- no Windows Makefile parity
- no Windows DLL/import-library package story
- no full reviewed macOS install/export parity
- no centralization of every repeated CI proof message

Not carried forward as unresolved Sprint 97 debt:

- build-topology duplication audit
- library source-list convergence design
- library source-list manifest/checker implementation
- reviewed Make compile-quality integration for `source-list-check`
- static-first package decision
- install/export proof static-shape assertion
- README/INSTALL/Makefile command wording alignment
- maintainer-guide platform wording calibration
- active Windows workflow stale sprint-history cleanup
- final validation and residual queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-build-topology-baseline.md](./artifacts/day1-build-topology-baseline.md)
- [day2-build-topology-duplication-audit.md](./artifacts/day2-build-topology-duplication-audit.md)
- [day3-convergence-architecture-design.md](./artifacts/day3-convergence-architecture-design.md)
- [day4-source-list-boundary-freeze.md](./artifacts/day4-source-list-boundary-freeze.md)
- [day5-source-list-reduction-batch1.md](./artifacts/day5-source-list-reduction-batch1.md)
- [day6-source-list-reduction-closeout.md](./artifacts/day6-source-list-reduction-closeout.md)
- [day7-package-surface-decision-audit.md](./artifacts/day7-package-surface-decision-audit.md)
- [day8-package-surface-decision.md](./artifacts/day8-package-surface-decision.md)
- [day9-consumer-proof-follow-through.md](./artifacts/day9-consumer-proof-follow-through.md)
- [day10-workflow-coherence-follow-through.md](./artifacts/day10-workflow-coherence-follow-through.md)
- [day11-cross-platform-calibration.md](./artifacts/day11-cross-platform-calibration.md)
- [day12-cross-platform-product-follow-through.md](./artifacts/day12-cross-platform-product-follow-through.md)
- [day13-validation-and-residual-queue.md](./artifacts/day13-validation-and-residual-queue.md)
- [day14-sprint-closeout-and-handoff.md](./artifacts/day14-sprint-closeout-and-handoff.md)

## Bottom Line

Sprint 97 achieved its goal:

- the highest-value build-topology duplication now has an enforced drift guard
- source membership remains reviewable in both Make and CMake
- the static-first package contract is clearer and better proven
- install/export scripts now fail if shared-library artifacts appear
- workflow and command descriptions match the current reviewed target behavior
- Linux, macOS, and Windows platform claims are calibrated without fake parity
- the branch validates cleanly under the strongest required checks for touched
  surfaces
- Sprint 98 receives a ranked handoff queue instead of a broad
  build/package/platform backlog
