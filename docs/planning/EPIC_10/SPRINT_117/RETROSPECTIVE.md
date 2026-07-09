# Sprint 117 Retrospective

**Sprint:** 117 - Final Integration, Competitive Calibration & Epic 10 Closeout
**Duration:** 14 days
**Status:** Complete

## Definition of Done Checklist

- [x] Created Sprint 117 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read the Sprint 100 state-of-the-art target and evidence contract.
- [x] Inventoried Sprint 100-116 closeout evidence and mapped Sprint 117
      project-plan items to day-level owners.
- [x] Built the end-state claim inventory against Sprint 100 target claims.
- [x] Decided earned, bounded, deferred, and explicit non-claim categories
      before public cleanup.
- [x] Designed the final reviewed and supplemental validation lanes for touched
      surfaces.
- [x] Ran documentation hygiene and the strongest local reviewed baseline:
      `make quality-review-full`.
- [x] Packaged the final validation evidence, changed-surface proof, skipped
      supplemental lane rationale, and validation residual risk.
- [x] Inventoried final comparison surfaces across solver, reorder, benchmark,
      coverage/dead-code, package/platform, adoption, and validation evidence.
- [x] Classified comparison artifacts as public claim evidence, local
      measurement context, supplemental proof, or residual background.
- [x] Rechecked public/support docs and recorded that no unsupported-claim edit
      was required.
- [x] Published the residual queue intake from Sprint 114, Sprint 115, and
      Sprint 116 deferred debt.
- [x] Published the post-Epic residual queue, future-epic candidates, optional
      scanability work, explicit non-claim register, and consciously closed
      prior work.
- [x] Finalized this retrospective.
- [x] Ran focused documentation hygiene after finalizing the retrospective.
- [x] Prepared the Epic 10 retrospective source inventory for Day 13.

## What Went Well

1. **Sprint 117 kept closeout work evidence-driven.**
   The sprint began with the Sprint 100 target and claim contract, then used
   Sprint 114-116 residual decisions as guardrails. That prevented broad
   state-of-the-art, ecosystem parity, package/platform, ABI, or performance
   claims from being promoted without evidence.

2. **Validation was stronger than the touched-surface minimum.**
   Sprint 117 changed planning documentation only, but Day 5 still ran
   `make quality-review-full`. The Makefile reviewed path, CMake reviewed
   parity path, CTest registration parity, and full CTest run all passed.

3. **Comparison evidence was classified before claim cleanup.**
   Day 7 separated public claim evidence from local measurement context,
   supplemental proof, and residual background. That made the Day 8 public
   wording pass mechanical rather than speculative.

4. **The public/support cleanup pass produced an explicit no-edit record.**
   Day 8 checked README, INSTALL, public docs, benchmark docs, examples, and
   maintainer-facing interpretation. The existing wording already fenced local
   benchmarks, static-first packaging, tiered platforms, Matrix Market public
   surface, and ecosystem parity non-claims.

5. **Residuals were published without reopening implementation scope.**
   Day 9 classified every Sprint 114-116 residual, and Day 10 published a
   usable post-Epic queue. No source movement, package/platform support,
   public API, ABI, package-manager, benchmark, workflow, helper-target, or
   CTest claim was silently widened.

6. **Skipped supplemental lanes were documented rather than hidden.**
   Package/install, benchmark, sanitizer, coverage, and guardrail regeneration
   were not rerun because their surfaces were untouched. The validation package
   records that those lanes are not fresh Sprint 117 proof.

## What Did Not Go Well

1. **Sprint 117 is documentation-heavy by necessity.**
   Closing an epic with accurate claims requires a large amount of artifact
   synthesis. The output is useful, but future closeout sprints may benefit
   from an earlier running residual register to reduce final-week synthesis.

2. **Benchmark and coverage freshness remain intentionally limited.**
   Day 7 found no fresh benchmark report or coverage output to cite. This is
   correct for the touched surface, but future performance or coverage claims
   still need their own regeneration lanes.

3. **Platform proof remains split between local validation and CI ownership.**
   Local `make quality-review-full` passed, but Linux/macOS/Windows workflow
   proof remains CI-owned. Sprint 117 kept the distinction explicit rather than
   claiming symmetric platform parity.

4. **Several residuals still require future product decisions.**
   Shared-library ABI, package-manager support, Windows install validation,
   macOS install/export parity, Linux install CI promotion, eigensolver source
   movement, shared direct/iterative oracle ownership, and SVD proof-helper
   ownership remain future work.

## Final Metrics

### Validation

| Metric | Sprint 117 close state |
|---|---:|
| documentation hygiene | `git diff --check` passed on Days 5, 7, 8, 9, 10, 11, and 12 |
| trailing-whitespace scan | passed on Days 5, 7, 8, 9, 10, 11, and 12 |
| strongest local reviewed baseline | `make quality-review-full` passed on Day 5 |
| Makefile reviewed path | passed: `format-check`, `lint`, `test`, `deadcode-check` |
| CMake reviewed parity path | passed: configure, clean build, `ctest -N`, count parity, full CTest |
| CMake registered tests | `54` |
| Makefile/CMake test-count parity | `54` vs `54` |
| CTest result | `54 / 54` passed |
| CTest failures | `0` |
| CTest real time | `242.37 sec` |
| changed `.c` files | `0` |
| changed `.h` files | `0` |
| changed Make/CMake/workflow/package/script files | `0` |
| changed benchmark/source/test/include files | `0` |

### Sprint Artifact Package

| Metric | Sprint 117 close state |
|---|---:|
| artifact files under `SPRINT_117/artifacts/` | `14` |
| artifact lines | `1795` |
| working notes lines | `405` |
| plan lines | `434` |
| retrospective files | `1` |
| retrospective lines | `212` |

## Claim And Comparison Outcomes

| Area | Outcome |
|---|---|
| Product maturity | Evidence supports bounded Epic 10 productization, validation, docs, support-tier, and claim-boundary progress, not an unqualified replacement claim. |
| Direct solvers | Selected Cholesky CSC, LDLT CSC, and linked-list LU evidence remains bounded to named external dense-reference lanes. |
| Iterative/eigensolver/SVD | Fixture-local residual, convergence, reconstruction, rank, and orthogonality evidence remains bounded; no broad ARPACK/LAPACK/SciPy/PETSc/Trilinos parity claim. |
| Benchmarks/performance | Benchmark and sentinel artifacts remain local measurement context, not portable performance proof. |
| Reorder/fill | Named fixture and report-contract evidence remains bounded; no universal reorder/fill superiority claim. |
| Package/platform | Static-first package support and tiered platform support remain the maintained product truth. |
| Matrix Market | Public support remains load/save functions with documented format boundaries, not a separate public Matrix I/O module or builder API. |
| Maintainability/source ownership | Touched-owner progress is documented; source-boundary and proof-owner residuals remain explicit. |

## Residual Deferred Debt

Most important carry-forward work:

- Move one eigensolver private owner only with exact old/new files,
  source-list and CMake updates, focused consumer proof, reviewed CTest count
  evidence where applicable, and rollback instructions.
- Revisit `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert
  setup/conversion, and `lanczos_iterate_op` movement with the dependency
  proof listed in Day 10.
- Decide whether QR, CG, GMRES, BiCGSTAB, and MINRES generated-RHS setup can
  share a direct/iterative oracle.
- Decide whether SVD reconstruction, U/Vt orthogonality, Moore-Penrose,
  low-rank, sparse-vs-dense, and condition-number helpers can share a proof
  owner.
- Promote Linux install proof to reviewed CI only with accepted CI/runtime
  ownership and support wording updates.
- Promote macOS CMake install/export parity only with reviewed CI proof.
- Add Windows install-validation only with MSVC install, downstream consumer,
  reviewed-count, and non-claim proof.
- Port or split Windows thread/fuzz/property proof only with native Windows
  behavior and CTest count updates.
- Add shared-library/dynamic ABI support only with build, package, loader,
  symbol, versioning, ABI-test, and platform ownership proof.
- Add package-manager support only with real recipes and install/consumer
  proof for each claimed manager/platform.
- Consider splitting `docs/algorithm.md` and adding generated benchmark
  artifact indexes as optional scanability work.

Still consciously constrained rather than silently solved:

- no unqualified state-of-the-art replacement claim;
- no broad ecosystem parity claim;
- no every-family external solver validation claim;
- no portable performance superiority claim;
- no universal reorder/fill superiority claim;
- no reviewed Linux install CI lane;
- no full reviewed macOS CMake install/export parity;
- no Windows install-validation, thread/fuzz/property, or Makefile parity;
- no shared-library package support;
- no dynamic ABI compatibility guarantee;
- no package-manager support;
- no public Matrix I/O module or public Matrix Market builder API;
- no proof-owner/internal-helper public contract expansion;
- no Sprint 117 public API, install-header, source-list, helper-target, CTest,
  workflow, package, benchmark, source, test, or implementation change.

Not carried forward as unresolved Sprint 117 debt:

- final integration intake and evidence map;
- end-state claim inventory and decision;
- validation design, execution, and final validation package;
- final comparison inventory and cleanup record;
- residual queue intake and final non-claim publication.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-final-integration-intake.md](./artifacts/day1-final-integration-intake.md)
- [day2-end-state-claim-inventory.md](./artifacts/day2-end-state-claim-inventory.md)
- [day3-end-state-claim-decision.md](./artifacts/day3-end-state-claim-decision.md)
- [day4-full-validation-design.md](./artifacts/day4-full-validation-design.md)
- [day5-validation-execution.md](./artifacts/day5-validation-execution.md)
- [day6-final-validation-package.md](./artifacts/day6-final-validation-package.md)
- [day7-final-comparison-inventory.md](./artifacts/day7-final-comparison-inventory.md)
- [day8-final-comparison-cleanup.md](./artifacts/day8-final-comparison-cleanup.md)
- [day9-residual-queue-intake.md](./artifacts/day9-residual-queue-intake.md)
- [day10-residual-queue-and-nonclaims.md](./artifacts/day10-residual-queue-and-nonclaims.md)
- [day11-sprint-retrospective-draft.md](./artifacts/day11-sprint-retrospective-draft.md)
- [day12-sprint-retrospective-finalization.md](./artifacts/day12-sprint-retrospective-finalization.md)
