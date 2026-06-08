# Sprint 59 Day 10 - Epic 5 closeout input audit

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Reduce the final Epic 5 closeout problem to a measured input set before any
summary or project-level residual writing lands.

This audit is intentionally narrower than a full closeout draft. Its job is
to fix:

- what Epic 5 can now legitimately claim as closed
- which validation anchors must appear in the final handoff
- which residuals are still consciously deferred
- what the remaining writing queue should be

## Closed work bands

The landed Sprint 50-59 branch history now reduces cleanly to these closed
Epic 5 work bands:

1. **Direct-solver lifecycle design fence**
   - Sprint 50
2. **Public direct lifecycle implementation and deeper analysis/refactor integration**
   - Sprints 51-52
3. **CSC direct-solver completion and dispatch follow-through**
   - Sprint 53
4. **Public repeated-run solver lifecycle completion**
   - Sprint 54
5. **Large-source decomposition**
   - Sprints 55-56
6. **Giant-test refactor and lifecycle/factor-many regression expansion**
   - Sprint 57
7. **Public-surface simplification**
   - Sprint 58
8. **Final quality/platform reconciliation and caller-story normalization**
   - Sprint 59 Days 1-9

## Validation anchors that must appear in the final Epic 5 handoff

The strongest maintained quality/truthfulness anchors remain:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake count anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity anchor:
  - `53 vs 53`
- full reviewed CMake `ctest` anchor:
  - `53 / 53`

As of Day 10, Sprint 59 has remained docs-only, so the latest fully measured
validation baseline is still inherited from Sprint 58 Day 13:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed
- reviewed CMake total time = `481.74 sec`

This is the baseline the Day 11 summary should reference until the Sprint 59
Day 13 validation sweep produces the final Epic 5 closeout baseline.

## Preserved compatibility fence

The final Epic 5 summary should carry one stable compatibility fence rather
than re-arguing it sprint by sprint:

- one-shot APIs remain first-class/default workflows
- repeated-run direct solves remain the explicit analysis/factors lifecycle:
  - analyze once
  - factor / solve
  - refactor / solve many
- repeated-run iterative handles remain limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated-run eigensolver handle remains limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces
- no broad public API redesign, raw internal storage exposure, or generic
  universal solver handle was introduced

## Consciously deferred residual queue

The remaining future-facing queue is now smaller than the original Epic 5
review implied.

### 1. Quality/platform residuals

- dead-code execution remains serialized
- macOS dead-code remains staged pending fresh measurement
- broader Windows reviewed-wrapper parity remains deferred
- Windows dead-code remains deferred/excluded
- coverage calibration is no longer an active residual

### 2. Maintainability residuals

- later iterative decomposition:
  - `GMRES`
  - shared block-wrapper scaffolding
- possible later eigensolver/private-header cleanup
- later CSC decomposition/comment cleanup if still justified
- deferred giant-test seams:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - intentionally retained dense `tests/test_integration.c`

### 3. Public-surface density residuals

- deeper long-form `README.md` chronology/performance-history cleanup
- broader docs-density reduction outside the bounded Sprint 58-59 target set

### 4. Non-residuals that should not be reopened by default

- generic direct-handle redesign
- raw CSC/native storage exposure
- broad repeated-run support-boundary expansion
- benchmark/example workflow redesign as a late Epic 5 surprise

## Ranked closeout-writing queue

### 1. Main Epic 5 summary / handoff artifact

The highest-value next writing target is the main Epic 5 handoff summary:

- organize by the eight closed work bands
- name the validation anchors explicitly
- state the preserved compatibility fence once
- state the consciously deferred residual queue once

### 2. Project-level plan/residual wording check

Only touch `PROJECT_PLAN.md` or other project-level residual wording if the
Day 11 summary exposes a real mismatch between:

- what Epic 5 now claims as closed
- what still needs an explicit future-facing defer state

### 3. Final Sprint 59 closeout package

The remaining sprint-local package should then close from the measured branch
state:

- Day 13 full validation sweep
- Day 14 Sprint 59 closeout/handoff
- later Sprint 59 retrospective

## Conclusion

The Epic 5 closeout now has a concrete measured input set:

- the completed work bands are explicit
- the final validation anchors are explicit
- the preserved compatibility fence is explicit
- the deferred residual queue is concrete and bounded
- the remaining writing queue is ranked

That is enough to start the main Epic 5 summary/handoff draft from measured
evidence rather than from generic retrospective language.
