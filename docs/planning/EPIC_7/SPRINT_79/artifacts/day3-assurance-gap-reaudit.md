# Sprint 79 Day 3 - Assurance Gap Re-audit

Date: 2026-06-18  
Branch: sprint-79

## Purpose
Re-rank the remaining highest-value oracle, property, lifecycle, and platform-confidence gaps after the main Epic 7 implementation work so Sprint 79 can spend its final assurance budget on the strongest real contradiction rather than on a generic closeout bucket.

## Main Result
Sprint 79's broad final-assurance problem is now reduced to one ranked contradiction map:
- strongest first target:
  - direct-family lifecycle/property assurance
- strongest second target:
  - platform-confidence-limited property coverage
- strongest third target:
  - family-local differential/oracle follow-through
- strongest later target:
  - support-surface overclaim sweep

## Strongest First Target
The strongest current contradiction center is the bounded residual direct-usability queue already named in maintainer policy:
- no-reorder linked-list Cholesky bit-identical cancellation restoration
- broader CSC progress-callback parity beyond the landed bounded Cholesky orchestration checkpoints
- any later LDL^T callback follow-through

This is the best first Sprint 79 target because it combines:
- the highest remaining user-facing correctness value in the explicit residual queue
- strong boundedness relative to broader subsystem work
- direct proof payoff at the public callback/cancel truth seam
- strong final-integration value because the same seam touches regression ownership, public caveat wording, and maintainer-policy interpretation

The strongest likely proof owners for that first lane are already clear:
- public oracle owner:
  - `tests/test_integration.c`
- bounded seeded generative/property owner:
  - `tests/test_fuzz.c`
- family-local support only if the first lane truly forces it:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`

## Strongest Second Target
Platform-confidence-limited property coverage is the strongest second lane:
- Linux and macOS still exercise the full `test_fuzz` binary in direct local and reviewed paths
- Windows still excludes `test_fuzz` from the reviewed CMake subset
- the docs already state that boundary truthfully, so this is a real confidence limit but not the strongest first landing

## Strongest Third Target
Family-local differential/oracle follow-through remains real but lower-ranked:
- QR, LDL^T, LDL^T CSC, and ND already carry strong residual-heavy proof
- those lanes still offer assurance-expansion value
- they currently read more like coherent large proof owners than the strongest unresolved contradiction center

## Weakest Current Lane
Support-surface overclaim is the weakest current assurance lane:
- callback/path-local limitations are already explicit
- benchmark/reporting truth is already explicit
- install/export proof boundaries are already explicit
- Windows fuzz exclusion is already explicit

That means Sprint 79 should not start from another wording-first pass unless the later integration sweep reveals a real contradiction.

## Where the Current Proof Package Is Already Coherent
The live Epic 7 tree is already coherent across:
- repeated-run direct workflow and benchmark/reporting ownership
- install/export proof split
- benchmark-governance and threshold-free reporting
- width/scalar/platform truth fences
- the direct-family caveat reading that family/path-local callback and cancellation semantics remain intentionally bounded

## Interpretation
The strongest Day 3 clarification is now explicit:
- Sprint 79 is not primarily missing broad residual-norm coverage.
- It is primarily missing the best final bounded expansion of lifecycle/property assurance on the direct-family callback/cancel seam.
- Final assurance work should therefore rank bounded lifecycle/property truth above broader proof churn or support-surface cleanup.

## Exit State
- The broad Sprint 79 assurance problem is reduced to one ranked contradiction map.
- The strongest first final-assurance lane is explicit.
- Day 4 can now freeze a first landing boundary from a real current-state ranking.
