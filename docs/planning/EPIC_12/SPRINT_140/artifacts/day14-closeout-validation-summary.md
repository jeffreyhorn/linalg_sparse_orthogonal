# Day 14 Closeout Validation Summary

## Sprint 140 Result

Sprint 140 closes the selected partial-SVD residual with a bounded,
fixture-local evidence lane for
`partial_svd_clustered_repeated_diag8x6_k3_v1`.

The sprint adds a deterministic 8x6 clustered/repeated diagonal fixture,
expected corpus rows, opt-in oracle/report support, a focused compiled proof
owner, helper cleanup, public documentation wording, explicit non-claims, and a
Sprint 141 report-index handoff.

## Project-Plan Item Inventory

| Item | Status | Evidence |
| --- | --- | --- |
| 1. Partial-SVD Residual Reaudit | Complete | Day 1 intake and Day 2 reaudit artifacts select the clustered/repeated top-k subspace and convergence-budget residual. |
| 2. Edge-Case Fixture Batch | Complete | Day 3 and Day 4 design artifacts plus Day 5 corpus fixture implementation define `partial_svd_clustered_repeated_diag8x6_k3_v1`. |
| 3. Comparison Semantics | Complete | Day 6 comparison semantics and Day 7 oracle implementation cover value, subspace, residual, status, diagnostic, stale, skip, and malformed-row handling. |
| 4. Convergence-Budget Tests | Complete | Day 8 proof design and Day 9 implementation add `tests/test_svd_partial_corpus.c` with default-budget, tight-budget fail-closed, and recovery checks. |
| 5. Proof-Owner Cleanup | Complete | Day 10 moves reusable projector and residual checks into `tests/test_svd_partial_shared_helpers.h` and keeps fixture-specific construction local. |
| 6. Validation | Complete | Day 12 runs corpus, oracle, focused test, source-list, CMake parity, full Make quality gate, and hygiene checks. Day 13 reruns the full Make quality gate after claim closure. |
| 7. Docs and Closeout | Complete | Day 11 publishes fixture-local wording, Day 13 closes the claim and Sprint 141 handoff, and Day 14 records closeout readiness. |

No Sprint 140 project-plan item is partially implemented without an explicit
boundary. Remaining work is deferred because it is outside the selected
fixture-local closure.

## Final Artifact Inventory

| Day | Artifact | Purpose |
| --- | --- | --- |
| 1 | `day1-partial-svd-residual-intake.md` | Inventories candidate partial-SVD residuals and validation boundaries. |
| 2 | `day2-partial-svd-residual-reaudit.md` | Re-ranks residuals and selects the priority closure target. |
| 3 | `day3-partial-svd-closure-design.md` | Defines the concrete fixture key, rows, claim boundary, and validation strategy. |
| 4 | `day4-partial-svd-fixture-batch-design.md` | Finalizes fixture metadata, expected rows, and non-claim wording. |
| 5 | `day5-partial-svd-fixture-implementation.md` | Records manifest, generator, schema, and expected TSV implementation. |
| 6 | `day6-partial-svd-comparison-semantics.md` | Defines row comparison semantics and failure classes. |
| 7 | `day7-partial-svd-oracle-implementation.md` | Records opt-in oracle/report implementation and generated-reference boundaries. |
| 8 | `day8-partial-svd-convergence-proof-design.md` | Designs the focused compiled proof owner and build ownership. |
| 9 | `day9-partial-svd-proof-implementation.md` | Records compiled test implementation and Make/CMake registration. |
| 10 | `day10-proof-owner-cleanup.md` | Records shared-helper cleanup and ownership boundaries. |
| 11 | `day11-documentation-update.md` | Records public documentation updates and preserved non-claims. |
| 12 | `day12-validation-pass.md` | Records focused and full validation evidence. |
| 13 | `day13-claim-closure.md` | Records closed claim, remaining non-claims, traceability, and Sprint 141 handoff. |
| 14 | `day14-closeout-validation-summary.md` | Records final inventory, residual summary, validation status, and retrospective input. |

## Validation Status

Final full changed-code quality gate:

```sh
make format && make lint && make test
```

Result: PASS on Day 14 after the closeout artifact was added. The full suite
included `test_svd` and the new `test_svd_partial_corpus` owner; the new owner
ran 5 tests with 0 failures and 107 assertions.

Day 14 final documentation hygiene checks:

```sh
git diff --check
rg -n "[[:blank:]]$" README.md docs/cookbook.md docs/tutorial.md docs/solver_selection.md docs/algorithm.md docs/maintainer_guide.md examples/README.md tests/corpus/README.md include/sparse_svd.h docs/planning/EPIC_12/SPRINT_140 tests/corpus scripts/validate_corpus_schema.py scripts/run_corpus_oracle.py src/sparse_svd_partial.c tests/test_svd_partial_corpus.c tests/test_svd_partial_helpers.h tests/test_svd_partial_shared_helpers.h Makefile CMakeLists.txt
```

Result: PASS. The branch contains `.c` and `.h` changes, so final pull-request
packaging should cite the Day 14 full Make quality gate.

## Closed Residual

Closed:

- fixture-local partial-SVD behavior for the generated 8x6 clustered/repeated
  diagonal fixture with `k = 3`;
- top-3 singular-value comparison;
- left and right top-k subspace-projector comparison;
- triplet residual and orthogonality checks;
- default-budget success;
- tight-budget fail-closed status; and
- no partial `sigma`, `U`, or `Vt` publication on tight-budget failure.

## Deferred Work

Deferred outside Sprint 140:

- broad partial-SVD correctness across arbitrary spectra and shapes;
- broad repeated-spectrum, rank-deficient, null-space, pseudoinverse, and
  minimum-norm behavior;
- low-rank sparse-output/drop-tolerance optimality;
- convergence-rate or portable iteration-count claims;
- partial-result guarantees after non-convergence;
- external-library parity;
- hosted-platform promotion;
- package, install, shared-library, or ABI claims;
- performance claims; and
- state-of-the-art claims.

These remain explicit non-claims in the public and maintainer documentation.

## Retrospective Input Notes

What worked:

- The sprint closed one narrow residual completely instead of expanding several
  partial-SVD surfaces at once.
- The fixture-local claim is traceable from corpus metadata through oracle rows,
  compiled tests, documentation, and validation.
- Subspace/projector comparison avoided brittle raw-vector identity checks for
  clustered and repeated leading singular values.
- Tight-budget behavior now has a clear fail-closed proof path.

What to watch:

- Generated-reference oracle rows must not be interpreted as hosted-platform or
  solver-backed pass evidence.
- The focused proof owner should stay small; broad partial-SVD expansion belongs
  in future planned residual closures.
- Report-index freshness is still not normalized enough to support durable
  cross-run evidence interpretation.

Sprint 141 should use the Day 13 handoff as its starting point for normalized
report indexes, freshness metadata, stale-report detection, and support-tier
interpretation.

## Closeout Decision

Sprint 140 is ready for retrospective creation and pull-request packaging. The
final Day 14 validation command set passed on the final working tree.
