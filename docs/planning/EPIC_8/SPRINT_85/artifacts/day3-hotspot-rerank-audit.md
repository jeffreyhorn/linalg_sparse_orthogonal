# Sprint 85 Day 3: Hotspot Rerank Audit

## Purpose

Reduce Sprint 85's broad maintainability problem to one ranked live
contradiction map so the sprint can choose one bounded decomposition lane
instead of another generic “split large files” bucket.

## Main Result

Sprint 85's broad maintainability problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - bounded iterative-source cleanup centered on `src/sparse_iterative.c`
- strongest second target:
  - bounded direct-family source cleanup centered on `src/sparse_chol_csc.c`
- strongest third target:
  - bounded giant-test architecture cleanup centered first on
    `tests/test_chol_csc.c`
- strongest fourth target:
  - next giant-test/source follow-through on `tests/test_qr.c`,
    `tests/test_integration.c`, `src/sparse_qr.c`, or `src/sparse_ldlt.c`
- strongest support-only but real target:
  - maintainer/docs wording only where cleanup changes proof-owner or helper
    boundaries

## Strongest Current Contradiction

The strongest current contradiction is not simply that large files exist:

- `src/sparse_iterative.c` is the largest implementation hotspot at `1985`
  lines
- it already sits on a retained reviewed proof-owner seam through
  `tests/test_iterative.c`
- it is also the most natural first bounded extraction lane because Sprint 84
  already closed from a stable assurance baseline and explicitly handed Sprint
  85 the maintainability-first queue

That fixes the strongest first Sprint 85 move:

- land one bounded iterative-source cleanup first
- preserve the current proof-owner contract while reducing local reasoning and
  review cost
- treat broader direct-family and giant-test cleanup as follow-through only if
  that first lane lands cleanly

## Second-Tier Contradictions

### Direct-Family Source Concentration

The strongest second contradiction is direct-family source concentration:

- `src/sparse_chol_csc.c` remains a very large source hotspot at `1841` lines
- `src/sparse_qr.c` = `1563`
- `src/sparse_ldlt.c` = `1535`
- `src/sparse_eigs.c` = `1534`

This is real Sprint 85 work, but it still reads as second after the first
iterative decomposition lane rather than the initial implementation center.

### Giant-Test Concentration

The strongest third contradiction is giant-test concentration:

- `tests/test_chol_csc.c` = `4964`
- `tests/test_qr.c` = `3234`
- `tests/test_integration.c` = `3197`
- `tests/test_ldlt.c` = `2921`
- `tests/test_iterative.c` = `2841`

The live tree shows helper and registration concentration across these proof
owners, especially the Cholesky CSC, QR, LDL^T, and integration seams. That
means giant-test architecture cleanup is real Sprint 85 work, but the current
rerank still favors one source-owner cleanup first rather than an immediate
test-only batch.

### Support-Only and Later Runtime Follow-Through

The strongest support-only surfaces remain bounded:

- `README.md` = `1050`
- `docs/maintainer_guide.md` = `726`

These remain follow-through only where landed decomposition changes helper,
owner, or rerun expectations.

Reviewed runtime-convergence pressure also remains intentionally later:

- Sprint 84 already handed `test_reorder_nd` runtime-dominance work to Sprint
  86
- Sprint 85 should therefore not widen early into runtime-tuning or benchmark
  ownership work

## Deferred Maintainability Claims

Broad maintainability-claim widening remains lower-value first work:

- no repo-wide claim that all large sources will be decomposed this sprint
- no proof-owner diffusion by moving helpers without a stronger owner contract
- no benchmark or example drift into correctness ownership
- no package/platform maturity claim widening
- no support-surface churn detached from a real landed hotspot seam
- no reopening Sprint 84's bounded assurance package as the first Sprint 85
  implementation center

## Interpretation

The useful Day 3 clarification is now explicit:

- the best first Sprint 85 move is not generic cleanup
- it is one bounded iterative-source decomposition on the current highest-value
  implementation hotspot
- bounded direct-family source cleanup follows next
- giant-test architecture cleanup follows after that where the first source
  landings expose the real helper seams
- support surfaces stay support-only unless implementation truly changes the
  owner or rerun contract
- runtime-convergence and `test_reorder_nd` pressure remains explicitly later
  than the first Sprint 85 implementation lane

## Exit State

- Sprint 85 now has one ranked live maintainability contradiction map grounded
  in the current tree.
- The first implementation center is fixed to one bounded iterative-source
  cleanup lane.
- Later direct-family source cleanup, giant-test architecture work, and
  support-only wording follow-through are explicitly ordered behind that first
  lane.
