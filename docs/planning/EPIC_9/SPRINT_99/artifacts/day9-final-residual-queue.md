# Sprint 99 Day 9: Final Residual Queue

## Purpose

Day 9 finalizes the post-Epic-9 residual queue. It consolidates residual work
from Sprints 90-98 and the Sprint 99 evidence sweep, removes duplicate/stale
entries, and separates true carry-forward work from deliberate non-claims and
already-resolved items.

## Classification Summary

| Classification | Count | Meaning |
|---|---:|---|
| post-Epic-9 carry-forward | 8 | real future work, but not a Sprint 99 closeout blocker |
| deliberate non-claim | 10 | scope intentionally not claimed by Epic 9 |
| unsupported claim to remove | 0 | no live positive unsupported claim found by Day 4-5 scans |
| already resolved | 12 | completed by Sprints 90-99 evidence package |

## Post-Epic-9 Carry-Forward Queue

| Item | Owner surface | Rationale | Validation expectation |
|---|---|---|---|
| broader LDLT CSC Matrix Market or indefinite corpus comparison | `tests/test_ldlt_csc.c`, `tests/ldlt_external_dense_reference.py`, future fixtures | current lane proves named deterministic KKT fixtures, not broad indefinite corpus behavior | design reference architecture first; run focused helper/test plus full C quality chain if C changes |
| iterative solver external comparison architecture | iterative tests, solver reference scripts if introduced | convergence semantics, preconditioning, restart behavior, and residual boundaries require design before proof | write boundary artifact before implementation; run focused iterative tests and full quality chain if C changes |
| eigensolver/LOBPCG external comparison architecture | eigensolver tests and future references | cluster behavior, tolerance policy, runtime cost, and reference eigenpairs need explicit limits | design fixtures and tolerance rules first; run focused eigensolver tests and full quality chain if C changes |
| QR/SVD external comparison architecture | `tests/test_qr.c`, `tests/test_svd.c`, future references | numerical tolerance and runtime cost are higher than direct solve comparison lanes | design per-family reference/tolerance ownership before implementation |
| generated reorder/fill report target if repeated captures justify it | `benchmarks/bench_reorder.c`, scripts, benchmark docs | Day 4 artifact command works; generated report target is only worth adding after repeated need is proven | preserve `nnz_L` as primary fill field; avoid portable timing thresholds; run focused benchmark/report commands |
| continued large-source extraction | largest source owners such as QR, eigs, LU CSR, LDLT, matrix, SVD | Sprint 96 improved selected owners but did not eliminate all mixed-role files | design extraction boundaries per family; run focused tests and full C quality chain for source/header changes |
| continued giant-test extraction | `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_integration.c`, `tests/test_etree.c`, `tests/test_graph.c`, and adjacent broad proof owners | proof concentration remains a review-cost issue but not a closeout blocker | split with registration parity checks; run focused tests plus Make/CMake count checks |
| lower-level chronology cleanup where useful | lower-level tests, implementation comments, selected docs | public surfaces are claim-safe, but some lower-level proof owners still read historically | avoid compatibility-breaking renames without migration plan; docs hygiene for docs-only work |

## Deliberate Non-claims to Preserve

| Non-claim | Why preserved | Guardrail owner |
|---|---|---|
| full compressed-first replacement of the linked-list shell | linked-list shell remains the mutable compatibility owner | README, `include/sparse_matrix.h`, `include/sparse_csr.h` |
| broad complex support | scalar widening does not implement broad complex semantics | `include/sparse_types.h`, README, maintainer guide |
| broad mixed-precision maturity | no broad mixed-precision implementation/proof exists | README, maintainer guide |
| broad backend-neutral acceleration maturity | backend seams improved, but not an industrial backend stack | maintainer guide, backend tests |
| shared-library-first package contract | install/export proof is intentionally static-first | `INSTALL.md`, install scripts |
| dynamic ABI guarantee | exact-version package proof exists, not dynamic ABI policy | `INSTALL.md`, CMake package proof |
| symmetric Linux/macOS/Windows reviewed parity | platform proof is intentionally asymmetric | workflows, README, INSTALL |
| Windows Makefile parity or install-validation lane | Windows reviewed scope is CMake-first subset only | `.github/workflows/windows-ci.yml` |
| portable timing superiority or universal reorder/fill superiority | runtime/fill evidence is local and bounded | benchmark docs, maintainer guide |
| every-solver-family external correctness comparison | maintained external lanes are Cholesky CSC and LDLT CSC only | maintainer guide, Sprint 99 closeout docs |

## Unsupported Claims to Remove

None found.

Day 4 and Day 5 scans found only negative guardrails and boundary language, not
positive unsupported broad claims. If future scans find a positive unsupported
claim, it should be removed or backed by a designed proof lane before any
closeout language repeats it.

## Already Resolved or No Longer Carried as Debt

| Item | Resolution evidence |
|---|---|
| Epic 9 target-state and claim fence | Sprint 90 planning package and retrospective |
| comparison and measurement contract | Sprint 90 Day 6 and Sprint 99 Day 3 |
| compressed-first constructor/public workflow proof | Sprint 91 closeout |
| bounded backend widening and LDLT backend observability | Sprint 92 closeout |
| bounded runtime/control cleanup | Sprint 93 closeout |
| bounded scalar/index capability widening | Sprint 94 closeout |
| public narrative, install, benchmark, and maintainer support cleanup | Sprint 95 closeout |
| selected source/proof-owner extraction | Sprint 96 closeout |
| source-list guard and static-first package proof | Sprint 97 closeout |
| LDLT CSC external correctness and bounded reorder/fill artifact | Sprint 98 closeout and Sprint 99 Day 4 |
| package/install/export evidence | Sprint 99 Day 5 |
| final fix/no-fix decision | Sprint 99 Day 6 |

## Duplicate and Stale Residual Cleanup

The following older residual entries are folded into consolidated queue items:

- "broader comparison depth" now means four explicit architecture queues:
  LDLT corpus, iterative, eigensolver/LOBPCG, and QR/SVD.
- "runtime/fill comparison" now means generated reporting only if repeated
  captures justify it; portable timing remains a non-claim.
- "large-source and proof-owner cleanup" now means focused family-local
  extraction, not broad simultaneous rewrites.
- "chronology cleanup" now excludes historical planning artifacts and focuses
  only on current proof/support surfaces where history obscures ownership.
- "build/package/workflow convergence" now excludes fake platform parity and
  focuses on future registration/checker reductions only where proof clarity
  improves.

## Day 9 Conclusion

No unresolved closeout blocker is hidden in the residual queue. The queue is
explicit, non-duplicative, and protected by deliberate non-claims. Sprint 99
can proceed to the full reviewed validation sweep without a pending final fix
batch.
