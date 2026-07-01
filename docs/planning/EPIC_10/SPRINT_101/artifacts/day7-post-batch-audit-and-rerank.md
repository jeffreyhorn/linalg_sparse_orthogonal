# Sprint 101 Day 7 Post-Batch Audit and Rerank

## Purpose

Day 7 audits the Day 6 constructor/import batch against the Day 4 API design
and Day 5 implementation boundary. It reconciles landed behavior before any
new edits, reranks remaining candidates, and decides whether a second
implementation batch is justified.

## Day 6 Boundary Compliance

| Day 5 boundary item | Day 6 result | status |
|---|---|---|
| clarify simple versus diagnostic constructor roles in `include/sparse_csr.h` | header comments distinguish `sparse_create_from_csr/csc` from `sparse_from_csr/csc` | complete |
| add focused CSR/CSC bad-input diagnostics in `tests/test_csr.c` | null, shape, pointer, monotonicity, index, duplicate, and unsorted cases added | complete |
| prove caller-owned input arrays are copied, not adopted | CSR and CSC array mutation-after-construction tests added | complete |
| add bounded solver-entry proof | CSR-built matrix enters one-shot LU factor/solve path | complete |
| preserve ABI and mutable-shell compatibility | no signatures or implementation behavior changed | complete |
| avoid broad direct CSR/CSC solver APIs | no new solver APIs added | complete |
| run required C quality gate | `make format`, `make lint`, and `make test` passed after fixing a lint finding | complete |

## Day 4 Design Reconciliation

| design requirement | evidence | assessment |
|---|---|---|
| promote existing constructors instead of inventing parallel APIs | Day 6 changed comments/tests only; no new public constructor family | aligned |
| clarify `sparse_from_csr/csc` as diagnostic front-door constructors | comments and tests now make diagnostic role explicit | aligned |
| preserve copy/build ownership | tests mutate caller arrays after construction and verify matrix values remain unchanged | aligned |
| preserve `SparseMatrix` as compatibility shell | returned matrix remains normal public shell; solver smoke uses existing LU entry | aligned |
| add proof without broad solver parity claim | one small LU smoke proves workflow entry only | aligned |
| defer docs/example narrative until implementation proof exists | no README/tutorial/example changes landed on Day 6 | aligned |

## Ownership, Lifetime, and Mutation Drift Check

| area | observed state | drift |
|---|---|---|
| caller-owned CSR arrays | read during construction; not retained; test proves later caller mutation does not affect matrix | none |
| caller-owned CSC arrays | read during construction; not retained; test proves later caller mutation does not affect matrix | none |
| returned matrix ownership | caller-owned `SparseMatrix *`, freed with `sparse_free` | none |
| diagnostic failure outputs | implementation sets output to `NULL` before validation/build; tests cover representative failures | none |
| mutable shell behavior | no source changes to `sparse_create`, `sparse_insert`, `sparse_remove`, or `sparse_set` | none |
| solver entry behavior | one existing one-shot LU path used; no solver ownership changed | none |
| public docs narrative | unchanged pending later documentation days | expected |

## Validation Status

Day 6 required and passed:

```bash
make format
make build/test_csr
./build/test_csr
make lint
make test
```

The first lint run found two non-portable sentinel pointer casts in the new
tests. Those were removed, and the full required gate passed on rerun.

## Remaining Candidate Rerank

| rank | candidate | current value | risk | decision |
|---:|---|---|---|---|
| 1 | lifecycle and ownership design across CSR/CSC import/export, matrix shells, factors, and handles | high | low | select for Day 8 design, not immediate code |
| 2 | public tutorial/docs compressed-input workflow | high | low | schedule after lifecycle design so wording includes ownership rules |
| 3 | examples README or focused compressed-input example | high | low-medium | schedule after docs design; may need example compile validation if `.c` example lands |
| 4 | additional solver smoke beyond LU | medium | medium | defer to Day 12 regression proof unless lifecycle design identifies a narrow need |
| 5 | internal CSR/CSC build-path optimization | medium | medium-high | defer; no urgent correctness or API clarity need remains |
| 6 | matrix-free CSR/CSC adapter pattern | medium | medium | defer; likely documentation/example topic, not core API change |
| 7 | adopt/no-copy constructors | medium | high | defer beyond Sprint 101 unless separately designed |
| 8 | direct CSR/CSC solver entry APIs | medium theoretical | high | reject for Sprint 101 |

## Second-Batch Decision

No second constructor/import implementation batch is justified immediately.

Reasons:

- Day 6 completed the selected Batch 1 scope.
- The remaining highest-value work is lifecycle/ownership design and public
  narrative, not another constructor patch.
- Additional solver smoke tests risk implying broader solver parity unless
  Day 8-9 define ownership and validation scope first.
- Internal build-path optimization is not needed to earn the bounded
  compressed-first front-door claim and would expand validation risk.

## Selected Next Step

Proceed to Day 8 lifecycle and ownership design before more code changes.
Day 8 should:

- reconcile CSR/CSC constructor copy rules with export ownership;
- map how constructed matrix shells interact with direct factors, analysis
  objects, iterative handles, eigensolver handles, and preconditioners;
- identify exact docs/tests needed for Day 10-12;
- preserve the non-claim that Sprint 101 does not replace `SparseMatrix` or
  provide direct CSR/CSC solver parity.

## Day 7 Conclusion

The Day 6 batch landed within boundary and supports the bounded
compressed-first front-door claim. The sprint should not immediately expand
constructor/import implementation. The next highest-value work is to make the
ownership and lifecycle story explicit, then update docs/examples and
regression coverage from that design.
