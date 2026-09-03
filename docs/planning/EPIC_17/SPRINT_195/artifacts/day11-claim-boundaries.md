# Sprint 195 Day 11: Claim Boundaries

## Purpose

Publish the selected `sparse_symbolic_cholesky()` allocation-failure proof
without widening reliability, platform, package, performance, or
state-of-the-art claims.

## Documentation Updates

| Surface | Update |
| --- | --- |
| `README.md` quality bullet | Added `make symbolic-allocation-failure-gate` as the selected symbolic Cholesky cleanup/retry allocation-failure proof. |
| `README.md` command map | Added the focused local symbolic allocation-failure gate command. |
| `README.md` repeated-run guidance | Added a non-claim paragraph limiting symbolic allocation-failure proof to selected `sparse_symbolic_cholesky()` behavior. |
| `INSTALL.md` support-readiness matrix | Added a local-only selected allocation-failure proof row with explicit non-claims. |
| `docs/maintainer_guide.md` proof-owner ledger | Added `test_etree` as the Sprint 195 bounded allocation-failure owner with test names, gate commands, CTest selector, guard, invariant artifacts, and non-claims. |

## Earned Claim

The earned Sprint 195 public claim is:

`make symbolic-allocation-failure-gate` provides a focused local proof for
selected `sparse_symbolic_cholesky()` allocation-failure status, output cleanup,
stale-output suppression, repeated cleanup after failure, and retry-after-reset
behavior on bounded fixtures.

## Retained Non-Claims

The Sprint 195 wording explicitly does not claim:

- broad allocation-failure coverage;
- `sparse_symbolic_lu()` allocation-failure proof;
- `sparse_analyze()` publication cleanup proof;
- standalone etree, postorder, or colcount allocation-failure proof;
- direct-solver, eigensolver, graph, sparse matrix construction, conversion,
  IO, package/install, or generated-tooling allocation-failure proof;
- OS-level OOM guarantees;
- concurrent allocation-hook behavior;
- hosted CI proof for this local gate unless a future hosted lane names it;
- platform, package, ABI, performance, release, or state-of-the-art
  reliability proof.

## Claim-Boundary Check

The public and maintainer wording points readers to the focused Make gate,
CTest selector, registration guard, and invariant artifacts while keeping the
support-readiness matrix local-only. No documentation text presents the new
symbolic lane as broad library reliability evidence.
