# Sprint 91 Day 5: Compressed-First Architecture Design

## Purpose

Define the bounded Sprint 91 contract for compressed-first
construction/import/publication and shell containment before the first code
landing.

## Main Result

Sprint 91 now has one explicit compressed-first product contract:

- linked-list shell:
  - remains the mutable sparse construction and one-shot direct-workflow
    compatibility surface
  - remains valid for pedagogy, mutation-heavy callers, and compatibility
    one-shot flows
  - stops being treated as the only natural public entry path for callers that
    already have compressed inputs

- CSC/CSR-backed construction and import:
  - should read as first-class public entry paths for callers that already own
    compressed sparse data
  - should preserve physical-index-space truth and existing compatibility
    semantics
  - should not require broader lifecycle or publication rewrites in the first
    batch

- public publication/export seams:
  - stay bounded behind the first batch
  - remain real Sprint 91 work, but as the second seam after entry-path
    improvement
  - should be re-read later as ownership wording and round-trip cost, not as a
    justification to broaden the first landing

## Public Workflow Role Split

The useful public role split is now explicit:

- shell-first path:
  - caller wants incremental insertion, mutation, or compatibility one-shot
    direct workflows
- compressed-first path:
  - caller already has CSR/CSC data and wants to enter the product without an
    unnecessary shell-centered conceptual detour
- repeated-run direct path:
  - caller wants long-lived symbolic and factor/workspace state through
    `sparse_analysis.h`

That means Sprint 91 should not try to erase the shell. It should make the
compressed-first path read like a real peer path on the highest-value direct
and interop workflows.

## Compatibility-Shim Policy

The compatibility policy is now fixed:

- acceptable to keep:
  - shell-centered one-shot direct APIs
  - shell-centered mutation APIs
  - conversion/export helpers that preserve current behavior while wording and
    ownership are still being tightened
- should stop being conceptual center stage:
  - the idea that every serious direct or interop workflow should begin by
    mentally adopting the linked-list shell as the primary owner
- explicitly out of scope for the first landing:
  - broad shell deprecation
  - broad repeated-run owner rewrites
  - family-wide compressed-native API redesign

## Day 6 Implementation Contract

The exact Day 6 implementation center is now fixed to:

- `include/sparse_csr.h`
- the matching import/construction implementation seam behind the public
  matrix-shell owner

The strongest directly forced support-only follow-through, if truly needed, is:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`

The strongest support-only wording, if the contract truly forces it, is:

- `README.md`
- `docs/maintainer_guide.md`

## Strongest Clarification

The strongest Day 5 clarification is now explicit:

- Sprint 91 should promote compressed inputs to first-class public entry
  paths
- it should not claim the whole product is already compressed-first
- it should keep the shell as a bounded mutable compatibility surface while
  removing the strongest unnecessary shell-first conceptual detour

## Exit State

- Sprint 91 now has one explicit compressed-first architecture contract.
- The shell is bounded conceptually even though compatibility remains.
- Day 6 can land the first code batch without reopening product intent.
