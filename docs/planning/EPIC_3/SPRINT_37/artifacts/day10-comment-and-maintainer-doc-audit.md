# Sprint 37 Day 10 Comment and Maintainer-Doc Audit

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Identify stale or duplicative maintainer-facing comments and support docs in
the reviewed-quality / cross-platform / dead-code workflow surfaces, then pick
the narrowest high-value wording cleanup batch for Day 11.

## Executive Summary

The repo’s workflow contract is mostly correct. The Day 10 debt is not
incorrect guidance; it is **too much repeated guidance** and **too much
sprint-history framing** in files that now function as maintained operator
surfaces.

Highest-value Day 11 cleanup batch:

1. Simplify workflow top-of-file headers in:
   - `.github/workflows/ci.yml`
   - `.github/workflows/macos-ci.yml`
   - `.github/workflows/windows-ci.yml`
2. Refresh stale Sprint 33-first wording in:
   - `scripts/deadcode_workflow.sh`
   - `scripts/deadcode_report.py`
3. Compress duplicated maintainer workflow wording in:
   - `README.md`

Not recommended:

- broad comment deletion
- numerical/doc tutorial rewrites
- removal of still-useful platform/tooling caveats

## Main Audit Findings

### 1. Workflow headers are accurate but over-explained

The workflow files currently do more than they need to at the top of the file:

- define current enforced/staged/supplemental contract
- preserve sprint provenance
- preserve detailed platform rationale
- duplicate information already summarized in `README.md`

Affected files:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

Why this matters:

- these are active operator surfaces
- maintainers reopening them for routine CI edits pay the cost of long header
  rereads before reaching the actual jobs
- the README already owns the fuller human-readable cross-platform contract

Recommended treatment:

- **refresh / simplify**
- keep a short pointer-style summary of the current role and exclusions
- point back to the authoritative README section for the fuller contract

### 2. Dead-code scripts still describe themselves primarily as Sprint 33 artifacts

This is the clearest stale wording in the maintained support layer.

Affected wording:

- `scripts/deadcode_workflow.sh`
  - `Sprint 33 Day 5 raw dead-code workflow`
  - `Sprint 33 Day 5 dead-code coverage notes`
- `scripts/deadcode_report.py`
  - `Generate and validate Sprint 33 dead-code reports.`
  - `Generate or validate dead-code reports from raw Sprint 33 artifacts.`
  - report title `Sprint 33 Dead-Code Report`

Why this matters:

- the tools are now active maintained utilities, not one-sprint throwaways
- Sprint 33 provenance is still useful, but should not dominate the primary
  operator label

Recommended treatment:

- **refresh**
- rename the primary wording around current maintained purpose
- keep provenance only where it still helps explain policy/history

### 3. `README.md` owns the right surface, but it can be compressed

The relevant README section correctly covers:

- dead-code workflow
- reviewed wrappers
- cross-platform CI contract
- operator command map
- rerun guidance
- tree-mutating mode reset guidance

The problem is repetition, not ownership.

Current duplication patterns:

- command names reappear in multiple adjacent sections
- contract distinctions are restated in both prose and lists more than once
- reset guidance after sanitizer/coverage modes duplicates what wrappers and
  Makefile comments already say

Recommended treatment:

- **refresh / simplify**
- keep README as the authoritative workflow surface
- reduce repeated wording and rely more on shorter pointer-style subsections

### 4. Some dense comments should remain

Not all historical or detailed comments are debt.

Keep:

- Linux TSan / libomp rationale in `.github/workflows/ci.yml`
- Apple Clang / libomp / GCC portability notes in `Makefile`
- coverage backend split rationale in `Makefile`
- partial-success handling rationale in `scripts/deadcode_workflow.sh`

Reason:

- these comments still encode active platform/tooling constraints
- removing them would hurt rather than help maintainability

Recommended treatment:

- **keep**

## Ranked Cleanup Queue

### Priority A

- workflow top-of-file header simplification:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- stale dead-code script/report wording refresh:
  - `scripts/deadcode_workflow.sh`
  - `scripts/deadcode_report.py`

### Priority B

- `README.md` maintainer workflow compression

### Priority C

- any small matching `Makefile` wording cleanup discovered during the Day 11
  batch, only if it reduces duplication without weakening ownership clarity

## Delete / Refresh / Keep Split

### Delete or heavily compress

- long sprint-history framing at the top of active workflow YAML files
- repeated contract framing that duplicates README without adding new operator
  value

### Refresh

- Sprint 33-first dead-code utility wording
- README workflow wording that repeats command maps and contract statements

### Keep

- platform/tooling rationale that still explains active constraints
- ownership comments in `Makefile` that clarify maintained vs helper vs
  tree-mutating targets

## Defined Day 11 Batch

Bounded Day 11 implementation scope:

1. Simplify workflow headers to short current-contract summaries
2. Refresh dead-code script/report headers and report title to current
   maintained wording
3. Compress the README maintainer workflow section so it preserves the current
   Sprint 34-Sprint 36 contract with less repeated wording

Explicitly out of scope:

- algorithm docs
- tutorial/public API prose
- benchmark commentary cleanup
- broad removal of detailed but still-useful implementation rationale

## Day 10 Conclusion

The maintainer workflow surfaces already say the right things. They just say
them too many times and too often through sprint-history framing. The best next
step is a bounded simplification pass on workflow headers, dead-code utility
labels, and the README workflow section rather than a broad comment purge.
