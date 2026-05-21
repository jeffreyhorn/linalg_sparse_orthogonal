# Sprint 37 Day 11 Maintainer Workflow Docs Batch

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Implement the bounded wording-cleanup batch chosen on Day 10 so the active
maintainer workflow surfaces are easier to scan, less repetitive, and less
dominated by sprint-history framing.

## Scope

Touched files:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- `README.md`

Untouched by design:

- numerical algorithm docs
- public API/tutorial docs
- benchmark-owner commentary
- dense but still-useful platform/tooling rationale in `Makefile`

## What Changed

### 1. Workflow headers are now pointer-style summaries

The top-of-file workflow comments were shortened so they now emphasize:

- the current platform role
- the major enforced/staged boundary
- the authoritative README section for the fuller contract

This removed a large amount of repeated Sprint 36 narration without changing:

- job names
- step names
- enforced/staged/supplemental meaning

### 2. Dead-code utilities now describe current purpose

The dead-code support files were refreshed from Sprint 33-first labels to
current maintained-purpose wording:

- `deadcode_workflow.sh` now presents itself as a raw evidence refresh script
- coverage-notes output now uses `Dead-code coverage notes`
- `deadcode_report.py` docstring and CLI description now refer to raw workflow
  artifacts
- the rendered markdown title is now `Dead-Code Report`

This keeps provenance implicit in the surrounding repo history instead of
making it the primary operator-facing label.

### 3. README maintainer workflow section is tighter

The workflow section in `README.md` now says the same thing with less repeated
framing:

- dead-code workflow intro is shorter
- reviewed-wrapper explanation is more compact
- cross-platform interpretation bullets are tighter
- tree-mutating mode reset guidance is still explicit but less repetitive

The authoritative ownership did not move:

- README still owns the maintainer-facing workflow contract
- workflow YAML files now point to it instead of retelling it in full

## Validation

Direct touched-surface validation passed:

- YAML parse:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- shell syntax:
  - `bash -n scripts/deadcode_workflow.sh`
- Python compile:
  - `python3 -m py_compile scripts/deadcode_report.py`

Dead-code validation note:

- one attempted concurrent rerun reproduced the known shared-path race between
  `deadcode-report` and `deadcode-check`
- the authoritative serial rerun passed:
  - `make deadcode-report`
  - `make deadcode-check`
  - `make deadcode-report && make deadcode-check`

Interpretation:

- the Day 11 wording cleanup did not change the workflow contract
- the only operational caveat remains the already-known serial-only dead-code
  constraint

## Residual Queue

Still deferred:

- dense but still-useful platform/tooling rationale comments
- the dead-code shared-path limitation itself
- any further README compression that would start trading away operator clarity

## Day 11 Conclusion

The active workflow surfaces now read more like maintained operator interfaces
and less like stitched-together sprint history:

- workflow YAML headers are shorter and clearer
- dead-code utilities describe current purpose directly
- README remains the authoritative workflow contract with less repetition

The real workflow behavior is unchanged, and the known dead-code serial-only
limitation remains explicit.
