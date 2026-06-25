# Sprint 90 Day 6: Comparison and Measurement Contract Design

## Purpose

Freeze the external correctness, runtime, package-shape, and workflow
comparison model for Epic 9 so later sprint work can widen evidence
intentionally instead of opportunistically.

## Main Result

Sprint 90 now has one explicit Epic 9 comparison-and-measurement contract:

- maintained correctness comparison lane:
  - bounded external SPD differential proof
- maintained package-shape comparison lane:
  - local install/export and downstream-consumer proof
- bounded runtime-reference lane:
  - touched reorder/runtime comparison slices plus canonical reporting
- advisory but not maintained comparison lanes:
  - broader ecosystem, broader platform, and broader solver-family comparison

The strongest evidence-class order is now fixed:

- first:
  - correctness evidence strong enough to support bounded product claims
- second:
  - package-shape and installed-consumer evidence strong enough to support
    bounded product-shape claims
- third:
  - bounded runtime-reference evidence strong enough to support calibration,
    not superiority
- fourth:
  - advisory ecosystem comparison strong enough to inform prioritization, not
    to carry maintained product claims

## Fixed Comparison Protocol

The exact maintained comparison protocol is now fixed:

- maintained correctness lane:
  - `./build/quality-review-cmake/test_chol_csc`
  - retained external-dense-reference readings for:
    - `nos4`
    - `bcsstk04`
  - interpret agreement through:
    - `max|x-x_ref|`
    - retained in-repo residual strength
- maintained package-shape lane:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
  - interpret through:
    - pass totals
    - fail totals
    - skip totals where applicable
    - continued agreement with the maintained static-first consumer/export
      contract
- bounded runtime-reference lane:
  - `make bench-reorder-sprint86`
  - `make bench-canonical-report`
  - interpret through:
    - touched AMD-vs-ND reorder/fill/runtime evidence on the Sprint 86 slice
    - retained canonical bundle generation and reproducible branch-local
      runtime reporting
  - do not interpret this lane as:
    - broad timing proof
    - portable benchmark parity
    - automatic superiority evidence

## Comparison Classes Epic 9 May Widen

The widening order is now explicit:

- correctness-only widening:
  - widen maintained external differential proof to one or more additional
    high-value solver lanes only where deterministic, reviewable, and
    dependency-bounded
- runtime/fill/reference widening:
  - widen touched runtime and fill/reference comparisons only where they stay
    bounded, reproducible, and clearly non-superiority-oriented
- advisory ecosystem widening:
  - broader solver-stack, reorder-stack, or platform comparisons may inform
    planning, but do not become maintained product-truth owners without a
    deliberate later contract change

## Claim Strength by Evidence Class

The claim fence by evidence class is now fixed:

- strong enough for bounded product claims:
  - maintained SPD external correctness lane
  - maintained install/export and downstream-consumer package-shape lane
- strong enough for calibration/support claims only:
  - bounded reorder/runtime reference slices
  - canonical benchmark-report generation
  - asymmetric macOS/Windows workflow evidence
- not strong enough for maintained product claims today:
  - broader ecosystem comparisons not yet executed under maintained proof
  - ad hoc local timing wins
  - platform-specific speed anecdotes
  - workflow completion outside the retained reviewed and script-owned lanes

## Runtime and Benchmark Claim Fence

The runtime evidence fence is now explicit:

- Epic 9 may use runtime evidence to claim:
  - bounded improvement on touched measured lanes
  - more competitive runtime posture than the Epic 8 close baseline
  - better calibrated runtime expectations by matrix family or workflow
- Epic 9 may not use runtime evidence to claim:
  - universal speed leadership
  - cross-platform timing parity
  - benchmark pass/fail product guarantees
  - superiority outside the bounded touched corpus and measured interpretation

## Package-Shape and Workflow Comparison Contract

The package/workflow comparison reading is now explicit:

- package-shape truth is owned by:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
- workflow evidence remains layered:
  - Linux = strongest reviewed source of truth
  - macOS = narrower reviewed Apple Clang lane plus supplemental static-first
    install confidence
  - Windows = reviewed CMake-first consumer subset only
- workflow asymmetry may support truthful product-shape statements
- workflow asymmetry may not be reinterpreted as reviewed cross-platform
  parity

## Strongest Clarification

The useful Day 6 clarification is now explicit:

- Epic 9 needs stronger comparison depth, but not every comparison class
  should be promoted equally
- correctness and package-shape proof are the maintained claim-bearing lanes
- runtime evidence remains intentionally bounded and calibration-oriented
- broader ecosystem comparison remains advisory until a later explicit
  maintained contract widens it

## Exit State

- Sprint 90 now has one explicit comparison-and-measurement contract.
- Correctness, package-shape, runtime-reference, and advisory ecosystem
  evidence classes are clearly separated.
- Later sprint planning can widen evidence intentionally instead of blurring
  local measurement, maintained proof, and product claims.
