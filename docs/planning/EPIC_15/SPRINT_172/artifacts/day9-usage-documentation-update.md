# Sprint 172 Day 9: Usage Documentation Update

## Purpose

Implement the Day 8 usage-documentation plan for the cleaned
`include/sparse_lu.h` contract.

## Edited Documentation

- `docs/tutorial.md`

## Change Made

Updated the LU tutorial refinement snippet from the stale five-argument call:

```c
sparse_lu_refine(A, LU, b, x, 3);
```

to the public header signature with an explicit convergence tolerance:

```c
sparse_lu_refine(A, LU, b, x, 3, 1e-15);
```

This aligns the tutorial with `include/sparse_lu.h`, `README.md`, and the test
suite.

## Behavior Preservation

Day 9 changed documentation only. No `.c` or `.h` files were edited for Day 9,
and no examples, tests, declarations, signatures, package metadata, generated
API output, or build files were changed.

## Signature Scan

Ran:

```sh
rg -n "sparse_lu_refine\([^\n]*\)" README.md docs/tutorial.md include/sparse_lu.h examples tests
```

The scan now shows the tutorial, README, checked-in header, and tests using
the six-argument `sparse_lu_refine(A, LU, b, x, max_iters, tol)` shape where
they call the API.

## Claim Scan

Ran:

```sh
rg -n "state-of-the-art|external-library parity|portable performance|performance guarantee|package-manager support|shared-library support|dynamic ABI|runtime-loader|broad Windows parity|Windows Makefile parity|Windows pkg-config parity|LU CSR parity" docs/tutorial.md
```

The scan found only existing non-claim boundary language in the tutorial:

- local evidence is not portable performance or platform proof;
- benchmark evidence is not a portable performance guarantee;
- examples do not claim parity, package/platform/ABI support, or
  state-of-the-art status;
- eigensolver docs do not claim portable state-of-the-art parity.

Day 9 did not add package-manager, shared-library, dynamic ABI,
runtime-loader, broad platform, portable performance, external-library parity,
LU CSR parity, or state-of-the-art claims.

## Validation

- `git diff --check` passed.
- Targeted LU refinement signature scan passed.
- Targeted tutorial unsupported-claim scan found retained non-claim wording
  only.

Because Day 9 changed documentation only, the full C quality gate was not
required for this day.

## Completion Status

Day 9 is complete. The tutorial LU workflow now matches the cleaned public
header contract and remains bounded to documentation usage alignment.

## Day 10 Handoff

Day 10 can design a lightweight guard that checks the selected LU header
sections and guards the tutorial LU refinement snippet against regressing to a
five-argument call.
