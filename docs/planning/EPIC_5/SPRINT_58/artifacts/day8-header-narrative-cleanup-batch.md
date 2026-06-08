# Sprint 58 Day 8 - header narrative cleanup batch

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Land the first bounded public-header cleanup batch by reducing stale
sprint-history and future-work wording in the highest-value public headers,
aligning the repeated-run and eigensolver public story with the simplified
README/tutorial wording, and preserving the stable API/lifecycle/support
contract.

## Touched headers

Primary landed set:

- `include/sparse_eigs.h`
- `include/sparse_iterative.h`

Intentionally deferred:

- `include/sparse_analysis.h`
- direct-family headers:
  - `include/sparse_cholesky.h`
  - `include/sparse_lu.h`
  - `include/sparse_ldlt.h`

## Landed changes

### `include/sparse_eigs.h`

The batch:

- removed stale sprint chronology from the file header, backend selector docs,
  threshold docs, option comments, result comments, and repeated-run handle
  summary
- removed stale future-work wording such as planned backend language and
  sprint-local follow-up notes
- shortened several overlong caller-facing explanations while keeping the
  important behavioral and ABI-compatibility facts
- normalized the public repeated-run handle wording so it now reads as one
  stable explicit lifecycle surface for:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit LOBPCG

Preserved contract:

- supported eigensolver backends unchanged
- AUTO routing semantics unchanged
- shift-invert / LDL^T composition unchanged
- repeated-run handle semantics unchanged
- ABI warnings preserved

### `include/sparse_iterative.h`

The batch:

- removed the remaining sprint-local framing from the progress/cancellation
  callback comments
- kept the explicit repeated-run handle wording stable for:
  - `CG`
  - `GMRES`
  - `MINRES`
- avoided a broader rewrite because the main repeated-run lifecycle section was
  already close to the new README/tutorial wording

Preserved contract:

- iterative handle semantics unchanged
- support boundary unchanged
- one-shot solver posture unchanged

## Measured result

Touched-surface line counts:

- `include/sparse_eigs.h`: `687 -> 646`
- `include/sparse_iterative.h`: `765 -> 765`
- `include/sparse_analysis.h`: unchanged at `375`

Diff shape:

- `2` files changed
- `77` insertions
- `118` deletions

## Validation

Because public headers changed, the required gate was run:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

## Conclusion

The Day 8 batch stayed inside the planned fence:

- it cleaned the strongest public-header narrative offender first
- it aligned the repeated-run iterative summary wording without widening scope
- it preserved the stable API/lifecycle/support contract
- it deferred lower-value header surfaces rather than turning the batch into a
  repo-wide wording sweep
