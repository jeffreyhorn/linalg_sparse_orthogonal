# Sprint 164 Day 11: Documentation Coherence Pass

## Purpose

Day 11 reviewed public documentation touched by or referenced from the selected
public-header cleanup batch and aligned stale API terms with the current
headers.

Selected headers:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Reviewed documentation surfaces:

- `README.md`
- `docs/tutorial.md`
- `docs/cookbook.md`
- `docs/solver_selection.md`
- `docs/api_reference.md`
- `docs/maintainer_guide.md`

## Documentation Updates

### README Eigensolver Summary

Updated the symmetric eigensolver API summary so
`sparse_eigs_sym(A, k, &opts, &result)` is no longer described as only a
grow-m Lanczos path.

The README now matches `include/sparse_eigs.h` and
`docs/solver_selection.md` by describing the public backend surface as:

- AUTO;
- grow-m Lanczos;
- thick-restart Lanczos;
- explicit LOBPCG.

It also now mentions `result.backend_used` alongside
`result.used_csc_path_ldlt` so the README API summary points users to the same
bounded routing telemetry described in the header.

### Tutorial Eigensolver Type Name

Corrected the tutorial's symmetric eigensolver example from the stale
`sparse_eigs_result_t` name to the public declaration name:

```c
sparse_eigs_t result = {
    .eigenvalues = eigenvalues,
    .eigenvectors = eigenvectors,
};
```

This aligns the runnable tutorial snippet with `include/sparse_eigs.h`.

## Surfaces Reviewed Without Edits

- `docs/cookbook.md`
  - Already starts eigensolver users with AUTO.
  - Already routes exact option, result, backend, and handle details to
    `include/sparse_eigs.h`.
  - Already excludes broad external-library, performance, platform, package,
    ABI, and state-of-the-art claims.
- `docs/solver_selection.md`
  - Already reflects the Day 8 eigensolver AUTO routing and workflow links.
  - Already distinguishes AUTO routing policy from backend-superiority claims.
- `docs/api_reference.md`
  - Already points exact declarations and call-site contracts to checked-in
    public headers.
  - Already retains generated HTML as local-only output and excludes broad
    ABI/package/platform/parity claims.
- `docs/maintainer_guide.md`
  - Already preserves the public-header cleanup policy and generated-reference
    non-claim boundaries.

## Non-Claim Trace

The documentation coherence pass preserved existing exclusions for:

- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad Windows Makefile or Windows `pkg-config` parity;
- external-library parity;
- portable runtime or performance guarantees;
- hosted generated documentation publication;
- source-controlled generated HTML;
- backend superiority;
- state-of-the-art coverage.

## Deferred Work

No selected-header declaration changes were needed.

The following remain outside Day 11 scope:

- broader non-selected-header API reference cleanup;
- generated HTML publication work;
- new backend tuning or performance claims;
- package/ABI product changes;
- exhaustive tutorial expansion for every option/result field.

## Validation

- Scoped stale-type scan confirmed `sparse_eigs_result_t` no longer appears in
  README/tutorial/cookbook/solver-selection/API-reference docs.
- Scoped stale-backend scan confirmed the README API summary no longer
  describes `sparse_eigs_sym` as only a grow-m Lanczos path.
- Scoped non-claim scan confirmed unsupported package, ABI, platform, external
  parity, hosted-docs, and state-of-the-art wording remains bounded to
  explicit disclaimers.
- `git diff --check`
- trailing whitespace scan over Sprint 164 planning docs
