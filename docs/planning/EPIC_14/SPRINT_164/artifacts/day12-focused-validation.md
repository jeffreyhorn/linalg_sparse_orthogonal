# Sprint 164 Day 12: Focused Validation

## Purpose

Day 12 ran the focused validation gates required for the accumulated Sprint
164 public-header and documentation changes before closeout work begins.

Changed validation surface:

- public headers:
  - `include/sparse_matrix.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
- public docs:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/solver_selection.md`
- Sprint 164 planning evidence under `docs/planning/EPIC_14/SPRINT_164/`

Because public `.h` files changed, Day 12 ran the full required
`make format && make lint && make test` gate.

## Declaration Preservation

Re-ran the normalized declaration capture after the full format gate.

Day 12 post-format checksum:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

This matches the Day 4 before-state baseline:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

The before/post-format diff command produced no output:

```sh
diff -u \
  build/sprint164/declarations/selected-public-headers.before.normalized.txt \
  build/sprint164/declarations/selected-public-headers.day12-postformat.normalized.txt
```

Result: zero public declaration drift.

## Generated API Reference

`make docs-check` passed:

```text
Generating API documentation with Doxygen...
doxygen Doxyfile
Documentation generated in docs/api/html/
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
```

Result: generated Doxygen coverage remains complete for checked-in public
headers, and `sparse_version.h` remains governed by the separate installed
header policy.

## Required C/Header Gate

The required public-header quality chain passed:

```sh
make format && make lint && make test
```

Observed result:

- `make format` completed.
- `make lint` completed, including strict syntax checks, `clang-tidy`, and
  `cppcheck`.
- `make test` completed with `All tests passed.`

## Documentation And Claim Scans

Scoped stale-reference scan passed with no hits:

```sh
rg -n "sparse_eigs_result_t|sparse_eigs_sym\\(A, k, &opts, &result\\).*growing-m|via Lanczos \\(growing-m" \
  README.md docs/tutorial.md docs/cookbook.md docs/solver_selection.md docs/api_reference.md
```

Scoped unsupported-claim scan found only explicit disclaimers and maintainer
policy boundaries for package, ABI, platform, external-parity, hosted-docs,
portable-performance, backend-superiority, and state-of-the-art claims.

## Generated And Local Evidence Status

Local generated evidence stayed untracked:

- `git status --short -- docs/api/html build/sprint164/declarations`

Result: no source-controlled generated HTML or local declaration-evidence churn.

## Diff Hygiene

`git diff --check` passed.

## Outcome

All required Day 12 validation gates passed. No fixes or blockers were found.
