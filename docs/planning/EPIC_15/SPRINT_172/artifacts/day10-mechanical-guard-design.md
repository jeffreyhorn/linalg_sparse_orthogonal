# Sprint 172 Day 10: Mechanical Guard Design

## Purpose

Design lightweight guard coverage for the selected Sprint 172 public header,
`include/sparse_lu.h`, and the Day 9 LU tutorial alignment.

Day 10 is design-only. It does not add scripts, Make targets, or code.

## Existing Guard Patterns Reviewed

| Pattern | Existing source | Applicability |
| --- | --- | --- |
| Shell non-claim guard | `scripts/package_manager_deferral_check.sh` | Good fit for simple required/forbidden text checks with clear failure output. |
| Shell static-package guard | `scripts/static_package_deferral_check.sh` | Good model for claim-boundary checks, but too broad for LU header/docs drift. |
| Python report/index checks | `scripts/check_api_docs_coverage.py`, `scripts/normalize_report_index.py` | Useful for structured report data, but heavier than needed for fixed header/tutorial checks. |
| Make quality targets | `make docs-check`, `make quality-review`, `make quality-review-full` | Possible wiring points after focused guard behavior is proven. |

The recommended Day 11 guard should be a small shell script because it only
needs deterministic text-presence and text-absence checks.

## Recommended Guard

Implement a focused shell guard:

```text
scripts/check_lu_header_docs_guard.sh
```

Recommended invocation:

```sh
bash scripts/check_lu_header_docs_guard.sh
```

Recommended failure prefix:

```text
lu-header-docs-guard: FAIL: <specific reason>
```

Recommended success lines should name each checked area:

```text
lu-header-docs-guard: header sections ok
lu-header-docs-guard: tutorial refinement signature ok
lu-header-docs-guard: unsupported claim absence ok
```

## Positive Checks

The guard should require `include/sparse_lu.h` to contain the workflow section
headings added in Day 7:

- `/* Options */`
- `/* Factorization */`
- `/* Solves */`
- `/* Conditioning and transpose solves */`
- `/* Advanced solver phases */`
- `/* Refinement */`

The guard should require the selected public declaration names to remain
present in `include/sparse_lu.h`:

- `sparse_lu_factor_opts`
- `sparse_lu_factor`
- `sparse_lu_solve`
- `sparse_lu_solve_block`
- `sparse_lu_condest`
- `sparse_lu_solve_transpose`
- `sparse_apply_row_perm`
- `sparse_apply_inv_col_perm`
- `sparse_forward_sub`
- `sparse_backward_sub`
- `sparse_lu_refine`

The guard should require `docs/tutorial.md` to contain the six-argument LU
refinement snippet:

```text
sparse_lu_refine(A, LU, b, x, 3, 1e-15)
```

## Negative Checks

The guard should reject a stale five-argument tutorial LU refinement snippet:

```text
sparse_lu_refine(A, LU, b, x, 3)
```

Use a pattern that does not false-positive on the valid six-argument call. A
simple extended-regex shape is:

```sh
grep -Eq 'sparse_lu_refine\(A, LU, b, x, 3\);' docs/tutorial.md
```

The guard should also reject unsupported claim wording in the selected header
and the edited tutorial LU section:

- `package-manager support`
- `shared-library support`
- `dynamic ABI`
- `runtime-loader`
- `broad Windows parity`
- `Windows Makefile parity`
- `Windows pkg-config parity`
- `external-library parity`
- `portable performance`
- `performance guarantee`
- `LU CSR parity`
- `state-of-the-art`

For Day 11, restrict the tutorial negative scan to the LU section or to the
touched snippet neighborhood. The full tutorial already contains intentional
non-claim language that mentions some of these terms; a whole-file negative
scan would fail on valid boundary text.

## Declaration Behavior Boundary

This guard must not infer ABI or API behavior from comments. It should only
prove that the selected header still exposes the expected declaration names
and that the Day 7 headings remain present.

Day 11 should not attempt to parse C signatures with ad hoc shell regex beyond
presence checks. If future work needs exact signature checking, use a separate
structured parser or compile-based check.

## Makefile Wiring Decision

Day 11 should first implement and run the guard directly. Makefile wiring can
be added only if the script is narrow, fast, deterministic, and does not depend
on generated files.

Preferred target if wired:

```make
.PHONY: lu-header-docs-guard
lu-header-docs-guard:
	@bash scripts/check_lu_header_docs_guard.sh
```

Do not wire this into `lint` or `test` during Day 11 unless the implementation
artifact explicitly records why the new guard is stable enough for the broader
quality path.

## Day 11 Validation Commands

Expected Day 11 validation:

```sh
bash scripts/check_lu_header_docs_guard.sh
git diff --check
```

If Day 11 edits `.c` or `.h` files, also run:

```sh
make format && make lint && make test
```

If Day 11 edits package/adoption/ABI/platform wording, also run:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
```

## Completion Status

Day 10 is complete. Guard behavior is scoped before implementation, the guard
cannot silently change API behavior, and unsupported package/ABI/platform
claims remain protected by explicit positive and negative checks.

## Day 11 Handoff

Implement `scripts/check_lu_header_docs_guard.sh` as a focused shell guard.
Check the LU header section headings, required LU declaration names, the
six-argument tutorial refinement snippet, and the scoped unsupported-claim
absence rules.
