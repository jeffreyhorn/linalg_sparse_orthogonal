# Sprint 184 Day 11: Mechanical Guard Implementation

## Purpose

Add focused mechanical protection for the selected Sprint 184 QR public-header
cleanup. The guard protects the organized `include/sparse_qr.h` section model,
selected QR declaration presence, and QR-facing documentation alignment without
making broader API, package, platform, performance, or parity claims.

## Files Added or Updated

| File | Change |
| --- | --- |
| `scripts/check_qr_header_docs_guard.sh` | New focused QR header/docs guard. |
| `Makefile` | Added `qr-header-docs-guard` target. |
| `docs/planning/EPIC_16/SPRINT_184/WORKING_NOTES.md` | Added Day 11 guard notes and validation. |
| `docs/planning/EPIC_16/SPRINT_184/artifacts/day11-mechanical-guard-implementation.md` | This artifact. |

## Guard Coverage

The guard checks:

- QR header section headings:
  - `Options and factor object`;
  - `Factorization and lifecycle`;
  - `Q operations`;
  - `Solve operations`;
  - `Rank, nullspace, and diagnostics`.
- Required QR declaration tokens:
  - `sparse_qr_opts_t`;
  - `sparse_qr_t`;
  - `sparse_qr_factor()`;
  - `sparse_qr_factor_opts()`;
  - `sparse_qr_free()`;
  - `sparse_qr_apply_q()`;
  - `sparse_qr_form_q()`;
  - `sparse_qr_solve()`;
  - `sparse_qr_refine()`;
  - `sparse_qr_solve_minnorm()`;
  - `sparse_qr_refine_minnorm()`;
  - `sparse_qr_rank()`;
  - `sparse_qr_nullspace()`;
  - `sparse_qr_diag_r()`;
  - `sparse_qr_rank_info_t`;
  - `sparse_qr_rank_info()`;
  - `sparse_qr_condest()`.
- Unsupported claim absence in the QR header for raw QR basis parity, global
  rank-threshold policy, broad rank-deficient solve behavior, broad
  minimum-norm behavior, external-library parity, package/ABI support,
  portable performance, Windows report freshness, and state-of-the-art claims.
- QR-facing docs alignment across README, API reference, cookbook,
  solver-selection, tutorial, and examples README.

## Focused Output

Command:

```sh
make qr-header-docs-guard
```

Output:

```text
qr-header-docs-guard: header sections ok
qr-header-docs-guard: header declarations ok
qr-header-docs-guard: header unsupported claim absence ok
qr-header-docs-guard: docs alignment ok
qr-header-docs-guard: passed
```

## Validation

- `bash -n scripts/check_qr_header_docs_guard.sh`: passed.
- `make qr-header-docs-guard`: passed.
- `make docs-check`: passed.
- `make api-docs-local-only`: passed.
- `git diff --check`: passed.
- Sorted comment-stripped QR declaration-set diff against `HEAD`: passed with
  no output.

Day 11 made no new `.c` or `.h` edits, so the full C quality gate was not
rerun for this day.

## Day 12 Handoff

Day 12 should include `make qr-header-docs-guard` in the focused validation
bundle alongside docs checks, API local-only checks, QR declaration-set checks,
and any example checks required by the final changed surface.
