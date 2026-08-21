# Sprint 174 Day 10: Freshness Gate Implementation

## Purpose

Close the selected comparison freshness gate for the newly added
`lu-nonsym-square-5` family by adding source-controlled negative-test coverage
and proving the positive Make-owned regeneration path remains repeatable.

## Freshness Checker Updates

Day 9 added the LU row IDs and artifact path to the production freshness
selection. Day 10 extended the focused normalizer tests so the selected
comparison gate is covered by source-controlled positive and negative cases
with LU included.

Updated `tests/test_normalize_report_index.py`:

- added the six `comparison_lu_nonsym_square_5_*` row IDs to the synthetic
  selected comparison fixture set;
- added `build/comparison/lu_nonsym_square_5/study.tsv` to the expected
  artifact diagnostic;
- added `SELECTED_LU_COMPARISON_ROW_IDS`;
- taught the synthetic selected-comparison writer to emit
  `lu_nonsym_square_5` rows into
  `build/comparison/lu_nonsym_square_5/study.tsv`;
- added positive assertions that LU rows normalize with pass status,
  `local_only` support tier, the expected artifact path, and bounded
  `no broad LU correctness` non-claim wording;
- changed the missing-row mismatch proof to drop
  `comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1`, proving
  LU-selected row loss fails closed.

No schema changes were required. The existing selected comparison freshness
logic already rejects missing, stale, duplicate, non-pass, skip/defer, and row
set mismatch cases; Day 10 makes the LU family part of that tested selected
surface.

## Positive Gate

The owning proof command remains:

```text
make report-index-comparison-freshness
```

It regenerates all selected comparison artifacts:

```text
build/comparison/qr_minnorm/study.tsv
build/comparison/qr_compatible_ls/study.tsv
build/comparison/partial_svd_diag6_k2/study.tsv
build/comparison/lu_nonsym_square_5/study.tsv
```

and then runs:

```text
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
```

The positive run completed with:

```text
normalize-report-index: freshness ok (32 rows)
report-index-comparison-freshness: passed (local-only generated comparison freshness)
```

## Negative Proof

Ran a controlled missing-artifact check with an empty temporary build root.
The command intentionally failed closed and included the LU artifact in the
remediation diagnostic:

```text
required generated family missing: comparison
artifacts=build/comparison/qr_minnorm/study.tsv,build/comparison/qr_compatible_ls/study.tsv,build/comparison/partial_svd_diag6_k2/study.tsv,build/comparison/lu_nonsym_square_5/study.tsv
run make report-index-comparison-freshness
```

The focused normalizer test suite also covers:

- complete selected comparison row set acceptance;
- selected row-set mismatch rejection;
- duplicate selected row rejection;
- stale selected row rejection;
- failed selected row rejection;
- skip/defer selected row rejection.

## Claim Boundary

The freshness gate remains fixture-local. It proves only that the selected
generated comparison rows for the named artifacts are present, pass, and match
the current source commit. It does not promote broad LU correctness,
nonsymmetric solve parity, LU CSR parity, external-library parity, hosted CI,
release, package-manager, ABI, performance, or state-of-the-art claims.

## Validation

Commands run:

```text
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
make report-index-comparison-freshness
python3 - <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix='sparse-comparison-missing-') as tmp:
    build_root = Path(tmp) / 'build'
    result = subprocess.run(
        [
            sys.executable,
            'scripts/normalize_report_index.py',
            '--build-root',
            str(build_root),
            '--family',
            'comparison',
            '--require-generated',
            'comparison',
            '--check-freshness',
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        raise SystemExit('expected missing-artifact freshness failure')
    for needle in (
        'required generated family missing: comparison',
        'build/comparison/lu_nonsym_square_5/study.tsv',
        'run make report-index-comparison-freshness',
    ):
        if needle not in result.stdout:
            raise SystemExit(f'missing expected diagnostic: {needle}')
PY
git diff --check
```

All passed.

No `.c` or `.h` files were modified. The full C quality gate is not required
for Day 10.
