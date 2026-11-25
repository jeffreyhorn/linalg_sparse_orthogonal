# Day 14 Final Quality Decision and Sprint 228 Handoff

## Final completion decision

Sprint 227 is **complete**. The implementation satisfies the frozen contract, the independent
oracle and failure matrix pass, and every required local closeout gate passes on the final source
tree. No unresolved Sprint 227 blocker remains.

The completion decision is capability-scoped. It does not promote the branch to a release
candidate and does not broaden the [Day 13 claim ledger](day13-claim-ledger.md).

## Capability accepted

- `sparse_add` validates pointer, shape, and original-state requirements before allocation, merges
  sorted rows, and links retained output nodes directly through one row tail and per-column tails.
- `sparse_add`, `sparse_add_inplace`, and `sparse_matmul` share the same unfactored identity-map
  requirement and return `SPARSE_ERR_BADARG` for unsupported state at the frozen precedence point.
- Out-of-place ownership and cleanup, in-place alias rejection and partial-failure state, matmul
  stale-output behavior, and downstream Cholesky fixtures match their documented contracts.
- Public declarations and layouts remain unchanged. Both installed index widths and Make/CMake
  package consumers pass positive and negative compatibility checks.

## Exact final validation packet

The full commands and aggregate outcomes are indexed in
[the closeout evidence index](day14-closeout-evidence-index.md). The decisive paths were:

```text
make format
make lint
make test
make QUALITY_REVIEW_CMAKE_DIR=build/s227-day14-cmake quality-review-cmake
fresh CMake ASan/UBSan build + focused arithmetic/matmul/Sprint 4 CTests
bash tests/test_make_idx_width.sh
bash tests/test_install.sh
bash tests/test_cmake_idx_width_functional.sh
bash tests/test_cmake_install.sh
bash tests/test_cmake_idx_width_install.sh
bash tests/test_pkgconfig_idx_width_install.sh
bash tests/test_installed_width_check.sh
python3 tests/test_static_install_contract.py
make format-check
make source-list-check
make docs-check
strict Day 12 profile revalidation and SHA-256 manifest checks
```

Every command exited zero. The final focused rerun passed 95 tests and 451,978 assertions. The
full CMake lane passed 63/63 tests in 136.28 seconds.

## Handoff to Sprint 228

Sprint 228 may rely on these closed properties:

1. The original-state guard and its pointer/shape/state precedence are shared, tested seams for
   sparse arithmetic entry points.
2. The sorted-output builder demonstrates one-output ownership, direct row/column publication,
   injectable allocation boundaries, and finish/abort cleanup in `src/sparse_matrix.c`.
3. `tests/arithmetic_contract_fixture.h` contains independent dense-oracle, snapshot, map, and
   cross-link helpers that may be reused when their assumptions fit reconstruction output.
4. The Day 11–13 measurement workflow demonstrates pre-registration, correctness-qualified
   samples, raw retention, strict parsing, hash manifests, and narrow claim language.

Sprint 228 must not assume that the file-local addition builder is a generic public builder or
copy it without re-proving its ordering and ownership assumptions. It must independently freeze
low-rank reconstruction thresholds, output ordering, allocation lifecycle, scratch accounting,
and dense-reference tolerances. Sprint 227 timing and memory observations do not transfer to SVD
reconstruction.

## Retained boundaries

- Keep rejection behavior for nonidentity arithmetic until a separate logical-view design is
  specified and tested.
- Keep the in-place partial-update warning visible; do not relabel it transactional.
- Preserve the 300-file Day 12 bundle and both historical/current source manifests unchanged.
- Treat hosted results after commit as additional platform evidence. Any failure must be repaired
  before merge; it does not retroactively authorize a narrower unsupported claim.
