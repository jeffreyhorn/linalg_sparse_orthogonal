# Sprint 44 Day 13: Full Validation Sweep

## Summary

Day 13 ran the full Sprint 40/41/42 validation anchor against the Sprint 44
Phase-2 graph decomposition and first large-test helper consolidation batch.

The result is clean:

- the mandatory code-change floor passed
- `make quality-review-full` passed
- reviewed CMake parity remained exact at `53`
- direct graph / ND and touched-QR reruns passed

No reconciliation queue surfaced.

## Validation Runs

### Mandatory floor

- `make format` → passed, `real 8.12`
- `make lint` → passed, `real 348.41`
- `make test` → passed, `real 96.91`

### Strong reviewed proof

- `make quality-review-full` → passed, `real 797.34`

Included reviewed-path components:

- reviewed Makefile path
- `deadcode-check`
- reviewed clean CMake rebuild
- `ctest -N`
- full reviewed CMake `ctest`

### Direct graph / ND and QR reruns

- `./build/test_graph` → passed, `Time: 2.228 s`
- `./build/test_graph_fm_buckets` → passed, `Time: 0.001 s`
- `./build/test_reorder_nd` → passed, `Time: 29.940 s`
- `./build/test_reorder_amd_qg` → passed, `Time: 0.310 s`
- `./build/test_qr` → passed, `Time: 2.113 s`

## Truthfulness Anchors

The preserved Sprint 40 validation/truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 194.88 sec`

Interpretation:

- Sprint 44 did not disturb the maintained local reviewed baseline
- the graph Phase-2 split and QR helper cleanup remain aligned with the
  established Makefile/CMake parity contract

## Sprint-44-Specific Outcome

The direct reruns matter because Sprint 44 changed two different high-risk
surfaces:

- the residual graph / ND subsystem:
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - residual orchestration in `src/sparse_graph.c`
- the first large-test maintainability seam:
  - local helper consolidation in `tests/test_qr.c`

The reruns confirm those touched surfaces still compose cleanly across:

- FM refinement
- separator-policy lifting
- top-level partition orchestration
- ND integration
- quotient-graph AMD integration
- QR reconstruction and solve validation paths

The explicit Sprint 44 seam protections stayed green:

- `test_edge_to_vertex_separator_balanced_boundary_prefers_smaller_boundary`
- `test_partition_fifo_balanced_boundary_smoke`
- `test_qr_rejects_factored_matrix_reuse`
- the Day 12 helper-adopted reconstruction and solve checks in `test_qr`

## Caveats

No new caveats surfaced beyond the maintained contract:

- dead-code execution remains serialized
- reviewed CMake remains the strongest shared reviewed baseline
- a standalone serial `make deadcode-report` rerun was not added for Day 13
  because Sprint 44 did not change dead-code scripts, reporting semantics, or
  dead-code Makefile wiring, and `deadcode-check` still passed inside
  `make quality-review-full`

## Outcome

Sprint 44 now enters Day 14 closeout from a measured, validated Phase-2 state:

- extracted FM refinement seam validated
- extracted separator-policy seam validated
- residual orchestration cleanup validated
- focused graph seam tests validated
- first large-test helper consolidation validated

That is the correct end-state before Sprint 44 closeout and handoff.
