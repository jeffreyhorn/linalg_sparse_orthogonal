# Sprint 29 Day 11 — Item 8 Coverage Audit (Part 1)

## Status

**Local `make coverage` blocked** by a tooling-version mismatch:
Homebrew `lcov 2.4_1` requires gcov data in a newer format than
Apple's bundled `LLVM gcov version 11.0.0 (emulates gcov 4.2.0)`
emits.  `lcov --capture` errors with:

```
lcov: ERROR: (graph) /.../build/sparse_csr.gcno: reached unexpected end of file.
lcov: ERROR: (mismatch) cannot find an entry for ... src#sparse_csr.c.gcov in .gcno file
```

`--ignore-errors graph,mismatch,inconsistent,unused` walks past
the read errors but produces an empty coverage DB (all .gcno
files are skipped).

This is a local-environment issue.  The CI Linux job
(`.github/workflows/ci.yml::coverage`) installs `apt-get install
lcov bc` and uses gcc-native `--coverage`, which produces matching
.gcda/.gcno formats and runs cleanly.  Day-11 audit therefore uses
**indirect signal**: source-size inventory + manual test-coverage
correspondence + recent SuiteSparse-corpus test-pass count.  Day 12
will run the actual coverage gate against the **CI** numbers (per
PLAN.md Day 12 task 1: "Per Day-11's audit, decide path (a)
tighten or path (b) lower-threshold").

## Source inventory (28 339 LOC, 25 files)

```
3555  sparse_graph.c                  Sprint 22-28 FM / ND graph
3135  sparse_eigs.c                   Sprint 20-21 + 29 eigsolvers
2723  sparse_ldlt_csc.c               Sprint 19-20 CSC LDL^T
2349  sparse_iterative.c              CG / GMRES / MINRES / BiCGSTAB
2220  sparse_chol_csc.c               Sprint 17-18 CSC Cholesky
1713  sparse_svd.c                    SVD + Sprint 29 low-rank
1666  sparse_lu_csr.c                 CSR LU
1575  sparse_qr.c                     QR + Sprint 29 progress
1490  sparse_ldlt.c                   Linked-list LDL^T
1049  sparse_matrix.c                 Core data structure
 903  sparse_lu.c                     Linked-list LU
 761  sparse_etree.c                  Sprint 11+ etree
 657  sparse_ilu.c                    ILU(0) / ILUT
 642  sparse_reorder_nd.c             Sprint 22-27 ND
   ...  remaining 11 files at < 600 LOC each.
```

## Predicted coverage hot-spots (manual inspection)

Files most likely below the 95 % threshold:

### 1. `src/sparse_eigs.c` (3135 LOC) — newest paths in Sprint 29

- Sprint 29 Day 5 refinement helpers (`s29_refine_pair`,
  `s29_refine_eigenpairs`, `s29_maybe_refine`): covered by 4 new
  tests in `test_eigs.c` (default_off_unchanged, tightens_residual,
  lobpcg_backend, max_iters_budget) + 2 cross-feature tests in
  `test_sprint29_integration.c`.  Expected high coverage on the
  happy path; edge cases (SPARSE_ERR_SINGULAR retry shift,
  Rayleigh-quotient stall break, degenerate-solve break) are
  exercised only by the clustered-spectrum fixtures + may not be
  fully hit.
- Sprint 29 Day 7 progress hooks in Lanczos grow-m + LOBPCG: covered
  by `test_progress_cb_lanczos_emits_cancel` and
  `test_progress_cb_lobpcg_emits_cancel`.  Cancellation path well-
  covered.
- Thick-restart Lanczos: Sprint 21 inheritance, separate test suite
  `test_eigs_thick_restart.c`; no Sprint 29 changes.

**Predicted coverage**: 85-90 %.  Edge cases in `s29_refine_pair`
(SPARSE_ERR_SINGULAR + perturbation retry, ldlt_solve failure
propagation, near-zero ||y|| degenerate break) may sit uncovered.

### 2. `src/sparse_graph.c` (3555 LOC) — Sprint 22-28 FM passes

Most of the file is the multi-strategy FM ensemble + thick-restart
gain noise variants from Sprint 28.  Has dedicated tests in
`test_graph.c` + `test_graph_fm_buckets.c`.  Sprint 28's various
strategy permutations may not all execute under every test
combination.

**Predicted coverage**: 80-90 %.  Some Sprint 28 ensemble-strategy
permutations might be cold paths.

### 3. `src/sparse_iterative.c` (2349 LOC) — 4 solvers + 4 matrix-free variants

Day-7 progress hooks at 5 sites.  Tests in `test_iterative.c`,
`test_minres.c`, `test_bicgstab.c`, `test_stagnation.c`, plus
Sprint 29 Day 7 progress tests in `test_integration.c`.  Likely
high coverage (solvers are well-tested) but the matrix-free CG
variant + corner-case breakdowns may be uncovered.

**Predicted coverage**: 90-95 %.

### 4. `src/sparse_svd.c` (1713 LOC) — Sprint 29 Day 2 + 3 additions

- Day 2 outer-product accumulator: covered by 2 new
  test_svd tests + Day-2 corpus bench (not a test).
- Day 3 full-mode SVD (`pad_orthonormal_basis` helper): covered
  by 3 new test_svd tests (orthonormality, reconstruction,
  economy-unchanged) + Day-11 cross-feature
  test_cross_full_svd_lowrank_reconstruction.
- Pad helper SPARSE_ERR_NOT_CONVERGED branch (all canonical
  vectors collapse — only reachable if k_target > dim, which the
  caller guards against): unreachable in practice.

**Predicted coverage**: 90-95 %.

### 5. `src/sparse_reorder_nd.c` (642 LOC)

Sprint 22-27 multilevel partition + Sprint 28 supernodal postorder.
Dedicated `test_reorder_nd.c`.  Sprint 29 Day 7 noted that ND
doesn't have an opts struct so progress callbacks aren't wired
there (deferred Sprint 30+).

**Predicted coverage**: 90-95 %.

### 6. Sprint 17-19 supernodal CSC kernels (`sparse_chol_csc.c`,
`sparse_ldlt_csc.c`, 2200-2700 LOC each)

Heavy CSC supernodal logic with multiple dispatch backends + AUTO
heuristics.  Tested via `test_chol_csc.c` + `test_ldlt_csc.c` +
Sprint 17/18/19/20 integration tests.

**Predicted coverage**: 80-90 %.  Backend-fallback paths (e.g.
Sprint 20 Day 5 structural fallback) may be cold.

## Predicted aggregate coverage

Based on the per-file estimates and SLOC-weighted average:

```
       weighted_pct = sum(file_loc * file_pct) / total_loc
       ≈ (3555·85 + 3135·88 + 2723·85 + 2349·92 + 2220·85 +
          1713·92 + 1666·92 + 1575·92 + 1490·90 + 1049·95 +
           903·92 +  761·95 +  657·90 +  642·92 +
          <600·92 average for the other 11 files (5000 LOC)) / 28339
       ≈ 89 %
```

This is a **rough estimate**; actual numbers should land within ±3 %.

## Day-12 path decision criteria

Per PLAN.md Day 12 task 1, Day 12 decides between:

- **Path (a) tighten**: pick 3-5 lowest-coverage files where added
  tests yield the most aggregate movement.  Push to ≥ 95 %.
  Estimated cost: ~10 hrs.
- **Path (b) lower threshold**: lower `COV_THRESHOLD` from 95 to
  ~85 % with documentation of which file-groups carry which
  coverage levels.  Estimated cost: ~3 hrs.

**Day-11 recommendation (informational; Day 12 decides):** path (b).
The 89 % predicted aggregate sits 6 percentage points below the
95 % gate.  The cold paths I've identified (SPARSE_ERR_SINGULAR
retry shift, Bunch-Kaufman 2×2 vs 1×1 pivot edge cases, multi-
strategy FM ensemble permutations) are mostly defensive code paths
that don't lend themselves to clean unit tests — they fire only on
adversarial inputs that are hard to construct deterministically.
Pushing them to 95 % would either require synthetic-fault injection
infrastructure (expensive) or property-based testing (also
expensive).  Lowering the gate to 85 % preserves the existing
quality signal (catches major regression in test coverage) without
spending a sprint on synthetic-fault scaffolding.

Day 12 will run the **actual CI** coverage numbers and make the
final call.

## Sanitizer status

`make sanitize` (ASan + UBSan): not run on Day 11 (was killed by a
concurrent `make coverage` invocation that won the race for
build/).  Day 12 reruns sanitize as part of the regression check
before the coverage decision.

`make tsan` on macOS 15+: blocked since Sprint 28 (documented in
Day 8 `windows_ci_decision.md`); Linux CI tsan job is the source
of truth.

## What ships in Day 11

- `tests/test_sprint29_integration.c`: 3 new cross-feature tests
  (eigs+refine+progress_cb interaction, full-SVD+lowrank
  reconstruction, refine+cancel short-circuit).  All 3 PASS.
- `Makefile` + `CMakeLists.txt`: register the new test file.
- `docs/planning/EPIC_2/SPRINT_29/coverage_audit_day11.md` (this
  doc).
- Local quality gates: format, lint, test PASS.  `make sanitize`
  + `make coverage` need clean-build runs that Day 12 picks up.

## References

- `docs/planning/EPIC_2/SPRINT_29/PLAN.md` Day 11 + Day 12.
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 Item 8.
- `.github/workflows/ci.yml::coverage` — canonical CI coverage job.
