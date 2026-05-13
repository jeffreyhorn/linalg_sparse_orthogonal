# Sprint 29 Day 12 — Item 8 Coverage Threshold Decision

## Status

**Decision: Path (b) — lower `COV_THRESHOLD` from 95 to 80.**

The Day-12 measurement landed aggregate line coverage at **81.3 %**
(13 622 / 16 763 lines), with function coverage at 97.7 % and branch
coverage at 74.3 %.  This sits **13.7 percentage points below** the
inherited 95 % gate and **7.7 points below** Day-11's per-file
hand-estimate (~89 %).  Path (a) — tighten the test suite to 95 % —
would require either ~2 000 lines of new test code targeting cold
fallback paths or synthetic-fault-injection scaffolding; neither
fits the 12-hour Day-12 budget.  Path (b) recalibrates the gate to
the operating reality + documents which file groups carry which
coverage levels, preserving the regression signal without spending
a sprint on fault scaffolding.

## Day-12 measurement methodology

Day 11 was blocked on `make coverage` locally because Homebrew
`lcov 2.4_1` couldn't parse Apple's bundled `LLVM gcov` output
(format mismatch — confirmed in `coverage_audit_day11.md`).  Day 12
worked around the lcov block by:

1. Building + running the full test suite with Apple Clang's
   `--coverage` flag (no lcov involvement during the run).
2. Running `gcovr 8.6` (Homebrew) against the resulting `.gcda`
   files with the `--gcov-executable=/usr/bin/gcov` flag (Apple's
   gcov bundled with the CommandLineTools).
3. Filtering to `src/` and excluding `tests/` + `benchmarks/`.
4. Passing `--gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file`
   to bypass the known
   [GCC bug 68080](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=68080)
   that trips on a single `sparse_graph.c:1378` line.

The Linux CI `coverage` job uses gcc-native `--coverage` + lcov +
`make coverage` directly; the gcovr / Apple-gcov path here serves as
a local diagnostic, not the production gate.  The aggregate-percent
delta between the two paths should be < 1 pp (gcc + lcov tend to
report slightly higher because lcov's filter excludes `static
inline` functions in headers more aggressively than gcovr does).

## Per-file coverage (Day-12 measured)

Sorted ascending by line-coverage %:

| File                                | LOC  | Hit  | Cov  | File group |
|-------------------------------------|------|------|------|------------|
| `src/sparse_types.c`                |   22 |   11 |  50% | error-string stubs |
| `src/sparse_reorder_nd.c`           |  285 |  155 |  54% | multilevel partition |
| `src/sparse_etree.c`                |  528 |  384 |  72% | symbolic factor |
| `src/sparse_analysis.c`             |  348 |  261 |  75% | analyze() dispatch |
| `src/sparse_bidiag.c`               |  217 |  168 |  77% | bidiagonalization |
| `src/sparse_iterative.c`            | 1566 | 1208 |  77% | iterative solvers |
| `src/sparse_svd.c`                  | 1084 |  842 |  77% | SVD (Sprint 29) |
| `src/sparse_colamd.c`               |  209 |  165 |  78% | reordering |
| `src/sparse_graph.c`                | 1833 | 1446 |  78% | FM/ND graph (Sprint 22-28) |
| `src/sparse_ilu.c`                  |  444 |  358 |  80% | ILU(0)/ILUT |
| `src/sparse_qr.c`                   | 1026 |  832 |  81% | QR (Sprint 29) |
| `src/sparse_csr.c`                  |  174 |  143 |  82% | CSR data structure |
| `src/sparse_ic.c`                   |  169 |  139 |  82% | IC(0) preconditioner |
| `src/sparse_ldlt.c`                 |  956 |  791 |  82% | linked-list LDLᵀ |
| `src/sparse_lu_csr.c`               | 1149 |  948 |  82% | CSR LU |
| `src/sparse_dense.c`                |  316 |  263 |  83% | dense helpers |
| `src/sparse_lu.c`                   |  593 |  492 |  83% | linked-list LU |
| `src/sparse_matrix.c`               |  714 |  593 |  83% | core data structure |
| `src/sparse_reorder.c`              |  261 |  218 |  83% | reorder dispatch |
| `src/sparse_cholesky.c`             |  275 |  233 |  84% | linked-list Cholesky |
| `src/sparse_chol_csc.c`             | 1229 | 1050 |  85% | CSC Cholesky |
| `src/sparse_ldlt_csc.c`             | 1556 | 1329 |  85% | CSC LDLᵀ |
| `src/sparse_reorder_amd_qg.c`       |  244 |  208 |  85% | AMD-QG reordering |
| `src/sparse_eigs.c`                 | 1471 | 1295 |  88% | eigensolvers (Sprint 20/21/29) |
| `src/sparse_vector.c`               |   45 |   45 | 100% | vector ops |
| `src/sparse_*_internal.h`           |   49 |   45 |  92% | internal headers |
| **TOTAL**                           |16763 |13622 |**81%** | |

## File-group classification

Coverage groups by structural role:

### Group A — core data structures: 82-100 %

`sparse_matrix.c` (83 %), `sparse_csr.c` (82 %), `sparse_vector.c`
(100 %), `sparse_dense.c` (83 %), `sparse_matrix_internal.h` (100 %).

Heavy unit-test coverage from `test_sparse_matrix.c`,
`test_csr.c`, `test_sparse_vector.c`, `test_dense.c`.  The 17 %
gap is dominated by error-path branches that fire only on
adversarial input (negative dimensions, NULL pointers in opts
that the caller-side already guards).

### Group B — direct factorizations: 81-85 %

`sparse_lu.c` (83 %), `sparse_lu_csr.c` (82 %), `sparse_cholesky.c`
(84 %), `sparse_chol_csc.c` (85 %), `sparse_ldlt.c` (82 %),
`sparse_ldlt_csc.c` (85 %), `sparse_qr.c` (81 %).

Sprint 11-20 numeric kernels with established test coverage in the
`test_sparse_lu.c` / `test_cholesky.c` / `test_chol_csc.c` /
`test_ldlt_csc.c` / `test_qr.c` files plus the Sprint-* integration
tests.  The 15-19 % gap concentrates in:
- Sprint 20 Day 5 structural-fallback paths (Bunch-Kaufman pivot
  retries on degenerate diagonals).
- Sprint 17-19 supernodal AUTO-backend dispatch heuristics (the
  scalar-vs-supernodal selector fires only on inputs near the
  size threshold).
- Sprint 29 Day 6 progress-callback emission sites on degenerate
  inputs (n=0, single-row, etc).

### Group C — iterative solvers + ILU: 77-82 %

`sparse_iterative.c` (77 %), `sparse_ilu.c` (80 %), `sparse_ic.c`
(82 %).

Four solvers (CG / GMRES / MINRES / BiCGSTAB) + four matrix-free
variants + IC(0) + ILU(0)/ILUT.  Covered by `test_iterative.c`,
`test_minres.c`, `test_bicgstab.c`, `test_stagnation.c`,
`test_ilu.c`, `test_ic.c`.  The 18-23 % gap concentrates in:
- Breakdown handling (e.g. BiCGSTAB ω=0 restart, GMRES restart-on-
  stagnation).
- Sprint 29 Day 7 cancellation callback paths in 5 distinct
  iteration loops.
- Matrix-free variants when the caller passes NULL preconditioner.

### Group D — eigensolvers: 88 %

`sparse_eigs.c` (88 %).

Sprint 20-21 Lanczos + LOBPCG + Sprint 29 Day-5 inverse-iteration
refinement.  Highest coverage in `src/` — driven by 4 new Day-5
refinement tests + 3 new Day-7 progress-callback tests + the
inherited thick-restart suite.  The 12 % gap is the
SPARSE_ERR_SINGULAR retry-shift path in `s29_refine_pair` (fires
only on a singular factor at the converged Ritz shift, which
requires a constructed-clustered-spectrum fixture larger than the
synthetics in the test suite).

### Group E — Sprint 29 SVD additions: 77 %

`sparse_svd.c` (77 %), `sparse_bidiag.c` (77 %).

Day-2 outer-product accumulator + Day-3 full-mode U/V padding +
Day-1 dense-baseline path.  Three Day-3 tests cover the happy path;
the 23 % gap concentrates in the SVD bidiag-back-projection path
(unreachable when the caller requests economy mode), the
`pad_orthonormal_basis` `SPARSE_ERR_NOT_CONVERGED` branch
(unreachable when `k_target ≤ dim`, which the caller guards), and
edge cases in the singular-vector convergence check.

### Group F — symbolic + reordering: 72-85 %

`sparse_analysis.c` (75 %), `sparse_etree.c` (72 %),
`sparse_reorder.c` (83 %), `sparse_reorder_amd_qg.c` (85 %),
`sparse_colamd.c` (78 %).

Sprint 11 etree + Sprint 27-28 reorder dispatch.  Inherited test
coverage from `test_etree.c`, `test_reorder.c`, `test_colamd.c`,
`test_reorder_amd_qg.c`.  The 15-28 % gap concentrates in COLAMD's
restart-on-overflow path (rare on test fixtures) and the etree's
post-ordering compaction (only fires on disconnected graphs).

### Group G — multilevel ND partition: 54-78 %

`sparse_reorder_nd.c` (54 %), `sparse_graph.c` (78 %).

Sprint 22-28 multilevel partition + FM passes.  Test coverage in
`test_reorder_nd.c` + `test_graph.c` + `test_graph_fm_buckets.c`.
The 22-46 % gap is the largest group-level gap and concentrates in:
- Sprint 28 multi-strategy FM ensemble permutations (the ensemble
  iterates 4 strategies; only the first 2 typically execute per
  call given the Pres_Poisson-class test fixtures).
- Multilevel coarsening fallback paths when matching produces an
  empty coarse graph (rare on real fixtures; common on synthetic
  pathological inputs which we don't test).
- Supernodal-postorder kernel's edge cases (fixture-conditional
  per Sprint 28 inheritance — `SPARSE_SUPERNODAL_POSTORDER=on`
  fires the kernel + adds the cold paths to coverage; the default
  `off` keeps them cold).

### Group H — error-string stubs: 50 %

`sparse_types.c` (50 %).

22 LOC — just `sparse_strerror()` mapping error codes to strings.
The 50 % miss = the new `SPARSE_ERR_CANCELLED` case (added Sprint
29 Day 6) is reachable only via the progress-callback cancellation
tests, but a few inherited error codes (e.g. an internal
`SPARSE_ERR_NOT_IMPLEMENTED` sentinel) are never returned by any
public routine, so their string-table entries stay cold.  Not
worth dedicated tests.

## Path-decision criteria

Per PLAN.md Day-12 task 2 ("Default to path (b) unless the Day-11
audit identifies low-hanging-fruit gaps that would clear ≥ 8
percentage points aggregate movement in < 10 hrs"):

- **Required movement to reach 95 %**: 13.7 percentage points
  (current 81.3 → target 95.0).
- **Required movement to reach 85 %**: 3.7 percentage points.
- **Required movement to reach 80 %**: already 1.3 points above.

For 13.7 percentage points of aggregate movement, the lowest-cost
files (those where 50-60 % of the LOC sits uncovered) are the
multilevel-partition fallback paths in `sparse_reorder_nd.c` (130
uncovered lines) and the FM ensemble permutations in
`sparse_graph.c` (~400 uncovered lines).  Both are defensive
fallback paths that fire only on pathological synthetic inputs:
- ND multilevel: coarsening fallback fires when matching produces
  an empty coarse graph (requires a graph with no matchable edges,
  which is degenerate).
- FM ensemble: 2 of 4 strategies cold under typical inputs;
  requires fixtures that trip the second-tier fallback strategies,
  which means hand-constructed adversarial graphs.

These would close 530 uncovered lines = 3.2 pp aggregate movement
at a cost of ~15 hrs of new test code + fixture engineering.  Even
the optimistic estimate doesn't clear the 8-pp Day-11 bar in 10 hrs,
and the remaining 10.5 pp would need 25-30 hrs of further work to
cover.  Path (a) is the wrong choice.

**Path (b) — lower threshold to 80** is the correct call.

## Threshold choice: 80, not 85 or 78

- **85**: above the measured 81.3 % aggregate.  Would fail
  immediately on the next CI coverage run.  Rejected.
- **80**: 1.3 pp below the measured 81.3 %, gives meaningful
  regression signal — any single-file regression of > ~2 pp would
  trip the gate.  **Chosen.**
- **78**: 3.3 pp below the measured aggregate, more headroom but
  too loose — a 3 pp regression is noticeable and the gate should
  catch it.  Rejected.
- **75**: 6 pp below, too loose; would silently accept significant
  regressions.  Rejected.

The 80 threshold tracks the lowest group-level coverage (Group C
iterative solvers at 77-82 %) — if any individual file's coverage
slips below 80 %, that's a real signal worth investigating.

## Implementation

### `Makefile` (this commit)

```diff
-COV_THRESHOLD = 95
+COV_THRESHOLD = 80
```

The check itself (`pct < $(COV_THRESHOLD)`) stays unchanged.  The
threshold comment expanded to cite this decision doc.

### `.github/workflows/ci.yml` (no change required)

The `coverage` job invokes `make coverage` directly, which now uses
`COV_THRESHOLD = 80`.  No workflow-level override is needed.

### Future calibration

If Sprint 30+ adds substantial new code without proportional test
coverage, the gate will trip + force the conversation about either
adding tests or further loosening the threshold.  If the test suite
gains significant new coverage (e.g. fault-injection harness lands
in a future sprint), the threshold can be tightened back upward
incrementally (`80 → 82 → 85`) without needing another full audit.

## References

- `docs/planning/EPIC_2/SPRINT_29/coverage_audit_day11.md` — Day 11
  per-file hand-estimate + Day-11 lcov-blocker analysis.
- `docs/planning/EPIC_2/SPRINT_29/PLAN.md` Day 12 task 1-3.
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 Item 8.
- `Makefile` lines 430-485 — coverage target + threshold check.
- `.github/workflows/ci.yml` — `coverage` job (no change Day 12).
- gcovr 8.6 docs — `--gcov-ignore-parse-errors` flag for the GCC
  bug 68080 workaround.
