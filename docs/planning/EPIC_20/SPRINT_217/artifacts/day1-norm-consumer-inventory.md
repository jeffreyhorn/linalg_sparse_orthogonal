# Sprint 217 Day 1: Norm Consumer Inventory

**Scope:** [Day 1 plan](../PLAN.md); public `vec_norm2`/`vec_norminf` behavior,
consumers, convergence dependencies, and focused local baseline. Audit only;
no numerical contract or code change is approved here.

## Current Helper Behavior

Source: `src/sparse_vector.c:4-23`; declarations: `include/sparse_vector.h:22,31`;
existing tests: `tests/test_sparse_vector.c:14-42`.

| Input | `vec_norm2` implementation result | `vec_norminf` implementation result | Evidence limit |
| --- | --- | --- | --- |
| NULL, any length | `0.0` | `0.0` | NULL tested with positive length |
| Nonpositive length, non-NULL | `0.0` | `0.0` | `n=0` tested only for 2-norm; negative length not tested |
| Ordinary finite values | `sqrt(sum(v[i]*v[i]))` | `max(fabs(v[i]))` | Ordinary cases tested |
| Extreme finite values | Squares/sum may overflow to `inf` or underflow to zero even when the mathematical norm is representable | Maximum finite magnitude, subject to ordinary floating-point behavior | No extreme-scale norm test |
| Infinity present, no NaN | `inf` | `inf` | Inferred from code, not tested |
| NaN present, no infinity | `NaN` | NaNs are ignored; all-NaN returns `0.0` | Inferred from code, not tested |
| Both NaN and infinity | `NaN` | `inf` | Inferred from code, not tested |

These are observed or code-inferred behaviors. Day 2 owns the desired public
contract, including mixed nonfinite values and representable results.

## Production Consumers

`rg -n 'vec_norm(2|inf)\(' src tests benchmarks examples include --glob '*.[ch]'`
found the following production calls. Every production consumer below calls
`vec_norm2`; `vec_norminf` has no production caller. No direct or eigensolver
source file calls either public helper.

| Location | Purpose | Risk from current norm |
| --- | --- | --- |
| `src/sparse_iterative.c:76` | Shared RHS norm/trivial-system decision for CG, matrix-free CG, GMRES, BiCGSTAB, and matrix-free BiCGSTAB | A tiny nonzero RHS may underflow to zero and be reported converged; zero denominator is avoided only by this branch; nonfinite denominator is not rejected |
| `src/sparse_iterative.c:280,335` | CG initial and updated residual norms; convergence/reporting by `rnorm/bnorm` | Overflow/underflow can distort threshold; `inf/inf` or NaN comparisons are unordered |
| `src/sparse_iterative.c:463,521` | Matrix-free CG initial and updated residual norms | Same ratio and status risk as CG |
| `src/sparse_iterative.c:651,683` | GMRES initial or zero-iteration true residual ratios | Nonfinite ratio can bypass ordinary `<= tol` checks and corrupt reported residual |
| `src/sparse_iterative.c:754,835` | GMRES Arnoldi basis scaling and new-vector norm | Zero/inf/NaN beta can invalidate basis normalization and downstream convergence |
| `src/sparse_iterative.c:931` | GMRES final true residual ratio | Final result may be nonfinite and comparisons unordered |
| `src/sparse_iterative.c:1054,1135,1182,1200,1249` | BiCGSTAB initial, half-step, updated, and final true residual norms | Intermediate and final relative residuals can be nonfinite; a NaN final ratio is not greater than tolerance |
| `src/sparse_iterative.c:1317,1372,1420,1435,1484` | Matrix-free BiCGSTAB counterpart | Same risk as matrix-backed path |
| `src/sparse_iterative_minres.c:46,95,99,173` | MINRES RHS trivial decision, Lanczos beta, initial true residual, updated beta | Tiny RHS can be classified as zero; beta normalization and convergence ratios can be invalid |
| `src/sparse_iterative_block.c:76,112,157` | Block CG per-column RHS and residual norms | Underflow can mark a nonzero column trivial; nonfinite ratio can be omitted from max-residual reporting |

The shared trivial helper is invoked from `src/sparse_iterative.c:244,429,633,
1027,1289`. Its zero-RHS branch zeros `x` and marks the result converged. In
block CG, the zero RHS branch at `src/sparse_iterative_block.c:76-83` likewise
zeros that solution column and replaces its denominator with `1.0`.

MINRES also computes true residuals by direct sum of squares at
`src/sparse_iterative_minres.c:201-207,253-262`; that code is not a public norm
call and will need separate review. BiCGSTAB checks some coefficients and an
updated residual for finiteness, but not every norm or final ratio; this table
describes possible behavior, not a reproduced failure for each path.

## Other Public-Helper Consumers

| Location | Purpose | Risk classification |
| --- | --- | --- |
| `tests/test_sparse_vector.c:14-42,244-321` | Direct helper behavior and vector-operation residual assertions | Coverage/diagnostic: lacks extreme and nonfinite cases |
| `tests/test_ilu.c:1157` | `west0067` ILUT solution norm, then `znorm > 0` assertion | Diagnostic gate: `inf > 0` passes; cannot prove finite solution |
| `tests/test_integration.c:55,122,131,181,327-328,416,1234,1816` | Direct/integration solver infinity-norm residual checks and relative residual | Diagnostic/assertion: NaNs can be ignored by infinity norm, weakening residual evidence |
| `tests/test_qr_external_ref_helpers.h:209,226,240,358,474,597,612` | QR reference nullspace normalization and residual assertions | Test oracle: overflow/underflow could change acceptance at extreme scales |
| `tests/test_qr_helpers.h:293-294,318,338` | QR residual/RHS/solution ratios | Test oracle: zero or nonfinite denominator can invalidate ratio |
| `tests/test_qr.c:1047,1086,1140,1207` | QR nullspace residual assertions | Test oracle: extreme-scale false pass/fail possible |
| `tests/test_qr_solve.c:173,290,495,547,584,633` | QR solve residual normalization and norm assertions | Test oracle: nonfinite or underflowing norm can distort acceptance |
| `tests/test_qr_corpus.c:143,258` | Corpus solution/basis norms | Test diagnostic/assertion risk at extreme scales |
| `tests/test_svd_selected_helpers.h:310,312` | Selected SVD solution norm assertion/report | Test diagnostic/assertion risk at extreme scales |
| `tests/test_sprint6_integration.c:27-28` | Integration residual/RHS normalization | Test oracle: zero/nonfinite denominator |
| `benchmarks/bench_convergence.c:47-48`, `benchmarks/bench_iterative_reuse.c:109-110` | Report relative residuals | Benchmark reporting: potentially nonfinite or misleading ratio |
| `benchmarks/bench_main.c:203,306`, `benchmarks/bench_scaling.c:106` | Report infinity-norm residual | Benchmark reporting: NaN entries can be hidden |

`tests/test_cholesky.c`, `tests/test_known_matrices.c`, and
`tests/test_sparse_lu.c` define file-local `vec_norminf`; `tests/test_colamd.c`
and `examples/example_minnorm.c` define file-local `vec_norm2`. These are not
public-helper consumers. `tests/test_solver_helpers.h` defines
`tf_vec_norm2`, also separate. No example calls the public norm helpers.

## Convergence Trace

| Solver | Initial/zero path | Update and final path | Numerical question for Day 2 |
| --- | --- | --- | --- |
| CG, matrix-backed and matrix-free | Shared RHS-zero helper, then initial `rnorm/bnorm <= tol` | Updated ratio determines convergence and reported norm | Distinguish true zero RHS from underflow; define nonfinite ratio status |
| GMRES | Shared helper; initial and zero-iteration true residual ratios | Arnoldi norms set basis scale; final true residual ratio reported | Guard zero/nonfinite beta and unordered final comparison |
| MINRES | Own `bnorm == 0` branch returns converged; initial beta and residual ratio | Updated beta; true residual recomputed by separate square/sum code | Define true-zero and nonfinite handling in both norm paths |
| BiCGSTAB, matrix-backed and matrix-free | Shared RHS-zero helper; initial residual ratio | Half-step and update checks; final true ratio can be NaN, making `true_rel_res > tol` false | Do not let unordered final check retain a converged candidate |
| Block CG | Per-column zero RHS branch, then residual ratios | Per-column final convergence; max residual uses `if (rel > max_res)` | NaN ratio cannot increase reported maximum; check each column |

These are risk paths identified by inspection, not claims that every listed
input has a demonstrated solver failure. A finite `bnorm` that underflows to
zero is enough to make several trivial branches suspect.

## Focused Baseline

Run on 2026-10-20, branch `sprint-217` at
`bde1d873c55a5fbda9fd39531f4ca2c05c7ee3f7`. Host: Darwin 24.6.0
x86_64; `cc --version`: Apple clang 17.0.0
(`clang-1700.6.4.2`), CommandLineTools target
`x86_64-apple-darwin24.6.0`. Commands were run locally in the repository.

| Command | Exit | Result |
| --- | ---: | --- |
| `make -s build/test_sparse_vector build/test_ilu build/test_iterative` | 0 | Three focused binaries available |
| `./build/test_sparse_vector` | 0 | 24 tests, 0 failures, 0 skips, 52 assertions |
| `./build/test_ilu` | 0 | 34 tests, 0 failures, 0 skips, 328 assertions |
| `./build/test_iterative` | 0 | 85 tests, 0 failures, 0 skips, 746 assertions |
| `make -s build/test_minres build/test_block_solvers` | 0 | Additional solver binaries available |
| `./build/test_minres` | 0 | 43 tests, 0 failures, 0 skips, 702 assertions |
| `./build/test_block_solvers` | 0 | 15 tests, 0 failures, 0 skips, 168 assertions |

Selected unmodified `test_ilu` output:

```text
west0067: ILU(0) returns singular or nearly singular matrix (expected)
west0067: ILUT returns success
west0067: ILUT solve norm=inf
[PASS] test_ilut_west0067
west0067 ILUT-right-GMRES(30): 34 iters, res=1.000e+00, conv=0
```

`tests/test_ilu.c:1122-1167` accepts an ILUT solve when `znorm > 0.0`, which
also accepts infinity. The output alone does not distinguish overflow in
`vec_norm2` from nonfinite solution entries, and the GMRES line is a separate
run. It is therefore an unresolved numerical diagnostic, not proof of a solver
defect or of a valid solve.

## Handoff

Day 2 must define exact norm and solver status behavior before implementation.
Later days should add discriminating extreme-scale fixtures and an independent
ILUT factor-product residual check. The current focused suite is green but
does not establish those properties.
