# Sprint 99 Day 3: Final Comparison Scope

## Purpose

Day 3 freezes the final comparison scope before Sprint 99 executes evidence
commands. This artifact defines the selected proof lanes, command checklist,
pass/fail expectations, output artifacts, and claim-language boundaries for
Days 4 and 5.

## Scope Inputs

The Day 3 scope is based on:

- Sprint 90 comparison-and-measurement contract
- Sprint 94 capability-surface closeout
- Sprint 97 build/package/product closeout
- Sprint 98 assurance and external comparison closeout
- Sprint 99 Day 2 contradiction-class re-audit

The goal is to verify the bounded Epic 9 closeout story, not to add new
comparison architecture.

## Final Evidence Lanes

| Lane | Proof owner | Day 4/5 command or check | Closeout role |
|---|---|---|---|
| Cholesky CSC external correctness | `tests/test_chol_csc.c`, `tests/chol_external_dense_reference.py` | `make build/test_chol_csc && ./build/test_chol_csc` | validates retained external dense-reference SPD lane on `nos4` and `bcsstk04` |
| LDLT CSC external correctness | `tests/test_ldlt_csc.c`, `tests/ldlt_external_dense_reference.py` | `python3 tests/ldlt_external_dense_reference.py kkt5`; `python3 tests/ldlt_external_dense_reference.py kkt10`; `make build/test_ldlt_csc && ./build/test_ldlt_csc` | validates Sprint 98 deterministic KKT external dense-reference lane |
| Aggregate reviewed correctness | `Makefile`, CMake registration, full test owners | `make quality-review-full` | final reviewed local baseline for correctness/build/test parity |
| Runtime/fill calibration | `benchmarks/bench_reorder.c`, `Makefile`, `benchmarks/README.md` | `make bench-reorder-sprint86` | validates bounded two-fixture reorder/fill artifact; `nnz_L` is claim-bearing fill field, `reorder_ms` is local context |
| Canonical benchmark reporting | canonical benchmark drivers, `scripts/bench_canonical_report.sh` | `make bench-canonical-report` | validates report generation; not a timing superiority claim |
| Make install/export | `tests/test_install.sh`, `INSTALL.md` | `bash tests/test_install.sh` | validates static-first Make install, uninstall, and `pkg-config` proof |
| CMake install/export | `tests/test_cmake_install.sh`, `examples/cmake_example/CMakeLists.txt`, `INSTALL.md` | `bash tests/test_cmake_install.sh` | validates CMake install/export and consumer target proof |
| Usability/docs coherence | `README.md`, `INSTALL.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, public headers | stale-claim and non-claim scans | validates that public/support language does not outrun maintained evidence |
| Workflow/CI coherence | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | workflow-scope scan and expected-count review | validates platform-scope wording and Windows reviewed CTest count expectation |

## Command Checklist

### Day 4: Correctness and Runtime/Fill

Run and capture:

```sh
python3 tests/ldlt_external_dense_reference.py kkt5
python3 tests/ldlt_external_dense_reference.py kkt10
python3 tests/ldlt_external_dense_reference.py nope
make build/test_chol_csc && ./build/test_chol_csc
make build/test_ldlt_csc && ./build/test_ldlt_csc
make bench-reorder-sprint86
make bench-canonical-report
```

Interpretation:

- helper positive fixtures should pass
- helper unknown fixture should fail closed with nonzero exit
- focused C tests should pass with zero failures
- `bench-reorder-sprint86` should emit bounded reorder/fill rows
- canonical report should generate maintained benchmark CSV artifacts
- runtime fields are local calibration context, not pass/fail product
  thresholds

### Day 5: Package, Usability, and Workflow

Run and capture:

```sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
rg -n "best-in-class|benchmark supremacy|full platform parity|shared-library-first|dynamic ABI|broad complex|mixed-precision|portable timing|universal .*superiority|all solver" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md include .github/workflows
rg -n "EXPECTED_WINDOWS_CTEST_COUNT|reviewed CMake subset|Linux is the enforced source-of-truth|macOS enforces|static-first|bench-reorder-sprint86|bench-canonical-report" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md .github/workflows
```

Interpretation:

- install/export scripts should pass
- stale-claim scan should find no positive unsupported broad claims
- boundary scan should show explicit static-first, benchmark, and platform
  fences
- any workflow expected-count mismatch becomes a final-fix candidate

### Final Broad Validation Candidate

Run after Day 6 or after any final fix batch, not before the decision point:

```sh
make quality-review-full
```

If `.c` or `.h` files change during the final fix batch, also run:

```sh
make format && make lint && make test
```

## Pass/Fail Expectations

| Evidence lane | Closeout-ready | Final-fix candidate | Residual/non-claim |
|---|---|---|---|
| external correctness helpers | positive fixtures pass; unknown fixture fails closed | helper missing, nondeterministic, or mismatched output | broader fixture/corpus expansion |
| focused Cholesky/LDLT CSC tests | zero failed tests | external-reference rows fail or segfault | broader solver-family external proof |
| reorder/fill benchmark | command runs and emits bounded artifact rows | command broken or docs contradict output fields | portable timing or full-corpus comparison |
| canonical report | CSV report generation works | report script broken or schema docs stale | benchmark supremacy or thresholds |
| install/export | scripts pass with static-first shape | install/export proof fails or docs contradict script | shared-library package maturity |
| docs/usability | no unsupported positive broad claims | stale overclaim found in public/support surface | future comparison architecture |
| workflow/CI | scope comments and expected counts match current proof | stale count/scope statement found | full cross-platform parity |

## Allowed Language

Sprint 99 closeout may say:

- Epic 9 materially improved compressed-first entry paths while retaining the
  linked-list shell as the mutable compatibility owner.
- Epic 9 improved backend and direct-family maturity on bounded maintained
  lanes.
- Cholesky CSC and LDLT CSC have maintained external dense-reference solve
  checks on named fixtures.
- Static-first install/export and CMake consumer proof are maintained and
  validated.
- Runtime/fill evidence is bounded, local, and calibration-oriented.
- Linux remains the strongest reviewed source of truth; macOS and Windows have
  intentionally narrower reviewed or supplemental proof roles.

## Disallowed Language

Sprint 99 closeout must not say or imply:

- the whole library is now compressed-first
- the project has broad complex or mixed-precision maturity
- the project has broad backend-neutral acceleration maturity
- the package story is shared-library-first or dynamically ABI-stable
- Linux, macOS, and Windows have symmetric reviewed parity
- benchmark output proves portable timing superiority
- reorder/fill output proves universal best choice
- every solver family has maintained external correctness comparison
- coverage topology was widened or coverage quality materially improved in
  Sprint 99 unless later evidence changes that

## Deferred Comparison Work

These remain residual unless a future architecture explicitly promotes them:

- broader LDLT CSC Matrix Market or indefinite corpus external comparison
- iterative solver external comparison around convergence semantics
- eigensolver/LOBPCG external comparison with cluster/tolerance/runtime bounds
- QR and SVD external comparison lanes
- generated reorder/fill report target
- broader ecosystem runtime and reorder/fill comparisons
- CI capture for benchmark artifacts
- coverage threshold or artifact ownership expansion

## Day 3 Conclusion

The final Sprint 99 comparison scope is frozen. Days 4 and 5 should execute
the selected commands and classify any failure or stale claim as a
final-fix-candidate. They should not widen proof lanes or promote residual
comparison work into closeout claims.
