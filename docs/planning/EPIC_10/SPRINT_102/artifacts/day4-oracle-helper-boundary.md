# Sprint 102 Day 4 Oracle Helper Boundary Freeze

## Purpose

Day 4 freezes the helper extraction boundary before any test-support code is
changed. It reviews the existing Cholesky and LDLT external dense-reference
patterns, identifies duplicated glue, and selects the smallest extraction that
improves proof ownership without widening solver APIs or hiding
family-specific numerical semantics.

## Existing External Helper Pattern

| lane | Python helper | C harness owner | current fixture input | output contract |
|---|---|---|---|---|
| Cholesky CSC external dense reference | `tests/chol_external_dense_reference.py` | `tests/test_chol_csc.c` | Matrix Market path such as `tests/data/suitesparse/nos4.mtx` | `OK n` plus `n` values; `SKIP reason`; `ERROR reason` |
| LDLT CSC external dense reference | `tests/ldlt_external_dense_reference.py` | `tests/test_ldlt_csc.c` | fixture key such as `kkt5` or `kkt10` | `OK n` plus `n` values; `ERROR reason` |

Both C harnesses:

- construct a `python3 ...` command string;
- open the command with `popen` or `_popen`;
- parse an `OK`, `SKIP`, or `ERROR` header;
- parse one dense reference vector entry per line;
- close the pipe and treat non-zero exit as an oracle failure;
- copy a human-readable reason into a caller-provided buffer.

Both C harnesses intentionally keep solver-family behavior local:

- matrix or fixture construction;
- direct solver invocation;
- permutation handling;
- tolerance choice;
- residual computation;
- `ASSERT_NEAR` interpretation;
- cleanup around solver objects.

## Duplication and Ownership Assessment

| concern | duplicated today | extraction decision |
|---|---|---|
| command construction | yes | keep family-local because helper path and argument quoting are lane-specific |
| `popen` / `_popen` compatibility | yes | centralize in test-support helper if a shared reader is added |
| `OK` / `SKIP` / `ERROR` header parsing | yes | extract |
| dense vector line parsing | yes | extract |
| dimension mismatch handling | yes | extract with caller-provided label |
| pipe close / non-zero exit handling | yes | extract |
| fixture construction | no, family-local | do not extract |
| solver execution | no, family-local | do not extract |
| RHS construction | similar but family-sensitive | do not extract in Day 5 |
| tolerance and residual checks | family-local | do not extract |
| Windows helper skip policy | similar but call-site-specific | leave at call site so each test controls skip wording |

## Selected Day 5 Extraction

Selected helper:

```c
tf_external_reference_status_t tf_read_external_reference_vector(
    const char *cmd,
    const char *label,
    double *x_out,
    idx_t n,
    char *reason,
    size_t reason_cap);
```

Target location:

- `tests/test_solver_helpers.h`

Rationale:

- `tests/test_solver_helpers.h` already exists as a narrow solver-test helper
  layer.
- The helper is test-support only and remains outside public headers,
  library sources, and installed API surfaces.
- The helper consolidates subprocess/vector parsing without creating a generic
  solver oracle framework.
- Cholesky and LDLT can keep fixture construction, solver path, tolerance
  policy, residual policy, and skip/fail assertions local.

Status enum:

```c
typedef enum {
    TF_EXTERNAL_REFERENCE_ERROR = -1,
    TF_EXTERNAL_REFERENCE_SKIP = 0,
    TF_EXTERNAL_REFERENCE_OK = 1
} tf_external_reference_status_t;
```

Input contract:

| input | contract |
|---|---|
| `cmd` | complete command string to execute; caller owns command construction |
| `label` | short family label used in error messages, such as `external dense reference` or `external LDLT reference` |
| `x_out` | caller-allocated dense vector of length `n` |
| `n` | expected vector length from the oracle header |
| `reason` | caller-provided reason buffer |
| `reason_cap` | reason buffer capacity; zero is invalid |

Output contract:

| status | meaning | call-site behavior |
|---|---|---|
| `TF_EXTERNAL_REFERENCE_OK` | helper produced `OK n` and exactly `n` parseable vector entries, command exited cleanly | compare against solver result |
| `TF_EXTERNAL_REFERENCE_SKIP` | helper produced `SKIP reason` or pipe open was unavailable | call `SKIP_TEST(reason)` if lane treats helper absence as unsupported |
| `TF_EXTERNAL_REFERENCE_ERROR` | helper produced `ERROR reason`, malformed output, dimension mismatch, truncated output, parse failure, or non-zero exit | fail the comparison |

The pipe-open case stays `SKIP` to preserve the existing behavior in both
harnesses, where an unavailable Python pipe is treated as helper unsupported
rather than numerical failure.

## Explicit Non-Extraction

Day 5 must not extract:

- dense Cholesky or dense LU/LDLT math into C;
- Matrix Market loading;
- fixture key dispatch;
- `x_true` and RHS construction;
- Cholesky/LDLT factor or solve calls;
- permutation handling for LDLT;
- residual calculations;
- tolerance constants;
- direct solver public APIs;
- any installed or library helper.

## Reuse Plan

| existing function | Day 5 action |
|---|---|
| `read_external_dense_reference_solution(...)` in `tests/test_chol_csc.c` | replace with a small command-construction wrapper that calls `tf_read_external_reference_vector(...)` |
| `read_ldlt_external_dense_reference_solution(...)` in `tests/test_ldlt_csc.c` | replace with a small command-construction wrapper that calls `tf_read_external_reference_vector(...)` |
| `assert_cholesky_external_dense_reference(...)` | keep family-local |
| `assert_ldlt_external_dense_reference(...)` | keep family-local |
| Python dense reference helpers | keep unchanged in Day 5 |

## Day 5 File Boundary

Expected touched files:

| file | intended change |
|---|---|
| `tests/test_solver_helpers.h` | add test-only external reference vector reader and status enum |
| `tests/test_chol_csc.c` | include/use helper and remove duplicated parser body |
| `tests/test_ldlt_csc.c` | include/use helper and remove duplicated parser body |
| `docs/planning/EPIC_10/SPRINT_102/artifacts/day5-oracle-helper-extraction.md` | implementation evidence and validation results |
| `docs/planning/EPIC_10/SPRINT_102/WORKING_NOTES.md` | Day 5 notes |

No `src/`, `include/`, public documentation, CMake, or Makefile changes are
planned for Day 5.

## Validation Plan for Day 5

Because Day 5 is expected to modify `.c` and `.h` test files, required
validation is:

```sh
make format
make build/test_chol_csc
./build/test_chol_csc
make build/test_ldlt_csc
./build/test_ldlt_csc
make lint
make test
git diff --check
rg -n "[ \t]+$" tests/test_solver_helpers.h tests/test_chol_csc.c tests/test_ldlt_csc.c docs/planning/EPIC_10/SPRINT_102
```

Focused helper behavior to preserve:

| lane | expected preserved behavior |
|---|---|
| Cholesky CSC | `nos4` and `bcsstk04` external dense-reference tests still compare solver output to Python dense reference |
| LDLT CSC | `kkt5` and `kkt10` external dense-reference tests still compare solver output to Python dense reference |
| Windows helper handling | external helper tests still skip where helper execution is unsupported |
| malformed helper output | malformed output remains a comparison failure, not a pass |

## Claim Boundaries

The Day 5 helper extraction, if implemented, earns only a maintainability
claim:

> External dense-reference vector parsing is shared by direct-solver tests.

It does not claim:

- new LU, QR, SVD, Cholesky, or LDLT oracle coverage;
- direct CSR/CSC solver APIs;
- external oracle coverage for every direct solver;
- portable performance superiority;
- public API behavior changes.

## Day 4 Conclusion

Sprint 102 should extract only the external-reference subprocess/vector parser
into `tests/test_solver_helpers.h` on Day 5. All solver-family math, fixture
construction, tolerance choices, residual checks, and public claim boundaries
remain local to the family tests and later Sprint 102 evidence artifacts.
