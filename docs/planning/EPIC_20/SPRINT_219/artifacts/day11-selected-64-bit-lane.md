# Day 11: Selected 64-Bit Functional Lane

**Date:** 2026-10-24

**Scope:** Sprint 219 item 219.5, first 10 of 20 hours
**Host:** macOS, native CMake static build

## Selected Cases And Expected Values

The exact local selector is `ctest --test-dir <fresh-64-bit-build> -R
'^test_idx_width_functional$' -V`. The test target is built from the same
fresh CMake tree as `sparse_lu_ortho`, configured with
`-DSPARSE_IDX_BITS=64`; it is not an installed-consumer or a broad 64-bit
algorithm claim. Before running it, the lane checks the CMake cache,
generated version header, and compile commands for the 64-bit selection.

| Case | Deterministic input | Required result |
| --- | --- | --- |
| Width identity | Built test and linked library | `sizeof(idx_t) * CHAR_BIT`, `SPARSE_BUILT_IDX_BITS`, and `sparse_idx_bits()` all equal 64. |
| Matrix insertion and LU | Symmetric 3x3 tridiagonal with diagonal 4 and off-diagonal 1; `b=[6,12,14]` | 7 nonzeros; inserted entries match; `lu_csr_factor_solve` returns `SPARSE_OK` and `x=[1,2,3]` within `1e-10`. |
| Matrix Market IO | Save and reload that same matrix | 3x3, 7 nonzeros, every entry matches exactly. |
| CSR/CSC conversion | Convert that matrix to each format and back | Pointer arrays `[0,2,5,7]`, ordered indices and values match the tridiagonal fixture, and reconstructed entries match. |
| Index text boundary | Format and parse `2147483648`, one above `INT32_MAX` | Decimal text is `2147483648`; parsed `idx_t` matches without large allocation. Also roundtrip `IDX_MAX`. |

## Execution Evidence

`bash tests/test_cmake_idx_width_functional.sh` created a fresh CMake tree,
configured `SPARSE_IDX_BITS:STRING=64`, built
`test_idx_width_functional`, and checked the generated
`SPARSE_BUILT_IDX_BITS 64` header plus the `-DSPARSE_IDX_BITS=64` compile
commands for both `sparse_types.c` and the test source. Verbose CTest ran
exactly `^test_idx_width_functional$`: **1/1 test passed**, with all five
named cases passing. Its runtime line was
`index width: typedef=64 header=64 archive=64`, proving that the selected
functional cases used the 64-bit archive rather than only a 64-bit typedef.

| Check | Result |
| --- | --- |
| Matrix creation, insertion, CSR LU solve | 3x3, 7 nonzeros, `SPARSE_OK`, `x=[1,2,3]` within `1e-10`. |
| Matrix Market save/load | 3x3, 7 nonzeros, all 9 logical values match exactly. |
| CSR and CSC conversions and reconstruction | Both pointer arrays `[0,2,5,7]`; expected ordered indices/values and reconstructed matrix match. |
| Index formatting/scanning | `2147483648` and `IDX_MAX` roundtrip through `SPARSE_PRIDX`/`SPARSE_SCNIDX` without a large allocation. |

The same test is registered in both CMake and Make. `make format-check`
passed; `bash -n tests/test_cmake_idx_width_functional.sh` passed;
`make -s BUILDDIR=build/s219-day11-quality -j4 lint` passed strict compile,
clang-tidy, and cppcheck; and
`make -s BUILDDIR=build/s219-day11-quality -j4 test` passed the full default
32-bit Make suite, including this test. No in-scope numerical or structural
failure arose, so no library source repair was needed.

This selected lane does not establish every algorithm's 64-bit behavior, a
large-allocation guarantee, or a shared-library ABI. Day 12 must register
the focused 64-bit build/test in hosted CI and report platform-specific
outcomes; hosted status is not inferred from this local pass.
