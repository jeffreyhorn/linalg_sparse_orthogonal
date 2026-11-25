# Day 13 Arithmetic Claim Ledger

## Purpose

This ledger freezes the statements supported by Sprint 227 and the statements that remain out of
scope. It connects the implemented contract to executable tests and limits the performance
observation to the exact Day 12 collection. It does not replace the raw evidence or authorize a
release claim.

| Topic | Accepted statement | Evidence | Boundary |
| --- | --- | --- | --- |
| Permutation state | `sparse_add`, `sparse_add_inplace`, and `sparse_matmul` require both operands to be in original identity-map, unfactored state. After pointer and shape validation, unsupported state returns `SPARSE_ERR_BADARG` before allocation or mutation. | Public contracts in `include/sparse_matrix.h`; shared guard in `src/sparse_matrix.c`; the 27-case cross-API matrix plus inverse-only, factored, and precedence regressions. | This is rejection safety, not support for arithmetic on logical permuted views. |
| Out-of-place ownership | Successful `sparse_add` returns a newly owned matrix. On documented errors it does not take ownership of either input; valid output storage is cleared according to the existing API contract. Inputs remain exact. | `test_s227_add_failure_output_ownership_matrix`, cancellation sweeps, repeated-call and caller-variable-alias tests. | Caller variables may alias, but the pointed-to matrix objects must follow the documented input/output roles. |
| In-place aliasing and failure | `sparse_add_inplace(A, A, ...)` is rejected before mutation. Pre-mutation validation failures preserve both matrices exactly. A post-validation allocation failure leaves A as a structurally valid scaled/completed-prefix state and leaves B exact. | Public contract; `test_s227_add_inplace_self_alias_rejected_unchanged`; `test_s227_add_inplace_allocation_failure_partial_contract`. | In-place failure is deliberately not transactionally atomic after validation. |
| Out-of-place construction complexity | For original-state inputs, `sparse_add` performs a sorted two-list merge and directly appends each retained output node once. Its construction work is `O(rows + cols + nnz(A) + nnz(B))`, with `O(cols)` temporary builder workspace in addition to result storage. | Builder implementation and comment in `src/sparse_matrix.c`; public note in `include/sparse_matrix.h`; structural, randomized, and two-slab allocation tests. | The bound describes this representation and out-of-place path; it excludes allocator implementation costs and does not apply to `sparse_add_inplace`. |
| In-place complexity | In-place addition retains normal sorted insertion for B-only entries, so its cost is pattern dependent. | `sparse_add_inplace` implementation and public note. | No linear-time in-place claim is made. |
| Selected local timing | On the exact Day 12 host, compiler, source pair, and three registered workloads, all 30 candidate samples were faster than their paired baselines and all candidate timing ranges were below baseline ranges. Baseline/candidate medians were 13.4310/0.1515 ms (wide), 2.1550/0.2470 ms (varied), and 3.5605/0.2190 ms (narrow). | [Day 12 profile](day12-add-profile.md), [60-row sample table](day12-samples.csv), [300-file raw bundle](day12-raw/), and [raw hashes](day12-raw-sha256.txt). | One short local collection; no portable speed, arbitrary-sparsity, whole-library, superiority, release, or state-of-the-art claim. |
| Memory | Whole-process RSS and footprint ranges overlap and pair directions are mixed. | Day 12 profile and raw `/usr/bin/time` records. | No reduced-memory or builder-workspace claim. |
| ABI and source compatibility | No public declaration, public structure layout, or installed symbol was changed by the arithmetic contract and builder work. | Header/source diff inspection and successful Make/CMake builds. | Behavioral rejection of unsupported matrix state is intentional and documented. |
| Post-collection edits | Day 13 changed comments and documentation only. Recompiling the candidate with the exact Day 12 compiler and flags produced a byte-identical `sparse_matrix.o` and byte-identical benchmark executable. | Object SHA-256 `6a96c2d82f204c294e6e4aaa11b9f8187e7707b8c67048620cd6b6f451cd3db2`; executable SHA-256 `516700b283d24997070eb2c2633b789fa8b37607ff9fd92d3edea0108ae090db`; `cmp` success. | Static archive container hashes differ because Apple `ar` records member timestamps; the historical Day 12 source manifest is retained unchanged. |

## Prohibited extensions

- Do not describe the three selected profiles as representative of arbitrary sparse addition.
- Do not infer a memory reduction from overlapping process-level measurements.
- Do not extend the timing result to in-place addition, multiplication, solvers, another host,
  another compiler, another index width, or another library.
- Do not use “release ready,” “best,” “superior,” or “state of the art” on this evidence.
- Do not rewrite the Day 12 source or raw manifests to reflect later comment-only edits.

Day 14 still owns the complete quality chain and sprint-closeout decision.
