# Sprint 184 Day 4: Lifecycle, Ownership, and Error Contracts

**Sprint:** 184 - Public Header Coherence Batch 3
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_184/`
**Status:** Complete

## Purpose

Day 4 performs the first comment-only QR public-header cleanup pass. The pass
normalizes lifecycle, ownership, borrowed input, caller-owned output, and
error-code wording for the highest-risk QR declarations while preserving the
public declaration surface.

## Files Changed

| File | Change type |
| --- | --- |
| `include/sparse_qr.h` | Comment-only public-header contract cleanup. |
| `docs/planning/EPIC_16/SPRINT_184/WORKING_NOTES.md` | Day 4 log update. |
| `docs/planning/EPIC_16/SPRINT_184/artifacts/day4-core-contract-cleanup.md` | Day 4 artifact. |

## Header Cleanup Summary

| Area | Day 4 update |
| --- | --- |
| File-level contract | Clarified that `include/sparse_qr.h` owns API-local contracts while runnable workflows and evidence boundaries live in examples/docs. |
| Factor object lifecycle | Clarified that callers own the `sparse_qr_t` storage, successful factorization stores owned factor data inside it, and `sparse_qr_free()` releases that owned data. |
| Reuse behavior | Preserved the warning that factor functions overwrite the struct without freeing prior contents and callers must free populated objects before reuse. |
| Factorization ownership | Marked `A` as borrowed/not modified and `qr` as caller-owned output that must be freed after success. |
| `sparse_qr_factor_opts()` errors | Expanded return-code docs for NULL, non-identity permutations, allocation failure, and callback cancellation. |
| Apply/form-Q ownership | Clarified borrowed factorization input and caller-owned output buffers. |
| Solve/refine ownership | Clarified borrowed factor/original matrix inputs, caller-owned solution/residual outputs, and temporary workspace allocation errors. |
| Unsupported claims | Kept evidence interpretation out of the header and did not add broad QR, parity, platform, package, ABI, performance, or state-of-the-art claims. |

## Implementation Cross-Check

Day 4 checked the edited return-code comments against `src/sparse_qr.c`:

| Declaration | Implementation behavior checked |
| --- | --- |
| `sparse_qr_factor_opts(...)` | Returns `SPARSE_ERR_NULL`, `SPARSE_ERR_BADARG`, `SPARSE_ERR_ALLOC`, and `SPARSE_ERR_CANCELLED` along documented paths. |
| `sparse_qr_apply_q(...)` | Returns `SPARSE_ERR_NULL` for NULL arguments and otherwise returns `SPARSE_OK`; the header does not claim an unfactored error. |
| `sparse_qr_form_q(...)` | Returns `SPARSE_ERR_NULL`, `SPARSE_ERR_ALLOC`, or `SPARSE_OK`; the header does not claim an unfactored error. |
| `sparse_qr_solve(...)` | Returns `SPARSE_ERR_NULL` for NULL inputs or missing factor data, and `SPARSE_ERR_ALLOC` for workspace allocation failures. |
| `sparse_qr_refine(...)` | Returns `SPARSE_ERR_NULL`, `SPARSE_ERR_SHAPE`, `SPARSE_ERR_ALLOC`, or propagated solve status. |

## Declaration Preservation Evidence

The Day 2 baseline used line-numbered declaration starts. That checksum changes
when comment edits move declarations to new line numbers, even if declarations
themselves are unchanged. Day 4 therefore uses a comment-stripped,
line-number-independent declaration comparison for the edited QR header.

Comparison command:

```sh
diff -u \
  <(git show HEAD:include/sparse_qr.h | perl -0pe 's@/\\*.*?\\*/@@gs; s@//.*$@@mg' | rg -o "^(typedef (struct|enum)|} sparse_[a-zA-Z0-9_]+|[a-zA-Z_][a-zA-Z0-9_ *]+\\s+sparse_qr[a-zA-Z0-9_]*\\()") \
  <(perl -0pe 's@/\\*.*?\\*/@@gs; s@//.*$@@mg' include/sparse_qr.h | rg -o "^(typedef (struct|enum)|} sparse_[a-zA-Z0-9_]+|[a-zA-Z_][a-zA-Z0-9_ *]+\\s+sparse_qr[a-zA-Z0-9_]*\\()")
```

| Check | Result |
| --- | --- |
| Pre-edit comment-stripped QR declaration hash | `5d20a4cf0cefb813c8eabc3d531d6ba31429f5023968c5d6f56c2506456d9a67` |
| Post-edit comment-stripped QR declaration hash | `5d20a4cf0cefb813c8eabc3d531d6ba31429f5023968c5d6f56c2506456d9a67` |
| Comment-stripped declaration diff | No output. |

## Deferred Cleanup For Day 5

Day 5 should continue with QR tolerance, workspace, option/result, and
cancellation wording:

- normalize rank tolerance language for `sparse_qr_rank()` and
  `sparse_qr_rank_info()`;
- clarify nullspace and diagnostic output buffer ownership;
- decide whether to mention callback propagation for minimum-norm paths after
  checking implementation details;
- keep evidence-boundary wording in docs, not in header promises.

## Validation

Because Day 4 modified a public `.h` file, the required validation is:

```sh
make format && make lint && make test
```

Result: passed.

Additional focused checks:

```sh
git diff --check
```

Result: passed.

The comment-stripped declaration comparison recorded above also passed with no
diff output.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Core contract wording is consistent with existing public API conventions. | Complete | Borrowed inputs, caller-owned outputs, free/reuse lifecycle, and return-code wording are normalized in `include/sparse_qr.h`. |
| No public declarations drift as a result of comment cleanup. | Complete | Comment-stripped declaration hashes match and the declaration diff is empty. |
| Unsupported or ambiguous behavior claims are removed or bounded. | Complete | Header wording stays API-local and points evidence/workflow interpretation to docs/examples. |
