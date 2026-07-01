# Sprint 101 Day 9 Lifecycle and Ownership Batch

## Purpose

Day 9 lands the focused lifecycle and ownership follow-through selected by
the Day 8 design. The Day 8 queue found that behavior was already covered by
Day 6 tests, while public wording still described the diagnostic CSR/CSC
constructors as compatibility wrappers. Day 9 therefore updates public docs
and records why no additional C test batch was needed.

## Changed Files

| file | change |
|---|---|
| `README.md` | clarified compressed-input ownership, diagnostic constructor role, and returned matrix ownership |
| `docs/tutorial.md` | added compressed-input workflow guidance near the tutorial entry point and included `sparse_csr.h` in the header list |
| `docs/planning/EPIC_10/SPRINT_101/WORKING_NOTES.md` | recorded Day 9 actions, validation expectations, and exit state |
| `docs/planning/EPIC_10/SPRINT_101/artifacts/day9-lifecycle-ownership-batch.md` | recorded lifecycle batch evidence |

## Lifecycle and Ownership Clarifications

| topic | clarification |
|---|---|
| simple constructors | `sparse_create_from_csr(...)` and `sparse_create_from_csc(...)` remain the smallest compressed-first entry path when input already lives in CSR/CSC storage |
| diagnostic constructors | `sparse_from_csr(...)` and `sparse_from_csc(...)` are diagnostic compressed-first constructors, not merely retained compatibility wrappers |
| input ownership | caller-owned CSR/CSC arrays are validated and copied during construction; they are not adopted |
| output ownership | successful construction returns a caller-owned `SparseMatrix *` that is freed with `sparse_free(...)` |
| solver entry | constructed matrices enter the normal one-shot or repeated-run solver surfaces as public matrix shells |
| mutable shell compatibility | insertion-based `SparseMatrix` construction remains supported, but it is no longer described as the only natural front door |

## Test Decision

No new C tests were added on Day 9.

Reasons:

- Day 6 already added focused invalid-input diagnostics for CSR and CSC
  construction.
- Day 6 already added copy-ownership tests proving that later caller mutation
  of CSR/CSC arrays does not change the returned matrix.
- Day 6 already added a bounded solver-entry smoke test from a CSR-built
  matrix into one-shot LU.
- Day 8 did not identify an untested no-op/free or mutation/factored-state
  behavior that required new implementation.

Day 12 remains the planned checkpoint for deciding whether additional
regression proof is needed after docs/examples settle.

## Validation

Day 9 changed public documentation and planning artifacts only. It did not
modify `.c` or `.h` files, build-system files, workflows, scripts, or example
source.

Required validation for this docs batch:

```bash
git diff --check
rg -n "[ \t]+$" README.md docs/tutorial.md docs/planning/EPIC_10/SPRINT_101
```

## Day 9 Conclusion

The compressed-first lifecycle story is now consistent across implementation
evidence, header comments, README guidance, and the tutorial entry point:
compressed arrays remain caller-owned, constructors copy into a normal
caller-owned `SparseMatrix`, and `sparse_from_csr/csc` provide explicit
diagnostic status. The next sprint day can design the broader compatibility
path documentation update without reopening constructor behavior.
