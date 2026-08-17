# Sprint 164 Day 10: Declaration Preservation Re-Capture

## Purpose

Day 10 re-ran the Day 4 declaration capture after the selected public-header
cleanup batch to prove that Sprint 164 comment and documentation cleanup did
not change public declarations.

Selected headers:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_matrix.h`

## Capture Method

The Day 3/Day 4 normalization method was reused with `after` output names:

- strip block and line comments;
- preserve non-comment declaration-like text in source order;
- collapse adjacent blank lines;
- write local ignored evidence under `build/sprint164/declarations/`;
- compute a SHA-256 checksum for the combined selected-header bundle.

Generated declaration evidence remains local ignored output and is not
committed.

## Before-State Baseline

Day 4 captured the before-state checksum:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41  build/sprint164/declarations/selected-public-headers.before.normalized.txt
```

Day 4 line counts:

| Local Generated File | Lines |
| --- | ---: |
| `build/sprint164/declarations/sparse_iterative.h.normalized.txt` | 151 |
| `build/sprint164/declarations/sparse_eigs.h.normalized.txt` | 102 |
| `build/sprint164/declarations/sparse_matrix.h.normalized.txt` | 88 |
| `build/sprint164/declarations/selected-public-headers.before.normalized.txt` | 346 |

## After-State Re-Capture

Day 10 captured the after-state checksum:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41  build/sprint164/declarations/selected-public-headers.after.normalized.txt
```

Day 10 after-state line counts:

| Local Generated File | Lines |
| --- | ---: |
| `build/sprint164/declarations/sparse_iterative.h.after.normalized.txt` | 151 |
| `build/sprint164/declarations/sparse_eigs.h.after.normalized.txt` | 102 |
| `build/sprint164/declarations/sparse_matrix.h.after.normalized.txt` | 88 |
| `build/sprint164/declarations/selected-public-headers.after.normalized.txt` | 346 |

## Before/After Comparison

The before and after checksums are identical:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

The Day 10 diff command produced no output:

```sh
diff -u \
  build/sprint164/declarations/selected-public-headers.before.normalized.txt \
  build/sprint164/declarations/selected-public-headers.after.normalized.txt
```

## Drift Investigation

No declaration drift was detected.

There were no changes to:

- function declarations;
- macro definitions;
- enum constants or values;
- typedef names or aliases;
- public struct names, field names, order, or types;
- include guards or public include dependencies;
- selected installed header names.

No intentional declaration drift needed documentation or review.

## Validation

- Day 3/Day 4 normalized declaration capture command re-run with `after`
  filenames.
- `diff -u` before/after comparison produced no output.
- before and after combined checksums matched exactly.
- `git status --short -- build/sprint164/declarations` showed no tracked
  local-evidence churn.
