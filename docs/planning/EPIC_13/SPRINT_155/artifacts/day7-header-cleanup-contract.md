# Sprint 155 Day 7 Header Cleanup Contract

## Purpose

Day 7 defines the cleanup contract for the selected Sprint 155 public-header
batch before implementation begins. It turns the Day 6 selection into allowed
edits, disallowed edits, declaration-preservation commands, maintainer review
rules, error-contract wording guidance, and the Day 8 implementation checklist.

Selected headers:

1. `include/sparse_ldlt.h`
2. `include/sparse_ic.h`
3. `include/sparse_eigs.h`
4. `include/sparse_analysis.h`

Day 8 should edit only `include/sparse_ldlt.h` and `include/sparse_ic.h`.
Day 9 should edit only `include/sparse_eigs.h` and
`include/sparse_analysis.h`.

## Allowed Edits

Header cleanup may:

- rewrite comments for clarity;
- shorten tutorial-scale examples when examples or tutorial docs already own
  the teaching path;
- replace sprint-history prose with current behavior and a doc handoff;
- clarify ownership and lifetime rules;
- clarify caller allocation and output-buffer shape;
- clarify NULL and `sparse_err_t` return behavior;
- clarify input mutation or non-mutation behavior;
- clarify shape, symmetry, SPD, identity-permutation, same-pattern, and
  backend preconditions;
- clarify zero-init defaults, option defaults, and result-field semantics;
- add concise non-claim wording when an API-local comment could otherwise be
  misread as package, ABI, platform, performance, external-parity,
  generated-report, or state-of-the-art proof;
- add cross-links to maintained owner docs when they reduce duplicated policy
  in the header.

## Disallowed Edits

Header cleanup must not:

- change public function declarations or signatures;
- change declaration order;
- change typedef names or layouts;
- change enum names, values, or order;
- change struct field names, order, types, or layout;
- change public macros or numeric values;
- change include guards;
- add, remove, or rename installed public headers;
- add or remove required includes unless a separate compile proof justifies it;
- change exported names;
- change documented default values;
- remove ownership/freeing requirements;
- remove error returns from function contracts;
- remove mutation/non-mutation guarantees;
- remove safety preconditions such as identity-permutation, symmetry, shape, or
  same-pattern requirements;
- imply shared-library support, dynamic ABI support, package-manager support,
  runtime-loader compatibility, broad Windows parity, portable performance,
  broad external-library parity, or state-of-the-art status.

If any disallowed edit appears necessary, Day 8 or Day 9 must stop and record a
separate API-change proposal rather than include it in cleanup.

## Declaration-Preservation Command Plan

Before Day 8 header edits, capture the baseline declaration-like surface:

```sh
rg -n "^[A-Za-z_][A-Za-z0-9_ \\*const]*[\\* ]+[A-Za-z_][A-Za-z0-9_]*\\([^;]*\\);$|^typedef |^typedef enum|^typedef struct|^#define |^#ifndef |^#define SPARSE_" \
  include/sparse_ldlt.h include/sparse_ic.h include/sparse_eigs.h include/sparse_analysis.h \
  > docs/planning/EPIC_13/SPRINT_155/artifacts/day8-header-declarations-before.txt
```

After Day 8 edits, capture the same surface:

```sh
rg -n "^[A-Za-z_][A-Za-z0-9_ \\*const]*[\\* ]+[A-Za-z_][A-Za-z0-9_]*\\([^;]*\\);$|^typedef |^typedef enum|^typedef struct|^#define |^#ifndef |^#define SPARSE_" \
  include/sparse_ldlt.h include/sparse_ic.h include/sparse_eigs.h include/sparse_analysis.h \
  > docs/planning/EPIC_13/SPRINT_155/artifacts/day8-header-declarations-after.txt
diff -u \
  docs/planning/EPIC_13/SPRINT_155/artifacts/day8-header-declarations-before.txt \
  docs/planning/EPIC_13/SPRINT_155/artifacts/day8-header-declarations-after.txt
```

For Day 9, repeat with Day 9 file names before and after the second tranche.

Focused diff checks after each tranche:

```sh
git diff -- include/sparse_ldlt.h include/sparse_ic.h
git diff --word-diff=porcelain -- include/sparse_ldlt.h include/sparse_ic.h
git diff --name-only -- '*.c' '*.h' '*.h.in'
git diff --check
```

After any public header edits in Days 8-9, run the required full gate:

```sh
make format && make lint && make test
```

## Claim Scan Plan

After each tranche, scan selected headers for unsupported positive claims:

```sh
rg -n "state-of-the-art|external-library parity|portable performance|performance guarantee|package-manager support|shared-library support|dynamic ABI|runtime-loader|broad Windows parity|Windows Makefile parity|Windows pkg-config parity" \
  include/sparse_ldlt.h include/sparse_ic.h include/sparse_eigs.h include/sparse_analysis.h
```

Matches are acceptable only when they are explicit non-claims or deferrals.

## Error-Contract Wording Guidance

Each edited function block should preserve or clarify:

- `SPARSE_OK` success behavior;
- `SPARSE_ERR_NULL` behavior for NULL inputs and outputs;
- `SPARSE_ERR_ALLOC` when allocation may fail;
- shape and domain errors such as `SPARSE_ERR_SHAPE`,
  `SPARSE_ERR_BADARG`, `SPARSE_ERR_NOT_SPD`, or `SPARSE_ERR_SINGULAR`;
- whether output pointers are set to NULL on error;
- whether caller-owned inputs are modified;
- which object the caller must free and with which free function;
- whether zeroed structs are valid for cleanup functions;
- whether inputs or outputs may alias.

If the existing function block lacks one of these details and the implementation
is not being inspected during the cleanup day, do not invent behavior. Record
the gap and defer it to the header summary unless existing tests/docs already
prove the wording.

## Maintainer Checklist

Use this checklist for each selected header:

| Check | Requirement |
| --- | --- |
| Scope | Comments only unless a separate API-change proposal exists. |
| Ownership | Caller-owned allocations, output buffers, and free functions remain visible. |
| Errors | Existing `SPARSE_ERR_*` returns are preserved. |
| Defaults | Zero-init and default-option behavior remains visible. |
| Mutation | Mutation/non-mutation and identity-permutation requirements remain visible. |
| Claims | Broad package, ABI, platform, performance, external-parity, generated-report, and state-of-the-art claims are absent or explicit non-claims. |
| Declarations | Declaration-like before/after capture is unchanged, or any differences are comment-only false positives explained in the artifact. |
| Quality | `git diff --check` and, after public header edits, `make format && make lint && make test` pass. |

## Maintainer Guidance Update

Day 7 updated `docs/maintainer_guide.md` under Documentation Ownership Rules
with a reusable public-header cleanup checklist. The update records that public
header cleanup is API-surface work even when it is intended to be comment-only
and names the required preservation and validation expectations.

## Day 8 Implementation Checklist

Day 8 should:

1. Capture `day8-header-declarations-before.txt`.
2. Edit only `include/sparse_ldlt.h` and `include/sparse_ic.h`.
3. Keep edits comment-only.
4. Preserve all declarations, typedefs, enum values, macros, struct fields,
   include guards, ownership rules, error returns, defaults, and preconditions.
5. Prefer concise doc handoffs over long maintainer policy blocks.
6. Capture `day8-header-declarations-after.txt`.
7. Diff before/after declaration captures.
8. Run `git diff --check`.
9. Run the unsupported-claim scan.
10. Run `make format && make lint && make test`.
11. Record the Day 8 cleanup summary and validation evidence.

## Day 7 Completion Check

- Allowed and disallowed header edits are defined.
- Declaration-preservation commands are defined.
- Claim scans are defined.
- Error-contract wording guidance is defined.
- Maintainer checklist exists.
- `docs/maintainer_guide.md` has reusable public-header cleanup guidance.
- Day 8 has a concrete implementation checklist.
