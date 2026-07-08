# Sprint 113 Day 12: Integrated Validation Plan

## Purpose

Define the final validation matrix for Sprint 113 touched surfaces so Day 13 can
run validation without redesigning scope or omitting required checks.

## Touched-Surface Inventory

| Surface | Files | Change owner | Validation owner |
|---|---|---|---|
| Eigensolver behavior proof | `tests/test_eigs.c` | Day 4 grow-m sizing and retry behavior tests | focused eigensolver test plus full quality gate |
| Direct solver oracle cleanup | `tests/test_ldlt_csc.c` | Day 8 LDLT CSC external dense-reference oracle cleanup | focused LDLT CSC test plus full quality gate |
| SVD helper cleanup | `tests/test_svd_partial_helpers.h` | Day 10 partial-SVD `A*v ~= sigma*u` residual helper cleanup | focused SVD test plus full quality gate |
| Sprint 113 planning docs | `docs/planning/EPIC_10/SPRINT_113/PLAN.md` | Day 1 plan artifact creation | doc hygiene and local link checks |
| Sprint 113 working notes | `docs/planning/EPIC_10/SPRINT_113/WORKING_NOTES.md` | Day 1-12 running evidence log | doc hygiene and local link checks |
| Sprint 113 artifacts | `docs/planning/EPIC_10/SPRINT_113/artifacts/*.md` | Day 1-12 proof artifacts | doc hygiene and local link checks |

No current Sprint 113 diff touches:

- `Makefile`;
- `CMakeLists.txt`;
- `cmake/`;
- `include/`;
- `src/`.

## Focused Validation Matrix

| Owner | Command | Required result |
|---|---|---|
| Eigensolver grow-m behavior | `make build/test_eigs && build/test_eigs` | `test_eigs` passes, including grow-m sizing/retry tests added in Day 4. |
| LDLT CSC external oracle cleanup | `make build/test_ldlt_csc && build/test_ldlt_csc` | `test_ldlt_csc` passes, including external dense-reference oracle tests. |
| SVD partial-vector residual cleanup | `make build/test_svd && build/test_svd` | `test_svd` passes, including partial-SVD `A*v ~= sigma*u` residual diagnostics. |

## Full Quality Gate Decision

The full quality gate is required on Day 13 because Sprint 113 changes `.c` and
`.h` files:

- `tests/test_eigs.c`;
- `tests/test_ldlt_csc.c`;
- `tests/test_svd_partial_helpers.h`.

Required command:

```sh
make format && make lint && make test
```

Required result:

- formatting completes;
- strict warning build completes;
- `clang-tidy` completes;
- `cppcheck` completes;
- all tests pass.

## Build and Membership Drift Checks

Because Day 11 confirmed no build-system or source-list files changed, Day 13
does not need a new CTest count assertion run. It must still verify drift with
the current worktree:

```sh
git diff --name-only -- Makefile CMakeLists.txt cmake include src tests | sort
git diff --name-only -- Makefile CMakeLists.txt cmake include src | sort
```

Expected result:

- first command lists only:
  - `tests/test_eigs.c`;
  - `tests/test_ldlt_csc.c`;
  - `tests/test_svd_partial_helpers.h`;
- second command prints no files.

If either command reports Make/CMake, `cmake/`, `include/`, or `src/` drift,
Day 13 must stop and reassess source-list, install-header, API, and reviewed
CTest membership scope before closeout.

## Documentation Hygiene Checks

Day 13 must run:

```sh
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_113 \
  tests/test_eigs.c tests/test_ldlt_csc.c tests/test_svd_partial_helpers.h
```

Day 13 must also run the local Markdown link checker over the Sprint 113 plan,
working notes, and all artifacts:

```sh
perl -MFile::Basename=dirname -MFile::Spec -ne 'while (/\[[^\]]+\]\(([^)]+)\)/g) { $u = $1; $u =~ s/[[:space:]].*$//; $u =~ s/#.*$//; next if $u eq q{} || $u =~ /^(https?:|mailto:)/; $p = File::Spec->catfile(dirname($ARGV), $u); print "$ARGV:$.: missing $u\n" unless -e $p; }' \
  docs/planning/EPIC_10/SPRINT_113/PLAN.md \
  docs/planning/EPIC_10/SPRINT_113/WORKING_NOTES.md \
  docs/planning/EPIC_10/SPRINT_113/artifacts/*.md
```

Expected result:

- no `git diff --check` output;
- no trailing-whitespace matches;
- no missing local Markdown links.

## Day 13 Execution Checklist

Run in this order:

1. `make build/test_eigs && build/test_eigs`
2. `make build/test_ldlt_csc && build/test_ldlt_csc`
3. `make build/test_svd && build/test_svd`
4. `git diff --name-only -- Makefile CMakeLists.txt cmake include src tests | sort`
5. `git diff --name-only -- Makefile CMakeLists.txt cmake include src | sort`
6. `make format && make lint && make test`
7. `git diff --check`
8. trailing-whitespace check
9. local Markdown link check
10. `git status --short --branch`
11. `git diff --stat`

## Blocking Criteria

Day 13 must stop before closeout if any of the following occurs:

- focused eigensolver validation fails;
- focused LDLT CSC validation fails;
- focused SVD validation fails;
- full quality gate fails;
- `Makefile`, `CMakeLists.txt`, `cmake/`, `include/`, or `src/` drift appears;
- documentation hygiene checks fail;
- local Markdown links are missing.

## Closeout Evidence to Capture

The Day 13 artifact should record:

- focused eigensolver result;
- focused LDLT CSC result;
- focused SVD result;
- full quality gate result;
- build/source/API/CTest drift result;
- documentation hygiene result;
- final `git status --short --branch`;
- final `git diff --stat`.
