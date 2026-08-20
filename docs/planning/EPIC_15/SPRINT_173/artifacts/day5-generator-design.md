# Day 5: Generator Command Normalization Design

## Purpose

Design the command, target, output-path, cleanup, and staging behavior needed
for the selected Sprint 173 generated API HTML path.

## Selected Path Recap

Day 4 selected guarded local-only generated API HTML with stronger
freshness/staging enforcement.

This means Sprint 173 should preserve:

- `docs/api/` as ignored generated output;
- `docs/api_reference.md` plus checked-in public headers as source-controlled
  API truth;
- `make docs-check` as the selected local command for generation plus page
  coverage;
- no hosted, committed, or CI artifact-only generated API HTML claim.

## Current Command Chain

Day 5 ran:

```bash
make -n docs-check
```

Dry-run output:

```text
echo "Generating API documentation with Doxygen..."
doxygen Doxyfile
echo "Documentation generated in docs/api/html/"
python3 scripts/check_api_docs_coverage.py
```

The current command chain is already concise and appropriate:

| Command | Behavior | Day 5 design decision |
| --- | --- | --- |
| `make docs` | Runs `doxygen Doxyfile`. | Keep as the raw local generation command. |
| `make api-docs-coverage` | Runs `scripts/check_api_docs_coverage.py`. | Keep as the page-coverage check. |
| `make docs-check` | Runs generation and page coverage. | Keep as the selected local validation command. |

Day 5 does not recommend renaming `make docs-check`; changing the command name
would create documentation churn without improving the selected local-only
contract.

## Output Path Design

| Path | Current behavior | Day 5 design |
| --- | --- | --- |
| `docs/api/` | Ignored generated root. | Preserve as ignored local output. |
| `docs/api/html/` | Doxygen HTML output. | Preserve as local generated HTML. |
| `docs/api/html/index.html` | Generated local index. | Keep local-only; never cite as source-controlled evidence. |
| `docs/api/html/search/` | Generated Doxygen search assets. | Keep ignored with the rest of `docs/api/`. |
| `include/sparse_version.h` | Generated installed header and ignored. | Keep outside current Doxygen page expectations. |

Day 5 ran:

```bash
git check-ignore -v docs/api docs/api/html docs/api/html/index.html
```

All three paths are covered by `.gitignore` rule `docs/api/`.

## Tracking And Staging Baseline

Day 5 checked:

```bash
git ls-files docs/api
git ls-files --others --exclude-standard docs/api
```

Both commands reported zero paths.

Interpretation:

- no generated API files under `docs/api/` are tracked;
- no non-ignored untracked generated API files under `docs/api/` are visible;
- existing generated HTML is ignored local state, not source-controlled state.

## Command Normalization Design

The selected command structure for Sprint 173 is:

| Use case | Command | Expected result |
| --- | --- | --- |
| Generate local Doxygen HTML only | `make docs` | Writes ignored output under `docs/api/html/`. |
| Check generated page coverage only | `make api-docs-coverage` | Verifies expected generated pages for checked-in public headers. |
| Validate local generated API view | `make docs-check` | Generates HTML and verifies page coverage. |
| Prove local-only tracking/staging boundary | new focused guard, proposed `scripts/check_api_docs_local_only.sh` | Verifies ignore/tracking/staging rules for `docs/api/`. |
| Aggregate local generated API validation | proposed Make target, name to decide Day 6 | Runs `docs-check` plus local-only guard if Make integration is selected. |

Day 5 recommends a small focused guard rather than overloading
`scripts/check_api_docs_coverage.py`. Page coverage and repository staging are
different concerns, and separate failure messages will be clearer.

## Proposed Day 6 Guard Behavior

Day 6 should implement a focused local-only guard with these checks:

1. `docs/api/` is ignored by the repository.
2. `docs/api/html/` is ignored by the repository.
3. `docs/api/html/index.html` is ignored if it exists.
4. `git ls-files docs/api` returns no tracked generated API files.
5. `git diff --cached --name-only -- docs/api` returns no staged generated API
   files.
6. `git ls-files --others --exclude-standard docs/api` returns no non-ignored
   untracked generated API files.
7. Failure messages state that generated API HTML remains local-only unless a
   future publication decision selects committed output.

Optional checks, if Day 6 keeps scope small:

- check that `docs/api_reference.md` still says generated HTML is local-only;
- check that `docs/maintainer_guide.md` still says generated HTML is not
  source-controlled, hosted, or release evidence.

Day 5 recommends implementing these optional wording checks only if they remain
low-noise. Otherwise Day 9 and Day 10 should own documentation wording.

## Cleanup Behavior

The selected local-only path does not require automatic cleanup before every
generation. Doxygen can update `docs/api/html/` in place, and the output is
ignored.

Required cleanup guidance:

- do not remove or rewrite ignored local generated HTML unless a focused guard
  or docs-check failure requires it;
- do not add automatic deletion of `docs/api/` to `make docs-check` unless
  stale-file behavior becomes a proven failure;
- if stale generated files become a problem later, add a separate
  `docs-clean`-style target before generation rather than silently deleting
  user-local output in unrelated targets.

Day 5 does not recommend changing cleanup behavior yet.

## Failure Behavior

| Failure | Desired message/action |
| --- | --- |
| Doxygen command fails | `make docs-check` fails immediately. |
| Expected header page missing | `api-docs-coverage: FAIL` with missing header/page mapping. |
| `docs/api/` not ignored | local-only guard fails and points to `.gitignore`/publication decision. |
| generated API file tracked | local-only guard fails and lists tracked paths. |
| generated API file staged | local-only guard fails and lists staged paths. |
| generated API file visible as non-ignored untracked | local-only guard fails and lists paths. |
| docs claim hosted/committed output | Day 9/Day 10 claim scans or docs review should fail. |

## CI Behavior

Sprint 173 has not selected hosted or artifact-only generated API HTML.

Day 5 design therefore keeps CI behavior conservative:

- do not add Doxygen HTML publishing;
- do not upload `docs/api/html/` as an artifact;
- do not claim hosted generated API freshness;
- allow a future sprint to add a docs-check CI lane only after defining
  install dependencies, retention, branch semantics, and claim wording.

If Day 6 adds a Make target for local-only enforcement, it may remain local
unless later days deliberately add it to a broader quality target.

## Docs And Guard Updates Needed

Recommended Day 6 files:

- add `scripts/check_api_docs_local_only.sh`;
- optionally add a Make target such as `api-docs-local-only` and a composed
  target such as `api-docs-validate`;
- update no user docs unless the new command name needs documentation.

Recommended Day 7/Day 8 follow-through:

- decide whether the guard should be freshness-only, staging-only, or both;
- implement fail-mode proof where practical;
- record generated-output staging evidence.

Recommended Day 9/Day 10 follow-through:

- review `README.md`, `docs/api_reference.md`, and `docs/maintainer_guide.md`
  for navigation wording;
- avoid hosted/generated artifact links;
- preserve package, ABI, platform, performance, external-parity, and
  state-of-the-art non-claims.

## Day 6 Implementation Checklist

Day 6 should proceed only within this scope:

1. keep `Doxyfile` input/output policy unchanged;
2. keep `docs/api/` ignored;
3. keep `make docs-check` behavior unchanged unless a small aggregate target is
   added;
4. add focused local-only guard logic;
5. run the new guard;
6. run `git diff --check`;
7. if any script or Makefile changes are made, run the focused command(s) they
   affect;
8. do not stage generated HTML.

## Completion Check

Day 5 completion criteria are met:

- implementation is scoped before editing generator commands;
- output paths and staging rules are unambiguous;
- generated output cannot drift into source control silently by design once the
  Day 6 guard is implemented.

No `.c` or `.h` files changed on Day 5, so the full C quality gate is not
required for this day.
