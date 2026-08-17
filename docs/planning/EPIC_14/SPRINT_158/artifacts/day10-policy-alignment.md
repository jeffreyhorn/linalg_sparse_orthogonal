# Day 10 Policy Alignment

## Scope

Day 10 aligns public and maintainer documentation with the Sprint 158 generated
API HTML decision.

The selected policy remains:

- source-controlled API truth stays in `docs/api_reference.md` and checked-in
  public headers under `include/`;
- generated Doxygen HTML stays local-only and ignored under `docs/api/html/`;
- `make docs-check` is the maintained local freshness and page-coverage guard;
- generated `sparse_version.h` remains an installed-header policy row, not an
  expected generated Doxygen page under the current input set.

## Documentation Updates

| File | Day 10 change |
| --- | --- |
| `docs/api_reference.md` | Replaced loose `make docs` freshness wording with `make docs-check`, local-only generated-output boundaries, and generated-version-header ownership wording. |
| `docs/maintainer_guide.md` | Replaced committed-output freshness rules with the selected local-only/ignored generated HTML policy and `make docs-check` interpretation. |
| `README.md` | Reviewed; no route change required because the front door already points API users to `docs/api_reference.md` and public headers. |
| `docs/tutorial.md` | Reviewed; no route change required because the tutorial already points declaration-oriented users to `docs/api_reference.md` and public headers. |

## Freshness Policy

Generated API HTML is current only for the branch and checkout where
`make docs-check` has just passed. It is not a durable source-controlled,
hosted, release, or platform-parity evidence surface.

`make docs-check` validates:

- Doxygen generation for the configured input set;
- generated page coverage for checked-in public headers;
- the generated `sparse_version.h` policy row by excluding it from expected
  Doxygen pages under the current input set.

## Source-Header-First Ownership

Day 10 preserves the source-header-first model:

- exact declarations and call-site contracts remain in checked-in public
  headers;
- `docs/api_reference.md` stays a compact user-facing index;
- local Doxygen output is a convenience rendering of configured public-header
  input, not a stronger contract.

## Non-Claim Boundaries

The aligned docs do not claim:

- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad platform parity;
- external-library parity;
- portable performance;
- hosted generated documentation;
- source-controlled generated HTML;
- completeness beyond the configured Doxygen input set.

## Validation

Day 10 changed Markdown documentation only. Validation focused on documentation
and generated-doc policy consistency:

```text
make docs-check
git diff --check
```

Results:

- `make docs-check` passed with zero Doxygen warnings and complete coverage for
  18 checked-in public headers.
- `git diff --check` passed.
- A stale wording scan found no live-doc claims that generated HTML must be
  committed or hosted for the selected Sprint 158 policy.

## Day 11 Handoff

Day 11 should finalize the publication path by verifying `.gitignore`,
generated paths, `make docs-check`, and documentation wording all agree with
the local-only generated API HTML decision.
