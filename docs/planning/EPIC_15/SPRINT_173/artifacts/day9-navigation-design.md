# Day 9: Documentation Navigation Design

## Purpose

Design README and docs-index navigation for the supported generated API
reference surface after Sprint 173 selected local-only generated API HTML and
added the `api-docs-freshness` target.

## Current Navigation Map

| Surface | Current role | Day 9 assessment |
| --- | --- | --- |
| `README.md` front-door steps | Routes advanced users to API reference and maintainer evidence. | Already points to `docs/api_reference.md`; no structural rewrite needed. |
| `README.md` Adoption Map | Routes exact declarations to `docs/api_reference.md` and public headers. | Already correct. |
| `README.md` command list | Lists `make docs`, `make docs-check`, and API reference entry point. | Needs `make api-docs-freshness` after Day 8. |
| `docs/api_reference.md` | Source-controlled API reference entry point. | Needs freshness wording updated from `docs-check` to selected freshness target. |
| `docs/maintainer_guide.md` | Maintainer generated-doc policy and command guidance. | Needs selected freshness/local-only staging target named. |
| `docs/tutorial.md` | Learning path; routes advanced users to API reference. | Already sufficient; no Day 10 update needed. |
| `docs/cookbook.md` | Workflow recipes. | No generated API HTML navigation update needed. |
| `docs/solver_selection.md` | Solver-family selection and evidence interpretation. | No generated API HTML navigation update needed. |
| `INSTALL.md` | Static-first install/downstream consumer docs. | No generated API HTML navigation update needed. |
| `benchmarks/README.md` | Benchmark/report interpretation. | No generated API HTML navigation update needed. |

## Supported API Reference Surface

Day 10 should preserve this hierarchy:

1. README routes users to `docs/api_reference.md` for exact API declarations.
2. `docs/api_reference.md` routes exact declarations to checked-in public
   headers under `include/`.
3. Generated Doxygen HTML under `docs/api/html/` remains local-only generated
   output.
4. `make api-docs-freshness` is the selected local command that regenerates
   Doxygen HTML, checks page coverage, and verifies generated output remains
   ignored, untracked, and unstaged.

## Proposed README Wording

In the command list, keep:

```text
make docs       # generate Doxygen API reference (requires doxygen)
make docs-check # generate and check local Doxygen API page coverage
```

Add:

```text
make api-docs-freshness # selected local Doxygen freshness plus local-only staging guard
```

Keep:

```text
# API reference entry point: docs/api_reference.md
```

Do not add a generated HTML URL or artifact link.

## Proposed `docs/api_reference.md` Wording

Update the Generated HTML section to name both command layers:

- `make docs-check` runs Doxygen and checks page coverage;
- `make api-docs-freshness` runs the selected local freshness proof:
  generation, page coverage, and local-only staging enforcement.

Replace current freshness sentence:

```text
Treat it as current only for the branch and checkout where `make docs-check`
has just passed.
```

with wording like:

```text
Treat it as current only for the branch and checkout where
`make api-docs-freshness` has just passed.
```

Also update the stale/missing generated HTML fallback sentence so exact
declarations remain owned by public headers until `make api-docs-freshness`
passes.

## Proposed `docs/maintainer_guide.md` Wording

Update the command block from:

```bash
make docs-check
```

to:

```bash
make api-docs-freshness
```

Then explain:

- `make docs-check` still runs Doxygen and page coverage;
- `make api-docs-freshness` runs `docs-check` plus local-only generated-output
  staging enforcement;
- local generated output under `docs/api/html/` remains not
  source-controlled, hosted, artifact-published, or release evidence.

Update stale/partial guidance so public header comment changes are fresh only
after the selected freshness target passes, not just after an ambiguous local
generation run.

## Non-Claim Wording Constraints

Day 10 must avoid wording that implies:

- hosted generated API HTML;
- committed generated API HTML;
- CI artifact-only generated API HTML;
- generated API HTML as release evidence;
- generated installed-header Doxygen coverage;
- package-manager provider support;
- shared-library support;
- dynamic ABI stability;
- runtime-loader behavior;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- portable performance guarantees;
- external-library parity;
- state-of-the-art sparse linear algebra coverage.

## Validation Commands For Day 10

If Day 10 updates only README and docs wording, run:

```bash
make api-docs-freshness
git diff --check
```

Run targeted claim scans:

```bash
rg -n "hosted|committed|source-controlled|artifact|release evidence|package-manager|shared-library|dynamic ABI|runtime-loader|platform parity|portable performance|external-library parity|state-of-the-art" README.md docs/api_reference.md docs/maintainer_guide.md
```

If Day 10 touches package, ABI, runtime-loader, or package-manager wording in a
way that could affect deferral boundaries, also run:

```bash
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
```

No full C quality gate is needed unless Day 10 modifies `.c` or `.h` files.

## Completion Check

Day 9 completion criteria are met:

- users can find the supported API reference path by design;
- docs wording can be aligned with the Day 4 publication decision;
- unselected hosted, committed, artifact-only, package, ABI, platform,
  performance, external-parity, and state-of-the-art modes remain non-claims.

No `.c` or `.h` files changed on Day 9, so the full C quality gate is not
required for this day.
