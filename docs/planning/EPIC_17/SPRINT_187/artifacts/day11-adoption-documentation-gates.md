# Sprint 187 Day 11: Adoption and Documentation Gates

## Purpose

Define exact acceptance gates for Sprint 194 adoption and API coherence work.
The goal is to make first use easier without widening package, platform,
performance, external-comparison, API, ABI, or state-of-the-art claims beyond
the evidence selected in Sprints 188 through 193.

## Current Documentation Ownership Split

| Surface | Current owner role | Sprint 194 risk |
| --- | --- | --- |
| `README.md` | Front-door project orientation, workflow chooser, compact status, and links to deeper docs. | Can become too long or repeat maintainer policy instead of guiding first use. |
| `INSTALL.md` | Operational setup, staged installs, installed consumers, package boundaries, and install validation. | Can imply package-manager, shared-library, or Windows parity if simplified carelessly. |
| `docs/tutorial.md` | Longer learning path and usage flow after the README. | Can duplicate cookbook/API material or carry stale status/diagnostic wording. |
| `docs/cookbook.md` | Task-oriented recipes for construction, CSR/CSC, solves, and Matrix Market workflows. | Can drift from solver-selection and API-reference truth. |
| `docs/solver_selection.md` | Solver-choice guidance, failure modes, and evidence boundaries by solver family. | Can imply broader correctness or external parity than selected evidence proves. |
| `docs/api_reference.md` | Source-controlled API reference entry point and header ownership map. | Can drift from public headers or generated Doxygen coverage. |
| `docs/api/html/` | Ignored local generated Doxygen output. | Must not become hosted, retained, or source-controlled API publication evidence. |
| `examples/README.md` | Executable example selection and example-local caveats. | Can present examples as stronger proof than tests/reports provide. |
| `examples/*.c` and `examples/cmake_example/` | Buildable adoption examples and installed CMake consumer sample. | Can become stale when API or install guidance changes. |
| `include/*.h` | Public declarations and Doxygen comments. | Can accumulate tutorial/policy narrative better owned by docs. |
| `docs/maintainer_guide.md` | Maintainer policy, validation interpretation, claim boundaries, and provenance links. | Can leak historical planning detail into user-facing docs if copied outward. |

## Sprint 194 Gate: Adoption Audit

| Requirement | Acceptance criteria | Failure state |
| --- | --- | --- |
| Surface inventory | README, INSTALL, tutorial, cookbook, solver selection, API reference, examples, selected public headers, and maintainer guide are listed with current owner role. | A user-facing surface changes without ownership context. |
| Duplication review | Repeated install, support-tier, solver-choice, diagnostics, package, Windows, report, comparison, and performance text is either consolidated or deliberately retained. | Same claim appears in multiple places with different support wording. |
| Friction review | First-use paths identify the smallest build, install, example, and API-reading route for a new user. | User must read planning artifacts or maintainer policy to start. |
| Evidence reconciliation | Public docs reflect Sprint 188 package, Sprint 190 Windows, Sprint 191 comparison, and Sprint 192 performance outcomes. | Docs pre-promote future support or lag behind proven support. |
| Planning separation | Historical sprint artifacts are linked only when they explain current support boundaries. | User-facing docs require planning archaeology for normal workflows. |

## Sprint 194 Gate: Support/Readiness Matrix

Sprint 194 must add or update a compact support/readiness matrix with these
rows:

| Row | Required content |
| --- | --- |
| Source build | Maintained local Make and CMake build/test routes. |
| Installed static package | Make install/`pkg-config` and CMake install/export support, including exact validation owners. |
| Package manager | Exact Sprint 188 Homebrew local proof outcome and retained provider non-claims. |
| Windows | CMake-first/static-first support plus Sprint 189/190 PowerShell/report outcome and retained parity non-claims. |
| macOS/Linux | Reviewed/supplemental support tiers that already exist without overpromoting platform parity. |
| Generated reports | Selected oracle/comparison/benchmark freshness lanes and stale-output boundaries. |
| External comparison | Exact bounded comparison families, fixture scope, and broad-parity non-claims. |
| Performance | Methodology-bound selected lane and portable-performance non-claims. |
| API docs | Source-controlled reference path, local Doxygen generation, and hosted/generated-output non-claims. |
| Reliability | Selected failure-path proof lanes and non-exhaustive reliability boundary. |

The matrix should give users a compact truth surface and link to deeper docs
instead of embedding sprint history.

## Sprint 194 Gate: Installed Consumer Tutorial

| Requirement | Acceptance criteria |
| --- | --- |
| Make/`pkg-config` route | Shows a minimal Unix installed consumer using `make install`, `pkg-config --cflags --libs sparse`, compile, run, and uninstall/staged-prefix caveats. |
| CMake route | Shows a minimal external project using `find_package(Sparse REQUIRED)` and `Sparse::sparse_lu_ortho`. |
| Example alignment | Links to `examples/cmake_example/` and executable examples without duplicating their full source. |
| Boundary wording | States static-first installed package support and excludes package-manager, shared-library, dynamic ABI, and broad platform claims. |
| Validation owner | Points maintainers to `tests/test_install.sh` and `tests/test_cmake_install.sh` when changing install or downstream-consumer docs. |

## Sprint 194 Gate: Diagnostics Coherence

Diagnostics wording must use consistent concepts across direct, iterative,
QR/SVD, and eigensolver workflows:

| Area | Required wording pattern |
| --- | --- |
| Status codes | Name `SPARSE_OK`, `SPARSE_ERR_ALLOC`, `SPARSE_ERR_BADARG`, `SPARSE_ERR_NOT_SPD`, `SPARSE_ERR_SINGULAR`, and convergence errors only where they are part of the public contract. |
| Residuals | Distinguish fixture-local residual evidence from broad solver correctness. |
| Convergence | Explain iterative/eigensolver non-convergence as a result state, not a crash or package/platform issue. |
| Retry | State when retry-after-reset or caller cleanup is proven and when it remains owner-local. |
| Cleanup | Keep ownership and free/destroy calls explicit in examples and tutorial snippets. |
| Unsupported breadth | Keep broad numerical, platform, package, ABI, performance, and state-of-the-art non-claims paired with any evidence summary. |

## Sprint 194 Gate: Header Narrative Cleanup

Public headers should keep declarations, Doxygen contracts, parameter/result
semantics, ownership notes, and essential caveats. Sprint 194 may move broad
workflow narrative to user docs when all of these remain true:

- declarations are unchanged;
- Doxygen coverage remains complete;
- public semantics and status-code contracts remain unchanged;
- examples/tutorial/cookbook receive any removed workflow guidance;
- header-specific guards still pass;
- generated API freshness is updated or explicitly left local-only;
- no ABI, package, platform, or performance claim is added.

Header cleanup must not delete necessary ownership, allocation-failure,
convergence, or cleanup notes just because they are verbose.

## Required Validation Commands

Minimum docs-only validation:

```sh
git diff --check
```

Recommended Markdown link check:

```sh
python3 - <<'PY'
from pathlib import Path
import re
missing = []
for path in sorted(Path('.').rglob('*.md')):
    if '.git' in path.parts or 'build' in path.parts:
        continue
    text = path.read_text(encoding='utf-8')
    for match in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        target = match.group(1).split('#', 1)[0]
        if not target or '://' in target or target.startswith('mailto:'):
            continue
        candidate = Path(target[1:]) if target.startswith('/') else (path.parent / target).resolve()
        if not candidate.exists():
            missing.append((str(path), target))
if missing:
    for path, target in missing:
        print(f'missing link: {path} -> {target}')
    raise SystemExit(1)
print('markdown links ok')
PY
```

API/Doxygen validation when public headers or API docs change:

```sh
make docs-check
make api-docs-freshness
```

Header/doc guard validation when matching surfaces change:

```sh
bash scripts/check_qr_header_docs_guard.sh
bash scripts/check_lu_header_docs_guard.sh
```

Install and downstream-consumer validation when install docs or examples
change:

```sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
```

Example validation when executable examples change:

```sh
make examples
```

C/header validation:

```sh
make format
make lint
make test
```

`make format && make lint && make test` is mandatory whenever `.c` or `.h`
files change.

## Retained Non-Claims

Sprint 194 does not promote:

- new solver behavior;
- broad numerical correctness;
- broad external-library parity;
- package-manager distribution beyond Sprint 188 proof;
- shared-library support;
- dynamic ABI support;
- broad Windows parity;
- Windows Makefile or Windows `pkg-config` execution parity;
- portable performance or performance superiority;
- hosted generated API publication;
- release-ready benchmark claims;
- unqualified state-of-the-art sparse linear algebra status.

## Completion Gate

Sprint 194 is complete when user-facing docs present a shorter, coherent
adoption path; support/readiness truth is compact and evidence-backed;
diagnostic wording is consistent across major workflows; examples and installed
consumer guidance build or remain guarded; public headers preserve declarations
and Doxygen coverage; and every widened or retained claim matches the evidence
from Sprints 188 through 193.

Sprint 194 must stop if docs imply unsupported package, platform, ABI,
performance, external-parity, hosted API, or state-of-the-art claims, or if
required docs/examples/install/header validation fails.

## Validation

Day 11 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
