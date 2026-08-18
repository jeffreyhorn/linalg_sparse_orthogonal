# Sprint 164 Day 3 Declaration Baseline Design

## Purpose

Day 3 defines the repeatable declaration-preservation method for the selected
Sprint 164 public-header cleanup batch before any header edits land.

Selected headers:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_matrix.h`

## Existing Validation Inputs

| Input | Finding |
| --- | --- |
| Sprint 157 quality surface map | Public header edits, including comment-only edits, require before/after declaration-preservation proof plus `make format && make lint && make test`. |
| Sprint 158 generated API policy | `make docs-check` owns local Doxygen generation and checked-in public-header page coverage; generated HTML remains ignored. |
| `docs/maintainer_guide.md` | Public headers own API-local call-site contracts and must not change declarations during comment cleanup. |
| `Makefile` | Provides `make docs`, `make api-docs-coverage`, `make docs-check`, `make format`, `make lint`, and `make test`. |
| Existing scripts | No dedicated declaration-preservation script exists today. Sprint 164 will use a recorded, repeatable normalization command unless later work adds a maintained helper. |

## Declaration Capture Method

Use two complementary records:

1. **Normalized non-comment header text.** Strip C block comments and `//`
   comments, trim trailing whitespace, collapse repeated blank lines, and keep
   preprocessor lines, typedefs, enums, structs, macros, and function
   declarations in source order.
2. **Checksum record.** Hash the normalized output so Day 10 can verify exact
   equivalence or name every drift explicitly.

This method is intentionally conservative. Comment-only cleanup should produce
an identical normalized declaration file and identical checksum. If it does
not, Day 10 must treat the difference as declaration drift until reviewed.

## Baseline Capture Command

Day 4 should run:

```sh
mkdir -p build/sprint164/declarations
python3 - <<'PY'
from pathlib import Path
import hashlib
import re

headers = [
    Path("include/sparse_iterative.h"),
    Path("include/sparse_eigs.h"),
    Path("include/sparse_matrix.h"),
]
out_dir = Path("build/sprint164/declarations")
out_dir.mkdir(parents=True, exist_ok=True)

def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text

def normalize(path: Path) -> str:
    lines = []
    previous_blank = False
    for raw in strip_comments(path.read_text()).splitlines():
        line = raw.rstrip()
        if not line.strip():
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"

combined = []
for header in headers:
    normalized = normalize(header)
    header_out = out_dir / f"{header.name}.normalized.txt"
    header_out.write_text(normalized)
    combined.append(f"===== {header.as_posix()} =====\n{normalized}")

bundle = "\n".join(combined)
bundle_path = out_dir / "selected-public-headers.before.normalized.txt"
bundle_path.write_text(bundle)
digest = hashlib.sha256(bundle.encode()).hexdigest()
(out_dir / "selected-public-headers.before.sha256").write_text(
    f"{digest}  {bundle_path.as_posix()}\n"
)
print(digest)
PY
```

Day 10 should run the same command with `after` file names and compare:

```sh
diff -u \
  build/sprint164/declarations/selected-public-headers.before.normalized.txt \
  build/sprint164/declarations/selected-public-headers.after.normalized.txt
```

Expected result for declaration-preserving cleanup: no diff.

Generated declaration evidence under `build/` remains local and ignored. Day 4
and Day 10 artifacts should summarize the checksum and any diff result rather
than committing generated baseline files.

## Drift Taxonomy

Treat each of the following as declaration drift unless explicitly reviewed and
documented before merging:

| Drift Class | Examples |
| --- | --- |
| Function declaration drift | Added, removed, renamed, reordered, or type-changed function declarations; parameter additions/removals; parameter type changes; constness changes. |
| Typedef drift | Added, removed, renamed, or re-aliased public typedefs. |
| Enum drift | Added, removed, renamed, reordered, or value-changed public enum constants. |
| Struct layout drift | Added, removed, renamed, reordered, or type-changed public struct fields. |
| Macro drift | Added, removed, renamed, or value-changed public macros and include-guard macros. |
| Visibility/install drift | Public header renamed, moved, excluded from install, or newly included in install without review. |
| Preprocessor contract drift | Changed `#include`, `#if`, `#ifdef`, `#define`, or compile-time configuration behavior. |
| ABI-adjacent drift | Any change that could affect downstream binary layout, even if ABI compatibility is not claimed. |

## Acceptable Non-Signature Edits

Allowed Sprint 164 cleanup edits, provided the normalized declaration diff is
empty:

- Doxygen or plain comment wording;
- section headings and comment organization;
- ownership/lifetime explanations;
- NULL/error-return explanations;
- output-buffer and result-field explanations;
- option-default explanations;
- backend behavior caveats that do not imply superiority;
- cross-link comments to README, tutorial, cookbook, solver-selection, API
  reference, or maintainer guide;
- removal of maintainer-history prose from public headers when the API-local
  contract remains clear;
- non-claim wording for ABI, package, runtime-loader, backend superiority,
  performance, platform, and state-of-the-art boundaries.

## Generated API Reference Check

After header comments change, Sprint 164 must apply Sprint 158 policy:

```sh
make docs-check
```

Expected result:

- Doxygen generation succeeds;
- API docs page coverage passes for checked-in public headers;
- generated `docs/api/html/` remains ignored and uncommitted.

## Required Quality Gate After Header Edits

Because the selected batch edits public headers, Day 12 must run:

```sh
make format && make lint && make test
```

This remains required even if all header edits are comment-only.

## Claim-Sensitive Scan

When public header or API docs wording changes, run a focused scan on touched
headers and related docs:

```sh
rg -n "ABI|shared-library|runtime-loader|package-manager|backend superiority|portable performance|state-of-the-art|hosted|release proof" \
  include/sparse_iterative.h include/sparse_eigs.h include/sparse_matrix.h \
  README.md docs/api_reference.md docs/tutorial.md docs/cookbook.md \
  docs/solver_selection.md docs/maintainer_guide.md
```

Hits are acceptable only when they are explicit non-claims or bounded evidence
statements.

## Day 4 Handoff

Day 4 should:

1. run the baseline capture command;
2. record the checksum in the Day 4 artifact;
3. summarize current generated API-reference state for the selected headers;
4. map current README/tutorial/cookbook/solver-selection references to the
   selected APIs;
5. identify pre-existing inconsistencies before edits begin.

## Validation Notes

Day 3 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 3.

## Completion Check

- Declaration-preservation proof can be repeated.
- Signature drift rules are explicit before editing.
- Comment cleanup boundaries are clear.
