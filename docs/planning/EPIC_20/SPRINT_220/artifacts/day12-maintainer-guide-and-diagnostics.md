# Sprint 220 Day 12: Maintainer Guide And Diagnostics

## Guide Update

Integrated the Day 11 outline into the existing **API reference and generated
Doxygen HTML** section of `docs/maintainer_guide.md`. The guide now gives:

- The `PyYAML==6.0.3` pin, a local virtual-environment setup, and the Linux/macOS
  CI provisioning context.
- The integrated `make api-docs-local-only` route and focused reader, policy,
  and mutation test commands; the Python tests do not replace the Doxygen gate.
- Supported YAML forms, shell precedence, path-input normalization, explicit
  POSIX/PowerShell staging coverage, and the `cmd`/unknown-shell limit.
- Fail-closed reader and policy cases, direct commands for separating setup,
  parse, and publication errors, and the file/job/step diagnostic scope.
- Repair guidance that preserves the local-only product decision. A passing
  check is not publication authorization.

The combined guard still has legacy workflow scans pending Day 13 removal.
Its first diagnostic may be less specific than the structured policy's, so
the guide recommends a direct policy rerun. A staged producer may be named
instead of its downstream upload step; both must be inspected in the same
job. The known checkout-into-`docs/` false positive remains Day 13 work,
not an allowed publication exception.

## Product Wording Review

`README.md` already calls `docs/api/html/` local-only ignored output and
explicitly excludes hosting, retained artifacts, source control, and release
evidence. `docs/api_reference.md` keeps this page and checked-in public
headers as the durable route and says generated HTML is neither hosted nor a
source-controlled publication surface. Neither file needed an edit. The
existing maintainer-guide decision text was retained.

## Local Checks

| Check | Result |
| --- | --- |
| Pinned virtual environment | PyYAML 6.0.3 present. |
| Reader suite | 20 tests passed. |
| Structured policy suite | 17 tests passed. |
| Mutation suite | 3 methods passed, including the 13-case matrix. |
| Direct reader CLI | Parsed 3 maintained workflows. |
| Direct policy CLI | Checked 110 job/step records. |
| Combined `check_api_docs_local_only.sh` | Passed Git-state, Doxyfile, wording, and workflow checks; 110 structured records. |
| Routing suite | Passed. |

These are local checks of the current checkout. Day 11 ran the full
`make api-docs-local-only` gate; Day 12 did not rerun its Doxygen and frozen
fixture phases. Hosted CI and Day 13 legacy-scan retirement remain separate
evidence. No C or header files changed.
