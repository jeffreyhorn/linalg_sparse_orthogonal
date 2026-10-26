# Sprint 220 Day 8: Shell Staging And Upload Policy

## Decision Boundary

`scripts/api_docs_workflow_staging.py` parses YAML-rendered `run` scalars using
the effective shell family from the reader. POSIX commands use `shlex` after
shell-comment removal; command separators and newlines remain boundaries.
PowerShell parsing preserves Windows backslashes and handles backtick line
continuations, quotes, named `-Path`/`-Destination` operands, and positional
`Copy-Item`/`Move-Item` operands. Neither parser searches inert `echo` text for
commands. Unrelated jobs and workflows never share a staging relationship.

The policy recognizes POSIX `cp`, `mv`, `rsync`, `tar`, `zip`, and `7z` archive
forms, plus PowerShell `Copy-Item`, `Move-Item`, and `Compress-Archive`.
Multiple source operands are inspected. Within a job, ordered staging
destinations remain tainted if a later staging command copies or moves them.
A matching later upload/deploy/release action path or explicit shell publisher
input rejects at the originating step with a file/line/column diagnostic.
Shell publishers covered here include `aws s3 cp/sync`, `rclone`, `gh release
upload`, `wrangler pages deploy`, `netlify deploy --dir`, `az storage blob
upload --file`, `scp`/remote `rsync`, and `surge` when they name a local input.
Unknown implicit publication inputs remain for Day 9 edge-case review.

The old staging-only regex branches in
`scripts/check_api_docs_local_only.sh` are retired so a docs copy followed by
an unrelated `build/` upload is not a false rejection. The old direct-path
and dynamic publication checks remain active pending Day 9 differential
comparison. No generated HTML publication was authorized.

## Paired Evidence

| Route | Expected decision |
| --- | --- |
| `cp -R docs artifact/` then upload `artifact/`; `mv`; `tar`; `zip`; `7z` | Reject at the staging step. |
| `cp build/ docs/ artifact/`; `tar cf ... build/ docs/`; grouped `(cp ...)` | Reject despite extra operands or grouping. |
| `Copy-Item -Path` backtick continuation then `.\docs\*` and upload `artifact/` | Reject without discarding Windows separators. |
| `cp docs staging/`, then `mv staging artifact/`, then upload `artifact/` | Reject the transitive same-job route. |
| `cp docs artifact/`, then `aws s3 cp artifact/ ...` or matching release asset | Reject explicit shell publication input. |
| Docs staged to `artifact/`, but action uploads `build/` | Allow. |
| Quoted staging text, literal shell comment, folded shell comment, or staging in a different job | Allow when no active matched route exists. |

The focused policy suite passed 16 tests after the final parser changes.
The integrated `make api-docs-local-only` gate passed with Doxygen coverage,
the real-workflow guard, and the 190-function historical fixture corpus.
Reader (20), policy-input (7), routing, shell syntax, and `git diff --check`
also passed locally. No C or header files changed, so the C format/lint/test
chain is not required for this day.
