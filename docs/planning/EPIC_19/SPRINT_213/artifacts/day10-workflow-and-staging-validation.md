# Sprint 213 Day 10: Workflow And Staging Validation

## Summary

Day 10 validates workflow paths, artifact staging, archive commands, and
publication semantics for the selected stronger local-only generated API
policy. The implementation hardens direct workflow generated-output scans so
quoted `#` characters cannot hide `docs/api` references.

## Workflow Audit

| Surface | Finding |
| --- | --- |
| Workflow files | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and `.github/workflows/windows-ci.yml` are the current workflow files. |
| Existing artifact uploads | Existing uploads are comparison freshness, selected performance freshness, dead-code, and coverage artifacts. |
| Generated API paths | No current workflow references `docs/api`, `docs/api/html`, or a generated API archive/deploy path. |
| Publication commands | No current workflow contains generated-doc Pages deployment, release upload, rclone/aws generated-doc publication, or docs archive publication commands. |

## Implementation

| Surface | Change | Rationale |
| --- | --- | --- |
| Direct workflow generated-path scans | `scripts/check_api_docs_local_only.sh` now uses the same quote-aware YAML comment stripper for direct generated API reference scans as for publication semantics scans. | Prevents a quoted `#` before `docs/api/html` from being treated as a YAML comment and hiding a generated-output reference. |
| Workflow extension coverage | `tests/test_api_docs_local_only_guard.py` adds a `.yaml` workflow fixture. | Confirms both GitHub workflow extensions are covered by the local-only guard. |
| Quoted hash fixture | `tests/test_api_docs_local_only_guard.py` adds a quoted-`#` direct generated API path regression. | Proves direct generated-output references fail even when a shell string contains `#` earlier on the line. |

## Preserved Boundaries

Day 10 does not:

- add generated API artifact uploads;
- add GitHub Pages or other generated API deployment;
- allow broad `docs/` or repository-root publication;
- allow generated API archive publication;
- commit `docs/api/`;
- change public C headers or implementation files.

## Validation

Ran for Day 10 closeout:

```sh
python3 tests/test_api_docs_local_only_guard.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

`python3 tests/test_api_docs_local_only_guard.py` and `make
api-docs-freshness` passed. Final hygiene checks are recorded in the final turn
summary.

No `.c` or `.h` files are modified, so the full C quality gate is not required
by the sprint instruction.

## Outcome

Sprint item 213.4 covers workflow references and generated-output staging for
the selected stronger local-only policy. Current project workflows do not
publish or archive generated API output, and the guard now treats direct
generated-output references consistently across quoted shell text and both
`.yml`/`.yaml` workflow extensions.
