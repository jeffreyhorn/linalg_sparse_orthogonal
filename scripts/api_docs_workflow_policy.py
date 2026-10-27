#!/usr/bin/env python3
"""Direct generated-API publication checks on structured workflow records."""

from __future__ import annotations

import argparse
import posixpath
import re
from pathlib import Path

if __package__:
    from .api_docs_workflow_policy_inputs import PolicyInput, read_policy_inputs
    from .api_docs_workflow_reader import ScalarValue, SourceLocation, WorkflowReadError
    from .api_docs_workflow_staging import (
        StagedCommand, glob_matches_path, path_reaches_upload, published_paths, stage_has_docs_source,
        stage_is_dynamic, staged_commands,
    )
    from .api_docs_workflow_yaml_common import strip_shell_comment_stateful
else:
    from api_docs_workflow_policy_inputs import PolicyInput, read_policy_inputs
    from api_docs_workflow_reader import ScalarValue, SourceLocation, WorkflowReadError
    from api_docs_workflow_staging import (
        StagedCommand, glob_matches_path, path_reaches_upload, published_paths, stage_has_docs_source,
        stage_is_dynamic, staged_commands,
    )
    from api_docs_workflow_yaml_common import strip_shell_comment_stateful


_ACTION = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?@[^\s@]+\Z")
_PUBLISHER = re.compile(r"publish|deploy|release|pages|artifact|upload|gh-pages", re.I)
_SAFE_ACTIONS = frozenset(("actions/checkout",))
_EXPLICIT_SOURCE_ACTIONS = frozenset(("actions/upload-artifact", "actions/upload-pages-artifact"))
_REFERENCE_WORD = re.compile(r"[^\s,;|&()<>]+")
_DYNAMIC_SOURCE = re.compile(r"\$\{|\$\(|\$[A-Za-z_0-9@*?#$!-]|%[A-Za-z_][A-Za-z_0-9]*%")


class PolicyViolation(ValueError):
    def __init__(self, item: PolicyInput, location: SourceLocation, detail: str):
        scope = f"{item.workflow_path}:{location.line}:{location.column}: jobs.{item.job_key}"
        if item.step_index is not None:
            scope += f".steps[{item.step_index}]"
        super().__init__(f"{scope}: {detail}")


def _normalized_words(value: str) -> tuple[str, ...]:
    words: list[str] = []
    for match in _REFERENCE_WORD.finditer(value):
        token = match.group().replace("'", "").replace('"', "").replace("\\", "/")
        token = token.strip("`{}[]")
        if token:
            words.append(posixpath.normpath(token.casefold()))
    return tuple(words)


def _contains_generated_path(value: str) -> bool:
    for word in _normalized_words(value):
        parts = re.split(r"[/=:]", word)
        if any(parts[index : index + 2] == ["docs", "api"] for index in range(len(parts) - 1)):
            return True
    return False


def _active_command(value: str, shell_family: str) -> str:
    if shell_family == "cmd":
        return "\n".join(
            line for line in value.splitlines()
            if not line.lstrip().casefold().startswith(("rem ", "::"))
        )
    state = (False, False, False)
    active: list[str] = []
    for line in value.splitlines():
        rendered, state = strip_shell_comment_stateful(
            line, state, powershell=shell_family == "powershell"
        )
        active.append(rendered)
    return "\n".join(active)


def _action_kind(item: PolicyInput) -> str:
    action = item.action_identity
    if action is None:
        return "run"
    identity = action.value.strip()
    if item.action_dynamic:
        raise PolicyViolation(item, action.location, "dynamic or unresolved action identity")
    if identity.startswith("./"):
        if len(identity) <= 2 or any(part == ".." for part in identity.split("/")):
            raise PolicyViolation(item, action.location, "unsupported local action form")
        return "unknown"
    if identity.startswith("docker://"):
        if not identity.removeprefix("docker://"):
            raise PolicyViolation(item, action.location, "unsupported Docker action form")
        return "publisher" if _PUBLISHER.search(identity) else "unknown"
    if not _ACTION.fullmatch(identity):
        raise PolicyViolation(item, action.location, "unsupported action form")
    name = identity.split("@", 1)[0].casefold()
    if name in _SAFE_ACTIONS:
        return "unrelated"
    return "publisher" if _PUBLISHER.search(name) else "unknown"


def _path_is_broad(canonical: str) -> bool:
    if (canonical in (".", "..", "/", "*", "**") or canonical.startswith(("../", "~", "/"))
            or re.match(r"^[A-Za-z]:/", canonical)):
        return True
    parts = canonical.split("/")
    if "docs" in parts:
        return True
    return any(
        glob_matches_path(canonical, candidate)
        for candidate in ("docs/api", "docs/api/html", "docs/api/html/index.html", "docs/readme.md")
    )


def check_direct_policy(inputs: tuple[PolicyInput, ...]) -> None:
    for item in inputs:
        kind = _action_kind(item)
        action = item.action_identity
        if action is not None and _contains_generated_path(action.value):
            raise PolicyViolation(item, action.location, "generated API output path in action identity")
        values: tuple[tuple[str, ScalarValue], ...] = item.with_inputs + item.env_values
        for name, value in values:
            if _contains_generated_path(value.value):
                raise PolicyViolation(item, value.location, f"generated API output path in {name}")
        run = item.rendered_command
        if run is not None and _contains_generated_path(_active_command(run.value, item.shell_family)):
            raise PolicyViolation(item, run.location, "generated API output path in run command")
        if run is not None:
            for _, source in published_paths(item):
                canonical = posixpath.normpath(source.replace("\\", "/")).casefold()
                if _DYNAMIC_SOURCE.search(source):
                    raise PolicyViolation(item, run.location, "dynamic publication path in run command")
                if _path_is_broad(canonical):
                    raise PolicyViolation(item, run.location, "publishes docs or repository roots via run command")
        if action is not None and action.value.split("@", 1)[0].casefold() in _EXPLICIT_SOURCE_ACTIONS:
            if not any(path.name.casefold() == "path" for path in item.path_inputs):
                raise PolicyViolation(item, action.location, "publisher action requires explicit source path")
        if kind == "unrelated":
            continue
        for path in item.path_inputs:
            if path.raw.tag != "tag:yaml.org,2002:str" or not path.entries:
                raise PolicyViolation(item, path.raw.location, f"unsupported publication path in {path.name}")
            for entry in path.entries:
                if entry.dynamic or entry.canonical is None:
                    raise PolicyViolation(item, path.raw.location, f"dynamic or unresolved publication path in {path.name}")
                if _path_is_broad(entry.canonical):
                    raise PolicyViolation(item, path.raw.location, f"publishes docs or repository roots via {path.name}")
    check_staging_policy(inputs)


def check_staging_policy(inputs: tuple[PolicyInput, ...]) -> None:
    tainted: list[tuple[PolicyInput, StagedCommand, str]] = []
    scope: tuple[Path, str] | None = None

    def reject(origin: PolicyInput, stage: StagedCommand) -> None:
        detail = ("uses dynamic staging paths" if stage_is_dynamic(stage) else
                  f"{stage.kind} docs for publication or artifact upload")
        assert origin.rendered_command is not None
        raise PolicyViolation(origin, origin.rendered_command.location, detail)

    for item in inputs:
        item_scope = (item.workflow_path, item.job_key)
        if item_scope != scope:
            scope = item_scope
            tainted.clear()
        events: list[tuple[int, str, StagedCommand | str]] = []
        events.extend((stage.command_index, "stage", stage) for stage in staged_commands(item))
        events.extend((index, "publish", path) for index, path in published_paths(item))
        events.sort(key=lambda event: event[0])
        for _, event_type, value in events:
            if event_type == "stage":
                stage = value
                assert isinstance(stage, StagedCommand)
                if stage_has_docs_source(stage):
                    tainted.append((item, stage, stage.destination))
                else:
                    for origin, first_stage, destination in tuple(tainted):
                        if path_reaches_upload(destination, stage.source):
                            tainted.append((origin, first_stage, stage.destination))
            else:
                assert isinstance(value, str)
                for origin, stage, destination in tainted:
                    if path_reaches_upload(destination, value):
                        reject(origin, stage)
        if _action_kind(item) != "unrelated":
            for path in item.path_inputs:
                for entry in path.entries:
                    for origin, stage, destination in tainted:
                        if path_reaches_upload(destination, entry.text):
                            reject(origin, stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        inputs = read_policy_inputs(args.root)
        check_direct_policy(inputs)
    except (WorkflowReadError, PolicyViolation) as exc:
        parser.exit(1, f"api-docs-workflow-policy: {exc}\n")
    print(f"api-docs-workflow-policy: checked {len(inputs)} job/step records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
