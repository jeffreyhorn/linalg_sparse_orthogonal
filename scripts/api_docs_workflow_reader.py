#!/usr/bin/env python3
"""Read GitHub Actions workflows without flattening job or step boundaries."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "api-docs-workflow-reader: PyYAML 6.0.3 is required; "
        "install requirements-api-docs.txt"
    ) from exc


class _WorkflowLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()
            if event.anchor in self.anchors:
                self.get_event()
                original = self.anchors[event.anchor]
                aliased = copy.copy(original)
                aliased.start_mark = event.start_mark
                aliased.end_mark = event.end_mark
                aliased.alias_origin = getattr(original, "alias_origin", original.start_mark)
                return aliased
        return super().compose_node(parent, index)


class WorkflowReadError(ValueError):
    def __init__(self, kind: str, path: Path, line: int, column: int, detail: str):
        self.kind = kind
        self.path = path
        self.line = line
        self.column = column
        super().__init__(f"{path}:{line}:{column}: {kind}: {detail}")


@dataclass(frozen=True)
class SourceLocation:
    path: Path
    line: int
    column: int
    definition_line: int | None = None
    definition_column: int | None = None


@dataclass(frozen=True)
class ScalarValue:
    value: str
    tag: str
    style: str | None
    location: SourceLocation
    source: str


@dataclass(frozen=True)
class StepRecord:
    workflow_path: Path
    job_key: str
    step_index: int
    location: SourceLocation
    runs_on: ScalarValue | None
    shell: ScalarValue | None
    effective_shell: str
    shell_source: str
    shell_family: str
    action_identity: ScalarValue | None
    with_inputs: tuple[tuple[str, ScalarValue], ...]
    with_location: SourceLocation | None
    run_value: ScalarValue | None
    env: tuple[tuple[str, ScalarValue], ...]


@dataclass(frozen=True)
class JobRecord:
    key: str
    location: SourceLocation
    runs_on: ScalarValue | None
    shell_default: ScalarValue | None
    uses: ScalarValue | None
    with_inputs: tuple[tuple[str, ScalarValue], ...]
    with_location: SourceLocation | None
    env: tuple[tuple[str, ScalarValue], ...]
    steps: tuple[StepRecord, ...]


@dataclass(frozen=True)
class WorkflowRecord:
    path: Path
    location: SourceLocation
    shell_default: ScalarValue | None
    env: tuple[tuple[str, ScalarValue], ...]
    jobs: tuple[JobRecord, ...]


def _location(path: Path, node: yaml.Node) -> SourceLocation:
    origin = getattr(node, "alias_origin", None)
    return SourceLocation(
        path,
        node.start_mark.line + 1,
        node.start_mark.column + 1,
        None if origin is None else origin.line + 1,
        None if origin is None else origin.column + 1,
    )


def _error(kind: str, path: Path, node: yaml.Node, detail: str) -> WorkflowReadError:
    location = _location(path, node)
    return WorkflowReadError(kind, path, location.line, location.column, detail)


def _validate_tree(path: Path, root: yaml.Node) -> None:
    active: set[int] = set()
    visited: set[int] = set()
    scalar_tags = {
        "tag:yaml.org,2002:str", "tag:yaml.org,2002:bool", "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:float", "tag:yaml.org,2002:null",
    }

    def visit(node: yaml.Node) -> None:
        identity = id(node)
        if identity in active:
            raise _error("parse", path, node, "cyclic alias")
        if identity in visited:
            return
        if isinstance(node, yaml.MappingNode):
            allowed_tags = {"tag:yaml.org,2002:map"}
        elif isinstance(node, yaml.SequenceNode):
            allowed_tags = {"tag:yaml.org,2002:seq"}
        elif isinstance(node, yaml.ScalarNode):
            allowed_tags = scalar_tags
        else:
            raise _error("unsupported", path, node, "unknown YAML node type")
        if node.tag not in allowed_tags:
            raise _error("unsupported", path, node, f"YAML tag {node.tag}")
        active.add(identity)
        if isinstance(node, yaml.MappingNode):
            keys: set[str] = set()
            for key, value in node.value:
                visit(key)
                if not isinstance(key, yaml.ScalarNode):
                    raise _error("unsupported", path, key, "non-scalar mapping key")
                if key.value in keys:
                    raise _error("parse", path, key, f"duplicate key {key.value!r}")
                keys.add(key.value)
                visit(value)
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                visit(child)
        active.remove(identity)
        visited.add(identity)

    visit(root)


def _mapping(path: Path, node: yaml.Node, label: str) -> dict[str, yaml.Node]:
    if not isinstance(node, yaml.MappingNode):
        raise _error("parse", path, node, f"{label} must be a mapping")
    return {key.value: value for key, value in node.value}


def _scalar(
    path: Path, node: yaml.Node, text: str, label: str, *, require_string: bool = True
) -> ScalarValue:
    if not isinstance(node, yaml.ScalarNode):
        raise _error("parse", path, node, f"{label} must be a scalar")
    if require_string and node.tag != "tag:yaml.org,2002:str":
        raise _error("parse", path, node, f"{label} must be a string")
    return ScalarValue(
        node.value,
        node.tag,
        node.style,
        _location(path, node),
        text[node.start_mark.index : node.end_mark.index],
    )


def _optional_scalar(
    path: Path, fields: dict[str, yaml.Node], text: str, key: str, label: str
) -> ScalarValue | None:
    node = fields.get(key)
    return None if node is None else _scalar(path, node, text, label)


def _scalar_items(
    path: Path, node: yaml.Node | None, text: str, label: str
) -> tuple[tuple[str, ScalarValue], ...]:
    if node is None:
        return ()
    fields = _mapping(path, node, label)
    return tuple(
        (key, _scalar(path, value, text, f"{label}.{key}", require_string=False))
        for key, value in fields.items()
    )


def _default_shell(
    path: Path, fields: dict[str, yaml.Node], text: str, label: str
) -> ScalarValue | None:
    defaults = fields.get("defaults")
    if defaults is None:
        return None
    run = _mapping(path, defaults, f"{label}.defaults").get("run")
    if run is None:
        return None
    return _optional_scalar(path, _mapping(path, run, f"{label}.defaults.run"), text, "shell", label)


def _effective_shell(
    shell: ScalarValue | None, job_shell: ScalarValue | None,
    workflow_shell: ScalarValue | None, runner: ScalarValue | None,
) -> tuple[str, str, str]:
    for source, candidate in (("step", shell), ("job", job_shell), ("workflow", workflow_shell)):
        if candidate is not None:
            value = candidate.value
            return value, source, _shell_family(value)
    if runner is None or "${{" in runner.value:
        return "unknown", "unknown", "unknown"
    runner_name = runner.value.lower()
    if runner_name.startswith("windows-"):
        return "pwsh", "runner", "powershell"
    if runner_name.startswith(("ubuntu-", "macos-")):
        return "bash", "runner", "posix"
    return "unknown", "unknown", "unknown"


def _shell_family(value: str) -> str:
    if "${{" in value:
        return "unknown"
    command = value.strip().split(maxsplit=1)[0].lower() if value.strip() else ""
    if command in ("pwsh", "powershell"):
        return "powershell"
    if command in ("bash", "sh"):
        return "posix"
    if command == "cmd":
        return "cmd"
    return "unknown"


def read_workflow(path: Path) -> WorkflowRecord:
    """Parse one workflow; reject ambiguity before extracting any records."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise WorkflowReadError("parse", path, 1, 1, f"cannot read UTF-8 workflow: {exc}") from exc
    try:
        documents = list(yaml.compose_all(text, Loader=_WorkflowLoader))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
        line = 1 if mark is None else mark.line + 1
        column = 1 if mark is None else mark.column + 1
        raise WorkflowReadError("parse", path, line, column, str(exc).splitlines()[0]) from exc
    if len(documents) != 1 or documents[0] is None:
        raise WorkflowReadError("parse", path, 1, 1, "expected exactly one YAML document")
    root = documents[0]
    _validate_tree(path, root)
    fields = _mapping(path, root, "workflow")
    jobs_node = fields.get("jobs")
    if jobs_node is None:
        raise _error("parse", path, root, "workflow requires jobs")
    jobs_map = _mapping(path, jobs_node, "jobs")
    if not jobs_map:
        raise _error("parse", path, jobs_node, "jobs must not be empty")
    workflow_shell = _default_shell(path, fields, text, "workflow")
    workflow_env = _scalar_items(path, fields.get("env"), text, "workflow.env")
    jobs: list[JobRecord] = []
    for job_key, job_node in jobs_map.items():
        job_fields = _mapping(path, job_node, f"jobs.{job_key}")
        job_shell = _default_shell(path, job_fields, text, f"jobs.{job_key}")
        job_env = _scalar_items(path, job_fields.get("env"), text, f"jobs.{job_key}.env")
        runner = _optional_scalar(path, job_fields, text, "runs-on", f"jobs.{job_key}.runs-on")
        job_uses = _optional_scalar(path, job_fields, text, "uses", f"jobs.{job_key}.uses")
        job_with = _scalar_items(path, job_fields.get("with"), text, f"jobs.{job_key}.with")
        steps_node = job_fields.get("steps")
        if job_uses is None and "with" in job_fields:
            raise _error("parse", path, job_fields["with"], f"jobs.{job_key}.with requires uses")
        if job_uses is not None and steps_node is not None:
            raise _error("parse", path, job_node, f"jobs.{job_key} has both uses and steps")
        if job_uses is None and steps_node is None:
            raise _error("parse", path, job_node, f"jobs.{job_key} requires steps or uses")
        if job_uses is None and runner is None:
            raise _error("parse", path, job_node, f"jobs.{job_key} requires runs-on")
        if steps_node is not None and not isinstance(steps_node, yaml.SequenceNode):
            raise _error("parse", path, steps_node, f"jobs.{job_key}.steps must be a sequence")
        if isinstance(steps_node, yaml.SequenceNode) and not steps_node.value:
            raise _error("parse", path, steps_node, f"jobs.{job_key}.steps must not be empty")
        steps: list[StepRecord] = []
        for index, step_node in enumerate(() if steps_node is None else steps_node.value):
            label = f"jobs.{job_key}.steps[{index}]"
            step_fields = _mapping(path, step_node, label)
            action = _optional_scalar(path, step_fields, text, "uses", f"{label}.uses")
            run = _optional_scalar(path, step_fields, text, "run", f"{label}.run")
            if (action is None) == (run is None):
                raise _error("parse", path, step_node, f"{label} requires exactly one of uses or run")
            if run is not None and "with" in step_fields:
                raise _error("parse", path, step_fields["with"], f"{label}.with requires uses")
            step_shell = _optional_scalar(path, step_fields, text, "shell", f"{label}.shell")
            effective_shell, shell_source, shell_family = _effective_shell(
                step_shell, job_shell, workflow_shell, runner
            )
            with_node = step_fields.get("with")
            steps.append(
                StepRecord(
                    workflow_path=path,
                    job_key=job_key,
                    step_index=index,
                    location=_location(path, step_node),
                    runs_on=runner,
                    shell=step_shell,
                    effective_shell=effective_shell,
                    shell_source=shell_source,
                    shell_family=shell_family,
                    action_identity=action,
                    with_inputs=_scalar_items(path, with_node, text, f"{label}.with"),
                    with_location=None if with_node is None else _location(path, with_node),
                    run_value=run,
                    env=_scalar_items(path, step_fields.get("env"), text, f"{label}.env"),
                )
            )
        jobs.append(
            JobRecord(
                key=job_key,
                location=_location(path, job_node),
                runs_on=runner,
                shell_default=job_shell,
                uses=job_uses,
                with_inputs=job_with,
                with_location=None if "with" not in job_fields else _location(path, job_fields["with"]),
                env=job_env,
                steps=tuple(steps),
            )
        )
    return WorkflowRecord(path, _location(path, root), workflow_shell, workflow_env, tuple(jobs))


def read_workflows(root: Path) -> tuple[WorkflowRecord, ...]:
    directory = Path(root) / ".github" / "workflows"
    paths = sorted((*directory.glob("*.yml"), *directory.glob("*.yaml")))
    if not paths:
        raise WorkflowReadError("parse", directory, 1, 1, "no workflow files found")
    return tuple(read_workflow(path) for path in paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        records = read_workflows(args.root)
    except WorkflowReadError as exc:
        parser.exit(1, f"api-docs-workflow-reader: {exc}\n")
    print(f"api-docs-workflow-reader: parsed {len(records)} workflows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
