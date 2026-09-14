#!/usr/bin/env python3
"""Validate local-only generated API routing docs."""

from __future__ import annotations

import argparse
import re
import sys
from urllib.parse import unquote
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
REFERENCE_LINK_PATTERN = re.compile(r"(?m)^ {0,3}\[[^\]]+\]:\s+(\S+)")
AUTOLINK_PATTERN = re.compile(r"<((?:https?:)?//[^>\s]+)>", re.IGNORECASE)
HTML_HREF_PATTERN = re.compile(
    r"""<a\s+[^>]*href\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))""", re.IGNORECASE
)
BARE_URL_PATTERN = re.compile(r"""https?://[^\s<>)"']+""", re.IGNORECASE)
PROTOCOL_RELATIVE_URL_PATTERN = re.compile(r"""(?<!:)//[^\s<>)"']+""")

API_ROUTING_FILES = (
    "README.md",
    "INSTALL.md",
    "docs/api_reference.md",
    "docs/maintainer_guide.md",
)

MAKEFILE_ROUTING_TARGET = re.compile(
    r"(?m)^api-docs-routing:\n"
    r"\t@python3 scripts/check_api_docs_routing\.py\n"
    r"\t@python3 tests/test_api_docs_routing\.py$"
)
MAKEFILE_VALIDATE_DEP = re.compile(
    r"(?m)^api-docs-validate:[^\n]*[ \t]api-docs-routing([ \t]|$)"
)
MAKEFILE_FRESHNESS_DEP = re.compile(
    r"(?m)^api-docs-freshness:[^\n]*[ \t]api-docs-validate([ \t]|$)"
)

REQUIRED_ROUTES = {
    "README.md": (
        "docs/api_reference.md",
        "include/",
        "INSTALL.md#support-readiness-matrix",
    ),
    "INSTALL.md": (),
    "docs/api_reference.md": (
        "../include/",
        "../Doxyfile",
        "../INSTALL.md#support-readiness-matrix",
        "tutorial.md",
        "cookbook.md",
        "solver_selection.md",
        "maintainer_guide.md",
    ),
    "docs/maintainer_guide.md": (),
}

REQUIRED_TEXT = {
    "docs/api_reference.md": (
        "The generated HTML tree is local-only generated output.",
        "is not a hosted or source-controlled publication surface.",
        "source-controlled API reference path.",
    ),
    "README.md": (
        "API reference entry point: docs/api_reference.md",
        "Generated API HTML is not hosted documentation, a retained CI artifact,",
    ),
    "INSTALL.md": (
        "| Local generated API HTML | local-only |",
        "No hosted API publication, retained generated-doc artifact, committed generated HTML,",
    ),
    "docs/maintainer_guide.md": (
        "`docs/api_reference.md` is the user-facing API reference entry point.",
        "`docs/api/html/` is generated Doxygen output",
        "`make api-docs-freshness` runs `docs-check` plus the local-only generated",
        "`api-docs-routing` proves user-facing docs route API readers",
        "source-controlled reference path is `docs/api_reference.md` plus checked-in",
        "retained generated-doc artifacts",
        "removes `api-docs-routing` from `make api-docs-freshness` without replacing",
    ),
}

GENERATED_API_PATH = "docs/api"
HOSTED_API_PUBLICATION_PATTERN = re.compile(
    r"(github\.io|readthedocs\.io|(^|[/:.-])(pages|api|doxygen|docs)([/:.-]|$))",
    re.IGNORECASE,
)
MAKEFILE_REQUIRED_VALIDATE_PREREQS = ("docs-check", "api-docs-local-only", "api-docs-routing")


class RoutingError(RuntimeError):
    pass


def markdown_links(text: str) -> list[str]:
    return [match.group(1).strip() for match in LINK_PATTERN.finditer(text)]


def publication_link_targets(text: str) -> list[str]:
    targets = markdown_links(text)
    targets.extend(match.group(1).strip() for match in REFERENCE_LINK_PATTERN.finditer(text))
    targets.extend(match.group(1).strip() for match in AUTOLINK_PATTERN.finditer(text))
    targets.extend(next(group for group in match.groups() if group).strip() for match in HTML_HREF_PATTERN.finditer(text))
    targets.extend(match.group(0).strip() for match in BARE_URL_PATTERN.finditer(text))
    targets.extend(match.group(0).strip() for match in PROTOCOL_RELATIVE_URL_PATTERN.finditer(text))
    return targets


def markdown_destination(target: str) -> str:
    target = target.strip()
    if target.startswith("<"):
        end = target.find(">")
        if end != -1:
            return target[: end + 1]
    return target.split(None, 1)[0] if target else target


def unwrap_link_target(target: str) -> str:
    target = markdown_destination(target)
    if target.startswith("<") and target.endswith(">"):
        return target[1:-1].strip()
    return target


def strip_fragment_and_query(target: str) -> str:
    return unquote(re.split(r"[#?]", unwrap_link_target(target), maxsplit=1)[0])


def fragment(target: str) -> str:
    parts = unwrap_link_target(target).split("#", 1)
    return parts[1] if len(parts) == 2 else ""


def is_external(target: str) -> bool:
    normalized = unwrap_link_target(target).lower()
    return normalized.startswith(("http://", "https://", "//", "mailto:"))


def is_forbidden_external_target(target: str) -> bool:
    normalized = unwrap_link_target(target)
    if normalized.lower().startswith("mailto:"):
        return False
    return bool(HOSTED_API_PUBLICATION_PATTERN.search(normalized))


def markdown_heading_fragment(heading: str) -> str:
    heading = re.sub(r"`([^`]*)`", r"\1", heading.strip().lower())
    heading = re.sub(r"[^a-z0-9 _-]", "", heading)
    heading = re.sub(r"\s+", "-", heading)
    return heading


def markdown_heading_fragments(text: str) -> set[str]:
    fragments = set()
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            fragments.add(markdown_heading_fragment(match.group(1)))
    return fragments


def validate_target_exists(root: Path, source: Path, target: str) -> None:
    if is_external(target):
        return

    path_part = strip_fragment_and_query(target)
    if not path_part:
        return

    resolved = (source.parent / path_part).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RoutingError(f"{source.relative_to(root)} link escapes repository: {target}") from exc

    if not resolved.exists():
        raise RoutingError(f"{source.relative_to(root)} links to missing API route target: {target}")

    target_fragment = fragment(target)
    if target_fragment and resolved.is_file():
        fragments = markdown_heading_fragments(resolved.read_text(encoding="utf-8"))
        if target_fragment not in fragments:
            raise RoutingError(f"{source.relative_to(root)} links to missing API route fragment: {target}")


def validate_required_routes(root: Path, rel_path: str, path: Path, text: str) -> None:
    links = set(markdown_links(text))
    missing = [target for target in REQUIRED_ROUTES[rel_path] if target not in links]
    if missing:
        joined = ", ".join(missing)
        raise RoutingError(f"{rel_path} missing required API route link(s): {joined}")

    for target in REQUIRED_ROUTES[rel_path]:
        validate_target_exists(root, path, target)

    for needle in REQUIRED_TEXT[rel_path]:
        if needle not in text:
            raise RoutingError(f"{rel_path} missing required local-only API routing text: {needle}")


def normalized_local_target(root: Path, source: Path, target: str) -> str:
    path_part = strip_fragment_and_query(target)
    if not path_part:
        return ""

    if path_part.startswith("/"):
        resolved = (root / path_part.lstrip("/")).resolve()
    else:
        resolved = (source.parent / path_part).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise RoutingError(f"{source.relative_to(root)} link escapes repository: {target}") from exc


def is_generated_api_path(rel_target: str) -> bool:
    return rel_target == GENERATED_API_PATH or rel_target.startswith(f"{GENERATED_API_PATH}/")


def validate_no_forbidden_links(root: Path, rel_path: str, source: Path, text: str) -> None:
    for target in publication_link_targets(text):
        if is_external(target) and is_forbidden_external_target(target):
            raise RoutingError(
                f"{rel_path} links to unsupported generated or hosted API publication target: {target}"
            )

        rel_target = normalized_local_target(root, source, target)
        if is_generated_api_path(rel_target):
            raise RoutingError(
                f"{rel_path} links to unsupported generated or hosted API publication target: {target}"
            )


def validate_makefile_wiring(root: Path) -> None:
    makefile = root / "Makefile"
    if not makefile.is_file():
        raise RoutingError("Makefile is missing; cannot verify api-docs-routing wiring")

    text = makefile.read_text(encoding="utf-8")
    if not MAKEFILE_ROUTING_TARGET.search(text):
        raise RoutingError("Makefile must define api-docs-routing with the routing guard and regression suite")
    if not MAKEFILE_VALIDATE_DEP.search(text):
        raise RoutingError("Makefile api-docs-validate must depend on api-docs-routing")
    if not MAKEFILE_FRESHNESS_DEP.search(text):
        raise RoutingError("Makefile api-docs-freshness must depend on api-docs-validate")

    validate_line = next(
        (line for line in text.splitlines() if line.startswith("api-docs-validate:")),
        "",
    )
    validate_prereqs = set(validate_line.split(":", 1)[1].split())
    missing = [prereq for prereq in MAKEFILE_REQUIRED_VALIDATE_PREREQS if prereq not in validate_prereqs]
    if missing:
        joined = ", ".join(missing)
        raise RoutingError(f"Makefile api-docs-validate missing required prerequisite(s): {joined}")


def validate_api_routes(root: Path) -> None:
    for rel_path in API_ROUTING_FILES:
        path = root / rel_path
        if not path.is_file():
            raise RoutingError(f"required API routing document missing: {rel_path}")

        text = path.read_text(encoding="utf-8")
        validate_required_routes(root, rel_path, path, text)
        validate_no_forbidden_links(root, rel_path, path, text)

    validate_makefile_wiring(root)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local-only generated API routing docs.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    try:
        validate_api_routes(root)
    except RoutingError as exc:
        print(f"api-docs-routing: FAIL: {exc}", file=sys.stderr)
        return 1

    print("api-docs-routing: PASS")
    print(f"  checked routing documents: {len(API_ROUTING_FILES)}")
    print("  Makefile api-docs-routing wiring: present")
    print("  generated API publication links: absent")
    print("  source-controlled API entry point: docs/api_reference.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
