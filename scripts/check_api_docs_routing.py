#!/usr/bin/env python3
"""Validate local-only generated API routing docs."""

from __future__ import annotations

import argparse
import html
import re
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOLINK_PATTERN = re.compile(r"<((?:[a-z][a-z0-9+.-]*:|//)[^>\s]+)>", re.IGNORECASE)
BARE_URL_PATTERN = re.compile(r"""https?://[^\s<>)"']+""", re.IGNORECASE)
PROTOCOL_RELATIVE_URL_PATTERN = re.compile(r"""(?<![:/])//[^\s<>)"']+""")

API_ROUTING_FILES = (
    "README.md",
    "INSTALL.md",
    "docs/api_reference.md",
    "docs/tutorial.md",
    "docs/cookbook.md",
    "docs/solver_selection.md",
    "docs/maintainer_guide.md",
)

MAKEFILE_ROUTING_TARGET = re.compile(
    r"(?m)^api-docs-routing:\n"
    r"\t@python3 scripts/check_api_docs_routing\.py\n"
    r"\t@python3 tests/test_api_docs_routing\.py$"
)
MAKEFILE_VALIDATE_DEP = re.compile(
    r"(?m)^api-docs-validate:[ \t]*\n"
    r"\t@[$][(]MAKE[)] docs-check\n"
    r"\t@[$][(]MAKE[)] api-docs-local-only\n"
    r"\t@[$][(]MAKE[)] api-docs-routing$"
)
MAKEFILE_FRESHNESS_DEP = re.compile(
    r"(?m)^api-docs-freshness:[ \t]*\n"
    r"\t@[$][(]MAKE[)] api-docs-validate$"
)
MAKEFILE_DOCS_CHECK_SERIAL = re.compile(
    r"(?m)^docs-check:[ \t]*docs[ \t]*\n"
    r"\t@[$][(]MAKE[)] api-docs-coverage$"
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
    "docs/tutorial.md": (),
    "docs/cookbook.md": (),
    "docs/solver_selection.md": (),
    "docs/maintainer_guide.md": (),
}

REQUIRED_TEXT = {
    "docs/api_reference.md": (
        "The generated HTML tree is local-only generated output.",
        "is not a hosted or source-controlled publication surface.",
        "source-controlled API reference path.",
    ),
    "docs/tutorial.md": (),
    "docs/cookbook.md": (),
    "docs/solver_selection.md": (),
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
        "hosted documentation publication, retained generated-doc artifacts,",
        "removes `api-docs-routing` from `make api-docs-freshness` without replacing",
    ),
}

GENERATED_API_PATH = "docs/api"
GENERATED_API_PUBLICATION_PATH_PATTERN = re.compile(r"(^|/)docs/api(?:[/?#!]|$)", re.IGNORECASE)
PROJECT_PUBLICATION_PATH_PATTERN = re.compile(
    r"(^|/)(api[-_]?reference|api|doxygen|pages)(?:[/?#!.]|$)",
    re.IGNORECASE,
)
PROJECT_PUBLICATION_HOST_PATTERN = re.compile(r"linalg[-_]sparse[-_]orthogonal", re.IGNORECASE)
SOURCE_CONTROLLED_API_REFERENCE_PATTERN = re.compile(
    r"^https?://github[.]com/jeffreyhorn/linalg_sparse_orthogonal/"
    r"(?:blob|tree)/.+/docs/api_reference[.]md(?:[?#].*)?$",
    re.IGNORECASE,
)
REPOSITORY_RELEASE_OR_ARTIFACT_PATTERN = re.compile(
    r"^https?://github[.]com/jeffreyhorn/linalg_sparse_orthogonal/"
    r"(?:releases/download/|actions/runs/[^/?#]+/artifacts(?:[/?#]|$)|suites/[^/?#]+/artifacts(?:[/?#]|$))",
    re.IGNORECASE,
)
ALLOWED_EXTERNAL_DOCS_PATTERN = re.compile(
    r"^https?://(?:docs[.]python[.]org(?:/|$)|github[.]com/jeffreyhorn/linalg_sparse_orthogonal(?:[/?#]|$))",
    re.IGNORECASE,
)
RAW_HTML_BLOCK_PATTERN = re.compile(
    r"(?is)<(article|aside|blockquote|details|div|figure|footer|header|li|ol|p|pre|section|table|ul)\b[^>]*>.*?</\1>"
)


class RoutingError(RuntimeError):
    pass


class AnchorHrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value is not None:
                self.hrefs.append(value.strip())


def closing_bracket(text: str, open_bracket: int) -> int:
    depth = 0
    escaped = False
    pos = open_bracket
    while pos < len(text):
        char = text[pos]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    return -1


def is_unescaped_image_label(text: str, open_label: int) -> bool:
    if open_label == 0 or text[open_label - 1] != "!":
        return False
    preceding_backslashes = 0
    pos = open_label - 2
    while pos >= 0 and text[pos] == "\\":
        preceding_backslashes += 1
        pos -= 1
    return preceding_backslashes % 2 == 0


def balanced_markdown_links(text: str, *, include_images: bool = True) -> list[str]:
    targets: list[str] = []
    index = 0
    while index < len(text):
        open_label = text.find("[", index)
        if open_label == -1:
            break
        if not include_images and is_unescaped_image_label(text, open_label):
            index = open_label + 1
            continue
        preceding_backslashes = 0
        pos = open_label - 1
        while pos >= 0 and text[pos] == "\\":
            preceding_backslashes += 1
            pos -= 1
        if preceding_backslashes % 2 == 1:
            index = open_label + 1
            continue
        close_label = closing_bracket(text, open_label)
        if close_label == -1 or close_label + 1 >= len(text) or text[close_label + 1] != "(":
            index = open_label + 1
            continue

        open_dest = close_label + 2
        pos = open_dest
        depth = 0
        escaped = False
        while pos < len(text):
            char = text[pos]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    targets.append(text[open_dest:pos].strip())
                    break
                depth -= 1
            pos += 1
        index = pos + 1 if pos < len(text) else open_label + 1
    return targets


def reference_label_key(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip()).casefold()


def balanced_reference_definitions(text: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = re.sub(r"^ {0,3}(?:>[ \t]?)+", "", lines[index])
        index += 1
        if not re.match(r"^ {0,3}\[", line):
            continue
        indent = len(line) - len(line.lstrip(" "))
        open_label = indent
        close_label = closing_bracket(line, open_label)
        if close_label == -1 or close_label + 1 >= len(line) or line[close_label + 1] != ":":
            continue
        target = line[close_label + 2 :].strip()
        if not target and index < len(lines):
            continuation = re.sub(r"^ {0,3}(?:>[ \t]?)+", "", lines[index])
            if re.match(r"^(?: {1,}|\t)\S", continuation):
                target = continuation.strip()
                index += 1
        if target and target.startswith(">"):
            target = re.sub(r"^(?:>[ \t]?)+", "", target).strip()
        if not target:
            continue
        definitions[reference_label_key(line[open_label + 1 : close_label])] = target
    return definitions


def balanced_reference_link_targets(text: str, definitions: dict[str, str]) -> list[str]:
    targets: list[str] = []
    index = 0
    while index < len(text):
        open_label = text.find("[", index)
        if open_label == -1:
            break
        preceding_backslashes = 0
        pos = open_label - 1
        while pos >= 0 and text[pos] == "\\":
            preceding_backslashes += 1
            pos -= 1
        if preceding_backslashes % 2 == 1:
            index = open_label + 1
            continue
        close_label = closing_bracket(text, open_label)
        if close_label == -1 or close_label + 1 >= len(text) or text[close_label + 1] != "[":
            index = open_label + 1
            continue
        close_ref = closing_bracket(text, close_label + 1)
        if close_ref == -1:
            index = open_label + 1
            continue
        label = text[open_label + 1 : close_label]
        ref = text[close_label + 2 : close_ref] or label
        target = definitions.get(reference_label_key(ref))
        if target:
            targets.append(target)
        index = close_ref + 1
    return targets


def publication_link_targets(text: str) -> list[str]:
    reference_definitions = balanced_reference_definitions(text)
    targets = [html.unescape(target) for target in balanced_markdown_links(text)]
    targets.extend(html.unescape(target) for target in reference_definitions.values())
    targets.extend(html.unescape(target) for target in balanced_reference_link_targets(text, reference_definitions))
    targets.extend(html.unescape(match.group(1).strip()) for match in AUTOLINK_PATTERN.finditer(text))
    parser = AnchorHrefParser()
    parser.feed(text)
    targets.extend(parser.hrefs)
    targets.extend(html.unescape(match.group(0).strip()) for match in BARE_URL_PATTERN.finditer(text))
    targets.extend(html.unescape(match.group(0).strip()) for match in PROTOCOL_RELATIVE_URL_PATTERN.finditer(text))
    return targets


def render_code_spans(text: str, *, keep_inline_code: bool) -> str:
    rendered: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "`":
            rendered.append(text[index])
            index += 1
            continue

        end = index + 1
        while end < len(text) and text[end] == "`":
            end += 1
        delimiter = text[index:end]
        close = text.find(delimiter, end)
        if close == -1:
            rendered.append(delimiter)
            index = end
            continue

        if keep_inline_code:
            rendered.append(text[end:close])
        index = close + len(delimiter)
    return "".join(rendered)


def rendered_markdown_text(
    text: str, *, keep_inline_code: bool = False, strip_raw_html_blocks: bool = True
) -> str:
    text = re.sub(r"(?s)<!--.*?-->", "", text)
    if strip_raw_html_blocks:
        text = RAW_HTML_BLOCK_PATTERN.sub("", text)
    rendered_lines: list[str] = []
    fence_marker = ""
    fence_in_blockquote = False
    for line in text.splitlines():
        in_blockquote = re.match(r"^ {0,3}>", line) is not None
        if fence_marker and fence_in_blockquote and not in_blockquote:
            fence_marker = ""
            fence_in_blockquote = False
        fence_line = re.sub(r"^ {0,3}(?:>[ \t]?)+", "", line)
        fence_match = re.match(r"^ {0,3}(```+|~~~+)", fence_line)
        if fence_match:
            marker = fence_match.group(1)
            rest = fence_line[fence_match.end() :]
            if fence_marker:
                if (
                    marker[0] == fence_marker[0]
                    and len(marker) >= len(fence_marker)
                    and rest.strip() == ""
                ):
                    fence_marker = ""
                    fence_in_blockquote = False
            else:
                fence_marker = marker
                fence_in_blockquote = in_blockquote
            continue
        if fence_marker:
            continue
        if re.match(r"^(?: {4}|\t)", fence_line):
            continue
        rendered_lines.append(line)
    text = "\n".join(rendered_lines)
    return render_code_spans(text, keep_inline_code=keep_inline_code)


def normalized_required_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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


def unescape_markdown_destination(target: str) -> str:
    unescaped = re.sub(
        r"""\\([!"#$%&'()*+,./:;<=>?@\[\\\]^_`{|}~-])""",
        r"\1",
        unwrap_link_target(target),
    )
    return html.unescape(unescaped)


def strip_fragment_and_query(target: str) -> str:
    return unquote(re.split(r"[#?]", unescape_markdown_destination(target), maxsplit=1)[0])


def fragment(target: str) -> str:
    parts = unescape_markdown_destination(target).split("#", 1)
    return unquote(parts[1]) if len(parts) == 2 else ""


def is_external(target: str) -> bool:
    normalized = unescape_markdown_destination(target).lower()
    return normalized.startswith("//") or re.match(r"^[a-z][a-z0-9+.-]*:", normalized) is not None


def is_forbidden_external_target(target: str) -> bool:
    normalized = unquote(unescape_markdown_destination(target))
    parsed = urlsplit(normalized if not normalized.startswith("//") else f"https:{normalized}")
    if SOURCE_CONTROLLED_API_REFERENCE_PATTERN.match(normalized):
        return False
    if REPOSITORY_RELEASE_OR_ARTIFACT_PATTERN.match(normalized):
        return True
    route = f"{parsed.path}"
    if parsed.query:
        route = f"{route}?{parsed.query}"
    if parsed.fragment:
        route = f"{route}#{parsed.fragment}"
    if GENERATED_API_PUBLICATION_PATH_PATTERN.search(route):
        return True
    project_owned = PROJECT_PUBLICATION_HOST_PATTERN.search(parsed.netloc) or PROJECT_PUBLICATION_HOST_PATTERN.search(
        parsed.path
    )
    if project_owned and PROJECT_PUBLICATION_PATH_PATTERN.search(route):
        return True
    if ALLOWED_EXTERNAL_DOCS_PATTERN.match(normalized):
        return False
    if project_owned:
        return True
    return False


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
        fragments = markdown_heading_fragments(
            rendered_markdown_text(resolved.read_text(encoding="utf-8"))
        )
        if target_fragment not in fragments:
            raise RoutingError(f"{source.relative_to(root)} links to missing API route fragment: {target}")


def validate_required_routes(root: Path, rel_path: str, path: Path, text: str) -> None:
    rendered_markdown_for_links = rendered_markdown_text(text)
    reference_definitions = balanced_reference_definitions(rendered_markdown_for_links)
    markdown_targets = balanced_markdown_links(
        rendered_markdown_for_links, include_images=False
    ) + balanced_reference_link_targets(rendered_markdown_for_links, reference_definitions)
    html_target_text = rendered_markdown_text(text, strip_raw_html_blocks=False)
    html_parser = AnchorHrefParser()
    html_parser.feed(html_target_text)
    route_targets = markdown_targets + html_parser.hrefs
    links = {
        normalized_route_key(root, path, target)
        for target in route_targets
        if not is_external(target)
    }
    required = {normalized_route_key(root, path, target): target for target in REQUIRED_ROUTES[rel_path]}
    missing = [target for key, target in required.items() if key not in links]
    if missing:
        joined = ", ".join(missing)
        raise RoutingError(f"{rel_path} missing required API route link(s): {joined}")

    for target in REQUIRED_ROUTES[rel_path]:
        validate_target_exists(root, path, target)

    rendered_text = normalized_required_text(rendered_markdown_text(text, keep_inline_code=True))
    for needle in REQUIRED_TEXT[rel_path]:
        rendered_needle = normalized_required_text(rendered_markdown_text(needle, keep_inline_code=True))
        if rendered_needle not in rendered_text:
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


def normalized_route_key(root: Path, source: Path, target: str) -> str:
    rel_target = normalized_local_target(root, source, target)
    target_fragment = fragment(target)
    return f"{rel_target}#{target_fragment}" if target_fragment else rel_target


def is_generated_api_path(rel_target: str) -> bool:
    normalized = rel_target.casefold()
    generated = GENERATED_API_PATH.casefold()
    return normalized == generated or normalized.startswith(f"{generated}/")


def validate_no_forbidden_links(root: Path, rel_path: str, source: Path, text: str) -> None:
    rendered_text = rendered_markdown_text(text, strip_raw_html_blocks=False)
    for target in publication_link_targets(rendered_text):
        if is_external(target):
            if is_forbidden_external_target(target):
                raise RoutingError(
                    f"{rel_path} links to unsupported generated or hosted API publication target: {target}"
                )
            continue

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
        raise RoutingError("Makefile api-docs-validate must serialize API docs validation phases")
    if not MAKEFILE_FRESHNESS_DEP.search(text):
        raise RoutingError("Makefile api-docs-freshness must serialize api-docs-validate")
    if not MAKEFILE_DOCS_CHECK_SERIAL.search(text):
        raise RoutingError("Makefile docs-check must serialize docs before api-docs-coverage")


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
