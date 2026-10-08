#!/usr/bin/env bash
# check_api_docs_local_only.sh - generated API HTML local-only guard.
#
# Proves that Doxygen HTML under docs/api/ remains ignored local generated
# output unless a future publication decision explicitly selects committed
# generated API HTML.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

fail() {
    echo "api-docs-local-only: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "api-docs-local-only: $1 ok"
}

require_ignored() {
    local path="$1"

    if ! git -C "$ROOT_DIR" check-ignore -q "$path"; then
        fail "$path is not ignored; generated API HTML must remain local-only unless a future publication decision selects committed output"
    fi

    pass "$path ignore rule"
}

require_empty() {
    local label="$1"
    local message="$2"
    local output="$3"

    if [ -n "$output" ]; then
        printf '%s\n' "$output" >&2
        fail "$message"
    fi

    pass "$label"
}

require_file_contains() {
    local path="$1"
    local needle="$2"
    local label="$3"
    local full_path="$ROOT_DIR/$path"

    if [ ! -f "$full_path" ]; then
        fail "$path is missing; cannot verify strengthened local-only generated API HTML product decision wording"
    fi

    if ! grep -Fq "$needle" "$full_path"; then
        fail "$path must state $label for the strengthened local-only generated API HTML product decision"
    fi

    pass "$path $label wording"
}

require_doxyfile_setting() {
    local key="$1"
    local value="$2"
    local label="$3"
    local doxyfile="$ROOT_DIR/Doxyfile"

    if [ ! -f "$doxyfile" ]; then
        fail "Doxyfile is missing; cannot verify generated API HTML local-only output contract"
    fi

    if ! grep -Eq "^[[:space:]]*$key[[:space:]]*=[[:space:]]*$value[[:space:]]*$" "$doxyfile"; then
        fail "Doxyfile must keep $key = $value for $label"
    fi

    pass "Doxyfile $key local-only contract"
}

strip_yaml_comments() {
    python3 - "$ROOT_DIR/scripts" "$1" <<'PY'
import re
import sys

sys.path.insert(0, sys.argv[1])
from api_docs_workflow_yaml_common import (
    quoted_scalar_continues,
    strip_shell_comment_stateful,
    strip_yaml_comment,
    strip_yaml_comment_continuation,
)

# Shared parsing helpers are imported from api_docs_workflow_yaml_common.py

path = sys.argv[2]
with open(path, encoding="utf-8") as handle:
    block_indent = None
    block_key = None
    quote_indent = None
    quote_char = None
    shell_state = (False, False, False)
    for raw_line in handle:
        line = raw_line.rstrip("\n")
        if quote_char is not None:
            if line.strip():
                line_indent = len(line) - len(line.lstrip(" "))
                if line_indent <= quote_indent:
                    quote_char = None
                else:
                    stripped = strip_yaml_comment_continuation(line, quote_char)
                    print(stripped)
                    if quoted_scalar_continues(f"{quote_char}{stripped}") is None:
                        quote_char = None
                    continue
            else:
                print(line)
                continue
        if block_indent is not None:
            if line.strip():
                line_indent = len(line) - len(line.lstrip(" "))
                if line_indent > block_indent:
                    if block_key == "run":
                        stripped_shell, shell_state = strip_shell_comment_stateful(
                            line,
                            shell_state,
                        )
                        if stripped_shell.strip():
                            print(stripped_shell)
                    else:
                        print(line)
                    continue
                block_indent = None
                block_key = None
                shell_state = (False, False, False)
            else:
                continue
        stripped = strip_yaml_comment(line)
        block_match = re.match(r"^[ ]*(?:-[ ]*)?([^:]+?)[ ]*:[ ]*[>|]", stripped)
        if block_match:
            block_indent = block_match.start(1)
            block_key = block_match.group(1).strip("'\"").lower()
            shell_state = (False, False, False)
        else:
            run_scalar_match = re.match(
                r"^(\s*(?:-[ ]*)?['\"]?run['\"]?[ ]*:[ ]*)(.*)$",
                stripped,
                re.IGNORECASE,
            )
            if run_scalar_match is not None:
                run_value, _ = strip_shell_comment_stateful(
                    run_scalar_match.group(2),
                    (False, False, False),
                )
                stripped = run_scalar_match.group(1) + run_value
        scalar_match = re.match(r"^[ ]*(?:-[ ]*)?['\"]?[^:]+['\"]?[ ]*:[ ]*(.*)$", stripped)
        if scalar_match:
            continuing_quote = quoted_scalar_continues(scalar_match.group(1))
            if continuing_quote is not None:
                quote_char = continuing_quote
                quote_indent = len(stripped) - len(stripped.lstrip(" "))
        line = stripped
        if line.strip():
            print(line)
PY
}

decode_yaml_publication_paths() {
    local path_decode_input
    path_decode_input="$(mktemp)"
    cat > "$path_decode_input"
    python3 - "$ROOT_DIR/scripts" "$path_decode_input" <<'PY'
import re
import sys

sys.path.insert(0, sys.argv[1])
from api_docs_workflow_yaml_common import (
    decode_plain_value,
    decode_value,
    quoted_value_tail,
    unquoted_yaml_text,
)

PUBLICATION_PATH_KEYS = (
    "path",
    "publish_dir",
    "publish-dir",
    "directory",
    "folder",
    "files",
    "asset_path",
    "uses",
)

KEY_PATTERN = "|".join(re.escape(key) for key in PUBLICATION_PATH_KEYS)
SCALAR_PATTERN = re.compile(
    rf"^(\s*-?\s*['\"]?(?:{KEY_PATTERN})['\"]?\s*:\s*)(['\"])(.*)\2(\s*)$",
    re.IGNORECASE,
)
SCALAR_START_PATTERN = re.compile(
    rf"^(\s*-?\s*['\"]?(?:{KEY_PATTERN})['\"]?\s*:\s*)(['\"])(.*)$",
    re.IGNORECASE,
)
DOUBLE_FLOW_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[{,\s])['\"]?(?:%s)['\"]?\s*:\s*)\"(?P<value>(?:\\.|[^\"])*)\"" % KEY_PATTERN,
    re.IGNORECASE,
)
SINGLE_FLOW_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[{,\s])['\"]?(?:%s)['\"]?\s*:\s*)'(?P<value>(?:''|[^'])*)'" % KEY_PATTERN,
    re.IGNORECASE,
)
FLOW_START_PATTERN = re.compile(
    r"(?P<prefix>.*(?:^|[{,\s])['\"]?(?:%s)['\"]?\s*:\s*)(?P<quote>['\"])(?P<value>.*)$" % KEY_PATTERN,
    re.IGNORECASE,
)
ANCHOR_NAME_PATTERN = r"[^\s\[\]\{\},]+"
NODE_ANCHOR_PATTERN = re.compile(
    r"^\s*(?:-\s*)?(?:['\"]?[^:'\"]+['\"]?\s*:\s*)?&(?P<anchor>%s)\s+(?P<value>(?:\"(?:\\.|[^\"])*\"|'(?:''|[^'])*'|[^#,\]}]+))"
    % ANCHOR_NAME_PATTERN,
)
ANCHORED_SCALAR_PATTERN = re.compile(
    rf"^(\s*-?\s*['\"]?(?:{KEY_PATTERN})['\"]?\s*:\s*)&(?P<anchor>{ANCHOR_NAME_PATTERN})\s+(?P<value>.*)$",
    re.IGNORECASE,
)
ALIAS_SCALAR_PATTERN = re.compile(
    rf"^(\s*-?\s*['\"]?(?:{KEY_PATTERN})['\"]?\s*:\s*)\*(?P<alias>{ANCHOR_NAME_PATTERN})(?P<suffix>\s*)$",
    re.IGNORECASE,
)
FLOW_ANCHORED_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[{,\s])['\"]?(?:%s)['\"]?\s*:\s*)&(?P<anchor>%s)\s+(?P<value>(?:\"(?:\\.|[^\"])*\"|'(?:''|[^'])*'|[^,}]+))" % (KEY_PATTERN, ANCHOR_NAME_PATTERN),
    re.IGNORECASE,
)
FLOW_ALIAS_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[{,\s])['\"]?(?:%s)['\"]?\s*:\s*)\*(?P<alias>%s)(?P<suffix>(?:[,\s}]|$))" % (KEY_PATTERN, ANCHOR_NAME_PATTERN),
    re.IGNORECASE,
)

# Shared YAML scalar helpers are imported from api_docs_workflow_yaml_common.py

def decoded_path_lines(prefix, value, quote, suffix=""):
    decoded = decode_value(value, quote)
    lines = [line.strip() for line in decoded.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return [prefix + suffix]
    return [
        prefix + line + (suffix if index == len(lines) - 1 else "")
        for index, line in enumerate(lines)
    ]

def decode_quoted_paths(line):
    line = DOUBLE_FLOW_PATTERN.sub(
        lambda match: "\n".join(
            decoded_path_lines(match.group("prefix"), match.group("value"), '"'),
        ),
        line,
    )
    line = SINGLE_FLOW_PATTERN.sub(
        lambda match: "\n".join(
            decoded_path_lines(match.group("prefix"), match.group("value"), "'"),
        ),
        line,
    )
    return line

def normalize_aliases_and_quoted_paths(line):
    line = FLOW_ANCHORED_PATTERN.sub(
        lambda match: (
            anchors.setdefault(match.group("anchor"), decode_plain_value(match.group("value")))
            and match.group("prefix") + anchors[match.group("anchor")]
        ),
        line,
    )
    line = FLOW_ALIAS_PATTERN.sub(
        lambda match: match.group("prefix")
        + anchors.get(match.group("alias"), "*" + match.group("alias"))
        + match.group("suffix"),
        line,
    )
    return decode_quoted_paths(line)

def block_scalar_key_column(line):
    match = re.match(r"^[ ]*(?:-[ ]*)?([^:]+?)[ ]*:[ ]*[>|]", line)
    if match is None:
        return None
    return match.start(1)

anchors = {}
pending_prefix = None
pending_quote = None
pending_value = None
block_indent = None

sys.stdin = open(sys.argv[2], encoding="utf-8")
for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")
    in_block_body = False
    if block_indent is not None:
        if line.strip():
            line_indent = len(line) - len(line.lstrip(" "))
            if line_indent > block_indent:
                in_block_body = True
            else:
                block_indent = None
        else:
            in_block_body = True
    if not in_block_body:
        next_block_indent = block_scalar_key_column(line)
        if next_block_indent is not None:
            block_indent = next_block_indent
        anchor_match = NODE_ANCHOR_PATTERN.match(unquoted_yaml_text(line))
        if anchor_match is not None:
            anchors[anchor_match.group("anchor")] = decode_plain_value(anchor_match.group("value"))
    if pending_prefix is not None:
        current = line.lstrip()
        if pending_value.endswith("\\"):
            current = f"{pending_value[:-1]}{current}"
        else:
            current = f"{pending_value} {current}"
        tail = quoted_value_tail(current, pending_quote)
        if tail is None:
            pending_value = current
            continue
        value, suffix = tail
        for decoded_line in decoded_path_lines(pending_prefix, value, pending_quote, suffix):
            print(normalize_aliases_and_quoted_paths(decoded_line))
        pending_prefix = None
        pending_quote = None
        pending_value = None
        continue

    anchored_scalar = ANCHORED_SCALAR_PATTERN.match(line)
    if anchored_scalar is not None:
        prefix = anchored_scalar.group(1)
        anchor = anchored_scalar.group("anchor")
        value = decode_plain_value(anchored_scalar.group("value"))
        anchors[anchor] = value
        print(prefix + value)
        continue

    alias_scalar = ALIAS_SCALAR_PATTERN.match(line)
    if alias_scalar is not None:
        prefix = alias_scalar.group(1)
        alias = alias_scalar.group("alias")
        suffix = alias_scalar.group("suffix")
        print(prefix + anchors.get(alias, "*" + alias) + suffix)
        continue

    start = SCALAR_START_PATTERN.match(line)
    if start is not None:
        prefix, quote, value = start.groups()
        tail = quoted_value_tail(value, quote)
        if tail is None:
            pending_prefix = prefix
            pending_quote = quote
            pending_value = value
            continue
        decoded_value, suffix = tail
        for decoded_line in decoded_path_lines(prefix, decoded_value, quote, suffix):
            print(normalize_aliases_and_quoted_paths(decoded_line))
        continue

    flow_start = FLOW_START_PATTERN.match(line)
    if flow_start is not None:
        prefix = flow_start.group("prefix")
        quote = flow_start.group("quote")
        value = flow_start.group("value")
        tail = quoted_value_tail(value, quote)
        if tail is None:
            pending_prefix = prefix
            pending_quote = quote
            pending_value = value
            continue

    print(normalize_aliases_and_quoted_paths(line))

if pending_prefix is not None:
    for decoded_line in decoded_path_lines(pending_prefix, pending_value, pending_quote):
        print(normalize_aliases_and_quoted_paths(decoded_line))
PY
    rm -f "$path_decode_input"
}

fold_yaml_run_blocks() {
    python3 - "$ROOT_DIR/scripts" "$1" <<'PY'
import re
import sys

sys.path.insert(0, sys.argv[1])
from api_docs_workflow_yaml_common import (
    quoted_scalar_continues,
    strip_shell_comment_stateful,
    strip_yaml_comment,
    yaml_double_unescape,
)

# Shared parsing helpers are imported from api_docs_workflow_yaml_common.py

path = sys.argv[2]
with open(path, encoding="utf-8") as handle:
    lines = [line.rstrip("\n") for line in handle]

index = 0
while index < len(lines):
    line = strip_yaml_comment(lines[index])
    flow_run = False
    match = re.match(
        r"^([ ]*)(?:-[ ]*)?['\"]?(run)['\"]?[ ]*:[ ]*(.*)$",
        line,
        re.IGNORECASE,
    )
    if not match:
        match = re.match(
            r"^([ ]*)-[ ]*[{][^#]*?['\"]?(run)['\"]?[ ]*:[ ]*(.*)$",
            line,
            re.IGNORECASE,
        )
        flow_run = match is not None
    if not match:
        index += 1
        continue
    run_key_column = line.index(match.group(2))
    boundary_column = len(match.group(1)) if flow_run else run_key_column
    value = match.group(3).strip()
    if flow_run:
        value = re.sub(r"\s*}[,]?\s*$", "", value).strip()
    style_match = re.match(r"([>|])", value)
    style = style_match.group(1) if style_match else None
    index += 1
    block_lines = []

    def emit_shell_lines(logical_lines):
        pending = ""
        pending_state = None
        shell_state = (False, False, False)
        def emit_one(logical_line):
            nonlocal pending, pending_state, shell_state
            if logical_line is None:
                if pending:
                    stripped, shell_state = strip_shell_comment_stateful(
                        pending,
                        pending_state,
                    )
                    if stripped:
                        print(stripped)
                    pending = ""
                pending_state = None
                shell_state = (False, False, False)
                return
            current = logical_line.rstrip()
            scan_state = shell_state
            if pending:
                current = f"{pending}{current.lstrip()}"
                scan_state = pending_state
            stripped, shell_state = strip_shell_comment_stateful(
                current,
                scan_state,
            )
            if stripped.endswith("\\"):
                pending = stripped[:-1]
                pending_state = scan_state
            else:
                if stripped:
                    print(stripped)
                pending = ""
                pending_state = None

        for logical_line in logical_lines:
            if logical_line is None:
                emit_one(None)
                continue
            for embedded_line in logical_line.split("\n"):
                emit_one(embedded_line)
        if pending:
            stripped, shell_state = strip_shell_comment_stateful(
                pending,
                pending_state,
            )
            if stripped:
                print(stripped)

    def unquote_yaml_scalar(value):
        value = value.strip()
        if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
            return value[1:-1].replace("''", "'")
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            return value[1:-1]
        return value

    def unquote_yaml_scalar_lines(lines):
        unquoted = list(lines)
        first_index = next(
            (line_index for line_index, line in enumerate(unquoted) if line is not None),
            None,
        )
        last_index = next(
            (
                line_index
                for line_index in range(len(unquoted) - 1, -1, -1)
                if unquoted[line_index] is not None
            ),
            None,
        )
        if first_index is None or last_index is None:
            return unquoted
        first = unquoted[first_index].strip()
        last = unquoted[last_index].strip()
        if first and first[0] in "'\"":
            quote = first[0]
            if not last.endswith(quote):
                raw_last = unquoted[last_index].rstrip()
                escaped = False
                index = 1 if first_index == last_index else 0
                while index < len(raw_last):
                    char = raw_last[index]
                    if quote == '"' and escaped:
                        escaped = False
                    elif quote == '"' and char == "\\":
                        escaped = True
                    elif (
                        quote == "'"
                        and char == "'"
                        and index + 1 < len(raw_last)
                        and raw_last[index + 1] == "'"
                    ):
                        index += 1
                    elif char == quote:
                        suffix = raw_last[index + 1:]
                        if not suffix.strip() or suffix.lstrip().startswith("#"):
                            unquoted[last_index] = raw_last[: index + 1]
                            last = unquoted[last_index].strip()
                        break
                    index += 1
        if first and first[0] in "'\"" and last.endswith(first[0]):
            quote = first[0]
            unquoted[first_index] = unquoted[first_index].lstrip()[1:]
            unquoted[last_index] = unquoted[last_index].rstrip()[:-1]
            if quote == "'":
                unquoted = [
                    line.replace("''", "'") if line is not None else None
                    for line in unquoted
                ]
            else:
                merged = []
                pending = None
                for line in unquoted:
                    if line is None:
                        if pending is not None:
                            merged.append(yaml_double_unescape(pending))
                            pending = None
                        merged.append(None)
                        continue
                    current = line
                    if pending is not None:
                        current = f"{pending}{current.lstrip()}"
                        pending = None
                    if current.endswith("\\"):
                        pending = current[:-1]
                    else:
                        merged.append(yaml_double_unescape(current))
                if pending is not None:
                    merged.append(yaml_double_unescape(pending))
                unquoted = merged
        return unquoted

    raw_block_lines = []
    while index < len(lines):
        candidate = lines[index]
        if not candidate.strip():
            raw_block_lines.append(None)
            index += 1
            continue
        line_indent = len(candidate) - len(candidate.lstrip(" "))
        if line_indent <= boundary_column:
            break
        raw_block_lines.append((line_indent, candidate))
        index += 1

    nonblank_raw_indents = [
        block_line[0] for block_line in raw_block_lines if block_line is not None
    ]
    content_indent = min(nonblank_raw_indents) if nonblank_raw_indents else None
    yaml_quote = quoted_scalar_continues(value) if style is None else None
    for block_line in raw_block_lines:
        if block_line is None:
            block_lines.append(None)
            continue
        line_indent, raw_content = block_line
        content = raw_content[content_indent:] if content_indent is not None else raw_content
        if style:
            block_lines.append((line_indent, content))
        elif yaml_quote is not None:
            block_lines.append((line_indent, content))
            if quoted_scalar_continues(f"{yaml_quote}{content}") is None:
                yaml_quote = None
        else:
            block_lines.append((line_indent, strip_yaml_comment(content)))

    if style is None:
        scalar_lines = []
        if value:
            scalar_lines.append(value)
        for block_line in block_lines:
            if block_line is None:
                scalar_lines.append(None)
            else:
                scalar_lines.append(block_line[1])
        if scalar_lines:
            if flow_run:
                for scalar_index in range(len(scalar_lines) - 1, -1, -1):
                    if scalar_lines[scalar_index] is not None:
                        scalar_lines[scalar_index] = re.sub(
                            r"\s*}[,]?\s*$",
                            "",
                            scalar_lines[scalar_index],
                        )
                        break
            scalar_lines = unquote_yaml_scalar_lines(scalar_lines)
            logical_lines = []
            paragraph = []
            for scalar_line in scalar_lines:
                if scalar_line is None:
                    if paragraph:
                        logical_lines.append(" ".join(paragraph))
                        paragraph = []
                    logical_lines.append(None)
                else:
                    paragraph.append(scalar_line)
            if paragraph:
                logical_lines.append(" ".join(paragraph))
            emit_shell_lines(logical_lines)
        continue

    if not block_lines:
        continue

    if style == "|":
        emit_shell_lines(
            block_line[1] if block_line is not None else None
            for block_line in block_lines
        )
    else:
        nonblank_indents = [
            block_line[0] for block_line in block_lines if block_line is not None
        ]
        base_indent = min(nonblank_indents)
        folded_lines = []
        paragraph = []
        for block_line in block_lines:
            if block_line is None:
                if paragraph:
                    folded_lines.append(" ".join(paragraph))
                    paragraph = []
                folded_lines.append(None)
                continue
            line_indent, content = block_line
            if line_indent > base_indent:
                if paragraph:
                    folded_lines.append(" ".join(paragraph))
                    paragraph = []
                folded_lines.append(content)
            else:
                paragraph.append(content)
        if paragraph:
            folded_lines.append(" ".join(paragraph))
        emit_shell_lines(folded_lines)
PY
}

require_workflows_do_not_reference() {
    local needle="$1"
    local label="$2"
    local workflows_dir="$ROOT_DIR/.github/workflows"
    local workflow_file
    local matches

    if [ ! -d "$workflows_dir" ]; then
        pass "no workflow directory for $label"
        return
    fi

    matches="$(
        for workflow_file in "$workflows_dir"/*.yml "$workflows_dir"/*.yaml; do
            [ -f "$workflow_file" ] || continue
            { strip_yaml_comments "$workflow_file"; fold_yaml_run_blocks "$workflow_file"; } |
                tr '\\' '/' |
                tr '[:upper:]' '[:lower:]' |
                grep -F -n "$needle" || true
        done
    )"
    if [ -n "$matches" ]; then
        printf '%s\n' "$matches" >&2
        fail "workflows must not reference $label while generated API HTML is strengthened local-only"
    fi

    pass "no workflow $label references"
}

require_workflows_do_not_match() {
    local pattern="$1"
    local label="$2"
    local workflows_dir="$ROOT_DIR/.github/workflows"
    local workflow_file
    local matches

    if [ ! -d "$workflows_dir" ]; then
        pass "no workflow directory for $label"
        return
    fi

    matches="$(
        for workflow_file in "$workflows_dir"/*.yml "$workflows_dir"/*.yaml; do
            [ -f "$workflow_file" ] || continue
            { strip_yaml_comments "$workflow_file"; fold_yaml_run_blocks "$workflow_file"; } |
                tr '\\' '/' |
                tr '[:upper:]' '[:lower:]' |
                grep -E -n "$pattern" || true
        done
    )"
    if [ -n "$matches" ]; then
        printf '%s\n' "$matches" >&2
        fail "workflows must not reference $label while generated API HTML is strengthened local-only"
    fi

    pass "no workflow $label references"
}

check_no_workflow_publication_semantics() {
    local workflows_dir="$ROOT_DIR/.github/workflows"
    local workflow_file
    local rel_path
    local publication_regex
    local command_publication_regex
    local generated_path_regex
    local broad_path_regex
    local broad_block_path_regex
    local broad_command_path_regex
    local docs_staging_command_regex
    local docs_archive_command_regex
    local dynamic_publication_path_regex
    local dynamic_command_publication_regex
    local broad_inline_path_regex
    local dynamic_inline_path_regex
    local stripped_text
    local normalized_text
    local folded_run_text
    local dynamic_block_path_matches

    if [ ! -d "$workflows_dir" ]; then
        pass "no workflow directory for generated API publication semantics"
        return
    fi

    publication_regex="actions/upload-artifact|actions/upload-pages-artifact|actions/deploy-pages|actions/upload-release-asset|softprops/action-gh-release|github-pages|gh-pages|pages:|['\"]?uses['\"]?[[:space:]]*:[[:space:]]*['\"]?([^[:space:]'\"]*(publish|deploy|release|pages|artifact|upload|gh-pages)[^[:space:]'\"]*(@|$)|[.]/[^[:space:]'\"]*(publish|deploy|release|pages|artifact|upload|gh-pages)[^[:space:]'\"]*|docker://[^[:space:]'\"]*(publish|deploy|release|pages|artifact|upload|gh-pages)[^[:space:]'\"]*)"
    command_publication_regex="aws[[:space:]]+s3[[:space:]]+(sync|cp)|gsutil[[:space:]]+(-m[[:space:]]+)?(rsync|cp)|az[[:space:]]+storage[[:space:]]+blob[[:space:]]+upload|netlify[[:space:]]+deploy|vercel[[:space:]]+deploy|firebase[[:space:]]+deploy|wrangler[[:space:]]+pages[[:space:]]+deploy|surge[[:space:]]|gh[[:space:]]+release[[:space:]]+upload|rclone[[:space:]]+(copy|sync|move)|rsync[[:space:]].*:[^[:space:]]*|scp[[:space:]].*:[^[:space:]]*"
    generated_path_regex="docs/api(/|$)|docs/api/html"
    broad_path_regex='^[[:space:]]*["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*["'"'"']?(\.|[.]/|[.]/[*][*]|[*][*]([/][*])?|/|([.]/)?([^[:space:]"'"'"']+/)*([.][.]/)?docs($|[/.]|[*]|["'"'"'])[^[:space:]"'"'"']*|[$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?(/docs($|[/.]|[*]|["'"'"'])[^[:space:]"'"'"']*)?)["'"'"']?[[:space:]]*$'
    broad_block_path_regex='^[[:space:]]*-?[[:space:]]*["'"'"']?(\.|[.]/|[.]/[*][*]|[*][*]([/][*])?|/|([.]/)?([^[:space:]"'"'"']+/)*([.][.]/)?docs($|[/.[:space:]#]|[*]|["'"'"'])[^"'"'"']*|[$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?(/docs($|[/.[:space:]#]|[*]|["'"'"'])[^"'"'"']*)?)["'"'"']?[[:space:]]*$'
    broad_command_path_regex='(^|[[:space:]])["'"'"']?(([$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?)?([.]/)?docs($|[/.*[:space:]])|([.]/)?([*][*]([/][*])?|[.]([/][*][*])?)($|[[:space:]])|/($|[[:space:]]))'
    docs_staging_command_regex='(^|[[:space:]])(cp|mv|rsync)[[:space:]][^;&|]*["'"'"']?([.]/)?docs($|[/[:space:]"'"'"'])'
    docs_archive_command_regex='(^|[[:space:]])(tar|zip|7z|7za|7zr)[[:space:]][^;&|]*([[:space:]]|=)["'"'"']?([.]/)?docs($|[/[:space:]"'"'"'])'
    dynamic_publication_path_regex='^[[:space:]]*["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[^#]*([$][{][{]|[$][A-Za-z_][A-Za-z0-9_]*|[%][A-Za-z_][A-Za-z0-9_]*[%])'
    dynamic_command_publication_regex="($command_publication_regex)[^;&|]*([$][A-Za-z_][A-Za-z0-9_]*|[$][{][A-Za-z_][A-Za-z0-9_]*[}]|[$][{][{][^}]+[}][}])"
    broad_inline_path_regex='(^|[,{][[:space:]]*)["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*["'"'"']?(\.|[.]/|[.]/[*][*]|[*][*]([/][*])?|/|([.]/)?([^[:space:],"'"'"'}]+/)*([.][.]/)?docs($|[/.]|[*]|["'"'"',}])[^[:space:],"'"'"'}]*|[$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?(/docs($|[/.]|[*]|["'"'"',}])[^[:space:],"'"'"'}]*)?)["'"'"']?([[:space:],}]|$)'
    dynamic_inline_path_regex='(^|[,{][[:space:]]*)["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[^,}]*([$][{][{]|[$][a-z_][a-z0-9_]*|[%][a-z_][a-z0-9_]*[%])'

    for workflow_file in "$workflows_dir"/*.yml "$workflows_dir"/*.yaml; do
        [ -f "$workflow_file" ] || continue
        rel_path="${workflow_file#$ROOT_DIR/}"
        stripped_text="$(strip_yaml_comments "$workflow_file" | decode_yaml_publication_paths)"
        normalized_text="$(printf '%s\n' "$stripped_text" | tr '\\' '/' | tr '[:upper:]' '[:lower:]')"
        folded_run_text="$(fold_yaml_run_blocks "$workflow_file" | tr '\\' '/' | tr '[:upper:]' '[:lower:]')"
        dynamic_block_path_matches="$(
            printf '%s\n' "$normalized_text" |
                awk '
                    /^[[:space:]]*["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*[|>][+-]?[0-9]*[+-]?/ {
                        in_path_block = 1
                        block_indent = match($0, /[^ ]/) - 1
                        next
                    }
                    in_path_block && /^[^[:space:]]/ {
                        in_path_block = 0
                    }
                    in_path_block && /^[[:space:]]*[^[:space:]]/ {
                        line_indent = match($0, /[^ ]/) - 1
                        if (line_indent <= block_indent) {
                            in_path_block = 0
                        } else if ($0 ~ /([$][{][{]|[$][a-z_][a-z0-9_]*|[%][a-z_][a-z0-9_]*[%])/) {
                            print
                        }
                    }
                '
        )"
        if printf '%s\n' "$normalized_text" | grep -Eq "$generated_path_regex" &&
            printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex"; then
            fail "$rel_path combines generated API output paths with publication, artifact, or Pages semantics while generated API HTML is local-only"
        fi
        if { printf '%s\n' "$normalized_text" | grep -Eq "$broad_path_regex" ||
            printf '%s\n' "$normalized_text" | grep -Eq "$broad_inline_path_regex"; } &&
            printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex"; then
            fail "$rel_path publishes docs or repository roots that can include local generated API HTML while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq '["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*[|>][+-]?[0-9]*[+-]?' &&
            printf '%s\n' "$normalized_text" | grep -Eq "$broad_block_path_regex" &&
            printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex"; then
            fail "$rel_path publishes docs or repository roots that can include local generated API HTML while generated API HTML is local-only"
        fi
        if { printf '%s\n' "$normalized_text" | grep -Eq "$command_publication_regex" ||
            printf '%s\n' "$folded_run_text" | grep -Eq "$command_publication_regex"; } &&
            { printf '%s\n' "$normalized_text" | grep -Eq "$broad_command_path_regex" ||
                printf '%s\n' "$folded_run_text" | grep -Eq "$broad_command_path_regex"; }; then
            fail "$rel_path publishes docs or repository roots that can include local generated API HTML while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex" &&
            { printf '%s\n' "$normalized_text" | grep -Eq "$docs_staging_command_regex" ||
                printf '%s\n' "$folded_run_text" | grep -Eq "$docs_staging_command_regex"; }; then
            fail "$rel_path stages docs for publication or artifact upload while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex" &&
            { printf '%s\n' "$normalized_text" | grep -Eq "$docs_archive_command_regex" ||
                printf '%s\n' "$folded_run_text" | grep -Eq "$docs_archive_command_regex"; }; then
            fail "$rel_path archives docs for publication or artifact upload while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex" &&
            { printf '%s\n' "$normalized_text" | grep -Eq "$dynamic_publication_path_regex" ||
                printf '%s\n' "$normalized_text" | grep -Eq "$dynamic_inline_path_regex" ||
                [ -n "$dynamic_block_path_matches" ]; }; then
            fail "$rel_path uses dynamic publication paths while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$dynamic_command_publication_regex" ||
            printf '%s\n' "$folded_run_text" | grep -Eq "$dynamic_command_publication_regex"; then
            fail "$rel_path uses dynamic publication paths while generated API HTML is local-only"
        fi
    done

    pass "no workflow generated API publication semantics"
}

check_ignore_rules() {
    require_ignored "docs/api"
    require_ignored "docs/api/html"
    require_ignored "docs/api/html/index.html"
}

check_tracked_and_staged_absence() {
    local tracked
    local staged
    local visible_untracked

    staged="$(git -C "$ROOT_DIR" diff --cached --name-only -- docs/api)"
    require_empty \
        "no staged generated API files" \
        "generated API files under docs/api/ are staged; unstage them unless a future publication decision selects committed output" \
        "$staged"

    tracked="$(git -C "$ROOT_DIR" ls-files docs/api)"
    require_empty \
        "no tracked generated API files" \
        "generated API files under docs/api/ are tracked; local-only generated HTML must not be source-controlled" \
        "$tracked"

    visible_untracked="$(git -C "$ROOT_DIR" ls-files --others --exclude-standard docs/api)"
    require_empty \
        "no non-ignored generated API files" \
        "generated API files under docs/api/ are visible as non-ignored untracked files; keep local generated output ignored" \
        "$visible_untracked"
}

check_doxyfile_contract() {
    require_doxyfile_setting "INPUT" "include/" "public-header generated API input"
    require_doxyfile_setting "FILE_PATTERNS" "\\*.h" "public-header generated API input"
    require_doxyfile_setting "RECURSIVE" "NO" "top-level public-header scope"
    require_doxyfile_setting "OUTPUT_DIRECTORY" "docs/api" "ignored generated API output"
    require_doxyfile_setting "GENERATE_HTML" "YES" "local generated API HTML"
    require_doxyfile_setting "HTML_OUTPUT" "html" "ignored generated API HTML output"
}

check_product_status_wording() {
    require_file_contains \
        "README.md" \
        "selected local Doxygen freshness plus local-only staging guard" \
        "local-only freshness"

    require_file_contains \
        "docs/api_reference.md" \
        "The generated HTML tree is local-only generated output." \
        "local-only generated output"

    require_file_contains \
        "docs/api_reference.md" \
        "is not a hosted or source-controlled publication surface." \
        "not hosted or source-controlled"

    require_file_contains \
        "docs/maintainer_guide.md" \
        "The maintained Sprint 179 product decision keeps this tree" \
        "Sprint 179 product decision"

    require_file_contains \
        "docs/maintainer_guide.md" \
        "hosted, artifact-published, or release evidence." \
        "not hosted, artifact-published, or release evidence"
}

check_no_workflow_publication_path() {
    check_no_workflow_publication_semantics
    require_workflows_do_not_match \
        '(^|[^[:alnum:]_.-])([.]/|/)?docs/api([^[:alnum:]_./-]|$)' \
        "generated API output root"
    require_workflows_do_not_reference "docs/api/html" "generated API HTML output path"
    require_workflows_do_not_reference "docs/api/" "generated API output tree"
}

check_tracked_and_staged_absence
check_ignore_rules
check_doxyfile_contract
check_product_status_wording
check_no_workflow_publication_path

echo "api-docs-local-only: passed"
