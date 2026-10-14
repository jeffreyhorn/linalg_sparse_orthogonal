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
    decode_plain_value,
    decode_value,
    quoted_scalar_continues,
    quoted_value_tail,
    strip_shell_comment_stateful,
    strip_yaml_comment,
    strip_yaml_comment_continuation,
)

# Shared parsing helpers are imported from api_docs_workflow_yaml_common.py

path = sys.argv[2]

def complete_quoted_scalar(value):
    stripped = value.strip()
    if not stripped or stripped[0] not in "'\"":
        return value
    parsed = quoted_value_tail(stripped[1:], stripped[0])
    if parsed is None:
        return value
    scalar_body, _ = parsed
    return stripped[0] + scalar_body + stripped[0]

def line_ends_with_unescaped_backslash(line):
    backslashes = 0
    for char in reversed(line):
        if char != "\\":
            break
        backslashes += 1
    return backslashes % 2 == 1

def fold_quoted_scalar_body(body, quote):
    lines = body.splitlines()
    if not lines:
        return body

    folded = []
    current = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current is not None:
                folded.append(current)
                current = None
            folded.append("")
            continue
        if current is None:
            current = line
        elif quote == '"' and line_ends_with_unescaped_backslash(current):
            current = current[:-1] + line
        else:
            current += " " + line

    if current is not None:
        folded.append(current)
    return "\n".join(folded)

def mapping_entry(line):
    prefix_match = re.match(r"^(\s*(?:-\s*)?)", line)
    prefix = prefix_match.group(1)
    index = len(prefix)
    if index >= len(line):
        return None
    key_column = index
    if line[index] in "'\"":
        quote = line[index]
        index += 1
        key_parts = []
        escaped = False
        while index < len(line):
            char = line[index]
            if quote == '"' and escaped:
                key_parts.append("\\" + char)
                escaped = False
                index += 1
                continue
            if quote == '"' and char == "\\":
                escaped = True
                index += 1
                continue
            if quote == "'" and char == "'" and index + 1 < len(line) and line[index + 1] == "'":
                key_parts.append("''")
                index += 2
                continue
            if char == quote:
                break
            key_parts.append(char)
            index += 1
        if index >= len(line):
            return None
        key = decode_value("".join(key_parts), quote)
        index += 1
    else:
        key_start = index
        while index < len(line) and line[index] not in ":,{}[]":
            index += 1
        key = line[key_start:index].strip()
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != ":":
        return None
    return prefix, key_column, key.lower(), line[index + 1:]

def print_decoded_run_scalar(prefix, parts):
    scalar = complete_quoted_scalar("\n".join(parts)).strip()
    if len(scalar) >= 2 and scalar[0] == scalar[-1] and scalar[0] in "'\"":
        decoded = decode_value(fold_quoted_scalar_body(scalar[1:-1], scalar[0]), scalar[0])
    else:
        decoded = decode_plain_value(scalar)
    run_shell_state = (False, False, False)
    for decoded_line in decoded.splitlines():
        if not decoded_line.strip():
            run_shell_state = (False, False, False)
            continue
        run_value, run_shell_state = strip_shell_comment_stateful(
            decoded_line,
            run_shell_state,
        )
        if run_value.strip():
            print(prefix + run_value)

with open(path, encoding="utf-8") as handle:
    block_indent = None
    block_key = None
    quote_indent = None
    quote_char = None
    quote_key = None
    quote_prefix = None
    quote_parts = []
    shell_state = (False, False, False)
    for raw_line in handle:
        line = raw_line.rstrip("\n")
        if quote_char is not None:
            if line.strip():
                line_indent = len(line) - len(line.lstrip(" "))
                if line_indent <= quote_indent:
                    quote_char = None
                    quote_key = None
                    quote_prefix = None
                    quote_parts = []
                else:
                    if quote_key == "run":
                        quote_parts.append(line.strip())
                        candidate = "\n".join(quote_parts)
                        if quoted_scalar_continues(candidate) is None:
                            print_decoded_run_scalar(quote_prefix, quote_parts)
                            quote_char = None
                            quote_key = None
                            quote_prefix = None
                            quote_parts = []
                        continue
                    stripped = strip_yaml_comment_continuation(line, quote_char)
                    print(stripped)
                    if quoted_scalar_continues(f"{quote_char}{stripped}") is None:
                        quote_char = None
                        quote_key = None
                        quote_prefix = None
                        quote_parts = []
                    continue
            else:
                if quote_key == "run":
                    quote_parts.append("")
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
        entry = mapping_entry(stripped)
        if entry is not None and re.match(r"\s*[>|]", entry[3]):
            block_indent = entry[1]
            block_key = entry[2]
            shell_state = (False, False, False)
        else:
            if entry is not None and entry[2] == "run":
                run_value, _ = strip_shell_comment_stateful(
                    decode_plain_value(entry[3]),
                    (False, False, False),
                )
                stripped = entry[0] + "run: " + run_value
                entry = mapping_entry(stripped)
        if entry is not None:
            scalar_key = entry[2]
            scalar_value = entry[3]
            continuing_quote = quoted_scalar_continues(scalar_value)
            if continuing_quote is not None:
                quote_char = continuing_quote
                quote_indent = len(stripped) - len(stripped.lstrip(" "))
                quote_key = scalar_key
                if quote_key == "run":
                    quote_prefix = entry[0] + "run: "
                    quote_parts = [scalar_value.strip()]
                    continue
                quote_key = None
                quote_prefix = None
                quote_parts = []
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
    quoted_scalar_continues,
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
PUBLICATION_PATH_KEY_SET = {key.lower() for key in PUBLICATION_PATH_KEYS}

KEY_PATTERN = "|".join(re.escape(key) for key in PUBLICATION_PATH_KEYS)
DOUBLE_KEY_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[{,\s])\s*)\"(?P<key>(?:\\.|[^\"])*)\"(?P<suffix>\s*:)",
)
SINGLE_KEY_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[{,\s])\s*)'(?P<key>(?:''|[^'])*)'(?P<suffix>\s*:)",
)
DOUBLE_KEY_START_PATTERN = re.compile(
    r'^(?P<prefix>\s*(?:-\s*)?)"(?P<key>(?:\\.|[^"])*)$',
)
SINGLE_KEY_START_PATTERN = re.compile(
    r"^(?P<prefix>\s*(?:-\s*)?)'(?P<key>(?:''|[^'])*)$",
)
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
GENERIC_DOUBLE_FLOW_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[{,\s])['\"]?[^:'\",{}\[\]]+['\"]?\s*:\s*)\"(?P<value>(?:\\.|[^\"])*)\"",
)
GENERIC_SINGLE_FLOW_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[{,\s])['\"]?[^:'\",{}\[\]]+['\"]?\s*:\s*)'(?P<value>(?:''|[^'])*)'",
)
GENERIC_SCALAR_START_PATTERN = re.compile(
    r"^(\s*-?\s*(?:\"(?:\\.|[^\"])*\"|'(?:''|[^'])*'|[^:'\",{}\[\]]+)\s*:\s*)(['\"])(.*)$",
)
ANCHOR_NAME_PATTERN = r"[^\s\[\]\{\},]+"
NODE_ANCHOR_PATTERN = re.compile(
    r"^\s*(?:-\s*)?(?:['\"]?[^:'\"]+['\"]?\s*:\s*)?&(?P<anchor>%s)\s+(?P<value>(?:\"(?:\\.|[^\"])*\"|'(?:''|[^'])*'|[^#,\]}]+))"
    % ANCHOR_NAME_PATTERN,
)
FLOW_NODE_ANCHOR_PATTERN = re.compile(
    r"(?:^|[{,\s])['\"]?[^:'\",{}\[\]]+['\"]?\s*:\s*&(?P<anchor>%s)\s+(?P<value>(?:\"(?:\\.|[^\"])*\"|'(?:''|[^'])*'|[^,}]+))"
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

def decode_quoted_publication_keys(line):
    def replace(match, quote):
        decoded_key = decode_value(match.group("key"), quote)
        if decoded_key.lower() not in PUBLICATION_PATH_KEY_SET:
            return match.group(0)
        return match.group("prefix") + decoded_key + match.group("suffix")

    line = DOUBLE_KEY_PATTERN.sub(lambda match: replace(match, '"'), line)
    return SINGLE_KEY_PATTERN.sub(lambda match: replace(match, "'"), line)

def publication_key_start(line):
    for pattern, quote in (
        (DOUBLE_KEY_START_PATTERN, '"'),
        (SINGLE_KEY_START_PATTERN, "'"),
    ):
        match = pattern.match(line)
        if match is not None:
            return match, quote
    return None

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
    line = GENERIC_DOUBLE_FLOW_PATTERN.sub(
        lambda match: match.group("prefix") + decode_value(match.group("value"), '"'),
        line,
    )
    line = GENERIC_SINGLE_FLOW_PATTERN.sub(
        lambda match: match.group("prefix") + decode_value(match.group("value"), "'"),
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
pending_anchor = None
pending_anchor_quote = None
pending_anchor_value = None
pending_anchored_prefix = None
pending_anchored_anchor = None
pending_anchored_quote = None
pending_anchored_value = None
pending_key_prefix = None
pending_key_quote = None
pending_key_value = None
pending_generic_prefix = None
pending_generic_quote = None
pending_generic_value = None
pending_prefix = None
pending_quote = None
pending_value = None
block_indent = None
quote_indent = None
quote_char = None

def register_anchor(anchor, value):
    stripped = value.strip()
    if stripped and stripped[0] in "'\"":
        tail = quoted_value_tail(stripped[1:], stripped[0])
        if tail is None:
            return False
        scalar_body, _ = tail
        anchors[anchor] = decode_value(scalar_body, stripped[0])
    else:
        anchors[anchor] = decode_plain_value(value)
    return True

sys.stdin = open(sys.argv[2], encoding="utf-8")
for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")
    if pending_key_prefix is not None:
        current = line.lstrip()
        if pending_key_value.endswith("\\"):
            current = f"{pending_key_value[:-1]}{current}"
        else:
            current = f"{pending_key_value} {current}"
        tail = quoted_value_tail(current, pending_key_quote)
        if tail is None:
            pending_key_value = current
            continue
        key, suffix = tail
        decoded_key = decode_value(key, pending_key_quote)
        line = pending_key_prefix + decoded_key + suffix
        pending_key_prefix = None
        pending_key_quote = None
        pending_key_value = None
    else:
        key_start = publication_key_start(line)
        if key_start is not None:
            key_match, key_quote = key_start
            pending_key_prefix = key_match.group("prefix")
            pending_key_quote = key_quote
            pending_key_value = key_match.group("key")
            continue
    line = decode_quoted_publication_keys(line)
    if pending_anchored_prefix is not None:
        current = line.lstrip()
        if pending_anchored_value.endswith("\\"):
            current = f"{pending_anchored_value[:-1]}{current}"
        else:
            current = f"{pending_anchored_value} {current}"
        tail = quoted_value_tail(current, pending_anchored_quote)
        if tail is None:
            pending_anchored_value = current
            continue
        value, suffix = tail
        decoded = decode_value(value, pending_anchored_quote)
        anchors[pending_anchored_anchor] = decoded
        print(pending_anchored_prefix + decoded + suffix)
        pending_anchored_prefix = None
        pending_anchored_anchor = None
        pending_anchored_quote = None
        pending_anchored_value = None
        continue
    if pending_anchor is not None:
        current = line.lstrip()
        if pending_anchor_value.endswith("\\"):
            current = f"{pending_anchor_value[:-1]}{current}"
        else:
            current = f"{pending_anchor_value} {current}"
        tail = quoted_value_tail(current, pending_anchor_quote)
        if tail is None:
            pending_anchor_value = current
            continue
        value, suffix = tail
        anchors[pending_anchor] = decode_value(value, pending_anchor_quote)
        pending_anchor = None
        pending_anchor_quote = None
        pending_anchor_value = None
        if not suffix.strip():
            continue
        line = suffix
    if pending_generic_prefix is not None:
        current = line.lstrip()
        if pending_generic_value.endswith("\\"):
            current = f"{pending_generic_value[:-1]}{current}"
        else:
            current = f"{pending_generic_value} {current}"
        tail = quoted_value_tail(current, pending_generic_quote)
        if tail is None:
            pending_generic_value = current
            continue
        value, suffix = tail
        decoded = decode_value(value, pending_generic_quote)
        print(normalize_aliases_and_quoted_paths(pending_generic_prefix + decoded + suffix))
        pending_generic_prefix = None
        pending_generic_quote = None
        pending_generic_value = None
        continue
    in_block_body = False
    in_quoted_body = False
    if block_indent is not None:
        if line.strip():
            line_indent = len(line) - len(line.lstrip(" "))
            if line_indent > block_indent:
                in_block_body = True
            else:
                block_indent = None
        else:
            in_block_body = True
    if quote_char is not None:
        if line.strip():
            line_indent = len(line) - len(line.lstrip(" "))
            if line_indent > quote_indent:
                in_quoted_body = True
                if quoted_scalar_continues(f"{quote_char}{line.strip()}") is None:
                    quote_indent = None
                    quote_char = None
            else:
                quote_indent = None
                quote_char = None
        else:
            in_quoted_body = True
    if not in_block_body and not in_quoted_body:
        next_block_indent = block_scalar_key_column(line)
        if next_block_indent is not None:
            block_indent = next_block_indent
        scalar_match = re.match(r"^[ ]*(?:-[ ]*)?['\"]?[^:]+['\"]?[ ]*:[ ]*(.*)$", line)
        if scalar_match is not None:
            continuing_quote = quoted_scalar_continues(scalar_match.group(1))
            if continuing_quote is not None:
                quote_indent = len(line) - len(line.lstrip(" "))
                quote_char = continuing_quote
        if ANCHORED_SCALAR_PATTERN.match(line) is None:
            anchor_match = NODE_ANCHOR_PATTERN.match(line)
            if anchor_match is not None:
                anchor_value = anchor_match.group("value")
                if not register_anchor(anchor_match.group("anchor"), anchor_value):
                    pending_anchor = anchor_match.group("anchor")
                    pending_anchor_quote = anchor_value.strip()[0]
                    pending_anchor_value = anchor_value.strip()[1:]
            if "{" in line:
                for anchor_match in FLOW_NODE_ANCHOR_PATTERN.finditer(unquoted_yaml_text(line)):
                    anchor_value = anchor_match.group("value")
                    if not register_anchor(anchor_match.group("anchor"), anchor_value):
                        pending_anchor = anchor_match.group("anchor")
                        pending_anchor_quote = anchor_value.strip()[0]
                        pending_anchor_value = anchor_value.strip()[1:]
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
        raw_value = anchored_scalar.group("value").strip()
        if raw_value and raw_value[0] in "'\"":
            tail = quoted_value_tail(raw_value[1:], raw_value[0])
            if tail is None:
                pending_anchored_prefix = prefix
                pending_anchored_anchor = anchor
                pending_anchored_quote = raw_value[0]
                pending_anchored_value = raw_value[1:]
                continue
            scalar_value, suffix = tail
            value = decode_value(scalar_value, raw_value[0])
            anchors[anchor] = value
            print(prefix + value + suffix)
            continue
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

    generic_start = GENERIC_SCALAR_START_PATTERN.match(line)
    if generic_start is not None:
        prefix, quote, value = generic_start.groups()
        tail = quoted_value_tail(value, quote)
        if tail is None:
            pending_generic_prefix = prefix
            pending_generic_quote = quote
            pending_generic_value = value
            continue
        decoded_value, suffix = tail
        print(normalize_aliases_and_quoted_paths(prefix + decode_value(decoded_value, quote) + suffix))
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
    decode_value,
    quoted_scalar_continues,
    strip_shell_comment_stateful,
    strip_yaml_comment,
    yaml_double_unescape,
)

# Shared parsing helpers are imported from api_docs_workflow_yaml_common.py

path = sys.argv[2]
with open(path, encoding="utf-8") as handle:
    lines = [line.rstrip("\n") for line in handle]

def flow_member_value(value):
    in_single = False
    in_double = False
    escaped = False
    depth = 0
    for value_index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char == "'" and not in_double:
            if in_single and value_index + 1 < len(value) and value[value_index + 1] == "'":
                escaped = True
                continue
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if char in "[{":
            depth += 1
            continue
        if char in "]}":
            if depth > 0:
                depth -= 1
            continue
        if char == "," and depth == 0:
            return value[:value_index].strip()
    return re.sub(r"\s*}[,]?\s*$", "", value).strip()

def flow_mapping_balance(value):
    in_single = False
    in_double = False
    escaped = False
    balance = 0
    for char in value:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
    return balance

ANCHOR_NAME_PATTERN = r"[^\s\[\]\{\},]+"
NODE_ANCHOR_PATTERN = re.compile(
    r"^\s*(?:-\s*)?(?:['\"]?[^:'\"]+['\"]?\s*:\s*)?&(?P<anchor>%s)\s+(?P<value>.+)$"
    % ANCHOR_NAME_PATTERN,
)
RUN_ALIAS_PATTERN = re.compile(r"^\*(?P<alias>%s)\s*$" % ANCHOR_NAME_PATTERN)
BLOCK_ANCHOR_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?:-\s*)?(?:(?P<key>['\"]?[^:'\"]+['\"]?)\s*:\s*)?&(?P<anchor>%s)\s*(?P<style>[>|])"
    % ANCHOR_NAME_PATTERN,
)

def decode_anchor_value(value):
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == "'" and stripped[-1] == "'":
        return decode_value(fold_quoted_anchor_body(stripped[1:-1], "'"), "'")
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return decode_value(fold_quoted_anchor_body(stripped[1:-1], '"'), '"')
    return stripped

def anchor_line_continues(value):
    stripped = value.strip()
    if not stripped or stripped[0] not in "'\"":
        return False
    return quoted_scalar_continues(stripped) is not None

def line_ends_with_unescaped_backslash(line):
    backslashes = 0
    for char in reversed(line):
        if char != "\\":
            break
        backslashes += 1
    return backslashes % 2 == 1

def fold_quoted_anchor_body(body, quote):
    lines = body.splitlines()
    if not lines:
        return body
    folded = []
    current = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current is not None:
                folded.append(current)
                current = None
            folded.append("")
            continue
        if current is None:
            current = line
        elif quote == '"' and line_ends_with_unescaped_backslash(current):
            current = current[:-1] + line
        else:
            current += " " + line
    if current is not None:
        folded.append(current)
    return "\n".join(folded)

def register_run_anchor(lines, index, anchors):
    line = lines[index]
    anchor_match = NODE_ANCHOR_PATTERN.match(line)
    if anchor_match is None:
        return index
    value_parts = [anchor_match.group("value").strip()]
    boundary_column = len(line) - len(line.lstrip(" "))
    next_index = index + 1
    while anchor_line_continues("\n".join(value_parts)) and next_index < len(lines):
        candidate = lines[next_index]
        if not candidate.strip():
            value_parts.append("")
            next_index += 1
            continue
        line_indent = len(candidate) - len(candidate.lstrip(" "))
        if line_indent <= boundary_column:
            break
        value_parts.append(candidate.strip())
        next_index += 1
    anchors[anchor_match.group("anchor")] = decode_anchor_value("\n".join(value_parts))
    return next_index - 1

def anchored_block_entry(line):
    match = BLOCK_ANCHOR_PATTERN.match(line)
    if match is None:
        return None
    key = match.group("key")
    if key is not None:
        key = key.strip()
        if len(key) >= 2 and key[0] == "'" and key[-1] == "'":
            key = decode_value(key[1:-1], "'")
        elif len(key) >= 2 and key[0] == '"' and key[-1] == '"':
            key = decode_value(key[1:-1], '"')
        key = key.lower()
    boundary_column = match.start("key") if key is not None else len(match.group("indent"))
    return boundary_column, key, match.group("anchor"), match.group("style")

def render_anchored_block(lines, start_index, boundary_column, style):
    raw_block_lines = []
    index = start_index
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

    nonblank_indents = [
        block_line[0] for block_line in raw_block_lines if block_line is not None
    ]
    content_indent = min(nonblank_indents) if nonblank_indents else None
    block_lines = []
    for block_line in raw_block_lines:
        if block_line is None:
            block_lines.append(None)
            continue
        _, raw_content = block_line
        block_lines.append((
            block_line[0],
            raw_content[content_indent:] if content_indent is not None else raw_content,
        ))

    if style == "|":
        return "\n".join(line[1] if line is not None else "" for line in block_lines), index

    paragraphs = []
    paragraph = []
    for block_line in block_lines:
        if block_line is None:
            if paragraph:
                paragraphs.append(" ".join(paragraph))
                paragraph = []
            paragraphs.append("")
            continue
        line_indent, content = block_line
        if line_indent > content_indent:
            if paragraph:
                paragraphs.append(" ".join(paragraph))
                paragraph = []
            paragraphs.append(content)
        else:
            paragraph.append(content)
    if paragraph:
        paragraphs.append(" ".join(paragraph))
    return "\n".join(paragraphs), index

def run_key_entry(line):
    prefix_match = re.match(r"^([ ]*)(?:-[ ]*)?", line)
    index = len(prefix_match.group(0))
    if index >= len(line):
        return None
    key_column = index
    if line[index] in "'\"":
        quote = line[index]
        index += 1
        key_parts = []
        escaped = False
        while index < len(line):
            char = line[index]
            if quote == '"' and escaped:
                key_parts.append("\\" + char)
                escaped = False
                index += 1
                continue
            if quote == '"' and char == "\\":
                escaped = True
                index += 1
                continue
            if quote == "'" and char == "'" and index + 1 < len(line) and line[index + 1] == "'":
                key_parts.append("''")
                index += 2
                continue
            if char == quote:
                break
            key_parts.append(char)
            index += 1
        if index >= len(line):
            return None
        key = decode_value("".join(key_parts), quote)
        index += 1
    else:
        key_start = index
        while index < len(line) and line[index] not in ":,{}[]":
            index += 1
        key = line[key_start:index].strip()
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != ":":
        return None
    if key.lower() != "run":
        return None
    return key_column, line[index + 1:].strip()

def steps_flow_sequence_entry(line):
    indent = len(line) - len(line.lstrip(" "))
    index = indent
    if index >= len(line):
        return None
    if line[index] in "'\"":
        quote = line[index]
        index += 1
        key_parts = []
        escaped = False
        while index < len(line):
            char = line[index]
            if quote == '"' and escaped:
                key_parts.append("\\" + char)
                escaped = False
                index += 1
                continue
            if quote == '"' and char == "\\":
                escaped = True
                index += 1
                continue
            if quote == "'" and char == "'" and index + 1 < len(line) and line[index + 1] == "'":
                key_parts.append("''")
                index += 2
                continue
            if char == quote:
                break
            key_parts.append(char)
            index += 1
        if index >= len(line):
            return None
        key = decode_value("".join(key_parts), quote)
        index += 1
    else:
        key_start = index
        while index < len(line) and line[index] not in ":,{}[]":
            index += 1
        key = line[key_start:index].strip()
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != ":":
        return None
    if key.lower() != "steps":
        return None
    index += 1
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != "[":
        return None
    return indent, index + 1

def flow_run_entry(line, open_flow_indent=None):
    in_steps_sequence = False
    start_match = re.match(r"^([ ]*)-[ ]*[{]", line)
    if start_match is not None:
        flow_indent = len(start_match.group(1))
        index = line.find("{", start_match.end() - 1) + 1
    elif (steps_entry := steps_flow_sequence_entry(line)) is not None:
        in_steps_sequence = True
        flow_indent, sequence_start = steps_entry
        flow_start = line.find("{", sequence_start)
        if flow_start == -1:
            return None
        index = flow_start + 1
    elif open_flow_indent is not None:
        in_steps_sequence = True
        if not line.strip():
            return None
        line_indent = len(line) - len(line.lstrip(" "))
        if line_indent <= open_flow_indent:
            return None
        flow_indent = open_flow_indent
        index = line_indent
        while index < len(line) and line[index] in " \t,":
            index += 1
        if index < len(line) and line[index] == "{":
            index += 1
    else:
        return None
    while index < len(line):
        while index < len(line) and line[index] in " \t,":
            index += 1
        if index >= len(line):
            return None
        if line[index] == "}":
            next_mapping = re.match(r"}\s*,\s*{", line[index:])
            if not in_steps_sequence or next_mapping is None:
                return None
            index += next_mapping.end()
            continue
        key_column = index
        if line[index] in "'\"":
            quote = line[index]
            index += 1
            key_start = index
            escaped = False
            key_parts = []
            while index < len(line):
                char = line[index]
                if quote == '"' and escaped:
                    key_parts.append("\\" + char)
                    escaped = False
                    index += 1
                    continue
                if quote == '"' and char == "\\":
                    escaped = True
                    index += 1
                    continue
                if quote == "'" and char == "'" and index + 1 < len(line) and line[index + 1] == "'":
                    key_parts.append("''")
                    index += 2
                    continue
                if char == quote:
                    break
                key_parts.append(char)
                index += 1
            if index >= len(line):
                return None
            key = decode_value("".join(key_parts), quote)
            index += 1
        else:
            key_start = index
            while index < len(line) and line[index] not in ":,{}[]":
                index += 1
            key = line[key_start:index].strip()
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line) or line[index] != ":":
            return None
        index += 1
        value_start = index
        if key.lower() == "run":
            return " " * flow_indent, key_column, line[value_start:]

        in_single = False
        in_double = False
        escaped = False
        depth = 0
        while index < len(line):
            char = line[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\" and in_double:
                escaped = True
                index += 1
                continue
            if char == "'" and not in_double:
                if in_single and index + 1 < len(line) and line[index + 1] == "'":
                    index += 2
                    continue
                in_single = not in_single
                index += 1
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                index += 1
                continue
            if in_single or in_double:
                index += 1
                continue
            if char in "[{":
                depth += 1
            elif char in "]}":
                if depth == 0:
                    break
                depth -= 1
            elif char == "," and depth == 0:
                index += 1
                break
            index += 1
    return None

def flow_member_start(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("}"):
        return False
    if stripped.startswith(","):
        stripped = stripped[1:].lstrip()
    if stripped.startswith("{"):
        stripped = stripped[1:].lstrip()
    if stripped.startswith(("'", '"')):
        quote = stripped[0]
        index = 1
        escaped = False
        while index < len(stripped):
            char = stripped[index]
            if quote == '"' and escaped:
                escaped = False
                index += 1
                continue
            if quote == '"' and char == "\\":
                escaped = True
                index += 1
                continue
            if quote == "'" and char == "'" and index + 1 < len(stripped) and stripped[index + 1] == "'":
                index += 2
                continue
            if char == quote:
                index += 1
                break
            index += 1
        else:
            return False
    else:
        index = 0
        while index < len(stripped) and stripped[index] not in ":,{}[]":
            index += 1
        if index == 0:
            return False
    while index < len(stripped) and stripped[index].isspace():
        index += 1
    return index < len(stripped) and stripped[index] == ":"

def normalize_shell_words(line):
    normalized = []
    in_single = False
    in_double = False
    escaped = False
    index = 0
    def is_shell_word_char(char):
        return char not in " \t\r\n;&|()<>"

    while index < len(line):
        char = line[index]
        if escaped:
            normalized.append(char)
            escaped = False
            index += 1
            continue
        if char == "\\" and not in_single:
            escaped = True
            index += 1
            continue
        if char == "'" and not in_double:
            if index + 1 < len(line) and line[index + 1] == "'":
                index += 2
                continue
            prev_char = line[index - 1] if index > 0 else ""
            next_char = line[index + 1] if index + 1 < len(line) else ""
            quoted_command = (
                re.search(r"(?:^|[;&|])\s*$", "".join(normalized)) is not None
                and re.match(r"'[A-Za-z_][A-Za-z0-9_-]*'(?=\s|[;&|]|$)", line[index:]) is not None
            )
            if in_single or quoted_command or (
                prev_char
                and next_char
                and is_shell_word_char(prev_char)
                and is_shell_word_char(next_char)
            ):
                pass
            else:
                normalized.append(char)
            in_single = not in_single
            index += 1
            continue
        if char == '"' and not in_single:
            if index + 1 < len(line) and line[index + 1] == '"':
                index += 2
                continue
            prev_char = line[index - 1] if index > 0 else ""
            next_char = line[index + 1] if index + 1 < len(line) else ""
            quoted_command = (
                re.search(r"(?:^|[;&|])\s*$", "".join(normalized)) is not None
                and re.match(r'"[A-Za-z_][A-Za-z0-9_-]*"(?=\s|[;&|]|$)', line[index:]) is not None
            )
            if in_double or quoted_command or (
                prev_char
                and next_char
                and is_shell_word_char(prev_char)
                and is_shell_word_char(next_char)
            ):
                pass
            else:
                normalized.append(char)
            in_double = not in_double
            index += 1
            continue
        normalized.append(char)
        index += 1
    if escaped:
        normalized.append("\\")
    return "".join(normalized)

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
                print(normalize_shell_words(stripped))
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
            print(normalize_shell_words(stripped))

index = 0
open_flow_indent = None
run_anchors = {}
while index < len(lines):
    line = strip_yaml_comment(lines[index])
    block_anchor = anchored_block_entry(line)
    if block_anchor is not None:
        boundary_column, key, anchor, block_style = block_anchor
        rendered_block, index = render_anchored_block(
            lines,
            index + 1,
            boundary_column,
            block_style,
        )
        run_anchors[anchor] = rendered_block
        if key == "run":
            emit_shell_lines(rendered_block.split("\n"))
        continue
    index = register_run_anchor(lines, index, run_anchors)
    if open_flow_indent is not None and line.strip():
        line_indent = len(line) - len(line.lstrip(" "))
        if line_indent <= open_flow_indent:
            open_flow_indent = None
    flow_run = False
    run_entry = run_key_entry(line)
    if run_entry is None:
        flow_entry = flow_run_entry(line, open_flow_indent)
        if flow_entry is not None:
            flow_indent, flow_key_column, flow_value = flow_entry
            run_entry = (flow_key_column, flow_value.strip())
            flow_run = True
    if run_entry is None:
        start_match = re.match(r"^([ ]*)-[ ]*[{]", line)
        if start_match is not None:
            remainder = line[line.find("{", start_match.end() - 1):]
            if flow_mapping_balance(remainder) > 0:
                open_flow_indent = len(start_match.group(1))
        elif (steps_entry := steps_flow_sequence_entry(line)) is not None:
            steps_indent, sequence_start = steps_entry
            flow_start = line.find("{", sequence_start)
            if flow_start != -1 and flow_mapping_balance(line[flow_start:]) > 0:
                open_flow_indent = steps_indent
            elif flow_start == -1 and not re.search(r"\]", line):
                open_flow_indent = steps_indent
        elif open_flow_indent is not None and flow_mapping_balance(line.lstrip()) < 0:
            open_flow_indent = None
        index += 1
        continue
    if flow_run:
        run_key_column = flow_key_column
        boundary_column = len(flow_indent)
        value = run_entry[1]
        open_flow_indent = None
    else:
        run_key_column = run_entry[0]
        boundary_column = run_key_column
        value = run_entry[1]
    if flow_run:
        value = flow_member_value(value)
    alias_match = RUN_ALIAS_PATTERN.match(value.strip())
    if alias_match is not None:
        value = run_anchors.get(alias_match.group("alias"), value)
    style_match = re.match(r"([>|])", value)
    style = style_match.group(1) if style_match else None
    index += 1
    block_lines = []

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
                        if (
                            not suffix.strip()
                            or suffix.lstrip().startswith(("#", "}", ","))
                        ):
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
        if flow_run and flow_member_start(candidate):
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

normalize_workflow_dot_segments() {
    local dot_segment_input
    dot_segment_input="$(mktemp)"
    cat > "$dot_segment_input"
    python3 - "$dot_segment_input" <<'PY'
import re
import sys

PATH_TOKEN = re.compile(r"(?P<prefix>^|[\s\"'=:{,\[])(?P<path>[.]?[^\s\"';&|,}]*[/][^\s\"';&|,}]*)")

def normalize_path(path):
    path = path.replace("\\", "/")
    absolute = path.startswith("/")
    trailing = path.endswith("/")
    parts = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not absolute:
                parts.append(part)
            continue
        parts.append(part)
    normalized = ("/" if absolute else "") + "/".join(parts)
    if trailing and normalized and not normalized.endswith("/"):
        normalized += "/"
    if not normalized:
        return "/" if absolute else "."
    return normalized

with open(sys.argv[1], encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.rstrip("\n")
        print(
            PATH_TOKEN.sub(
                lambda match: match.group("prefix") + normalize_path(match.group("path")),
                line,
            )
        )
PY
    rm -f "$dot_segment_input"
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
            { strip_yaml_comments "$workflow_file" | decode_yaml_publication_paths; fold_yaml_run_blocks "$workflow_file"; } |
                tr '\\' '/' |
                tr '[:upper:]' '[:lower:]' |
                normalize_workflow_dot_segments |
                awk -v needle="$needle" 'index($0, needle) { print NR ":" $0 }' || true
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
            { strip_yaml_comments "$workflow_file" | decode_yaml_publication_paths; fold_yaml_run_blocks "$workflow_file"; } |
                tr '\\' '/' |
                tr '[:upper:]' '[:lower:]' |
                normalize_workflow_dot_segments |
                awk -v pattern="$pattern" '$0 ~ pattern { print NR ":" $0 }' || true
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
    local dynamic_staging_command_regex
    local dynamic_powershell_staging_regex
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
    docs_staging_command_regex='(^|[[:space:]])(cp|mv|rsync|copy-item|move-item)[[:space:]][^;&|]*["'"'"']?([.]/)?docs($|[/[:space:]"'"'"'])'
    docs_archive_command_regex='(^|[[:space:]])(tar|zip|7z|7za|7zr|compress-archive)[[:space:]][^;&|]*([[:space:]]|=)["'"'"']?([.]/)?docs($|[/[:space:]"'"'"'])'
    dynamic_publication_path_regex='^[[:space:]]*["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[^#]*([$][{][{]|[$][A-Za-z_][A-Za-z0-9_]*|[%][A-Za-z_][A-Za-z0-9_]*[%])'
    dynamic_command_publication_regex="($command_publication_regex)[^;&|]*([$][A-Za-z_][A-Za-z0-9_]*|[$][{][A-Za-z_][A-Za-z0-9_]*[}]|[$][{][{][^}]+[}][}])"
    dynamic_staging_command_regex='(^|[[:space:]])(cp|mv|rsync)([[:space:]]+-[^[:space:]]+)*[[:space:]]+["'"'"']?[^"'"'"';&|[:space:]]*([$][(]|[$][A-Za-z_][A-Za-z0-9_]*|[$][{][^}]+[}]|[$][{][{][^}]+[}][}])[^"'"'"';&|[:space:]]*["'"'"']?([/[:space:]]|$)'
    broad_inline_path_regex='(^|[,{][[:space:]]*)["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*["'"'"']?(\.|[.]/|[.]/[*][*]|[*][*]([/][*])?|/|([.]/)?([^[:space:],"'"'"'}]+/)*([.][.]/)?docs($|[/.]|[*]|["'"'"',}])[^[:space:],"'"'"'}]*|[$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?(/docs($|[/.]|[*]|["'"'"',}])[^[:space:],"'"'"'}]*)?)["'"'"']?([[:space:],}]|$)'
    dynamic_inline_path_regex='(^|[,{][[:space:]]*)["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[^,}]*([$][{][{]|[$][a-z_][a-z0-9_]*|[%][a-z_][a-z0-9_]*[%])'
    dynamic_powershell_staging_regex='(^|[[:space:]])(copy-item|move-item|compress-archive)[[:space:]][^;&|]*([$][(]|[$][a-z_][a-z0-9_:]*|[$][{][^}]+[}])'

    for workflow_file in "$workflows_dir"/*.yml "$workflows_dir"/*.yaml; do
        [ -f "$workflow_file" ] || continue
        rel_path="${workflow_file#$ROOT_DIR/}"
        stripped_text="$(strip_yaml_comments "$workflow_file" | decode_yaml_publication_paths)"
        normalized_text="$(printf '%s\n' "$stripped_text" | tr '\\' '/' | tr '[:upper:]' '[:lower:]' | normalize_workflow_dot_segments)"
        folded_run_text="$(fold_yaml_run_blocks "$workflow_file" | tr '\\' '/' | tr '[:upper:]' '[:lower:]' | normalize_workflow_dot_segments)"
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
            { printf '%s\n' "$normalized_text" | grep -Eq "$dynamic_staging_command_regex" ||
                printf '%s\n' "$folded_run_text" | grep -Eq "$dynamic_staging_command_regex" ||
                printf '%s\n' "$normalized_text" | grep -Eq "$dynamic_powershell_staging_regex" ||
                printf '%s\n' "$folded_run_text" | grep -Eq "$dynamic_powershell_staging_regex"; }; then
            fail "$rel_path uses dynamic staging paths while generated API HTML is local-only"
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
