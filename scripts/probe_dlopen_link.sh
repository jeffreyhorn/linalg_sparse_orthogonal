#!/bin/sh

set -eu

mode=${1:?usage: probe_dlopen_link.sh <plain|libdl>}
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/sparse-dlopen.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

cat >"$tmpdir/probe.c" <<'EOF'
#include <dlfcn.h>

int main(void) { return dlopen(0, RTLD_LAZY) == 0; }
EOF

set -- ${CFLAGS:-}

if [ "$mode" = "plain" ]; then
    "${CC:-cc}" "$@" -x c -o /dev/null "$tmpdir/probe.c" >/dev/null 2>&1
elif [ "$mode" = "libdl" ]; then
    "${CC:-cc}" "$@" -x c -o /dev/null "$tmpdir/probe.c" -ldl >/dev/null 2>&1
else
    echo "unknown mode: $mode" >&2
    exit 2
fi
