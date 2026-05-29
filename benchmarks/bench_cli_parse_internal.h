#ifndef BENCH_CLI_PARSE_INTERNAL_H
#define BENCH_CLI_PARSE_INTERNAL_H

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    int value;
} sparse_cli_choice_t;

static inline int sparse_cli_parse_long_arg(const char *tool, const char *flag, const char *s,
                                            long min_value, long max_value, long *out) {
    if (!s || !*s) {
        fprintf(stderr, "%s: %s requires a value\n", tool, flag);
        return 0;
    }

    errno = 0;
    char *end = NULL;
    long value = strtol(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0') {
        fprintf(stderr, "%s: %s: not a valid integer: '%s'\n", tool, flag, s);
        return 0;
    }
    if (value < min_value || value > max_value) {
        fprintf(stderr, "%s: %s: out of range [%ld, %ld]: %ld\n", tool, flag, min_value,
                max_value, value);
        return 0;
    }

    *out = value;
    return 1;
}

static inline int sparse_cli_parse_int_arg(const char *tool, const char *flag, const char *s,
                                           int min_value, int max_value, int *out) {
    long value = 0;
    if (!sparse_cli_parse_long_arg(tool, flag, s, (long)min_value, (long)max_value, &value))
        return 0;
    *out = (int)value;
    return 1;
}

static inline int sparse_cli_parse_finite_double_arg(const char *tool, const char *flag,
                                                     const char *s, double min_value,
                                                     double *out) {
    if (!s || !*s) {
        fprintf(stderr, "%s: %s requires a value\n", tool, flag);
        return 0;
    }

    errno = 0;
    char *end = NULL;
    double value = strtod(s, &end);
    if (errno == ERANGE || end == s || *end != '\0' || !isfinite(value)) {
        fprintf(stderr, "%s: %s: not a valid finite number: '%s'\n", tool, flag, s);
        return 0;
    }
    if (value < min_value) {
        fprintf(stderr, "%s: %s: must be >= %g: %g\n", tool, flag, min_value, value);
        return 0;
    }

    *out = value;
    return 1;
}

static inline int sparse_cli_parse_choice_arg(const char *tool, const char *flag, const char *s,
                                              const sparse_cli_choice_t *choices, size_t count,
                                              int *out, const char *accepted_desc) {
    if (!s || !*s) {
        fprintf(stderr, "%s: %s requires a value\n", tool, flag);
        return 0;
    }

    for (size_t i = 0; i < count; i++) {
        if (!strcmp(s, choices[i].name)) {
            *out = choices[i].value;
            return 1;
        }
    }

    if (accepted_desc && *accepted_desc) {
        fprintf(stderr, "%s: unknown %s '%s' (%s)\n", tool, flag, s, accepted_desc);
    } else {
        fprintf(stderr, "%s: unknown %s '%s'\n", tool, flag, s);
    }
    return 0;
}

#endif
