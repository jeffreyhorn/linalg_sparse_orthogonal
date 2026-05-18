/* _POSIX_C_SOURCE 200809L: needed for `setenv` / `unsetenv` on POSIX
 * because this self-check exercises the new opt-in test wrappers. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 200809L
#endif

#include "sparse_types.h"
#include "test_framework.h"

static int executed_default = 0;
static int executed_skip_body = 0;
static int executed_slow = 0;
static int executed_experimental = 0;

static void test_run_test_executes_normally(void) {
    executed_default++;
    ASSERT_EQ(executed_default, 1);
}

static void test_skip_macro_marks_current_test_skipped(void) {
    executed_skip_body++;
    SKIP_TEST("intentional self-check skip");
}

static void test_skip_accounting_records_body_skip(void) {
    ASSERT_EQ(executed_skip_body, 1);
    ASSERT_EQ(tf_tests_skipped, 1);
}

static void test_slow_wrapper_body(void) {
    executed_slow++;
    ASSERT_EQ(executed_slow, 1);
}

static void test_experimental_wrapper_body(void) {
    executed_experimental++;
    ASSERT_EQ(executed_experimental, 1);
}

static void test_default_optin_wrappers_skip_when_disabled(void) {
    ASSERT_EQ(executed_slow, 0);
    ASSERT_EQ(executed_experimental, 0);
    ASSERT_EQ(tf_tests_skipped, 3);
}

static void test_slow_wrapper_runs_when_enabled(void) {
    ASSERT_EQ(executed_slow, 1);
    ASSERT_EQ(tf_tests_skipped, 3);
}

static void test_experimental_wrapper_runs_when_enabled(void) {
    ASSERT_EQ(executed_experimental, 1);
    ASSERT_EQ(tf_tests_skipped, 3);
}

int main(void) {
    TEST_SUITE_BEGIN("test_framework_optin");

    RUN_TEST(test_run_test_executes_normally);
    RUN_TEST(test_skip_macro_marks_current_test_skipped);
    RUN_TEST(test_skip_accounting_records_body_skip);

    RUN_TEST_SLOW(test_slow_wrapper_body);
    RUN_TEST_EXPERIMENTAL(test_experimental_wrapper_body);
    RUN_TEST(test_default_optin_wrappers_skip_when_disabled);

    if (tf_setenv("SPARSE_TEST_SLOW", "1") != 0) {
        fprintf(stderr, "failed to enable SPARSE_TEST_SLOW\n");
        return 1;
    }
    RUN_TEST_SLOW(test_slow_wrapper_body);
    tf_unsetenv("SPARSE_TEST_SLOW");
    RUN_TEST(test_slow_wrapper_runs_when_enabled);

    if (tf_setenv("SPARSE_TEST_EXPERIMENTAL", "1") != 0) {
        fprintf(stderr, "failed to enable SPARSE_TEST_EXPERIMENTAL\n");
        return 1;
    }
    RUN_TEST_EXPERIMENTAL(test_experimental_wrapper_body);
    tf_unsetenv("SPARSE_TEST_EXPERIMENTAL");
    RUN_TEST(test_experimental_wrapper_runs_when_enabled);

    TEST_SUITE_END();
}
