#ifndef TEST_THREAD_HELPERS_H
#define TEST_THREAD_HELPERS_H

typedef void *(*test_thread_fn)(void *);

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>

typedef struct {
    HANDLE handle;
    test_thread_fn fn;
    void *arg;
    void *result;
} test_thread_t;

static unsigned __stdcall test_thread_entry(void *arg) {
    test_thread_t *thread = (test_thread_t *)arg;
    thread->result = thread->fn(thread->arg);
    return 0;
}

static int test_thread_create(test_thread_t *thread, test_thread_fn fn, void *arg) {
    if (!thread || !fn)
        return 1;

    thread->fn = fn;
    thread->arg = arg;
    thread->result = NULL;
    thread->handle = (HANDLE)_beginthreadex(NULL, 0, test_thread_entry, thread, 0, NULL);
    return thread->handle ? 0 : 1;
}

static int test_thread_join(test_thread_t *thread) {
    int rc = 0;

    if (!thread || !thread->handle)
        return 1;

    if (WaitForSingleObject(thread->handle, INFINITE) != WAIT_OBJECT_0)
        rc = 1;
    if (!CloseHandle(thread->handle))
        rc = 1;
    thread->handle = NULL;
    return rc;
}

#else
#include <pthread.h>

typedef struct {
    pthread_t handle;
} test_thread_t;

static int test_thread_create(test_thread_t *thread, test_thread_fn fn, void *arg) {
    if (!thread || !fn)
        return 1;

    return pthread_create(&thread->handle, NULL, fn, arg);
}

static int test_thread_join(test_thread_t *thread) {
    if (!thread)
        return 1;

    return pthread_join(thread->handle, NULL);
}
#endif

#endif /* TEST_THREAD_HELPERS_H */
