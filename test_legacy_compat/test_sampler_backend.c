/* SPDX-License-Identifier: LGPL-3.0-only */

#include "mgl_sampler_backend.h"

#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;
static int fail_calloc;

void *mglTestCalloc(size_t count, size_t size)
{
    return fail_calloc ? NULL : calloc(count, size);
}

#define CHECK(condition, message)                  \
    do {                                           \
        if (!(condition)) {                        \
            fprintf(stderr, "FAIL: %s\n", message); \
            failures++;                            \
        }                                          \
    } while (0)

static CFTypeRef make_state(char value)
{
    CFMutableStringRef state = CFStringCreateMutable(NULL, 0);
    CFStringAppendCString(state, (char[]){value, '\0'}, kCFStringEncodingUTF8);
    return state;
}

static CFTypeRef observe(CFTypeRef state)
{
    CFRetain(state);
    return state;
}

int main(void)
{
    mglSamplerBackendResetState(NULL);
    mglSamplerBackendRelease(NULL);
    CHECK(mglSamplerBackendState(NULL) == NULL, "NULL sampler has no state");

    Sampler sampler = {0};
    CFTypeRef first = observe(make_state('a'));
    mglSamplerBackendSetState(&sampler, (void *)first);
    CHECK(sampler.backend != NULL, "first state allocates backend");
    CHECK(mglSamplerBackendState(&sampler) == first, "first state is retained");
    CHECK(CFGetRetainCount(first) == 2, "backend owns the handed-over reference");

    CFTypeRef second = observe(make_state('b'));
    mglSamplerBackendSetState(&sampler, (void *)second);
    CHECK(CFGetRetainCount(first) == 1, "replacement releases previous state");
    CHECK(CFGetRetainCount(second) == 2, "replacement owns new state");
    mglSamplerBackendResetState(&sampler);
    CHECK(mglSamplerBackendState(&sampler) == NULL, "reset clears state");
    CHECK(sampler.backend != NULL, "reset keeps backend allocation");
    CHECK(CFGetRetainCount(second) == 1, "reset releases state reference");
    CFRelease(first);
    CFRelease(second);

    CFTypeRef third = observe(make_state('c'));
    Sampler failed = {0};
    fail_calloc = 1;
    mglSamplerBackendSetState(&failed, (void *)third);
    fail_calloc = 0;
    CHECK(failed.backend == NULL, "failed allocation leaves backend absent");
    CHECK(CFGetRetainCount(third) == 1,
          "failed allocation releases the handed-over state");
    CFRelease(third);

    CFTypeRef fourth = observe(make_state('d'));
    mglSamplerBackendSetState(&sampler, (void *)fourth);
    mglSamplerBackendRelease(&sampler);
    CHECK(sampler.backend == NULL, "release clears backend ownership");
    CHECK(mglSamplerBackendState(&sampler) == NULL,
          "released sampler has no state");
    CHECK(CFGetRetainCount(fourth) == 1, "release drops state reference");
    CFRelease(fourth);

    if (failures) {
        fprintf(stderr, "test_sampler_backend: %d failures\n", failures);
        return 1;
    }
    printf("test_sampler_backend: ok\n");
    return 0;
}
