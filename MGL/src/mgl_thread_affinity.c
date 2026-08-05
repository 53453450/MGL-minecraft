#include "mgl_thread_affinity.h"

#if defined(DEBUG) || defined(MGL_ENABLE_THREAD_CHECKS)
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

_Atomic uint64_t g_mglGLThreadID = 0;

static uint64_t mglCurrentThreadID(void)
{
    uint64_t tid = 0;
    pthread_threadid_np(NULL, &tid);
    return tid;
}

void mglClaimGLThread(void)
{
    uint64_t expected = 0;
    uint64_t self = mglCurrentThreadID();
    /* First arrival claims; later calls are idempotent. */
    atomic_compare_exchange_strong(&g_mglGLThreadID, &expected, self);
}

void mglAssertGLThreadImpl(const char *fn, const char *file, int line)
{
    uint64_t owner = atomic_load_explicit(&g_mglGLThreadID, memory_order_relaxed);
    uint64_t self  = mglCurrentThreadID();
    if (owner != 0 && owner != self) {
        fprintf(stderr,
                "MGL FATAL: thread affinity violation in %s (%s:%d)\n"
                "  GL thread = %llu, calling thread = %llu\n",
                fn, file, line,
                (unsigned long long)owner, (unsigned long long)self);
        abort();    /* fail at the first violation, not in later heap corruption */
    }
}
#endif /* DEBUG || MGL_ENABLE_THREAD_CHECKS */