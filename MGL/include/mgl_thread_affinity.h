/*
 * Thread-affinity contract for the GL calling thread.
 *
 * MGL Metal-layer state is owned exclusively by one thread at a time (the GL
 * calling thread).  This facility records who that thread is and turns the
 * "must be single-threaded" contract into a checkable assertion.
 *
 * AUDIT-061: keep the check in Release as well — the load + pthread_threadid
 * cost is tiny, and affinity violations otherwise surface as silent UAF.
 * Opt out only with MGL_DISABLE_THREAD_CHECKS (tests / intentional bypass).
 */
#ifndef MGL_THREAD_AFFINITY_H
#define MGL_THREAD_AFFINITY_H

#include <stdint.h>
#include <stdbool.h>

#if !defined(MGL_DISABLE_THREAD_CHECKS)

#ifdef __cplusplus
extern "C" {
#endif

extern _Atomic uint64_t g_mglGLThreadID;   /* 0 = no thread has claimed yet */

void mglClaimGLThread(void);                       /* first GL entry, idempotent */
void mglAssertGLThreadImpl(const char *fn, const char *file, int line);

#ifdef __cplusplus
}
#endif

#define MGL_ASSERT_GL_THREAD() \
    mglAssertGLThreadImpl(__func__, __FILE__, __LINE__)

#else
#define mglClaimGLThread()      ((void)0)
#define MGL_ASSERT_GL_THREAD()  ((void)0)
#endif /* !MGL_DISABLE_THREAD_CHECKS */

#endif /* MGL_THREAD_AFFINITY_H */
