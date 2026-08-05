/*
 * Thread-affinity contract for the GL calling thread.
 *
 * MGL Metal-layer state is owned exclusively by one thread at a time (the GL
 * calling thread).  This facility records who that thread is and turns the
 * "must be single-threaded" contract into a checkable assertion in DEBUG /
 * MGL_ENABLE_THREAD_CHECKS builds.  Compiled out entirely in Release.
 */
#ifndef MGL_THREAD_AFFINITY_H
#define MGL_THREAD_AFFINITY_H

#include <stdint.h>
#include <stdbool.h>

#if defined(DEBUG) || defined(MGL_ENABLE_THREAD_CHECKS)

extern _Atomic uint64_t g_mglGLThreadID;   /* 0 = no thread has claimed yet */

void mglClaimGLThread(void);                       /* first GL entry, idempotent */
void mglAssertGLThreadImpl(const char *fn, const char *file, int line);

#define MGL_ASSERT_GL_THREAD() \
    mglAssertGLThreadImpl(__func__, __FILE__, __LINE__)

#else
#define mglClaimGLThread()      ((void)0)
#define MGL_ASSERT_GL_THREAD()  ((void)0)
#endif /* DEBUG || MGL_ENABLE_THREAD_CHECKS */

#endif /* MGL_THREAD_AFFINITY_H */