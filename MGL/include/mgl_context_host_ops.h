/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_context_host_ops.h — the C ABI between an out-of-tree windowing host
 * (the GLFW fork) and MGL's platform renderer.
 *
 * The host used to drive MGLRenderer through objc_msgSend.  Because ObjC
 * message dispatch resolves at run time, deleting a selector on the MGL side
 * (mglSetSwapInterval:, 2026-09-13) could only ever fail as an
 * "unrecognized selector" crash inside the consumer — the two artifacts are
 * built independently, so the boundary must be verifiable at compile time
 * and degrade gracefully at load time.  This ops table is that boundary:
 *
 *   - A removed or renamed entry is a compile-time error for a rebuilt
 *     consumer and a NULL function pointer for a stale one.
 *   - `version` + `size` let a stale consumer detect a newer MGL and skip
 *     entries instead of calling through garbage.
 *   - No MGL selector is ever sent by name; the host holds opaque handles.
 *
 * Pure C, no Objective-C and no GL header dependency: includable from any
 * TU without pulling glcorearb.h.
 *
 * Implemented in MGL/src/mgl_platform_shell.cpp; design rationale in
 * docs/GLFW_MGL_INVOCATION_PLAN.md §3.3.
 */

#ifndef MGL_CONTEXT_HOST_OPS_H
#define MGL_CONTEXT_HOST_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MGL_CONTEXT_HOST_OPS_VERSION 1u

typedef struct MGLContextHostOps {
    /* version must stay the first field; size second. */
    uint32_t version; /* MGL_CONTEXT_HOST_OPS_VERSION */
    uint32_t size;    /* sizeof(MGLContextHostOps) */

    /* Create the platform renderer and bind it to `glm_ctx` and the
     * consumer's view (`view` is an opaque NSView handle).  Returns an
     * opaque owner handle holding one reference, or NULL on failure.
     * Release it with release_owner; never free/CFRelease it directly. */
    void *(*create_and_bind)(void *glm_ctx, void *view);

    /* Non-zero once the renderer finished Metal device/queue/layer setup
     * (the same predicate as the renderer's mglRendererIsReady). */
    int (*renderer_is_ready)(void *owner);

    /* glfwSwapInterval backend: clamps negatives to 0, records the value,
     * and drives the CAMetalLayer's displaySyncEnabled. */
    void (*set_swap_interval)(void *owner, int interval);

    /* Release the owner handle (balances create_and_bind's +1). */
    void (*release_owner)(void *owner);
} MGLContextHostOps;

/* Stable per-process table; the returned pointer never changes.  Consumers
 * fetch it once per context and validate version/size before use. */
const MGLContextHostOps *mglContextHostOps(void);

#ifdef __cplusplus
}
#endif

#endif /* MGL_CONTEXT_HOST_OPS_H */
