/*
 * mgl_metal_ref.h
 * MGL
 *
 * Metal Object Lifecycle Subsystem.
 *
 * Unified release helpers for Metal objects stored as `void *` (ARC-bridged)
 * in C structs (Texture::mtl_data, Buffer::data.mtl_data, Shader::mtl_data.*,
 * Sampler::mtl_data, Program::mtl_data, MGLShaderModule::mtl_function / mtl_library,
 * Sync::mtl_event / mtl_command_buffer, etc.).
 *
 * Problem: the codebase had three parallel release paths:
 *   1. renderer callback deletion from C files
 *   2. CFBridgingRelease(ptr); ptr = NULL;        — MGLRenderer.m (24 sites)
 *   3. CFRelease(ptr); ptr = NULL;                — program.c MGLShaderModule + 3-way
 *                                                   if/else fallbacks (9 sites)
 * All three ultimately do the same CFBridgingRelease.  The function-pointer
 * indirection in path 1 exists only because C files cannot message ObjC
 * objects directly; internally it just calls CFBridgingRelease.
 *
 * Solution: a single `static inline` helper that takes `void **slot`, calls
 * CFBridgingRelease, and nulls the slot.  This:
 *   - Eliminates the forgotten-`= NULL` class of bugs (22 sites were missing it)
 *   - Removes the dead `else CFRelease` fallback branches (3 sites)
 *   - Unifies program.c's raw CFRelease on MGLShaderModule fields (4 sites)
 *   - Is a pure C function (no ObjC messaging), so it works in both .c and .m TUs
 *
 * Scope: this header ONLY covers the generic `void *` slot pattern.
 * Sync objects keep their own `mtlReleaseSync` path (needs @try/@catch for
 * legacy MTLSharedEvent robustness). The renderer itself is retained by the
 * backend operation context and released during backend destruction.
 *
 * Dependencies: CoreFoundation (CFBridgingRelease) + stddef.h (NULL).
 * No Metal framework dependency — works in pure C TUs.
 */

#ifndef MGL_METAL_REF_H
#define MGL_METAL_REF_H

#include <stddef.h>

/* CoreFoundation is needed for CFBridgingRelease / CFAutorelease.
 * On Apple platforms this is always available as a system framework.
 * In ARC mode, CFBridgingRelease transfers ownership from CF to ARC. */
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>

#ifdef __OBJC__
#include <objc/runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* === Metal object lifecycle counters ===
 *
 * Cross-TU accounting used by the MGL PERF summary to expose net-live
 * object growth (created minus released) per Metal object kind. */

typedef enum MGLMetalKind {
    MGLMetalKindBuffer   = 0, /* MTLBuffer */
    MGLMetalKindTexture  = 1, /* MTLTexture */
    MGLMetalKindSampler  = 2, /* MTLSamplerState */
    MGLMetalKindLibrary  = 3, /* MTLLibrary */
    MGLMetalKindFunction = 4, /* MTLFunction */
    MGLMetalKindPSO      = 5, /* MTLRenderPipelineState / MTLComputePipelineState */
    MGLMetalKindOther    = 6,
    MGLMetalKindCount    = 7
} MGLMetalKind;

void mglMetalCountCreate(int kind);
void mglMetalCountRelease(int kind);
uint64_t mglMetalGetCreated(int kind);
uint64_t mglMetalGetReleased(int kind);

/* Classify a bridged Metal object by its runtime class name.  The concrete
 * classes are private (MTLIOAccelBuffer etc.), so match stable suffixes on
 * object_getClassName.  In pure C TUs (no ObjC runtime access) this can only
 * return MGLMetalKindOther; ObjC TUs get the precise class. */
static inline int mglMetalObjKindOf(void *obj)
{
    if (!obj) {
        return MGLMetalKindOther;
    }
#ifdef __OBJC__
    const char *cls = object_getClassName((__bridge id)obj);
    if (!cls) {
        return MGLMetalKindOther;
    }
    if (strstr(cls, "Buffer"))       return MGLMetalKindBuffer;
    if (strstr(cls, "Texture"))      return MGLMetalKindTexture;
    if (strstr(cls, "Sampler"))      return MGLMetalKindSampler;
    if (strstr(cls, "Library"))      return MGLMetalKindLibrary;
    if (strstr(cls, "Function"))     return MGLMetalKindFunction;
    if (strstr(cls, "Pipeline"))     return MGLMetalKindPSO;
    return MGLMetalKindOther;
#else
    return MGLMetalKindOther;
#endif
}

/* === Core release helper ===
 *
 * Releases a Metal object stored as `void *` in a C struct slot and nulls
 * the slot.  Safe to call with NULL slot or NULL *slot (no-op).
 *
 * `slot` is a pointer to the void* field (e.g. &tex->mtl_data).
 * The object must have been retained via CFBridgingRetain or
 * __bridge_retained.  Uses CFBridgingRelease for correct ARC bridge
 * semantics (NOT raw CFRelease, which would confuse ARC's bookkeeping).
 *
 * After this call, *slot is NULL. */
static inline void mglSafeReleaseMetalObj(void **slot)
{
    if (slot && *slot) {
        mglMetalCountRelease(mglMetalObjKindOf(*slot));
#ifdef __OBJC__
        CFBridgingRelease(*slot);
#else
        CFRelease(*slot);
#endif
        *slot = NULL;
    }
}

/* === Convenience: release + don't-null (for destructor paths) ===
 *
 * Same as mglSafeReleaseMetalObj but does NOT null the slot.  Use only
 * when the owning struct is about to be free()'d and the slot value will
 * never be read again.  Prefer mglSafeReleaseMetalObj in all other cases. */
static inline void mglReleaseMetalObjNoNull(void *obj)
{
    if (obj) {
        mglMetalCountRelease(mglMetalObjKindOf(obj));
#ifdef __OBJC__
        CFBridgingRelease(obj);
#else
        CFRelease(obj);
#endif
    }
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_METAL_REF_H */
