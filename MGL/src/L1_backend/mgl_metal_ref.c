/*
 * mgl_metal_ref.c
 * MGL
 *
 * Shared cross-TU counters for Metal object lifecycle accounting.
 * Created counts are bumped at the main allocation sites
 * (bindMTLBufferLocked, createMTLTextureFromGLTexture,
 * createMTLSamplerForTexParam, MSL library/function creation, PSO creation);
 * released counts are bumped inside mglSafeReleaseMetalObj (and its no-null
 * variant), which covers every C-storage release path automatically.
 * mglPrintPerfSummary diffs the two to expose net-live object growth.
 */

#include "mgl_metal_ref.h"

#include <stdatomic.h>

_Atomic uint64_t g_mgl_metal_created[MGLMetalKindCount];
_Atomic uint64_t g_mgl_metal_released[MGLMetalKindCount];

void mglMetalCountCreate(int kind)
{
    if (kind < 0 || kind >= MGLMetalKindCount) {
        kind = MGLMetalKindOther;
    }
    atomic_fetch_add_explicit(&g_mgl_metal_created[kind], 1, memory_order_relaxed);
}

void mglMetalCountRelease(int kind)
{
    if (kind < 0 || kind >= MGLMetalKindCount) {
        kind = MGLMetalKindOther;
    }
    atomic_fetch_add_explicit(&g_mgl_metal_released[kind], 1, memory_order_relaxed);
}

uint64_t mglMetalGetCreated(int kind)
{
    if (kind < 0 || kind >= MGLMetalKindCount) {
        return 0;
    }
    return atomic_load_explicit(&g_mgl_metal_created[kind], memory_order_relaxed);
}

uint64_t mglMetalGetReleased(int kind)
{
    if (kind < 0 || kind >= MGLMetalKindCount) {
        return 0;
    }
    return atomic_load_explicit(&g_mgl_metal_released[kind], memory_order_relaxed);
}
