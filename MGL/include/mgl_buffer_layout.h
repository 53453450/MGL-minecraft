/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_buffer_layout.h -- where a shader stage's buffer resources land in the
 * Metal buffer argument table (pure C, no Metal, no GL object).
 *
 * Both sides of the shader ABI need the same numbers: the AIR backend declares
 * each buffer with a location_index in the metallib, and reflection tells the
 * renderer which slot to bind it to.  They used to compute the layout
 * separately ("must match" comments in both places).  This is the one
 * computation; each side counts its resources and asks for the bases.
 *
 * Layout of a stage, from slot 0 (TES starts at 1; the tessellation contract
 * buffer owns slot 0):
 *
 *     [plain-uniform pack] [vertex attributes (VS)] [SSBOs] [UBOs] [atomic counters]
 *
 * Every instance of a block array takes its own slot.
 */

#ifndef MGL_BUFFER_LAYOUT_H
#define MGL_BUFFER_LAYOUT_H

#include <stdint.h>

#include "mgl_shader_abi.h" /* MGL_STAGE_* */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLBufferSlotCounts {
    int stage;                       /* MGL_STAGE_* */
    uint32_t attr_slots;             /* VS: locations taken by stage inputs */
    int has_plain_pack;              /* plain uniforms packed into one buffer */
    uint32_t ssbo_slots;             /* SSBO instances (array elements count) */
    uint32_t ubo_slots;              /* UBO instances */
    uint32_t atomic_counter_slots;
    /* Highest slot atomic counters may reach (MAX_BINDABLE_BUFFERS). */
    uint32_t atomic_counter_limit;
} MGLBufferSlotCounts;

typedef struct MGLBufferSlotLayout {
    uint32_t plain_base;    /* slot of the plain-uniform pack, when there is one */
    uint32_t plain_slots;   /* 0 or 1 */
    uint32_t ssbo_base;
    uint32_t ubo_base;
    uint32_t atomic_counter_base;
    uint32_t end;           /* first slot after the atomic counters */
} MGLBufferSlotLayout;

enum {
    MGL_BUFFER_LAYOUT_OK = 0,
    MGL_BUFFER_LAYOUT_BAD_ARGS = -1,
    MGL_BUFFER_LAYOUT_OVERFLOW = -2,          /* a slot number exceeds uint32 */
    MGL_BUFFER_LAYOUT_ATOMIC_EXCEEDS = -3     /* atomic counters pass the limit */
};

/* Fill `out` for the stage's resource counts.  Returns MGL_BUFFER_LAYOUT_OK or
 * the reason the layout cannot be formed (`out` is then zeroed). */
int mglBufferSlotLayoutBuild(const MGLBufferSlotCounts *in,
                             MGLBufferSlotLayout *out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BUFFER_LAYOUT_H */
