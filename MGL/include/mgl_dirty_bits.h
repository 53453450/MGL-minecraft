/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holder.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_dirty_bits.h — the authoritative GLMState dirty-bit masks.
 *
 * Single source (STATE_MACHINE_REVIEW 4.2 / M0): the batch-restore plan used
 * to hand-copy this enum into mgl_batch_restore.c ("keep in sync" by
 * convention, no compile-time binding), so renumbering a bit in the
 * authoritative header silently drifted the restore plan.  Both sides now
 * include this header:
 *
 *   - mgl_types_state.h  (renderer-facing consumers, GLMState.dirty_bits)
 *   - mgl_batch_restore.c (pure-C plan TU, linkable without GL headers)
 *
 * Deliberately dependency-free: no GL types, no other MGL headers, so the
 * plan TU can include it standalone (test_batch_restore).
 */

#ifndef MGL_DIRTY_BITS_H
#define MGL_DIRTY_BITS_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    dirtyVAO = 0,
    dirtyState,
    dirtyBuffer,
    dirtyTexture,
    dirtyTexParam,
    dirtyTexBinding,
    dirtySampler,
    dirtyShader,
    dirtyProgram,
    dirtyFBO,
    dirtyDrawable,
    dirtyRenderState,
    dirtyAlphaState,
    dirtyImageUnit,
    dirtyBufferBase,
    maxDirtyState,
    dirtyAllBit = 31
};

#define DIRTY_VAO       (0x1 << dirtyVAO)
#define DIRTY_STATE     (0x1 << dirtyState)
#define DIRTY_BUFFER    (0x1 << dirtyBuffer)
#define DIRTY_TEX       (0x1 << dirtyTexture)
#define DIRTY_TEX_PARAM   (0x1 << dirtyTexParam)
#define DIRTY_TEX_BINDING (0x1 << dirtyTexBinding)
#define DIRTY_SAMPLER (0x1 << dirtySampler)
#define DIRTY_SHADER    (0x1 << dirtyShader)
#define DIRTY_PROGRAM   (0x1 << dirtyProgram)
#define DIRTY_FBO       (0x1 << dirtyFBO)
/* DIRTY_DRAWABLE: written by createGLMContext sRGB path (glm_context.c).
 * DIRTY_SHADER: also used on Shader_t.dirty_bits (shaders.c); GLMState restore
 * plans do not fold either bit (STATE_DATAFLOW T8 — keep, do not delete). */
#define DIRTY_DRAWABLE      (0x1 << dirtyDrawable)
#define DIRTY_RENDER_STATE  (0x1 << dirtyRenderState)
#define DIRTY_ALPHA_STATE   (0x1 << dirtyAlphaState)
#define DIRTY_IMAGE_UNIT_STATE   (0x1 << dirtyImageUnit)
#define DIRTY_BUFFER_BASE_STATE   (0x1 << dirtyBufferBase)
#define DIRTY_ALL_BIT   ((unsigned)0x1 << dirtyAllBit)    // so we know the dirty all was set.
#define DIRTY_ALL       (0xFFFFFFFF)

/* Pin the numeric contract the batch-restore plan depends on: these bits are
 * folded into replay dirty masks and their values are part of the plan's
 * tested behavior (test_batch_restore exercises the domain masks).  A renumber
 * must be a deliberate, reviewed change — not an accidental enum insertion. */
_Static_assert(DIRTY_VAO == 0x0001u && DIRTY_BUFFER == 0x0004u &&
                   DIRTY_TEX == 0x0008u && DIRTY_PROGRAM == 0x0100u &&
                   DIRTY_RENDER_STATE == 0x0800u && DIRTY_ALPHA_STATE == 0x1000u &&
                   DIRTY_IMAGE_UNIT_STATE == 0x2000u &&
                   DIRTY_BUFFER_BASE_STATE == 0x4000u,
               "dirty-bit numbering changed; the batch restore plan and its "
               "tests assume these values");

#ifdef __cplusplus
}
#endif

#endif /* MGL_DIRTY_BITS_H */
