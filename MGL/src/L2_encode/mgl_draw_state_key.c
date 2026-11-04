/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Split from draw_command.c (Phase 7 soft god-TU target).
 */
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <stdatomic.h>

#include "glm_context.h"
#include "draw_command.h"
#include "mgl_byte_hash.h"
#include "mgl_safety.h"
#include "mgl_frame_activity.h"
#include "mgl_trace_log.h"
#include "mgl_sampler_compat.h"
#include "mgl_batch_hazard.h"
#include "mgl_env_flag.h"
#include "mgl_render_api_buffer.h"
#include "mgl_buffer_backend.h"
#include "mgl_draw_command_shared.h"

extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern void mglSparseScrubBoundImages(GLMContext ctx);

static uint64_t mglTexturePointerNameBits(const Texture *tex)
{
    if (!tex)
        return 0u;
    return (uint64_t)(uintptr_t)tex ^ ((uint64_t)tex->name << 1) ^
           (tex->identity_generation << 17);
}

static uint64_t mglSamplerPointerNameBits(const Sampler *sampler)
{
    if (!sampler)
        return 0u;
    return (uint64_t)(uintptr_t)sampler ^ ((uint64_t)sampler->name << 1) ^
           (sampler->identity_generation << 17);
}

static uint64_t mglComputeTextureHash(GLMContext ctx)
{
    uint64_t hash = 0;
    unsigned *mask = ctx->active_state->active_texture_mask;
    for (int w = 0; w < 4; w++) {
        unsigned bits = mask[w];
        while (bits) {
            int i = __builtin_ctz(bits);
            bits &= bits - 1;
            int unit = w * 32 + i;
            if (unit < TEXTURE_UNITS) {
                Texture *tex = ctx->active_state->active_textures[unit];
                hash ^= mglRotateLeft64(mglTexturePointerNameBits(tex), unit & 63);

                TextureUnit *tex_unit = &ctx->active_state->texture_units[unit];
                for (int t = 0; t < _MAX_TEXTURE_TYPES; t++) {
                    Texture *typed_tex = tex_unit->textures[t];
                    hash ^= mglRotateLeft64(
                        mglTexturePointerNameBits(typed_tex) + (uint64_t)(t + 1),
                        (unit + t) & 63);
                }

                Sampler *sampler = ctx->active_state->texture_samplers[unit];
                hash ^= mglRotateLeft64(mglSamplerPointerNameBits(sampler),
                                        (unit + 17) & 63);
            }
        }
    }

    /* BindImageTexture layer/level/access change Metal storage views but not
     * sampled texture bindings. Hash image units so batch same-key restore
     * skip and merge cannot keep a stale Type2D slice across rebinds. */
    GLuint max_iu = ctx->active_state->var.max_image_units;
    if (max_iu > TEXTURE_UNITS)
        max_iu = TEXTURE_UNITS;
    for (GLuint i = 0; i < max_iu; i++) {
        const ImageUnit *iu = &ctx->active_state->image_units[i];
        /* iu->texture is already the GL name; keep the same mix as textures. */
        uint64_t tex_bits = iu->tex
            ? mglTexturePointerNameBits(iu->tex)
            : ((uint64_t)iu->texture << 1);
        hash ^= mglRotateLeft64(tex_bits, (i + 41u) & 63);
        hash ^= mglRotateLeft64(((uint64_t)iu->level << 32) |
                                ((uint64_t)(iu->layered ? 1u : 0u) << 31) |
                                ((uint64_t)(uint32_t)iu->layer),
                                (i + 53u) & 63);
        hash ^= mglRotateLeft64((uint64_t)iu->access ^
                                ((uint64_t)iu->internalformat << 16),
                                (i + 59u) & 63);
    }
    return hash;
}

static void mglHashBufferBaseBinding(uint64_t *hash,
                                     const BufferBaseTarget *binding,
                                     uint64_t salt)
{
    if (!hash || !binding) return;

    uint64_t binding_ptr = binding->buf ? (uint64_t)(uintptr_t)binding->buf : 0;
    *hash ^= mglRotateLeft64(binding_ptr ^ ((uint64_t)binding->buffer << 1), salt & 63);
    *hash ^= mglRotateLeft64((uint64_t)binding->offset, (salt + 17u) & 63);
    *hash ^= mglRotateLeft64((uint64_t)binding->size, (salt + 37u) & 63);
}

/* Hash one buffer_base target.  Empty slots are fully zeroed at
 * unbind/delete time, so their contribution is 0 and iterating only
 * active_mask bits is bit-identical to scanning all 84 slots (the
 * DEBUG sampled assert below verifies this). */
static void mglHashBufferBaseTarget(uint64_t *hash,
                                    const BufferBase *base,
                                    int target,
                                    bool use_mask)
{
    if (use_mask) {
        for (unsigned w = 0; w < MGL_BUFFER_BASE_ACTIVE_WORDS; w++) {
            uint64_t mask = base->active_mask[w];
            while (mask) {
                int i = (int)(w * 64u) + __builtin_ctzll(mask);
                mask &= mask - 1;
                mglHashBufferBaseBinding(hash,
                                         &base->buffers[i],
                                         ((uint64_t)(target + 1) * 131u) + (uint64_t)i);
            }
        }
    } else {
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            mglHashBufferBaseBinding(hash,
                                     &base->buffers[i],
                                     ((uint64_t)(target + 1) * 131u) + (uint64_t)i);
        }
    }
}

static uint64_t mglComputeDrawBufferBindingHashScan(GLMContext ctx, bool use_mask)
{
    uint64_t hash = 0;

    /* Keep this list aligned with the buffer_base slots captured by
     * mglCopyHotStateFields for deferred batch replay. */
    const int draw_buffer_targets[] = {
        _UNIFORM_BUFFER,
        _UNIFORM_CONSTANT,
        _SHADER_STORAGE_BUFFER,
        _ATOMIC_COUNTER_BUFFER,
        _TRANSFORM_FEEDBACK_BUFFER
    };

    for (size_t t = 0; t < sizeof(draw_buffer_targets) / sizeof(draw_buffer_targets[0]); t++) {
        int target = draw_buffer_targets[t];
        mglHashBufferBaseTarget(&hash,
                                &ctx->active_state->buffer_base[target],
                                target,
                                use_mask);
    }

    Program *program = ctx->active_state->program;
    if (program) {
        /* plain_uniform_active_mask skips empty slots (buf==NULL contributes
         * 0 to the hash, so skipping them is bit-identical to the linear
         * scan).  Typical MC draws have a handful of plain uniforms, so
         * this turns ~84 empty-slot iterations into ~5-10 active ones. */
#ifdef DEBUG
        uint64_t hash_before_plain = hash;
        /* DEBUG path: linear scan + assert that mask-scan produces the same
         * incremental contribution.  Run both to detect mask drift. */
        for (int i = 0; i < MAX_PLAIN_UNIFORM_LOCATIONS; i++) {
            mglHashBufferBaseBinding(&hash,
                                     &program->plain_uniform_buffers[i],
                                     0x700u + (uint64_t)i);
        }
        static uint32_t s_plainMaskCheck = 0;
        if ((++s_plainMaskCheck & 0xFFu) == 0u) {
            uint64_t verify = hash_before_plain;
            for (int w = 0; w < (int)MGL_PLAIN_UNIFORM_ACTIVE_WORDS; w++) {
                uint64_t bits = program->plain_uniform_active_mask[w];
                while (bits) {
                    int i = (int)((w * 64) + __builtin_ctzll(bits));
                    bits &= bits - 1;
                    mglHashBufferBaseBinding(&verify,
                                             &program->plain_uniform_buffers[i],
                                             0x700u + (uint64_t)i);
                }
            }
            assert(verify == hash);
        }
#else
        for (int w = 0; w < (int)MGL_PLAIN_UNIFORM_ACTIVE_WORDS; w++) {
            uint64_t bits = program->plain_uniform_active_mask[w];
            while (bits) {
                int i = (int)((w * 64) + __builtin_ctzll(bits));
                bits &= bits - 1;
                mglHashBufferBaseBinding(&hash,
                                         &program->plain_uniform_buffers[i],
                                         0x700u + (uint64_t)i);
            }
        }
#endif
    }

    return hash;
}

static uint64_t mglComputeDrawBufferBindingHash(GLMContext ctx)
{
    uint64_t hash = mglComputeDrawBufferBindingHashScan(ctx, true);
#ifdef DEBUG
    static uint32_t s_maskCheck = 0;
    if ((++s_maskCheck & 0xFFu) == 0u) {
        assert(hash == mglComputeDrawBufferBindingHashScan(ctx, false));
    }
#endif
    return hash;
}

static uint64_t mglComputeUniformBufferBindingHash(GLMContext ctx)
{
    uint64_t hash = 0;
    mglHashBufferBaseTarget(&hash,
                            &ctx->active_state->buffer_base[_UNIFORM_BUFFER],
                            _UNIFORM_BUFFER,
                            true);
    return hash;
}

static void mglHashBufferPointerName(uint64_t *hash, const Buffer *buffer, uint64_t salt)
{
    if (!hash) return;

    uint64_t ptr = buffer ? (uint64_t)(uintptr_t)buffer : 0u;
    uint64_t name = buffer ? (uint64_t)buffer->name : 0u;
    *hash ^= mglRotateLeft64(ptr ^ (name << 1), salt & 63);
}

static void mglHashCurrentVertexAttrib(uint64_t *hash,
                                       const CurrentVertexAttrib *current,
                                       uint64_t salt)
{
    if (!hash || !current) return;

    *hash ^= mglHashBytes64(current->f, sizeof(current->f), 0x63766166u + salt);
    *hash ^= mglHashBytes64(current->i, sizeof(current->i), 0x63766169u + salt);
    *hash ^= mglHashBytes64(current->u, sizeof(current->u), 0x63766175u + salt);
    *hash ^= mglHashBytes64(current->d, sizeof(current->d), 0x63766164u + salt);
    *hash ^= mglRotateLeft64((uint64_t)current->type, (salt + 1u) & 63);
    *hash ^= mglRotateLeft64((uint64_t)current->integer, (salt + 2u) & 63);
    *hash ^= mglRotateLeft64((uint64_t)current->long_attribute, (salt + 3u) & 63);
}

static void mglHashElementArrayState(uint64_t *hash, const VertexArray *vao)
{
    if (!hash || !vao) return;

    mglHashBufferPointerName(hash, vao->element_array.buffer, 0x900u);
    *hash ^= mglRotateLeft64((uint64_t)vao->element_array.type, 11);
    *hash ^= mglRotateLeft64((uint64_t)vao->element_array.size, 13);
    *hash ^= mglRotateLeft64((uint64_t)(uintptr_t)vao->element_array.ptr, 17);
}

static uint64_t mglComputeVertexArrayStateHashVAO(const VertexArray *vao,
                                                  const CurrentVertexAttrib *current_attribs,
                                                  bool uses_elements,
                                                  bool include_element)
{
    if (!vao) return 0u;

    uint64_t hash = 0x56414f5354415445ULL;
    hash ^= mglRotateLeft64((uint64_t)vao->name, 3);
    hash ^= mglRotateLeft64((uint64_t)vao->enabled_attribs, 7);

    GLuint maxAttribs = MAX_ATTRIBS;
    for (GLuint i = 0; i < maxAttribs; i++) {
        const VertexAttrib *attrib = &vao->attrib[i];
        uint64_t salt = 0x100u + (uint64_t)i * 17u;
        if (include_element)
            mglHashBufferPointerName(&hash, attrib->buffer, salt);
        hash ^= mglRotateLeft64((uint64_t)attrib->size, (salt + 1u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->type, (salt + 2u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->normalized, (salt + 3u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->integer, (salt + 4u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->long_attribute, (salt + 5u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->stride, (salt + 6u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->divisor, (salt + 7u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->relativeoffset, (salt + 8u) & 63);
        if (include_element)
            hash ^= mglRotateLeft64((uint64_t)attrib->binding_offset,
                                    (salt + 9u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->buffer_bindingindex, (salt + 10u) & 63);
        if ((vao->enabled_attribs & (1u << i)) == 0u && current_attribs) {
            mglHashCurrentVertexAttrib(&hash, &current_attribs[i],
                                       0x300u + (uint64_t)i * 23u);
        }
    }

    for (GLuint i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++) {
        const BufferBinding *binding = &vao->bindings[i];
        uint64_t salt = 0x500u + (uint64_t)i * 19u;
        if (include_element)
            mglHashBufferPointerName(&hash, binding->buffer, salt);
        if (include_element)
            hash ^= mglRotateLeft64((uint64_t)binding->offset, (salt + 1u) & 63);
        hash ^= mglRotateLeft64((uint64_t)binding->stride, (salt + 2u) & 63);
        hash ^= mglRotateLeft64((uint64_t)binding->divisor, (salt + 3u) & 63);
    }

    if (uses_elements && include_element)
        mglHashElementArrayState(&hash, vao);

    return hash;
}

static uint64_t mglComputeVertexArrayStateHash(GLMContext ctx, bool uses_elements)
{
    return mglComputeVertexArrayStateHashVAO(
        ctx ? ctx->active_state->vao : NULL,
        ctx ? ctx->active_state->current_vertex_attrib : NULL,
        uses_elements, true);
}

static uint64_t mglComputeRenderStateHash(GLMContext ctx)
{
    uint64_t hash = mglHashBytes64(&ctx->active_state->caps,
                                   sizeof(ctx->active_state->caps),
                                   0xfeedfacecafebeefULL);
    GLMParams *var = &ctx->active_state->var;

    hash ^= mglHashBytes64(ctx->active_state->viewport, sizeof(ctx->active_state->viewport), 0x101u);
    hash ^= mglRotateLeft64((uint64_t)ctx->active_state->draw_buffer, 7);
    hash ^= mglRotateLeft64((uint64_t)ctx->active_state->draw_buffer_count, 13);
    hash ^= mglHashBytes64(ctx->active_state->draw_buffers, sizeof(ctx->active_state->draw_buffers), 0x102u);

    hash ^= mglHashBytes64(&var->point_size, sizeof(var->point_size), 0x201u);
    hash ^= mglHashBytes64(&var->line_width, sizeof(var->line_width), 0x202u);
    hash ^= mglRotateLeft64((uint64_t)var->polygon_mode, 5);
    hash ^= mglRotateLeft64((uint64_t)var->cull_face_mode, 11);
    hash ^= mglRotateLeft64((uint64_t)var->front_face, 17);
    hash ^= mglHashBytes64(var->depth_range, sizeof(var->depth_range), 0x203u);
    hash ^= var->depth_writemask ? 0xAAAA5555ULL : 0ULL;
    hash ^= mglRotateLeft64((uint64_t)var->depth_func, 8);

    hash ^= mglRotateLeft64((uint64_t)var->logic_op, 9);
    hash ^= mglRotateLeft64((uint64_t)var->logic_op_mode, 10);
    hash ^= mglHashBytes64(var->color_writemask, sizeof(var->color_writemask), 0x204u);
    hash ^= mglHashBytes64(var->blend_color, sizeof(var->blend_color), 0x205u);
    hash ^= mglHashBytes64(var->blend_dst_rgb, sizeof(var->blend_dst_rgb), 0x206u);
    hash ^= mglHashBytes64(var->blend_src_rgb, sizeof(var->blend_src_rgb), 0x207u);
    hash ^= mglHashBytes64(var->blend_dst_alpha, sizeof(var->blend_dst_alpha), 0x208u);
    hash ^= mglHashBytes64(var->blend_src_alpha, sizeof(var->blend_src_alpha), 0x209u);
    hash ^= mglHashBytes64(var->blend_equation_rgb, sizeof(var->blend_equation_rgb), 0x20au);
    hash ^= mglHashBytes64(var->blend_equation_alpha, sizeof(var->blend_equation_alpha), 0x20bu);

    hash ^= mglRotateLeft64((uint64_t)var->stencil_func, 24);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_value_mask, 25);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_fail, 26);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_pass_depth_fail, 27);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_pass_depth_pass, 28);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_ref, 29);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_writemask, 30);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_func, 31);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_fail, 32);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_pass_depth_fail, 33);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_pass_depth_pass, 34);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_ref, 35);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_value_mask, 36);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_writemask, 37);

    hash ^= mglHashBytes64(&var->polygon_offset_units, sizeof(var->polygon_offset_units), 0x20cu);
    hash ^= mglHashBytes64(&var->polygon_offset_factor, sizeof(var->polygon_offset_factor), 0x20du);
    hash ^= mglHashBytes64(&var->sample_coverage_value, sizeof(var->sample_coverage_value), 0x20eu);
    hash ^= mglHashBytes64(&var->polygon_offset_clamp, sizeof(var->polygon_offset_clamp), 0x20fu);
    hash ^= mglRotateLeft64((uint64_t)var->sample_coverage_invert, 38);
    /* Sample-mask and sample-shading state.  Both live in GLMParams and both
     * change what a draw produces (the mask gates which samples the fragment
     * coverage enables; min_sample_shading drives the emulated per-sample draw
     * loop), so two draws that differ only in glSampleMaski /
     * glMinSampleShading must not be merged into one batch entry.  The
     * SAMPLE_MASK *enable* is in GLMCaps and is already covered by the caps
     * hash above; these are the two fields that were missing. */
    hash ^= mglRotateLeft64((uint64_t)var->sample_mask_value, 44);
    hash ^= mglHashBytes64(&var->min_sample_shading,
                           sizeof(var->min_sample_shading), 0x210u);
    hash ^= mglRotateLeft64((uint64_t)var->primitive_restart_index, 39);
    hash ^= mglRotateLeft64((uint64_t)var->clip_origin, 40);
    hash ^= mglRotateLeft64((uint64_t)var->clip_depth_mode, 41);
    hash ^= mglRotateLeft64((uint64_t)var->provoking_vertex, 42);
    hash ^= mglRotateLeft64((uint64_t)var->point_sprite_coord_origin, 43);

    return hash;
}

static uint8_t mglModeToPrimitiveType(GLenum mode)
{
    switch (mode) {
        case GL_POINTS:         return 0;  /* MTLPrimitiveTypePoint */
        case GL_LINES:          return 1;  /* MTLPrimitiveTypeLine */
        case GL_LINE_STRIP:     return 2;  /* MTLPrimitiveTypeLineStrip */
        case GL_TRIANGLES:      return 3;  /* MTLPrimitiveTypeTriangle */
        case GL_TRIANGLE_STRIP: return 4;  /* MTLPrimitiveTypeTriangleStrip */
        /* Metal has no native support for these modes; they are emulated
         * in the draw dispatch path (MGLRenderer.m). Return 0xFF so the
         * batch system routes them through the per-draw emulation path
         * instead of multi-draw-indirect. */
        case GL_TRIANGLE_FAN:
        case GL_LINE_LOOP:
        case GL_PATCHES:
        case GL_QUADS:
        case GL_LINES_ADJACENCY:
        case GL_LINE_STRIP_ADJACENCY:
        case GL_TRIANGLES_ADJACENCY:
        case GL_TRIANGLE_STRIP_ADJACENCY:
            return 0xFF;
        default:
            return 0xFF;
    }
}

static uint16_t mglComputeCapsFlags(GLMContext ctx)
{
    uint16_t flags = 0;
    GLMCaps *caps = &ctx->active_state->caps;

    if (caps->cull_face)        flags |= (1u << 0);
    if (caps->depth_test)       flags |= (1u << 1);
    if (caps->stencil_test)     flags |= (1u << 2);
    if (caps->blend)            flags |= (1u << 3);
    if (caps->scissor_test)     flags |= (1u << 4);
    if (caps->polygon_offset_fill)  flags |= (1u << 5);
    if (caps->polygon_offset_line)  flags |= (1u << 6);
    if (caps->polygon_offset_point) flags |= (1u << 7);
    if (ctx->active_state->var.cull_face_mode == GL_FRONT_AND_BACK) flags |= (1u << 8);

    return flags;
}

#if defined(DEBUG) || defined(MGL_DEBUG)
/* G6 (STATE_DATAFLOW_TODO §7.3): when the dirty-hash cache path skips a
 * recompute, recompute anyway and abort on mismatch.  Default ON in debug
 * builds; set MGL_HASH_CACHE_ORACLE=0 to disable. */
static int mglHashCacheOracleEnabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("MGL_HASH_CACHE_ORACLE");
        if (!env || env[0] == '\0') {
            cached = 1;
        } else {
            cached = mgl_env_flag_value_enabled(env);
        }
    }
    return cached;
}
#endif

void mglComputeStateKey(GLMContext ctx, GLenum mode, bool uses_elements, MGLStateKey *out)
{
    MGL_SIGNPOST_BEGIN(ComputeStateKey);
    if (!ctx || !out) {
        MGL_SIGNPOST_END(ComputeStateKey);
        return;
    }

    /* Every other field is written unconditionally below.  These are only
     * written conditionally (program names when a pipeline is bound without a
     * direct program; scissor[] only when scissor is enabled; _padding is
     * never written), so zero them explicitly to keep memcmp-based equality
     * correct without a full-struct memset.  _padding is the struct's only
     * unnamed-value gap (draw_command.h); leaving it as stack residue made
     * identical states compare unequal at random (STATE_MACHINE_REVIEW 4.1). */
    out->vertex_program_name = 0;
    out->fragment_program_name = 0;
    out->_padding = 0;
    out->scissor[0] = 0;
    out->scissor[1] = 0;
    out->scissor[2] = 0;
    out->scissor[3] = 0;

    out->program_name = ctx->active_state->program_name;
    out->program_pipeline_name = ctx->active_state->var.program_pipeline_binding;
    if (out->program_pipeline_name == 0 && ctx->active_state->program_pipeline) {
        out->program_pipeline_name = ctx->active_state->program_pipeline->name;
    }
    if (ctx->active_state->program_name == 0 && out->program_pipeline_name != 0) {
        ProgramPipeline *pipeline = ctx->active_state->program_pipeline;
        if (!pipeline || pipeline->name != out->program_pipeline_name) {
            pipeline = (ProgramPipeline *)searchHashTable(&ctx->active_state->program_pipeline_table,
                                                          out->program_pipeline_name);
        }
        out->vertex_program_name = pipeline && pipeline->stage_programs[_VERTEX_SHADER]
            ? pipeline->stage_programs[_VERTEX_SHADER]->name
            : 0u;
        out->fragment_program_name = pipeline && pipeline->stage_programs[_FRAGMENT_SHADER]
            ? pipeline->stage_programs[_FRAGMENT_SHADER]->name
            : 0u;
    }
    out->vao_name = ctx->active_state->vao ? ctx->active_state->vao->name : 0;
    out->fbo_name = ctx->active_state->framebuffer ? ctx->active_state->framebuffer->name : 0;

    for (int i = 0; i < 4; i++) {
        out->viewport[i] = (int32_t)ctx->active_state->viewport[i];
    }

    out->scissor_enabled = ctx->active_state->caps.scissor_test ? 1 : 0;
    if (out->scissor_enabled) {
        for (int i = 0; i < 4; i++) {
            /* Full GL scissor range — do not narrow to int16 (A03). */
            out->scissor[i] = (int32_t)ctx->active_state->var.scissor_box[i];
        }
    }

    out->primitive_type = mglModeToPrimitiveType(mode);
    out->caps_flags = mglComputeCapsFlags(ctx);

    /* Dirty-Flag Hash Optimization
     * Only recompute hashes when dirty flags are set (texture bindings changed,
     * VAO changed, render state changed). This reduces redundant hash computation
     * from ~1200 → ~360 per frame when state is stable across draw calls.
     *
     * NOTE: cached hashes live in GLMState (ctx->active_state), NOT in the
     * MGLStateKey output struct.  mglStateKeysEqual() uses memcmp on the full
     * MGLStateKey, so the key must contain only per-draw-computed values —
     * cached hash fields would be stale garbage there and break batch-merge
     * comparisons.  (The struct's _padding is written by the fixed-zero list
     * above, so memcmp stays exact.)
     *
     * NOTE: `mode` is a per-draw parameter, not persistent GL state, so it is
     * NOT part of the cached render-state hash.  It is XORed in on every call
     * so that different draw modes produce different keys. */
    static int dirty_hash_enabled = -1;
    if (dirty_hash_enabled < 0) {
        const char *env = getenv("MGL_OPT_DIRTY_HASH");
        dirty_hash_enabled = (env == NULL || atoi(env) != 0) ? 1 : 0;
    }

    if (dirty_hash_enabled) {
        /* Legacy dirty bits remain set until Metal consumes them. Hash-cache
         * invalidation is tracked separately so stable draws do not recompute
         * the same hashes while renderer state is still pending.
         * G5 note: *_dirty latches can stay set after
         * mglClearStateDirtyBitsPreservingHashInvalidation clears dirty_bits,
         * so "recompute ⇒ dirty_bits contains domain" is not a valid assert. */
        if (ctx->active_state->texture_dirty) {
            ctx->active_state->cached_texture_hash = mglComputeTextureHash(ctx);
            ctx->active_state->texture_dirty = 0;
            MGL_PERF_INC(g_mglHashRecomputeTextureSinceSwap);
        }
#if defined(DEBUG) || defined(MGL_DEBUG)
        else if (mglHashCacheOracleEnabled()) {
            uint64_t full = mglComputeTextureHash(ctx);
            if (full != ctx->active_state->cached_texture_hash) {
                fprintf(stderr,
                        "MGL Error: hash-cache oracle mismatch (texture) "
                        "cached=%016llx full=%016llx\n",
                        (unsigned long long)ctx->active_state->cached_texture_hash,
                        (unsigned long long)full);
                abort();
            }
        }
#endif
        out->texture_hash = ctx->active_state->cached_texture_hash;

        if (ctx->active_state->vertex_layout_dirty) {
            /* Cache only the draw-arrays-independent portion.  Element-array
             * state is added below for indexed draws so changing draw kind
             * cannot reuse a hash computed with the opposite uses_elements
             * value. */
            ctx->active_state->cached_vertex_layout_hash =
                mglComputeVertexArrayStateHash(ctx, false);
            ctx->active_state->vertex_layout_dirty = 0;
            MGL_PERF_INC(g_mglHashRecomputeVertexSinceSwap);
        }
#if defined(DEBUG) || defined(MGL_DEBUG)
        else if (mglHashCacheOracleEnabled()) {
            uint64_t full = mglComputeVertexArrayStateHash(ctx, false);
            if (full != ctx->active_state->cached_vertex_layout_hash) {
                fprintf(stderr,
                        "MGL Error: hash-cache oracle mismatch (vertex) "
                        "cached=%016llx full=%016llx\n",
                        (unsigned long long)ctx->active_state->cached_vertex_layout_hash,
                        (unsigned long long)full);
                abort();
            }
        }
#endif
        out->vertex_layout_hash = ctx->active_state->cached_vertex_layout_hash;
        if (uses_elements)
            mglHashElementArrayState(&out->vertex_layout_hash, ctx->active_state->vao);

        if (ctx->active_state->render_state_dirty) {
            ctx->active_state->cached_render_state_hash =
                mglComputeRenderStateHash(ctx) ^
                mglComputeDrawBufferBindingHash(ctx);
            ctx->active_state->render_state_dirty = 0;
            MGL_PERF_INC(g_mglHashRecomputeRenderSinceSwap);
        }
#if defined(DEBUG) || defined(MGL_DEBUG)
        else if (mglHashCacheOracleEnabled()) {
            uint64_t full = mglComputeRenderStateHash(ctx) ^
                            mglComputeDrawBufferBindingHash(ctx);
            if (full != ctx->active_state->cached_render_state_hash) {
                fprintf(stderr,
                        "MGL Error: hash-cache oracle mismatch (render) "
                        "cached=%016llx full=%016llx\n",
                        (unsigned long long)ctx->active_state->cached_render_state_hash,
                        (unsigned long long)full);
                abort();
            }
        }
#endif
        out->render_state_hash = ctx->active_state->cached_render_state_hash ^
                                 mglRotateLeft64((uint64_t)mode, 21);

        /* uniform_buffer_hash is cached like the other hash domains instead
         * of being recomputed every draw.  Its inputs (buffer_base[_UNIFORM_
         * BUFFER]) are a strict subset of render_state_hash inputs, so it
         * shares the same dirty bits (DIRTY_BUFFER_BASE_STATE |
         * DIRTY_PROGRAM).  Kept as a separate field because
         * mglStateKeysEqualIgnoringUniformRanges XORs it back out of
         * render_state_hash during batch merge comparisons. */
        if (ctx->active_state->uniform_buffer_dirty) {
            ctx->active_state->cached_uniform_buffer_hash =
                mglComputeUniformBufferBindingHash(ctx);
            ctx->active_state->uniform_buffer_dirty = 0;
            MGL_PERF_INC(g_mglHashRecomputeUniformSinceSwap);
        }
#if defined(DEBUG) || defined(MGL_DEBUG)
        else if (mglHashCacheOracleEnabled()) {
            uint64_t full = mglComputeUniformBufferBindingHash(ctx);
            if (full != ctx->active_state->cached_uniform_buffer_hash) {
                fprintf(stderr,
                        "MGL Error: hash-cache oracle mismatch (uniform) "
                        "cached=%016llx full=%016llx\n",
                        (unsigned long long)ctx->active_state->cached_uniform_buffer_hash,
                        (unsigned long long)full);
                abort();
            }
        }
#endif
        out->uniform_buffer_hash = ctx->active_state->cached_uniform_buffer_hash;
    } else {
        /* Legacy path: unconditional hash computation */
        out->texture_hash = mglComputeTextureHash(ctx);
        out->vertex_layout_hash = mglComputeVertexArrayStateHash(ctx, uses_elements);
        out->render_state_hash = mglComputeRenderStateHash(ctx) ^
                                 mglComputeDrawBufferBindingHash(ctx) ^
                                 mglRotateLeft64((uint64_t)mode, 21);
        out->uniform_buffer_hash = mglComputeUniformBufferBindingHash(ctx);
    }

    MGL_SIGNPOST_END(ComputeStateKey);
}

bool mglStateKeysEqual(const MGLStateKey *a, const MGLStateKey *b)
{
    if (!a || !b) return false;
    return memcmp(a, b, sizeof(MGLStateKey)) == 0;
}

bool mglStateKeysEqualIgnoringUniformRanges(const MGLStateKey *a,
                                                   const MGLStateKey *b)
{
    if (!a || !b) return false;

    /* render_state_hash includes the UBO hash for legacy key compatibility.
     * XOR it back out before comparing the non-UBO render/buffer state. */
    if ((a->render_state_hash ^ a->uniform_buffer_hash) !=
        (b->render_state_hash ^ b->uniform_buffer_hash)) {
        return false;
    }

    /* Compare fields directly instead of copying both keys to the stack,
     * zeroing the two range-hash fields, and running a full-struct memcmp.
     * The range hashes are intentionally excluded from this comparison. */
    return a->program_name == b->program_name &&
           a->program_pipeline_name == b->program_pipeline_name &&
           a->vertex_program_name == b->vertex_program_name &&
           a->fragment_program_name == b->fragment_program_name &&
           a->vao_name == b->vao_name &&
           a->fbo_name == b->fbo_name &&
           a->viewport[0] == b->viewport[0] &&
           a->viewport[1] == b->viewport[1] &&
           a->viewport[2] == b->viewport[2] &&
           a->viewport[3] == b->viewport[3] &&
           a->scissor[0] == b->scissor[0] &&
           a->scissor[1] == b->scissor[1] &&
           a->scissor[2] == b->scissor[2] &&
           a->scissor[3] == b->scissor[3] &&
           a->scissor_enabled == b->scissor_enabled &&
           a->primitive_type == b->primitive_type &&
           a->caps_flags == b->caps_flags &&
           a->texture_hash == b->texture_hash &&
           a->vertex_layout_hash == b->vertex_layout_hash;
}

/* Merge tolerance for per-section data movement (MC 1.21.11 terrain: one
 * GpuBuffer pair + one UBO slice + one EBO per section).  Compares every
 * key field except the UBO-range and vertex-data-location hashes; the two
 * VAOS' layout hashes are compared with data locations excluded.  The direct
 * issue path resolves each command's own index buffer, and the merged
 * command records its own dynamic uniform/vertex bindings, so the replay
 * never applies a neighbour's data location. */

/* Data-location movement between the snapshot and the live VAO is only
 * natively rebindable per command when the moved attribs are NVVB-managed
 * (their binding-table entry carries the buffer).  Classic attrib->buffer
 * movement would push the dynamic apply path into a whole-buffer format
 * conversion per command (mglDynApplyMapperFallback), which is far slower
 * than not merging — so classic-style movement must not be merged. */
static bool mglVertexDataMovementIsNativelyRebindable(
    const VertexArray *vao_a, const VertexArray *vao_b)
{
    if (!vao_a || !vao_b) {
        return false;
    }
    GLuint enabled = (GLuint)(vao_a->enabled_attribs | vao_b->enabled_attribs);
    while (enabled) {
        int i = __builtin_ctz(enabled);
        enabled &= enabled - 1;
        const VertexAttrib *aa = &vao_a->attrib[i];
        const VertexAttrib *ab = &vao_b->attrib[i];
        if (aa->buffer == ab->buffer) {
            continue;
        }
        GLuint bi = aa->buffer_bindingindex;
        if (bi != ab->buffer_bindingindex ||
            bi >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
            return false;
        }
        if (vao_a->bindings[bi].buffer == NULL ||
            vao_b->bindings[bi].buffer == NULL) {
            /* Classic pointer attrib: the location moved without the
             * bindings table carrying it. */
            return false;
        }
    }
    return true;
}

bool mglStateKeysEqualIgnoringUniformAndElementRanges(
    const MGLStateKey *a, const MGLStateKey *b,
    const VertexArray *vao_a, const VertexArray *vao_b,
    const CurrentVertexAttrib *current_attribs_a,
    const CurrentVertexAttrib *current_attribs_b, bool uses_elements)
{
    if ((a->render_state_hash ^ a->uniform_buffer_hash) !=
        (b->render_state_hash ^ b->uniform_buffer_hash)) {
        return false;
    }
    if (a->program_name != b->program_name ||
        a->program_pipeline_name != b->program_pipeline_name ||
        a->vertex_program_name != b->vertex_program_name ||
        a->fragment_program_name != b->fragment_program_name ||
        a->vao_name != b->vao_name ||
        a->fbo_name != b->fbo_name ||
        a->viewport[0] != b->viewport[0] ||
        a->viewport[1] != b->viewport[1] ||
        a->viewport[2] != b->viewport[2] ||
        a->viewport[3] != b->viewport[3] ||
        a->scissor[0] != b->scissor[0] ||
        a->scissor[1] != b->scissor[1] ||
        a->scissor[2] != b->scissor[2] ||
        a->scissor[3] != b->scissor[3] ||
        a->scissor_enabled != b->scissor_enabled ||
        a->primitive_type != b->primitive_type ||
        a->caps_flags != b->caps_flags ||
        a->texture_hash != b->texture_hash) {
        return false;
    }
    if (a->vertex_layout_hash == b->vertex_layout_hash) {
        return true;
    }
    if (!vao_a || !vao_b) {
        return false;
    }
    if (!mglVertexDataMovementIsNativelyRebindable(vao_a, vao_b)) {
        return false;
    }
    uint64_t hash_a = mglComputeVertexArrayStateHashVAO(vao_a, current_attribs_a,
                                                        uses_elements, false);
    uint64_t hash_b = mglComputeVertexArrayStateHashVAO(vao_b, current_attribs_b,
                                                        uses_elements, false);
    return hash_a == hash_b;
}

/* Texture-tolerant variant: like the element variant, but the texture-hash
 * field may also differ — every merged command records its own dynamic
 * texture binds (mglCaptureDynamicTextureBindings, which fails closed on
 * sampler/active-mask changes), so the replay applies each draw's binds. */
bool mglStateKeysEqualIgnoringUniformElementAndTextures(
    const MGLStateKey *a, const MGLStateKey *b,
    const VertexArray *vao_a, const VertexArray *vao_b,
    const CurrentVertexAttrib *current_attribs_a,
    const CurrentVertexAttrib *current_attribs_b, bool uses_elements)
{
    if ((a->render_state_hash ^ a->uniform_buffer_hash) !=
        (b->render_state_hash ^ b->uniform_buffer_hash)) {
        return false;
    }
    if (a->program_name != b->program_name ||
        a->program_pipeline_name != b->program_pipeline_name ||
        a->vertex_program_name != b->vertex_program_name ||
        a->fragment_program_name != b->fragment_program_name ||
        a->vao_name != b->vao_name ||
        a->fbo_name != b->fbo_name ||
        a->viewport[0] != b->viewport[0] ||
        a->viewport[1] != b->viewport[1] ||
        a->viewport[2] != b->viewport[2] ||
        a->viewport[3] != b->viewport[3] ||
        a->scissor[0] != b->scissor[0] ||
        a->scissor[1] != b->scissor[1] ||
        a->scissor[2] != b->scissor[2] ||
        a->scissor[3] != b->scissor[3] ||
        a->scissor_enabled != b->scissor_enabled ||
        a->primitive_type != b->primitive_type ||
        a->caps_flags != b->caps_flags) {
        return false;
    }
    if (a->vertex_layout_hash == b->vertex_layout_hash &&
        a->texture_hash == b->texture_hash) {
        return true;
    }
    if (!vao_a || !vao_b) {
        return false;
    }
    if (!mglVertexDataMovementIsNativelyRebindable(vao_a, vao_b)) {
        return false;
    }
    uint64_t hash_a = mglComputeVertexArrayStateHashVAO(vao_a, current_attribs_a,
                                                        uses_elements, false);
    uint64_t hash_b = mglComputeVertexArrayStateHashVAO(vao_b, current_attribs_b,
                                                        uses_elements, false);
    return hash_a == hash_b;
}

bool mglCaptureDynamicUniformRanges(GLMContext ctx,
                                           const MGLDrawBatch *batch,
                                           MGLDrawCommand *cmd,
                                           bool capture_all_bound_ranges)
{
    if (!ctx || !batch || !batch->state_snapshot || !cmd) return false;

    const GLMState *snapshot = (const GLMState *)batch->state_snapshot;
    const BufferBase *base_ubo = &snapshot->buffer_base[_UNIFORM_BUFFER];
    const BufferBase *current_ubo = &ctx->active_state->buffer_base[_UNIFORM_BUFFER];

    cmd->dynamic_uniform_binding_count = 0;

    /* The active_mask bitmap mirrors buf != NULL (maintained by
     * mglBufferBaseSetActive / ClearActive at bind/unbind/delete time), so:
     *   - Bits set only in `current` (added) or only in `base` (removed)
     *     mean an object was bound/unbound since snapshot — fall back to
     *     the conservative hazard/flush path by returning false.
     *   - Bits set in both are the only slots we need to inspect for
     *     offset/size overrides.
     *   - Bits set in neither are skipped: both buf are NULL, and the
     *     bind/unbind contract keeps buffer names 0 when buf is NULL. */
    uint64_t base_mask[2]  = { base_ubo->active_mask[0],  base_ubo->active_mask[1] };
    uint64_t curr_mask[2]  = { current_ubo->active_mask[0], current_ubo->active_mask[1] };

    /* Object additions or removals since snapshot => cannot capture
     * dynamic overrides, fall back to hazard path. */
    if ((base_mask[0] ^ curr_mask[0]) != 0u ||
        (base_mask[1] ^ curr_mask[1]) != 0u) {
        return false;
    }

    /* Early bound-count check (replaces the per-slot ++bound_count). */
    uint16_t bound_count = (uint16_t)(__builtin_popcountll(curr_mask[0]) +
                                      __builtin_popcountll(curr_mask[1]));
    if (bound_count > MGL_MAX_DYNAMIC_UNIFORM_BINDINGS) {
        return false;
    }

    /* Iterate only the active slots. */
    for (uint32_t w = 0; w < MGL_BUFFER_BASE_ACTIVE_WORDS; w++) {
        uint64_t bits = curr_mask[w];
        while (bits) {
            int i = __builtin_ctzll(bits);
            bits &= bits - 1;
            uint16_t slot = (uint16_t)((w << 6) + i);

            const BufferBaseTarget *base = &base_ubo->buffers[slot];
            const BufferBaseTarget *current = &current_ubo->buffers[slot];

            /* Both slots are active per the mask check above, but verify
             * the underlying pointers/names match (cheap pointer compare). */
            if (base->buffer != current->buffer || base->buf != current->buf) {
                return false;
            }
            if (!capture_all_bound_ranges &&
                base->offset == current->offset && base->size == current->size) {
                continue;
            }
            if (cmd->dynamic_uniform_binding_count >= MGL_MAX_DYNAMIC_UNIFORM_BINDINGS) {
                return false;
            }

            MGLDynamicUniformBinding *override =
                &cmd->dynamic_uniform_bindings[cmd->dynamic_uniform_binding_count++];
            override->binding_index = slot;
            override->offset = current->offset;
            override->size = current->size;
        }
    }
    return cmd->dynamic_uniform_binding_count > 0;
}

static const Buffer *mglVertexArrayEffectiveBuffer(const VertexArray *vao,
                                                   GLuint attrib_index)
{
    if (!vao || attrib_index >= MAX_ATTRIBS) return NULL;
    const VertexAttrib *attrib = &vao->attrib[attrib_index];
    if (attrib->buffer_bindingindex < MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        const BufferBinding *binding =
            &vao->bindings[attrib->buffer_bindingindex];
        if (binding->buffer) return binding->buffer;
    }
    return attrib->buffer;
}

static bool mglVertexAttribLayoutsEqual(const VertexAttrib *a,
                                        const VertexAttrib *b)
{
    /* Format + binding-index identity only. VERTEX_BINDING_OFFSET belongs to
     * BindVertexBuffer state (GL 4.6 Table 23.4) and is captured as a per-draw
     * absolute override — do not treat offset as layout incompatibility. */
    return a && b &&
           a->size == b->size &&
           a->type == b->type &&
           a->normalized == b->normalized &&
           a->integer == b->integer &&
           a->long_attribute == b->long_attribute &&
           a->stride == b->stride &&
           a->divisor == b->divisor &&
           a->relativeoffset == b->relativeoffset &&
           a->buffer_bindingindex == b->buffer_bindingindex;
}

static bool mglVertexArraysHaveCompatibleBindings(GLMContext ctx,
                                                  const GLMState *snapshot,
                                                  const VertexArray *base,
                                                  const VertexArray *current)
{
    if (!ctx || !snapshot || !base || !current ||
        base->enabled_attribs == 0u ||
        base->enabled_attribs != current->enabled_attribs) {
        return false;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if (!mglVertexAttribLayoutsEqual(&base->attrib[i], &current->attrib[i])) {
            return false;
        }
        if ((base->enabled_attribs & (1u << i)) == 0u &&
            memcmp(&snapshot->current_vertex_attrib[i],
                   &ctx->active_state->current_vertex_attrib[i],
                   sizeof(CurrentVertexAttrib)) != 0) {
            return false;
        }
    }

    for (GLuint i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++) {
        const BufferBinding *a = &base->bindings[i];
        const BufferBinding *b = &current->bindings[i];
        if (a->stride != b->stride || a->divisor != b->divisor) {
            return false;
        }
    }

    /* Metal slot assignment groups attributes that share one VBO.  Preserve
     * that equivalence relation even though the actual buffer objects differ
     * between base and current.
     *
     * Cache the effective buffer per attribute once, then give each attribute
     * a canonical partition label: the smallest enabled index sharing the same
     * buffer.  Two partitions are equal iff the canonical labels match
     * element-wise. */
    const Buffer *base_bufs[MAX_ATTRIBS];
    const Buffer *current_bufs[MAX_ATTRIBS];
    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        base_bufs[i] = NULL;
        current_bufs[i] = NULL;
        if ((base->enabled_attribs & (1u << i)) == 0u) continue;
        base_bufs[i] = mglVertexArrayEffectiveBuffer(base, i);
        current_bufs[i] = mglVertexArrayEffectiveBuffer(current, i);
        if (!base_bufs[i] || !current_bufs[i]) return false;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if ((base->enabled_attribs & (1u << i)) == 0u) continue;
        /* Find smallest enabled j < i with the same buffer on each side. */
        GLuint base_lbl = i;
        GLuint current_lbl = i;
        for (GLuint j = 0; j < i; j++) {
            if ((base->enabled_attribs & (1u << j)) == 0u) continue;
            if (base_lbl == i && base_bufs[j] == base_bufs[i]) {
                base_lbl = j;
            }
            if (current_lbl == i && current_bufs[j] == current_bufs[i]) {
                current_lbl = j;
            }
            if (base_lbl != i && current_lbl != i) break;
        }
        if (base_lbl != current_lbl) return false;
    }
    return true;
}

bool mglCaptureDynamicVertexBindings(GLMContext ctx,
                                            const MGLDrawBatch *batch,
                                            MGLDrawCommand *cmd)
{
    if (!ctx || !batch || !batch->vao_snapshot || !cmd ||
        !ctx->active_state->vao) {
        return false;
    }

    const VertexArray *base = (const VertexArray *)batch->vao_snapshot;
    const VertexArray *current = ctx->active_state->vao;
    const GLMState *snapshot = (const GLMState *)batch->state_snapshot;
    if (!mglVertexArraysHaveCompatibleBindings(ctx, snapshot, base, current)) {
        return false;
    }

    cmd->dynamic_vertex_binding_count = 0;
    uint16_t captured_mask = 0u;
    for (GLuint attrib_index = 0; attrib_index < MAX_ATTRIBS; attrib_index++) {
        if ((base->enabled_attribs & (1u << attrib_index)) == 0u) continue;

        const VertexAttrib *base_attrib = &base->attrib[attrib_index];
        const VertexAttrib *current_attrib = &current->attrib[attrib_index];
        GLuint binding_index = base_attrib->buffer_bindingindex;
        if (binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS ||
            binding_index != current_attrib->buffer_bindingindex) {
            return false;
        }
        uint16_t binding_bit = (uint16_t)(1u << binding_index);
        if ((captured_mask & binding_bit) != 0u) continue;

        const BufferBinding *base_binding = &base->bindings[binding_index];
        const BufferBinding *current_binding = &current->bindings[binding_index];
        const Buffer *base_buffer = base_binding->buffer
            ? base_binding->buffer : base_attrib->buffer;
        Buffer *current_buffer = current_binding->buffer
            ? current_binding->buffer : current_attrib->buffer;
        GLintptr base_offset = base_binding->buffer
            ? base_binding->offset : base_attrib->binding_offset;
        GLintptr current_offset = current_binding->buffer
            ? current_binding->offset : current_attrib->binding_offset;
        /* Absolute VERTEX_BINDING_OFFSET — VAO rotation may wrap to a smaller
         * slice than the batch base; that is still a valid rebind. */
        if (!base_buffer || !current_buffer ||
            current_offset < 0 || base_offset < 0) {
            return false;
        }
        if ((uint64_t)current_offset > UINT32_MAX ||
            cmd->dynamic_vertex_binding_count >= MGL_MAX_DYNAMIC_VERTEX_BINDINGS) {
            return false;
        }

        MGLDynamicVertexBinding *override =
            &cmd->dynamic_vertex_bindings[cmd->dynamic_vertex_binding_count++];
        override->buffer_name = current_buffer->name;
        override->offset = (uint32_t)current_offset;
        override->binding_index = (uint8_t)binding_index;
        memset(override->reserved, 0, sizeof(override->reserved));
        captured_mask |= binding_bit;
    }

    /* Attributes sharing one Metal slot must slide by the same signed delta
     * relative to the batch base VAO (absolute offsets themselves may differ). */
    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if ((base->enabled_attribs & (1u << i)) == 0u) continue;
        for (GLuint j = i + 1; j < MAX_ATTRIBS; j++) {
            if ((base->enabled_attribs & (1u << j)) == 0u ||
                mglVertexArrayEffectiveBuffer(base, i) !=
                    mglVertexArrayEffectiveBuffer(base, j)) {
                continue;
            }
            uint8_t bi = (uint8_t)base->attrib[i].buffer_bindingindex;
            uint8_t bj = (uint8_t)base->attrib[j].buffer_bindingindex;
            uint32_t abs_i = 0u, abs_j = 0u;
            GLintptr base_i = 0, base_j = 0;
            for (uint8_t k = 0; k < cmd->dynamic_vertex_binding_count; k++) {
                const MGLDynamicVertexBinding *binding =
                    &cmd->dynamic_vertex_bindings[k];
                if (binding->binding_index == bi) abs_i = binding->offset;
                if (binding->binding_index == bj) abs_j = binding->offset;
            }
            {
                const BufferBinding *bb = &base->bindings[bi];
                const BufferBinding *bjb = &base->bindings[bj];
                base_i = bb->buffer ? bb->offset : base->attrib[i].binding_offset;
                base_j = bjb->buffer ? bjb->offset : base->attrib[j].binding_offset;
            }
            int64_t delta_i = (int64_t)abs_i - (int64_t)base_i;
            int64_t delta_j = (int64_t)abs_j - (int64_t)base_j;
            if (delta_i != delta_j) {
                return false;
            }
        }
    }
    /* Indexed commands already carry an immutable element-buffer name.
     * Treat an EBO-only VAO hash change as a capturable per-draw binding so
     * A -> B -> A index-buffer switches can remain in one direct batch. */
    GLuint base_element_name = base->element_array.buffer
        ? base->element_array.buffer->name : 0u;
    bool element_binding_changed =
        cmd->element_buffer_name &&
        cmd->element_buffer_name != base_element_name;
    return cmd->dynamic_vertex_binding_count > 0 || element_binding_changed;
}

bool mglStateKeysEqualIgnoringDynamicBindings(const MGLStateKey *a,
                                                      const MGLStateKey *b)
{
    if (!a || !b ||
        (a->render_state_hash ^ a->uniform_buffer_hash) !=
        (b->render_state_hash ^ b->uniform_buffer_hash)) {
        return false;
    }

    MGLStateKey a_without_vao = *a;
    MGLStateKey b_without_vao = *b;
    a_without_vao.vao_name = 0;
    b_without_vao.vao_name = 0;
    a_without_vao.vertex_layout_hash = 0;
    b_without_vao.vertex_layout_hash = 0;
    a_without_vao.texture_hash = 0;
    b_without_vao.texture_hash = 0;
    a_without_vao.render_state_hash = 0;
    b_without_vao.render_state_hash = 0;
    a_without_vao.uniform_buffer_hash = 0;
    b_without_vao.uniform_buffer_hash = 0;
    return mglStateKeysEqual(&a_without_vao, &b_without_vao);
}

bool mglCaptureDynamicTextureBindings(GLMContext ctx,
                                             const MGLDrawBatch *batch,
                                             MGLDrawCommand *cmd)
{
    if (!ctx || !batch || !cmd || !batch->state_snapshot) return false;

    const GLMState *snapshot = (const GLMState *)batch->state_snapshot;
    cmd->dynamic_texture_binding_count = 0;

    /* Iterate only the units the current program/pipeline actually samples,
     * using the cached merged mask (bitmap scan).  Falls back to scanning
     * all 128 units only when no program/pipeline is bound (mask all-ones). */
    const uint32_t *sampled_mask = mglGetActiveSampledTextureUnitMask(ctx);
    if (!sampled_mask) {
        return cmd->dynamic_texture_binding_count > 0;
    }
    const uint32_t *cur_active_mask = ctx->active_state->active_texture_mask;
    const uint32_t *snap_active_mask = snapshot->active_texture_mask;

    for (int w = 0; w < 4; w++) {
        uint32_t bits = sampled_mask[w];
        while (bits) {
            int bit = __builtin_ctz(bits);
            bits &= bits - 1;
            GLuint unit = (GLuint)((w * 32) + bit);
            if (unit >= TEXTURE_UNITS) continue;

            bool snapshot_active =
                (snap_active_mask[unit / 32u] & (1u << (unit % 32u))) != 0u;
            bool current_active =
                (cur_active_mask[unit / 32u] & (1u << (unit % 32u))) != 0u;
            /* Pointer compare only: snapshot texture/sampler pointers may be
             * dangling after delete, so name cannot be read from them (T2).
             * Apply uses binding->texture_name from the live current object;
             * same-address reuse after free still needs generation (T1-2). */
            if (snapshot_active != current_active ||
                snapshot->texture_samplers[unit] !=
                    ctx->active_state->texture_samplers[unit]) {
                return false;
            }
            if (!current_active) continue;

            bool found_active = ctx->active_state->active_textures[unit] == NULL;
            for (GLuint target = 0; target < _MAX_TEXTURE_TYPES; target++) {
                Texture *base = snapshot->texture_units[unit].textures[target];
                Texture *current = ctx->active_state->texture_units[unit].textures[target];
                if ((base == NULL) != (current == NULL)) return false;
                if (!current) continue;
                if (current->index != target || (base && base->index != target) ||
                    current->target != base->target ||
                    cmd->dynamic_texture_binding_count >=
                        MGL_MAX_DYNAMIC_TEXTURE_BINDINGS) {
                    return false;
                }

                MGLDynamicTextureBinding *binding =
                    &cmd->dynamic_texture_bindings[
                        cmd->dynamic_texture_binding_count++];
                binding->unit = (uint8_t)unit;
                binding->target_index = (uint8_t)target;
                binding->is_active =
                    ctx->active_state->active_textures[unit] == current ? 1u : 0u;
                binding->texture_name = current->name;
                found_active = found_active || binding->is_active != 0u;
            }
            if (!found_active) return false;
        }
    }

    return cmd->dynamic_texture_binding_count > 0;
}

bool mglCaptureDynamicDrawBindings(GLMContext ctx,
                                          const MGLDrawBatch *batch,
                                          const MGLStateKey *key,
                                          MGLDrawCommand *cmd)
{
    if (!ctx || !batch || !key || !cmd || !batch->state_snapshot) {
        return false;
    }

    bool capture_vao = batch->has_dynamic_vertex_bindings ||
        key->vao_name != batch->key.vao_name ||
        key->vertex_layout_hash != batch->key.vertex_layout_hash;
    if (capture_vao) {
        if (!mglCaptureDynamicVertexBindings(ctx, batch, cmd)) return false;
    }

    bool capture_uniforms = batch->has_dynamic_uniform_bindings ||
        key->uniform_buffer_hash != batch->key.uniform_buffer_hash;
    if (capture_uniforms &&
        !mglCaptureDynamicUniformRanges(ctx, batch, cmd,
                                        batch->has_dynamic_uniform_bindings)) {
        return false;
    }
    bool capture_textures = batch->has_dynamic_texture_bindings ||
        key->texture_hash != batch->key.texture_hash;
    if (capture_textures &&
        !mglCaptureDynamicTextureBindings(ctx, batch, cmd)) {
        return false;
    }
    return true;
}

/* Called once at batch creation to compute the initial mdi_compatible
 * value.  Checks batch-level invariants (primitive type, element-usage
 * match) plus the first command's draw mode.  Batch-level invariants never
 * change after creation, so they must NOT be re-checked on each append. */
