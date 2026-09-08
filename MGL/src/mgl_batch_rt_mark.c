/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_batch_rt_mark.h"

#include <stdio.h>

int mgl_batch_rt_attachment_active(uint32_t bitfield, uint32_t index,
                                   uint32_t max_attachments)
{
    if (index >= max_attachments) {
        return 0;
    }
    return ((bitfield >> index) & 1u) != 0u;
}

int mgl_batch_rt_yflip_authority(int has_injected_yflip,
                                 int yflip_sampler_explicit,
                                 int has_in_sampler_named,
                                 int has_diffuse_sampler_named)
{
    if (!has_injected_yflip || !yflip_sampler_explicit) {
        return 0;
    }
    if (has_in_sampler_named || has_diffuse_sampler_named) {
        return 0;
    }
    return 1;
}

int mgl_batch_rt_should_trace_write_mark(uint64_t hit)
{
    return hit <= 128ull || (hit % 256ull) == 0ull;
}

int mgl_batch_rt_should_cross_mark(int already_marked, int attachment_active)
{
    return !already_marked && attachment_active;
}

int mgl_batch_rt_should_diag_attachment0(uint32_t attachment_index,
                                         int trace_enabled, int can_use_copy)
{
    return attachment_index == 0u && trace_enabled && can_use_copy;
}

void mgl_batch_trace_fs_slot_flags(const MGLBatchTraceFsSlot *slots, uint32_t n,
                                   int *has_rt_out, int *used_copy_out)
{
    int has_rt = 0;
    int used_copy = 0;
    if (slots) {
        for (uint32_t i = 0; i < n; i++) {
            if (slots[i].rt_write_version != 0u) {
                has_rt = 1;
            }
            if (slots[i].used_sampled_copy) {
                used_copy = 1;
            }
        }
    }
    if (has_rt_out) {
        *has_rt_out = has_rt;
    }
    if (used_copy_out) {
        *used_copy_out = used_copy;
    }
}

int mgl_batch_trace_format_texslots_line(
    char *buf, size_t buflen, uint64_t flush_id, uint32_t batch_index,
    uint32_t cmd_index, uint32_t program, uint32_t vs, uint32_t fs,
    uint32_t pipe_program, const MGLBatchTraceTexSlotView slots[4])
{
    if (!buf || buflen == 0u || !slots) {
        return -1;
    }
    return snprintf(
        buf, buflen,
        "REPLAY_CMD_TEXSLOTS flush=%llu batch=%u cmd=%u program=%u vs=%u fs=%u "
        "pipelineProgram=%u "
        "s0(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u "
        "rtVer=%u sampledVer=%u size=%llux%llu fmt=%llu type=%llu) "
        "s1(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u "
        "rtVer=%u sampledVer=%u size=%llux%llu fmt=%llu type=%llu) "
        "s2(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u "
        "rtVer=%u sampledVer=%u size=%llux%llu fmt=%llu type=%llu) "
        "s3(tex=%u unit=%u prog=%u mtl=%p direct=%p copy=%p useCopy=%u fallback=%u "
        "rtVer=%u sampledVer=%u size=%llux%llu fmt=%llu type=%llu)",
        (unsigned long long)flush_id, (unsigned)batch_index, (unsigned)cmd_index,
        (unsigned)program, (unsigned)vs, (unsigned)fs, (unsigned)pipe_program,
        (unsigned)slots[0].gl_texture_name, (unsigned)slots[0].sampler_unit,
        (unsigned)slots[0].program_name, slots[0].mtl_texture_ptr,
        slots[0].direct_mtl_texture_ptr, slots[0].sampled_copy_ptr,
        (unsigned)slots[0].used_sampled_copy, (unsigned)slots[0].used_fallback,
        (unsigned)slots[0].rt_write_version, (unsigned)slots[0].sampled_write_version,
        (unsigned long long)slots[0].width, (unsigned long long)slots[0].height,
        (unsigned long long)slots[0].pixel_format,
        (unsigned long long)slots[0].texture_type,
        (unsigned)slots[1].gl_texture_name, (unsigned)slots[1].sampler_unit,
        (unsigned)slots[1].program_name, slots[1].mtl_texture_ptr,
        slots[1].direct_mtl_texture_ptr, slots[1].sampled_copy_ptr,
        (unsigned)slots[1].used_sampled_copy, (unsigned)slots[1].used_fallback,
        (unsigned)slots[1].rt_write_version, (unsigned)slots[1].sampled_write_version,
        (unsigned long long)slots[1].width, (unsigned long long)slots[1].height,
        (unsigned long long)slots[1].pixel_format,
        (unsigned long long)slots[1].texture_type,
        (unsigned)slots[2].gl_texture_name, (unsigned)slots[2].sampler_unit,
        (unsigned)slots[2].program_name, slots[2].mtl_texture_ptr,
        slots[2].direct_mtl_texture_ptr, slots[2].sampled_copy_ptr,
        (unsigned)slots[2].used_sampled_copy, (unsigned)slots[2].used_fallback,
        (unsigned)slots[2].rt_write_version, (unsigned)slots[2].sampled_write_version,
        (unsigned long long)slots[2].width, (unsigned long long)slots[2].height,
        (unsigned long long)slots[2].pixel_format,
        (unsigned long long)slots[2].texture_type,
        (unsigned)slots[3].gl_texture_name, (unsigned)slots[3].sampler_unit,
        (unsigned)slots[3].program_name, slots[3].mtl_texture_ptr,
        slots[3].direct_mtl_texture_ptr, slots[3].sampled_copy_ptr,
        (unsigned)slots[3].used_sampled_copy, (unsigned)slots[3].used_fallback,
        (unsigned)slots[3].rt_write_version, (unsigned)slots[3].sampled_write_version,
        (unsigned long long)slots[3].width, (unsigned long long)slots[3].height,
        (unsigned long long)slots[3].pixel_format,
        (unsigned long long)slots[3].texture_type);
}

int mgl_batch_trace_format_batch_line(char *buf, size_t buflen,
                                      const MGLBatchTraceBatchView *v)
{
    if (!buf || buflen == 0u || !v) return -1;
    return snprintf(
        buf, buflen,
        "REPLAY_BATCH_%s flush=%llu batch=%u commands=%u stream=%d mdiCompat=%d "
        "usesElements=%d key(program=%u pipeline=%u vs=%u fs=%u fbo=%u vao=%u prim=%u) "
        "snapshot(program=%u pipeline=%u current=%u fbo=%u vao=%p) "
        "restored(program=%u current=%u pipeline=%u vs=%u fs=%u fbo=%u vao=%p enabled=0x%x) "
        "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
        "drawBuf=0x%x readBuf=0x%x colorMask=%d%d%d%d depth(test=%d write=%d func=0x%x) "
        "blend=%d cull=%d cullFace=0x%x frontFace=0x%x dirty=0x%x encoder=%p "
        "pipelineState=%p rpFbo=%u rpColor=%p rpDepth=%p",
        v->phase ? v->phase : "STATE",
        (unsigned long long)v->flush_id, (unsigned)v->batch_index,
        (unsigned)v->command_count, (int)v->stream_merged, (int)v->mdi_compatible,
        (int)v->uses_elements, (unsigned)v->key_program, (unsigned)v->key_pipeline,
        (unsigned)v->key_vs, (unsigned)v->key_fs, (unsigned)v->key_fbo,
        (unsigned)v->key_vao, (unsigned)v->key_prim, (unsigned)v->snap_program,
        (unsigned)v->snap_pipeline, (unsigned)v->snap_current, (unsigned)v->snap_fbo,
        v->snap_vao, (unsigned)v->restored_current, (unsigned)v->restored_program,
        (unsigned)v->restored_pipeline, (unsigned)v->restored_vs,
        (unsigned)v->restored_fs, (unsigned)v->restored_fbo, v->restored_vao,
        (unsigned)v->enabled_attribs, (int)v->viewport[0], (int)v->viewport[1],
        (int)v->viewport[2], (int)v->viewport[3], v->scissor_test,
        (int)v->scissor[0], (int)v->scissor[1], (int)v->scissor[2],
        (int)v->scissor[3], (unsigned)v->draw_buf, (unsigned)v->read_buf,
        v->color_mask[0], v->color_mask[1], v->color_mask[2], v->color_mask[3],
        v->depth_test, v->depth_write, (unsigned)v->depth_func, v->blend, v->cull,
        (unsigned)v->cull_face, (unsigned)v->front_face, (unsigned)v->dirty,
        v->encoder, v->pipeline_state, (unsigned)v->rp_fbo, v->rp_color,
        v->rp_depth);
}

int mgl_batch_trace_format_cmd_line(char *buf, size_t buflen,
                                    const MGLBatchTraceCmdView *v)
{
    if (!buf || buflen == 0u || !v) return -1;
    return snprintf(
        buf, buflen,
        "REPLAY_CMD_%s flush=%llu batch=%u cmd=%u type=%s reason=%s "
        "program=%u vs=%u fs=%u mode=0x%x count=%d first=%d indexType=0x%x "
        "indexOffset=%u instances=%d baseVertex=%d baseInstance=%u ebo=%u eboPtr=%p "
        "encoder=%p pipelineState=%p fbo=%u rpFbo=%u rpColor=%p rpDepth=%p "
        "rpColorSize=%llux%llu rpDepthSize=%llux%llu rpLA/SA=%s/%s depthLA/SA=%s/%s "
        "fboColor0(tex=%u target=0x%x level=%u ptr=%p size=%ux%u mtl=%p init=%u/%u/%u "
        "rtVer=%u sampledVer=%u) "
        "fboDepth(tex=%u target=0x%x level=%u ptr=%p size=%ux%u mtl=%p init=%u/%u/%u "
        "rtVer=%u sampledVer=%u) "
        "units(u0 active=%u tex2D=%u u1 active=%u tex2D=%u u2 active=%u tex2D=%u) "
        "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) drawBuf=0x%x readBuf=0x%x "
        "depth(test=%d write=%d func=0x%x clear=%.6f) blend=%d cull=%d colorMask=%d%d%d%d",
        v->phase ? v->phase : "STATE", (unsigned long long)v->flush_id,
        (unsigned)v->batch_index, (unsigned)v->command_index,
        v->type_name ? v->type_name : "?", v->reason ? v->reason : "",
        (unsigned)v->program, (unsigned)v->vs, (unsigned)v->fs, (unsigned)v->mode,
        (int)v->count, (int)v->first, (unsigned)v->index_type,
        (unsigned)v->index_offset, (int)v->instances, (int)v->base_vertex,
        (unsigned)v->base_instance, (unsigned)v->ebo_name, v->ebo, v->encoder,
        v->pipeline_state, (unsigned)v->fbo_name, (unsigned)v->rp_fbo, v->rp_color,
        v->rp_depth, (unsigned long long)v->rp_color_w,
        (unsigned long long)v->rp_color_h, (unsigned long long)v->rp_depth_w,
        (unsigned long long)v->rp_depth_h, v->rp_la ? v->rp_la : "?",
        v->rp_sa ? v->rp_sa : "?", v->depth_la ? v->depth_la : "?",
        v->depth_sa ? v->depth_sa : "?", (unsigned)v->color0_tex,
        (unsigned)v->color0_target, (unsigned)v->color0_level, v->color0_ptr,
        (unsigned)v->color0_w, (unsigned)v->color0_h, v->color0_mtl,
        (unsigned)v->color0_ever, (unsigned)v->color0_full,
        (unsigned)v->color0_source, (unsigned)v->color0_rt_ver,
        (unsigned)v->color0_sampled_ver, (unsigned)v->depth_tex,
        (unsigned)v->depth_target, (unsigned)v->depth_level, v->depth_ptr,
        (unsigned)v->depth_w, (unsigned)v->depth_h, v->depth_mtl,
        (unsigned)v->depth_ever, (unsigned)v->depth_full,
        (unsigned)v->depth_source, (unsigned)v->depth_rt_ver,
        (unsigned)v->depth_sampled_ver, (unsigned)v->u0_active,
        (unsigned)v->u0_tex2d, (unsigned)v->u1_active, (unsigned)v->u1_tex2d,
        (unsigned)v->u2_active, (unsigned)v->u2_tex2d, (int)v->viewport[0],
        (int)v->viewport[1], (int)v->viewport[2], (int)v->viewport[3],
        v->scissor_test, (int)v->scissor[0], (int)v->scissor[1],
        (int)v->scissor[2], (int)v->scissor[3], (unsigned)v->draw_buf,
        (unsigned)v->read_buf, v->depth_test, v->depth_write,
        (unsigned)v->depth_func, v->depth_clear, v->blend, v->cull,
        v->color_mask[0], v->color_mask[1], v->color_mask[2], v->color_mask[3]);
}

void mgl_batch_trace_copy_state_to_batch(MGLBatchTraceBatchView *v,
                                         const MGLBatchTraceStatePod *s)
{
    if (!v || !s) return;
    for (int i = 0; i < 4; i++) v->viewport[i] = s->viewport[i];
    v->scissor_test = s->scissor_test;
    for (int i = 0; i < 4; i++) v->scissor[i] = s->scissor[i];
    v->draw_buf = s->draw_buf;
    v->read_buf = s->read_buf;
    for (int i = 0; i < 4; i++) v->color_mask[i] = s->color_mask[i];
    v->depth_test = s->depth_test;
    v->depth_write = s->depth_write;
    v->depth_func = s->depth_func;
    v->blend = s->blend;
    v->cull = s->cull;
    v->cull_face = s->cull_face;
    v->front_face = s->front_face;
    v->dirty = s->dirty;
}

void mgl_batch_trace_copy_state_to_cmd(MGLBatchTraceCmdView *v,
                                       const MGLBatchTraceStatePod *s)
{
    if (!v || !s) return;
    for (int i = 0; i < 4; i++) v->viewport[i] = s->viewport[i];
    v->scissor_test = s->scissor_test;
    for (int i = 0; i < 4; i++) v->scissor[i] = s->scissor[i];
    v->draw_buf = s->draw_buf;
    v->read_buf = s->read_buf;
    v->depth_test = s->depth_test;
    v->depth_write = s->depth_write;
    v->depth_func = s->depth_func;
    v->depth_clear = s->depth_clear;
    v->blend = s->blend;
    v->cull = s->cull;
    for (int i = 0; i < 4; i++) v->color_mask[i] = s->color_mask[i];
}

int mgl_batch_trace_format_rt_write_mark(char *buf, size_t buflen,
                                         const MGLBatchTraceRtWriteView *v)
{
    if (!buf || buflen == 0u || !v) return -1;
    return snprintf(
        buf, buflen,
        "RT_SAMPLE_COPY_WRITE_MARK hit=%llu fbo=%u program=%u rtTex=%u "
        "label=\"%s\" depthTex=%u depthLabel=\"%s\" viewport=%d,%d,%d,%d "
        "scissor(en=%d box=%d,%d,%d,%d) depth(test=%d write=%d func=0x%x) "
        "blend=%d cull=%d colorMask=%d%d%d%d level=%u "
        "texInit(ever=%u full=%u source=%u) levels=%u mips=%u mipmapped=%u "
        "mtlColor=%p fmt=%llu size=%llux%llu rpColor=%p rpDepth=%p depthMTL=%p",
        (unsigned long long)v->hit, (unsigned)v->fbo_name, (unsigned)v->program,
        (unsigned)v->rt_tex, v->rt_label ? v->rt_label : "",
        (unsigned)v->depth_tex, v->depth_label ? v->depth_label : "",
        (int)v->viewport[0], (int)v->viewport[1], (int)v->viewport[2],
        (int)v->viewport[3], v->scissor_en, (int)v->scissor[0],
        (int)v->scissor[1], (int)v->scissor[2], (int)v->scissor[3],
        v->depth_test, v->depth_write, (unsigned)v->depth_func, v->blend,
        v->cull, v->color_mask[0], v->color_mask[1], v->color_mask[2],
        v->color_mask[3], (unsigned)v->level, (unsigned)v->ever,
        (unsigned)v->full, (unsigned)v->source, (unsigned)v->levels,
        (unsigned)v->mips, (unsigned)v->mipmapped, v->mtl_color,
        (unsigned long long)v->fmt, (unsigned long long)v->width,
        (unsigned long long)v->height, v->rp_color, v->rp_depth, v->depth_mtl);
}

void mgl_batch_trace_copy_state_to_rt(MGLBatchTraceRtWriteView *v,
                                      const MGLBatchTraceStatePod *s)
{
    if (!v || !s) {
        return;
    }
    for (int i = 0; i < 4; i++) {
        v->viewport[i] = s->viewport[i];
        v->scissor[i] = s->scissor[i];
        v->color_mask[i] = s->color_mask[i];
    }
    v->scissor_en = s->scissor_test;
    v->depth_test = s->depth_test;
    v->depth_write = s->depth_write;
    v->depth_func = s->depth_func;
    v->blend = s->blend;
    v->cull = s->cull;
}

void mgl_batch_rt_run_draw_attachments(const MGLBatchRtDrawMarkOps *ops)
{
    if (!ops || !ops->mark_attachment || ops->max_attachments == 0u) {
        return;
    }
    uint8_t marked[64];
    if (ops->max_attachments > 64u) {
        return;
    }
    for (uint32_t i = 0; i < ops->max_attachments; i++) {
        marked[i] = 0u;
    }
    if (ops->resolve_draw_slot) {
        for (uint32_t slot = 0; slot < ops->draw_buffer_count; slot++) {
            uint32_t att = 0u;
            if (!ops->resolve_draw_slot(ops->ctx, slot, &att)) {
                continue;
            }
            ops->mark_attachment(ops->ctx, att);
            if (att < ops->max_attachments) {
                marked[att] = 1u;
            }
        }
    }
    if (!ops->has_rp_owner || !ops->attachment_mtl || !ops->rp_has_mtl) {
        return;
    }
    for (uint32_t att = 0; att < ops->max_attachments; att++) {
        if (!mgl_batch_rt_should_cross_mark(
                marked[att] ? 1 : 0,
                mgl_batch_rt_attachment_active(ops->color_attachment_bitfield,
                                               att, ops->max_attachments))) {
            continue;
        }
        void *mtl = ops->attachment_mtl(ops->ctx, att);
        if (!mtl) {
            continue;
        }
        if (ops->rp_has_mtl(ops->ctx, mtl)) {
            ops->mark_attachment(ops->ctx, att);
        }
    }
}

