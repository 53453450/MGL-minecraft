/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_batch_rt_mark.h — A3 / O2.5: framebuffer RT-write mark plans (no Metal).
 *
 * ObjC markCurrentFramebuffer* fills POD flags and applies mtl_data bumps.
 */

#ifndef MGL_BATCH_RT_MARK_H
#define MGL_BATCH_RT_MARK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* True when color attachment index is in-range and present in the bitfield. */
int mgl_batch_rt_attachment_active(uint32_t bitfield, uint32_t index,
                                   uint32_t max_attachments);

/*
 * Minecraft GUI/mesh RT Y-flip authority: VS framebuffer Y-flip injection
 * wrote GL-visible orientation, so RT_SAMPLE_COPY must not flip again —
 * unless the program samples InSampler / DiffuseSampler (true FB inputs).
 */
int mgl_batch_rt_yflip_authority(int has_injected_yflip,
                                 int yflip_sampler_explicit,
                                 int has_in_sampler_named,
                                 int has_diffuse_sampler_named);

/* Rate-limit RT_SAMPLE_COPY_WRITE_MARK diag (attachment0 + can_use_copy). */
int mgl_batch_rt_should_trace_write_mark(uint64_t hit);


/* Cross-check candidate: active attachment not already marked by draw-buffers. */
int mgl_batch_rt_should_cross_mark(int already_marked, int attachment_active);

/* Whether attachment0 write-mark should emit RT_SAMPLE_COPY diag. */
int mgl_batch_rt_should_diag_attachment0(uint32_t attachment_index,
                                         int trace_enabled, int can_use_copy);


/* Trace early-exit: any of N FS slots has RT write or used sampled copy. */
typedef struct MGLBatchTraceFsSlot {
    uint32_t rt_write_version;
    uint8_t used_sampled_copy;
} MGLBatchTraceFsSlot;

void mgl_batch_trace_fs_slot_flags(const MGLBatchTraceFsSlot *slots, uint32_t n,
                                   int *has_rt_out, int *used_copy_out);

typedef struct MGLBatchTraceTexSlotView {
    uint32_t gl_texture_name;
    uint32_t sampler_unit;
    uint32_t program_name;
    const void *mtl_texture_ptr;
    const void *direct_mtl_texture_ptr;
    const void *sampled_copy_ptr;
    uint8_t used_sampled_copy;
    uint8_t used_fallback;
    uint32_t rt_write_version;
    uint32_t sampled_write_version;
    uint64_t width;
    uint64_t height;
    uint64_t pixel_format;
    uint64_t texture_type;
} MGLBatchTraceTexSlotView;

/* Format REPLAY_CMD_TEXSLOTS body (no prefix). Returns bytes or -1. */
int mgl_batch_trace_format_texslots_line(
    char *buf, size_t buflen, uint64_t flush_id, uint32_t batch_index,
    uint32_t cmd_index, uint32_t program, uint32_t vs, uint32_t fs,
    uint32_t pipe_program, const MGLBatchTraceTexSlotView slots[4]);


typedef struct MGLBatchTraceBatchView {
    const char *phase;
    uint64_t flush_id;
    uint32_t batch_index;
    uint32_t command_count;
    uint8_t stream_merged;
    uint8_t mdi_compatible;
    uint8_t uses_elements;
    uint32_t key_program;
    uint32_t key_pipeline;
    uint32_t key_vs;
    uint32_t key_fs;
    uint32_t key_fbo;
    uint32_t key_vao;
    uint32_t key_prim;
    uint32_t snap_program;
    uint32_t snap_pipeline;
    uint32_t snap_current;
    uint32_t snap_fbo;
    const void *snap_vao;
    uint32_t restored_current;
    uint32_t restored_program;
    uint32_t restored_pipeline;
    uint32_t restored_vs;
    uint32_t restored_fs;
    uint32_t restored_fbo;
    const void *restored_vao;
    uint32_t enabled_attribs;
    int32_t viewport[4];
    int scissor_test;
    int32_t scissor[4];
    uint32_t draw_buf;
    uint32_t read_buf;
    int color_mask[4];
    int depth_test;
    int depth_write;
    uint32_t depth_func;
    int blend;
    int cull;
    uint32_t cull_face;
    uint32_t front_face;
    uint32_t dirty;
    const void *encoder;
    const void *pipeline_state;
    uint32_t rp_fbo;
    const void *rp_color;
    const void *rp_depth;
} MGLBatchTraceBatchView;


/* Fill viewport/scissor/depth/blend/cull/colorMask shared by batch+cmd traces. */
typedef struct MGLBatchTraceStatePod {
    int32_t viewport[4];
    int scissor_test;
    int32_t scissor[4];
    uint32_t draw_buf;
    uint32_t read_buf;
    int color_mask[4];
    int depth_test;
    int depth_write;
    uint32_t depth_func;
    double depth_clear;
    int blend;
    int cull;
    uint32_t cull_face;
    uint32_t front_face;
    uint32_t dirty;
} MGLBatchTraceStatePod;

void mgl_batch_trace_copy_state_to_batch(MGLBatchTraceBatchView *v,
                                         const MGLBatchTraceStatePod *s);

int mgl_batch_trace_format_batch_line(char *buf, size_t buflen,
                                      const MGLBatchTraceBatchView *v);

typedef struct MGLBatchTraceCmdView {
    const char *phase;
    const char *reason;
    const char *type_name;
    uint64_t flush_id;
    uint32_t batch_index;
    uint32_t command_index;
    uint32_t program;
    uint32_t vs;
    uint32_t fs;
    uint32_t mode;
    int32_t count;
    int32_t first;
    uint32_t index_type;
    uint32_t index_offset;
    int32_t instances;
    int32_t base_vertex;
    uint32_t base_instance;
    uint32_t ebo_name;
    const void *ebo;
    const void *encoder;
    const void *pipeline_state;
    uint32_t fbo_name;
    uint32_t rp_fbo;
    const void *rp_color;
    const void *rp_depth;
    uint64_t rp_color_w, rp_color_h;
    uint64_t rp_depth_w, rp_depth_h;
    const char *rp_la;
    const char *rp_sa;
    const char *depth_la;
    const char *depth_sa;
    uint32_t color0_tex;
    uint32_t color0_target;
    uint32_t color0_level;
    const void *color0_ptr;
    uint32_t color0_w, color0_h;
    const void *color0_mtl;
    uint32_t color0_ever, color0_full, color0_source;
    uint32_t color0_rt_ver, color0_sampled_ver;
    uint32_t depth_tex;
    uint32_t depth_target;
    uint32_t depth_level;
    const void *depth_ptr;
    uint32_t depth_w, depth_h;
    const void *depth_mtl;
    uint32_t depth_ever, depth_full, depth_source;
    uint32_t depth_rt_ver, depth_sampled_ver;
    uint32_t u0_active, u0_tex2d, u1_active, u1_tex2d, u2_active, u2_tex2d;
    int32_t viewport[4];
    int scissor_test;
    int32_t scissor[4];
    uint32_t draw_buf;
    uint32_t read_buf;
    int depth_test;
    int depth_write;
    uint32_t depth_func;
    double depth_clear;
    int blend;
    int cull;
    int color_mask[4];
} MGLBatchTraceCmdView;

void mgl_batch_trace_copy_state_to_cmd(MGLBatchTraceCmdView *v,
                                       const MGLBatchTraceStatePod *s);

int mgl_batch_trace_format_cmd_line(char *buf, size_t buflen,
                                    const MGLBatchTraceCmdView *v);




typedef struct MGLBatchTraceRtWriteView {
    uint64_t hit;
    uint32_t fbo_name;
    uint32_t program;
    uint32_t rt_tex;
    const char *rt_label;
    uint32_t depth_tex;
    const char *depth_label;
    int32_t viewport[4];
    int scissor_en;
    int32_t scissor[4];
    int depth_test;
    int depth_write;
    uint32_t depth_func;
    int blend;
    int cull;
    int color_mask[4];
    uint32_t level;
    uint32_t ever, full, source;
    uint32_t levels, mips, mipmapped;
    const void *mtl_color;
    uint64_t fmt, width, height;
    const void *rp_color;
    const void *rp_depth;
    const void *depth_mtl;
} MGLBatchTraceRtWriteView;

int mgl_batch_trace_format_rt_write_mark(char *buf, size_t buflen,
                                         const MGLBatchTraceRtWriteView *v);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_RT_MARK_H */
