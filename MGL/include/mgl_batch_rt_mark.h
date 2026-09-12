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

#include "glm_context.h"  /* GLMContext (host ports) */
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

void mgl_batch_trace_copy_state_to_rt(MGLBatchTraceRtWriteView *v,
                                      const MGLBatchTraceStatePod *s);

/* Draw-buffer marks + RP cross-check loops (ObjC supplies resolve/mark/mtl). */
typedef struct MGLBatchRtDrawMarkOps {
    void *ctx;
    uint32_t max_attachments;
    uint32_t draw_buffer_count;
    uint32_t color_attachment_bitfield;
    int (*resolve_draw_slot)(void *ctx, uint32_t slot, uint32_t *att_out);
    void (*mark_attachment)(void *ctx, uint32_t att);
    int has_rp_owner;
    void *(*attachment_mtl)(void *ctx, uint32_t att); /* NULL → skip */
    int (*rp_has_mtl)(void *ctx, void *mtl);
} MGLBatchRtDrawMarkOps;

void mgl_batch_rt_run_draw_attachments(const MGLBatchRtDrawMarkOps *ops);

/* ---- host ports (ObjC-zeroing T4) -------------------------------------
 * These were the Objective-C category methods
 * -[MGLRenderer markCurrentFramebufferColorAttachmentWrittenAtIndex:] and
 * -[MGLRenderer markCurrentFramebufferDrawAttachmentsWritten]; the renderer is
 * passed as a handle so C callers can drive them. */

/* Marks one colour attachment as written (also emits the RT-write trace when
 * the plan asks for it). */
void mglBatchRtMarkColorAttachmentWritten(void *renderer, GLMContext ctx,
                                          uint32_t attachment_index);

/* Marks every draw attachment of the current framebuffer as written. */
void mglBatchRtMarkCurrentFramebufferDrawAttachments(void *renderer,
                                                     GLMContext ctx);

/* ---- A3: trace fill helpers (shrink replay_trace ObjC) ---- */

int mgl_batch_trace_should_emit(int log_enabled, int should_log_replay,
                                int fs_has_rt, int fs_used_copy);
int mgl_batch_trace_is_submit_phase(const char *phase);

void mgl_batch_trace_batch_fill_key_flags(
    MGLBatchTraceBatchView *v, uint32_t command_count, int stream_merged,
    int mdi_compatible, int uses_elements, uint32_t key_program,
    uint32_t key_pipeline, uint32_t key_vs, uint32_t key_fs, uint32_t key_fbo,
    uint32_t key_vao, uint32_t key_prim);

void mgl_batch_trace_cmd_fill_draw(MGLBatchTraceCmdView *v,
                                   const char *type_name, uint32_t mode,
                                   int32_t count, int32_t first,
                                   uint32_t index_type, uint32_t index_offset,
                                   int32_t instances, int32_t base_vertex,
                                   uint32_t base_instance);

typedef struct MGLBatchTraceAttPod {
    uint32_t tex;
    uint32_t target;
    uint32_t level;
    const void *ptr;
    uint32_t w, h;
    const void *mtl;
    uint32_t ever, full, source;
    uint32_t rt_ver, sampled_ver;
} MGLBatchTraceAttPod;

void mgl_batch_trace_cmd_set_color0(MGLBatchTraceCmdView *v,
                                    const MGLBatchTraceAttPod *p);
void mgl_batch_trace_cmd_set_depth(MGLBatchTraceCmdView *v,
                                   const MGLBatchTraceAttPod *p);
void mgl_batch_trace_cmd_set_units(MGLBatchTraceCmdView *v, uint32_t u0a,
                                   uint32_t u0t, uint32_t u1a, uint32_t u1t,
                                   uint32_t u2a, uint32_t u2t);

/* Single color-attachment mark orchestration. */
typedef struct MGLBatchRtMarkOneOps {
    void *ctx;
    void (*mark_level)(void *ctx);
    int yflip;
    void (*apply_yflip)(void *ctx);
    int should_diag;
    void (*emit_diag)(void *ctx, uint64_t hit);
} MGLBatchRtMarkOneOps;

void mgl_batch_rt_mark_one_attachment(uint32_t attachment_index,
                                      uint32_t bitfield, uint32_t max_attachments,
                                      uint64_t *diag_hit_inout,
                                      const MGLBatchRtMarkOneOps *ops);

/* Draw-submission records: frame counters, last-draw snapshot and the RT-mark
 * for one submitted draw.  Defined in mgl_batch_rt_mark_host.c. */
void mglBatchRecordArrayDrawSubmitted(void *renderer, GLMContext ctx, GLenum mode,
                                      uint64_t vertex_count);
void mglBatchRecordElementDrawSubmitted(void *renderer, GLMContext ctx, GLenum mode,
                                        uint64_t index_count);

/* === Replay trace drivers (former -[MGLRenderer traceReplayBatch:...] /
 * -[MGLRenderer traceReplayCommand:...]; defined in mgl_batch_replay_trace.c) ===
 * They read renderer state through mgl_renderer_ports.h and only emit when the
 * trace gates in mgl_trace_strategy.h say so. */
void mglBatchTraceReplayBatch(void *renderer, MGLDrawBatch *batch,
                              GLMContext glm_ctx, uint64_t flush_id,
                              uint32_t batch_index, const char *phase);
void mglBatchTraceReplayCommand(void *renderer, MGLDrawBatch *batch,
                                MGLDrawCommand *cmd, GLMContext glm_ctx,
                                uint64_t flush_id, uint32_t batch_index,
                                uint32_t command_index, const char *phase,
                                const char *reason);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_RT_MARK_H */
