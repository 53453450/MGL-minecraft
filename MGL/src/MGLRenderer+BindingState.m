/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+BindingState.m
// Vertex/fragment buffer, attribute and texture binding methods
// extracted from MGLRenderer+Draw.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "pixel_utils.h"

enum {
    MGL_BINDING_RESOURCE_STORAGE_SHARED = 0u,
    MGL_BINDING_VERTEX_FORMAT_INVALID = 0u,
    MGL_BINDING_PIXEL_FORMAT_INVALID = 0u,
    /* Match MGLTextureType / MTLTextureType values from mgl_render_values.h.
     * (A prior local enum used 3D=4, which is actually 2DMultisample.) */
    MGL_BINDING_TEXTURE_TYPE_1D = MGLTextureType1D,
    MGL_BINDING_TEXTURE_TYPE_1D_ARRAY = MGLTextureType1DArray,
    MGL_BINDING_TEXTURE_TYPE_2D = MGLTextureType2D,
    MGL_BINDING_TEXTURE_TYPE_2D_ARRAY = MGLTextureType2DArray,
    MGL_BINDING_TEXTURE_TYPE_3D = MGLTextureType3D,
    MGL_BINDING_TEXTURE_TYPE_CUBE = MGLTextureTypeCube,
    MGL_BINDING_TEXTURE_TYPE_CUBE_ARRAY = MGLTextureTypeCubeArray,
};

static uint64_t mglBindingStateBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo((__bridge void *)buffer, &info) == 0
        ? info.length : 0u;
}

static MGLRenderTextureInfo mglBindingStateTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

static uint32_t mglBindingStateTexturePixelFormat(id texture)
{
    return mglBindingStateTextureInfo(texture).pixel_format;
}

static uint32_t mglBindingStateTextureType(id texture)
{
    return mglBindingStateTextureInfo(texture).texture_type;
}

/* UBOs use the reflected block size (not the generic 256-byte stage floor).
 * When the client bound the whole store but it is shorter than the padded
 * reflection size, require only the bytes actually visible in the binding. */
static NSUInteger mglRequiredBindingBytesForMap(const BufferMap *map,
                                                NSUInteger reflectedRequiredBytes)
{
    GLsizeiptr vis = map ? mglBufferMapVisibleSize(map) : 0;
    return (NSUInteger)mglRenderRequiredBindingBytesForMap(
        map ? (int)map->resource_type : -1, (uint32_t)reflectedRequiredBytes,
        (int64_t)vis, (uint32_t)kMGLMinimumStageBindingSize);
}

static uint64_t mglBindingStateTextureWidth(id texture)
{
    return mglBindingStateTextureInfo(texture).width;
}

static uint64_t mglBindingStateTextureHeight(id texture)
{
    return mglBindingStateTextureInfo(texture).height;
}

static uint64_t mglBindingStateTextureArrayLength(id texture)
{
    return mglBindingStateTextureInfo(texture).array_length;
}

static uint64_t mglBindingStateTextureMipmapLevelCount(id texture)
{
    return mglBindingStateTextureInfo(texture).mipmap_level_count;
}

static BOOL mglBindingStateRenderPassUsesColorTexture(
    void *owner,
    void *texture,
    NSUInteger *attachmentIndexOut)
{
    uint32_t attachmentIndex = MAX_COLOR_ATTACHMENTS;
    const BOOL found = mglRenderPassUsesColorTextureOwner(
        owner, texture, &attachmentIndex);
    if (attachmentIndexOut) {
        *attachmentIndexOut = attachmentIndex;
    }
    return found;
}

static BOOL mglBindingStateHasActiveEncoder(const MGLEncodeContext *encCtx)
{
    if (!encCtx) {
        return NO;
    }
    return mglRenderEncoderOwnerHasCurrent(
        encCtx->render_encoder_owner) != 0;
}

static id mglBindingStateCreateBuffer(
    id device,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglBindingStateCreateBufferWithBytes(
    id device,
    const void *bytes,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglBindingStateCreateTextureLevelView(
    id texture,
    NSUInteger level,
    NSUInteger sliceCount)
{
    const MGLRenderTextureInfo info = mglBindingStateTextureInfo(texture);
    if (!texture || info.width == 0u || info.height == 0u) return nil;
    void *view = NULL;
    if (mglRenderCreateTextureViewRange(
            (__bridge void *)texture, info.pixel_format,
            info.texture_type, level, 1, 0, sliceCount,
            0, 0, 0, 0, 0, &view) == 0 && view) {
        return (__bridge_transfer id)view;
    }
    return nil;
}

static id mglBindingStateCacheImageUnitView(ImageUnit *iu, id fallback, void *view)
{
    if (!view) {
        return fallback;
    }
    if (iu->mtl_image_view) {
        mglRenderReleaseMetalObject(iu->mtl_image_view);
        iu->mtl_image_view = NULL;
    }
    iu->mtl_image_view = view; /* +1 from newTextureView */
    return (__bridge id)iu->mtl_image_view;
}

/* Metal pixel format for BindImageTexture <format>.  Same-size reinterprets
 * (e.g. RGBA8 storage bound as R32I/R32UI) require a PixelFormatView so AIR
 * integer atomics/load/store hit the intended layout (CTS advanced-cast). */
static uint32_t mglBindingStateImageBindPixelFormat(const ImageUnit *iu,
                                                    uint32_t native_format)
{
    if (!iu) {
        return native_format;
    }
    const uint32_t bind_format =
        mtlFormatForGLInternalFormat(iu->internalformat);
    return mglRenderImageBindPixelFormat(iu->internalformat, native_format,
                                         bind_format);
}

/* For non-layered BindImageTexture on array/3D/cube textures, GLSL image2D
 * (etc.) expects a single 2D (or 1D) slice.  Create a Type2D/Type1D view of
 * that layer so AIR write/read_texture_2d matches the bound Metal type.
 * Multisample images are intentionally backed as Type2DArray (sample→layer)
 * and must keep that type.  Views are cached on ImageUnit so they outlive
 * the bind call until the unit is rebound/reset.
 *
 * Also applies BindImageTexture <format> via PixelFormatView so float and
 * integer imageLoad/Store pack/unpack match the bound internalformat (CTS
 * multiple-uniforms / advanced-cast). Shared by VS/FS, GS, and TCS/TES. */
void *mglRendererStorageImageTexture(void *base_texture, ImageUnit *iu)
{
    id texture = (__bridge id)base_texture;
    if (!texture || !iu) {
        return base_texture;
    }
    if (iu->mtl_image_view) {
        return iu->mtl_image_view;
    }
    const MGLRenderTextureInfo info = mglBindingStateTextureInfo(texture);
    if (info.width == 0u) {
        return base_texture;
    }
    const NSUInteger level = (NSUInteger)iu->level;
    /* Mutable textures may BindImage a mip that was never defined. Metal
     * rejects views past mipmapLevelCount — leave unbound so loads read 0
     * and stores are ignored (CTS incomplete_textures). */
    if (!mglRenderImageLevelInRange((uint32_t)level,
                                    (uint32_t)info.mipmap_level_count)) {
        return NULL;
    }
    const uint32_t srcType = info.texture_type;
    const uint32_t bindFormat =
        mglBindingStateImageBindPixelFormat(iu, info.pixel_format);
    const GLenum glTarget = iu->tex ? iu->tex->target : (GLenum)0;
    const int isMsTarget = mglRenderImageTargetIsMultisample((uint32_t)glTarget);
    uint32_t dstType = 0u;
    if (mglRenderImageNeedsNonLayeredSlice(iu->layered ? 1 : 0, isMsTarget,
                                           srcType, &dstType)) {
            void *view = NULL;
            int rc = mglRenderCreateTextureViewRange(
                    base_texture, bindFormat, dstType,
                    level, 1u, (uint64_t)iu->layer, 1u,
                    0, 0, 0, 0, 0, &view);
            if (rc == 0 && view) {
                return (__bridge void *)mglBindingStateCacheImageUnitView(
                    iu, texture, view);
            }
    }

    if (mglRenderImageNeedsFormatOrMipView(level, bindFormat,
                                           info.pixel_format)) {
        NSUInteger sliceCount = (NSUInteger)mglRenderImageViewSliceCount(
            srcType, mglBindingStateTextureArrayLength(texture));
        void *view = NULL;
        if (mglRenderCreateTextureViewRange(
                base_texture, bindFormat,
                info.texture_type, level, 1u, 0u, sliceCount,
                0, 0, 0, 0, 0, &view) == 0 && view) {
            return (__bridge void *)mglBindingStateCacheImageUnitView(
                iu, texture, view);
        }
    }
    return base_texture;
}

static id mglBindingStateCreateStorageImageView(id texture, ImageUnit *iu)
{
    return (__bridge id)mglRendererStorageImageTexture(
        (__bridge void *)texture, iu);
}

static void mglBindingStateSetVertexBuffer(
    void *renderEncoderOwner,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglBindingStateSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetRenderBytesForOwner(
        renderEncoderOwner, bytes, length,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglBindingStateSetFragmentBuffer(
    void *renderEncoderOwner,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_BINDING_STAGE_FRAGMENT, (uint32_t)index);
}

static void mglBindingStateSetFragmentBytes(
    void *renderEncoderOwner,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetRenderBytesForOwner(
        renderEncoderOwner, bytes, length,
        MGL_RENDER_BINDING_STAGE_FRAGMENT, (uint32_t)index);
}

static bool mglBindingStateCollectResourceBinding(
    MGLRenderResourceBindingSnapshot *snapshot,
    uint32_t stage,
    uint32_t kind,
    void *resource,
    uint32_t index)
{
    if (!snapshot || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        kind > MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return false;
    }
    uint32_t *count = stage == MGL_RENDER_BINDING_STAGE_VERTEX
        ? &snapshot->vertex_op_count : &snapshot->fragment_op_count;
    MGLRenderResourceBindingOp *ops =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? snapshot->vertex_ops : snapshot->fragment_ops;
    if (*count >= MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS) {
        return false;
    }
    ops[(*count)++] = (MGLRenderResourceBindingOp){
        .kind = kind,
        .index = index,
        .resource = resource,
    };
    return true;
}

static bool mglBindingStateQueueResourceBinding(
    BOOL collect,
    void *bindingStateOwner,
    void *renderEncoderOwner,
    MGLRenderResourceBindingSnapshot *snapshot,
    uint32_t stage,
    uint32_t kind,
    void *resource,
    uint32_t index)
{
    if (collect) {
        return mglBindingStateCollectResourceBinding(
            snapshot, stage, kind, resource, index);
    }
    if (kind == MGL_RENDER_RESOURCE_BINDING_TEXTURE) {
        return mglRenderBindingSetTextureForOwner(
            bindingStateOwner, renderEncoderOwner,
            resource, stage, index) >= 0;
    }
    if (kind == MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return mglRenderBindingSetSamplerForOwner(
            bindingStateOwner, renderEncoderOwner,
            resource, stage, index) >= 0;
    }
    return false;
}

static bool mglBindingStateFlushResourceBindings(
    void *bindingStateOwner,
    void *renderEncoderOwner,
    MGLRenderResourceBindingSnapshot *snapshot)
{
    if (!snapshot ||
        (snapshot->vertex_op_count == 0 &&
         snapshot->fragment_op_count == 0)) {
        return true;
    }
    if (mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
            bindingStateOwner, renderEncoderOwner, snapshot, NULL, 0) != 0) {
        return false;
    }
    *snapshot = (MGLRenderResourceBindingSnapshot){0};
    return true;
}


/* O3.3: fill POD input for mglBindingStagePlanMapEntry (thin set*Buffer ports). */
static void mglBindingStateFillStageBindInput(
    MGLStageBufferBindInput *in, int is_fragment, int phase,
    const BufferMap *map, Buffer *ptr, int is_base_binding,
    int attrib_reserved, uint32_t max_metal_slots, uint32_t reflected,
    uint64_t visible_cpu, int64_t visible_range, int allow_isolate_when_gpu)
{
    memset(in, 0, sizeof(*in));
    in->phase = phase;
    in->is_fragment = is_fragment ? 1 : 0;
    in->is_base_binding = is_base_binding ? 1 : 0;
    in->has_metal_binding = map && map->has_metal_binding ? 1 : 0;
    in->metal_binding_index =
        map ? (int32_t)map->metal_binding_index : -1;
    in->gl_binding_index = map ? (int32_t)map->buffer_base_index : -1;
    in->resource_type = map ? (uint32_t)map->resource_type : 0u;
    in->offset = map ? map->offset : 0;
    in->buffer_size = ptr ? ptr->size : -1;
    in->has_buffer = ptr ? 1 : 0;
    in->has_cpu_data = ptr && ptr->data.buffer_data ? 1 : 0;
    in->has_mtl_data = ptr && ptr->data.mtl_data ? 1 : 0;
    in->cpu_ptr = ptr ? (const void *)(uintptr_t)ptr->data.buffer_data : NULL;
    in->mtl_ptr = ptr ? ptr->data.mtl_data : NULL;
    in->cpu_dirty =
        ptr && mglRenderBufferHasCPUDirty(ptr->data.dirty_bits) ? 1 : 0;
    in->gpu_write_target = ptr && ptr->gpu_write_target ? 1 : 0;
    in->allow_isolate_when_gpu = allow_isolate_when_gpu ? 1 : 0;
    in->attrib_slot_reserved = attrib_reserved ? 1 : 0;
    in->max_metal_slots = max_metal_slots;
    in->max_gl_bindings = (uint32_t)MAX_BINDABLE_BUFFERS;
    in->reflected_required = reflected;
    in->min_stage_bytes = (uint32_t)kMGLMinimumStageBindingSize;
    in->scratch_cap = (uint32_t)kMGLStageBindingStackScratchSize;
    in->visible_cpu = visible_cpu;
    in->visible_range = visible_range;
    if (is_fragment) {
        in->mtl_usable =
            ptr && ptr->data.mtl_data &&
                    (uintptr_t)ptr->data.mtl_data >= 0x100000000ULL
                ? 1
                : 0;
    } else {
        in->mtl_usable =
            ptr && mglRenderMetalDataPointerUsable(ptr->data.mtl_data) ? 1 : 0;
    }
}


/* O3.3 apply-mask / port-collapse: compact log + shared V/F snapshot emit. */
#define MGL_EMIT_DR_LOG(...) \ do { \ MGLBindingDepthLog _mgl_dr_log = {__VA_ARGS__}; \ mglBindingLogDepthRecover(&_mgl_dr_log);                                 \
    } while (0)

#define MGL_EMIT_RT_LOG(...) \ do { \ MGLBindingRTCopyLog _mgl_rt_log = {__VA_ARGS__}; \ mglBindingLogRTSampleCopy(&_mgl_rt_log);                                 \
    } while (0)

/* Shared V/F binding-snapshot apply ports (frag=0 vertex, 1 fragment).
 * Residual VATTR/VFB/VPS/FFB wrappers dereference *snapshot onto these. */
#define MGL_BIND_SNAP_FLUSH(snap, frag, scratchUsed)                             \
    do {                                                                         \
        uint32_t *_mgl_cnt = (frag) ? &(snap).fragment_op_count                   \
                                    : &(snap).vertex_op_count;                   \
        if (*_mgl_cnt > 0) {                                                     \
            mglRenderEncodeBindingSnapshotForRenderEncoderOwner(                 \
                encCtx->render_encoder_owner, &(snap), NULL, 0);                 \
            (snap) = (MGLRenderBindingSnapshot){0};                              \
            (scratchUsed) = 0;                                                   \
        }                                                                        \
    } while (0)
#define MGL_BIND_SNAP_COLLECT_BUFFER(snap, frag, scratchUsed, slot, bufPtr, off)  \
    do {                                                                         \
        uint32_t *_mgl_cnt = (frag) ? &(snap).fragment_op_count                   \
                                    : &(snap).vertex_op_count;                   \
        MGLRenderBindingOp *_mgl_ops =                                           \
            (frag) ? (snap).fragment_ops : (snap).vertex_ops;                    \
        if (*_mgl_cnt >= MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                  \
            MGL_BIND_SNAP_FLUSH(snap, frag, scratchUsed);                        \
            _mgl_cnt = (frag) ? &(snap).fragment_op_count                        \
                              : &(snap).vertex_op_count;                         \
            _mgl_ops = (frag) ? (snap).fragment_ops : (snap).vertex_ops;         \
        }                                                                        \
        _mgl_ops[(*_mgl_cnt)++] =                                                \
            (MGLRenderBindingOp){0u, (uint32_t)(slot), (uint64_t)(off),          \
                                 (void *)(bufPtr), NULL, 0u};                    \
    } while (0)
#define MGL_BIND_SNAP_COLLECT_BYTES(snap, frag, scratch, scratchUsed, scratchCap, slot, src, len) \
    do {                                                                         \
        const void *src_ = (src);                                                \
        size_t len_ = (len);                                                     \
        uint32_t *_mgl_cnt = (frag) ? &(snap).fragment_op_count                   \
                                    : &(snap).vertex_op_count;                   \
        MGLRenderBindingOp *_mgl_ops =                                           \
            (frag) ? (snap).fragment_ops : (snap).vertex_ops;                    \
        if ((scratchUsed) + len_ > (scratchCap) ||                               \
            *_mgl_cnt >= MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                  \
            MGL_BIND_SNAP_FLUSH(snap, frag, scratchUsed);                        \
            _mgl_cnt = (frag) ? &(snap).fragment_op_count                        \
                              : &(snap).vertex_op_count;                         \
            _mgl_ops = (frag) ? (snap).fragment_ops : (snap).vertex_ops;         \
        }                                                                        \
        uint8_t *dst_ = (scratch) + (scratchUsed);                               \
        memcpy(dst_, src_, len_);                                                \
        (scratchUsed) += len_;                                                   \
        _mgl_ops[(*_mgl_cnt)++] =                                                \
            (MGLRenderBindingOp){1u, (uint32_t)(slot), 0, NULL, dst_,            \
                                 (uint32_t)len_};                                \
    } while (0)

@implementation MGLRenderer (Draw)

- (bool) bindVertexBuffersToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx
{
    static uint64_t s_vbindCallCount = 0;
    static double s_vbindLastCallTime = 0.0;
    static uint64_t s_vbindLastCallCount = 0;
    uint64_t vbindCall = ++s_vbindCallCount;
    double vbindStartSeconds = mglTraceNowSeconds();
    uint64_t vbindStartNS = mglTraceClockNS();
    mglLogLoopHeartbeat("vbind.loop",
                        vbindCall,
                        vbindStartSeconds,
                        &s_vbindLastCallTime,
                        &s_vbindLastCallCount,
                        0.25);

    BufferMap *map;
    Buffer *ptr;
    GLintptr offset;
    NSUInteger bindingIndex;
    bool isBaseBinding;
    bool anyBindingPresent[MAX_MAPPED_BUFFERS] = {false};
    bool baseBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    bool attribBindingReserved[MAX_MAPPED_BUFFERS] = {false};
    int attribBindingIndex[MAX_ATTRIBS];
    Program *activeProgram;
    VertexArray *vao;
    GLuint mapCount;

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL VBIND begin ctx=%p vao=%p owner=%p",
              ctx, ctx ? MGL_STATE(ctx)->vao : NULL,
              encCtx->render_encoder_owner);
    }

    if (!ctx || !mglBindingStateHasActiveEncoder(encCtx)) {
        NSLog(@"MGL VBIND skip: encoder/ctx nil");
        return false;
    }

    vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    if (!vao) {
        NSLog(@"MGL VBIND skip: vao nil/invalid");
        return false;
    }
    activeProgram = _tessellation.nativeTESActive
        ? _tessellation.nativeTESProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;


    const BOOL useVertexBindingSnapshot = YES;
    MGLRenderBindingSnapshot vbindSnapshot = {0};
    uint8_t vbindByteScratch[4096];
    size_t vbindByteScratchUsed = 0;

#define MGL_VBIND_FLUSH_SNAPSHOT() MGL_BIND_SNAP_FLUSH(vbindSnapshot, 0, vbindByteScratchUsed)
#define MGL_VBIND_COLLECT_BUFFER(slot, bufPtr, off) MGL_BIND_SNAP_COLLECT_BUFFER(vbindSnapshot, 0, vbindByteScratchUsed, slot, bufPtr, off)
#define MGL_VBIND_COLLECT_BYTES(slot, src, len) MGL_BIND_SNAP_COLLECT_BYTES(vbindSnapshot, 0, vbindByteScratch, vbindByteScratchUsed, sizeof(vbindByteScratch), slot, src, len)
#define MGL_VBIND_EMIT_BUFFER(slot, bufPtr, off) do { if (useVertexBindingSnapshot) { MGL_VBIND_COLLECT_BUFFER(slot, bufPtr, off); } else { mglBindingStateSetVertexBuffer(encCtx->render_encoder_owner, (__bridge id)(bufPtr), (off), (slot)); } } while (0)
#define MGL_VBIND_EMIT_BYTES(slot, src, len) do { if (useVertexBindingSnapshot) { MGL_VBIND_COLLECT_BYTES(slot, src, len); } else { mglBindingStateSetVertexBytes(encCtx->render_encoder_owner, (src), (len), (slot)); } } while (0)
#define MGL_VBIND_EMIT_CLEAR(slot) do { if (useVertexBindingSnapshot) { MGL_VBIND_COLLECT_BUFFER(slot, NULL, 0); } else { mglBindingStateSetVertexBuffer(encCtx->render_encoder_owner, nil, 0, (slot)); } } while (0)

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL VBIND vao=%p magic=0x%x", vao, vao->magic);
    }
    mapCount = MGL_STATE(ctx)->vertex_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        static uint64_t s_vbindMapCountOverflow = 0;
        uint64_t hit = ++s_vbindMapCountOverflow;
        if (hit <= 16ull || (hit % 4096ull) == 0ull) {
            NSLog(@"MGL WARNING: VBIND mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping (hit=%llu)",
                  mapCount, MAX_MAPPED_BUFFERS, (unsigned long long)hit);
        }
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        attribBindingIndex[i] = -1;
    }

    // Resolve attribute slot reservations first so base/resource bindings do not
    // overwrite shader-required vertex input slots.
    bool attribsEnabledByApp = (vao->enabled_attribs != 0u);
    GLuint reserveMaxAttribs = MAX_ATTRIBS;
    for (GLuint attrib = 0; attrib < reserveMaxAttribs; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            continue;
        }

        int mappedIndex = [self getVertexBufferIndexWithAttributeSet:(int)attrib];
        if (mappedIndex < 0 || mappedIndex >= (int)kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL ERROR: VBIND reserve attrib=%u unresolved mapping=%d", attrib, mappedIndex);
            continue;
        }

        attribBindingIndex[attrib] = mappedIndex;
        attribBindingReserved[mappedIndex] = true;
    }

    if (kMGLVerboseBindLogs) {
        for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
            BOOL enabled = attribsEnabledByApp && ((vao->enabled_attribs >> i) & 0x1u) != 0;
            MGLResolvedVertexAttribBinding resolved = {0};
            Buffer *attribBuffer = mglRendererResolveVertexAttribBinding(
                                       ctx, vao, i, __FUNCTION__, &resolved)
                ? resolved.buffer
                : NULL;
            NSLog(@"MGL VBIND attrib=%u en=%d buf=%u off=%lld rel=0x%llx stride=%u size=%u type=0x%x norm=%u div=%u bind=%u table=%d mtl=%p ever=%u written=[%lld,%lld)",
                  i, enabled ? 1 : 0,
                  attribBuffer ? attribBuffer->name : 0u,
                  (long long)(attribBuffer ? resolved.binding_offset
                                           : vao->attrib[i].binding_offset),
                  (unsigned long long)(uintptr_t)vao->attrib[i].relativeoffset,
                  (unsigned)(attribBuffer ? resolved.stride : vao->attrib[i].stride),
                  (unsigned)vao->attrib[i].size, (unsigned)vao->attrib[i].type,
                  (unsigned)vao->attrib[i].normalized,
                  (unsigned)(attribBuffer ? resolved.divisor : vao->attrib[i].divisor),
                  (unsigned)vao->attrib[i].buffer_bindingindex,
                  attribBuffer && resolved.uses_binding_table ? 1 : 0,
                  attribBuffer ? attribBuffer->data.mtl_data : NULL,
                  attribBuffer ? (unsigned)attribBuffer->ever_written : 0u,
                  attribBuffer ? (long long)attribBuffer->written_min : 0ll,
                  attribBuffer ? (long long)attribBuffer->written_max : 0ll);
        }
    }

    for (int i = 0; i < (int)mapCount; i++) {
        map = &MGL_STATE(ctx)->vertex_buffer_map_list.buffers[i];
        ptr = mglRendererGetValidatedBuffer(ctx, map->buf, __FUNCTION__, (NSUInteger)i);
        offset = map->offset;
        isBaseBinding = mglRenderBufferMapIsBaseBinding(map->attribute_mask) != 0;
        GLuint glBindingIndex = map->buffer_base_index;
        NSInteger metalResolved = map->has_metal_binding
            ? (NSInteger)map->metal_binding_index
            : mglRendererGetProgramMetalBufferIndexForStage(ctx, vertexStage, glBindingIndex);

        NSUInteger reflectedRequiredBytes = 0;
        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            reflectedRequiredBytes = map->has_metal_binding
                ? mglRendererGetProgramBindingRequiredSize(
                      ctx, vertexStage, (int)map->resource_type,
                      (int)map->resource_index)
                : mglRendererGetProgramBindingRequiredSizeForStage(
                      ctx, vertexStage, glBindingIndex);
        }
        uint64_t visibleCpu = ptr
            ? (uint64_t)mglBufferMapVisibleBackingBytes(map, ptr->data.buffer_size)
            : 0u;

        MGLStageBufferBindInput bin = {0};
        mglBindingStateFillStageBindInput(
            &bin, /*is_fragment=*/0, MGL_SB_PHASE_PRE_MTL, map, ptr,
            isBaseBinding ? 1 : 0, /*attrib_reserved=*/0,
            (uint32_t)kMGLMaxMetalVertexBufferCount,
            (uint32_t)reflectedRequiredBytes, visibleCpu,
            map ? mglBufferMapVisibleSize(map) : 0,
            _tessellation.nativeTESActive ? 1 : 0);
        /* Provisional slot for attrib-reserved check before plan. */
        if (isBaseBinding &&
            mglRenderBufferSlotInRange((int32_t)metalResolved,
                                       (uint32_t)kMGLMaxMetalVertexBufferCount) &&
            attribBindingReserved[(NSUInteger)metalResolved]) {
            bin.attrib_slot_reserved = 1;
        }
        if (isBaseBinding && map->has_metal_binding) {
            bin.metal_binding_index = (int32_t)metalResolved;
            bin.has_metal_binding = 1;
        } else if (isBaseBinding) {
            bin.metal_binding_index = (int32_t)metalResolved;
            bin.has_metal_binding = 1; /* resolved program slot */
        }

        MGLStageBufferBindPlan plan = {0};
        if (mglBindingStagePlanMapEntry(&bin, &plan) != 0) {
            continue;
        }
        bindingIndex = plan.metal_slot;
        if (plan.mark_base_present && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            baseBindingPresent[glBindingIndex] = true;
        }

        if (plan.action == MGL_SB_ACTION_SKIP) {
            continue;
        }
        if (plan.action == MGL_SB_ACTION_CLEAR) {
            MGL_VBIND_EMIT_CLEAR(bindingIndex);
            mglRenderBindingClearVertexBuffer(_bindingStateOwner,
                                              (uint32_t)bindingIndex);
            continue;
        }
        if (plan.action == MGL_SB_ACTION_INLINE_BYTES) {
            uint8_t padded[kMGLStageBindingStackScratchSize];
            const void *inlineBytes = mglBindingStageInlineBytesSrc(
                padded, (uint32_t)sizeof(padded),
                (const void *)((const uint8_t *)bin.cpu_ptr + plan.inline_src_offset),
                plan.inline_visible, plan.inline_length);
            MGL_VBIND_EMIT_BYTES(bindingIndex, inlineBytes, plan.inline_length);
            if (plan.invalidate_last_bound) {
                [self invalidateLastBoundVertexBufferAtIndex:bindingIndex];
            }
            if (plan.mark_any_present) {
                anyBindingPresent[bindingIndex] = true;
            }
            if (plan.clear_cpu_dirty_if_no_mtl && ptr && !ptr->data.mtl_data) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            if (plan.clear_cpu_dirty && ptr) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            continue;
        }

        /* NEED_MTL → ensure → POST plan */
        if (plan.action == MGL_SB_ACTION_NEED_MTL) {
            if (!ptr->data.mtl_data) {
                [self bindMTLBuffer:ptr];
            } else if (mglRenderBufferHasCPUDirty(ptr->data.dirty_bits)) {
                [self updateDirtyBuffer:ptr];
            }
            id buffer = nil;
            if (ptr->data.mtl_data &&
                mglRenderMetalDataPointerUsable(ptr->data.mtl_data)) {
                buffer = (__bridge id)(ptr->data.mtl_data);
            }
            NSUInteger metalLen = buffer ? mglBindingStateBufferLength(buffer) : 0u;
            NSUInteger availableBytes = buffer
                ? mglBufferMapVisibleBackingBytes(map, metalLen)
                : 0u;
            bin.phase = MGL_SB_PHASE_POST_MTL;
            bin.has_mtl_data = ptr->data.mtl_data ? 1 : 0;
            bin.mtl_ptr = ptr->data.mtl_data;
            bin.mtl_usable =
                mglRenderMetalDataPointerUsable(ptr->data.mtl_data) ? 1 : 0;
            bin.metal_len = (uint64_t)metalLen;
            bin.visible_mtl = (uint64_t)availableBytes;
            bin.binding_state_valid =
                mglBindingStateIsValid(_bindingStateOwner) ? 1 : 0;
            bin.buffer_matches =
                buffer &&
                        mglBindingStateBufferMatches(
                            _bindingStateOwner, MGL_RENDER_BINDING_STAGE_VERTEX,
                            (__bridge void *)buffer, (NSUInteger)offset,
                            (uint32_t)plan.metal_slot)
                    ? 1
                    : 0;
            if (mglBindingStagePlanMapEntry(&bin, &plan) != 0) {
                continue;
            }
            bindingIndex = plan.metal_slot;

            if (plan.action == MGL_SB_ACTION_CLEAR) {
                MGL_VBIND_EMIT_CLEAR(bindingIndex);
                mglRenderBindingClearVertexBuffer(_bindingStateOwner,
                                                  (uint32_t)bindingIndex);
                continue;
            }
            if (plan.action == MGL_SB_ACTION_ISOLATE) {
                id isolated =
                    [self isolatedStageBindingBufferForMap:map
                                                     source:buffer
                                             requiredLength:plan.required_bytes];
                if (!isolated) {
                    NSLog(@"MGL WARNING: VBIND failed to isolate undersized buffer=%u slot=%lu required=%u available=%u",
                          ptr->name, (unsigned long)bindingIndex,
                          plan.required_bytes, plan.available_bytes);
                    MGL_VBIND_EMIT_CLEAR(bindingIndex);
                    mglRenderBindingClearVertexBuffer(_bindingStateOwner,
                                                      (uint32_t)bindingIndex);
                    continue;
                }
                if (plan.needs_copy_back && buffer && plan.available_bytes > 0 &&
                    ![self recordStageBindingCopyBack:&_tessellation.nativeTESCopyBacks
                                               atIndex:bindingIndex
                                             temporary:isolated
                                           destination:buffer
                                     destinationBuffer:ptr
                                    destinationOffset:(NSUInteger)offset
                                                length:plan.available_bytes]) {
                    return false;
                }
                MGL_VBIND_EMIT_BUFFER(bindingIndex, (__bridge void *)isolated, 0);
                if (plan.needs_flush_snapshot) {
                    MGL_VBIND_FLUSH_SNAPSHOT();
                }
                mglRenderBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)isolated, 0,
                    (uint32_t)bindingIndex);
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                anyBindingPresent[bindingIndex] = true;
                continue;
            }
            if (plan.action == MGL_SB_ACTION_SKIP_MATCHED) {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                anyBindingPresent[bindingIndex] = true;
                continue;
            }
            if (plan.action == MGL_SB_ACTION_BIND_BUFFER) {
                MGL_VBIND_EMIT_BUFFER(bindingIndex, (__bridge void *)buffer,
                                      (NSUInteger)plan.bind_offset);
                mglRenderBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)buffer,
                    (NSUInteger)plan.bind_offset, (uint32_t)bindingIndex);
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                mglNoteBufferEncoded(ptr);
                anyBindingPresent[bindingIndex] = true;
                continue;
            }
        }
    }


    if (useVertexBindingSnapshot && vbindSnapshot.vertex_op_count > 0) {
        mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
            encCtx->render_encoder_owner, &vbindSnapshot, NULL, 0);
        vbindSnapshot = (MGLRenderBindingSnapshot){0};
        vbindByteScratchUsed = 0;
    }

    if (![self bindVertexAttributesFromVAO:vao
                              activeProgram:activeProgram
                        attribsEnabledByApp:attribsEnabledByApp
                        attribBindingIndex:attribBindingIndex
                          anyBindingPresent:anyBindingPresent
                              encodeContext:encCtx
                             bindingSnapshot:&vbindSnapshot
                                 useSnapshot:useVertexBindingSnapshot]) {
        return false;
    }

    if (mapCount > 0) {
        [self bindVertexFallbackBuffersToCurrentRenderEncoder:activeProgram
                                          anyBindingPresent:anyBindingPresent
                                          baseBindingPresent:baseBindingPresent
                                              encodeContext:encCtx
                                           bindingSnapshot:&vbindSnapshot
                                               useSnapshot:useVertexBindingSnapshot];
    }

    [self bindPointSizeParamsIfNeeded:anyBindingPresent
                        encodeContext:encCtx
                      bindingSnapshot:&vbindSnapshot
                          byteScratch:vbindByteScratch
                        byteScratchUsed:&vbindByteScratchUsed
                    byteScratchCapacity:sizeof(vbindByteScratch)
                           useSnapshot:useVertexBindingSnapshot];

    /* Snapshot the final Metal slot set only after VAO, fallback, and
     * generated point-size bindings have all updated anyBindingPresent.  The
     * cache mask accumulates for the encoder lifetime because older dedup
     * entries remain valid until invalidateLastBoundState. */
    uint32_t boundVertexBufferMask = mglBindingStageBuildPresentMask(
        (const uint8_t *)anyBindingPresent, (uint32_t)kMGLMaxBufferSlots);
    mglRenderBindingOrVertexBufferMask(_bindingStateOwner,
                                          boundVertexBufferMask);

    if (mglEnvFlagEnabled("MGL_TRACE_SPARSE_BINDING")) {
        static uint64_t s_vbind_trace_count = 0;
        if ((++s_vbind_trace_count % 500) == 1) {
            NSLog(@"MGL SPARSE VBIND: mask=0x%x activeSlots=%u/31",
                  boundVertexBufferMask,
                  mglBindingStageCountPresent((const uint8_t *)anyBindingPresent,
                                              (uint32_t)kMGLMaxBufferSlots));
        }
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(vbindCall)) {
        NSUInteger boundSlots = mglBindingStageCountPresent(
            (const uint8_t *)anyBindingPresent,
            (uint32_t)kMGLMaxMetalVertexBufferCount);
        NSUInteger reservedSlots = mglBindingStageCountPresent(
            (const uint8_t *)attribBindingReserved,
            (uint32_t)kMGLMaxMetalVertexBufferCount);
        NSUInteger baseSlots = mglBindingStageCountPresent(
            (const uint8_t *)baseBindingPresent, (uint32_t)MAX_BINDABLE_BUFFERS);
        mglTraceLogNSString(@"MGL TRACE vbind.end call=%llu mapCount=%u boundSlots=%lu reservedSlots=%lu baseSlots=%lu elapsed=%.1fus",
              (unsigned long long)vbindCall,
              (unsigned)mapCount,
              (unsigned long)boundSlots,
              (unsigned long)reservedSlots,
              (unsigned long)baseSlots,
              (mglTraceClockNS() - vbindStartNS) / 1000.0);
    }

    /* Mark the dedup cache as valid for the current encoder so subsequent
     * binds can be skipped when the resource and offset are unchanged. */
    mglRenderBindingSetValid(_bindingStateOwner, 1);
    return true;
}


- (bool)bindVertexAttributesFromVAO:(VertexArray *)vao
                      activeProgram:(Program *)activeProgram
                attribsEnabledByApp:(bool)attribsEnabledByApp
                attribBindingIndex:(int *)attribBindingIndex
                  anyBindingPresent:(bool *)anyBindingPresent
                      encodeContext:(const MGLEncodeContext *)encCtx
                     bindingSnapshot:(MGLRenderBindingSnapshot *)bindingSnapshot
                         useSnapshot:(BOOL)useSnapshot
{
    NSUInteger bindingIndex;


    MGLRenderBindingSnapshot *vattrSnapshot = bindingSnapshot;
    const BOOL vattrUseSnapshot = useSnapshot && vattrSnapshot != NULL;
    size_t vattrScratchDummy = 0;
#define MGL_VATTR_FLUSH_SNAPSHOT() do { if (vattrUseSnapshot) MGL_BIND_SNAP_FLUSH(*vattrSnapshot, 0, vattrScratchDummy); } while (0)
#define MGL_VATTR_EMIT_BUFFER(slot, bufPtr, off) do { if (vattrUseSnapshot) { MGL_BIND_SNAP_COLLECT_BUFFER(*vattrSnapshot, 0, vattrScratchDummy, slot, bufPtr, off); } else { mglBindingStateSetVertexBuffer(encCtx->render_encoder_owner, (__bridge id)(bufPtr), (off), (slot)); } } while (0)

    // Attribute bindings must use the same mapping as generateVertexDescriptorState.
    // Do this pass directly from the VAO so pipeline creation does not depend on map list timing.
    // O3.3: orchestration → mglBindingStagePlanAttribEntry; ObjC = plan + setVertexBuffer.
    GLuint maxAttribs = MAX_ATTRIBS;
    for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
        BOOL usesCurrentValue = mglRendererVertexAttribUsesCurrentValue(vao, attrib);
        MGLResolvedVertexAttribBinding resolved = {0};
        bool hasAttribBinding = mglRendererResolveVertexAttribBinding(ctx,
                                                                      vao,
                                                                      attrib,
                                                                      __FUNCTION__,
                                                                      &resolved);
        int mappedIndex = (attrib < MAX_ATTRIBS) ? attribBindingIndex[attrib] : -1;
        uint32_t plannedFormat = 0u;
        int effectiveNormalized = 0;
        int conversionKind = MGL_ATTRIB_CONV_NONE;

        MGLAttribBindInput ain = {0};
        ain.phase = MGL_ATTR_PHASE_SELECT;
        ain.program_uses_attrib =
            mglRendererProgramUsesVertexAttrib(activeProgram, attrib) ? 1 : 0;
        ain.uses_current_value = usesCurrentValue ? 1 : 0;
        ain.has_attrib_binding = hasAttribBinding ? 1 : 0;
        ain.mapped_index = mappedIndex;
        ain.max_metal_slots = (uint32_t)kMGLMaxMetalVertexBufferCount;
        if (hasAttribBinding && resolved.buffer) {
            ain.offsets_valid = mglRenderAttribOffsetsValid(
                                    resolved.binding_offset,
                                    resolved.relativeoffset)
                                    ? 1
                                    : 0;
            int64_t attrOffset = 0, attrSpan = 0, attrEnd = 0;
            ain.span_status = mglRenderPlanVertexAttribSpan(
                (int64_t)resolved.binding_offset,
                (int64_t)resolved.relativeoffset,
                (uint32_t)resolved.attrib->type,
                (uint32_t)resolved.attrib->size, &attrOffset, &attrSpan,
                &attrEnd);
            if (kMGLVerboseBindLogs &&
                mglRenderAttribWrittenRangeTracked(resolved.buffer->written_min,
                                                   resolved.buffer->written_max) &&
                mglRenderAttribOutsideWrittenRange(
                    attrOffset, attrEnd, resolved.buffer->written_min,
                    resolved.buffer->written_max)) {
                static uint64_t s_vbindWrittenRangeWarningCount = 0;
                uint64_t hit = ++s_vbindWrittenRangeWarningCount;
                if (hit <= 16ull || (hit % 4096ull) == 0ull) {
                    NSLog(@"MGL VBIND WARNING draw: attrib=%u buffer=%u attrRange=[%lld,%lld) outside written range [%lld,%lld) (type=0x%x size=%u) - allowing, Sodium arena buffers use sub-ranges hit=%llu",
                          attrib, resolved.buffer->name, (long long)attrOffset,
                          (long long)attrEnd,
                          (long long)resolved.buffer->written_min,
                          (long long)resolved.buffer->written_max,
                          (unsigned)resolved.attrib->type,
                          (unsigned)resolved.attrib->size,
                          (unsigned long long)hit);
                }
            }
            MGLShaderResource *attrRes =
                mglRendererProgramVertexAttribResource(activeProgram, attrib);
            GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
            int needsConversion = 0;
            mglRenderPlanVertexAttribFormat(
                (uint32_t)resolved.attrib->type,
                (uint32_t)resolved.attrib->size,
                resolved.attrib->integer ? 1 : 0,
                resolved.attrib->normalized ? 1 : 0,
                mglRendererVertexAttribIsColorInput(activeProgram, attrib) ? 1
                                                                           : 0,
                (uint32_t)shaderGlType, &plannedFormat, &needsConversion,
                &effectiveNormalized, &conversionKind);
            (void)needsConversion;
            ain.conversion_kind = conversionKind;
            ain.already_present =
                (mappedIndex >= 0 &&
                 mappedIndex < (int)kMGLMaxMetalVertexBufferCount &&
                 anyBindingPresent[mappedIndex])
                    ? 1
                    : 0;
            ain.binding_offset = (uint64_t)resolved.binding_offset;
            ain.absolute_vertex_offsets =
                _batching.absoluteVertexBindingOffsets ? 1 : 0;
        }

        MGLAttribBindPlan plan = {0};
        if (mglBindingStagePlanAttribEntry(&ain, &plan) != 0) {
            continue;
        }
        if (plan.action == MGL_ATTR_ACTION_SKIP ||
            plan.action == MGL_ATTR_ACTION_SKIP_ALREADY) {
            if (plan.reason == MGL_ATTR_REASON_BAD_MAP) {
                NSLog(@"MGL ERROR: VBIND attrib=%u unresolved mapping=%d", attrib,
                      mappedIndex);
            } else if (plan.reason == MGL_ATTR_REASON_NO_BINDING &&
                       !usesCurrentValue && !hasAttribBinding) {
                /* disabled attrib */
            } else if (plan.reason == MGL_ATTR_REASON_NO_BINDING) {
                NSLog(@"MGL VBIND skip attrib=%u: enabled but buffer is invalid",
                      attrib);
            }
            continue;
        }
        if (plan.action == MGL_ATTR_ACTION_BLOCK) {
            if (plan.reason == MGL_ATTR_REASON_BAD_OFFSET) {
                NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u negative bindingOffset=%lld relativeOffset=%lld",
                      attrib, resolved.buffer->name,
                      (long long)resolved.binding_offset,
                      (long long)resolved.relativeoffset);
            } else {
                NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u attr span overflow (type=0x%x size=%u)",
                      attrib, resolved.buffer->name,
                      (unsigned)resolved.attrib->type,
                      (unsigned)resolved.attrib->size);
            }
            MGL_VATTR_FLUSH_SNAPSHOT();
            return false;
        }

        bindingIndex = (NSUInteger)plan.metal_slot;

        if (plan.action == MGL_ATTR_ACTION_CURRENT) {
            uint8_t poolValues[MAX_ATTRIBS][16];
            memset(poolValues, 0, sizeof(poolValues));
            for (GLuint a = 0; a < (GLuint)MAX_ATTRIBS; a++) {
                uint8_t tmp[16] = {0};
                NSUInteger built = mglRendererBuildCurrentVertexAttribBytes(
                    ctx, a, &vao->attrib[a], tmp);
                if (built == 0u || built > 16u) {
                    continue;
                }
                memcpy(poolValues[a], tmp, 16u);
            }
            id currentAttribBuffer = (__bridge id)
                mglRendererBackendGetPackedCurrentAttribBuffer(
                    _backend, poolValues, (uint32_t)sizeof(poolValues),
                    kMGLCurrentAttribRepeatCount);
            if (currentAttribBuffer == nil) {
                NSUInteger poolBytes = (NSUInteger)MAX_ATTRIBS *
                                       kMGLCurrentAttribPoolStride;
                NSMutableData *pool = [NSMutableData dataWithLength:poolBytes];
                if (!pool) {
                    NSLog(@"MGL VBIND skip attrib=%u: failed to allocate packed "
                          @"current vertex attrib pool", attrib);
                    continue;
                }
                mglRenderPackCurrentAttribPool(
                    (const uint8_t *)poolValues, (uint32_t)MAX_ATTRIBS,
                    (uint8_t *)pool.mutableBytes, (uint64_t)poolBytes,
                    kMGLCurrentAttribRepeatCount, kMGLCurrentAttribValueBytes);
                currentAttribBuffer = mglBindingStateCreateBufferWithBytes(
                    _device, pool.bytes, pool.length,
                    MGL_BINDING_RESOURCE_STORAGE_SHARED);
                if (!currentAttribBuffer) {
                    NSLog(@"MGL VBIND skip attrib=%u: failed to allocate packed "
                          @"current vertex attrib Metal buffer", attrib);
                    continue;
                }
                if (mglRendererBackendSetPackedCurrentAttribBuffer(
                        _backend, poolValues, (uint32_t)sizeof(poolValues),
                        kMGLCurrentAttribRepeatCount,
                        (__bridge void *)currentAttribBuffer) != 0) {
                    NSLog(@"MGL VBIND skip attrib=%u: failed to retain packed "
                          @"current vertex attrib cache buffer", attrib);
                    continue;
                }
            }
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_BINDING_STAGE_VERTEX,
                    (__bridge void *)currentAttribBuffer, 0,
                    (uint32_t)bindingIndex)) {
                MGL_VATTR_EMIT_BUFFER(bindingIndex,
                                      (__bridge void *)currentAttribBuffer, 0);
                mglRenderBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)currentAttribBuffer, 0,
                    (uint32_t)bindingIndex);
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
            anyBindingPresent[bindingIndex] = true;
            static uint64_t s_traceFileCurrentAttribBindLogs = 0;
            if (mglProgramNeedsTraceLog(activeProgram) &&
                mglShouldLogTraceFileBindingForProgram(
                    activeProgram, &s_traceFileCurrentAttribBindLogs)) {
                MGLShaderResource *resource =
                    mglRendererProgramVertexAttribResource(activeProgram, attrib);
                mglTraceLog("VATTR_BIND_CURRENT_PACKED program=%u attrib=%u resource=%s loc=%u metalSlot=%lu poolOffset=%lu valueF=(%.6f,%.6f,%.6f,%.6f)",
                            activeProgram ? (unsigned)activeProgram->name : 0u,
                            (unsigned)attrib,
                            resource && resource->name ? resource->name : "(unknown)",
                            resource ? (unsigned)resource->location : 0xffffffffu,
                            (unsigned long)bindingIndex,
                            (unsigned long)((NSUInteger)attrib *
                                            kMGLCurrentAttribPoolStride),
                            MGL_STATE(ctx)->current_vertex_attrib[attrib].f[0],
                            MGL_STATE(ctx)->current_vertex_attrib[attrib].f[1],
                            MGL_STATE(ctx)->current_vertex_attrib[attrib].f[2],
                            MGL_STATE(ctx)->current_vertex_attrib[attrib].f[3]);
            }
            continue;
        }

        Buffer *attribBuffer = resolved.buffer;
        const VertexAttrib *attribState = resolved.attrib;
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL VBIND attrib map attrib=%u -> index=%lu buffer=%u bindingOffset=%lld table=%d",
                  attrib, (unsigned long)bindingIndex,
                  (unsigned)attribBuffer->name,
                  (long long)resolved.binding_offset,
                  resolved.uses_binding_table ? 1 : 0);
        }

        if (plan.action == MGL_ATTR_ACTION_CONVERT) {
            MGLShaderResource *convRes =
                mglRendererProgramVertexAttribResource(activeProgram, attrib);
            BOOL integerConvDstIsInt = mglRenderIntegerAttribDstIsInt(
                convRes ? convRes->gl_type : 0u) != 0;
            NSUInteger convertedStride = 0;
            id convertedBuffer =
                [self convertedVertexBufferForAttribKind:ain.conversion_kind
                                                  source:attribBuffer
                                                resolved:&resolved
                                                    size:attribState->size
                                                    type:attribState->type
                                              normalized:attribState->normalized
                                               dstIsInt:integerConvDstIsInt
                                               outStride:&convertedStride];
            if (!convertedBuffer) {
                NSLog(@"MGL VBIND skip attrib=%u buffer=%u: failed to convert vertex attrib kind=%d type=0x%x",
                      attrib, attribBuffer->name, ain.conversion_kind,
                      (unsigned)attribState->type);
                continue;
            }
            (void)convertedStride;
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_BINDING_STAGE_VERTEX,
                    (__bridge void *)convertedBuffer, 0,
                    (uint32_t)bindingIndex)) {
                MGL_VATTR_EMIT_BUFFER(bindingIndex,
                                      (__bridge void *)convertedBuffer, 0);
                MGL_VATTR_FLUSH_SNAPSHOT();
                mglRenderBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)convertedBuffer, 0,
                    (uint32_t)bindingIndex);
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
            anyBindingPresent[bindingIndex] = true;
            continue;
        }

        /* NEED_MTL → ensure → POST plan */
        if (!attribBuffer->data.mtl_data) {
            [self bindMTLBuffer:attribBuffer];
        }
        if (!attribBuffer->data.mtl_data) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: no Metal backing", attrib,
                  attribBuffer->name);
            continue;
        }
        if (!mglRenderMetalDataPointerUsable(attribBuffer->data.mtl_data)) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: suspicious mtl_data=%p",
                  attrib, attribBuffer->name, attribBuffer->data.mtl_data);
            continue;
        }
        id attribMetalBuffer = (__bridge id)(attribBuffer->data.mtl_data);
        if (!attribMetalBuffer) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: Metal bridge failed",
                  attrib, attribBuffer->name);
            continue;
        }
        NSUInteger attribBindingOffset = (NSUInteger)resolved.binding_offset;
        ain.phase = MGL_ATTR_PHASE_POST_MTL;
        ain.has_mtl_data = 1;
        ain.mtl_usable = 1;
        ain.metal_len = (uint64_t)mglBindingStateBufferLength(attribMetalBuffer);
        ain.binding_state_valid =
            mglBindingStateIsValid(_bindingStateOwner) ? 1 : 0;
        uint64_t provisionalOff = mglRenderVertexMetalBindOffset(
            ain.absolute_vertex_offsets, ain.binding_offset);
        ain.buffer_matches =
            mglBindingStateBufferMatches(
                _bindingStateOwner, MGL_RENDER_BINDING_STAGE_VERTEX,
                (__bridge void *)attribMetalBuffer, provisionalOff,
                (uint32_t)bindingIndex)
                ? 1
                : 0;
        if (mglBindingStagePlanAttribEntry(&ain, &plan) != 0) {
            continue;
        }
        if (plan.action == MGL_ATTR_ACTION_SKIP) {
            if (plan.reason == MGL_ATTR_REASON_BAD_MTL) {
                NSLog(@"MGL VBIND skip attrib=%u buffer=%u: bindingOffset=%lu >= metalLen=%lu",
                      attrib, attribBuffer->name,
                      (unsigned long)attribBindingOffset,
                      (unsigned long)ain.metal_len);
            }
            continue;
        }
        NSUInteger metalBindOffset = (NSUInteger)plan.metal_bind_offset;
        if (plan.action == MGL_ATTR_ACTION_SKIP_MATCHED) {
            MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            anyBindingPresent[bindingIndex] = true;
            continue;
        }
        MGL_VATTR_EMIT_BUFFER(bindingIndex, (__bridge void *)attribMetalBuffer,
                              metalBindOffset);
        mglRenderBindingUpdateVertexBuffer(
            _bindingStateOwner, (__bridge void *)attribMetalBuffer,
            metalBindOffset, (uint32_t)bindingIndex);
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        mglNoteBufferEncoded(attribBuffer);
        anyBindingPresent[bindingIndex] = true;
        static uint64_t s_traceFileVertexAttribBindLogs = 0;
        if (mglProgramNeedsTraceLog(activeProgram) &&
            mglShouldLogTraceFileBindingForProgram(
                activeProgram, &s_traceFileVertexAttribBindLogs)) {
            MGLShaderResource *resource =
                mglRendererProgramVertexAttribResource(activeProgram, attrib);
            GLboolean effectiveNormalizedLog = effectiveNormalized != 0;
            uint32_t format = mglRenderAttribFormatOrFallback(
                plannedFormat, (uint32_t)attribState->type,
                (uint32_t)attribState->size,
                effectiveNormalizedLog ? 1 : 0);
            mglTraceLog("VATTR_BIND program=%u attrib=%u resource=%s loc=%u metalSlot=%lu glBuffer=%u bindingIndex=%u bindingOffset=%lu relOffset=%lld stride=%u size=%u type=0x%x normalized=%u/%u divisor=%u table=%d metalLen=%lu format=%lu(%s)",
                        activeProgram ? (unsigned)activeProgram->name : 0u,
                        (unsigned)attrib,
                        resource && resource->name ? resource->name : "(unknown)",
                        resource ? (unsigned)resource->location : 0xffffffffu,
                        (unsigned long)bindingIndex,
                        (unsigned)attribBuffer->name,
                        (unsigned)resolved.binding_index,
                        (unsigned long)attribBindingOffset,
                        (long long)resolved.relativeoffset,
                        (unsigned)resolved.stride,
                        (unsigned)attribState->size,
                        (unsigned)attribState->type,
                        (unsigned)attribState->normalized,
                        (unsigned)effectiveNormalized,
                        (unsigned)resolved.divisor,
                        resolved.uses_binding_table ? 1 : 0,
                        (unsigned long)ain.metal_len,
                        (unsigned long)format, mglVertexFormatName(format));
        }
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL SET VERTEX ATTRIB BUFFER index=%lu glName=%u offset=%lu available=%lu attrib=%u stride=%u attrOffset=0x%llx mtl=%p",
                  (unsigned long)bindingIndex, attribBuffer->name,
                  (unsigned long)attribBindingOffset,
                  (unsigned long)ain.metal_len, attrib,
                  (unsigned)resolved.stride,
                  (unsigned long long)(uintptr_t)resolved.relativeoffset,
                  attribBuffer->data.mtl_data);
        }
    }


    MGL_VATTR_FLUSH_SNAPSHOT();
#undef MGL_VATTR_EMIT_BUFFER
#undef MGL_VATTR_FLUSH_SNAPSHOT
    return true;
}


- (void)bindVertexFallbackBuffersToCurrentRenderEncoder:(Program *)activeProgram
                                     anyBindingPresent:(bool *)anyBindingPresent
                                     baseBindingPresent:(bool *)baseBindingPresent
                                         encodeContext:(const MGLEncodeContext *)encCtx
                                     bindingSnapshot:(MGLRenderBindingSnapshot *)bindingSnapshot
                                         useSnapshot:(BOOL)useSnapshot
{
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    [self bindStageFallbackBuffersForStage:vertexStage
                           isFragmentStage:NO
                                   program:activeProgram
                         anyBindingPresent:anyBindingPresent
                        baseBindingPresent:baseBindingPresent
                             encodeContext:encCtx
                           bindingSnapshot:bindingSnapshot
                               useSnapshot:useSnapshot
                              maxMetalSlots:kMGLMaxMetalVertexBufferCount
                           enableAllSlotFill:kMGLEnableVertexAllSlotFallback];
}


- (void)bindPointSizeParamsIfNeeded:(bool *)anyBindingPresent
                      encodeContext:(const MGLEncodeContext *)encCtx
                    bindingSnapshot:(MGLRenderBindingSnapshot *)bindingSnapshot
                        byteScratch:(uint8_t *)byteScratch
                      byteScratchUsed:(size_t *)byteScratchUsed
                  byteScratchCapacity:(size_t)byteScratchCapacity
                         useSnapshot:(BOOL)useSnapshot
{
    BOOL needsPointSizeParams = NO;


    MGLRenderBindingSnapshot *vpointSnapshot = bindingSnapshot;
    uint8_t *vpointByteScratch = byteScratch;
    size_t *vpointByteScratchUsed = byteScratchUsed;
    const BOOL vpointUseSnapshot =
        useSnapshot && vpointSnapshot != NULL && vpointByteScratch != NULL &&
        vpointByteScratchUsed != NULL;
#define MGL_VPS_FLUSH_SNAPSHOT() do { if (vpointUseSnapshot) MGL_BIND_SNAP_FLUSH(*vpointSnapshot, 0, *vpointByteScratchUsed); } while (0)
#define MGL_VPS_EMIT_BYTES(slot, src, len) do { if (vpointUseSnapshot) { MGL_BIND_SNAP_COLLECT_BYTES(*vpointSnapshot, 0, vpointByteScratch, *vpointByteScratchUsed, byteScratchCapacity, slot, src, len); } else { mglBindingStateSetVertexBytes(encCtx->render_encoder_owner, (src), (len), (slot)); } } while (0)
    int pointSizeStages[] = { _VERTEX_SHADER, _TESS_EVALUATION_SHADER, _GEOMETRY_SHADER };
    for (NSUInteger ps = 0; ps < sizeof(pointSizeStages) / sizeof(pointSizeStages[0]); ps++) {
        Program *pointProgram = mglResolveProgramForStageFromState(ctx, pointSizeStages[ps]);
        if (!pointProgram) continue;
        if (pointProgram->uses_point_size_params) {
            needsPointSizeParams = YES;
            break;
        }
    }
    if (needsPointSizeParams) {
        float pointSizeParams[2] = {
            ctx && MGL_STATE(ctx)->var.point_size > 0.0f ? MGL_STATE(ctx)->var.point_size : 1.0f,
            ctx && MGL_STATE(ctx)->caps.program_point_size ? 1.0f : 0.0f
        };
        MGL_VPS_EMIT_BYTES(kMGLPointSizeParamBufferIndex, pointSizeParams,
                          sizeof(pointSizeParams));
        [self invalidateLastBoundVertexBufferAtIndex:kMGLPointSizeParamBufferIndex];
        anyBindingPresent[kMGLPointSizeParamBufferIndex] = true;
    }


    MGL_VPS_FLUSH_SNAPSHOT();
#undef MGL_VPS_EMIT_BYTES
#undef MGL_VPS_FLUSH_SNAPSHOT
}


- (bool) bindFragmentBuffersToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx
{
    static uint64_t s_fbindCallCount = 0;
    static double s_fbindLastCallTime = 0.0;
    static uint64_t s_fbindLastCallCount = 0;
    uint64_t fbindCall = ++s_fbindCallCount;
    double fbindStartSeconds = mglTraceNowSeconds();
    uint64_t fbindStartNS = mglTraceClockNS();
    mglLogLoopHeartbeat("fbind.loop",
                        fbindCall,
                        fbindStartSeconds,
                        &s_fbindLastCallTime,
                        &s_fbindLastCallCount,
                        0.25);

    GLuint mapCount;
    BufferMap *map;
    Buffer *ptr;
    GLintptr offset;
    NSUInteger bindingIndex;
    bool isBaseBinding;
    bool anyBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    bool baseBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    Program *activeProgram = NULL;

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL FBIND begin ctx=%p owner=%p", ctx,
              encCtx->render_encoder_owner);
    }

    if (!ctx || !mglBindingStateHasActiveEncoder(encCtx)) {
        NSLog(@"MGL FBIND skip: ctx/encoder nil");
        return false;
    }
    activeProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);


    const BOOL useBindingSnapshot = YES;
    MGLRenderBindingSnapshot snapshot = {0};
    uint8_t fbindByteScratch[4096];
    size_t fbindByteScratchUsed = 0;

#define MGL_FBIND_FLUSH_SNAPSHOT() MGL_BIND_SNAP_FLUSH(snapshot, 1, fbindByteScratchUsed)
#define MGL_FBIND_COLLECT_BUFFER(slot, bufPtr, off) MGL_BIND_SNAP_COLLECT_BUFFER(snapshot, 1, fbindByteScratchUsed, slot, bufPtr, off)
#define MGL_FBIND_COLLECT_BYTES(slot, src, len) MGL_BIND_SNAP_COLLECT_BYTES(snapshot, 1, fbindByteScratch, fbindByteScratchUsed, sizeof(fbindByteScratch), slot, src, len)
#define MGL_FBIND_EMIT_BUFFER(slot, bufPtr, off) do { if (useBindingSnapshot) { MGL_FBIND_COLLECT_BUFFER(slot, bufPtr, off); } else { mglBindingStateSetFragmentBuffer(encCtx->render_encoder_owner, (__bridge id)(bufPtr), (off), (slot)); } } while (0)
#define MGL_FBIND_EMIT_BYTES(slot, src, len) do { if (useBindingSnapshot) { MGL_FBIND_COLLECT_BYTES(slot, src, len); } else { mglBindingStateSetFragmentBytes(encCtx->render_encoder_owner, (src), (len), (slot)); } } while (0)
#define MGL_FBIND_EMIT_CLEAR(slot) do { if (useBindingSnapshot) { MGL_FBIND_COLLECT_BUFFER(slot, NULL, 0); } else { mglBindingStateSetFragmentBuffer(encCtx->render_encoder_owner, nil, 0, (slot)); } } while (0)

    mapCount = MGL_STATE(ctx)->fragment_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        static uint64_t s_fbindMapCountOverflow = 0;
        uint64_t hit = ++s_fbindMapCountOverflow;
        if (hit <= 16ull || (hit % 4096ull) == 0ull) {
            NSLog(@"MGL WARNING: FBIND mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping (hit=%llu)",
                  mapCount, MAX_MAPPED_BUFFERS, (unsigned long long)hit);
        }
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++) {
        map = &MGL_STATE(ctx)->fragment_buffer_map_list.buffers[i];
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL FBIND slot=%u candidate=%p mask=0x%x baseIndex=%u offset=%lld",
                  i, map->buf, map->attribute_mask, map->buffer_base_index,
                  (long long)map->offset);
        }
        ptr = mglRendererGetValidatedBuffer(ctx, map->buf, __FUNCTION__, (NSUInteger)i);
        offset = map->offset;
        isBaseBinding = mglRenderBufferMapIsBaseBinding(map->attribute_mask) != 0;
        GLuint glBindingIndex = map->buffer_base_index;
        NSInteger metalResolved = map->has_metal_binding
            ? (NSInteger)map->metal_binding_index
            : mglRendererGetProgramMetalBufferIndexForStage(
                  ctx, _FRAGMENT_SHADER, glBindingIndex);

        NSUInteger reflectedRequiredBytes = 0;
        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            reflectedRequiredBytes = map->has_metal_binding
                ? mglRendererGetProgramBindingRequiredSize(
                      ctx, _FRAGMENT_SHADER, (int)map->resource_type,
                      (int)map->resource_index)
                : mglRendererGetProgramBindingRequiredSizeForStage(
                      ctx, _FRAGMENT_SHADER, glBindingIndex);
        }
        uint64_t visibleCpu = ptr
            ? (uint64_t)mglBufferMapVisibleBackingBytes(map, ptr->data.buffer_size)
            : 0u;

        MGLStageBufferBindInput bin = {0};
        mglBindingStateFillStageBindInput(
            &bin, /*is_fragment=*/1, MGL_SB_PHASE_PRE_MTL, map, ptr,
            isBaseBinding ? 1 : 0, /*attrib_reserved=*/0,
            (uint32_t)MAX_BINDABLE_BUFFERS, (uint32_t)reflectedRequiredBytes,
            visibleCpu, map ? mglBufferMapVisibleSize(map) : 0,
            /*allow_isolate_when_gpu=*/0);
        if (isBaseBinding) {
            bin.has_metal_binding = 1;
            bin.metal_binding_index = (int32_t)metalResolved;
        }

        MGLStageBufferBindPlan plan = {0};
        if (mglBindingStagePlanMapEntry(&bin, &plan) != 0) {
            continue;
        }
        bindingIndex = plan.metal_slot;
        if (plan.mark_base_present && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            baseBindingPresent[glBindingIndex] = true;
        }

        if (plan.action == MGL_SB_ACTION_SKIP) {
            continue;
        }
        if (plan.action == MGL_SB_ACTION_CLEAR) {
            if (!ptr) {
                map->buf = NULL;
            }
            MGL_FBIND_EMIT_CLEAR(bindingIndex);
            mglRenderBindingClearFragmentBuffer(_bindingStateOwner,
                                                (uint32_t)bindingIndex);
            continue;
        }
        if (plan.action == MGL_SB_ACTION_INLINE_BYTES) {
            uint8_t padded[kMGLStageBindingStackScratchSize];
            const void *inlineBytes = mglBindingStageInlineBytesSrc(
                padded, (uint32_t)sizeof(padded),
                (const void *)((const uint8_t *)bin.cpu_ptr + plan.inline_src_offset),
                plan.inline_visible, plan.inline_length);
            MGL_FBIND_EMIT_BYTES(bindingIndex, inlineBytes, plan.inline_length);
            if (plan.invalidate_last_bound) {
                [self invalidateLastBoundFragmentBufferAtIndex:bindingIndex];
            }
            if (plan.mark_any_present) {
                anyBindingPresent[bindingIndex] = true;
            }
            if (plan.clear_cpu_dirty_if_no_mtl && ptr && !ptr->data.mtl_data) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            if (plan.clear_cpu_dirty && ptr) {
                ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            }
            continue;
        }

        if (plan.action == MGL_SB_ACTION_NEED_MTL) {
            if (!plan.use_mtl_as_inline_src) {
                if (!ptr->data.mtl_data) {
                    [self bindMTLBuffer:ptr];
                } else if (mglRenderBufferHasCPUDirty(ptr->data.dirty_bits)) {
                    [self updateDirtyBuffer:ptr];
                }
            }
            id buffer = nil;
            if (ptr->data.mtl_data &&
                (uintptr_t)ptr->data.mtl_data >= 0x100000000ULL) {
                buffer = (__bridge id)(ptr->data.mtl_data);
            } else if (!plan.use_mtl_as_inline_src && ptr->data.mtl_data &&
                       mglRenderMetalDataPointerUsable(ptr->data.mtl_data)) {
                /* Base UBO path may still use MetalDataPointerUsable. */
                buffer = (__bridge id)(ptr->data.mtl_data);
            }
            /* Align POST usability with historical FS thresholds. */
            if (plan.use_mtl_as_inline_src) {
                bin.mtl_usable =
                    ptr->data.mtl_data &&
                            (uintptr_t)ptr->data.mtl_data >= 0x100000000ULL
                        ? 1
                        : 0;
                buffer = bin.mtl_usable ? (__bridge id)(ptr->data.mtl_data) : nil;
            } else {
                bin.mtl_usable =
                    ptr->data.mtl_data &&
                            (uintptr_t)ptr->data.mtl_data >= 0x100000000ULL
                        ? 1
                        : 0;
                if (!bin.mtl_usable && ptr->data.mtl_data &&
                    mglRenderMetalDataPointerUsable(ptr->data.mtl_data)) {
                    bin.mtl_usable = 1;
                    buffer = (__bridge id)(ptr->data.mtl_data);
                }
            }
            NSUInteger metalLen = buffer ? mglBindingStateBufferLength(buffer) : 0u;
            NSUInteger availableBytes = buffer
                ? mglBufferMapVisibleBackingBytes(map, metalLen)
                : 0u;
            bin.phase = MGL_SB_PHASE_POST_MTL;
            bin.has_mtl_data = ptr->data.mtl_data ? 1 : 0;
            bin.mtl_ptr = ptr->data.mtl_data;
            bin.metal_len = (uint64_t)metalLen;
            bin.visible_mtl = (uint64_t)availableBytes;
            bin.binding_state_valid =
                mglBindingStateIsValid(_bindingStateOwner) ? 1 : 0;
            bin.buffer_matches =
                buffer &&
                        mglBindingStateBufferMatches(
                            _bindingStateOwner,
                            MGL_RENDER_BINDING_STAGE_FRAGMENT,
                            (__bridge void *)buffer, (NSUInteger)offset,
                            (uint32_t)plan.metal_slot)
                    ? 1
                    : 0;
            if (mglBindingStagePlanMapEntry(&bin, &plan) != 0) {
                continue;
            }
            bindingIndex = plan.metal_slot;

            if (plan.action == MGL_SB_ACTION_CLEAR) {
                MGL_FBIND_EMIT_CLEAR(bindingIndex);
                mglRenderBindingClearFragmentBuffer(_bindingStateOwner,
                                                    (uint32_t)bindingIndex);
                if (plan.clear_cpu_dirty && ptr) {
                    ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
                }
                continue;
            }
            if (plan.action == MGL_SB_ACTION_ISOLATE) {
                id isolated =
                    [self isolatedStageBindingBufferForMap:map
                                                     source:buffer
                                             requiredLength:plan.required_bytes];
                if (!isolated) {
                    NSLog(@"MGL WARNING: FBIND failed to isolate undersized buffer=%u slot=%lu required=%u available=%u",
                          ptr->name, (unsigned long)bindingIndex,
                          plan.required_bytes, plan.available_bytes);
                    MGL_FBIND_EMIT_CLEAR(bindingIndex);
                    mglRenderBindingClearFragmentBuffer(_bindingStateOwner,
                                                        (uint32_t)bindingIndex);
                    continue;
                }
                MGL_FBIND_EMIT_BUFFER(bindingIndex, (__bridge void *)isolated, 0);
                if (plan.needs_flush_snapshot) {
                    MGL_FBIND_FLUSH_SNAPSHOT();
                }
                mglRenderBindingUpdateFragmentBuffer(
                    _bindingStateOwner, (__bridge void *)isolated, 0,
                    (uint32_t)bindingIndex);
                MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                anyBindingPresent[bindingIndex] = true;
                continue;
            }
            if (plan.action == MGL_SB_ACTION_SKIP_MATCHED) {
                MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                anyBindingPresent[bindingIndex] = true;
                if (plan.clear_cpu_dirty && ptr) {
                    ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
                }
                continue;
            }
            if (plan.action == MGL_SB_ACTION_BIND_BUFFER) {
                MGL_FBIND_EMIT_BUFFER(bindingIndex, (__bridge void *)buffer,
                                      (NSUInteger)plan.bind_offset);
                mglRenderBindingUpdateFragmentBuffer(
                    _bindingStateOwner, (__bridge void *)buffer,
                    (NSUInteger)plan.bind_offset, (uint32_t)bindingIndex);
                MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                anyBindingPresent[bindingIndex] = true;
                if (plan.clear_cpu_dirty && ptr) {
                    ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
                }
                continue;
            }
        }
    }


    if (useBindingSnapshot && snapshot.fragment_op_count > 0) {
        mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
            encCtx->render_encoder_owner, &snapshot, NULL, 0);
        snapshot = (MGLRenderBindingSnapshot){0};
        fbindByteScratchUsed = 0;
    }

    if (mapCount > 0) {
        [self bindFragmentFallbackBuffersToCurrentRenderEncoder:activeProgram
                                             anyBindingPresent:anyBindingPresent
                                             baseBindingPresent:baseBindingPresent
                                                 encodeContext:encCtx
                                             bindingSnapshot:&snapshot
                                                 useSnapshot:useBindingSnapshot];
    }

    /* Fallback bindings are real Metal slots and must be included in the
     * worker snapshot. */
    uint32_t boundFragmentBufferMask = mglBindingStageBuildPresentMask(
        (const uint8_t *)anyBindingPresent, (uint32_t)kMGLMaxBufferSlots);
    mglRenderBindingOrFragmentBufferMask(_bindingStateOwner, boundFragmentBufferMask);

    if (mglEnvFlagEnabled("MGL_TRACE_SPARSE_BINDING")) {
        static uint64_t s_fbind_trace_count = 0;
        if ((++s_fbind_trace_count % 500) == 1) {
            int textureSlotCount =
                mglBindingStateTextureSlotCount(_bindingStateOwner);
            NSLog(@"MGL SPARSE FBIND: fbuf=0x%x(%u/31) textureSlots=%d/128",
                  boundFragmentBufferMask,
                  mglBindingStageCountPresent((const uint8_t *)anyBindingPresent,
                                              (uint32_t)kMGLMaxBufferSlots),
                  textureSlotCount);
        }
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(fbindCall)) {
        NSUInteger boundSlots = mglBindingStageCountPresent(
            (const uint8_t *)anyBindingPresent, (uint32_t)MAX_BINDABLE_BUFFERS);
        NSUInteger baseSlots = mglBindingStageCountPresent(
            (const uint8_t *)baseBindingPresent, (uint32_t)MAX_BINDABLE_BUFFERS);
        mglTraceLogNSString(@"MGL TRACE fbind.end call=%llu mapCount=%u boundSlots=%lu baseSlots=%lu elapsed=%.1fus",
              (unsigned long long)fbindCall,
              (unsigned)mapCount,
              (unsigned long)boundSlots,
              (unsigned long)baseSlots,
              (mglTraceClockNS() - fbindStartNS) / 1000.0);
    }

    /* Mark the dedup cache as valid for the current encoder so subsequent
     * binds can be skipped when the resource and offset are unchanged. */
    mglRenderBindingSetValid(_bindingStateOwner, 1);
    return true;
}


- (void)bindFragmentFallbackBuffersToCurrentRenderEncoder:(Program *)activeProgram
                                       anyBindingPresent:(bool *)anyBindingPresent
                                       baseBindingPresent:(bool *)baseBindingPresent
                                           encodeContext:(const MGLEncodeContext *)encCtx
                                       bindingSnapshot:(MGLRenderBindingSnapshot *)bindingSnapshot
                                           useSnapshot:(BOOL)useSnapshot
{
    [self bindStageFallbackBuffersForStage:_FRAGMENT_SHADER
                           isFragmentStage:YES
                                   program:activeProgram
                         anyBindingPresent:anyBindingPresent
                        baseBindingPresent:baseBindingPresent
                             encodeContext:encCtx
                           bindingSnapshot:bindingSnapshot
                               useSnapshot:useSnapshot
                              maxMetalSlots:MAX_BINDABLE_BUFFERS
                           enableAllSlotFill:YES];
}

/* O3.3: shared V/F stage-buffer fallback — plan@C slot + thin snap emit. */
- (void)bindStageFallbackBuffersForStage:(int)shaderStage
                         isFragmentStage:(BOOL)isFragment
                                 program:(Program *)activeProgram
                       anyBindingPresent:(bool *)anyBindingPresent
                      baseBindingPresent:(bool *)baseBindingPresent
                           encodeContext:(const MGLEncodeContext *)encCtx
                         bindingSnapshot:(MGLRenderBindingSnapshot *)bindingSnapshot
                             useSnapshot:(BOOL)useSnapshot
                            maxMetalSlots:(NSUInteger)maxMetalSlots
                         enableAllSlotFill:(BOOL)enableAllSlotFill
{
    MGLRenderBindingSnapshot *snap = bindingSnapshot;
    const BOOL useSnap = useSnapshot && snap != NULL;
    const int frag = isFragment ? 1 : 0;
    size_t scratchDummy = 0;
    const uint32_t metalStage = isFragment ? MGL_RENDER_BINDING_STAGE_FRAGMENT
                                           : MGL_RENDER_BINDING_STAGE_VERTEX;
#define MGL_SFB_FLUSH() do { if (useSnap) MGL_BIND_SNAP_FLUSH(*snap, frag, scratchDummy); } while (0)
#define MGL_SFB_EMIT_BUFFER(slot, bufPtr, off) do { \
    if (useSnap) { MGL_BIND_SNAP_COLLECT_BUFFER(*snap, frag, scratchDummy, slot, bufPtr, off); } \
    else if (isFragment) { mglBindingStateSetFragmentBuffer(encCtx->render_encoder_owner, (__bridge id)(bufPtr), (off), (slot)); } \
    else { mglBindingStateSetVertexBuffer(encCtx->render_encoder_owner, (__bridge id)(bufPtr), (off), (slot)); } \
} while (0)

    void *fallbackBindingBuffer = mglRendererBackendGetFallbackBindingBuffer(
        _backend, kMGLDefaultStageFallbackBufferSize);

    uint32_t resourceTypes[MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT];
    uint32_t resourceTypeCount =
        mglBindingStageFallbackResourceTypes(resourceTypes,
                                             MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT);
    for (uint32_t t = 0; t < resourceTypeCount; t++) {
        int resourceType = (int)resourceTypes[t];
        int count = mglRendererGetProgramBindingCount(ctx, shaderStage, resourceType);
        Program *program = activeProgram;
        for (int i = 0; i < count; i++) {
            if (!program || resourceType < 0 || resourceType >= MGL_MAX_SHADER_RESOURCES ||
                i >= (int)program->shader_resources_list[shaderStage][resourceType].count) {
                continue;
            }
            MGLShaderResource *resource =
                &program->shader_resources_list[shaderStage][resourceType].list[i];
            if (mglShouldSkipStageBufferResource(program, shaderStage, resourceType,
                                                 resource)) {
                continue;
            }
            GLuint elementCount =
                mglStageBufferResourceElementCount(resourceType, resource);
            for (GLuint element = 0; element < elementCount; element++) {
                GLuint clientBinding = mglClientBufferBindingForResourceElement(
                    resourceType, resource, element);
                if (clientBinding >= MAX_BINDABLE_BUFFERS) {
                    continue;
                }
                NSInteger metalBinding =
                    (NSInteger)mglMetalResourceSlotForElement(resource, element);
                if (metalBinding < 0 || metalBinding >= (NSInteger)maxMetalSlots) {
                    continue;
                }
                NSUInteger slot = (NSUInteger)metalBinding;
                int matches =
                    mglBindingStateIsValid(_bindingStateOwner) &&
                    mglBindingStateBufferMatches(_bindingStateOwner, metalStage,
                                                 fallbackBindingBuffer, 0,
                                                 (uint32_t)slot);
                uint32_t action = mglBindingStagePlanFallbackSlot(
                    anyBindingPresent[slot] ? 1 : 0, fallbackBindingBuffer ? 1 : 0,
                    mglBindingStateIsValid(_bindingStateOwner) ? 1 : 0,
                    matches ? 1 : 0);
                if (action == MGL_FB_SLOT_SKIP) {
                    continue;
                }
                if (action == MGL_FB_SLOT_EMIT) {
                    MGL_SFB_EMIT_BUFFER(slot, fallbackBindingBuffer, 0);
                    if (isFragment) {
                        mglRenderBindingUpdateFragmentBuffer(
                            _bindingStateOwner, fallbackBindingBuffer, 0,
                            (uint32_t)slot);
                        MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                    } else {
                        mglRenderBindingUpdateVertexBuffer(
                            _bindingStateOwner, fallbackBindingBuffer, 0,
                            (uint32_t)slot);
                        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                    }
                } else {
                    if (isFragment) {
                        MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                    } else {
                        MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                    }
                }
                baseBindingPresent[clientBinding] = true;
                anyBindingPresent[slot] = true;
            }
        }
    }

    if (enableAllSlotFill && fallbackBindingBuffer) {
        for (NSUInteger s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            int matches =
                mglBindingStateIsValid(_bindingStateOwner) &&
                mglBindingStateBufferMatches(_bindingStateOwner, metalStage,
                                             fallbackBindingBuffer, 0, (uint32_t)s);
            uint32_t action = mglBindingStagePlanFallbackSlot(
                anyBindingPresent[s] ? 1 : 0, 1,
                mglBindingStateIsValid(_bindingStateOwner) ? 1 : 0,
                matches ? 1 : 0);
            if (action == MGL_FB_SLOT_SKIP) {
                continue;
            }
            if (action == MGL_FB_SLOT_EMIT) {
                MGL_SFB_EMIT_BUFFER(s, fallbackBindingBuffer, 0);
                if (isFragment) {
                    mglRenderBindingUpdateFragmentBuffer(
                        _bindingStateOwner, fallbackBindingBuffer, 0, (uint32_t)s);
                    MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                } else {
                    mglRenderBindingUpdateVertexBuffer(
                        _bindingStateOwner, fallbackBindingBuffer, 0, (uint32_t)s);
                    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                }
            } else {
                if (isFragment) {
                    MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                } else {
                    MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                }
            }
            anyBindingPresent[s] = true;
        }
    }

    MGL_SFB_FLUSH();
#undef MGL_SFB_EMIT_BUFFER
#undef MGL_SFB_FLUSH
}


static const NSUInteger kMaxFragmentSamplerSlots = 16;

#define MGL_ABORT_TBIND_IF_ENCODER_CLOSED() do { \
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) == 0) { \
        if (ctx) { \
            mglMarkRendererDirtyBits(ctx->active_state, (DIRTY_TEX | DIRTY_TEX_BINDING | DIRTY_RENDER_STATE)); \
        } \
        return false; \
    } \
} while (0)

- (bool) bindTexturesToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx
{
    static uint64_t s_bindTexturesCallCount = 0;
    uint64_t bindCall = ++s_bindTexturesCallCount;
    bool traceBind = mglShouldTraceCall(bindCall);
    GLuint vertexSampledCount = 0;
    GLuint vertexBoundTextures = 0;
    GLuint vertexFallbackTextures = 0;
    GLuint boundSampledTextures = 0;
    GLuint nilSampledTextures = 0;
    GLuint fallbackSampledTextures = 0;
    GLuint boundSampledSamplers = 0;
    Program *vertexProgram = NULL;
    Program *fragmentProgram = NULL;
    GLuint vertexProgramName = 0u;
    GLuint fragmentProgramName = 0u;
    const BOOL useResourceSnapshot = YES;
    MGLRenderResourceBindingSnapshot resourceSnapshot = {0};

    if (!mglBindingStateHasActiveEncoder(encCtx)) {
        // No active render encoder yet (or it was rotated). Texture/sampler binding
        // can be deferred until the next encoder is created.
        return true;
    }

    /*
     * This array is the per-draw sampler snapshot used later by replay logging
     * and by the RT-sampled-copy cull bypass decision.  Do not let bindings from
     * the previous program survive in slots the current program does not touch.
     */
    if (mglTraceLogIsEnabled()) {
        mglTraceFragmentTextureTraceBindings("CLEAR",
                                             "bind_textures_begin",
                                             _resourceFallback.fragmentTextureTraceBindings,
                                             TEXTURE_UNITS,
                                             ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                             _pipelineCache.state->pipelineProgramName);

        memset(_resourceFallback.fragmentTextureTraceBindings, 0,
               sizeof(_resourceFallback.fragmentTextureTraceBindings));
    } else {
        /* When trace is disabled, only clear the functional flag fields read
         * by non-trace consumers (~384 bytes vs ~12 KB). */
        mglClearFragmentTextureTraceFunctionalFlags(
            _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
    }


    const int vertexResourceStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    vertexProgram = _tessellation.nativeTESActive
        ? _tessellation.nativeTESProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    vertexProgramName = vertexProgram ? vertexProgram->name : mglCurrentRenderProgramKey(ctx);
    fragmentProgramName = fragmentProgram ? fragmentProgram->name : mglCurrentRenderProgramKey(ctx);

    id defaultSampler = [self fallbackSamplerState];
    if (defaultSampler) {
        if (vertexProgram) {
            (void)mglProgramSamplesTextureUnit(vertexProgram, 0);
        }
        if (fragmentProgram && fragmentProgram != vertexProgram) {
            (void)mglProgramSamplesTextureUnit(fragmentProgram, 0);
        }
        MGLSamplerWarmupPlan warm = {0};
        mglBindingTexturePlanSamplerWarmup(
            1, vertexProgram ? 1 : 0, fragmentProgram ? 1 : 0,
            vertexProgram ? vertexProgram->sampled_texture_unit_mask : NULL,
            (fragmentProgram && fragmentProgram != vertexProgram)
                ? fragmentProgram->sampled_texture_unit_mask
                : NULL,
            (uint32_t)TEXTURE_UNITS, (uint32_t)kMaxFragmentSamplerSlots, &warm);
        for (uint32_t s = 0; s < warm.warmup_count; s++) {
            if (warm.mode == MGL_SW_MODE_MASK &&
                !mglBindingTextureSamplerWarmupSlotActive(warm.mask, s)) {
                continue;
            }
            if (!mglBindingStateQueueResourceBinding(
                    useResourceSnapshot, _bindingStateOwner,
                    _renderPassManager.state->currentRenderEncoderOwner,
                    &resourceSnapshot, MGL_RENDER_BINDING_STAGE_VERTEX,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    (__bridge void *)defaultSampler, s) ||
                !mglBindingStateQueueResourceBinding(
                    useResourceSnapshot, _bindingStateOwner,
                    _renderPassManager.state->currentRenderEncoderOwner,
                    &resourceSnapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    (__bridge void *)defaultSampler, s)) {
                return false;
            }
        }
    }

    if (useResourceSnapshot &&
        !mglBindingStateFlushResourceBindings(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            &resourceSnapshot)) {
        return false;
    }


    GLuint sampledCount = 0;
    GLuint separateSamplerCount = 0;
    GLuint boundSeparateSamplers = 0;

    // Metal validates every active stage resource. Bind vertex-stage sampled
    // images as well, even though most Minecraft pipelines only sample in FS.
    vertexSampledCount = mglRendererGetProgramBindingCount(ctx, vertexResourceStage, _SAMPLED_IMAGE_RES);
    if (![self bindVertexSampledTexturesToEncoder:vertexProgram
                                  vertexProgramName:vertexProgramName
                                     defaultSampler:defaultSampler
                                            bindCall:bindCall
                                          traceBind:traceBind
                                  vertexSampledCount:vertexSampledCount
                                         boundCount:&vertexBoundTextures
                                      fallbackCount:&vertexFallbackTextures]) {
        return false;
    }

    // Bind sampled images (texture + sampler).
    if (![self bindFragmentSampledTexturesToEncoder:fragmentProgram
                                  fragmentProgramName:fragmentProgramName
                                     vertexProgramName:vertexProgramName
                                        defaultSampler:defaultSampler
                                               bindCall:bindCall
                                             traceBind:traceBind
                                  boundSampledTextures:&boundSampledTextures
                                    nilSampledTextures:&nilSampledTextures
                               fallbackSampledTextures:&fallbackSampledTextures
                                 boundSampledSamplers:&boundSampledSamplers
                                            sampledCount:&sampledCount]) {
        return false;
    }

    /* Vertex/Fragment-stage storage image binding. */
    if (![self bindStorageImagesForVertexProgram:vertexProgram
                              fragmentProgram:fragmentProgram]) {
        return false;
    }

    // Bind separate samplers explicitly.
    if (![self bindSeparateSamplersAndArrayTextures:vertexProgram
                                      fragmentProgram:fragmentProgram
                                fragmentProgramName:fragmentProgramName
                                  vertexProgramName:vertexProgramName
                                     defaultSampler:defaultSampler
                                            bindCall:bindCall
                                          traceBind:traceBind
                                 separateSamplerCount:&separateSamplerCount
                                   boundSeparateSamplers:&boundSeparateSamplers]) {
        return false;
    }

    BOOL interestingTextureBind =
        (sampledCount > 0 && boundSampledTextures == 0) ||
        fallbackSampledTextures > 0 ||
        vertexFallbackTextures > 0;
    BOOL logTextureSummary = traceBind;
    if (interestingTextureBind) {
        static uint64_t s_interestingTextureSummaryCount = 0;
        uint64_t hit = ++s_interestingTextureSummaryCount;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            logTextureSummary = YES;
        }
    }
    if (logTextureSummary) {
        GLuint programName = mglCurrentRenderProgramKey(ctx);
        mglTraceLogNSString(@"MGL TRACE texbind.summary call=%llu program=%u vertexSampled=%u vertexBoundTex=%u vertexFallback=%u sampled=%u boundTex=%u nilTex=%u fallbackTex=%u sampledSamplers=%u separateSamplers=%u boundSeparate=%u",
              (unsigned long long)bindCall,
              (unsigned)programName,
              (unsigned)vertexSampledCount,
              (unsigned)vertexBoundTextures,
              (unsigned)vertexFallbackTextures,
              (unsigned)sampledCount,
              (unsigned)boundSampledTextures,
              (unsigned)nilSampledTextures,
              (unsigned)fallbackSampledTextures,
              (unsigned)boundSampledSamplers,
              (unsigned)separateSamplerCount,
              (unsigned)boundSeparateSamplers);
    }

    return true;
}

- (bool)bindVertexSampledTexturesToEncoder:(Program *)vertexProgram
                          vertexProgramName:(GLuint)vertexProgramName
                             defaultSampler:(id)defaultSampler
                                    bindCall:(uint64_t)bindCall
                                  traceBind:(bool)traceBind
                          vertexSampledCount:(GLuint)vertexSampledCount
                                 boundCount:(GLuint *)boundCount
                              fallbackCount:(GLuint *)fallbackCount
{
    GLuint vertexBoundTextures = *boundCount;
    GLuint vertexFallbackTextures = *fallbackCount;
    const BOOL useResourceSnapshot = YES;
    MGLRenderResourceBindingSnapshot resourceSnapshot = {0};
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;

    for (GLuint i = 0; i < vertexSampledCount; i++)
    {
        Program *currentProgram = vertexProgram;
        MGLShaderResource *sampledResource = NULL;
        const char *sampledName = "";
        if (currentProgram) {
            MGLShaderResourceList *list =
                &currentProgram->shader_resources_list[vertexStage][_SAMPLED_IMAGE_RES];
            GLuint ordinal = i;
            for (GLuint ri = 0; ri < list->count; ri++) {
                GLuint elements = mglRenderShaderResourceElementCount(
                    (uint32_t)list->list[ri].gl_array_size);
                if (ordinal < elements) {
                    sampledResource = &list->list[ri];
                    break;
                }
                ordinal -= elements;
            }
            if (sampledResource) sampledName = sampledResource->name;
        }
        /* read binding/gl_binding directly from the already-resolved
         * MGLShaderResource instead of re-resolving the program per query. When
         * sampledResource is NULL (no program / index OOR), mirror the
         * query-method semantics of returning 0. */
        GLuint spirvBinding = sampledResource
            ? (GLuint)mglRendererGetProgramBinding(ctx, vertexStage,
                                                   _SAMPLED_IMAGE_RES, (int32_t)i) : 0u;
        GLuint glBinding = sampledResource
            ? (GLuint)mglRendererGetProgramGLBinding(ctx, vertexStage,
                                                     _SAMPLED_IMAGE_RES, (int32_t)i) : 0u;
        MGLSampledTextureBindInput sin = {0};
        sin.phase = MGL_ST_PHASE_GATE;
        sin.spirv_binding = spirvBinding;
        sin.gl_binding = glBinding;
        sin.max_units = TEXTURE_UNITS;
        sin.skip_resource = mglShouldSkipStageTextureResource(
                                currentProgram, vertexStage, _SAMPLED_IMAGE_RES,
                                sampledResource)
                                ? 1
                                : 0;
        sin.has_resource = sampledResource ? 1 : 0;
        MGLSampledTextureBindPlan splan = {0};
        if (mglBindingTexturePlanSampled(&sin, &splan) != 0 ||
            splan.action == MGL_ST_ACTION_SKIP) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                        program:currentProgram
                                                    metalBinding:spirvBinding
                                                           stage:vertexStage];
        /* derive texture types/data kind directly from sampledResource
         * via C helpers, skipping per-resource mglResolveProgramForStageFromState. */
        uint32_t expectedType = (uint32_t)
            mglExpectedTextureTypeForResource(currentProgram, vertexStage, sampledResource);
        uint32_t lookupType = (uint32_t)
            mglDeclaredTextureTypeFromResource(sampledResource);
        MGLTextureDataKind expectedKind = (MGLTextureDataKind)
            mglExpectedTextureDataKindForResource(
                currentProgram, vertexStage, sampledResource);
        Texture *ptr = [self textureForSampledResource:sampledResource
                                          metalBinding:spirvBinding
                                                  stage:vertexStage
                                           expectedType:(lookupType ? lookupType : expectedType)
                                          textureUnit:textureUnit];
        id texture = nil;
        id sampler = defaultSampler;
        BOOL usedTypeFallback = NO;

        if (ptr) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            if (ptr->mtl_data) {
                texture = (__bridge id)(ptr->mtl_data);
                texture = (__bridge id)mglSampledTextureViewForBaseLevel(ptr, (__bridge void *)texture);
            }
            texture = [self applySampledCompatFallbackPlan:ptr
                                                   texture:texture
                                              expectedType:expectedType
                                              expectedKind:expectedKind
                                                     stage:"vertex"
                                               programName:vertexProgramName
                                              spirvBinding:spirvBinding
                                             sampleProgram:currentProgram
                                          usedFallbackOut:&usedTypeFallback];
            sampler = [self materializeSampledSamplerForTexture:ptr
                                                    textureUnit:textureUnit
                                                defaultSampler:defaultSampler
                                                  forceDefault:NO
                                                 samplerTarget:ptr ? ptr->target : 0u
                                                   programName:vertexProgramName
                                                  spirvBinding:spirvBinding
                                                         stage:"vertex"
                                                       texture:texture];
        }


        if (![self applySampledRenderTargetCopyPlan:ptr
                                            texture:&texture
                                      sampleProgram:currentProgram
                                       expectedType:expectedType
                                       expectedKind:expectedKind
                                  usedTypeFallback:usedTypeFallback
                                            stage:"vertex"
                                       programName:vertexProgramName
                                       spirvBinding:spirvBinding
                                         textureUnit:textureUnit
                                         sampledName:sampledName
                                usedSampledCopyOut:NULL
                              directTextureForTrace:NULL
                              sampledCopyForTrace:NULL]) {
            return false;
        }

        GLuint samplerBinding = sampledResource && sampledResource->has_combined_sampler
            ? mglMetalCombinedSamplerSlot(sampledResource)
            : spirvBinding;
        sin.phase = MGL_ST_PHASE_FINAL;
        sin.has_bound_texture = texture ? 1 : 0;
        sin.suppress_missing_fallback = 0;
        sin.used_type_fallback = usedTypeFallback ? 1 : 0;
        sin.has_combined_sampler =
            sampledResource && sampledResource->has_combined_sampler ? 1 : 0;
        sin.sampler_binding = samplerBinding;
        sin.max_sampler_slots = kMaxFragmentSamplerSlots;
        sin.has_sampler = sampler ? 1 : 0;
        if (!texture) {
            texture = [self fallbackSampledTextureForExpectedType:expectedType
                                                         dataKind:expectedKind];
            if (texture) {
                /* Match historical: count here; do not set usedTypeFallback
                 * (end-of-loop only counts type/kind fallbacks). */
                vertexFallbackTextures++;
                sin.has_bound_texture = 1;
            }
        }
        if (mglBindingTexturePlanSampled(&sin, &splan) != 0 ||
            splan.action != MGL_ST_ACTION_QUEUE) {
            continue;
        }
        if (splan.queue_texture &&
            !mglBindingStateQueueResourceBinding(
                useResourceSnapshot, _bindingStateOwner,
                _renderPassManager.state->currentRenderEncoderOwner,
                &resourceSnapshot, MGL_RENDER_BINDING_STAGE_VERTEX,
                MGL_RENDER_RESOURCE_BINDING_TEXTURE,
                (__bridge void *)texture, splan.texture_slot)) {
            return false;
        }
        if (splan.queue_sampler &&
            !mglBindingStateQueueResourceBinding(
                useResourceSnapshot, _bindingStateOwner,
                _renderPassManager.state->currentRenderEncoderOwner,
                &resourceSnapshot, MGL_RENDER_BINDING_STAGE_VERTEX,
                MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                (__bridge void *)sampler, splan.sampler_slot)) {
            return false;
        }
        static uint64_t s_focusedVertexTextureBindLogs = 0;
        static uint64_t s_traceFileVertexTextureBindLogs = 0;
        [self emitSampledDiagPortsForProgram:currentProgram
                                       stage:"vertex"
                             stageIsFragment:NO
                                 sampledName:sampledName
                                spirvBinding:spirvBinding
                                 textureUnit:textureUnit
                            sampledResource:sampledResource
                                         ptr:ptr
                                     texture:texture
                                     sampler:sampler
                               usedFallback:usedTypeFallback
                              expectedType:expectedType
                                lookupType:lookupType
                                   bindCall:bindCall
                                programName:vertexProgramName
                           vertexProgramName:vertexProgramName
                         fragmentProgramName:0u
                        usedSampledCopyTrace:NO
                       directTextureForTrace:nil
                       sampledCopyForTrace:nil
                           focusedCounter:&s_focusedVertexTextureBindLogs
                         traceFileCounter:&s_traceFileVertexTextureBindLogs];
        if (texture) {
            vertexBoundTextures++;
            if (usedTypeFallback) {
                vertexFallbackTextures++;
            }
        }
    }

    if (useResourceSnapshot &&
        !mglBindingStateFlushResourceBindings(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            &resourceSnapshot)) {
        return false;
    }
    *boundCount = vertexBoundTextures;
    *fallbackCount = vertexFallbackTextures;
    return true;
}

- (bool)bindFragmentSampledTexturesToEncoder:(Program *)fragmentProgram
                          fragmentProgramName:(GLuint)fragmentProgramName
                             vertexProgramName:(GLuint)vertexProgramName
                                defaultSampler:(id)defaultSampler
                                       bindCall:(uint64_t)bindCall
                                     traceBind:(bool)traceBind
                          boundSampledTextures:(GLuint *)boundSampledTexturesPtr
                            nilSampledTextures:(GLuint *)nilSampledTexturesPtr
                       fallbackSampledTextures:(GLuint *)fallbackSampledTexturesPtr
                         boundSampledSamplers:(GLuint *)boundSampledSamplersPtr
                                    sampledCount:(GLuint *)sampledCount
{
    GLuint boundSampledTextures = *boundSampledTexturesPtr;
    GLuint nilSampledTextures = *nilSampledTexturesPtr;
    GLuint fallbackSampledTextures = *fallbackSampledTexturesPtr;
    GLuint boundSampledSamplers = *boundSampledSamplersPtr;
    const BOOL useResourceSnapshot = YES;
    MGLRenderResourceBindingSnapshot resourceSnapshot = {0};

    // Bind sampled images (texture + sampler).
    *sampledCount = mglRendererGetProgramBindingCount(ctx, _FRAGMENT_SHADER, _SAMPLED_IMAGE_RES);
    for (GLuint i = 0; i < *sampledCount; i++)
    {
        Program *sampleProgram = fragmentProgram;
        MGLShaderResource *sampledResource = NULL;
        const char *sampledName = "";
        if (sampleProgram) {
            MGLShaderResourceList *list =
                &sampleProgram->shader_resources_list[_FRAGMENT_SHADER][_SAMPLED_IMAGE_RES];
            GLuint ordinal = i;
            for (GLuint ri = 0; ri < list->count; ri++) {
                GLuint elements = mglRenderShaderResourceElementCount(
                    (uint32_t)list->list[ri].gl_array_size);
                if (ordinal < elements) {
                    sampledResource = &list->list[ri];
                    break;
                }
                ordinal -= elements;
            }
            if (sampledResource) sampledName = sampledResource->name;
        }
        /* read binding/gl_binding directly from the already-resolved
         * MGLShaderResource instead of re-resolving the program per query. */
        GLuint spirvBinding = sampledResource
            ? (GLuint)mglRendererGetProgramBinding(ctx, _FRAGMENT_SHADER,
                                                   _SAMPLED_IMAGE_RES, (int32_t)i) : 0u;
        GLuint glBinding = sampledResource
            ? (GLuint)mglRendererGetProgramGLBinding(ctx, _FRAGMENT_SHADER,
                                                     _SAMPLED_IMAGE_RES, (int32_t)i) : 0u;
        MGLSampledTextureBindInput fsin = {0};
        fsin.phase = MGL_ST_PHASE_GATE;
        fsin.spirv_binding = spirvBinding;
        fsin.gl_binding = glBinding;
        fsin.max_units = TEXTURE_UNITS;
        fsin.skip_resource = mglShouldSkipStageTextureResource(
                                 sampleProgram, _FRAGMENT_SHADER,
                                 _SAMPLED_IMAGE_RES, sampledResource)
                                 ? 1
                                 : 0;
        fsin.has_resource = sampledResource ? 1 : 0;
        MGLSampledTextureBindPlan fsplan = {0};
        if (mglBindingTexturePlanSampled(&fsin, &fsplan) != 0 ||
            fsplan.action == MGL_ST_ACTION_SKIP) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                        program:sampleProgram
                                                    metalBinding:spirvBinding
                                                           stage:_FRAGMENT_SHADER];

        /* derive texture types/data kind directly from sampledResource
         * via C helpers, skipping per-resource mglResolveProgramForStageFromState. */
        uint32_t expectedType = (uint32_t)
            mglExpectedTextureTypeForResource(sampleProgram, _FRAGMENT_SHADER, sampledResource);
        uint32_t lookupType = (uint32_t)
            mglDeclaredTextureTypeFromResource(sampledResource);
        MGLTextureDataKind expectedKind = (MGLTextureDataKind)
            mglExpectedTextureDataKindForResource(
                sampleProgram, _FRAGMENT_SHADER, sampledResource);
        Texture *ptr = [self textureForSampledResource:sampledResource
                                          metalBinding:spirvBinding
                                                  stage:_FRAGMENT_SHADER
                                           expectedType:(lookupType ? lookupType : expectedType)
                                          textureUnit:textureUnit];
        id texture = nil;
        id sampler = nil;
        id directTextureForTrace = nil;
        id sampledCopyForTrace = nil;
        BOOL usedFallbackTexture = NO;
        BOOL suppressMissingTextureFallback = NO;
        BOOL usedSampledCopyForTrace = NO;

        if (ptr) {
            if (![self recoverFragmentSampledDepthTexture:&ptr
                                                   texture:&texture
                                               sampledName:sampledName
                                              spirvBinding:spirvBinding
                                                textureUnit:textureUnit
                                               expectedType:expectedType
                                               expectedKind:expectedKind
                                       fragmentProgramName:fragmentProgramName
                            suppressMissingTextureFallback:&suppressMissingTextureFallback
                                      usedFallbackTexture:&usedFallbackTexture]) {
                return false;
            }
            if (![self applySampledRenderTargetCopyPlan:ptr
                                                texture:&texture
                                          sampleProgram:sampleProgram
                                           expectedType:expectedType
                                           expectedKind:expectedKind
                                      usedTypeFallback:usedFallbackTexture
                                                stage:"fragment"
                                           programName:fragmentProgramName
                                           spirvBinding:spirvBinding
                                             textureUnit:textureUnit
                                             sampledName:sampledName
                                    usedSampledCopyOut:&usedSampledCopyForTrace
                                  directTextureForTrace:&directTextureForTrace
                                  sampledCopyForTrace:&sampledCopyForTrace]) {
                return false;
            }
            texture = [self applySampledCompatFallbackPlan:ptr
                                                   texture:texture
                                              expectedType:expectedType
                                              expectedKind:expectedKind
                                                     stage:"fragment"
                                               programName:fragmentProgramName
                                              spirvBinding:spirvBinding
                                             sampleProgram:sampleProgram
                                          usedFallbackOut:&usedFallbackTexture];
            if (usedFallbackTexture) {
                usedSampledCopyForTrace = NO;
            }
            sampler = [self materializeSampledSamplerForTexture:ptr
                                                    textureUnit:textureUnit
                                                defaultSampler:sampler
                                                  forceDefault:NO
                                                 samplerTarget:ptr ? ptr->target : 0u
                                                   programName:fragmentProgramName
                                                  spirvBinding:spirvBinding
                                                         stage:"fragment"
                                                       texture:texture];
            if (mglMipDiagEnabled() && ptr && textureUnit < TEXTURE_UNITS) {
                Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
                const TextureParameter *effective =
                    glSampler ? &glSampler->params : &ptr->params;
                uint64_t signature = mglBindingTextureMipDiagSignature(
                    ptr->name, effective->min_filter, effective->mag_filter,
                    ptr->params.base_level, ptr->params.max_level,
                    texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u,
                    (uint64_t)(uintptr_t)texture, usedSampledCopyForTrace ? 1 : 0,
                    ptr->mtl_gl_sampled_levels, ptr->mtl_gl_sampled_dirty_mip_mask,
                    ptr->mtl_gl_sampled_write_version !=
                        ptr->mtl_render_target_write_version);
                static uint64_t s_fragSamplerState[TEXTURE_UNITS];
                if (mglMipDiagStateChanged(&s_fragSamplerState[textureUnit], signature)) {
                    mglBindingLogMipDiagFrag(
                        textureUnit, spirvBinding, fragmentProgramName, ptr->name,
                        glSampler ? "glSampler" : "texParams", effective->min_filter,
                        effective->mag_filter, effective->min_lod, effective->max_lod,
                        effective->max_anisotropy, ptr->params.base_level,
                        ptr->params.max_level, ptr->num_levels,
                        texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u,
                        texture ? mglBindingStateTextureWidth(texture) : 0u,
                        texture ? mglBindingStateTextureHeight(texture) : 0u, texture,
                        ptr->is_render_target ? 1 : 0, usedSampledCopyForTrace ? 1 : 0,
                        ptr->mtl_gl_sampled_levels, ptr->mtl_gl_sampled_dirty_mip_mask,
                        ptr->mtl_render_target_write_version,
                        ptr->mtl_gl_sampled_write_version);
                }
            }
        }

        if (!texture && !suppressMissingTextureFallback) {
            texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
            if (texture) {
                usedFallbackTexture = YES;
                usedSampledCopyForTrace = NO;
                mglFocusLoadingProgram(fragmentProgramName,
                                       "sample-fallback",
                                       bindCall);
                fallbackSampledTextures++;
                static uint64_t s_fragmentFallbackLogCount = 0;
                if (mglBindingTextureRateLogHit(&s_fragmentFallbackLogCount, 32ull,
                                                512ull)) {
                    mglBindingLogTexFallbackEx(s_fragmentFallbackLogCount, spirvBinding,
                                               fragmentProgramName,
                                               ptr ? ptr->name : 0u, 0, NULL, 0u);
                }
            }
        } else if (!texture && suppressMissingTextureFallback) {
            static uint64_t s_fragmentFallbackSuppressedLogCount = 0;
            if (mglBindingTextureRateLogHit(&s_fragmentFallbackSuppressedLogCount,
                                            64ull, 512ull)) {
                mglBindingLogTexFallbackEx(
                    s_fragmentFallbackSuppressedLogCount, spirvBinding,
                    fragmentProgramName, ptr ? ptr->name : 0u, 1, sampledName,
                    textureUnit);
            }
        }

        {
            MGLSamplerMaterializeInput fin = {
                .force_default =
                    (usedFallbackTexture && expectedKind == MGLTextureDataKindDepth)
                        ? 1
                        : 0,
                .unit_in_range = 0,
                .has_gl_sampler = 0,
                .require_tex_params_mtl = 1,
            };
            MGLSamplerMaterializePlan fplan = {0};
            (void)mglBindingTexturePlanSamplerMaterialize(&fin, &fplan);
            if (fplan.action == MGL_SM_ACTION_USE_DEFAULT || !sampler) {
                sampler = defaultSampler;
            }
        }

        GLuint sampleProgramName =
            sampleProgram ? sampleProgram->name : fragmentProgramName;

        if (!mglBindingStateQueueResourceBinding(
                useResourceSnapshot, _bindingStateOwner,
                _renderPassManager.state->currentRenderEncoderOwner,
                &resourceSnapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                MGL_RENDER_RESOURCE_BINDING_TEXTURE,
                (__bridge void *)texture, spirvBinding)) {
            return false;
        }
        if (spirvBinding < TEXTURE_UNITS) {
            MGLFragmentTextureTraceBinding *traceBinding = &_resourceFallback.fragmentTextureTraceBindings[spirvBinding];
            memset(traceBinding, 0, sizeof(*traceBinding));
            traceBinding->gl_texture_name = ptr ? ptr->name : 0u;
            traceBinding->sampler_unit = textureUnit;
            traceBinding->metal_binding = spirvBinding;
            traceBinding->program_name = sampleProgramName;
            traceBinding->rt_write_version = ptr ? ptr->mtl_render_target_write_version : 0u;
            traceBinding->sampled_write_version = ptr ? ptr->mtl_gl_sampled_write_version : 0u;
            traceBinding->gl_texture_ptr = ptr;
            traceBinding->mtl_texture_ptr = (__bridge void *)texture;
            traceBinding->direct_mtl_texture_ptr = (__bridge void *)(directTextureForTrace ? directTextureForTrace : texture);
            traceBinding->sampled_copy_ptr = (__bridge void *)sampledCopyForTrace;
            traceBinding->width = texture ? mglBindingStateTextureWidth(texture) : 0u;
            traceBinding->height = texture ? mglBindingStateTextureHeight(texture) : 0u;
            traceBinding->pixel_format = texture ? mglBindingStateTexturePixelFormat(texture) : MGL_BINDING_PIXEL_FORMAT_INVALID;
            traceBinding->texture_type = texture ? mglBindingStateTextureType(texture) : 0u;
            traceBinding->used_sampled_copy = usedSampledCopyForTrace ? 1u : 0u;
            traceBinding->used_fallback = usedFallbackTexture ? 1u : 0u;
        }
        static uint64_t s_focusedFragmentTextureBindLogs = 0;
        static uint64_t s_traceFileFragmentTextureBindLogs = 0;
        [self emitSampledDiagPortsForProgram:sampleProgram
                                       stage:"fragment"
                             stageIsFragment:YES
                                 sampledName:sampledName
                                spirvBinding:spirvBinding
                                 textureUnit:textureUnit
                            sampledResource:sampledResource
                                         ptr:ptr
                                     texture:texture
                                     sampler:sampler
                               usedFallback:usedFallbackTexture
                              expectedType:expectedType
                                lookupType:lookupType
                                   bindCall:bindCall
                                programName:sampleProgramName
                           vertexProgramName:vertexProgramName
                         fragmentProgramName:fragmentProgramName
                        usedSampledCopyTrace:usedSampledCopyForTrace
                       directTextureForTrace:directTextureForTrace
                       sampledCopyForTrace:sampledCopyForTrace
                           focusedCounter:&s_focusedFragmentTextureBindLogs
                         traceFileCounter:&s_traceFileFragmentTextureBindLogs];
        switch (mglBindingTextureSampledMarkKind(texture ? 1 : 0,
                                                 usedFallbackTexture ? 1 : 0)) {
        case MGL_ST_MARK_BOUND:
            boundSampledTextures++;
            break;
        case MGL_ST_MARK_FALLBACK:
            /* Keep nilTex as original GL failure count; Metal gets fallback. */
            nilSampledTextures++;
            break;
        default:
            nilSampledTextures++;
            break;
        }
        GLuint samplerBinding = sampledResource && sampledResource->has_combined_sampler
            ? mglMetalCombinedSamplerSlot(sampledResource)
            : spirvBinding;
        if (sampler &&
            (!sampledResource || sampledResource->has_combined_sampler) &&
            samplerBinding < kMaxFragmentSamplerSlots) {
            if (!mglBindingStateQueueResourceBinding(
                    useResourceSnapshot, _bindingStateOwner,
                    _renderPassManager.state->currentRenderEncoderOwner,
                    &resourceSnapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    (__bridge void *)sampler, samplerBinding)) {
                return false;
            }
            boundSampledSamplers++;
        }

        if (traceBind && i < 6) {
            TextureLevel *level0 = NULL;
            if (ptr && ptr->faces[0].levels) {
                level0 = &ptr->faces[0].levels[0];
            }
            uint32_t cpuFirstTexel = 0u;
            bool cpuFirstTexelValid = false;
            if (level0 && level0->data && level0->data_size >= sizeof(cpuFirstTexel) &&
                ((uintptr_t)level0->data >= 0x1000ull)) {
                memcpy(&cpuFirstTexel, (const void *)level0->data, sizeof(cpuFirstTexel));
                cpuFirstTexelValid = true;
            }

            mglTraceLogNSString(@"MGL TRACE texbind.sampled call=%llu idx=%u binding=%u glTex=%u target=0x%x internal=0x%x "
                  "l0=%ux%ux%u l0bytes=%lu l0first=0x%08x(valid=%d) "
                  "l0src(source=%u upload=%lu srcPtr=%p hash=0x%016llx init(ever=%u full=%u zero=%u)) "
                  "mtlTex=%p size=%lux%lu sampler=%p fallback=%d",
                  (unsigned long long)bindCall,
                  (unsigned)i,
                  (unsigned)spirvBinding,
                  ptr ? (unsigned)ptr->name : 0u,
                  ptr ? (unsigned)ptr->target : 0u,
                  ptr ? (unsigned)ptr->internalformat : 0u,
                  level0 ? (unsigned)level0->width : 0u,
                  level0 ? (unsigned)level0->height : 0u,
                  level0 ? (unsigned)level0->depth : 0u,
                  (unsigned long)(level0 ? level0->data_size : 0u),
                  (unsigned)cpuFirstTexel,
                  cpuFirstTexelValid ? 1 : 0,
                  (unsigned)(level0 ? level0->last_init_source : 0u),
                  (unsigned long)(level0 ? level0->last_upload_size : 0u),
                  (void *)(level0 ? level0->last_src_ptr : NULL),
                  (unsigned long long)(level0 ? level0->last_src_hash : 0ull),
                  (unsigned)(level0 ? level0->ever_written : 0u),
                  (unsigned)(level0 ? level0->has_initialized_data : 0u),
                  (unsigned)(level0 ? level0->suspicious_zero_upload : 0u),
                  texture,
                  (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
                  (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
                  sampler,
                  usedFallbackTexture ? 1 : 0);
        }
    }

    if (useResourceSnapshot &&
        !mglBindingStateFlushResourceBindings(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            &resourceSnapshot)) {
        return false;
    }
    *boundSampledTexturesPtr = boundSampledTextures;
    *nilSampledTexturesPtr = nilSampledTextures;
    *fallbackSampledTexturesPtr = fallbackSampledTextures;
    *boundSampledSamplersPtr = boundSampledSamplers;
    return true;
}

- (bool)recoverFragmentSampledDepthTexture:(Texture **)ptrPtr
                                    texture:(id *)texturePtr
                                sampledName:(const char *)sampledName
                                spirvBinding:(GLuint)spirvBinding
                                  textureUnit:(GLuint)textureUnit
                                 expectedType:(uint32_t)expectedType
                                 expectedKind:(MGLTextureDataKind)expectedKind
                         fragmentProgramName:(GLuint)fragmentProgramName
                  suppressMissingTextureFallback:(BOOL *)suppressMissingTextureFallbackPtr
                            usedFallbackTexture:(BOOL *)usedFallbackTexturePtr
{
    Texture *ptr = *ptrPtr;
    id texture = *texturePtr;
    BOOL suppressMissingTextureFallback = *suppressMissingTextureFallbackPtr;
    BOOL usedFallbackTexture = *usedFallbackTexturePtr;

    RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
    MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
    if (ptr->mtl_data) {
        texture = (__bridge id)(ptr->mtl_data);
        /* Defer base-level views for render targets until after the Y-flip
         * decision.  Creating a live-RT view here pollutes
         * mtl_base_level_view; a later sampled-copy bind can then race the
         * cache across MRT slots (texture_barrier color1+). */
        if (!ptr->is_render_target) {
            texture = (__bridge id)mglSampledTextureViewForBaseLevel(
                ptr, (__bridge void *)texture);
        }
    }

    TextureLevel *depthSampleLevel0 = mglTraceTextureBaseLevel(ptr);
    MGLDepthRecoverInput gin = {
        .phase = MGL_DR_PHASE_GATE,
        .has_texture = texture ? 1 : 0,
        .is_insampler = mglBindingTextureSampledNameIsInSampler(sampledName),
        .is_depth_or_stencil = texture && mglMetalPixelFormatIsDepthOrStencil(
                                              mglBindingStateTexturePixelFormat(texture)),
        .is_render_target = ptr && ptr->is_render_target ? 1 : 0,
        .level0_ever_written = depthSampleLevel0 && depthSampleLevel0->ever_written,
        .level0_has_init = depthSampleLevel0 && depthSampleLevel0->has_initialized_data,
    };
    MGLDepthRecoverPlan gplan = {0};
    if (mglBindingTexturePlanDepthRecover(&gin, &gplan) != 0 ||
        gplan.action == MGL_DR_ACTION_KEEP) {
        goto done;
    }

    if (gplan.action == MGL_DR_ACTION_ENTER_INSAMPLER) {
        GLuint pairedFboName = 0u;
        Texture *pairedColor =
            mglFindFramebufferColorTexturePairedWithDepth(ctx, ptr, &pairedFboName);
        Texture *recoverTexture = NULL;
        id recoverMTL = nil;
        const char *recoverReason = "none";
        BOOL recoveredFromSampledCopy = NO, recoveredFromPreviousVersion = NO;
        NSUInteger recoverAtt = MAX_COLOR_ATTACHMENTS;
        NSUInteger curAtt = MAX_COLOR_ATTACHMENTS;
        BOOL pairedCur = mglCurrentDrawFramebufferUsesColorTexture(
            ctx, pairedColor, pairedFboName, &curAtt);
        id pairedMTL = nil;
        if (pairedColor) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:pairedColor]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            pairedMTL = pairedColor->mtl_data ? (__bridge id)(pairedColor->mtl_data) : nil;
            if (!pairedCur && pairedMTL) {
                pairedCur = mglBindingStateRenderPassUsesColorTexture(
                    _renderPassManager.state->renderPassStateOwner,
                    (__bridge void *)pairedMTL, &curAtt);
            }
        }
        MGLDepthRecoverInput iin = {
            .phase = MGL_DR_PHASE_INSAMPLER,
            .has_paired_color = pairedColor ? 1 : 0,
            .paired_is_current_draw = pairedCur ? 1 : 0,
            .has_paired_mtl = pairedMTL ? 1 : 0,
            .paired_is_depth_or_stencil =
                pairedMTL && mglMetalPixelFormatIsDepthOrStencil(
                                 mglBindingStateTexturePixelFormat(pairedMTL)),
            .unit_in_range = textureUnit < TEXTURE_UNITS ? 1 : 0,
        };
        MGLDepthRecoverPlan iplan = {0};
        (void)mglBindingTexturePlanDepthRecover(&iin, &iplan);

        if (iplan.action == MGL_DR_ACTION_PROBE_PAIRED_COPY) {
            static uint64_t s_histSup = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_histSup)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_HIST_SUPPRESSED, .hit = s_histSup, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .fbo = pairedFboName, .color_att = curAtt, .depth_tex = ptr ? ptr->name : 0u, .paired_color = pairedColor ? pairedColor->name : 0u);
            }
            id pairedCopy = nil;
            BOOL usedPrev = NO;
            int usable = pairedColor && mglRendererGLSampledCopyLooksUsable(
                pairedColor, expectedType, expectedKind, YES, &pairedCopy, &usedPrev);
            MGLDepthRecoverInput cin = {.phase = MGL_DR_PHASE_COPY,
                                        .paired_copy_usable = usable ? 1 : 0};
            MGLDepthRecoverPlan cplan = {0};
            (void)mglBindingTexturePlanDepthRecover(&cin, &cplan);
            if (cplan.action == MGL_DR_ACTION_USE_RECOVER) {
                recoverTexture = pairedColor;
                recoverMTL = pairedCopy;
                recoverReason = cplan.reason_tag ? cplan.reason_tag : "paired-current-copy";
                recoveredFromSampledCopy = YES;
                recoveredFromPreviousVersion = usedPrev;
                recoverAtt = curAtt;
            } else {
                static uint64_t s_noCopy = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_noCopy)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_NO_COPY, .hit = s_noCopy, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .fbo = pairedFboName, .color_att = curAtt, .depth_tex = ptr ? ptr->name : 0u, .color_tex = pairedColor ? pairedColor->name : 0u, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .sampled_ver = pairedColor ? pairedColor->mtl_gl_sampled_write_version : 0u, .rt_ver = pairedColor ? pairedColor->mtl_render_target_write_version : 0u);
                }
                texture = nil;
                suppressMissingTextureFallback = YES;
            }
        } else if (iplan.action == MGL_DR_ACTION_USE_PAIRED_DIRECT) {
            static uint64_t s_rec = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_rec)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_PAIRED_DIRECT, .hit = s_rec, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .fbo = pairedFboName, .depth_tex = ptr ? ptr->name : 0u, .color_tex = pairedColor->name, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .color_fmt = mglBindingStateTexturePixelFormat(pairedMTL), .w = mglBindingStateTextureWidth(pairedMTL), .h = mglBindingStateTextureHeight(pairedMTL));
            }
            ptr = pairedColor;
            texture = pairedMTL;
        } else if (iplan.action == MGL_DR_ACTION_SCAN_HISTORY) {
            for (GLuint hi = 0; hi < MGL_RECENT_SAMPLED_2D_HISTORY; hi++) {
                Texture *cand =
                    MGL_STATE(ctx)->recent_sampled_2d_textures[textureUnit][hi];
                if (!cand || cand == ptr || cand == pairedColor ||
                    !mglRendererTextureLooksLikeSampledColor2D(ctx, cand))
                    continue;
                id candMTL = cand->mtl_data ? (__bridge id)(cand->mtl_data) : nil;
                NSUInteger candAtt = MAX_COLOR_ATTACHMENTS;
                BOOL candCur =
                    mglCurrentDrawFramebufferUsesColorTexture(ctx, cand, 0u, &candAtt) ||
                    mglBindingStateRenderPassUsesColorTexture(
                        _renderPassManager.state->renderPassStateOwner,
                        (__bridge void *)candMTL, &candAtt);
                if (!candCur && (!cand->mtl_data || cand->dirty_bits)) {
                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture:cand]);
                    MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
                    candMTL = cand->mtl_data ? (__bridge id)(cand->mtl_data) : nil;
                    candAtt = MAX_COLOR_ATTACHMENTS;
                    candCur =
                        mglCurrentDrawFramebufferUsesColorTexture(ctx, cand, 0u, &candAtt) ||
                        mglBindingStateRenderPassUsesColorTexture(
                            _renderPassManager.state->renderPassStateOwner,
                            (__bridge void *)candMTL, &candAtt);
                }
                id candCopy = nil;
                BOOL usedPrev = NO;
                int copyOk = cand->is_render_target &&
                    mglRendererGLSampledCopyLooksUsable(cand, expectedType, expectedKind,
                                                        candCur, &candCopy, &usedPrev);
                MGLDepthRecoverInput hin = {
                    .phase = MGL_DR_PHASE_HISTORY,
                    .candidate_valid = 1,
                    .candidate_is_rt = cand->is_render_target ? 1 : 0,
                    .candidate_is_current_draw = candCur ? 1 : 0,
                    .candidate_has_mtl = candMTL ? 1 : 0,
                    .candidate_copy_usable = copyOk ? 1 : 0,
                    .candidate_is_depth_or_stencil =
                        candMTL && mglMetalPixelFormatIsDepthOrStencil(
                                       mglBindingStateTexturePixelFormat(candMTL)),
                    .candidate_type_ok = !candMTL || expectedType == 0 ||
                        mglBindingStateTextureType(candMTL) == expectedType,
                    .candidate_kind_ok = !candMTL ||
                        mglTexturePixelFormatCompatibleWithExpectedDataKind(
                            mglBindingStateTexturePixelFormat(candMTL), expectedKind),
                };
                MGLDepthRecoverPlan hplan = {0};
                (void)mglBindingTexturePlanDepthRecover(&hin, &hplan);
                if (hplan.action == MGL_DR_ACTION_HISTORY_USE_COPY) {
                    recoverTexture = cand;
                    recoverMTL = candCopy;
                    recoverReason = hplan.reason_tag ? hplan.reason_tag : "history-copy";
                    recoveredFromSampledCopy = YES;
                    recoveredFromPreviousVersion = usedPrev;
                    recoverAtt = candAtt;
                    break;
                }
                if (hplan.action == MGL_DR_ACTION_HISTORY_USE_DIRECT) {
                    recoverTexture = cand;
                    recoverMTL = candMTL;
                    recoverReason = hplan.reason_tag ? hplan.reason_tag : "history-direct";
                    recoverAtt = candAtt;
                    break;
                }
            }
        } else if (iplan.action == MGL_DR_ACTION_LOG_UNPAIRED) {
            static uint64_t s_unp = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_unp)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_UNPAIRED, .hit = s_unp, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .depth_tex = ptr ? ptr->name : 0u, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .w = mglBindingStateTextureWidth(texture), .h = mglBindingStateTextureHeight(texture));
            }
        }

        if (recoverTexture && recoverMTL) {
            static uint64_t s_histRec = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_histRec)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_HISTORY_RECOVERY, .hit = s_histRec, .reason = recoverReason, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .fbo = pairedFboName, .color_att = recoverAtt, .depth_tex = ptr ? ptr->name : 0u, .recover_tex = recoverTexture ? recoverTexture->name : 0u, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .recover_fmt = mglBindingStateTexturePixelFormat(recoverMTL), .w = mglBindingStateTextureWidth(recoverMTL), .h = mglBindingStateTextureHeight(recoverMTL), .copy = recoveredFromSampledCopy ? 1 : 0, .prev_ver = recoveredFromPreviousVersion ? 1 : 0, .sampled_ver = recoverTexture ? recoverTexture->mtl_gl_sampled_write_version : 0u, .rt_ver = recoverTexture ? recoverTexture->mtl_render_target_write_version : 0u, .paired_color = pairedColor ? pairedColor->name : 0u, .paired_current = pairedCur ? 1 : 0);
            }
            ptr = recoverTexture;
            texture = recoverMTL;
        }
        goto done;
    }

    /* ENTER_RT */
    {
        Texture *unitActive =
            textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
        Texture *unit2D = textureUnit < TEXTURE_UNITS
            ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
        Texture *last2D = textureUnit < TEXTURE_UNITS
            ? MGL_STATE(ctx)->last_sampled_2d_textures[textureUnit] : NULL;
        Texture *recoverTexture = NULL;
        const char *recoverReason = "none";
        GLuint recoverFboName = 0u;
        Texture *pairedColor =
            mglFindFramebufferColorTexturePairedWithDepth(ctx, ptr, &recoverFboName);
        if (pairedColor) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:pairedColor]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            id pairedMTL = pairedColor->mtl_data ? (__bridge id)(pairedColor->mtl_data) : nil;
            NSUInteger drawAtt = MAX_COLOR_ATTACHMENTS;
            BOOL pairedCur = mglBindingStateRenderPassUsesColorTexture(
                _renderPassManager.state->renderPassStateOwner, (__bridge void *)pairedMTL,
                &drawAtt);
            MGLDepthRecoverInput rin = {
                .phase = MGL_DR_PHASE_RT,
                .rt_sub = 0,
                .has_paired_color = 1,
                .has_paired_mtl = pairedMTL ? 1 : 0,
                .paired_is_current_draw = pairedCur ? 1 : 0,
                .paired_is_depth_or_stencil =
                    pairedMTL && mglMetalPixelFormatIsDepthOrStencil(
                                     mglBindingStateTexturePixelFormat(pairedMTL)),
                .candidate_type_ok = !pairedMTL || expectedType == 0 ||
                    mglBindingStateTextureType(pairedMTL) == expectedType,
                .candidate_kind_ok = !pairedMTL ||
                    mglTexturePixelFormatCompatibleWithExpectedDataKind(
                        mglBindingStateTexturePixelFormat(pairedMTL), expectedKind),
            };
            MGLDepthRecoverPlan rplan = {0};
            (void)mglBindingTexturePlanDepthRecover(&rin, &rplan);
            if (rplan.action == MGL_DR_ACTION_RT_USE_PAIRED) {
                recoverTexture = pairedColor;
                recoverReason = rplan.reason_tag ? rplan.reason_tag : "paired-color";
            } else if (rplan.action == MGL_DR_ACTION_RT_SKIP_CURRENT) {
                static uint64_t s_skip = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_skip)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_SKIP, .hit = s_skip, .program = fragmentProgramName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .fbo = recoverFboName, .color_att = drawAtt, .depth_tex = ptr ? ptr->name : 0u, .color_tex = pairedColor ? pairedColor->name : 0u);
                }
            }
        }
        if (!recoverTexture &&
            mglRendererTextureLooksRecoverableSampled2D(ctx, last2D, expectedType, expectedKind)) {
            static uint64_t s_sup = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_sup)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_SUPPRESS_LAST2D, .hit = s_sup, .program = fragmentProgramName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .depth_tex = ptr ? ptr->name : 0u, .last2d = last2D->name);
            }
        }
        if (recoverTexture) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:recoverTexture]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            id recoverMTL =
                recoverTexture->mtl_data ? (__bridge id)(recoverTexture->mtl_data) : nil;
            if (recoverMTL &&
                !mglMetalPixelFormatIsDepthOrStencil(
                    mglBindingStateTexturePixelFormat(recoverMTL)) &&
                (expectedType == 0 ||
                 mglBindingStateTextureType(recoverMTL) == expectedType) &&
                mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    mglBindingStateTexturePixelFormat(recoverMTL), expectedKind)) {
                Framebuffer *currentFbo = ctx ? MGL_STATE(ctx)->framebuffer : NULL;
                GLuint colorTexName = 0u, depthTexName = 0u;
                if (currentFbo && mglRendererObjectPointerLikelyValid(currentFbo) &&
                    mglPointerRangeIsReadable(currentFbo, sizeof(*currentFbo))) {
                    colorTexName = currentFbo->color_attachments[0].texture;
                    depthTexName = currentFbo->depth.texture;
                }
                static uint64_t s_rt = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_rt)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_RECOVER, .hit = s_rt, .reason = recoverReason, .program = fragmentProgramName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .depth_tex = ptr ? ptr->name : 0u, .recover_tex = recoverTexture->name, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .recover_fmt = mglBindingStateTexturePixelFormat(recoverMTL), .w = mglBindingStateTextureWidth(texture), .h = mglBindingStateTextureHeight(texture), .level = depthSampleLevel0, .ever = depthSampleLevel0 ? depthSampleLevel0->ever_written : 0u, .init = depthSampleLevel0 ? depthSampleLevel0->has_initialized_data : 0u, .unit_active = mglTraceTextureName(unitActive), .unit_tex2d = mglTraceTextureName(unit2D), .unit_last2d = mglTraceTextureName(last2D), .recover_fbo = recoverFboName, .current_fbo = currentFbo ? currentFbo->name : 0u, .color_tex = colorTexName, .fbo_depth_tex = depthTexName);
                }
                ptr = recoverTexture;
                texture = recoverMTL;
            }
        }
        if (texture &&
            mglMetalPixelFormatIsDepthOrStencil(mglBindingStateTexturePixelFormat(texture))) {
            id fallbackTexture =
                [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
            if (fallbackTexture) {
                static uint64_t s_fb = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_fb)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_FALLBACK, .hit = s_fb, .program = fragmentProgramName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .depth_tex = ptr ? ptr->name : 0u, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .w = mglBindingStateTextureWidth(texture), .h = mglBindingStateTextureHeight(texture), .level = depthSampleLevel0, .ever = depthSampleLevel0 ? depthSampleLevel0->ever_written : 0u, .init = depthSampleLevel0 ? depthSampleLevel0->has_initialized_data : 0u, .unit_active = mglTraceTextureName(unitActive), .unit_tex2d = mglTraceTextureName(unit2D), .unit_last2d = mglTraceTextureName(last2D));
                }
                texture = fallbackTexture;
                usedFallbackTexture = YES;
            }
        }
    }

done:
    *ptrPtr = ptr;
    *texturePtr = texture;
    *suppressMissingTextureFallbackPtr = suppressMissingTextureFallback;
    *usedFallbackTexturePtr = usedFallbackTexture;
    return true;
}



/* O3.3: collapsed TBIND + sample-detail + texbuffer + FS gui/readback ports. */
- (void)emitSampledDiagPortsForProgram:(Program *)program
                                 stage:(const char *)stage
                       stageIsFragment:(BOOL)stageIsFragment
                           sampledName:(const char *)sampledName
                          spirvBinding:(GLuint)spirvBinding
                           textureUnit:(GLuint)textureUnit
                      sampledResource:(MGLShaderResource *)sampledResource
                                   ptr:(Texture *)ptr
                               texture:(id)texture
                               sampler:(id)sampler
                         usedFallback:(BOOL)usedFallback
                        expectedType:(uint32_t)expectedType
                          lookupType:(uint32_t)lookupType
                             bindCall:(uint64_t)bindCall
                          programName:(GLuint)programName
                     vertexProgramName:(GLuint)vertexProgramName
                   fragmentProgramName:(GLuint)fragmentProgramName
                  usedSampledCopyTrace:(BOOL)usedSampledCopyTrace
                 directTextureForTrace:(id)directTextureForTrace
                 sampledCopyForTrace:(id)sampledCopyForTrace
                     focusedCounter:(uint64_t *)focusedCounter
                   traceFileCounter:(uint64_t *)traceFileCounter
{
    TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
    int expectedIndex =
        [self textureIndexForExpectedMetalType:(lookupType ? lookupType
                                                           : expectedType)];
    Texture *unitActive = NULL, *unitExpected = NULL, *unit2D = NULL, *unitCube = NULL;
    if (textureUnit < TEXTURE_UNITS) {
        unitActive = MGL_STATE(ctx)->active_textures[textureUnit];
        unit2D = MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D];
        unitCube = MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP];
        if (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES) {
            unitExpected =
                MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex];
        }
    }

    if (mglProgramNeedsBindingTrace(program) &&
        mglShouldLogFocusedBinding(focusedCounter)) {
        mglBindingLogTBINDFocused(
            stage, program ? program->name : 0u, sampledName, spirvBinding,
            textureUnit, ptr ? ptr->name : 0u, ptr ? ptr->target : 0u, texture,
            texture ? mglBindingStateTextureType(texture) : 0,
            texture ? mglBindingStateTextureWidth(texture) : 0,
            texture ? mglBindingStateTextureHeight(texture) : 0,
            level0 ? level0->width : 0u, level0 ? level0->height : 0u,
            level0 ? level0->ever_written : 0u,
            level0 ? level0->has_initialized_data : 0u,
            level0 ? level0->last_init_source : 0u);
    }
    if (mglProgramNeedsTraceLog(program) &&
        mglShouldLogTraceFileBindingForProgram(program, traceFileCounter)) {
        mglBindingLogTBINDTraceFile(
            stage, program ? program->name : 0u, sampledName, spirvBinding,
            textureUnit, sampledResource ? (int)sampledResource->sampler_unit : -1,
            (sampledResource && sampledResource->sampler_unit_explicit) ? 1 : 0,
            ptr ? ptr->name : 0u, ptr ? ptr->target : 0u, usedFallback ? 1 : 0,
            expectedType, lookupType, expectedIndex, mglTraceTextureName(unitActive),
            mglTraceTextureName(unitExpected), mglTraceTextureName(unit2D),
            mglTraceTextureName(unitCube), texture,
            texture ? mglBindingStateTextureType(texture) : 0,
            texture ? mglBindingStateTextureWidth(texture) : 0,
            texture ? mglBindingStateTextureHeight(texture) : 0,
            level0 ? level0->width : 0u, level0 ? level0->height : 0u,
            level0 ? level0->ever_written : 0u,
            level0 ? level0->has_initialized_data : 0u,
            level0 ? level0->last_init_source : 0u);
    }

    MGLSampledDiagGateInput din = {
        .stage_is_fragment = stageIsFragment ? 1 : 0,
        .used_fallback =
            usedFallback || (stageIsFragment && ptr && ptr->name == 13u) ? 1 : 0,
        .is_gui_rt_copy_eligible =
            stageIsFragment && ptr &&
                    mglTextureCanUseGLSampledRenderTargetCopy(ptr)
                ? 1
                : 0,
        .focused_loading_window =
            stageIsFragment && mglIsFocusedLoadingProgram(programName) &&
                    (bindCall <= 2048ull || ((bindCall % 512ull) == 0ull))
                ? 1
                : 0,
        .vertex_focus_program =
            !stageIsFragment &&
                    ((program && program->name == 34u) ||
                     (!program && programName == 34u))
                ? 1
                : 0,
        .level0_suspicious_zero = level0 && level0->suspicious_zero_upload,
        .level0_never_written = level0 && !level0->ever_written,
        .level0_uninit = level0 && !level0->has_initialized_data,
        .has_bound_texture = texture ? 1 : 0,
        .is_texel_buffer =
            ptr && mglRenderTextureTargetIsBuffer((uint32_t)ptr->target) ? 1 : 0,
    };
    MGLSampledDiagGatePlan dplan = {0};
    (void)mglBindingTexturePlanSampledDiag(&din, &dplan);

    if (dplan.log_texel_buffer) {
        static uint64_t s_texelBufferBindLogs = 0;
        if (mglBindingTextureRateLogHit(&s_texelBufferBindLogs, 8ull, 2048ull)) {
            Texture *unitBuffer =
                textureUnit < TEXTURE_UNITS
                    ? MGL_STATE(ctx)
                          ->texture_units[textureUnit]
                          .textures[_TEXTURE_BUFFER_TARGET]
                    : NULL;
            mglBindingLogTexBufferBind(
                s_texelBufferBindLogs, programName, spirvBinding, textureUnit,
                ptr ? ptr->name : 0u, mglTraceTextureName(unitActive),
                mglTraceTextureName(unitBuffer), expectedType, lookupType, texture,
                texture ? mglBindingStateTextureType(texture) : 0,
                texture ? mglBindingStateTextureWidth(texture) : 0,
                texture ? mglBindingStateTextureHeight(texture) : 0,
                texture ? mglBindingStateTexturePixelFormat(texture) : 0, sampler);
        }
    }
    if (dplan.log_detail) {
        static uint64_t s_sampleDetailLogCount[2] = {0, 0};
        uint64_t *ctr = &s_sampleDetailLogCount[stageIsFragment ? 1 : 0];
        uint64_t early = stageIsFragment ? 256ull : 128ull;
        if (mglBindingTextureRateLogHit(ctr, early, 512ull)) {
            uint64_t levelDataHash =
                (level0 && level0->data && level0->data_size > 0)
                    ? mglTraceHashBytes((const void *)(uintptr_t)level0->data,
                                        level0->data_size)
                    : 0ull;
            mglBindingLogSampleDetail(
                bindCall, *ctr, stage, programName, sampledName, spirvBinding,
                textureUnit, expectedType, expectedIndex, mglTraceTextureName(ptr),
                ptr, ptr ? ptr->target : 0u, usedFallback ? 1 : 0, texture,
                texture ? mglBindingStateTextureType(texture) : 0,
                texture ? mglBindingStateTextureWidth(texture) : 0,
                texture ? mglBindingStateTextureHeight(texture) : 0,
                mglTraceTextureName(unitActive), mglTraceTextureName(unitExpected),
                stageIsFragment ? mglTraceTextureName(unit2D) : 0u,
                stageIsFragment ? mglTraceTextureName(unitCube) : 0u,
                level0 ? level0->width : 0u, level0 ? level0->height : 0u,
                level0 ? level0->depth : 0u, level0 ? level0->data_size : 0u,
                level0 ? level0->ever_written : 0u,
                level0 ? level0->has_initialized_data : 0u,
                level0 ? level0->suspicious_zero_upload : 0u,
                level0 ? level0->last_init_source : 0u,
                level0 ? level0->last_upload_size : 0u,
                level0 ? (void *)level0->last_src_ptr : NULL,
                level0 ? level0->last_src_hash : 0ull, levelDataHash);
        }
    }
    if (dplan.log_gui_rt) {
        static uint64_t s_guiRTSampleLogCount = 0;
        if (mglBindingTextureRateLogHit(&s_guiRTSampleLogCount, 128ull, 256ull)) {
            id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
            id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            mglBindingLogRTSampleCopySample(
                s_guiRTSampleLogCount, bindCall, programName, vertexProgramName,
                fragmentProgramName, sampledName, spirvBinding, textureUnit,
                mglTraceTextureName(ptr), mglTraceTextureLabel(ptr),
                usedFallback ? 1 : 0, usedSampledCopyTrace ? 1 : 0, ptr, texture,
                directTextureForTrace, sampledCopyForTrace,
                texture ? mglBindingStateTexturePixelFormat(texture)
                        : MGL_BINDING_PIXEL_FORMAT_INVALID,
                texture ? mglBindingStateTextureType(texture) : 0,
                texture ? mglBindingStateTextureWidth(texture) : 0,
                texture ? mglBindingStateTextureHeight(texture) : 0,
                ctx && MGL_STATE(ctx)->framebuffer ? MGL_STATE(ctx)->framebuffer->name
                                                   : 0u,
                _renderPassManager.state->renderPassFramebufferName, rpColor0,
                rpDepth);
        }
    }
    if (dplan.log_readback && texture && level0) {
        static uint64_t s_sampleReadbackCount = 0;
        if (mglBindingTextureRateLogHit(&s_sampleReadbackCount, 32ull, 512ull)) {
            [self traceSampledTextureReadback:texture
                                        glTex:ptr
                                        level:level0
                                      program:programName
                                      binding:spirvBinding
                                        stage:stageIsFragment ? @"fragment"
                                                              : @"vertex"
                                       reason:(level0->suspicious_zero_upload
                                                   ? @"zero-level"
                                                   : (!level0->ever_written
                                                          ? @"never-written"
                                                          : @"not-initialized"))
                                          hit:s_sampleReadbackCount];
        }
    }
}

/* O3.3: COMPAT type/kind fallback — plan@C + thin fallback port (V/F shared). */
- (id)applySampledCompatFallbackPlan:(Texture *)ptr
                             texture:(id)texture
                        expectedType:(uint32_t)expectedType
                        expectedKind:(MGLTextureDataKind)expectedKind
                               stage:(const char *)stage
                         programName:(GLuint)programName
                        spirvBinding:(GLuint)spirvBinding
                       sampleProgram:(Program *)sampleProgram
                    usedFallbackOut:(BOOL *)usedFallbackOut
{
    MGLSampledTextureBindInput cin = {0};
    cin.phase = MGL_ST_PHASE_COMPAT;
    cin.has_mtl_texture = texture ? 1 : 0;
    cin.mtl_type = texture ? mglBindingStateTextureType(texture) : 0u;
    cin.expected_type = expectedType;
    cin.format_kind_ok =
        !texture ||
                mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    mglBindingStateTexturePixelFormat(texture), expectedKind)
            ? 1
            : 0;
    MGLSampledTextureBindPlan cplan = {0};
    if (mglBindingTexturePlanSampled(&cin, &cplan) != 0 ||
        (cplan.action != MGL_ST_ACTION_TYPE_FALLBACK &&
         cplan.action != MGL_ST_ACTION_KIND_FALLBACK)) {
        return texture;
    }
    static uint64_t s_compatMismatchLogCount = 0;
    if (mglBindingTextureRateLogHit(&s_compatMismatchLogCount, 32ull, 512ull)) {
        mglBindingLogTexCompatMismatch(
            cplan.action == MGL_ST_ACTION_TYPE_FALLBACK ? "TYPE" : "DATA",
            stage, spirvBinding, programName, ptr ? ptr->name : 0u, cin.mtl_type,
            expectedType, s_compatMismatchLogCount);
    }
    if (sampleProgram) {
        mglWriteProgramMSLDump(
            sampleProgram,
            [NSString stringWithFormat:@"tex-%@-mismatch-%s-binding-%u",
                                       cplan.action == MGL_ST_ACTION_TYPE_FALLBACK
                                           ? @"type"
                                           : @"data",
                                       stage ? stage : "x", spirvBinding]);
    }
    texture = [self fallbackSampledTextureForExpectedType:expectedType
                                                 dataKind:expectedKind];
    if (usedFallbackOut) {
        *usedFallbackOut = YES;
    }
    return texture;
}

/* O3.3: sampler materialize — plan@C + thin createMTLSampler / bridge ports. */
- (id)materializeSampledSamplerForTexture:(Texture *)ptr
                              textureUnit:(GLuint)textureUnit
                          defaultSampler:(id)defaultSampler
                            forceDefault:(BOOL)forceDefault
                           samplerTarget:(GLuint)samplerTarget
                             programName:(GLuint)programName
                            spirvBinding:(GLuint)spirvBinding
                                   stage:(const char *)stage
                                 texture:(id)texture
{
    Sampler *glSampler = (textureUnit < TEXTURE_UNITS)
                             ? MGL_STATE(ctx)->texture_samplers[textureUnit]
                             : NULL;
    MGLSamplerMaterializeInput in = {
        .force_default = forceDefault ? 1 : 0,
        .unit_in_range = textureUnit < TEXTURE_UNITS ? 1 : 0,
        .has_gl_sampler = glSampler ? 1 : 0,
        .gl_sampler_dirty = glSampler && glSampler->dirty_bits ? 1 : 0,
        .has_gl_sampler_mtl = glSampler && glSampler->mtl_data ? 1 : 0,
        .has_tex_params_mtl = ptr && ptr->params.mtl_data ? 1 : 0,
        .require_tex_params_mtl = (stage && stage[0] == 'v') ? 1 : 0,
    };
    MGLSamplerMaterializePlan plan = {0};
    if (mglBindingTexturePlanSamplerMaterialize(&in, &plan) != 0) {
        return defaultSampler;
    }
    if (plan.action == MGL_SM_ACTION_USE_DEFAULT) {
        return defaultSampler;
    }
    id sampler = defaultSampler;
    const TextureParameter *params = NULL;
    GLuint samplerName = 0u;
    if (plan.action == MGL_SM_ACTION_USE_GL_SAMPLER && glSampler) {
        if (plan.recreate_gl_sampler_mtl) {
            if (glSampler->mtl_data) {
                mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
            }
            GLuint target = samplerTarget
                                ? samplerTarget
                                : (ptr ? ptr->target : GL_TEXTURE_2D);
            glSampler->mtl_data = (void *)CFBridgingRetain(
                [self createMTLSamplerForTexParam:&glSampler->params target:target]);
        }
        if (plan.clear_gl_sampler_dirty) {
            glSampler->dirty_bits = 0;
        }
        sampler = (__bridge id)(glSampler->mtl_data);
        params = &glSampler->params;
        samplerName = glSampler->name;
    } else if (plan.action == MGL_SM_ACTION_USE_TEX_PARAMS && ptr) {
        sampler = (__bridge id)(ptr->params.mtl_data);
        params = &ptr->params;
    } else {
        return defaultSampler;
    }
    if (params && mglTraceLogIsEnabled()) {
        mglBindingLogSamplerResolve(
            (stage && stage[0] == 'v') ? "VERT" : "FRAG", programName,
            spirvBinding, textureUnit, plan.source_tag ? plan.source_tag : "?",
            samplerName, params->min_filter, params->mag_filter, params->wrap_s,
            params->wrap_t, params->min_lod, params->max_lod,
            ptr ? ptr->name : 0u, ptr ? ptr->params.base_level : 0u,
            ptr ? ptr->params.max_level : 0u, ptr ? ptr->width : 0u,
            ptr ? ptr->height : 0u,
            texture ? mglBindingStateTextureWidth(texture) : 0u,
            texture ? mglBindingStateTextureHeight(texture) : 0u,
            texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u);
    }
    return sampler;
}

/* O3.3: shared RT sampled-copy spine — plan@C + thin texture ports. */
- (bool)applySampledRenderTargetCopyPlan:(Texture *)ptr
                                 texture:(id *)texturePtr
                             sampleProgram:(Program *)sampleProgram
                              expectedType:(uint32_t)expectedType
                              expectedKind:(MGLTextureDataKind)expectedKind
                         usedTypeFallback:(BOOL)usedTypeFallback
                                   stage:(const char *)stage
                            programName:(GLuint)programName
                            spirvBinding:(GLuint)spirvBinding
                              textureUnit:(GLuint)textureUnit
                              sampledName:(const char *)sampledName
                     usedSampledCopyOut:(BOOL *)usedSampledCopyOut
                   directTextureForTrace:(id *)directTextureForTrace
                   sampledCopyForTrace:(id *)sampledCopyForTrace
{
    if (!texturePtr || usedTypeFallback || !ptr || !ptr->is_render_target) {
        return true;
    }
    id texture = *texturePtr;
    MGLYFlipDecision yflip = mglDecideYFlipForSampledRT(ptr, sampleProgram);
    if (mglTraceRTYFlipDiagnosticsEnabled()) {
        mglBindingLogRTYFlipDecision(
            stage, programName, sampledName, spirvBinding, textureUnit, ptr->name,
            mglTraceTextureLabel(ptr), mglYFlipDecisionName(yflip), (int)yflip,
            ptr->mtl_render_yflip_authority, ptr->mtl_render_target_write_version,
            ptr->mtl_gl_sampled_write_version, ptr->mtl_gl_sampled_data ? 1 : 0,
            mglProgramHasExistingFramebufferSampleYFlip(sampleProgram) ? 1 : 0);
    }

    MGLSampledTextureBindInput in = {0};
    in.phase = MGL_ST_PHASE_RT;
    in.used_type_fallback = usedTypeFallback ? 1 : 0;
    in.is_render_target = 1;
    in.yflip = (int)yflip;
    in.has_sampled_copy = ptr->mtl_gl_sampled_data ? 1 : 0;
    in.copy_fresh = mglGLSampledCopyContentFresh(ptr) ? 1 : 0;
    in.can_use_rt_copy = mglTextureCanUseGLSampledRenderTargetCopy(ptr) ? 1 : 0;
    in.want_base_level_on_original =
        (stage && stage[0] == 'f') ? 1 : 0;
    id sampledCopy = ptr->mtl_gl_sampled_data
                         ? (__bridge id)(ptr->mtl_gl_sampled_data)
                         : nil;
    in.copy_type_ok =
        sampledCopy &&
                (expectedType == 0 ||
                 mglBindingStateTextureType(sampledCopy) == expectedType)
            ? 1
            : 0;
    in.copy_kind_ok =
        sampledCopy &&
                mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    mglBindingStateTexturePixelFormat(sampledCopy), expectedKind)
            ? 1
            : 0;

    MGLSampledTextureBindPlan plan = {0};
    if (mglBindingTexturePlanSampled(&in, &plan) != 0) {
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_USE_COPY && sampledCopy) {
        if (directTextureForTrace) {
            *directTextureForTrace = texture;
        }
        if (sampledCopyForTrace) {
            *sampledCopyForTrace = sampledCopy;
        }
        if (mglTraceLogIsEnabled()) {
            MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_BIND, .stage = stage, .program = programName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .tex = ptr->name, .label = mglTraceTextureLabel(ptr), .original = (__bridge const void *)texture, .copy = (__bridge const void *)sampledCopy);
        }
        id chosen = sampledCopy;
        if (plan.apply_base_level_view) {
            chosen = (__bridge id)mglSampledTextureViewForBaseLevel(
                ptr, (__bridge void *)sampledCopy);
        }
        *texturePtr = chosen;
        if (usedSampledCopyOut) {
            *usedSampledCopyOut = YES;
        }
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_REPAIR) {
        id repairedCopy =
            [self freshGLSampledRenderTargetCopyForSampling:ptr
                                                      source:texture
                                                       stage:stage
                                                     program:programName
                                                     binding:spirvBinding
                                                        unit:textureUnit
                                                expectedType:expectedType
                                                expectedKind:expectedKind];
        if (!repairedCopy) {
            return true;
        }
        in.repaired_available = 1;
        in.repaired_fresh = mglGLSampledCopyContentFresh(ptr) ? 1 : 0;
        if (mglBindingTexturePlanSampled(&in, &plan) != 0) {
            return true;
        }
        if (plan.action == MGL_ST_ACTION_RT_RETRY) {
            return false;
        }
        if (plan.action == MGL_ST_ACTION_RT_USE_COPY) {
            id chosen = repairedCopy;
            if (plan.apply_base_level_view) {
                chosen = (__bridge id)mglSampledTextureViewForBaseLevel(
                    ptr, (__bridge void *)repairedCopy);
            }
            *texturePtr = chosen;
            if (usedSampledCopyOut) {
                *usedSampledCopyOut = YES;
            }
        }
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_GATE_MISS && mglTraceLogIsEnabled()) {
        MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_GATE_MISS, .stage = stage, .program = programName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .tex = ptr->name, .label = mglTraceTextureLabel(ptr), .is_rt = 1, .has_copy = ptr->mtl_gl_sampled_data ? 1 : 0, .can_use = in.can_use_rt_copy, .expected_type = expectedType);
    } else if (plan.action == MGL_ST_ACTION_RT_ORIGINAL) {
        static uint64_t s_rtSampleCopySkipExistingFlipLogCount = 0;
        if (mglTraceLogIsEnabled() &&
            mglBindingTextureRateLogHit(&s_rtSampleCopySkipExistingFlipLogCount,
                                        32ull, 512ull)) {
            MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_SKIP_YFLIP, .hit = s_rtSampleCopySkipExistingFlipLogCount, .stage = stage, .program = programName, .name = sampledName, .binding = spirvBinding, .tex = ptr ? ptr->name : 0u, .decision_name = mglYFlipDecisionName(yflip), .decision = (int)yflip);
        }
        if (plan.apply_base_level_view && texture) {
            *texturePtr = (__bridge id)mglSampledTextureViewForBaseLevel(
                ptr, (__bridge void *)texture);
        }
    }
    return true;
}

- (bool)bindStorageImagesForStage:(int)shaderStage
                          program:(Program *)program
                        bindStage:(uint32_t)metalBindStage
{
    const BOOL useResourceSnapshot = YES;
    MGLRenderResourceBindingSnapshot resourceSnapshot = {0};
    GLuint count =
        mglRendererGetProgramBindingCount(ctx, shaderStage, _STORAGE_IMAGE_RES);
    const char *restoreTag =
        metalBindStage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? "vs-storage-image-bind"
            : "storage-image-bind";

    for (int pass = MGL_SI_PASS_ENSURE; pass <= MGL_SI_PASS_BIND; pass++) {
        for (GLuint i = 0; i < count; i++) {
            MGLShaderResource *resource = NULL;
            GLuint element = 0u;
            if (program) {
                MGLShaderResourceList *list =
                    &program->shader_resources_list[shaderStage][_STORAGE_IMAGE_RES];
                GLuint ordinal = i;
                for (GLuint ri = 0; ri < list->count; ri++) {
                    GLuint elements = mglRenderShaderResourceElementCount(
                        (uint32_t)list->list[ri].gl_array_size);
                    if (ordinal < elements) {
                        resource = &list->list[ri];
                        element = ordinal;
                        break;
                    }
                    ordinal -= elements;
                }
            }
            const uint32_t fallbackMetal =
                (GLuint)mglRendererGetProgramBinding(ctx, shaderStage,
                                                     _STORAGE_IMAGE_RES, (int)i);
            const uint32_t provisionalSlot = mglRenderResourceMetalSlot(
                resource ? 1 : 0, resource ? resource->binding : 0u, element,
                fallbackMetal);
            const int explicitUnit =
                program && provisionalSlot < TEXTURE_UNITS &&
                program->sampler_units_explicit_by_stage[shaderStage][provisionalSlot];
            MGLStorageImageBindInput in = {
                .pass = pass,
                .skip_resource = mglShouldSkipStageTextureResource(
                                     program, shaderStage, _STORAGE_IMAGE_RES,
                                     resource)
                                     ? 1
                                     : 0,
                .has_resource = resource ? 1 : 0,
                .resource_binding = resource ? resource->binding : 0u,
                .element = element,
                .fallback_metal_slot = fallbackMetal,
                .use_resource_unit = (explicitUnit || resource) ? 1 : 0,
                .explicit_by_slot = explicitUnit ? 1 : 0,
                .explicit_unit =
                    explicitUnit
                        ? (uint32_t)program
                              ->sampler_units_by_stage[shaderStage][provisionalSlot]
                        : 0u,
                .sampler_unit = resource ? resource->sampler_unit : -1,
                .resource_gl_binding = resource ? resource->gl_binding : 0u,
                .fallback_gl_binding =
                    (GLuint)mglRendererGetProgramGLBinding(
                        ctx, shaderStage, _STORAGE_IMAGE_RES, (int)i),
                .max_units = TEXTURE_UNITS,
            };
            MGLStorageImageBindPlan plan = {0};
            if (mglBindingTexturePlanStorageImage(&in, &plan) != 0 ||
                plan.action == MGL_SI_ACTION_SKIP) {
                continue;
            }
            Texture *ptr = plan.gl_unit < TEXTURE_UNITS
                               ? MGL_STATE(ctx)->image_units[plan.gl_unit].tex
                               : NULL;
            if (plan.action == MGL_SI_ACTION_ENSURE_TEX) {
                if (ptr) {
                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
                }
                continue;
            }
            id texture = nil;
            if (ptr) {
                MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
                texture = (__bridge id)(ptr->mtl_data);
                texture = mglBindingStateCreateStorageImageView(
                    texture, &MGL_STATE(ctx)->image_units[plan.gl_unit]);
            }
            if (!mglBindingStateQueueResourceBinding(
                    useResourceSnapshot, _bindingStateOwner,
                    _renderPassManager.state->currentRenderEncoderOwner,
                    &resourceSnapshot, metalBindStage,
                    MGL_RENDER_RESOURCE_BINDING_TEXTURE,
                    (__bridge void *)texture, plan.metal_slot)) {
                return false;
            }
        }
        if (pass == MGL_SI_PASS_ENSURE &&
            mglRenderEncoderOwnerHasCurrent(
                _renderPassManager.state->currentRenderEncoderOwner) == 0) {
            RETURN_FALSE_ON_FAILURE(
                [self restoreRenderEncoderAfterTextureUploadForDraw:restoreTag]);
        }
    }
    if (useResourceSnapshot &&
        !mglBindingStateFlushResourceBindings(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            &resourceSnapshot)) {
        return false;
    }
    return true;
}

- (bool)bindStorageImagesForVertexProgram:(Program *)vertexProgram
                          fragmentProgram:(Program *)fragmentProgram
{
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    if (![self bindStorageImagesForStage:vertexStage
                                 program:vertexProgram
                               bindStage:MGL_RENDER_BINDING_STAGE_VERTEX]) {
        return false;
    }
    if (![self bindStorageImagesForStage:_FRAGMENT_SHADER
                                 program:fragmentProgram
                               bindStage:MGL_RENDER_BINDING_STAGE_FRAGMENT]) {
        return false;
    }
    return true;
}

- (bool)bindSeparateSamplersAndArrayTextures:(Program *)vertexProgram
                              fragmentProgram:(Program *)fragmentProgram
                        fragmentProgramName:(GLuint)fragmentProgramName
                          vertexProgramName:(GLuint)vertexProgramName
                             defaultSampler:(id)defaultSampler
                                    bindCall:(uint64_t)bindCall
                                  traceBind:(bool)traceBind
                         separateSamplerCount:(GLuint *)separateSamplerCount
                           boundSeparateSamplers:(GLuint *)boundSeparateSamplers
{
    const BOOL useResourceSnapshot = YES;
    MGLRenderResourceBindingSnapshot resourceSnapshot = {0};
    // Bind separate samplers explicitly.
    *separateSamplerCount = mglRendererGetProgramBindingCount(ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES);
    *boundSeparateSamplers = 0;
    for (GLuint i = 0; i < *separateSamplerCount; i++)
    {
        GLuint spirvBinding = mglRendererGetProgramBinding(ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES, (int)i);
        GLuint glBinding = mglRendererGetProgramGLBinding(ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES, (int)i);
        if (!mglBindingTextureSeparateSamplerInRange(spirvBinding, glBinding,
                                                     TEXTURE_UNITS)) {
            continue;
        }
        Program *sampleProgram = fragmentProgram;
        MGLShaderResource *samplerResource = NULL;
        if (sampleProgram &&
            i < sampleProgram->shader_resources_list[_FRAGMENT_SHADER][_SEPARATE_SAMPLERS_RES].count) {
            samplerResource = &sampleProgram->shader_resources_list[_FRAGMENT_SHADER][_SEPARATE_SAMPLERS_RES].list[i];
        }
        if (mglShouldSkipStageSamplerResource(sampleProgram,
                                              _FRAGMENT_SHADER,
                                              _SEPARATE_SAMPLERS_RES,
                                              samplerResource)) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:samplerResource
                                                    metalBinding:spirvBinding
                                                           stage:_FRAGMENT_SHADER];

        id sampler = [self materializeSampledSamplerForTexture:NULL
                                                   textureUnit:textureUnit
                                               defaultSampler:defaultSampler
                                                 forceDefault:NO
                                                samplerTarget:(GLuint)mglRenderSamplerObjectTarget()
                                                  programName:fragmentProgramName
                                                 spirvBinding:spirvBinding
                                                        stage:"fragment"
                                                      texture:nil];
        if (sampler && spirvBinding < kMaxFragmentSamplerSlots) {
            if (!mglBindingStateQueueResourceBinding(
                    useResourceSnapshot, _bindingStateOwner,
                    _renderPassManager.state->currentRenderEncoderOwner,
                    &resourceSnapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    (__bridge void *)sampler, spirvBinding)) {
                return false;
            }
            boundSeparateSamplers++;
        }

        if (traceBind && i < 6) {
            mglTraceLogNSString(@"MGL TRACE texbind.separateSampler call=%llu idx=%u binding=%u unit=%u sampler=%p",
                  (unsigned long long)bindCall,
                  (unsigned)i,
                  (unsigned)spirvBinding,
                  (unsigned)textureUnit,
                  sampler);
        }
    }

    Program *arrayPrograms[] = { vertexProgram, fragmentProgram };
    int arrayStages[] = {
        _tessellation.nativeTESActive
            ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER,
        _FRAGMENT_SHADER
    };
    for (NSUInteger programIndex = 0; programIndex < 2; programIndex++) {
        Program *arrayProgram = arrayPrograms[programIndex];
        int arrayStage = arrayStages[programIndex];
        if (!arrayProgram) {
            continue;
        }

        MGLShaderResourceList *arrayResources =
            &arrayProgram->shader_resources_list[arrayStage][_SAMPLED_IMAGE_RES];
        for (GLuint resourceIndex = 0; arrayResources->list && resourceIndex < arrayResources->count; resourceIndex++) {
            MGLShaderResource *resource = &arrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            uint32_t expectedType = (uint32_t)
                mglRendererGetProgramExpectedTextureType(ctx, arrayStage, _SAMPLED_IMAGE_RES, (int)resourceIndex);
            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                GLuint samplerSlot =
                    mglMetalCombinedSamplerSlotForElement(resource,
                                                          (GLuint)element);
                if (!mglBindingTextureArrayElementSlotOk(metalSlot, TEXTURE_UNITS)) {
                    break;
                }

                GLuint textureUnit = [self textureUnitForSampledResource:NULL
                                                             metalBinding:metalSlot
                                                                    stage:arrayStage];
                Texture *arrayTexture = [self textureForSampledResource:NULL
                                                            metalBinding:metalSlot
                                                                    stage:arrayStage
                                                             expectedType:expectedType];
                id metalTexture = nil;
                id metalSampler = defaultSampler;
                if (arrayTexture && [self bindMTLTexture:arrayTexture]) {
                    metalTexture = (__bridge id)(arrayTexture->mtl_data);
                    metalSampler = [self materializeSampledSamplerForTexture:arrayTexture
                                                                 textureUnit:textureUnit
                                                             defaultSampler:defaultSampler
                                                               forceDefault:NO
                                                             samplerTarget:arrayTexture->target
                                                                programName:arrayProgram->name
                                                               spirvBinding:metalSlot
                                                                      stage:"vertex"
                                                                    texture:metalTexture];
                }
                if (!metalTexture) {
                    metalTexture = [self fallbackSampledTextureForExpectedType:expectedType
                                                                      dataKind:MGLTextureDataKindFloat];
                }

                uint32_t bindStage =
                    mglRenderTextureBindingStageForShader(arrayStage);
                if (!mglBindingStateQueueResourceBinding(
                        useResourceSnapshot, _bindingStateOwner,
                        _renderPassManager.state->currentRenderEncoderOwner,
                        &resourceSnapshot, bindStage,
                        MGL_RENDER_RESOURCE_BINDING_TEXTURE,
                        (__bridge void *)metalTexture, metalSlot)) {
                    return false;
                }
                if (mglBindingTextureShouldBindCombinedSampler(
                        resource->has_combined_sampler ? 1 : 0,
                        metalSampler ? 1 : 0, samplerSlot,
                        kMaxFragmentSamplerSlots)) {
                    if (!mglBindingStateQueueResourceBinding(
                            useResourceSnapshot, _bindingStateOwner,
                            _renderPassManager.state->currentRenderEncoderOwner,
                            &resourceSnapshot, bindStage,
                            MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                            (__bridge void *)metalSampler, samplerSlot)) {
                        return false;
                    }
                }
            }
        }
    }
    if (useResourceSnapshot &&
        !mglBindingStateFlushResourceBindings(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            &resourceSnapshot)) {
        return false;
    }
    return true;
}


@end
