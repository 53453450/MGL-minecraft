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

#define MGL_VBIND_FLUSH_SNAPSHOT()                                              \
    do {                                                                        \
        if (vbindSnapshot.vertex_op_count > 0) {                                \
            mglRenderEncodeBindingSnapshotForRenderEncoderOwner(             \
                encCtx->render_encoder_owner, &vbindSnapshot, NULL, 0);         \
            vbindSnapshot = (MGLRenderBindingSnapshot){0};                   \
            vbindByteScratchUsed = 0;                                           \
        }                                                                       \
    } while (0)

#define MGL_VBIND_COLLECT_BUFFER(slot, bufPtr, off)                             \
    do {                                                                        \
        if (vbindSnapshot.vertex_op_count >=                                    \
            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                          \
            MGL_VBIND_FLUSH_SNAPSHOT();                                         \
        }                                                                       \
        vbindSnapshot.vertex_ops[vbindSnapshot.vertex_op_count++] =             \
            (MGLRenderBindingOp){/* kind */ 0u,                              \
                                    /* index */ (uint32_t)(slot),               \
                                    /* offset */ (uint64_t)(off),               \
                                    /* buffer */ (void *)(bufPtr),              \
                                    /* bytes */ NULL,                           \
                                    /* length */ 0u};                           \
    } while (0)

#define MGL_VBIND_COLLECT_BYTES(slot, src, len)                                 \
    do {                                                                        \
        const void *src_ = (src);                                               \
        size_t len_ = (len);                                                    \
        if (vbindByteScratchUsed + len_ > sizeof(vbindByteScratch)) {           \
            MGL_VBIND_FLUSH_SNAPSHOT();                                         \
        }                                                                       \
        if (vbindSnapshot.vertex_op_count >=                                    \
            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                          \
            MGL_VBIND_FLUSH_SNAPSHOT();                                         \
        }                                                                       \
        uint8_t *dst_ = vbindByteScratch + vbindByteScratchUsed;                \
        memcpy(dst_, src_, len_);                                               \
        vbindByteScratchUsed += len_;                                           \
        vbindSnapshot.vertex_ops[vbindSnapshot.vertex_op_count++] =             \
            (MGLRenderBindingOp){/* kind */ 1u,                              \
                                    /* index */ (uint32_t)(slot),               \
                                    /* offset */ 0,                             \
                                    /* buffer */ NULL,                          \
                                    /* bytes */ dst_,                           \
                                    /* length */ (uint32_t)len_};               \
    } while (0)

#define MGL_VBIND_EMIT_BUFFER(slot, bufPtr, off)                                \
    do {                                                                        \
        if (useVertexBindingSnapshot) {                                         \
            MGL_VBIND_COLLECT_BUFFER(slot, bufPtr, off);                        \
        } else {                                                                \
            mglBindingStateSetVertexBuffer(                                     \
                encCtx->render_encoder_owner,                  \
                (__bridge id)(bufPtr),                           \
                (off), (slot));                                                 \
        }                                                                       \
    } while (0)

#define MGL_VBIND_EMIT_BYTES(slot, src, len)                                    \
    do {                                                                        \
        if (useVertexBindingSnapshot) {                                         \
            MGL_VBIND_COLLECT_BYTES(slot, src, len);                            \
        } else {                                                                \
            mglBindingStateSetVertexBytes(                                      \
                encCtx->render_encoder_owner,                  \
                (src), (len), (slot));                                          \
        }                                                                       \
    } while (0)

#define MGL_VBIND_EMIT_CLEAR(slot)                                              \
    do {                                                                        \
        if (useVertexBindingSnapshot) {                                         \
            MGL_VBIND_COLLECT_BUFFER(slot, NULL, 0);                            \
        } else {                                                                \
            mglBindingStateSetVertexBuffer(                                     \
                encCtx->render_encoder_owner,                  \
                nil, 0, (slot));                                                \
        }                                                                       \
    } while (0)

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
            Buffer *attribBuffer = mglRendererResolveVertexAttribBinding(ctx,
                                                                         vao,
                                                                         i,
                                                                         __FUNCTION__,
                                                                         &resolved)
                ? resolved.buffer
                : NULL;
            GLuint attribBufferName = attribBuffer ? attribBuffer->name : 0;
            NSLog(@"MGL VBIND attrib=%u enabled=%d buf=%p bufName=%u bindOffset=%lld ptr=0x%llx stride=%u size=%u type=0x%x normalized=%u divisor=%u binding=%u table=%d",
                  i,
                  enabled ? 1 : 0,
                  attribBuffer,
                  attribBufferName,
                  (long long)(attribBuffer ? resolved.binding_offset : vao->attrib[i].binding_offset),
                  (unsigned long long)(uintptr_t)vao->attrib[i].relativeoffset,
                  (unsigned)(attribBuffer ? resolved.stride : vao->attrib[i].stride),
                  (unsigned)vao->attrib[i].size,
                  (unsigned)vao->attrib[i].type,
                  (unsigned)vao->attrib[i].normalized,
                  (unsigned)(attribBuffer ? resolved.divisor : vao->attrib[i].divisor),
                  (unsigned)vao->attrib[i].buffer_bindingindex,
                  attribBuffer && resolved.uses_binding_table ? 1 : 0);

            if (enabled && attribBuffer) {
                NSLog(@"MGL VBIND buffer detail attrib=%u name=%u size=%lld mtl=%p data=%p init(ever=%u full=%u range=[%lld,%lld) source=%u off=%lld size=%lld src=%p hash=0x%016llx)",
                      i,
                      attribBuffer->name,
                      (long long)attribBuffer->size,
                      attribBuffer->data.mtl_data,
                      (void *)attribBuffer->data.buffer_data,
                      (unsigned)attribBuffer->ever_written,
                      (unsigned)attribBuffer->has_initialized_data,
                      (long long)attribBuffer->written_min,
                      (long long)attribBuffer->written_max,
                      (unsigned)attribBuffer->last_init_source,
                      (long long)attribBuffer->last_write_offset,
                      (long long)attribBuffer->last_write_size,
                      attribBuffer->last_write_src_ptr,
                      (unsigned long long)attribBuffer->last_write_src_hash);
            }
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
            const void *inlineBytes =
                (const void *)((const uint8_t *)bin.cpu_ptr + plan.inline_src_offset);
            if (plan.inline_length > plan.inline_visible) {
                memcpy(padded, inlineBytes, plan.inline_visible);
                memset(padded + plan.inline_visible, 0,
                       plan.inline_length - plan.inline_visible);
                inlineBytes = padded;
            }
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
    uint32_t boundVertexBufferMask = 0;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        if (anyBindingPresent[i]) {
            boundVertexBufferMask |= 1U << i;
        }
    }
    mglRenderBindingOrVertexBufferMask(_bindingStateOwner,
                                          boundVertexBufferMask);

    if (mglEnvFlagEnabled("MGL_TRACE_SPARSE_BINDING")) {
        static uint64_t s_vbind_trace_count = 0;
        if ((++s_vbind_trace_count % 500) == 1) {
            NSLog(@"MGL SPARSE VBIND: mask=0x%x activeSlots=%d/31",
                  boundVertexBufferMask,
                  __builtin_popcount(boundVertexBufferMask));
        }
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(vbindCall)) {
        NSUInteger boundSlots = 0;
        NSUInteger reservedSlots = 0;
        NSUInteger baseSlots = 0;
        for (NSUInteger s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            if (anyBindingPresent[s]) {
                boundSlots++;
            }
            if (attribBindingReserved[s]) {
                reservedSlots++;
            }
        }
        for (NSUInteger s = 0; s < MAX_BINDABLE_BUFFERS; s++) {
            if (baseBindingPresent[s]) {
                baseSlots++;
            }
        }
        mglTraceLogNSString(@"MGL TRACE vbind.end call=%llu mapCount=%u boundSlots=%lu reservedAttribSlots=%lu baseSlots=%lu elapsed=%.1fus",
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
#define MGL_VATTR_FLUSH_SNAPSHOT()                                              \
    do {                                                                        \
        if (vattrUseSnapshot && vattrSnapshot->vertex_op_count > 0) {           \
            mglRenderEncodeBindingSnapshotForRenderEncoderOwner(             \
                encCtx->render_encoder_owner, vattrSnapshot, NULL, 0);          \
            *vattrSnapshot = (MGLRenderBindingSnapshot){0};                  \
        }                                                                       \
    } while (0)

#define MGL_VATTR_EMIT_BUFFER(slot, bufPtr, off)                                \
    do {                                                                        \
        if (vattrUseSnapshot) {                                                 \
            if (vattrSnapshot->vertex_op_count >=                               \
                MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                      \
                MGL_VATTR_FLUSH_SNAPSHOT();                                     \
            }                                                                   \
            vattrSnapshot->vertex_ops[vattrSnapshot->vertex_op_count++] =       \
                (MGLRenderBindingOp){/* kind */ 0u,                          \
                                        /* index */ (uint32_t)(slot),           \
                                        /* offset */ (uint64_t)(off),           \
                                        /* buffer */ (void *)(bufPtr),          \
                                        /* bytes */ NULL,                       \
                                        /* length */ 0u};                       \
        } else {                                                                \
            mglBindingStateSetVertexBuffer(                                     \
                encCtx->render_encoder_owner,                  \
                (__bridge id)(bufPtr),                           \
                (off), (slot));                                                 \
        }                                                                       \
    } while (0)

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

    MGLRenderBindingSnapshot *vfallbackSnapshot = bindingSnapshot;
    const BOOL vfallbackUseSnapshot =
        useSnapshot && vfallbackSnapshot != NULL;
#define MGL_VFB_FLUSH_SNAPSHOT()                                                \
    do {                                                                        \
        if (vfallbackUseSnapshot &&                                             \
            vfallbackSnapshot->vertex_op_count > 0) {                           \
            mglRenderEncodeBindingSnapshotForRenderEncoderOwner(             \
                encCtx->render_encoder_owner, vfallbackSnapshot, NULL, 0);      \
            *vfallbackSnapshot = (MGLRenderBindingSnapshot){0};              \
        }                                                                       \
    } while (0)

#define MGL_VFB_EMIT_BUFFER(slot, bufPtr, off)                                  \
    do {                                                                        \
        if (vfallbackUseSnapshot) {                                             \
            if (vfallbackSnapshot->vertex_op_count >=                           \
                MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                      \
                MGL_VFB_FLUSH_SNAPSHOT();                                       \
            }                                                                   \
            vfallbackSnapshot                                                     \
                ->vertex_ops[vfallbackSnapshot->vertex_op_count++] =            \
                (MGLRenderBindingOp){/* kind */ 0u,                          \
                                        /* index */ (uint32_t)(slot),           \
                                        /* offset */ (uint64_t)(off),           \
                                        /* buffer */ (void *)(bufPtr),          \
                                        /* bytes */ NULL,                       \
                                        /* length */ 0u};                       \
        } else {                                                                \
            mglBindingStateSetVertexBuffer(                                     \
                encCtx->render_encoder_owner,                  \
                (__bridge id)(bufPtr),                           \
                (off), (slot));                                                 \
        }                                                                       \
    } while (0)
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;

    void *fallbackBindingBuffer = mglRendererBackendGetFallbackBindingBuffer(
        _backend, kMGLDefaultStageFallbackBufferSize);

    // Bind fallback buffer for required stage buffer bindings that were not mapped.
    // This prevents Metal validation aborts on missing buffer slots.
    uint32_t resourceTypes[MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT];
    uint32_t resourceTypeCount =
        mglBindingStageFallbackResourceTypes(resourceTypes,
                                             MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT);
    for (uint32_t t = 0; t < resourceTypeCount; t++) {
        int resourceType = (int)resourceTypes[t];
        int count = mglRendererGetProgramBindingCount(ctx, vertexStage, resourceType);
        Program *program = activeProgram;
        for (int i = 0; i < count; i++) {
            if (!program || resourceType < 0 || resourceType >= MGL_MAX_SHADER_RESOURCES ||
                i >= (int)program->shader_resources_list[vertexStage][resourceType].count) {
                continue;
            }
            MGLShaderResource *resource = &program->shader_resources_list[vertexStage][resourceType].list[i];
            if (mglShouldSkipStageBufferResource(program, vertexStage, resourceType, resource)) {
                continue;
            }
            GLuint elementCount = mglStageBufferResourceElementCount(resourceType, resource);
            for (GLuint element = 0; element < elementCount; element++) {
                GLuint clientBinding =
                    mglClientBufferBindingForResourceElement(resourceType, resource, element);
                if (clientBinding >= MAX_BINDABLE_BUFFERS) {
                    continue;
                }
                NSInteger metalBinding =
                    (NSInteger)mglMetalResourceSlotForElement(resource, element);
                if (metalBinding < 0 || metalBinding >= (NSInteger)kMGLMaxMetalVertexBufferCount) {
                    continue;
                }
                if (mglBindingStageFallbackNeedsBind(
                        anyBindingPresent[(NSUInteger)metalBinding] ? 1 : 0,
                        fallbackBindingBuffer ? 1 : 0)) {
                    NSUInteger _slot = (NSUInteger)metalBinding;
                    if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_BINDING_STAGE_VERTEX,
                    fallbackBindingBuffer, 0, (uint32_t)_slot)) {
                        MGL_VFB_EMIT_BUFFER(_slot,
                                            fallbackBindingBuffer,
                                            0);
                        mglRenderBindingUpdateVertexBuffer(
                    _bindingStateOwner, fallbackBindingBuffer, 0,
                    (uint32_t)_slot);
                        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                    } else {
                        MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                    }
                    baseBindingPresent[clientBinding] = true;
                    anyBindingPresent[_slot] = true;
                }
            }
        }
    }

    // Conservative safety net:
    // Ensure every stage buffer slot has a valid binding before draw validation.
    // This avoids hard aborts when reflection misses hidden/generated buffer args.
    if (kMGLEnableVertexAllSlotFallback && fallbackBindingBuffer) {
        for (NSUInteger s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            if (!anyBindingPresent[s]) {
                if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_BINDING_STAGE_VERTEX,
                fallbackBindingBuffer, 0, (uint32_t)s)) {
                    MGL_VFB_EMIT_BUFFER(s, fallbackBindingBuffer,
                                         0);
                    mglRenderBindingUpdateVertexBuffer(
                    _bindingStateOwner, fallbackBindingBuffer, 0,
                    (uint32_t)s);
                    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
                } else {
                    MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
                }
                anyBindingPresent[s] = true;
            }
        }
    }


    MGL_VFB_FLUSH_SNAPSHOT();
#undef MGL_VFB_EMIT_BUFFER
#undef MGL_VFB_FLUSH_SNAPSHOT
}


/* Bind point-size parameters if the active shader references them.
 * Extracted from bindVertexBuffersToCurrentRenderEncoder. */
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
#define MGL_VPS_FLUSH_SNAPSHOT()                                                \
    do {                                                                        \
        if (vpointUseSnapshot &&                                                \
            vpointSnapshot->vertex_op_count > 0) {                              \
            mglRenderEncodeBindingSnapshotForRenderEncoderOwner(             \
                encCtx->render_encoder_owner, vpointSnapshot, NULL, 0);         \
            *vpointSnapshot = (MGLRenderBindingSnapshot){0};                 \
            *vpointByteScratchUsed = 0;                                         \
        }                                                                       \
    } while (0)

#define MGL_VPS_EMIT_BYTES(slot, src, len)                                      \
    do {                                                                        \
        const void *src_ = (src);                                               \
        size_t len_ = (len);                                                    \
        if (vpointUseSnapshot) {                                                \
            if (*vpointByteScratchUsed + len_ > byteScratchCapacity) {          \
                MGL_VPS_FLUSH_SNAPSHOT();                                       \
            }                                                                   \
            if (vpointSnapshot->vertex_op_count >=                              \
                MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                      \
                MGL_VPS_FLUSH_SNAPSHOT();                                       \
            }                                                                   \
            uint8_t *dst_ = vpointByteScratch + *vpointByteScratchUsed;         \
            memcpy(dst_, src_, len_);                                           \
            *vpointByteScratchUsed += len_;                                     \
            vpointSnapshot->vertex_ops[vpointSnapshot->vertex_op_count++] =     \
                (MGLRenderBindingOp){/* kind */ 1u,                          \
                                        /* index */ (uint32_t)(slot),           \
                                        /* offset */ 0,                         \
                                        /* buffer */ NULL,                      \
                                        /* bytes */ dst_,                       \
                                        /* length */ (uint32_t)len_};           \
        } else {                                                                \
            mglBindingStateSetVertexBytes(                                      \
                encCtx->render_encoder_owner,                  \
                (src), (len), (slot));                                          \
        }                                                                       \
    } while (0)
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

#define MGL_FBIND_FLUSH_SNAPSHOT()                                              \
    do {                                                                        \
        if (snapshot.fragment_op_count > 0) {                                   \
            mglRenderEncodeBindingSnapshotForRenderEncoderOwner(             \
                encCtx->render_encoder_owner, &snapshot, NULL, 0);              \
            snapshot = (MGLRenderBindingSnapshot){0};                        \
            fbindByteScratchUsed = 0;                                           \
        }                                                                       \
    } while (0)

#define MGL_FBIND_COLLECT_BUFFER(slot, bufPtr, off)                             \
    do {                                                                        \
        if (snapshot.fragment_op_count >=                                       \
            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                          \
            MGL_FBIND_FLUSH_SNAPSHOT();                                         \
        }                                                                       \
        snapshot.fragment_ops[snapshot.fragment_op_count++] =                   \
            (MGLRenderBindingOp){/* kind */ 0u,                              \
                                    /* index */ (uint32_t)(slot),               \
                                    /* offset */ (uint64_t)(off),               \
                                    /* buffer */ (void *)(bufPtr),              \
                                    /* bytes */ NULL,                           \
                                    /* length */ 0u};                           \
    } while (0)

#define MGL_FBIND_COLLECT_BYTES(slot, src, len)                                 \
    do {                                                                        \
        const void *src_ = (src);                                               \
        size_t len_ = (len);                                                    \
        if (fbindByteScratchUsed + len_ > sizeof(fbindByteScratch)) {           \
            MGL_FBIND_FLUSH_SNAPSHOT();                                         \
        }                                                                       \
        if (snapshot.fragment_op_count >=                                       \
            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                          \
            MGL_FBIND_FLUSH_SNAPSHOT();                                         \
        }                                                                       \
        uint8_t *dst_ = fbindByteScratch + fbindByteScratchUsed;                \
        memcpy(dst_, src_, len_);                                               \
        fbindByteScratchUsed += len_;                                           \
        snapshot.fragment_ops[snapshot.fragment_op_count++] =                   \
            (MGLRenderBindingOp){/* kind */ 1u,                              \
                                    /* index */ (uint32_t)(slot),               \
                                    /* offset */ 0,                             \
                                    /* buffer */ NULL,                          \
                                    /* bytes */ dst_,                           \
                                    /* length */ (uint32_t)len_};               \
    } while (0)

#define MGL_FBIND_EMIT_BUFFER(slot, bufPtr, off)                                \
    do {                                                                        \
        if (useBindingSnapshot) {                                               \
            MGL_FBIND_COLLECT_BUFFER(slot, bufPtr, off);                        \
        } else {                                                                \
            mglBindingStateSetFragmentBuffer(                                   \
                encCtx->render_encoder_owner,                  \
                (__bridge id)(bufPtr),                           \
                (off), (slot));                                                 \
        }                                                                       \
    } while (0)

#define MGL_FBIND_EMIT_BYTES(slot, src, len)                                    \
    do {                                                                        \
        if (useBindingSnapshot) {                                               \
            MGL_FBIND_COLLECT_BYTES(slot, src, len);                            \
        } else {                                                                \
            mglBindingStateSetFragmentBytes(                                    \
                encCtx->render_encoder_owner,                  \
                (src), (len), (slot));                                          \
        }                                                                       \
    } while (0)

#define MGL_FBIND_EMIT_CLEAR(slot)                                              \
    do {                                                                        \
        if (useBindingSnapshot) {                                               \
            MGL_FBIND_COLLECT_BUFFER(slot, NULL, 0);                            \
        } else {                                                                \
            mglBindingStateSetFragmentBuffer(                                   \
                encCtx->render_encoder_owner,                  \
                nil, 0, (slot));                                                \
        }                                                                       \
    } while (0)

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
            const void *inlineBytes =
                (const void *)((const uint8_t *)bin.cpu_ptr + plan.inline_src_offset);
            if (plan.reason == MGL_SB_REASON_INLINE_UC &&
                plan.inline_length > plan.inline_visible) {
                memcpy(padded, inlineBytes, plan.inline_visible);
                memset(padded + plan.inline_visible, 0,
                       plan.inline_length - plan.inline_visible);
                inlineBytes = padded;
            }
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
    uint32_t boundFragmentBufferMask = 0;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        if (anyBindingPresent[i]) {
            boundFragmentBufferMask |= 1U << i;
        }
    }
    mglRenderBindingOrFragmentBufferMask(_bindingStateOwner, boundFragmentBufferMask);

    if (mglEnvFlagEnabled("MGL_TRACE_SPARSE_BINDING")) {
        static uint64_t s_fbind_trace_count = 0;
        if ((++s_fbind_trace_count % 500) == 1) {
            int textureSlotCount =
                mglBindingStateTextureSlotCount(_bindingStateOwner);
            NSLog(@"MGL SPARSE FBIND: fbuf=0x%x(%d/31) textureSlots=%d/128",
                  boundFragmentBufferMask,
                  __builtin_popcount(boundFragmentBufferMask),
                  textureSlotCount);
        }
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(fbindCall)) {
        NSUInteger boundSlots = 0;
        NSUInteger baseSlots = 0;
        for (NSUInteger s = 0; s < MAX_BINDABLE_BUFFERS; s++) {
            if (anyBindingPresent[s]) {
                boundSlots++;
            }
            if (baseBindingPresent[s]) {
                baseSlots++;
            }
        }
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
    /* Keep fallback emits in the same per-draw snapshot as the main fragment
     * binding loop.  This is the final fragment binding segment, so replaying
     * at method exit preserves the direct path's ordering while removing the
     * last fragment-stage ObjC setter body from the gate-on path. */
    MGLRenderBindingSnapshot *ffallbackSnapshot = bindingSnapshot;
    const BOOL ffallbackUseSnapshot = useSnapshot && ffallbackSnapshot != NULL;
#define MGL_FFB_FLUSH_SNAPSHOT()                                               \
    do {                                                                        \
        if (ffallbackUseSnapshot &&                                             \
            ffallbackSnapshot->fragment_op_count > 0) {                        \
            mglRenderEncodeBindingSnapshotForRenderEncoderOwner(            \
                encCtx->render_encoder_owner, ffallbackSnapshot, NULL, 0);     \
            *ffallbackSnapshot = (MGLRenderBindingSnapshot){0};              \
        }                                                                       \
    } while (0)
#define MGL_FFB_EMIT_BUFFER(slot, bufPtr, off)                                  \
    do {                                                                        \
        if (ffallbackUseSnapshot) {                                             \
            if (ffallbackSnapshot->fragment_op_count >=                        \
                MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {                     \
                MGL_FFB_FLUSH_SNAPSHOT();                                      \
            }                                                                   \
            ffallbackSnapshot->fragment_ops[                                   \
                ffallbackSnapshot->fragment_op_count++] =                      \
                (MGLRenderBindingOp){/* kind */ 0u,                         \
                                        /* index */ (uint32_t)(slot),            \
                                        /* offset */ (uint64_t)(off),            \
                                        /* buffer */ (void *)(bufPtr),           \
                                        /* bytes */ NULL,                        \
                                        /* length */ 0u};                        \
        } else {                                                                \
            mglBindingStateSetFragmentBuffer(                                   \
                encCtx->render_encoder_owner,                  \
                (__bridge id)(bufPtr),                           \
                (off), (slot));                                                 \
        }                                                                       \
    } while (0)

    void *fallbackBindingBuffer = mglRendererBackendGetFallbackBindingBuffer(
        _backend, kMGLDefaultStageFallbackBufferSize);

    // Bind fallback buffer for required stage buffer bindings that were not mapped.
    uint32_t resourceTypes[MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT];
    uint32_t resourceTypeCount =
        mglBindingStageFallbackResourceTypes(resourceTypes,
                                             MGL_STAGE_FALLBACK_RESOURCE_TYPE_COUNT);
    for (uint32_t t = 0; t < resourceTypeCount; t++) {
        int resourceType = (int)resourceTypes[t];
        int count = mglRendererGetProgramBindingCount(ctx, _FRAGMENT_SHADER, resourceType);
        Program *program = activeProgram;
        for (int i = 0; i < count; i++) {
            if (!program || resourceType < 0 || resourceType >= MGL_MAX_SHADER_RESOURCES ||
                i >= (int)program->shader_resources_list[_FRAGMENT_SHADER][resourceType].count) {
                continue;
            }
            MGLShaderResource *resource = &program->shader_resources_list[_FRAGMENT_SHADER][resourceType].list[i];
            if (mglShouldSkipStageBufferResource(program, _FRAGMENT_SHADER, resourceType, resource)) {
                continue;
            }
            GLuint elementCount = mglStageBufferResourceElementCount(resourceType, resource);
            for (GLuint element = 0; element < elementCount; element++) {
                GLuint clientBinding =
                    mglClientBufferBindingForResourceElement(resourceType, resource, element);
                if (clientBinding >= MAX_BINDABLE_BUFFERS) {
                    continue;
                }
                NSInteger metalBinding =
                    (NSInteger)mglMetalResourceSlotForElement(resource, element);
                if (metalBinding < 0 || metalBinding >= (NSInteger)MAX_BINDABLE_BUFFERS) {
                    continue;
                }
                if (mglBindingStageFallbackNeedsBind(
                        anyBindingPresent[(NSUInteger)metalBinding] ? 1 : 0,
                        fallbackBindingBuffer ? 1 : 0)) {
                    NSUInteger _slot = (NSUInteger)metalBinding;
                    if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    fallbackBindingBuffer, 0, (uint32_t)_slot)) {
                        MGL_FFB_EMIT_BUFFER(_slot,
                                           fallbackBindingBuffer,
                                           0);
                        mglRenderBindingUpdateFragmentBuffer(
                    _bindingStateOwner, fallbackBindingBuffer, 0,
                    (uint32_t)_slot);
                        MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                    } else {
                        MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                    }
                    baseBindingPresent[clientBinding] = true;
                    anyBindingPresent[_slot] = true;
                }
            }
        }
    }

    if (fallbackBindingBuffer) {
        for (NSUInteger s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            if (!anyBindingPresent[s]) {
                if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                fallbackBindingBuffer, 0, (uint32_t)s)) {
                    MGL_FFB_EMIT_BUFFER(s, fallbackBindingBuffer,
                                        0);
                    mglRenderBindingUpdateFragmentBuffer(
                    _bindingStateOwner, fallbackBindingBuffer, 0,
                    (uint32_t)s);
                    MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                } else {
                    MGL_PERF_INC(g_mglSetFragmentBufferSkipsSinceSwap);
                }
                anyBindingPresent[s] = true;
            }
        }
    }

    MGL_FFB_FLUSH_SNAPSHOT();
#undef MGL_FFB_EMIT_BUFFER
#undef MGL_FFB_FLUSH_SNAPSHOT
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
        /* Only warmup sampler slots the program actually samples, using the
         * sampled_texture_unit_mask bitmap to skip unused slots, instead of
         * blindly setting all TEXTURE_UNITS. */
        uint32_t activeMask[4] = {0, 0, 0, 0};
        if (vertexProgram) {
            (void)mglProgramSamplesTextureUnit(vertexProgram, 0); /* trigger lazy build */
            for (int i = 0; i < 4; i++)
                activeMask[i] |= vertexProgram->sampled_texture_unit_mask[i];
        }
        if (fragmentProgram && fragmentProgram != vertexProgram) {
            (void)mglProgramSamplesTextureUnit(fragmentProgram, 0);
            for (int i = 0; i < 4; i++)
                activeMask[i] |= fragmentProgram->sampled_texture_unit_mask[i];
        }

        NSUInteger warmupCount = TEXTURE_UNITS;
        if (warmupCount > kMaxFragmentSamplerSlots) {
            warmupCount = kMaxFragmentSamplerSlots;
        }
        /* If no program is bound (both NULL), fall back to warming all slots
         * to avoid Metal assertions on stale sampler state. */
        bool hasActiveProgram = (vertexProgram != nil || fragmentProgram != nil);
        bool maskEmpty = mglBindingTextureSamplerMaskEmpty(activeMask) != 0;
        if (hasActiveProgram && !maskEmpty) {
            for (NSUInteger s = 0; s < warmupCount; s++) {
                if (!mglBindingTextureSamplerWarmupSlotActive(activeMask, (uint32_t)s))
                    continue;
                if (!mglBindingStateQueueResourceBinding(
                        useResourceSnapshot, _bindingStateOwner,
                        _renderPassManager.state->currentRenderEncoderOwner,
                        &resourceSnapshot, MGL_RENDER_BINDING_STAGE_VERTEX,
                        MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                        (__bridge void *)defaultSampler, (uint32_t)s) ||
                    !mglBindingStateQueueResourceBinding(
                        useResourceSnapshot, _bindingStateOwner,
                        _renderPassManager.state->currentRenderEncoderOwner,
                        &resourceSnapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                        MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                        (__bridge void *)defaultSampler, (uint32_t)s)) {
                    return false;
                }
            }
        } else {
            for (NSUInteger s = 0; s < warmupCount; s++) {
                if (!mglBindingStateQueueResourceBinding(
                        useResourceSnapshot, _bindingStateOwner,
                        _renderPassManager.state->currentRenderEncoderOwner,
                        &resourceSnapshot, MGL_RENDER_BINDING_STAGE_VERTEX,
                        MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                        (__bridge void *)defaultSampler, (uint32_t)s) ||
                    !mglBindingStateQueueResourceBinding(
                        useResourceSnapshot, _bindingStateOwner,
                        _renderPassManager.state->currentRenderEncoderOwner,
                        &resourceSnapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                        MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                        (__bridge void *)defaultSampler, (uint32_t)s)) {
                    return false;
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
            sin.phase = MGL_ST_PHASE_COMPAT;
            sin.has_mtl_texture = texture ? 1 : 0;
            sin.mtl_type = texture ? mglBindingStateTextureType(texture) : 0u;
            sin.expected_type = expectedType;
            sin.format_kind_ok =
                !texture ||
                        mglTexturePixelFormatCompatibleWithExpectedDataKind(
                            mglBindingStateTexturePixelFormat(texture),
                            expectedKind)
                    ? 1
                    : 0;
            if (mglBindingTexturePlanSampled(&sin, &splan) == 0 &&
                (splan.action == MGL_ST_ACTION_TYPE_FALLBACK ||
                 splan.action == MGL_ST_ACTION_KIND_FALLBACK)) {
                static uint64_t s_vertexCompatMismatchLogCount = 0;
                uint64_t hit = ++s_vertexCompatMismatchLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX %@ MISMATCH vertex binding=%u program=%u glTex=%u glTarget=0x%x mtlType=%lu expected=%lu hit=%llu",
                          splan.action == MGL_ST_ACTION_TYPE_FALLBACK ? @"TYPE"
                                                                      : @"DATA",
                          (unsigned)spirvBinding, (unsigned)vertexProgramName,
                          (unsigned)ptr->name, (unsigned)ptr->target,
                          (unsigned long)sin.mtl_type,
                          (unsigned long)expectedType, (unsigned long long)hit);
                }
                mglWriteProgramMSLDump(
                    currentProgram,
                    [NSString stringWithFormat:
                                  @"tex-%@-mismatch-vertex-binding-%u",
                              splan.action == MGL_ST_ACTION_TYPE_FALLBACK
                                  ? @"type"
                                  : @"data",
                              spirvBinding]);
                texture = [self fallbackSampledTextureForExpectedType:expectedType
                                                             dataKind:expectedKind];
                usedTypeFallback = YES;
            }

            if (textureUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[textureUnit]) {
                Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (glSampler->mtl_data == NULL) {
                    glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge id)(glSampler->mtl_data);
                mglTraceLogExternal("VERT_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=glSampler samplerName=%u minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
                                    (unsigned)vertexProgramName,
                                    (unsigned)spirvBinding,
                                    (unsigned)textureUnit,
                                    (unsigned)glSampler->name,
                                    (unsigned)glSampler->params.min_filter,
                                    (unsigned)glSampler->params.mag_filter,
                                    (unsigned)glSampler->params.wrap_s,
                                    (unsigned)glSampler->params.wrap_t,
                                    (double)glSampler->params.min_lod,
                                    (double)glSampler->params.max_lod,
                                    (unsigned)ptr->name,
                                    (unsigned)ptr->params.base_level,
                                    (unsigned)ptr->params.max_level,
                                    (unsigned)ptr->width,
                                    (unsigned)ptr->height,
                                    (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0u),
                                    (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0u),
                                    (unsigned long)(texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u));
            } else if (ptr->params.mtl_data) {
                sampler = (__bridge id)(ptr->params.mtl_data);
                mglTraceLogExternal("VERT_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=texParamsFallback samplerName=0 minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
                                    (unsigned)vertexProgramName,
                                    (unsigned)spirvBinding,
                                    (unsigned)textureUnit,
                                    (unsigned)ptr->params.min_filter,
                                    (unsigned)ptr->params.mag_filter,
                                    (unsigned)ptr->params.wrap_s,
                                    (unsigned)ptr->params.wrap_t,
                                    (double)ptr->params.min_lod,
                                    (double)ptr->params.max_lod,
                                    (unsigned)ptr->name,
                                    (unsigned)ptr->params.base_level,
                                    (unsigned)ptr->params.max_level,
                                    (unsigned)ptr->width,
                                    (unsigned)ptr->height,
                                    (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0u),
                                    (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0u),
                                    (unsigned long)(texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u));
            }
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
        Program *focusedTextureProgram = currentProgram;
        if (mglProgramNeedsBindingTrace(focusedTextureProgram)) {
            static uint64_t s_focusedVertexTextureBindLogs = 0;
            if (mglShouldLogFocusedBinding(&s_focusedVertexTextureBindLogs)) {
                TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
                NSLog(@"MGL TBIND focused stage=vertex program=%u resource=%s metalTextureSlot=%u samplerUnit=%u glTex=%u target=0x%x mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
                      (unsigned)focusedTextureProgram->name,
                      sampledName ? sampledName : "",
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      ptr ? (unsigned)ptr->name : 0u,
                      ptr ? (unsigned)ptr->target : 0u,
                      texture,
                      (unsigned long)(texture ? mglBindingStateTextureType(texture) : 0),
                      (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
                      (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
                      level0 ? (unsigned)level0->width : 0u,
                      level0 ? (unsigned)level0->height : 0u,
                      level0 ? (unsigned)level0->ever_written : 0u,
                      level0 ? (unsigned)level0->has_initialized_data : 0u,
                      level0 ? (unsigned)level0->last_init_source : 0u);
            }
        }
        static uint64_t s_traceFileVertexTextureBindLogs = 0;
        if (mglProgramNeedsTraceLog(focusedTextureProgram) &&
            mglShouldLogTraceFileBindingForProgram(focusedTextureProgram, &s_traceFileVertexTextureBindLogs)) {
            TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
            int expectedIndex = [self textureIndexForExpectedMetalType:(lookupType ? lookupType : expectedType)];
            Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
            Texture *unitExpected = (textureUnit < TEXTURE_UNITS &&
                                     expectedIndex >= 0 &&
                                     expectedIndex < _MAX_TEXTURE_TYPES)
                ? MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex]
                : NULL;
            Texture *unit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
            Texture *unitCube = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP] : NULL;
            mglTraceLog("TBIND stage=vertex program=%u resource=%s metalTextureSlot=%u samplerUnit=%u resUnit=%d explicit=%d glTex=%u target=0x%x fallback=%d expectedType=%lu lookupType=%lu expectedIndex=%d unit(active=%u expected=%u tex2D=%u cube=%u) mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
                        (unsigned)focusedTextureProgram->name,
                        sampledName ? sampledName : "",
                        (unsigned)spirvBinding,
                        (unsigned)textureUnit,
                        sampledResource ? (int)sampledResource->sampler_unit : -1,
                        (sampledResource && sampledResource->sampler_unit_explicit) ? 1 : 0,
                        ptr ? (unsigned)ptr->name : 0u,
                        ptr ? (unsigned)ptr->target : 0u,
                        usedTypeFallback ? 1 : 0,
                        (unsigned long)expectedType,
                        (unsigned long)lookupType,
                        expectedIndex,
                        mglTraceTextureName(unitActive),
                        mglTraceTextureName(unitExpected),
                        mglTraceTextureName(unit2D),
                        mglTraceTextureName(unitCube),
                        texture,
                        (unsigned long)(texture ? mglBindingStateTextureType(texture) : 0),
                        (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
                        (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
                        level0 ? (unsigned)level0->width : 0u,
                        level0 ? (unsigned)level0->height : 0u,
                        level0 ? (unsigned)level0->ever_written : 0u,
                        level0 ? (unsigned)level0->has_initialized_data : 0u,
                        level0 ? (unsigned)level0->last_init_source : 0u);
        }
        if (ptr && mglRenderTextureTargetIsBuffer((uint32_t)ptr->target)) {
            static uint64_t s_vertexTexelBufferBindLogs = 0;
            uint64_t hit = ++s_vertexTexelBufferBindLogs;
            if (hit <= 8ull || (hit % 2048ull) == 0ull) {
                Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
                Texture *unitBuffer = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_BUFFER_TARGET] : NULL;
                NSLog(@"MGL TEXBUFFER BIND vertex hit=%llu program=%u binding=%u unit=%u ptrTex=%u active=%u bufferSlot=%u expectedType=%lu lookupType=%lu mtlTex=%p mtlType=%lu size=%lux%lu format=%lu sampler=%p",
                      (unsigned long long)hit,
                      (unsigned)vertexProgramName,
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      (unsigned)ptr->name,
                      mglTraceTextureName(unitActive),
                      mglTraceTextureName(unitBuffer),
                      (unsigned long)expectedType,
                      (unsigned long)lookupType,
                      texture,
                      (unsigned long)(texture ? mglBindingStateTextureType(texture) : 0),
                      (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
                      (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
                      (unsigned long)(texture ? mglBindingStateTexturePixelFormat(texture) : 0),
                      sampler);
            }
        }
        if (ptr && !mglRenderTextureTargetIsBuffer((uint32_t)ptr->target)) {
            Program *sampleProgram = currentProgram;
            GLuint sampleProgramName = sampleProgram ? sampleProgram->name : vertexProgramName;
            TextureLevel *sampleLevel0 = mglTraceTextureBaseLevel(ptr);
            BOOL focusedVertexSample =
                (sampleProgramName == 34u) ||
                (sampleLevel0 &&
                 (sampleLevel0->suspicious_zero_upload ||
                  !sampleLevel0->ever_written ||
                  !sampleLevel0->has_initialized_data));
            if (focusedVertexSample) {
                static uint64_t s_vertexSampleDetailLogCount = 0;
                uint64_t hit = ++s_vertexSampleDetailLogCount;
                if (hit <= 128ull || (hit % 512ull) == 0ull) {
                    int expectedIndex = [self textureIndexForExpectedMetalType:expectedType];
                    Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
                    Texture *unitExpected = (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES)
                        ? MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex]
                        : NULL;
                    uint64_t levelDataHash = (sampleLevel0 && sampleLevel0->data && sampleLevel0->data_size > 0)
                        ? mglTraceHashBytes((const void *)(uintptr_t)sampleLevel0->data, sampleLevel0->data_size)
                        : 0ull;

                    mglTraceLogNSString(@"MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=vertex program=%u name=%s binding=%u "
                          "unit=%u expectedType=%lu expectedIndex=%d ptrTex=%u ptr=%p target=0x%x fallback=%d mtlTex=%p mtlType=%lu mtlSize=%lux%lu "
                          "unit(active=%u expected=%u) "
                          "l0=%ux%ux%u bytes=%lu init(ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx dataHash=0x%016llx)",
                          (unsigned long long)bindCall,
                          (unsigned long long)hit,
                          sampleProgramName,
                          sampledName ? sampledName : "",
                          (unsigned)spirvBinding,
                          (unsigned)textureUnit,
                          (unsigned long)expectedType,
                          expectedIndex,
                          mglTraceTextureName(ptr),
                          ptr,
                          ptr ? (unsigned)ptr->target : 0u,
                          usedTypeFallback ? 1 : 0,
                          texture,
                          (unsigned long)(texture ? mglBindingStateTextureType(texture) : 0),
                          (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
                          (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
                          mglTraceTextureName(unitActive),
                          mglTraceTextureName(unitExpected),
                          sampleLevel0 ? (unsigned)sampleLevel0->width : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->height : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->depth : 0u,
                          (unsigned long)(sampleLevel0 ? sampleLevel0->data_size : 0u),
                          sampleLevel0 ? (unsigned)sampleLevel0->ever_written : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->has_initialized_data : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->suspicious_zero_upload : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->last_init_source : 0u,
                          (unsigned long)(sampleLevel0 ? sampleLevel0->last_upload_size : 0u),
                          sampleLevel0 ? (void *)sampleLevel0->last_src_ptr : NULL,
                          (unsigned long long)(sampleLevel0 ? sampleLevel0->last_src_hash : 0ull),
                          (unsigned long long)levelDataHash);
                }
            }
        }
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
            if (![self resolveFragmentSampledYFlipAndSampler:ptr
                                                      texture:&texture
                                                      sampler:&sampler
                                                  sampledName:sampledName
                                                 spirvBinding:spirvBinding
                                                   textureUnit:textureUnit
                                                  expectedType:expectedType
                                                  expectedKind:expectedKind
                                          fragmentProgramName:fragmentProgramName
                                           vertexProgramName:vertexProgramName
                                                 sampleProgram:sampleProgram
                                             usedFallbackTexture:&usedFallbackTexture
                                        usedSampledCopyForTrace:&usedSampledCopyForTrace
                                           directTextureForTrace:&directTextureForTrace
                                           sampledCopyForTrace:&sampledCopyForTrace]) {
                return false;
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
                uint64_t hit = ++s_fragmentFallbackLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX FALLBACK fragment sampled binding=%u program=%u glTex=%u hit=%llu",
                          (unsigned)spirvBinding,
                          (unsigned)fragmentProgramName,
                          ptr ? (unsigned)ptr->name : 0u,
                          (unsigned long long)hit);
                }
            }
        } else if (!texture && suppressMissingTextureFallback) {
            static uint64_t s_fragmentFallbackSuppressedLogCount = 0;
            uint64_t hit = ++s_fragmentFallbackSuppressedLogCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL TEX FALLBACK SUPPRESSED fragment sampled binding=%u program=%u name=%s glTex=%u unit=%u reason=insampler-current-target-no-copy hit=%llu",
                      (unsigned)spirvBinding,
                      (unsigned)fragmentProgramName,
                      sampledName ? sampledName : "",
                      ptr ? (unsigned)ptr->name : 0u,
                      (unsigned)textureUnit,
                      (unsigned long long)hit);
            }
        }

	        if (!sampler) {
	            sampler = defaultSampler;
	        }
        if (usedFallbackTexture && expectedKind == MGLTextureDataKindDepth) {
            sampler = defaultSampler;
        }

                GLuint sampleProgramName = sampleProgram ? sampleProgram->name : fragmentProgramName;
                TextureLevel *sampleLevel0 = mglTraceTextureBaseLevel(ptr);
	        BOOL focusedSample =
		            mglIsFocusedLoadingProgram(sampleProgramName) &&
		            (bindCall <= 2048ull || ((bindCall % 512ull) == 0ull));
                BOOL guiRTSample =
                    ptr &&
                    mglTextureCanUseGLSampledRenderTargetCopy(ptr);
		        BOOL suspiciousSample =
		            usedFallbackTexture ||
		            (ptr && ptr->name == 13u) ||
                    guiRTSample ||
		    focusedSample ||
		            (sampleLevel0 &&
		             (sampleLevel0->suspicious_zero_upload ||
		              !sampleLevel0->ever_written ||
		              !sampleLevel0->has_initialized_data));
        if (suspiciousSample) {
            static uint64_t s_fragmentSampleDetailLogCount = 0;
            uint64_t hit = ++s_fragmentSampleDetailLogCount;
            if (hit <= 256ull || (hit % 512ull) == 0ull) {
	                int expectedIndex = [self textureIndexForExpectedMetalType:expectedType];
	                Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
	                Texture *unitExpected = (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES)
	                    ? MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex]
	                    : NULL;
	                Texture *unit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
	                Texture *unitCube = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP] : NULL;
	                uint32_t actualType = texture ? mglBindingStateTextureType(texture) : 0;
	                uint64_t levelDataHash = (sampleLevel0 && sampleLevel0->data && sampleLevel0->data_size > 0)
	                    ? mglTraceHashBytes((const void *)(uintptr_t)sampleLevel0->data, sampleLevel0->data_size)
	                    : 0ull;

	                mglTraceLogNSString(@"MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=fragment program=%u name=%s binding=%u "
	                      "unit=%u expectedType=%lu expectedIndex=%d ptrTex=%u ptr=%p target=0x%x fallback=%d mtlTex=%p mtlType=%lu mtlSize=%lux%lu "
	                      "unit(active=%u expected=%u tex2D=%u cube=%u) "
	                      "l0=%ux%ux%u bytes=%lu init(ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx dataHash=0x%016llx)",
	                      (unsigned long long)bindCall,
	                      (unsigned long long)hit,
		                      sampleProgramName,
                          sampledName ? sampledName : "",
	                      (unsigned)spirvBinding,
	                      (unsigned)textureUnit,
	                      (unsigned long)expectedType,
	                      expectedIndex,
	                      mglTraceTextureName(ptr),
	                      ptr,
	                      ptr ? (unsigned)ptr->target : 0u,
	                      usedFallbackTexture ? 1 : 0,
	                      texture,
	                      (unsigned long)actualType,
	                      (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
	                      (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
	                      mglTraceTextureName(unitActive),
	                      mglTraceTextureName(unitExpected),
	                      mglTraceTextureName(unit2D),
	                      mglTraceTextureName(unitCube),
	                      sampleLevel0 ? (unsigned)sampleLevel0->width : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->height : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->depth : 0u,
	                      (unsigned long)(sampleLevel0 ? sampleLevel0->data_size : 0u),
	                      sampleLevel0 ? (unsigned)sampleLevel0->ever_written : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->has_initialized_data : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->suspicious_zero_upload : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->last_init_source : 0u,
	                      (unsigned long)(sampleLevel0 ? sampleLevel0->last_upload_size : 0u),
	                      sampleLevel0 ? (void *)sampleLevel0->last_src_ptr : NULL,
	                      (unsigned long long)(sampleLevel0 ? sampleLevel0->last_src_hash : 0ull),
	                      (unsigned long long)levelDataHash);
	            }

                if (guiRTSample) {
                    static uint64_t s_guiRTSampleLogCount = 0;
	                    uint64_t atlasHit = ++s_guiRTSampleLogCount;
                    if (atlasHit <= 128ull || (atlasHit % 256ull) == 0ull) {
                            int atlasExpectedIndex = [self textureIndexForExpectedMetalType:expectedType];
                            Texture *atlasUnitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
                            Texture *atlasUnitExpected = (atlasExpectedIndex >= 0 && atlasExpectedIndex < _MAX_TEXTURE_TYPES)
                                ? MGL_STATE(ctx)->texture_units[textureUnit].textures[atlasExpectedIndex]
                                : NULL;
                            Texture *atlasUnit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
                            Texture *atlasUnitCube = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP] : NULL;
	                        id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                                _renderPassManager.state->renderPassStateOwner,
                                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
	                        id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                                _renderPassManager.state->renderPassStateOwner,
                                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
                        mglTraceLog("RT_SAMPLE_COPY_SAMPLE hit=%llu bindCall=%llu program=%u stateProgram=%u current=%u pipeline=%u vs=%u fs=%u pipelineProgram=%u name=%s binding=%u unit=%u "
                                    "rtTex=%u label=\"%s\" fallback=%d useCopy=%d ptr=%p mtl=%p direct=%p copy=%p fmt=%lu type=%lu size=%lux%lu "
                                    "unit(active=%u expected=%u tex2D=%u cube=%u) "
                                    "l0(ever=%u full=%u zero=%u source=%u upload=%lu) "
                                    "drawFbo=%u rpFbo=%u rpColor=%p rpDepth=%p depthTest=%d blend=%d",
                                    (unsigned long long)atlasHit,
                                    (unsigned long long)bindCall,
                                    sampleProgramName,
                                    (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
                                    (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
                                    (unsigned)(ctx ? MGL_STATE(ctx)->var.program_pipeline_binding : 0u),
                                    (unsigned)vertexProgramName,
                                    (unsigned)fragmentProgramName,
                                    (unsigned)_pipelineCache.state->pipelineProgramName,
                                    sampledName ? sampledName : "",
                                    (unsigned)spirvBinding,
                                    (unsigned)textureUnit,
                                    (unsigned)mglTraceTextureName(ptr),
                                    mglTraceTextureLabel(ptr),
                                    usedFallbackTexture ? 1 : 0,
                                    usedSampledCopyForTrace ? 1 : 0,
                                    ptr,
                                    texture,
                                    directTextureForTrace,
                                    sampledCopyForTrace,
                                    (unsigned long)(texture ? mglBindingStateTexturePixelFormat(texture) : MGL_BINDING_PIXEL_FORMAT_INVALID),
                                    (unsigned long)(texture ? mglBindingStateTextureType(texture) : 0),
                                    (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
                                    (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
                                    mglTraceTextureName(atlasUnitActive),
                                    mglTraceTextureName(atlasUnitExpected),
                                    mglTraceTextureName(atlasUnit2D),
                                    mglTraceTextureName(atlasUnitCube),
                                    sampleLevel0 ? (unsigned)sampleLevel0->ever_written : 0u,
                                    sampleLevel0 ? (unsigned)sampleLevel0->has_initialized_data : 0u,
                                    sampleLevel0 ? (unsigned)sampleLevel0->suspicious_zero_upload : 0u,
                                    sampleLevel0 ? (unsigned)sampleLevel0->last_init_source : 0u,
                                    (unsigned long)(sampleLevel0 ? sampleLevel0->last_upload_size : 0u),
                                    (unsigned)(ctx && MGL_STATE(ctx)->framebuffer ? MGL_STATE(ctx)->framebuffer->name : 0u),
                                    (unsigned)_renderPassManager.state->renderPassFramebufferName,
                                    rpColor0,
                                    rpDepth,
                                    ctx && MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
                                    ctx && MGL_STATE(ctx)->caps.blend ? 1 : 0);
                    }
                }

		            if (texture && sampleLevel0 &&
		                (sampleLevel0->suspicious_zero_upload ||
		                 !sampleLevel0->ever_written ||
		                 !sampleLevel0->has_initialized_data)) {
		                static uint64_t s_fragmentSampleReadbackCount = 0;
		                uint64_t rbHit = ++s_fragmentSampleReadbackCount;
		                if (rbHit <= 32ull || (rbHit % 512ull) == 0ull) {
	                    [self traceSampledTextureReadback:texture
	                                                glTex:ptr
	                                                level:sampleLevel0
	                                              program:sampleProgramName
	                                              binding:spirvBinding
	                                                stage:@"fragment"
		                                               reason:(sampleLevel0->suspicious_zero_upload ? @"zero-level" :
		                                                       (!sampleLevel0->ever_written ? @"never-written" : @"not-initialized"))
		                                                  hit:rbHit];
		                }
		            }
	        }

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
        Program *focusedTextureProgram = sampleProgram;
        if (mglProgramNeedsBindingTrace(focusedTextureProgram)) {
            static uint64_t s_focusedFragmentTextureBindLogs = 0;
            if (mglShouldLogFocusedBinding(&s_focusedFragmentTextureBindLogs)) {
                TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
                NSLog(@"MGL TBIND focused stage=fragment program=%u resource=%s metalTextureSlot=%u samplerUnit=%u glTex=%u target=0x%x mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
                      (unsigned)focusedTextureProgram->name,
                      sampledName ? sampledName : "",
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      ptr ? (unsigned)ptr->name : 0u,
                      ptr ? (unsigned)ptr->target : 0u,
                      texture,
                      (unsigned long)(texture ? mglBindingStateTextureType(texture) : 0),
                      (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
                      (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
                      level0 ? (unsigned)level0->width : 0u,
                      level0 ? (unsigned)level0->height : 0u,
                      level0 ? (unsigned)level0->ever_written : 0u,
                      level0 ? (unsigned)level0->has_initialized_data : 0u,
                      level0 ? (unsigned)level0->last_init_source : 0u);
            }
        }
        static uint64_t s_traceFileFragmentTextureBindLogs = 0;
        if (mglProgramNeedsTraceLog(focusedTextureProgram) &&
            mglShouldLogTraceFileBindingForProgram(focusedTextureProgram, &s_traceFileFragmentTextureBindLogs)) {
            TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
            int expectedIndex = [self textureIndexForExpectedMetalType:(lookupType ? lookupType : expectedType)];
            Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
            Texture *unitExpected = (textureUnit < TEXTURE_UNITS &&
                                     expectedIndex >= 0 &&
                                     expectedIndex < _MAX_TEXTURE_TYPES)
                ? MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex]
                : NULL;
            Texture *unit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
            Texture *unitCube = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP] : NULL;
            mglTraceLog("TBIND stage=fragment program=%u resource=%s metalTextureSlot=%u samplerUnit=%u resUnit=%d explicit=%d glTex=%u target=0x%x fallback=%d expectedType=%lu lookupType=%lu expectedIndex=%d unit(active=%u expected=%u tex2D=%u cube=%u) mtl=%p mtlType=%lu size=%lux%lu level0=%ux%u init(ever=%u full=%u source=%u)",
                        (unsigned)focusedTextureProgram->name,
                        sampledName ? sampledName : "",
                        (unsigned)spirvBinding,
                        (unsigned)textureUnit,
                        sampledResource ? (int)sampledResource->sampler_unit : -1,
                        (sampledResource && sampledResource->sampler_unit_explicit) ? 1 : 0,
                        ptr ? (unsigned)ptr->name : 0u,
                        ptr ? (unsigned)ptr->target : 0u,
                        usedFallbackTexture ? 1 : 0,
                        (unsigned long)expectedType,
                        (unsigned long)lookupType,
                        expectedIndex,
                        mglTraceTextureName(unitActive),
                        mglTraceTextureName(unitExpected),
                        mglTraceTextureName(unit2D),
                        mglTraceTextureName(unitCube),
                        texture,
                        (unsigned long)(texture ? mglBindingStateTextureType(texture) : 0),
                        (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0),
                        (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0),
                        level0 ? (unsigned)level0->width : 0u,
                        level0 ? (unsigned)level0->height : 0u,
                        level0 ? (unsigned)level0->ever_written : 0u,
                        level0 ? (unsigned)level0->has_initialized_data : 0u,
                        level0 ? (unsigned)level0->last_init_source : 0u);
        }
        if (texture && !usedFallbackTexture) {
            boundSampledTextures++;
        } else if (usedFallbackTexture) {
            // Keep nilTex as the original GL binding failure count, while Metal receives fallback texture.
            nilSampledTextures++;
        } else {
            nilSampledTextures++;
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
                mglBindingLogInSamplerDepthHistorySuppressed(
                    s_histSup, fragmentProgramName, spirvBinding, textureUnit,
                    pairedFboName, curAtt, ptr ? ptr->name : 0u,
                    pairedColor ? pairedColor->name : 0u);
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
                    mglBindingLogInSamplerDepthNoCopy(
                        s_noCopy, fragmentProgramName, spirvBinding, textureUnit,
                        pairedFboName, curAtt, ptr ? ptr->name : 0u,
                        pairedColor ? pairedColor->name : 0u,
                        mglBindingStateTexturePixelFormat(texture),
                        pairedColor ? pairedColor->mtl_gl_sampled_write_version : 0u,
                        pairedColor ? pairedColor->mtl_render_target_write_version : 0u);
                }
                texture = nil;
                suppressMissingTextureFallback = YES;
            }
        } else if (iplan.action == MGL_DR_ACTION_USE_PAIRED_DIRECT) {
            static uint64_t s_rec = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_rec)) {
                mglBindingLogInSamplerDepthPairedDirect(
                    s_rec, fragmentProgramName, spirvBinding, textureUnit, pairedFboName,
                    ptr ? ptr->name : 0u, pairedColor->name,
                    mglBindingStateTexturePixelFormat(texture),
                    mglBindingStateTexturePixelFormat(pairedMTL),
                    mglBindingStateTextureWidth(pairedMTL),
                    mglBindingStateTextureHeight(pairedMTL));
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
                mglBindingLogInSamplerDepthUnpaired(
                    s_unp, fragmentProgramName, spirvBinding, textureUnit,
                    ptr ? ptr->name : 0u, mglBindingStateTexturePixelFormat(texture),
                    mglBindingStateTextureWidth(texture),
                    mglBindingStateTextureHeight(texture));
            }
        }

        if (recoverTexture && recoverMTL) {
            static uint64_t s_histRec = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_histRec)) {
                mglBindingLogInSamplerDepthHistoryRecovery(
                    s_histRec, recoverReason, fragmentProgramName, spirvBinding,
                    textureUnit, pairedFboName, recoverAtt, ptr ? ptr->name : 0u,
                    recoverTexture ? recoverTexture->name : 0u,
                    mglBindingStateTexturePixelFormat(texture),
                    mglBindingStateTexturePixelFormat(recoverMTL),
                    mglBindingStateTextureWidth(recoverMTL),
                    mglBindingStateTextureHeight(recoverMTL),
                    recoveredFromSampledCopy ? 1 : 0,
                    recoveredFromPreviousVersion ? 1 : 0,
                    recoverTexture ? recoverTexture->mtl_gl_sampled_write_version : 0u,
                    recoverTexture ? recoverTexture->mtl_render_target_write_version : 0u,
                    pairedColor ? pairedColor->name : 0u, pairedCur ? 1 : 0);
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
                    mglBindingLogSampledDepthRtSkip(
                        s_skip, fragmentProgramName, sampledName, spirvBinding,
                        textureUnit, recoverFboName, drawAtt, ptr ? ptr->name : 0u,
                        pairedColor ? pairedColor->name : 0u);
                }
            }
        }
        if (!recoverTexture &&
            mglRendererTextureLooksRecoverableSampled2D(ctx, last2D, expectedType, expectedKind)) {
            static uint64_t s_sup = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_sup)) {
                mglBindingLogSampledDepthRtSuppressLast2D(
                    s_sup, fragmentProgramName, sampledName, spirvBinding, textureUnit,
                    ptr ? ptr->name : 0u, last2D->name);
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
                    mglBindingLogSampledDepthRtRecover(
                        s_rt, recoverReason, fragmentProgramName, sampledName, spirvBinding,
                        textureUnit, ptr ? ptr->name : 0u, recoverTexture->name,
                        mglBindingStateTexturePixelFormat(texture),
                        mglBindingStateTexturePixelFormat(recoverMTL),
                        mglBindingStateTextureWidth(texture),
                        mglBindingStateTextureHeight(texture), depthSampleLevel0,
                        depthSampleLevel0 ? depthSampleLevel0->ever_written : 0u,
                        depthSampleLevel0 ? depthSampleLevel0->has_initialized_data : 0u,
                        mglTraceTextureName(unitActive), mglTraceTextureName(unit2D),
                        mglTraceTextureName(last2D), recoverFboName,
                        currentFbo ? currentFbo->name : 0u, colorTexName, depthTexName);
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
                    mglBindingLogSampledDepthRtFallback(
                        s_fb, fragmentProgramName, sampledName, spirvBinding, textureUnit,
                        ptr ? ptr->name : 0u, mglBindingStateTexturePixelFormat(texture),
                        mglBindingStateTextureWidth(texture),
                        mglBindingStateTextureHeight(texture), depthSampleLevel0,
                        depthSampleLevel0 ? depthSampleLevel0->ever_written : 0u,
                        depthSampleLevel0 ? depthSampleLevel0->has_initialized_data : 0u,
                        mglTraceTextureName(unitActive), mglTraceTextureName(unit2D),
                        mglTraceTextureName(last2D));
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

- (bool)resolveFragmentSampledYFlipAndSampler:(Texture *)ptr
                                       texture:(id *)texturePtr
                                       sampler:(id *)samplerPtr
                                   sampledName:(const char *)sampledName
                                spirvBinding:(GLuint)spirvBinding
                                  textureUnit:(GLuint)textureUnit
                                 expectedType:(uint32_t)expectedType
                                 expectedKind:(MGLTextureDataKind)expectedKind
                         fragmentProgramName:(GLuint)fragmentProgramName
                          vertexProgramName:(GLuint)vertexProgramName
                                sampleProgram:(Program *)sampleProgram
                            usedFallbackTexture:(BOOL *)usedFallbackTexturePtr
                       usedSampledCopyForTrace:(BOOL *)usedSampledCopyForTracePtr
                          directTextureForTrace:(id *)directTextureForTracePtr
                          sampledCopyForTrace:(id *)sampledCopyForTracePtr
{
    id texture = *texturePtr;
    id sampler = *samplerPtr;
    BOOL usedFallbackTexture = *usedFallbackTexturePtr;
    BOOL usedSampledCopyForTrace = *usedSampledCopyForTracePtr;
    id directTextureForTrace = *directTextureForTracePtr;
    id sampledCopyForTrace = *sampledCopyForTracePtr;


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
        if (mglBindingTexturePlanSampled(&cin, &cplan) == 0 &&
            (cplan.action == MGL_ST_ACTION_TYPE_FALLBACK ||
             cplan.action == MGL_ST_ACTION_KIND_FALLBACK)) {
            static uint64_t s_fragmentCompatMismatchLogCount = 0;
            uint64_t hit = ++s_fragmentCompatMismatchLogCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL TEX %@ MISMATCH fragment binding=%u program=%u glTex=%u mtlType=%lu expected=%lu hit=%llu",
                      cplan.action == MGL_ST_ACTION_TYPE_FALLBACK ? @"TYPE"
                                                                  : @"DATA",
                      (unsigned)spirvBinding, (unsigned)fragmentProgramName,
                      (unsigned)ptr->name, (unsigned long)cin.mtl_type,
                      (unsigned long)expectedType, (unsigned long long)hit);
            }
            mglWriteProgramMSLDump(
                sampleProgram,
                [NSString
                    stringWithFormat:@"tex-%@-mismatch-fragment-binding-%u",
                                     cplan.action == MGL_ST_ACTION_TYPE_FALLBACK
                                         ? @"type"
                                         : @"data",
                                     spirvBinding]);
            texture = [self fallbackSampledTextureForExpectedType:expectedType
                                                         dataKind:expectedKind];
            usedFallbackTexture = YES;
            usedSampledCopyForTrace = NO;
        }
    }

    if (textureUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[textureUnit]) {
        Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
        if (glSampler->dirty_bits && glSampler->mtl_data) {
            mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
        }
        if (glSampler->mtl_data == NULL) {
            glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:ptr->target]);
            glSampler->dirty_bits = 0;
        }
        sampler = (__bridge id)(glSampler->mtl_data);
        mglTraceLogExternal("FRAG_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=glSampler samplerName=%u minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
                            (unsigned)fragmentProgramName,
                            (unsigned)spirvBinding,
                            (unsigned)textureUnit,
                            (unsigned)glSampler->name,
                            (unsigned)glSampler->params.min_filter,
                            (unsigned)glSampler->params.mag_filter,
                            (unsigned)glSampler->params.wrap_s,
                            (unsigned)glSampler->params.wrap_t,
                            (double)glSampler->params.min_lod,
                            (double)glSampler->params.max_lod,
                            (unsigned)ptr->name,
                            (unsigned)ptr->params.base_level,
                            (unsigned)ptr->params.max_level,
                            (unsigned)ptr->width,
                            (unsigned)ptr->height,
                            (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0u),
                            (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0u),
                            (unsigned long)(texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u));
    } else {
        sampler = (__bridge id)(ptr->params.mtl_data);
        mglTraceLogExternal("FRAG_SAMPLER_RESOLVE program=%u binding=%u unit=%u source=texParamsFallback samplerName=0 minFilter=0x%x magFilter=0x%x wrapS=0x%x wrapT=0x%x minLod=%.3f maxLod=%.3f glTex=%u base=%u max=%u texSize=%ux%u boundSize=%lux%lu boundLevels=%lu",
                            (unsigned)fragmentProgramName,
                            (unsigned)spirvBinding,
                            (unsigned)textureUnit,
                            (unsigned)ptr->params.min_filter,
                            (unsigned)ptr->params.mag_filter,
                            (unsigned)ptr->params.wrap_s,
                            (unsigned)ptr->params.wrap_t,
                            (double)ptr->params.min_lod,
                            (double)ptr->params.max_lod,
                            (unsigned)ptr->name,
                            (unsigned)ptr->params.base_level,
                            (unsigned)ptr->params.max_level,
                            (unsigned)ptr->width,
                            (unsigned)ptr->height,
                            (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0u),
                            (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0u),
                            (unsigned long)(texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u));
    }

    if (mglMipDiagEnabled() && ptr) {
        Sampler *glSampler = (textureUnit < TEXTURE_UNITS)
            ? MGL_STATE(ctx)->texture_samplers[textureUnit] : NULL;
        const TextureParameter *effective = glSampler ? &glSampler->params : &ptr->params;
        uint64_t signature = 1469598103934665603ULL;
        signature = mglMipDiagMixState(signature, ptr->name);
        signature = mglMipDiagMixState(signature, effective->min_filter);
        signature = mglMipDiagMixState(signature, effective->mag_filter);
        signature = mglMipDiagMixState(signature, ptr->params.base_level);
        signature = mglMipDiagMixState(signature, ptr->params.max_level);
        signature = mglMipDiagMixState(signature, texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u);
        signature = mglMipDiagMixState(signature, (uint64_t)(uintptr_t)texture);
        /* A render-target atlas is sampled through the Y-flip copy, so a mip
         * level left dirty or a version mismatch is what a stale mip looks like. */
        signature = mglMipDiagMixState(signature, usedSampledCopyForTrace ? 1u : 0u);
        signature = mglMipDiagMixState(signature, ptr->mtl_gl_sampled_levels);
        signature = mglMipDiagMixState(signature, ptr->mtl_gl_sampled_dirty_mip_mask);
        signature = mglMipDiagMixState(signature,
            (uint64_t)(ptr->mtl_gl_sampled_write_version != ptr->mtl_render_target_write_version));

        static uint64_t s_fragSamplerState[TEXTURE_UNITS];
        if (textureUnit < TEXTURE_UNITS &&
            mglMipDiagStateChanged(&s_fragSamplerState[textureUnit], signature)) {
            NSLog(@"MGL MIP_DIAG frag unit=%u binding=%u program=%u glTex=%u "
                  @"source=%s minFilter=0x%x magFilter=0x%x minLod=%.1f maxLod=%.1f aniso=%.1f "
                  @"base=%u max=%u glLevels=%u mtlLevels=%lu mtlW=%lu mtlH=%lu mtlTex=%p "
                  @"renderTarget=%d viaCopy=%d copyLevels=%u dirtyMips=0x%x rtVer=%u copyVer=%u",
                  (unsigned)textureUnit,
                  (unsigned)spirvBinding,
                  (unsigned)fragmentProgramName,
                  (unsigned)ptr->name,
                  glSampler ? "glSampler" : "texParams",
                  (unsigned)effective->min_filter,
                  (unsigned)effective->mag_filter,
                  (double)effective->min_lod,
                  (double)effective->max_lod,
                  (double)effective->max_anisotropy,
                  (unsigned)ptr->params.base_level,
                  (unsigned)ptr->params.max_level,
                  (unsigned)ptr->num_levels,
                  (unsigned long)(texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u),
                  (unsigned long)(texture ? mglBindingStateTextureWidth(texture) : 0u),
                  (unsigned long)(texture ? mglBindingStateTextureHeight(texture) : 0u),
                  texture,
                  ptr->is_render_target ? 1 : 0,
                  usedSampledCopyForTrace ? 1 : 0,
                  (unsigned)ptr->mtl_gl_sampled_levels,
                  (unsigned)ptr->mtl_gl_sampled_dirty_mip_mask,
                  (unsigned)ptr->mtl_render_target_write_version,
                  (unsigned)ptr->mtl_gl_sampled_write_version);
        }
    }

    *texturePtr = texture;
    *samplerPtr = sampler;
    *usedFallbackTexturePtr = usedFallbackTexture;
    *usedSampledCopyForTracePtr = usedSampledCopyForTrace;
    *directTextureForTracePtr = directTextureForTrace;
    *sampledCopyForTracePtr = sampledCopyForTrace;
    return true;
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
        mglTraceLog("RT_YFLIP_DECISION stage=%s program=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" decision=%s(%d) authority=0x%x rtVer=%u copyVer=%u hasCopy=%d sampleYFlip=%d",
                    stage ? stage : "?",
                    (unsigned)programName,
                    sampledName ? sampledName : "",
                    (unsigned)spirvBinding,
                    (unsigned)textureUnit,
                    (unsigned)ptr->name,
                    mglTraceTextureLabel(ptr),
                    mglYFlipDecisionName(yflip),
                    (int)yflip,
                    (unsigned)ptr->mtl_render_yflip_authority,
                    (unsigned)ptr->mtl_render_target_write_version,
                    (unsigned)ptr->mtl_gl_sampled_write_version,
                    ptr->mtl_gl_sampled_data ? 1 : 0,
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
            mglTraceLog("RT_SAMPLE_COPY_BIND stage=%s program=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" original=%p copy=%p",
                        stage ? stage : "?", (unsigned)programName,
                        sampledName ? sampledName : "", (unsigned)spirvBinding,
                        (unsigned)textureUnit, (unsigned)ptr->name,
                        mglTraceTextureLabel(ptr), texture, sampledCopy);
        }
        *texturePtr =
            (__bridge id)mglSampledTextureViewForBaseLevel(ptr, (__bridge void *)sampledCopy);
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
            *texturePtr = (__bridge id)mglSampledTextureViewForBaseLevel(
                ptr, (__bridge void *)repairedCopy);
            if (usedSampledCopyOut) {
                *usedSampledCopyOut = YES;
            }
        }
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_GATE_MISS && mglTraceLogIsEnabled()) {
        mglTraceLog("RT_SAMPLE_COPY_GATE_MISS stage=%s program=%u name=%s binding=%u unit=%u tex=%u label=\"%s\" isRT=%d hasCopy=%d canUse=%d expectedType=%lu",
                    stage ? stage : "?", (unsigned)programName,
                    sampledName ? sampledName : "", (unsigned)spirvBinding,
                    (unsigned)textureUnit, (unsigned)ptr->name,
                    mglTraceTextureLabel(ptr), 1,
                    ptr->mtl_gl_sampled_data ? 1 : 0, in.can_use_rt_copy,
                    (unsigned long)expectedType);
    } else if (plan.action == MGL_ST_ACTION_RT_ORIGINAL) {
        static uint64_t s_rtSampleCopySkipExistingFlipLogCount = 0;
        uint64_t hit = ++s_rtSampleCopySkipExistingFlipLogCount;
        if (mglTraceLogIsEnabled() && (hit <= 32ull || (hit % 512ull) == 0ull)) {
            mglTraceLog("RT_SAMPLE_COPY_SKIP_EXISTING_YFLIP hit=%llu stage=%s program=%u name=%s binding=%u tex=%u decision=%s(%d)",
                        (unsigned long long)hit, stage ? stage : "?",
                        (unsigned)programName, sampledName ? sampledName : "",
                        (unsigned)spirvBinding, (unsigned)(ptr ? ptr->name : 0u),
                        mglYFlipDecisionName(yflip), (int)yflip);
        }
        /* Fragment path: ensure base-level view on original RT. */
        if (texture && ptr->is_render_target && stage &&
            stage[0] == 'f') {
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

        id sampler = nil;
        if (textureUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[textureUnit]) {
            Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
            if (glSampler->dirty_bits && glSampler->mtl_data) {
                mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
            }
            if (glSampler->mtl_data == NULL) {
                glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:(GLuint)mglRenderSamplerObjectTarget()]);
                glSampler->dirty_bits = 0;
            }
            sampler = (__bridge id)(glSampler->mtl_data);
        }

        if (!sampler) {
            sampler = defaultSampler;
        }
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
                    if (textureUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[textureUnit]) {
                        Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
                        if (glSampler->mtl_data == NULL) {
                            glSampler->mtl_data = (void *)CFBridgingRetain(
                                [self createMTLSamplerForTexParam:&glSampler->params target:arrayTexture->target]);
                            glSampler->dirty_bits = 0;
                        }
                        metalSampler = (__bridge id)(glSampler->mtl_data);
                    } else if (arrayTexture->params.mtl_data) {
                        metalSampler = (__bridge id)(arrayTexture->params.mtl_data);
                    }
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
