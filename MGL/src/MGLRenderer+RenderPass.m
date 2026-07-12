// MGLRenderer+RenderPass.m
// Render pass lifecycle methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+RenderPass_Private.h"

/* === MGLDepthStencilCacheKey ===
 * Lightweight value key covering every field of MTLDepthStencilDescriptor that
 * affects the created id<MTLDepthStencilState>.  Used only when the
 * MGL_DS_CACHE env var gates the depth/stencil state cache (default ON; =0 off).  All fields
 * are stored as primitives for fast isEqual:/hash.
 *
 * Note: MTLDepthStencilDescriptor in the target SDK exposes
 * depthCompareFunction, depthWriteEnabled, frontFaceStencil and
 * backFaceStencil (each stencil descriptor exposes stencilCompareFunction,
 * readMask, writeMask, stencilFailureOperation, depthFailureOperation and
 * depthStencilPassOperation).  depthClipMode is NOT a descriptor property in
 * this SDK — it is set on the render encoder directly and does not affect the
 * created state object, so it is intentionally not part of the key. */
@interface MGLDepthStencilCacheKey : NSObject <NSCopying>
    @property (nonatomic) MTLCompareFunction depthCompareFunction;
    @property (nonatomic) BOOL depthWriteEnabled;
    /* front stencil */
    @property (nonatomic) BOOL frontStencilPresent;
    @property (nonatomic) MTLCompareFunction frontStencilCompareFunction;
    @property (nonatomic) uint32_t frontReadMask;
    @property (nonatomic) uint32_t frontWriteMask;
    @property (nonatomic) MTLStencilOperation frontStencilFailureOperation;
    @property (nonatomic) MTLStencilOperation frontDepthFailureOperation;
    @property (nonatomic) MTLStencilOperation frontDepthStencilPassOperation;
    /* back stencil */
    @property (nonatomic) BOOL backStencilPresent;
    @property (nonatomic) MTLCompareFunction backStencilCompareFunction;
    @property (nonatomic) uint32_t backReadMask;
    @property (nonatomic) uint32_t backWriteMask;
    @property (nonatomic) MTLStencilOperation backStencilFailureOperation;
    @property (nonatomic) MTLStencilOperation backDepthFailureOperation;
    @property (nonatomic) MTLStencilOperation backDepthStencilPassOperation;
@end

@implementation MGLDepthStencilCacheKey

- (BOOL)isEqual:(id)object
{
    if (self == object) {
        return YES;
    }
    if (![object isKindOfClass:[MGLDepthStencilCacheKey class]]) {
        return NO;
    }
    MGLDepthStencilCacheKey *other = object;
    if (_depthCompareFunction != other.depthCompareFunction) return NO;
    if (_depthWriteEnabled != other.depthWriteEnabled) return NO;
    if (_frontStencilPresent != other.frontStencilPresent) return NO;
    if (_frontStencilCompareFunction != other.frontStencilCompareFunction) return NO;
    if (_frontReadMask != other.frontReadMask) return NO;
    if (_frontWriteMask != other.frontWriteMask) return NO;
    if (_frontStencilFailureOperation != other.frontStencilFailureOperation) return NO;
    if (_frontDepthFailureOperation != other.frontDepthFailureOperation) return NO;
    if (_frontDepthStencilPassOperation != other.frontDepthStencilPassOperation) return NO;
    if (_backStencilPresent != other.backStencilPresent) return NO;
    if (_backStencilCompareFunction != other.backStencilCompareFunction) return NO;
    if (_backReadMask != other.backReadMask) return NO;
    if (_backWriteMask != other.backWriteMask) return NO;
    if (_backStencilFailureOperation != other.backStencilFailureOperation) return NO;
    if (_backDepthFailureOperation != other.backDepthFailureOperation) return NO;
    if (_backDepthStencilPassOperation != other.backDepthStencilPassOperation) return NO;
    return YES;
}

- (NSUInteger)hash
{
    NSUInteger h = 0;
    h = h * 31 + (NSUInteger)_depthCompareFunction;
    h = h * 31 + (NSUInteger)(_depthWriteEnabled ? 1 : 0);
    h = h * 31 + (NSUInteger)(_frontStencilPresent ? 1 : 0);
    h = h * 31 + (NSUInteger)_frontStencilCompareFunction;
    h = h * 31 + (NSUInteger)_frontReadMask;
    h = h * 31 + (NSUInteger)_frontWriteMask;
    h = h * 31 + (NSUInteger)_frontStencilFailureOperation;
    h = h * 31 + (NSUInteger)_frontDepthFailureOperation;
    h = h * 31 + (NSUInteger)_frontDepthStencilPassOperation;
    h = h * 31 + (NSUInteger)(_backStencilPresent ? 1 : 0);
    h = h * 31 + (NSUInteger)_backStencilCompareFunction;
    h = h * 31 + (NSUInteger)_backReadMask;
    h = h * 31 + (NSUInteger)_backWriteMask;
    h = h * 31 + (NSUInteger)_backStencilFailureOperation;
    h = h * 31 + (NSUInteger)_backDepthFailureOperation;
    h = h * 31 + (NSUInteger)_backDepthStencilPassOperation;
    return h;
}

- (id)copyWithZone:(NSZone *)zone
{
    MGLDepthStencilCacheKey *copy = [[MGLDepthStencilCacheKey allocWithZone:zone] init];
    copy.depthCompareFunction = _depthCompareFunction;
    copy.depthWriteEnabled = _depthWriteEnabled;
    copy.frontStencilPresent = _frontStencilPresent;
    copy.frontStencilCompareFunction = _frontStencilCompareFunction;
    copy.frontReadMask = _frontReadMask;
    copy.frontWriteMask = _frontWriteMask;
    copy.frontStencilFailureOperation = _frontStencilFailureOperation;
    copy.frontDepthFailureOperation = _frontDepthFailureOperation;
    copy.frontDepthStencilPassOperation = _frontDepthStencilPassOperation;
    copy.backStencilPresent = _backStencilPresent;
    copy.backStencilCompareFunction = _backStencilCompareFunction;
    copy.backReadMask = _backReadMask;
    copy.backWriteMask = _backWriteMask;
    copy.backStencilFailureOperation = _backStencilFailureOperation;
    copy.backDepthFailureOperation = _backDepthFailureOperation;
    copy.backDepthStencilPassOperation = _backDepthStencilPassOperation;
    return copy;
}

@end

/* === Static C helpers used only by RenderPass methods === */

static bool mglGeometryShaderIsPassthrough(const Shader *shader)
{
    const char *src = shader ? shader->src : NULL;
    if (!src) {
        return false;
    }

    /* Metal has no geometry stage.  A few CTS paths insert a geometry shader
     * whose only job is to re-emit each input vertex unchanged while copying
     * clip/cull distance arrays.  Those programs are equivalent to running
     * the VS->FS pipeline directly, and blocking them regresses otherwise
     * valid cull-distance coverage.  Keep this deliberately narrow so real
     * geometry expansion/rewriting remains unsupported instead of being
     * silently misrendered. */
    return strstr(src, "EmitVertex()") &&
           strstr(src, "EndPrimitive()") &&
           strstr(src, "gl_Position = gl_in[n_vertex_index].gl_Position") &&
           !strstr(src, "gl_PrimitiveID") &&
           !strstr(src, "gl_Layer") &&
           !strstr(src, "gl_ViewportIndex");
}

@implementation MGLRenderer (RenderPass)

/* Returns a cached id<MTLDepthStencilState> for the given descriptor, creating
 * and caching a new one on miss.  Only used when _dsCacheEnabled == YES (gated
 * by MGL_DS_CACHE, default ON).  LRU eviction at 64 entries: NSDictionary preserves
 * insertion order on macOS 10.12+, so on a hit we remove+re-add the key to
 * mark it most-recently-used, and evict the first (oldest) key when the cache
 * exceeds capacity.  The MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap)
 * counter is the caller's responsibility (increment only on miss). */
- (id<MTLDepthStencilState>)cachedDepthStencilStateForDescriptor:(MTLDepthStencilDescriptor *)descriptor
{
    if (!_dsCacheEnabled || descriptor == nil) {
        return [_device newDepthStencilStateWithDescriptor:descriptor];
    }

    MGLDepthStencilCacheKey *key = [MGLDepthStencilCacheKey new];
    key.depthCompareFunction = descriptor.depthCompareFunction;
    key.depthWriteEnabled = descriptor.depthWriteEnabled;

    MTLStencilDescriptor *front = descriptor.frontFaceStencil;
    if (front) {
        key.frontStencilPresent = YES;
        key.frontStencilCompareFunction = front.stencilCompareFunction;
        key.frontReadMask = front.readMask;
        key.frontWriteMask = front.writeMask;
        key.frontStencilFailureOperation = front.stencilFailureOperation;
        key.frontDepthFailureOperation = front.depthFailureOperation;
        key.frontDepthStencilPassOperation = front.depthStencilPassOperation;
    }

    MTLStencilDescriptor *back = descriptor.backFaceStencil;
    if (back) {
        key.backStencilPresent = YES;
        key.backStencilCompareFunction = back.stencilCompareFunction;
        key.backReadMask = back.readMask;
        key.backWriteMask = back.writeMask;
        key.backStencilFailureOperation = back.stencilFailureOperation;
        key.backDepthFailureOperation = back.depthFailureOperation;
        key.backDepthStencilPassOperation = back.depthStencilPassOperation;
    }

    id<MTLDepthStencilState> cached = _depthStencilStateCache[key];
    if (cached) {
        /* LRU: move key to most-recently-used position by reinserting. */
        [_depthStencilStateCache removeObjectForKey:key];
        _depthStencilStateCache[key] = cached;
        return cached;
    }

    id<MTLDepthStencilState> state = [_device newDepthStencilStateWithDescriptor:descriptor];
    MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
    if (state) {
        _depthStencilStateCache[key] = state;
        if (_depthStencilStateCache.count > 64) {
            /* Evict least-recently-used (first inserted) entry. */
            MGLDepthStencilCacheKey *oldestKey = [_depthStencilStateCache.allKeys firstObject];
            if (oldestKey) {
                [_depthStencilStateCache removeObjectForKey:oldestKey];
            }
        }
    }
    return state;
}

- (void) bindMTLBuffer:(Buffer *) ptr
{
    METAL_LOCK();
    [self bindMTLBufferLocked:ptr];
    METAL_UNLOCK();
}

- (void) bindMTLBufferLocked:(Buffer *) ptr
{
    MTLResourceOptions options;
    const size_t kMaxSafeBufferSize = (size_t)2 * 1024 * 1024 * 1024; // 2 GiB safety cap

    if (!ptr) {
        NSLog(@"MGL ERROR: bindMTLBuffer called with NULL buffer");
        return;
    }

    // Corrupted buffer sizes can crash Metal validation immediately.
    if (ptr->size == 0 || ptr->size > kMaxSafeBufferSize) {
        NSLog(@"MGL ERROR: Refusing to create Metal buffer with suspicious size=%zu for buffer %u",
              (size_t)ptr->size, ptr->name);
        ptr->data.mtl_data = NULL;
        return;
    }

    options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared;

    // ways we will only write to this
    if ((ptr->storage_flags & GL_MAP_READ_BIT) == 0)
    {
        options |= MTLResourceCPUCacheModeWriteCombined;
    }

    if (ptr->transient_batch_buffer)
    {
        if (!ptr->data.buffer_data) {
            NSLog(@"MGL ERROR: transient batch buffer has no CPU backing (size=%zu)", (size_t)ptr->size);
            ptr->data.mtl_data = NULL;
            return;
        }

        id<MTLBuffer> buffer = [_device newBufferWithBytes:(void *)(uintptr_t)ptr->data.buffer_data
                                                     length:(NSUInteger)ptr->size
                                                    options:options];
        if (!buffer) {
            NSLog(@"MGL ERROR: Failed to create transient batch Metal buffer (size=%zu)", (size_t)ptr->size);
            ptr->data.mtl_data = NULL;
            return;
        }

        ptr->data.mtl_data = (void *)CFBridgingRetain(buffer);
        return;
    }

    if (ptr->storage_flags & GL_CLIENT_STORAGE_BIT)
    {
        if (!ptr->data.buffer_data) {
            NSLog(@"MGL ERROR: GL_CLIENT_STORAGE_BIT set but buffer_data is NULL for buffer %u", ptr->name);
            ptr->data.mtl_data = NULL;
            return;
        }

        id<MTLBuffer> buffer = [_device newBufferWithBytesNoCopy:(void *)(ptr->data.buffer_data)
                                                           length:ptr->size
                                                          options:options
                                                      deallocator:^(void *pointer, NSUInteger length)
                              {
                                  kern_return_t err;
                                  err = vm_deallocate((vm_map_t) mach_task_self(),
                                                      (vm_address_t) pointer,
                                                      length);
                                  if (err != 0) {
                                      NSLog(@"MGL WARNING: vm_deallocate failed for Metal no-copy buffer err=%d ptr=%p len=%lu",
                                            err,
                                            pointer,
                                            (unsigned long)length);
                                  }
                              }];

        ptr->data.mtl_data = (void *)CFBridgingRetain(buffer);
    }
    else
    {
        id<MTLBuffer> buffer;
        
        if (ptr->data.buffer_data)
        {
            size_t safeBufferSize = ptr->data.buffer_size;
            if (safeBufferSize == 0 || safeBufferSize > kMaxSafeBufferSize) {
                safeBufferSize = ptr->size;
            }

            buffer = [_device newBufferWithBytes:(void *)ptr->data.buffer_data
                                           length:safeBufferSize
                                          options:options];
            if (!buffer) {
                NSLog(@"MGL ERROR: Failed to create Metal buffer from CPU backing (size=%zu, buffer=%u)",
                      safeBufferSize, ptr->name);
                ptr->data.mtl_data = NULL;
                return;
            }
        }
        else
        {
            buffer = [_device newBufferWithLength: ptr->size // allocate by size
                                                        options: options];
            if (!buffer) {
                NSLog(@"MGL ERROR: Failed to allocate Metal buffer with length=%zu (buffer=%u)",
                      (size_t)ptr->size, ptr->name);
                ptr->data.mtl_data = NULL;
                return;
            }

            ptr->data.buffer_data = (vm_address_t)NULL;
        }

        ptr->data.mtl_data = (void *)CFBridgingRetain(buffer);
        if ((ptr->storage_flags & GL_MAP_PERSISTENT_BIT) &&
            (ptr->data.dirty_bits & DIRTY_BUFFER_DATA) &&
            buffer.length > 0 &&
            buffer.storageMode == MTLStorageModeManaged) {
            [buffer didModifyRange:NSMakeRange(0, buffer.length)];
        }
    }
}

- (void)mtlInvalidateRenderPass:(GLMContext)glm_ctx
{
    if (!glm_ctx || glm_ctx != ctx || !_currentRenderEncoder) {
        return;
    }

    /*
     * When the active framebuffer matches the render-pass framebuffer and the
     * draw-buffer selection hasn't changed, the invalidation is a spurious
     * side-effect of state setup (e.g. glDrawBuffers restoring the same values
     * after a glBindFramebuffer round-trip).  Skip it — no content is lost.
     */
    Framebuffer *curFbo = glm_ctx->state.framebuffer;
    if (curFbo == _renderPassFramebuffer &&
        glm_ctx->state.draw_buffer == _renderPassDrawBuffer) {
        return;
    }

    static uint64_t s_renderPassInvalidateCount = 0;
    uint64_t hit = ++s_renderPassInvalidateCount;
    if (mglTraceLogIsEnabled() && (hit <= 64ull || (hit % 512ull) == 0ull)) {
        Framebuffer *fbo = glm_ctx->state.framebuffer;
        mglTraceLog("RENDERPASS_INVALIDATE hit=%llu fbo=%u(%p) drawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x",
                    (unsigned long long)hit,
                    (unsigned)(fbo ? fbo->name : 0u),
                    fbo,
                    (unsigned)glm_ctx->state.draw_buffer,
                    (unsigned)_renderPassFramebufferName,
                    _renderPassFramebuffer,
                    (unsigned)_renderPassDrawBuffer);
        mglLogRenderPassLifecycle("invalidate-before-end",
                                  hit,
                                  glm_ctx,
                                  _currentCommandBuffer,
                                  _currentRenderEncoder,
                                  _renderPassDescriptor,
                                  _drawable,
                                  _renderPassFramebuffer,
                                  _renderPassFramebufferName,
                                  _renderPassDrawBuffer,
                                  _renderPassDrawBufferCount);
    }

    [self flushDrawBuffer:glm_ctx];
    [self endRenderEncoding];
}

- (Texture *)framebufferAttachmentTexture: (FBOAttachment *)fbo_attachment
{
    Texture *tex = NULL;

    if (!fbo_attachment) {
        NSLog(@"MGL ERROR: framebufferAttachmentTexture called with NULL attachment");
        return NULL;
    }

    if (fbo_attachment->textarget == GL_RENDERBUFFER)
    {
        if (fbo_attachment->buf.rbo) {
            tex = fbo_attachment->buf.rbo->tex;
        }
    }
    else
    {
        tex = fbo_attachment->buf.tex;
        if (!tex && fbo_attachment->texture != 0 && fbo_attachment->textarget != GL_RENDERBUFFER)
        {
            tex = findTexture(ctx, fbo_attachment->texture);
            if (tex)
            {
                fbo_attachment->buf.tex = tex;
            }
        }
    }
    if (!tex) {
        NSLog(@"MGL WARN: framebuffer attachment has no texture (target=0x%x)", fbo_attachment->textarget);
    }

    return tex;
}

- (bool)currentRenderPassMatchesCurrentFramebuffer
{
    if (!ctx || !_renderPassDescriptor) {
        return true;
    }

    Framebuffer *fbo = ctx->state.framebuffer;
    GLuint fboName = fbo ? fbo->name : 0u;
    if (_renderPassFramebuffer != fbo ||
        _renderPassFramebufferName != fboName ||
        _renderPassDrawBuffer != ctx->state.draw_buffer ||
        _renderPassDrawBufferCount != mglMetalDrawBufferCount(ctx)) {
        return false;
    }
    for (GLsizei i = 0; i < _renderPassDrawBufferCount; ++i) {
        if (_renderPassDrawBuffers[i] != mglMetalDrawBufferAt(ctx, (GLuint)i)) {
            return false;
        }
    }

    if (!fbo) {
        GLuint mgl_drawbuffer = mglDefaultDrawBufferIndexForGL(ctx->state.draw_buffer);
        id<MTLTexture> expectedColor0 = nil;
        id<MTLTexture> actualColor0 = _renderPassDescriptor.colorAttachments[0].texture;

        if (mgl_drawbuffer == _FRONT) {
            expectedColor0 = _drawable ? _drawable.texture : nil;
        } else if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            expectedColor0 = _drawBuffers[mgl_drawbuffer].drawbuffer;
        }

        if (actualColor0 != expectedColor0) {
            return false;
        }

        id<MTLTexture> expectedDepth = nil;
        id<MTLTexture> expectedStencil = nil;
        if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            BOOL defaultPassNeedsDepth = ctx->state.caps.depth_test ||
                                         _drawBuffers[mgl_drawbuffer].depthbuffer != nil;
            BOOL defaultPassNeedsStencil = ctx->state.caps.stencil_test ||
                                           ctx->stencil_format.format ||
                                           _drawBuffers[mgl_drawbuffer].stencilbuffer != nil;
            expectedDepth = defaultPassNeedsDepth ? _drawBuffers[mgl_drawbuffer].depthbuffer : nil;
            expectedStencil = defaultPassNeedsStencil ? _drawBuffers[mgl_drawbuffer].stencilbuffer : nil;
            if (ctx->state.caps.depth_test && !expectedDepth) {
                return false;
            }
            if ((ctx->state.caps.stencil_test || ctx->stencil_format.format) && !expectedStencil) {
                return false;
            }
        }

        if (_renderPassDescriptor.depthAttachment.texture != expectedDepth) {
            return false;
        }
        if (_renderPassDescriptor.stencilAttachment.texture != expectedStencil) {
            return false;
        }

        return true;
    }

    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        BOOL drawSlotPresent =
            mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, i),
                                                  &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) != 0u;
        FBOAttachment *attachment = drawSlotPresent ? &fbo->color_attachments[attachmentIndex] : NULL;
        Texture *tex = drawSlotPresent ? [self framebufferAttachmentTexture:attachment] : NULL;
        id<MTLTexture> expected = nil;

        if (tex) {
            tex->is_render_target = true;
            if (!tex->mtl_data) {
                if (![self bindMTLTexture:tex]) {
                    return false;
                }
            }
            expected = (__bridge id<MTLTexture>)(tex->mtl_data);
        }

        id<MTLTexture> actual = _renderPassDescriptor.colorAttachments[colorSlot].texture;
        if (actual != expected) {
            return false;
        }

        if (attachment && actual) {
            MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(attachment);
            if (!mglMetalRenderPassColorAttachmentMatchesSubresource(_renderPassDescriptor.colorAttachments[colorSlot], subresource)) {
                return false;
            }
        }

        if (i + 1u >= MAX_COLOR_ATTACHMENTS ||
            (mglMetalDrawBufferAt(ctx, i + 1u) == GL_NONE &&
             !_renderPassDescriptor.colorAttachments[i + 1u].texture)) {
            break;
        }
    }

    id<MTLTexture> expectedDepth = nil;
    if (fbo->depth.texture) {
        Texture *depthTex = [self framebufferAttachmentTexture:&fbo->depth];
        if (depthTex && !depthTex->mtl_data) {
            depthTex->is_render_target = true;
            if (![self bindMTLTexture:depthTex]) {
                return false;
            }
        }
        expectedDepth = depthTex ? (__bridge id<MTLTexture>)(depthTex->mtl_data) : nil;
    }
    if (_renderPassDescriptor.depthAttachment.texture != expectedDepth) {
        return false;
    }
    if (fbo->depth.texture && expectedDepth) {
        MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
        if (!mglMetalRenderPassDepthAttachmentMatchesSubresource(_renderPassDescriptor.depthAttachment, subresource)) {
            return false;
        }
    }

    id<MTLTexture> expectedStencil = nil;
    if (fbo->stencil.texture) {
        Texture *stencilTex = [self framebufferAttachmentTexture:&fbo->stencil];
        if (stencilTex && !stencilTex->mtl_data) {
            stencilTex->is_render_target = true;
            if (![self bindMTLTexture:stencilTex]) {
                return false;
            }
        }
        expectedStencil = stencilTex ? (__bridge id<MTLTexture>)(stencilTex->mtl_data) : nil;
    }
    if (_renderPassDescriptor.stencilAttachment.texture != expectedStencil) {
        return false;
    }
    if (fbo->stencil.texture && expectedStencil) {
        MGLMetalAttachmentSubresource subresource = mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
        if (!mglMetalRenderPassStencilAttachmentMatchesSubresource(_renderPassDescriptor.stencilAttachment, subresource)) {
            return false;
        }
    }

    return true;
}

- (bool)ensureCurrentRenderPassMatchesFramebufferForDraw
{
    if (!ctx) {
        return true;
    }

    if (!_currentRenderEncoder) {
        return true;
    }

    if ([self currentRenderPassMatchesCurrentFramebuffer]) {
        return true;
    }

    static uint64_t s_fboPassMismatchCount = 0;
    uint64_t hit = ++s_fboPassMismatchCount;
    if (hit <= 32ull || (hit % 256ull) == 0ull) {
        Framebuffer *fbo = ctx->state.framebuffer;
        id<MTLTexture> color0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
        GLuint mglDefaultDrawbuffer = fbo ? 0u : mglDefaultDrawBufferIndexForGL(ctx->state.draw_buffer);
        id<MTLTexture> expectedDefaultColor0 = nil;
        if (!fbo) {
            expectedDefaultColor0 = (mglDefaultDrawbuffer == _FRONT)
                ? (_drawable ? _drawable.texture : nil)
                : ((mglDefaultDrawbuffer < _MAX_DRAW_BUFFERS) ? _drawBuffers[mglDefaultDrawbuffer].drawbuffer : nil);
        }
        GLuint fboName = fbo ? fbo->name : 0u;
        GLuint attachment0Name = (fbo && (fbo->color_attachment_bitfield & 1u)) ? fbo->color_attachments[0].texture : 0u;
        NSLog(@"MGL WARNING: render pass/FBO mismatch before draw hit=%llu fbo=%u drawBuffer=0x%x attachment0=%u passColor0=%p expectedDefaultColor0=%p defaultDrawBuffer=%u; rebuilding encoder",
              (unsigned long long)hit,
              (unsigned)fboName,
              (unsigned)(ctx ? ctx->state.draw_buffer : 0u),
              (unsigned)attachment0Name,
              color0,
              expectedDefaultColor0,
              (unsigned)mglDefaultDrawbuffer);
        mglLogRenderPassLifecycle(fbo ? "fbo-mismatch-before-rebuild" : "default-fbo-mismatch-before-rebuild",
                                  hit,
                                  ctx,
                                  _currentCommandBuffer,
                                  _currentRenderEncoder,
                                  _renderPassDescriptor,
                                  _drawable,
                                  _renderPassFramebuffer,
                                  _renderPassFramebufferName,
                                  _renderPassDrawBuffer,
                                  _renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    ctx->state.dirty_bits |= (DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE | DIRTY_VAO);
    return [self newRenderEncoder];
}

- (void)endRenderPassIfFramebufferChangedForNonDraw:(uint64_t)processCall
{
    if (!ctx || !_currentRenderEncoder) {
        return;
    }

    if ([self currentRenderPassMatchesCurrentFramebuffer]) {
        return;
    }

    static uint64_t s_nonDrawFboMismatchCount = 0;
    uint64_t hit = ++s_nonDrawFboMismatchCount;
    if (mglTraceLogIsEnabled() && (hit <= 32ull || (hit % 256ull) == 0ull)) {
        Framebuffer *fbo = ctx->state.framebuffer;
        GLuint fboName = fbo ? fbo->name : 0u;
        mglTraceLog("RENDERPASS_NON_DRAW_MISMATCH processCall=%llu hit=%llu "
                    "ctxFbo=%u(%p) ctxDrawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x",
                    (unsigned long long)processCall,
                    (unsigned long long)hit,
                    (unsigned)fboName,
                    fbo,
                    (unsigned)ctx->state.draw_buffer,
                    (unsigned)_renderPassFramebufferName,
                    _renderPassFramebuffer,
                    (unsigned)_renderPassDrawBuffer);
        mglLogRenderPassLifecycle("non-draw-mismatch-before-end",
                                  hit,
                                  ctx,
                                  _currentCommandBuffer,
                                  _currentRenderEncoder,
                                  _renderPassDescriptor,
                                  _drawable,
                                  _renderPassFramebuffer,
                                  _renderPassFramebufferName,
                                  _renderPassDrawBuffer,
                                  _renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    ctx->state.dirty_bits |= (DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE);
}

- (bool)bindMTLTexture:(Texture *)tex
{
    METAL_LOCK();
    bool result = [self bindMTLTextureLocked:tex];
    METAL_UNLOCK();
    return result;
}

- (bool)bindMTLTextureLocked:(Texture *)tex
{
    if (tex && tex->target == GL_TEXTURE_BUFFER && tex->texture_buffer &&
        tex->texture_buffer->data.dirty_bits) {
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    }

    // If this texture is now used as a render target but was previously created
    // without render-target usage, force a recreate with proper usage flags.
    // When the old texture already has GPU-written data (e.g. from imageStore
    // in a compute shader), preserve it via a GPU-to-GPU blit instead of
    // re-uploading potentially stale CPU data.
    if (tex->mtl_data && tex->is_render_target) {
        id<MTLTexture> existingTexture = (__bridge id<MTLTexture>)(tex->mtl_data);
        MTLTextureUsage requiredRenderTargetUsage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        NSUInteger requiredMipLevels =
            (tex->target == GL_RENDERBUFFER || tex->samples > 1u)
                ? 1u
                : ((tex->mipmap_levels > 1u) ? (NSUInteger)tex->mipmap_levels : 1u);
        BOOL usageMismatch = existingTexture &&
            ((existingTexture.usage & requiredRenderTargetUsage) != requiredRenderTargetUsage);
        BOOL mipCountMismatch = existingTexture &&
            requiredMipLevels > existingTexture.mipmapLevelCount;
        if (existingTexture && (usageMismatch || mipCountMismatch)) {
            NSLog(@"MGL WARNING: Recreating texture %u for render-target use (old usage=0x%lx oldMips=%lu requiredMips=%lu)",
                  tex->name,
                  (unsigned long)existingTexture.usage,
                  (unsigned long)existingTexture.mipmapLevelCount,
                  (unsigned long)requiredMipLevels);

            // Keep a strong reference to the old texture so we can blit its GPU
            // data to the new one after releasing tex->mtl_data.
            __strong id<MTLTexture> oldTexture = existingTexture;

            mglSafeReleaseMetalObj((void **)&tex->mtl_data);
            [self releaseGLSampledRenderTargetCopyForTexture:tex];

            // Create a new texture with correct usage.  Don't set
            // DIRTY_TEXTURE_DATA so that createMTLTextureFromGLTexture
            // skips CPU data upload — we'll blit GPU data instead.
            id<MTLTexture> newTexture = [self createMTLTextureFromGLTexture:tex];
            if (newTexture && oldTexture &&
                newTexture.width == oldTexture.width &&
                newTexture.height == oldTexture.height &&
                newTexture.depth == oldTexture.depth) {
                // Blit GPU data from old texture to new texture to preserve
                // any writes (e.g. imageStore) that occurred before the
                // is_render_target transition.
                [self endRenderEncodingLocked];
                if ([self ensureWritableCommandBufferLocked:"is_render_target_blit"]) {
                    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
                    if (blit) {
                        NSUInteger copySlices = MIN(oldTexture.arrayLength, newTexture.arrayLength);
                        NSUInteger copyLevels = MIN(oldTexture.mipmapLevelCount, newTexture.mipmapLevelCount);
                        for (NSUInteger slice = 0; slice < copySlices; slice++) {
                            for (NSUInteger level = 0; level < copyLevels; level++) {
                                NSUInteger lw = (oldTexture.width >> level);
                                NSUInteger lh = (oldTexture.height >> level);
                                if (lw == 0 || lh == 0) continue;
                                MTLSize levelSize = MTLSizeMake(lw, lh, 1);
                                [blit copyFromTexture:oldTexture
                                          sourceSlice:slice
                                          sourceLevel:level
                                         sourceOrigin:MTLOriginMake(0, 0, 0)
                                           sourceSize:levelSize
                                            toTexture:newTexture
                                       destinationSlice:slice
                                       destinationLevel:level
                                      destinationOrigin:MTLOriginMake(0, 0, 0)];
                            }
                        }
                        [blit endEncoding];
                    }
                }
                tex->mtl_data = (void *)CFBridgingRetain(newTexture);
                tex->dirty_bits = 0;
            } else {
                // Fallback: use the old CPU-data-upload path
                tex->dirty_bits |= DIRTY_TEXTURE_DATA;
            }
        }
    }

    if (tex->dirty_bits)
    {
        bool textureNeedsRebuild =
            (tex->dirty_bits & (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_ACCESS)) != 0;
        bool samplerNeedsRebuild =
            textureNeedsRebuild || ((tex->dirty_bits & DIRTY_TEXTURE_PARAM) != 0);
        bool storageShapeChanged =
            (tex->dirty_bits & (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_ACCESS)) != 0;

        if (tex->mtl_data &&
            !storageShapeChanged &&
            (tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) {
            id<MTLTexture> existingTexture = (__bridge id<MTLTexture>)(tex->mtl_data);
            if (existingTexture &&
                [self uploadFullCPUTextureDataIntoTexture:tex
                                                     metal:existingTexture
                                                    reason:"bindMTLTexture.dirtyData"]) {
                textureNeedsRebuild =
                    (tex->dirty_bits & (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_ACCESS)) != 0;
                samplerNeedsRebuild =
                    textureNeedsRebuild || ((tex->dirty_bits & DIRTY_TEXTURE_PARAM) != 0);
            }
        }

        // Texture parameter changes only affect the Metal sampler object. Do
        // not throw away texture storage for wrap/filter/lod updates; doing so
        // can turn Minecraft's frequent sampler changes into render-pass and
        // upload storms.
        if (tex->mtl_data)
        {
            if (textureNeedsRebuild) {
                mglSafeReleaseMetalObj((void **)&tex->mtl_data);
                [self releaseGLSampledRenderTargetCopyForTexture:tex];
            }
        }

        if (samplerNeedsRebuild && tex->params.mtl_data)
        {
            mglSafeReleaseMetalObj((void **)&tex->params.mtl_data);
        }
    }

    if (tex->mtl_data == NULL)
    {
        tex->mtl_data = (void *)CFBridgingRetain([self createMTLTextureFromGLTexture: tex]);

        // AGX-SAFE: Handle NULL texture gracefully when in GPU recovery mode
        if (!tex->mtl_data) {
            // Circuit breaker: limit fallback texture creations to prevent infinite loops
            static int s_fallbackTextureCount = 0;
            static NSTimeInterval s_fallbackTextureWindowStart = 0;
            NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
            if (now - s_fallbackTextureWindowStart > 5.0) {
                s_fallbackTextureCount = 0;
                s_fallbackTextureWindowStart = now;
            }
            if (s_fallbackTextureCount >= 4096) {
                NSLog(@"MGL AGX: Fallback texture limit reached (%d in %.1fs), suppressing further fallbacks",
                      s_fallbackTextureCount, now - s_fallbackTextureWindowStart);
                tex->mtl_data = NULL;
                tex->dirty_bits = 0;
            } else {
                s_fallbackTextureCount++;
                NSLog(@"MGL AGX: Primary texture creation returned NULL, attempting fallback texture creation (%d/4096)",
                      s_fallbackTextureCount);
                // Create a simple fallback texture to prevent crashes
                tex->mtl_data = (void *)CFBridgingRetain([self createFallbackMTLTexture: tex]);

                if (tex->mtl_data) {
                    NSLog(@"MGL SUCCESS: Fallback texture created successfully");
                    tex->dirty_bits = 0;
                } else {
                    NSLog(@"MGL ERROR: Even fallback texture creation failed - this texture will remain NULL");
                }
            }
        } else {
            if (kMGLDiagnosticStateLogs) {
                MGLTraceNSLog(@"MGL SUCCESS: Primary texture created successfully");
            }
        }

    }

    if (tex->params.mtl_data == NULL)
    {
        tex->params.mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&tex->params target:tex->target]);
        // Sampler creation should not fail even in recovery mode
        if (!tex->params.mtl_data) {
            NSLog(@"MGL WARNING: Sampler creation failed, using default");
            tex->params.mtl_data = (void *)CFBridgingRetain([_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]]);
        }
        if ((tex->name == 21u || tex->name == 27u) &&
            mglEnvFlagEnabled("MGL_TRACE_TEXTURE_NAMES")) {
            mglTraceLogExternal("SAMPLER_CREATE tex=%u minFilter=0x%x magFilter=0x%x mipFilter=%d minLod=%.3f maxLod=%.3f base=%u max=%u mips=%u",
                                (unsigned)tex->name,
                                (unsigned)tex->params.min_filter,
                                (unsigned)tex->params.mag_filter,
                                (tex->params.min_filter >= 0x2700) ? 1 : 0,
                                (double)tex->params.min_lod,
                                (double)tex->params.max_lod,
                                (unsigned)tex->params.base_level,
                                (unsigned)tex->params.max_level,
                                (unsigned)tex->mipmap_levels);
        }
    }

    if (tex->params.mtl_data) {
        tex->dirty_bits &= ~DIRTY_TEXTURE_PARAM;
    }

    return true;
}

- (bool)restoreRenderEncoderAfterTextureUploadForDraw:(const char *)reason
{
    if (_currentRenderEncoder) {
        return true;
    }
    if (!ctx || !_renderPassDescriptor) {
        return false;
    }

    static uint64_t s_restoreAfterTextureUploadCount = 0;
    uint64_t hit = ++s_restoreAfterTextureUploadCount;
    if (hit <= 64ull || (hit % 512ull) == 0ull) {
        NSLog(@"MGL TEXTURE UPLOAD closed render encoder; restoring for draw reason=%s hit=%llu",
              reason ? reason : "(null)",
              (unsigned long long)hit);
    }

    if (![self ensureWritableCommandBuffer:reason ? reason : "restore_render_encoder_after_texture_upload"]) {
        return false;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (_renderPassDescriptor.colorAttachments[i].texture) {
            _renderPassDescriptor.colorAttachments[i].loadAction = MTLLoadActionLoad;
            _renderPassDescriptor.colorAttachments[i].storeAction = MTLStoreActionStore;
        }
    }
    if (_renderPassDescriptor.depthAttachment.texture) {
        _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionLoad;
        _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
    }
    if (_renderPassDescriptor.stencilAttachment.texture) {
        _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
        _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
    }

    @try {
        _currentRenderEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:_renderPassDescriptor];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload failed to create encoder: %@",
              exception.reason);
        _currentRenderEncoder = NULL;
        [self recordGPUError];
        return false;
    }
    if (!_currentRenderEncoder) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload returned nil encoder reason=%s",
              reason ? reason : "(null)");
        [self recordGPUError];
        return false;
    }
    _currentRenderEncoder.label = @"GL Render Encoder";
    mglTraceFragmentTextureTraceBindings("CLEAR",
                                         reason ? reason : "restore_render_encoder_after_texture_upload",
                                         _fragmentTextureTraceBindings,
                                         TEXTURE_UNITS,
                                         ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                         _pipelineProgramName);
    memset(_fragmentTextureTraceBindings, 0, sizeof(_fragmentTextureTraceBindings));
    _renderPassFramebuffer = ctx ? ctx->state.framebuffer : NULL;
    _renderPassFramebufferName = _renderPassFramebuffer ? _renderPassFramebuffer->name : 0u;
    _renderPassDrawBuffer = ctx ? ctx->state.draw_buffer : 0u;
    _renderPassDrawBufferCount = ctx ? mglMetalDrawBufferCount(ctx) : 0;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        _renderPassDrawBuffers[i] = (ctx && i < _renderPassDrawBufferCount)
            ? mglMetalDrawBufferAt(ctx, (GLuint)i)
            : GL_NONE;
    }
    [self updateCurrentRenderEncoder];

    if (!_pipelineState) {
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }

    @try {
        [_currentRenderEncoder setRenderPipelineState:_pipelineState];
        _lastPipelineState = _pipelineState;
        MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: restoring render encoder after texture upload failed to bind pipeline: %@",
              exception.reason);
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }

    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder]);
    return true;
}

- (bool)bindFramebufferTexture:(FBOAttachment *)fbo_attachment isDrawBuffer:(bool) isDrawBuffer
{
    Texture *tex;

    tex = [self framebufferAttachmentTexture: fbo_attachment];
    if (!tex) {
        // Incomplete/missing attachment. Do not crash.
        return true;
    }

    if (isDrawBuffer) {
        tex->is_render_target = true;
    }

    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);

    return true;
}

- (void)invalidateCurrentPipelineStateForReason:(NSString *)reason
{
    if (_pipelineState) {
        static uint64_t s_pipelineInvalidateCount = 0;
        uint64_t hit = ++s_pipelineInvalidateCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL WARNING: Invalidating current pipeline state after %@ hit=%llu",
                  reason ?: @"pipeline failure",
                  (unsigned long long)hit);
        }
    }
    _pipelineState = nil;
    _pipelineColor0Format = MTLPixelFormatInvalid;
    _pipelineDepthFormat = MTLPixelFormatInvalid;
    _pipelineStencilFormat = MTLPixelFormatInvalid;
    _pipelineProgramName = 0u;
}

-(bool)bindMTLProgram:(Program *)ptr
{
    METAL_LOCK();
    bool result = [self bindMTLProgramLocked:ptr];
    METAL_UNLOCK();
    return result;
}

-(bool)bindMTLProgramLocked:(Program *)ptr
{
    if (ptr->dirty_bits & DIRTY_PROGRAM)
    {
        // release mtl shaders
        for(int i=_VERTEX_SHADER; i<_MAX_SHADER_TYPES; i++)
        {
            Shader *shader;
            shader = ptr->shader_slots[i];

	            if (shader)
	            {
	                if (shader->mtl_data.library)
	                {
	                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.library);
	                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.function);
	                }
	                if (shader->mtl_data.zero_to_one_library)
	                {
	                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.zero_to_one_library);
	                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.zero_to_one_function);
	                }
	                if (shader->mtl_data.upper_left_library)
	                {
	                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.upper_left_library);
	                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.upper_left_function);
	                }
	                if (shader->mtl_data.upper_left_zero_to_one_library)
	                {
	                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.upper_left_zero_to_one_library);
	                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.upper_left_zero_to_one_function);
	                }
	            }
	        }

        ptr->dirty_bits &= ~DIRTY_PROGRAM;
    }

	    // bind mtl functions to shaders
	    for(int i=_VERTEX_SHADER; i<_MAX_SHADER_TYPES; i++)
	    {
	        Shader *shader;
	        shader = ptr->shader_slots[i];

        if (shader)
        {
            if (i == _GEOMETRY_SHADER) {
                if (shader->mtl_data.library) {
                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.library);
                }
                if (shader->mtl_data.function) {
                    mglSafeReleaseMetalObj((void **)&shader->mtl_data.function);
                }
                if (mglGeometryShaderIsPassthrough(shader)) {
                    static uint64_t s_passthroughGeometryShaderSkipCount = 0;
                    uint64_t hit = ++s_passthroughGeometryShaderSkipCount;
                    if (hit <= 16ull || (hit % 512ull) == 0ull) {
                        NSLog(@"MGL INFO: Skipping passthrough geometry shader for Metal program=%u hit=%llu",
                              (unsigned)ptr->name,
                              (unsigned long long)hit);
                    }
                    continue;
                }
                static uint64_t s_geometryShaderMetalSkipCount = 0;
                uint64_t hit = ++s_geometryShaderMetalSkipCount;
                if (hit <= 16ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL WARNING: Blocking draw for unsupported geometry shader program=%u hit=%llu",
                          (unsigned)ptr->name,
                          (unsigned long long)hit);
                }
                return false;
            }
            if (!ptr->spirv[i].msl_str) {
                NSLog(@"MGL WARNING: Program %u stage %d has reflection but no MSL; skipping Metal bind",
                      (unsigned)ptr->name,
                      i);
                shader->mtl_data.library = NULL;
                shader->mtl_data.function = NULL;
                return false;
            }
            if (shader->mtl_data.library == NULL)
            {
                id<MTLLibrary> library;
                id<MTLFunction> function;

                if (mglProgramExplicitlyTraced(ptr)) {
                    mglWriteProgramMSLDump(ptr, @"explicit-trace");
                }
                const char *compileMSL = ptr->spirv[i].msl_str;
                /* Cull distance emulation: Metal does not support gl_CullDistance
                 * primitive culling natively. When the vertex shader writes to
                 * mgl_CullDistance, inject code that reads the cull distance
                 * values of all vertices in the same primitive and moves the
                 * vertex off-screen if all vertices have a negative cull
                 * distance for any single cull distance entry.
                 *
                 * The draw path binds the vertex buffer to slot 29 and a
                 * params buffer to slot 28 so the shader can index into
                 * sibling vertices. */
                if (i == _VERTEX_SHADER && compileMSL && strstr(compileMSL, "mgl_CullDistance")) {
                    NSString *mslNS = [NSString stringWithUTF8String:compileMSL];
                    /* Inject the params struct definition after the includes. */
                    NSString *structDef = @"\nstruct MGLCullDistanceParams {\n"
                                           "    uint prim_vertex_count;\n"
                                           "    uint culldist_offset;\n"
                                           "    uint vertex_stride;\n"
                                           "    uint culldist_size;\n"
                                           "};\n";
                    NSString *includeMarker = @"using namespace metal;";
                    NSRange includeRange = [mslNS rangeOfString:includeMarker];
                    if (includeRange.location != NSNotFound) {
                        NSRange afterInclude = NSMakeRange(includeRange.location + includeRange.length, 0);
                        mslNS = [mslNS stringByReplacingCharactersInRange:afterInclude withString:structDef];
                    }
                    BOOL cullParamsInjected = NO;
                    NSRange closeParenRange =
                        mglRendererFindMSLEntryParameterClose(mslNS, shader->entry_point);
                    if (closeParenRange.location != NSNotFound) {
                        NSString *cullParams = [NSString stringWithFormat:@", uint mgl_vid [[vertex_id]], "
                                                 "device const float* mgl_cull_buf [[buffer(%d)]], "
                                                 "constant MGLCullDistanceParams* mgl_cull_params [[buffer(%d)]])",
                                                 kMGLCullDistanceVertexBufferIndex,
                                                 kMGLCullDistanceParamsBufferIndex];
                        mslNS = [mslNS stringByReplacingCharactersInRange:closeParenRange
                                                               withString:cullParams];
                        cullParamsInjected = YES;
                    }
                    /* Inject the cull check before "return out;". */
                    NSString *cullCheck = @"\n    /* MGL cull distance emulation: if all vertices in this primitive "
                                            "have a negative value for any single cull distance entry, move "
                                            "the vertex off-screen to cull the entire primitive. For points "
                                            "(prim_vertex_count==1) this degenerates to checking the single "
                                            "vertex's own cull distance values. */\n"
                                            "    {\n"
                                            "        uint mgl_base = mgl_vid - (mgl_vid % mgl_cull_params->prim_vertex_count);\n"
                                            "        bool mgl_should_cull = false;\n"
                                            "        for (uint mgl_j = 0u; mgl_j < mgl_cull_params->culldist_size && !mgl_should_cull; mgl_j++) {\n"
                                            "            bool mgl_all_neg = true;\n"
                                            "            for (uint mgl_i = 0u; mgl_i < mgl_cull_params->prim_vertex_count; mgl_i++) {\n"
                                            "                uint mgl_other = mgl_base + mgl_i;\n"
                                            "                float mgl_d = mgl_cull_buf[mgl_other * (mgl_cull_params->vertex_stride / 4u) "
                                            "+ (mgl_cull_params->culldist_offset / 4u) + mgl_j];\n"
                                            "                if (mgl_d >= 0.0) { mgl_all_neg = false; break; }\n"
                                            "            }\n"
                                            "            if (mgl_all_neg) { mgl_should_cull = true; }\n"
                                            "        }\n"
                                            "        if (mgl_should_cull) {\n"
                                            "            out.gl_Position = float4(2.0, 2.0, 2.0, 1.0);\n"
                                            "        }\n"
                                            "    }\n"
                                            "    return out;";
                    /* Replace the last "return out;" occurrence. SPIRV-Cross
                     * vertex shaders always end with this.  Use NSBackwardsSearch
                     * to find the last occurrence (the entry function's return),
                     * and search for "return out;" without leading spaces because
                     * mglInjectMSLPointSizeBuiltin may have inserted
                     * "out.mgl_injected_point_size = 1.0; " before "return out;". */
                    NSRange returnRange = [mslNS rangeOfString:@"return out;"
                                                       options:NSBackwardsSearch];
                    if (cullParamsInjected && returnRange.location != NSNotFound) {
                        mslNS = [mslNS stringByReplacingCharactersInRange:returnRange withString:cullCheck];
                        compileMSL = [mslNS UTF8String];
                        if (getenv("MGL_DUMP_MSL_POST_PACK")) {
                            FILE *cullFP = fopen("/tmp/mgl_cull_emulation.msl", "w");
                            if (cullFP) { fputs(compileMSL, cullFP); fclose(cullFP); }
                        }
                    }
                }
                library = [self compileShader: compileMSL];
                if (!library) {
                    const char *stageName = "shader";
                    switch (i) {
                        case _VERTEX_SHADER: stageName = "vertex"; break;
                        case _TESS_CONTROL_SHADER: stageName = "tess-control"; break;
                        case _TESS_EVALUATION_SHADER: stageName = "tess-evaluation"; break;
                        case _GEOMETRY_SHADER: stageName = "geometry"; break;
                        case _FRAGMENT_SHADER: stageName = "fragment"; break;
                        case _COMPUTE_SHADER: stageName = "compute"; break;
                    }
                    NSLog(@"MGL ERROR: Failed to compile %s shader, skipping render", stageName);
                    shader->mtl_data.library = NULL;
                    shader->mtl_data.function = NULL;
                    return false;  // Signal shader compilation failure
                }
                NSString *entryName = [NSString stringWithUTF8String:shader->entry_point];
                function = [self newFunctionFromLibrary:library
                                              entryName:entryName
                                                 source:ptr->spirv[i].msl_str
                                                  label:entryName];
                if (!function) {
                    NSLog(@"MGL ERROR: Failed to find function '%s' in compiled shader", shader->entry_point);
                    shader->mtl_data.library = NULL;
                    shader->mtl_data.function = NULL;
                    return false;  // Signal function lookup failure
                }
                shader->mtl_data.library = (void *)CFBridgingRetain(library);
	                shader->mtl_data.function = (void *)CFBridgingRetain(function);
	            }
	        }
	    }

	    if (ctx &&
	        ctx->state.var.clip_depth_mode == GL_ZERO_TO_ONE &&
	        ptr->shader_slots[_VERTEX_SHADER] &&
	        ptr->shader_slots[_VERTEX_SHADER]->mtl_data.zero_to_one_library == NULL)
	    {
	        Shader *vertexShader = ptr->shader_slots[_VERTEX_SHADER];
	        NSString *variantSource = mglZeroToOneVertexMSLSource(ptr, vertexShader);
	        NSString *variantEntry = mglZeroToOneVertexEntryName(vertexShader);
	        if (!variantSource || !variantEntry) {
	            NSLog(@"MGL ERROR: Failed to create ZERO_TO_ONE vertex variant source for program=%u",
	                  (unsigned)ptr->name);
	            return false;
	        }

	        __autoreleasing NSError *error = nil;
	        id<MTLLibrary> library = [self newMetalLibraryWithSource:variantSource
	                                                         options:nil
	                                                           label:@"MGL ZERO_TO_ONE vertex shader"
	                                                           error:&error];
	        if (!library) {
	            NSLog(@"MGL ERROR: Failed to compile ZERO_TO_ONE vertex shader for program=%u: %@",
	                  (unsigned)ptr->name,
	                  error.localizedDescription ?: error);
	            return false;
	        }

	        id<MTLFunction> function = [self newFunctionFromLibrary:library
	                                                      entryName:variantEntry
	                                                         source:variantSource.UTF8String
	                                                          label:@"ZERO_TO_ONE vertex shader"];
	        if (!function) {
	            NSLog(@"MGL ERROR: Failed to find ZERO_TO_ONE vertex function '%@' for program=%u",
	                  variantEntry,
	                  (unsigned)ptr->name);
	            return false;
	        }

	        vertexShader->mtl_data.zero_to_one_library = (void *)CFBridgingRetain(library);
	        vertexShader->mtl_data.zero_to_one_function = (void *)CFBridgingRetain(function);
	    }

	    if (ctx &&
	        ctx->state.var.clip_origin == GL_UPPER_LEFT &&
	        ptr->shader_slots[_VERTEX_SHADER])
	    {
	        Shader *vertexShader = ptr->shader_slots[_VERTEX_SHADER];
	        BOOL zeroToOneDepth = (ctx->state.var.clip_depth_mode == GL_ZERO_TO_ONE);
	        BOOL needsVariant = zeroToOneDepth
	            ? (vertexShader->mtl_data.upper_left_zero_to_one_library == NULL)
	            : (vertexShader->mtl_data.upper_left_library == NULL);

	        if (needsVariant) {
	            NSString *variantSource = mglUpperLeftVertexMSLSource(ptr, vertexShader, zeroToOneDepth);
	            NSString *variantEntry = mglUpperLeftVertexEntryName(vertexShader, zeroToOneDepth);
	            if (!variantSource || !variantEntry) {
	                NSLog(@"MGL ERROR: Failed to create UPPER_LEFT vertex variant source for program=%u depthMode=0x%x",
	                      (unsigned)ptr->name,
	                      (unsigned)ctx->state.var.clip_depth_mode);
	                return false;
	            }

	            __autoreleasing NSError *error = nil;
	            id<MTLLibrary> library = [self newMetalLibraryWithSource:variantSource
	                                                             options:nil
	                                                               label:@"MGL UPPER_LEFT vertex shader"
	                                                               error:&error];
	            if (!library) {
	                NSLog(@"MGL ERROR: Failed to compile UPPER_LEFT vertex shader for program=%u depthMode=0x%x: %@",
	                      (unsigned)ptr->name,
	                      (unsigned)ctx->state.var.clip_depth_mode,
	                      error.localizedDescription ?: error);
	                return false;
	            }

	            id<MTLFunction> function = [self newFunctionFromLibrary:library
	                                                          entryName:variantEntry
	                                                             source:variantSource.UTF8String
	                                                              label:@"UPPER_LEFT vertex shader"];
	            if (!function) {
	                NSLog(@"MGL ERROR: Failed to find UPPER_LEFT vertex function '%@' for program=%u",
	                      variantEntry,
	                      (unsigned)ptr->name);
	                return false;
	            }

	            if (zeroToOneDepth) {
	                vertexShader->mtl_data.upper_left_zero_to_one_library = (void *)CFBridgingRetain(library);
	                vertexShader->mtl_data.upper_left_zero_to_one_function = (void *)CFBridgingRetain(function);
	            } else {
	                vertexShader->mtl_data.upper_left_library = (void *)CFBridgingRetain(library);
	                vertexShader->mtl_data.upper_left_function = (void *)CFBridgingRetain(function);
	            }
	        }
	    }

	    return true;
	}

- (void) updateCurrentRenderEncoder
{
    GLMState *state = MGL_STATE(ctx);
    BOOL passHasDepthAttachment =
        (_renderPassDescriptor != nil &&
         _renderPassDescriptor.depthAttachment.texture != nil);
    BOOL passHasStencilAttachment =
        (_renderPassDescriptor != nil &&
         _renderPassDescriptor.stencilAttachment.texture != nil);
    BOOL useDepthState = state->caps.depth_test && passHasDepthAttachment;
    BOOL useStencilState = state->caps.stencil_test && passHasStencilAttachment;

    if (state->caps.depth_test && !passHasDepthAttachment) {
        static uint64_t s_missingDepthAttachmentCount = 0;
        uint64_t hit = ++s_missingDepthAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            NSLog(@"MGL WARNING: depth test/write requested without depth attachment, disabling depth for this pass hit=%llu fbo=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  state->draw_buffer);
        }
    }

    if (state->caps.stencil_test && !passHasStencilAttachment) {
        static uint64_t s_missingStencilAttachmentCount = 0;
        uint64_t hit = ++s_missingStencilAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            NSLog(@"MGL WARNING: stencil test requested without stencil attachment, disabling stencil for this pass hit=%llu fbo=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  state->draw_buffer);
        }
    }

    if (useDepthState || useStencilState)
    {
        MTLDepthStencilDescriptor *dsDesc = [[MTLDepthStencilDescriptor alloc] init];

        if (useDepthState)
        {
            if (!mglIsValidGLCompareFunction(state->var.depth_func)) {
                mglLogRenderStateRepair("depth_func", state->var.depth_func, GL_LESS);
                state->var.depth_func = GL_LESS;
                state->dirty_bits |= DIRTY_RENDER_STATE;
            }

            dsDesc.depthCompareFunction =
                mglMTLCompareFunctionForGL(state->var.depth_func,
                                           MTLCompareFunctionLess,
                                           "depth");
            dsDesc.depthWriteEnabled = state->var.depth_writemask;
        }

        if (useStencilState)
        {
            {
                if (!mglIsValidGLCompareFunction(state->var.stencil_func)) {
                    mglLogRenderStateRepair("stencil_func", state->var.stencil_func, GL_ALWAYS);
                    state->var.stencil_func = GL_ALWAYS;
                    state->dirty_bits |= DIRTY_RENDER_STATE;
                }

                MTLStencilDescriptor *frontSDesc = [[MTLStencilDescriptor alloc] init];

                frontSDesc.stencilCompareFunction =
                    mglMTLCompareFunctionForGL(state->var.stencil_func,
                                               MTLCompareFunctionAlways,
                                               "front-stencil");
                frontSDesc.stencilFailureOperation = [self mtlStencilOpForGLOp:state->var.stencil_fail ];
                frontSDesc.depthFailureOperation = [self mtlStencilOpForGLOp:state->var.stencil_pass_depth_fail];
                frontSDesc.depthStencilPassOperation = [self mtlStencilOpForGLOp:state->var.stencil_pass_depth_pass];
                frontSDesc.writeMask = state->var.stencil_writemask;
                frontSDesc.readMask = state->var.stencil_value_mask;    // ????

                dsDesc.frontFaceStencil = frontSDesc;
            }

            {
                if (!mglIsValidGLCompareFunction(state->var.stencil_back_func)) {
                    mglLogRenderStateRepair("stencil_back_func", state->var.stencil_back_func, GL_ALWAYS);
                    state->var.stencil_back_func = GL_ALWAYS;
                    state->dirty_bits |= DIRTY_RENDER_STATE;
                }

                MTLStencilDescriptor *backSDesc = [[MTLStencilDescriptor alloc] init];

                backSDesc.stencilCompareFunction =
                    mglMTLCompareFunctionForGL(state->var.stencil_back_func,
                                               MTLCompareFunctionAlways,
                                               "back-stencil");
                backSDesc.stencilFailureOperation = [self mtlStencilOpForGLOp:state->var.stencil_back_fail ];
                backSDesc.depthFailureOperation = [self mtlStencilOpForGLOp:state->var.stencil_back_pass_depth_fail];
                backSDesc.depthStencilPassOperation = [self mtlStencilOpForGLOp:state->var.stencil_back_pass_depth_pass];
                backSDesc.writeMask = state->var.stencil_back_writemask;
                backSDesc.readMask = state->var.stencil_back_value_mask;    // ????

                dsDesc.backFaceStencil = backSDesc;
            }
        }

        id <MTLDepthStencilState> dsState;
        if (_dsCacheEnabled) {
            /* Cache miss path inside cachedDepthStencilStateForDescriptor:
             * increments g_mglDepthStencilStateCreatesSinceSwap; cache hits
             * do not, so the counter reflects real device calls only. */
            dsState = [self cachedDepthStencilStateForDescriptor:dsDesc];
        } else {
            dsState = [_device newDepthStencilStateWithDescriptor:dsDesc];
            MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
        }

        if (!_lastBoundValid || _lastDepthStencilState != dsState) {
            [_currentRenderEncoder setDepthStencilState: dsState];
            _lastDepthStencilState = dsState;
        } else {
            MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
        }
        if (useStencilState) {
            [_currentRenderEncoder setStencilFrontReferenceValue:(uint32_t)state->var.stencil_ref
                                              backReferenceValue:(uint32_t)state->var.stencil_back_ref];
        }
    }
    else
    {
        MTLDepthStencilDescriptor *disabledDSDesc = [[MTLDepthStencilDescriptor alloc] init];
        disabledDSDesc.depthCompareFunction = MTLCompareFunctionAlways;
        disabledDSDesc.depthWriteEnabled = NO;

        id <MTLDepthStencilState> disabledDSState;
        if (_dsCacheEnabled) {
            disabledDSState = [self cachedDepthStencilStateForDescriptor:disabledDSDesc];
        } else {
            disabledDSState = [_device newDepthStencilStateWithDescriptor:disabledDSDesc];
            MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
        }
        if (disabledDSState) {
            if (!_lastBoundValid || _lastDepthStencilState != disabledDSState) {
                [_currentRenderEncoder setDepthStencilState:disabledDSState];
                _lastDepthStencilState = disabledDSState;
            } else {
                MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
            }
        }
    }

    [_currentRenderEncoder setBlendColorRed:state->var.blend_color[0]
                                      green:state->var.blend_color[1]
                                       blue:state->var.blend_color[2]
                                      alpha:state->var.blend_color[3]];

    /* GL_SAMPLE_MASK: Metal does not expose a per-draw sample mask setter on
     * MTLRenderCommandEncoder.  Sample coverage in Metal is controlled via
     * alpha-to-coverage and shader-side [[sample_mask]], neither of which
     * maps cleanly to GL_SAMPLE_MASK.  This remains a known limitation. */

    [self updateViewportAndScissorLocked];

    if (state->var.front_face != GL_CW && state->var.front_face != GL_CCW) {
        mglLogRenderStateRepair("front_face", state->var.front_face, GL_CCW);
        state->var.front_face = GL_CCW;
        state->dirty_bits |= DIRTY_RENDER_STATE;
    }

    BOOL defaultFramebufferSampledPass =
        state->framebuffer == NULL &&
        !state->caps.depth_test &&
        [self getProgramBindingCount:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE] > 0;
    BOOL rtSampledCopyDraw = _currentDrawUsesRTSampledCopy;

    if (state->caps.cull_face && !defaultFramebufferSampledPass && !rtSampledCopyDraw)
    {
        MTLCullMode cull_mode;

        switch(state->var.cull_face_mode)
        {
            case GL_BACK: cull_mode = MTLCullModeBack; break;
            case GL_FRONT: cull_mode = MTLCullModeFront; break;
            default:
                cull_mode = MTLCullModeNone;
        }

        if (!_lastBoundValid || _lastCullMode != cull_mode) {
            [_currentRenderEncoder setCullMode:cull_mode];
            _lastCullMode = cull_mode;
        }
        MTLWinding _winding =
            mglMaybeInvertMTLWinding(mglMTLWindingForGL(state->var.front_face),
                                     state->var.clip_origin == GL_UPPER_LEFT);
        if (!_lastBoundValid || _lastFrontFacingWinding != _winding) {
            [_currentRenderEncoder setFrontFacingWinding:_winding];
            _lastFrontFacingWinding = _winding;
        }
    }
    else
    {
        if (!_lastBoundValid || _lastCullMode != MTLCullModeNone) {
            [_currentRenderEncoder setCullMode:MTLCullModeNone];
            _lastCullMode = MTLCullModeNone;
        }
        MTLWinding _winding =
            mglMaybeInvertMTLWinding(mglMTLWindingForGL(state->var.front_face),
                                     state->var.clip_origin == GL_UPPER_LEFT);
        if (!_lastBoundValid || _lastFrontFacingWinding != _winding) {
            [_currentRenderEncoder setFrontFacingWinding:_winding];
            _lastFrontFacingWinding = _winding;
        }

        if (state->caps.cull_face && defaultFramebufferSampledPass) {
            static uint64_t s_defaultSampledCullBypassCount = 0;
            uint64_t hit = ++s_defaultSampledCullBypassCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                MGLTraceNSLog(@"MGL TRACE default sampled pass cull bypass hit=%llu program=%u drawBuf=0x%x",
                      (unsigned long long)hit,
                      (unsigned)(ctx ? state->program_name : 0u),
                      (unsigned)(ctx ? state->draw_buffer : 0u));
            }
        }
        if (state->caps.cull_face && rtSampledCopyDraw) {
            static uint64_t s_rtSampledCopyCullBypassCount = 0;
            uint64_t hit = ++s_rtSampledCopyCullBypassCount;
            if (hit <= 64ull || (hit % 256ull) == 0ull) {
                mglTraceLog("RT_SAMPLE_COPY_CULL_BYPASS hit=%llu program=%u pipelineProgram=%u fbo=%u rpFbo=%u depth(test=%d write=%d func=0x%x) blend=%d cullFace=0x%x frontFace=0x%x",
                            (unsigned long long)hit,
                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                            (unsigned)_pipelineProgramName,
                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
                            (unsigned)_renderPassFramebufferName,
                            (ctx && state->caps.depth_test) ? 1 : 0,
                            (ctx && state->var.depth_writemask) ? 1 : 0,
                            (unsigned)(ctx ? state->var.depth_func : 0u),
                            (ctx && state->caps.blend) ? 1 : 0,
                            (unsigned)(ctx ? state->var.cull_face_mode : 0u),
                            (unsigned)(ctx ? state->var.front_face : 0u));
            }
        }
    }

    if (state->caps.depth_clamp)
    {
        [_currentRenderEncoder setDepthClipMode: MTLDepthClipModeClamp];
    }

    if (state->caps.polygon_offset_fill ||
        state->caps.polygon_offset_line ||
        state->caps.polygon_offset_point)
    {
        float _bias = state->var.polygon_offset_units;
        float _slope = state->var.polygon_offset_factor;
        float _clamp = 0.0f;
        if (!_lastBoundValid || _lastDepthBias != _bias ||
            _lastDepthBiasClamp != _clamp || _lastDepthSlopeScale != _slope) {
            [_currentRenderEncoder setDepthBias:_bias
                                     slopeScale:_slope
                                          clamp:_clamp];
            _lastDepthBias = _bias;
            _lastDepthBiasClamp = _clamp;
            _lastDepthSlopeScale = _slope;
        }
    }
    else
    {
        if (!_lastBoundValid || _lastDepthBias != 0.0f ||
            _lastDepthBiasClamp != 0.0f || _lastDepthSlopeScale != 0.0f) {
            [_currentRenderEncoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
            _lastDepthBias = 0.0f;
            _lastDepthBiasClamp = 0.0f;
            _lastDepthSlopeScale = 0.0f;
        }
    }

    MTLTriangleFillMode triangleFillMode = MTLTriangleFillModeFill;
    if (state->var.polygon_mode == GL_LINE)
    {
        triangleFillMode = MTLTriangleFillModeLines;
    }
    else if (state->var.polygon_mode != GL_FILL &&
             state->var.polygon_mode != GL_POINT)
    {
        mglLogRenderStateRepair("polygon_mode", state->var.polygon_mode, GL_FILL);
        state->var.polygon_mode = GL_FILL;
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];
}
/*
 * Viewport and scissor setup extracted from updateCurrentRenderEncoder.
 * Resolves render-pass dimensions, applies the scissor rect (with GL-to-Metal
 * origin conversion), and sets the viewport. Uses MGL_STATE(ctx) for
 * snapshot-based state access (Principle 2 compliance).
 */
- (void)updateViewportAndScissorLocked
{
    GLMState *state = MGL_STATE(ctx);
    // Metal validates viewport/scissor strictly against the active render pass dimensions.
    // Always derive pass size from the current attachments first (not from window drawable fallback).
    {
        static uint64_t s_encoderStateUpdateCount = 0;
        bool traceEncoderState = kMGLDiagnosticStateLogs || mglShouldTraceCall(++s_encoderStateUpdateCount);

        NSUInteger passWidth = 0;
        NSUInteger passHeight = 0;
        id<MTLTexture> passTexture = nil;

        if (_renderPassDescriptor) {
            passWidth = _renderPassDescriptor.renderTargetWidth;
            passHeight = _renderPassDescriptor.renderTargetHeight;

            if (passWidth == 0 || passHeight == 0) {
                for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                    id<MTLTexture> candidate = _renderPassDescriptor.colorAttachments[i].texture;
                    if (candidate) {
                        passTexture = candidate;
                        break;
                    }
                }

                if (!passTexture) {
                    passTexture = _renderPassDescriptor.depthAttachment.texture;
                }
                if (!passTexture) {
                    passTexture = _renderPassDescriptor.stencilAttachment.texture;
                }

                if (passTexture) {
                    passWidth = passTexture.width;
                    passHeight = passTexture.height;
                    _renderPassDescriptor.renderTargetWidth = passWidth;
                    _renderPassDescriptor.renderTargetHeight = passHeight;
                    NSLog(@"MGL INFO: Resolved render pass size from attachment %lux%lu (rtw/rth were unset)",
                          (unsigned long)passWidth, (unsigned long)passHeight);
                }
            }
        }

        if ((passWidth == 0 || passHeight == 0) && _drawable && _drawable.texture) {
            passWidth = _drawable.texture.width;
            passHeight = _drawable.texture.height;
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: Falling back to drawable size for encoder state: %lux%lu",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if ((passWidth == 0 || passHeight == 0) && _layer) {
            CGSize drawableSize = _layer.drawableSize;
            if (drawableSize.width > 0 && drawableSize.height > 0) {
                passWidth = (NSUInteger)drawableSize.width;
                passHeight = (NSUInteger)drawableSize.height;
            } else {
                NSRect frame = [_layer frame];
                if (frame.size.width > 0 && frame.size.height > 0) {
                    passWidth = (NSUInteger)frame.size.width;
                    passHeight = (NSUInteger)frame.size.height;
                }
            }
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: Falling back to layer size for encoder state: %lux%lu",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if (passWidth > 0 && passHeight > 0) {
            GLint rawSx = 0;
            GLint rawSy = 0;
            GLint rawSw = (GLint)passWidth;
            GLint rawSh = (GLint)passHeight;

            GLint sx = 0;
            GLint sy = 0;
            GLint sw = (GLint)passWidth;
            GLint sh = (GLint)passHeight;

            if (state->caps.scissor_test) {
                rawSx = (GLint)state->var.scissor_box[0];
                rawSy = (GLint)state->var.scissor_box[1];
                rawSw = (GLint)state->var.scissor_box[2];
                rawSh = (GLint)state->var.scissor_box[3];

                sx = rawSx;
                sy = rawSy;
                sw = rawSw;
                sh = rawSh;

                // GL allows negative x/y; clamp origin and shrink extent accordingly.
                if (sx < 0) {
                    sw += sx;
                    sx = 0;
                }
                if (sy < 0) {
                    sh += sy;
                    sy = 0;
                }

                if (sx >= (GLint)passWidth || sy >= (GLint)passHeight) {
                    sx = 0;
                    sy = 0;
                    sw = (GLint)passWidth;
                    sh = (GLint)passHeight;
                } else {
                    GLint maxWidth = (GLint)passWidth - sx;
                    GLint maxHeight = (GLint)passHeight - sy;

                    if (sw > maxWidth) {
                        sw = maxWidth;
                    }
                    if (sh > maxHeight) {
                        sh = maxHeight;
                    }

                    if (sw <= 0 || sh <= 0) {
                        sx = 0;
                        sy = 0;
                        sw = (GLint)passWidth;
                        sh = (GLint)passHeight;
                    }
                }
            }

            GLint metalSy = sy;
            if (state->var.clip_origin != GL_UPPER_LEFT) {
                metalSy = (GLint)passHeight - (sy + sh);
                if (metalSy < 0) {
                    metalSy = 0;
                }
            }

            if (traceEncoderState || sx != rawSx || sy != rawSy || sw != rawSw || sh != rawSh || metalSy != sy) {
                NSLog(@"MGL SCISSOR apply pass=%lux%lu scissorEnabled=%d origin=0x%x raw=(%d,%d,%d,%d) glResolved=(%d,%d,%d,%d) metal=(%d,%d,%d,%d)",
                      (unsigned long)passWidth, (unsigned long)passHeight,
                      state->caps.scissor_test ? 1 : 0,
                      state->var.clip_origin,
                      rawSx, rawSy, rawSw, rawSh,
                      sx, sy, sw, sh,
                      sx, metalSy, sw, sh);
            }

            MTLScissorRect rect;
            rect.x = (NSUInteger)sx;
            rect.y = (NSUInteger)metalSy;
            rect.width = (NSUInteger)sw;
            rect.height = (NSUInteger)sh;
            [self setScissorRectIfNeeded:rect];

            GLdouble rawVx = (GLdouble)state->viewport[0];
            GLdouble rawVy = (GLdouble)state->viewport[1];
            GLdouble rawVw = (GLdouble)state->viewport[2];
            GLdouble rawVh = (GLdouble)state->viewport[3];

            GLdouble vx = rawVx;
            GLdouble vy = rawVy;
            GLdouble vw = rawVw;
            GLdouble vh = rawVh;

            if (vw <= 0.0 || vh <= 0.0) {
                vx = 0.0;
                vy = 0.0;
                vw = (GLdouble)passWidth;
                vh = (GLdouble)passHeight;
            }

            if (vx < 0.0) {
                vw += vx;
                vx = 0.0;
            }
            if (vy < 0.0) {
                vh += vy;
                vy = 0.0;
            }

            if (vx >= (GLdouble)passWidth || vy >= (GLdouble)passHeight) {
                vx = 0.0;
                vy = 0.0;
                vw = (GLdouble)passWidth;
                vh = (GLdouble)passHeight;
            } else {
                GLdouble maxVw = (GLdouble)passWidth - vx;
                GLdouble maxVh = (GLdouble)passHeight - vy;
                if (vw > maxVw) {
                    vw = maxVw;
                }
                if (vh > maxVh) {
                    vh = maxVh;
                }
                if (vw <= 0.0 || vh <= 0.0) {
                    vx = 0.0;
                    vy = 0.0;
                    vw = (GLdouble)passWidth;
                    vh = (GLdouble)passHeight;
                }
            }

            /*
             * glViewport's x/y select the same framebuffer rectangle regardless
             * of glClipControl origin.  The origin only changes how clip-space Y
             * maps within that rectangle; Metal still addresses the texture from
             * the top, so always convert GL's lower-left viewport rectangle to a
             * Metal top-left origin here.
             */
            GLdouble metalVy = (GLdouble)passHeight - (vy + vh);
            if (metalVy < 0.0) {
                metalVy = 0.0;
            }

            Texture *guiRTColor = NULL;
            Texture *guiRTDepth = NULL;
            BOOL guiRTPass =
                mglTraceLogIsEnabled() &&
                mglFramebufferLooksLikeGLSampledCopyRenderTarget(ctx,
                                                                 state->framebuffer,
                                                                 &guiRTColor,
                                                                 &guiRTDepth);
            if (guiRTPass) {
                static uint64_t s_guiRTEncoderStateLogCount = 0;
                uint64_t hit = ++s_guiRTEncoderStateLogCount;
                if (hit <= 128ull || (hit % 256ull) == 0ull) {
                    Program *program = mglResolveProgramFromState(ctx);
                    id<MTLTexture> c0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
                    id<MTLTexture> d0 = _renderPassDescriptor ? _renderPassDescriptor.depthAttachment.texture : nil;
                    mglTraceLog("RT_SAMPLE_COPY_ENCODER hit=%llu fbo=%u rpFbo=%u program=%u rtTex=%u label=\"%s\" depthTex=%u depthLabel=\"%s\" "
                          "pass=%lux%lu c0=%p fmt=%lu depth=%p fmt=%lu "
                          "loadStore(c=%s/%s d=%s/%s) clipOrigin=0x%x "
                          "scissor(en=%d raw=%d,%d,%d,%d metal=%d,%d,%d,%d) "
                          "viewport(raw=%.1f,%.1f,%.1f,%.1f metal=%.1f,%.1f,%.1f,%.1f) "
                          "depth(test=%d write=%d func=0x%x) blend=%d cull=%d levels=%u mips=%u mipmapped=%u",
                          (unsigned long long)hit,
                          state->framebuffer ? (unsigned)state->framebuffer->name : 0u,
                          (unsigned)_renderPassFramebufferName,
                          program ? (unsigned)program->name : (unsigned)state->program_name,
                          (unsigned)mglTraceTextureName(guiRTColor),
                          mglTraceTextureLabel(guiRTColor),
                          (unsigned)mglTraceTextureName(guiRTDepth),
                          mglTraceTextureLabel(guiRTDepth),
                          (unsigned long)passWidth,
                          (unsigned long)passHeight,
                          c0,
                          (unsigned long)(c0 ? c0.pixelFormat : MTLPixelFormatInvalid),
                          d0,
                          (unsigned long)(d0 ? d0.pixelFormat : MTLPixelFormatInvalid),
                          mglLoadActionName(_renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare),
                          mglStoreActionName(_renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare),
                          mglLoadActionName(_renderPassDescriptor ? _renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare),
                          mglStoreActionName(_renderPassDescriptor ? _renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare),
                          state->var.clip_origin,
                          state->caps.scissor_test ? 1 : 0,
                          rawSx, rawSy, rawSw, rawSh,
                          sx, metalSy, sw, sh,
                          rawVx, rawVy, rawVw, rawVh,
                          vx, metalVy, vw, vh,
                          state->caps.depth_test ? 1 : 0,
                          state->var.depth_writemask ? 1 : 0,
                          (unsigned)state->var.depth_func,
                          state->caps.blend ? 1 : 0,
                          state->caps.cull_face ? 1 : 0,
                          guiRTColor ? (unsigned)guiRTColor->num_levels : 0u,
                          guiRTColor ? (unsigned)guiRTColor->mipmap_levels : 0u,
                          guiRTColor ? (unsigned)guiRTColor->mipmapped : 0u);
                }
            }

            BOOL viewportWasClamped = (vx != rawVx || vy != rawVy || vw != rawVw || vh != rawVh);
            BOOL viewportOriginConverted = (metalVy != vy);
            if (traceEncoderState) {
                MGLTraceNSLog(@"MGL VIEWPORT apply pass=%lux%lu origin=0x%x raw=(%.3f,%.3f,%.3f,%.3f) resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                              (unsigned long)passWidth, (unsigned long)passHeight,
                              state->var.clip_origin,
                              rawVx, rawVy, rawVw, rawVh,
                              vx, vy, vw, vh,
                              vx, metalVy, vw, vh);
            }

            if (kMGLDiagnosticStateLogs && (viewportWasClamped || viewportOriginConverted)) {
                static uint64_t s_viewportClampDetailCount = 0;
                uint64_t clampHit = ++s_viewportClampDetailCount;
                BOOL logClampDetail = (clampHit <= 80ull || (clampHit % 120ull) == 0ull);

                if (logClampDetail) {
                    Framebuffer *debugFbo = state->framebuffer;
                    BOOL debugFboValid = (debugFbo != NULL &&
                                          mglRendererObjectPointerLikelyValid(debugFbo) &&
                                          mglRendererPointerInHashTable(&state->framebuffer_table, debugFbo) &&
                                          mglPointerRangeIsReadable(debugFbo, sizeof(*debugFbo)));
                    id<MTLTexture> rpColor0 = _renderPassDescriptor.colorAttachments[0].texture;
                    id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
                    id<MTLTexture> drawableTexture = (_drawable ? _drawable.texture : nil);

                    MGLTraceNSLog(@"MGL VIEWPORT CLAMP DETAIL hit=%llu fbo=%p valid=%d fboName=%u drawBuffer=0x%x pass=%lux%lu "
                                  "rpColor0=%p(%lux%lu) rpDepth=%p(%lux%lu) drawable=%p(%lux%lu) raw=(%.3f,%.3f,%.3f,%.3f) "
                                  "resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                                  (unsigned long long)clampHit,
                                  debugFbo,
                                  debugFboValid ? 1 : 0,
                                  (debugFboValid ? debugFbo->name : 0),
                                  state->draw_buffer,
                                  (unsigned long)passWidth,
                                  (unsigned long)passHeight,
                                  rpColor0,
                                  (unsigned long)(rpColor0 ? rpColor0.width : 0),
                                  (unsigned long)(rpColor0 ? rpColor0.height : 0),
                                  rpDepth,
                                  (unsigned long)(rpDepth ? rpDepth.width : 0),
                                  (unsigned long)(rpDepth ? rpDepth.height : 0),
                                  drawableTexture,
                                  (unsigned long)(drawableTexture ? drawableTexture.width : 0),
                                  (unsigned long)(drawableTexture ? drawableTexture.height : 0),
                                  rawVx, rawVy, rawVw, rawVh,
                                  vx, vy, vw, vh,
                                  vx, metalVy, vw, vh);

                    if (debugFboValid) {
                        for (int attIndex = 0; attIndex < MAX_COLOR_ATTACHMENTS; attIndex++) {
                            FBOAttachment *attachment = &debugFbo->color_attachments[attIndex];
                            if (attachment->texture == 0 && attachment->buf.tex == NULL && attachment->buf.rbo == NULL) {
                                continue;
                            }

                            Texture *attachmentTexture = NULL;
                            if (attachment->textarget == GL_RENDERBUFFER) {
                                attachmentTexture = attachment->buf.rbo ? attachment->buf.rbo->tex : NULL;
                            } else {
                                attachmentTexture = attachment->buf.tex;
                                if (!attachmentTexture && attachment->texture != 0) {
                                    attachmentTexture = findTexture(ctx, attachment->texture);
                                }
                            }

                            id<MTLTexture> attachmentMtl = (attachmentTexture && attachmentTexture->mtl_data)
                                ? (__bridge id<MTLTexture>)(attachmentTexture->mtl_data)
                                : nil;
                            id<MTLTexture> rpAttachment = _renderPassDescriptor.colorAttachments[attIndex].texture;

                            MGLTraceNSLog(@"MGL VIEWPORT CLAMP FBO att=%d name=%u textarget=0x%x level=%d layer=%d tex=%p "
                                          "texName=%u texTarget=0x%x texSize=%ux%ux%u mtl=%p(%lux%lu) rpTex=%p(%lux%lu)",
                                          attIndex,
                                          attachment->texture,
                                          attachment->textarget,
                                          attachment->level,
                                          attachment->layer,
                                          attachmentTexture,
                                          attachmentTexture ? attachmentTexture->name : 0,
                                          attachmentTexture ? attachmentTexture->target : 0,
                                          attachmentTexture ? attachmentTexture->width : 0,
                                          attachmentTexture ? attachmentTexture->height : 0,
                                          attachmentTexture ? attachmentTexture->depth : 0,
                                          attachmentMtl,
                                          (unsigned long)(attachmentMtl ? attachmentMtl.width : 0),
                                          (unsigned long)(attachmentMtl ? attachmentMtl.height : 0),
                                          rpAttachment,
                                          (unsigned long)(rpAttachment ? rpAttachment.width : 0),
                                          (unsigned long)(rpAttachment ? rpAttachment.height : 0));
                        }
                    }
                }
            }

            [self setViewportIfNeeded:(MTLViewport){vx, metalVy, vw, vh,
                                       state->var.depth_range[0], state->var.depth_range[1]}];
        } else {
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: updateCurrentRenderEncoder could not resolve pass size; using raw GL viewport");
            }
            [self setViewportIfNeeded:(MTLViewport){state->viewport[0], state->viewport[1],
                                       state->viewport[2], state->viewport[3],
                                       state->var.depth_range[0], state->var.depth_range[1]}];
        }
    }
}


- (bool) newRenderEncoder
{
    METAL_LOCK();
    bool result = [self newRenderEncoderLocked];
    METAL_UNLOCK();
    return result;
}

/*
 * Stage 4.2 — DontCare load-action inference for a color attachment.
 *
 * Returns YES only when it is provably safe to skip loading the attachment's
 * existing tile contents at pass start, i.e. the pass fully defines them:
 *   - env flag MGL_ENABLE_DONTCARE_LOAD is enabled (default OFF);
 *   - a real backing texture exists;
 *   - blending is disabled (a blend reads the destination, so contents live);
 *   - this is the texture's FIRST render-target use this frame (its stamped
 *     generation differs from the renderer's current frame generation) — a
 *     later pass to the same attachment must Load to preserve earlier draws.
 *
 * On a YES it stamps the texture with the current generation so any subsequent
 * pass this frame correctly falls back to Load. Callers invoke this only on the
 * no-pending-clear branch (a clear already fully defines contents via Clear).
 */
- (BOOL)shouldUseDontCareLoadForColorTexture:(Texture *)tex
                             firstUseThisFrame:(BOOL)firstUseThisFrame
{
    /* Pure decision — the caller is responsible for stamping the texture's
     * frame generation on EVERY render-target use (clear/load/dontcare), so
     * this predicate must not mutate state. DontCare is safe only on a frame's
     * first use of the attachment, with no pending clear (caller gates that),
     * blending off, and the flag enabled. */
    if (!mglEnvFlagEnabled("MGL_ENABLE_DONTCARE_LOAD")) {
        return NO;
    }
    if (!tex || !tex->mtl_data) {
        return NO;
    }
    if (ctx && ctx->state.caps.blend) {
        return NO;
    }
    if (!firstUseThisFrame) {
        return NO;
    }
    return YES;
}

- (bool) configureUserFBOAttachmentsLocked
{
    Framebuffer *fbo;

    fbo = ctx->state.framebuffer;

    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    for (int i = 0; i < drawBufferCount; i++)
    {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        if (mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, (GLuint)i),
                                                  &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            (fbo->color_attachment_bitfield & (1u << attachmentIndex)) &&
            fbo->color_attachments[attachmentIndex].texture)
        {
            Texture *tex;

            tex = [self framebufferAttachmentTexture: &fbo->color_attachments[attachmentIndex]];
            if (!tex) {
                continue;
            }

            // Ensure attachment textures are created with RenderTarget usage.
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
            if (!tex->mtl_data) {
                continue;
            }

            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->color_attachments[attachmentIndex]);
            _renderPassDescriptor.colorAttachments[colorSlot].texture =
                mglApplySRGBStateToRenderTarget((__bridge id<MTLTexture> _Nullable)(tex->mtl_data), ctx);
            _renderPassDescriptor.colorAttachments[colorSlot].level = subresource.level;
            _renderPassDescriptor.colorAttachments[colorSlot].slice = subresource.slice;
            _renderPassDescriptor.colorAttachments[colorSlot].depthPlane = subresource.depthPlane;

            if (tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY ||
                tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
                tex->target == GL_TEXTURE_2D_ARRAY) {
                id<MTLTexture> rpTex = _renderPassDescriptor.colorAttachments[colorSlot].texture;
                (void)rpTex;
            }

            // Keep render pass dimensions aligned with attached color targets.
            // Some FBO paths use textures (not renderbuffers), and Metal still requires
            // scissor/viewport to be bounded by the attachment dimensions.
            NSUInteger attWidth = mglMetalTextureLevelDimension((NSUInteger)tex->width,
                                                                subresource.level);
            NSUInteger attHeight = mglMetalTextureLevelDimension((NSUInteger)tex->height,
                                                                 subresource.level);
            if (attWidth > 0 && attHeight > 0) {
                if (_renderPassDescriptor.renderTargetWidth == 0 || _renderPassDescriptor.renderTargetHeight == 0) {
                    _renderPassDescriptor.renderTargetWidth = attWidth;
                    _renderPassDescriptor.renderTargetHeight = attHeight;
                } else if (_renderPassDescriptor.renderTargetWidth != attWidth ||
                           _renderPassDescriptor.renderTargetHeight != attHeight) {
                    NSUInteger oldWidth = _renderPassDescriptor.renderTargetWidth;
                    NSUInteger oldHeight = _renderPassDescriptor.renderTargetHeight;
                    _renderPassDescriptor.renderTargetWidth = MIN(_renderPassDescriptor.renderTargetWidth, attWidth);
                    _renderPassDescriptor.renderTargetHeight = MIN(_renderPassDescriptor.renderTargetHeight, attHeight);
                    NSLog(@"MGL WARNING: FBO color attachment size mismatch slot=%d old=%lux%lu new=%lux%lu resolved=%lux%lu",
                          i,
                          (unsigned long)oldWidth,
                          (unsigned long)oldHeight,
                          (unsigned long)attWidth,
                          (unsigned long)attHeight,
                          (unsigned long)_renderPassDescriptor.renderTargetWidth,
                          (unsigned long)_renderPassDescriptor.renderTargetHeight);
                }
            }
        }
    }

    // depth attachment
    if (fbo->depth.texture)
    {
        Texture *tex;

        tex = [self framebufferAttachmentTexture: &fbo->depth];
        if (tex) {
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
        }
        if (tex && tex->mtl_data) {
            _renderPassDescriptor.depthAttachment.texture = (__bridge id<MTLTexture> _Nullable)(tex->mtl_data);
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
            _renderPassDescriptor.depthAttachment.level = subresource.level;
            _renderPassDescriptor.depthAttachment.slice = subresource.slice;
            _renderPassDescriptor.depthAttachment.depthPlane = subresource.depthPlane;
        }
    }

    // stencil attachment
    if (fbo->stencil.texture)
    {
        Texture *tex;

        tex = [self framebufferAttachmentTexture: &fbo->stencil];
        if (tex) {
            tex->is_render_target = true;
            RETURN_FALSE_ON_FAILURE([self bindMTLTextureLocked: tex]);
        }
        if (tex && tex->mtl_data) {
            _renderPassDescriptor.stencilAttachment.texture = (__bridge id<MTLTexture> _Nullable)(tex->mtl_data);
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
            _renderPassDescriptor.stencilAttachment.level = subresource.level;
            _renderPassDescriptor.stencilAttachment.slice = subresource.slice;
            _renderPassDescriptor.stencilAttachment.depthPlane = subresource.depthPlane;
        }
    }
    return true;
}

- (bool) configureDefaultFramebufferAttachmentsLocked
{
    GLuint mgl_drawbuffer;
    id<MTLTexture> texture = nil;
    id<MTLTexture> depth_texture = nil;
    id<MTLTexture> stencil_texture = nil;

    switch(ctx->state.draw_buffer)
    {
        case GL_FRONT: mgl_drawbuffer = _FRONT; break;
        case GL_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_FRONT_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_FRONT_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_BACK_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_BACK_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_FRONT_AND_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_COLOR_ATTACHMENT0: mgl_drawbuffer = _FRONT; break;
        case GL_NONE:
            // Handle GL_NONE gracefully - no draw buffer selected
            mgl_drawbuffer = _FRONT; // fallback to front
            DEBUG_PRINT("MGL: draw_buffer is GL_NONE, falling back to FRONT\n");
            break;
        default:
            DEBUG_PRINT("MGL: Unknown draw_buffer value: 0x%x, falling back to FRONT\n", ctx->state.draw_buffer);
            mgl_drawbuffer = _FRONT; // fallback to front instead of failing render setup
            NSLog(@"MGL WARNING: Unknown draw_buffer value 0x%x, using FRONT fallback", ctx->state.draw_buffer);
            break;
    }

    if(![self checkDrawBufferSize:mgl_drawbuffer])
    {
        _drawBuffers[mgl_drawbuffer].drawbuffer = NULL;
        _drawBuffers[mgl_drawbuffer].depthbuffer = NULL;
        _drawBuffers[mgl_drawbuffer].stencilbuffer = NULL;
        _drawBuffers[mgl_drawbuffer].width = 0;
        _drawBuffers[mgl_drawbuffer].height = 0;
    }

    // attach color buffer
    if (mgl_drawbuffer == _FRONT)
    {
        // SAFETY: Ensure we have a valid drawable with texture
        if (!_drawable) {
            NSLog(@"MGL ERROR: No drawable available for front buffer");
            return false;
        }

        texture = _drawable.texture;

        // sleep mode will return a null texture - handle gracefully without crashing
        if (!texture) {
            NSLog(@"MGL WARNING: Drawable texture is NULL (sleep mode or window not visible), attempting to get new drawable");

            // Try to get a new drawable
            _drawable = [_layer nextDrawable];
            if (_drawable) {
                texture = _drawable.texture;
                NSLog(@"MGL INFO: Successfully obtained new drawable with texture");
            } else {
                NSLog(@"MGL ERROR: Still no drawable texture available");
                return false;
            }
        }
    }
    else if(_drawBuffers[mgl_drawbuffer].drawbuffer)
    {
        texture = _drawBuffers[mgl_drawbuffer].drawbuffer;
    }
    else
    {
        texture = [self newDrawBuffer: ctx->pixel_format.mtl_pixel_format isDepthStencil:false];
        _drawBuffers[mgl_drawbuffer].drawbuffer = texture;
    }

    // attach depth. The default framebuffer must have a usable depth
    // attachment whenever GL depth testing is active, even if the legacy
    // context format fields were left unset by the window/bootstrap path.
    BOOL defaultPassNeedsDepth = ctx->state.caps.depth_test ||
                                 _drawBuffers[mgl_drawbuffer].depthbuffer != nil;
    if (defaultPassNeedsDepth)
    {
        MTLPixelFormat depthFormat = ctx->depth_format.mtl_pixel_format;
        if (depthFormat == MTLPixelFormatInvalid) {
            depthFormat = MTLPixelFormatDepth32Float;
        }

        if(_drawBuffers[mgl_drawbuffer].depthbuffer)
        {
            depth_texture = _drawBuffers[mgl_drawbuffer].depthbuffer;
        }
        else
        {
            depth_texture = [self newDrawBufferWithCustomSize:depthFormat isDepthStencil:true customSize: CGSizeMake(texture.width, texture.height) ];
            _drawBuffers[mgl_drawbuffer].depthbuffer = depth_texture;
            if (depth_texture) {
                static uint64_t s_defaultDepthCreateCount = 0;
                uint64_t hit = ++s_defaultDepthCreateCount;
                if (kMGLDiagnosticStateLogs && hit <= 8) {
                    MGLTraceNSLog(@"MGL DEFAULT FBO: created depth attachment fmt=%lu size=%lux%lu drawBuffer=%u",
                                  (unsigned long)depthFormat,
                                  (unsigned long)depth_texture.width,
                                  (unsigned long)depth_texture.height,
                                  mgl_drawbuffer);
                }
            }
        }
    }

    // attach stencil
    BOOL defaultPassNeedsStencil = ctx->state.caps.stencil_test ||
                                   ctx->stencil_format.format ||
                                   _drawBuffers[mgl_drawbuffer].stencilbuffer != nil;
    if (defaultPassNeedsStencil)
    {
        MTLPixelFormat stencilFormat = ctx->stencil_format.mtl_pixel_format;
        if (stencilFormat == MTLPixelFormatInvalid ||
            stencilFormat == MTLPixelFormatDepth32Float_Stencil8) {
            stencilFormat = MTLPixelFormatStencil8;
        }

        if(_drawBuffers[mgl_drawbuffer].stencilbuffer)
        {
            stencil_texture = _drawBuffers[mgl_drawbuffer].stencilbuffer;
        }
        else
        {
            stencil_texture = [self newDrawBufferWithCustomSize:stencilFormat isDepthStencil:true customSize: CGSizeMake(texture.width, texture.height) ];
            _drawBuffers[mgl_drawbuffer].stencilbuffer = stencil_texture;
        }
    }

    _renderPassDescriptor.colorAttachments[0].texture =
        mglApplySRGBStateToRenderTarget(texture, ctx);
    _renderPassDescriptor.depthAttachment.texture = depth_texture;
    _renderPassDescriptor.stencilAttachment.texture = stencil_texture;

    _renderPassDescriptor.renderTargetWidth = texture.width;
    _renderPassDescriptor.renderTargetHeight = texture.height;
    _drawBuffers[mgl_drawbuffer].width = (GLuint)texture.width;
    _drawBuffers[mgl_drawbuffer].height = (GLuint)texture.height;
    return true;
}

- (void) ensureTransientDepthForDefaultFramebufferLocked
{
    if (!ctx->state.framebuffer &&
        ctx->state.caps.depth_test &&
        !_renderPassDescriptor.depthAttachment.texture) {
        NSUInteger depthWidth = _renderPassDescriptor.renderTargetWidth;
        NSUInteger depthHeight = _renderPassDescriptor.renderTargetHeight;

        if (depthWidth == 0 || depthHeight == 0) {
            id<MTLTexture> color0 = _renderPassDescriptor.colorAttachments[0].texture;
            if (color0) {
                depthWidth = color0.width;
                depthHeight = color0.height;
            }
        }

        if (depthWidth > 0 && depthHeight > 0) {
            if (!_transientDepthTexture ||
                _transientDepthTextureWidth != depthWidth ||
                _transientDepthTextureHeight != depthHeight) {
                MTLTextureDescriptor *depthDesc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                       width:depthWidth
                                                                      height:depthHeight
                                                                   mipmapped:NO];
                depthDesc.usage = MTLTextureUsageRenderTarget;
                depthDesc.storageMode = MTLStorageModePrivate;
                _transientDepthTexture = [_device newTextureWithDescriptor:depthDesc];
                _transientDepthTextureWidth = depthWidth;
                _transientDepthTextureHeight = depthHeight;

                if (_transientDepthTexture) {
                    static uint64_t s_transientDepthCreateCount = 0;
                    uint64_t hit = ++s_transientDepthCreateCount;
                    if (hit <= 16 || (hit % 128) == 0) {
                        NSLog(@"MGL TRANSIENT FBO: created depth attachment fmt=%lu size=%lux%lu fbo=%u",
                              (unsigned long)MTLPixelFormatDepth32Float,
                              (unsigned long)depthWidth,
                              (unsigned long)depthHeight,
                              (unsigned)(mglRendererSafeFramebufferName(ctx)));
                    }
                } else {
                    NSLog(@"MGL ERROR: failed to create transient depth attachment size=%lux%lu fbo=%u",
                          (unsigned long)depthWidth,
                          (unsigned long)depthHeight,
                          (unsigned)(mglRendererSafeFramebufferName(ctx)));
                }
            }

            if (_transientDepthTexture) {
                _renderPassDescriptor.depthAttachment.texture = _transientDepthTexture;
                _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
                _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionDontCare;
                _renderPassDescriptor.depthAttachment.clearDepth = ctx->state.var.depth_clear_value;
            }
        }
    }
}

- (void) configureUserFBOLoadStoreActionsLocked:(GLuint *)outFboColorClearCount
                                  fboColorClearMask:(GLbitfield *)outFboColorClearMask
                     fboColorAttachment0ClearMask:(GLbitfield *)outFboColorAttachment0ClearMask
{
    Framebuffer *fbo = ctx->state.framebuffer;
    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    /* Stage 4.2: read the DontCare flag once per pass so the feature-off
     * (default) path skips both the per-attachment stamp write and the
     * shouldUse call entirely — avoiding per-attachment getenv and a
     * cache-line write on the common no-DontCare path. */
    BOOL dontCareLoadEnabled = mglEnvFlagEnabled("MGL_ENABLE_DONTCARE_LOAD");
    for (int i = 0; i < drawBufferCount; ++i) {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        if (!mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                   mglMetalDrawBufferAt(ctx, (GLuint)i),
                                                   &attachmentIndex) ||
            attachmentIndex >= MAX_COLOR_ATTACHMENTS ||
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            _renderPassDescriptor.colorAttachments[colorSlot].loadAction = MTLLoadActionLoad;
            continue;
        }

        FBOAttachment *att = &fbo->color_attachments[attachmentIndex];
        if (attachmentIndex == 0) {
            *outFboColorAttachment0ClearMask = att->clear_bitmask;
        }

        Texture *attachmentTextureForClear = [self framebufferAttachmentTexture:att];
        /* Stage 4.2: stamp this attachment's frame generation on EVERY
         * render-target use (clear/load/dontcare), capturing whether this
         * is its first use this frame BEFORE stamping. A clear-then-resume
         * within one frame must record the clear as a use so the resume is
         * not mistaken for a first use (which would wrongly DontCare and
         * discard the cleared+drawn content). */
        BOOL colorFirstUseThisFrame = NO;
        if (dontCareLoadEnabled && attachmentTextureForClear) {
            colorFirstUseThisFrame =
                (attachmentTextureForClear->mtl_rt_frame_generation != _dontCareFrameGeneration);
            attachmentTextureForClear->mtl_rt_frame_generation = _dontCareFrameGeneration;
        }
        if (att->clear_bitmask & GL_COLOR_BUFFER_BIT) {
            if (attachmentTextureForClear &&
                attachmentTextureForClear->name == 8u &&
                mglTraceLogIsEnabled()) {
                mglTraceLog("PENDING_COLOR_CLEAR_CONSUME tex=%u fbo=%u attachment=%u slot=%d program=%u clearMask=0x%x rgba=(%.3f,%.3f,%.3f,%.3f) drawBuf=0x%x readBuf=0x%x scissor(test=%d box=%d,%d,%d,%d) colorMask=%d%d%d%d depth(test=%d write=%d)",
                            (unsigned)attachmentTextureForClear->name,
                            (unsigned)fbo->name,
                            (unsigned)attachmentIndex,
                            i,
                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                            (unsigned)att->clear_bitmask,
                            att->clear_color[0],
                            att->clear_color[1],
                            att->clear_color[2],
                            att->clear_color[3],
                            (unsigned)(ctx ? ctx->state.draw_buffer : 0u),
                            (unsigned)(ctx ? ctx->state.read_buffer : 0u),
                            (ctx && ctx->state.caps.scissor_test) ? 1 : 0,
                            (int)(ctx ? ctx->state.var.scissor_box[0] : 0),
                            (int)(ctx ? ctx->state.var.scissor_box[1] : 0),
                            (int)(ctx ? ctx->state.var.scissor_box[2] : 0),
                            (int)(ctx ? ctx->state.var.scissor_box[3] : 0),
                            (ctx && ctx->state.var.color_writemask[0][0]) ? 1 : 0,
                            (ctx && ctx->state.var.color_writemask[0][1]) ? 1 : 0,
                            (ctx && ctx->state.var.color_writemask[0][2]) ? 1 : 0,
                            (ctx && ctx->state.var.color_writemask[0][3]) ? 1 : 0,
                            (ctx && ctx->state.caps.depth_test) ? 1 : 0,
                            (ctx && ctx->state.var.depth_writemask) ? 1 : 0);
            }
            _renderPassDescriptor.colorAttachments[colorSlot].clearColor =
                MTLClearColorMake(att->clear_color[0],
                                  att->clear_color[1],
                                  att->clear_color[2],
                                  att->clear_color[3]);
            _renderPassDescriptor.colorAttachments[colorSlot].loadAction = MTLLoadActionClear;
            _renderPassDescriptor.colorAttachments[colorSlot].storeAction = MTLStoreActionStore;

            att->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
            mglMarkTextureLevelRenderTargetWritten(attachmentTextureForClear, att->level);

            (*outFboColorClearCount)++;
            *outFboColorClearMask |= (GLbitfield)(1u << attachmentIndex);
        } else if (dontCareLoadEnabled &&
                   [self shouldUseDontCareLoadForColorTexture:attachmentTextureForClear
                                                firstUseThisFrame:colorFirstUseThisFrame]) {
            /* Stage 4.2: first render-target use this frame, no clear, no
             * blend — prior tile contents are dead, skip the load. */
            _renderPassDescriptor.colorAttachments[colorSlot].loadAction = MTLLoadActionDontCare;
        } else {
            _renderPassDescriptor.colorAttachments[colorSlot].loadAction = MTLLoadActionLoad;
        }
    }

    /* Consume any pending color clears on attachments that could not be
     * resolved through a draw buffer — prevents infinite retry when an
     * attachment's clear_bitmask is set but mglMetalResolveFboDrawAttachmentIndex
     * fails for its draw buffer. */
    for (GLuint ai = 0; ai < MAX_COLOR_ATTACHMENTS; ++ai) {
        if ((fbo->color_attachments[ai].clear_bitmask & GL_COLOR_BUFFER_BIT) &&
            ((fbo->color_attachment_bitfield >> ai) & 1u) == 0u) {
            fbo->color_attachments[ai].clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        }
    }

    if (fbo->depth.clear_bitmask & GL_DEPTH_BUFFER_BIT) {
        _renderPassDescriptor.depthAttachment.clearDepth = fbo->depth.clear_color[0];
        _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
        _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
        fbo->depth.clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionLoad;
        if (_renderPassDescriptor.depthAttachment.texture) {
            _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
        }
    }

    if (fbo->stencil.clear_bitmask & GL_STENCIL_BUFFER_BIT) {
        _renderPassDescriptor.stencilAttachment.clearStencil = (uint32_t)fbo->stencil.clear_color[0];
        _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionClear;
        _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
        fbo->stencil.clear_bitmask &= ~GL_STENCIL_BUFFER_BIT;
    } else {
        _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
        if (_renderPassDescriptor.stencilAttachment.texture) {
            _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    }
}

- (void) configureDefaultFramebufferLoadStoreActionsLocked
{
    Framebuffer *fbo = ctx->state.framebuffer;
    GLbitfield defaultClearMask = ctx->state.default_fbo_clear_bitmask;
    if (defaultClearMask & GL_COLOR_BUFFER_BIT) {
        _renderPassDescriptor.colorAttachments[0].clearColor =
            MTLClearColorMake(ctx->state.default_clear_color[0],
                              ctx->state.default_clear_color[1],
                              ctx->state.default_clear_color[2],
                              ctx->state.default_clear_color[3]);
        _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        _renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
        ctx->state.default_fbo_clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    } else {
        _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
        static uint64_t s_defaultFboLoadLogCount = 0;
        uint64_t hit = ++s_defaultFboLoadLogCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            MGLTraceNSLog(@"MGL DEFAULT FBO: using Load (no clear mask) call=%llu drawBuf=0x%x fbo=%u",
                          (unsigned long long)hit,
                          ctx->state.draw_buffer,
                          fbo ? (unsigned)fbo->name : 0u);
        }
    }

    if (defaultClearMask & GL_DEPTH_BUFFER_BIT) {
        _renderPassDescriptor.depthAttachment.clearDepth = STATE_VAR(depth_clear_value);
        _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
        _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
        ctx->state.default_fbo_clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionLoad;
        if (_renderPassDescriptor.depthAttachment.texture) {
            _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
        }
    }

    if (defaultClearMask & GL_STENCIL_BUFFER_BIT) {
        _renderPassDescriptor.stencilAttachment.clearStencil = STATE_VAR(stencil_clear_value);
        _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionClear;
        _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
        ctx->state.default_fbo_clear_bitmask &= ~GL_STENCIL_BUFFER_BIT;
    } else {
        _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
        if (_renderPassDescriptor.stencilAttachment.texture) {
            _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    }
}

- (void) logRenderPassClearResolveLocked:(uint64_t)renderEncoderCall
                      traceRenderEncoder:(bool)traceRenderEncoder
                        fboColorClearCount:(GLuint)fboColorClearCount
                         fboColorClearMask:(GLbitfield)fboColorClearMask
            fboColorAttachment0ClearMask:(GLbitfield)fboColorAttachment0ClearMask
                 fboDepthClearMaskBefore:(GLbitfield)fboDepthClearMaskBefore
               fboStencilClearMaskBefore:(GLbitfield)fboStencilClearMaskBefore
                             defaultClearMask:(GLbitfield)defaultClearMask
                                         fbo:(Framebuffer *)fbo
{
	    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
	        MTLClearColor c0 = _renderPassDescriptor.colorAttachments[0].clearColor;
	        MGLTraceNSLog(@"MGL TRACE clear.resolve call=%llu fbo=%u "
	              "fboColorClears=%u fboColorMask=0x%x fboAtt0ClearMask=0x%x c0LA=%s depthLA=%s stencilLA=%s "
	              "c0Clear=(%.3f,%.3f,%.3f,%.3f) depthClear=%.3f stencilClear=%u",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)fboColorClearCount,
              (unsigned)fboColorClearMask,
              (unsigned)fboColorAttachment0ClearMask,
              mglLoadActionName(_renderPassDescriptor.colorAttachments[0].loadAction),
              mglLoadActionName(_renderPassDescriptor.depthAttachment.loadAction),
              mglLoadActionName(_renderPassDescriptor.stencilAttachment.loadAction),
              c0.red,
              c0.green,
              c0.blue,
              c0.alpha,
	              _renderPassDescriptor.depthAttachment.clearDepth,
	              (unsigned)_renderPassDescriptor.stencilAttachment.clearStencil);
	    }

            BOOL clearResolveInteresting =
                (fboColorClearCount != 0) ||
                (fboColorAttachment0ClearMask != 0) ||
                (_renderPassDescriptor.depthAttachment.loadAction == MTLLoadActionClear) ||
                (fboDepthClearMaskBefore & GL_DEPTH_BUFFER_BIT) ||
                (!fbo && (defaultClearMask & GL_DEPTH_BUFFER_BIT));
            if (clearResolveInteresting) {
                static uint64_t s_clearResolveDetailLogCount = 0;
                uint64_t hit = ++s_clearResolveDetailLogCount;
                if (mglTraceLogIsEnabled() && (hit <= 256ull || (hit % 512ull) == 0ull)) {
	            MTLClearColor c0 = _renderPassDescriptor.colorAttachments[0].clearColor;
	            id<MTLTexture> c0Tex = _renderPassDescriptor.colorAttachments[0].texture;
	            id<MTLTexture> dTex = _renderPassDescriptor.depthAttachment.texture;
	            id<MTLTexture> sTex = _renderPassDescriptor.stencilAttachment.texture;
		            mglTraceLog("RENDERPASS_CLEAR call=%llu hit=%llu fbo=%u drawBuf=0x%x readBuf=0x%x "
	                        "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                        "fboColorClears=%u fboColorMask=0x%x fboAtt0Mask=0x%x pending(global=0x%x default=0x%x depth=0x%x stencil=0x%x) "
	                        "c0LA=%s c0SA=%s depthLA=%s depthSA=%s stencilLA=%s stencilSA=%s "
	                        "c0Tex=%p fmt=%lu size=%lux%lu depthTex=%p fmt=%lu size=%lux%lu stencilTex=%p "
	                        "clearRGBA=(%.6f,%.6f,%.6f,%.6f) depthClear=%.6f stencilClear=%u depthState(test=%d write=%d func=0x%x)",
	                        (unsigned long long)renderEncoderCall,
	                        (unsigned long long)hit,
	                        (unsigned)(mglRendererSafeFramebufferName(ctx)),
	                        (unsigned)ctx->state.draw_buffer,
	                        (unsigned)ctx->state.read_buffer,
	                        (int)ctx->state.viewport[0],
	                        (int)ctx->state.viewport[1],
	                        (int)ctx->state.viewport[2],
	                        (int)ctx->state.viewport[3],
	                        ctx->state.caps.scissor_test ? 1 : 0,
	                        (int)ctx->state.var.scissor_box[0],
	                        (int)ctx->state.var.scissor_box[1],
	                        (int)ctx->state.var.scissor_box[2],
	                        (int)ctx->state.var.scissor_box[3],
	                        (unsigned)fboColorClearCount,
	                        (unsigned)fboColorClearMask,
	                        (unsigned)fboColorAttachment0ClearMask,
	                        (unsigned)ctx->state.clear_bitmask,
	                        (unsigned)ctx->state.default_fbo_clear_bitmask,
	                        (unsigned)fboDepthClearMaskBefore,
	                        (unsigned)fboStencilClearMaskBefore,
	                        mglLoadActionName(_renderPassDescriptor.colorAttachments[0].loadAction),
	                        mglStoreActionName(_renderPassDescriptor.colorAttachments[0].storeAction),
	                        mglLoadActionName(_renderPassDescriptor.depthAttachment.loadAction),
	                        mglStoreActionName(_renderPassDescriptor.depthAttachment.storeAction),
	                        mglLoadActionName(_renderPassDescriptor.stencilAttachment.loadAction),
	                        mglStoreActionName(_renderPassDescriptor.stencilAttachment.storeAction),
	                        c0Tex,
	                        (unsigned long)(c0Tex ? c0Tex.pixelFormat : MTLPixelFormatInvalid),
	                        (unsigned long)(c0Tex ? c0Tex.width : 0),
	                        (unsigned long)(c0Tex ? c0Tex.height : 0),
	                        dTex,
	                        (unsigned long)(dTex ? dTex.pixelFormat : MTLPixelFormatInvalid),
	                        (unsigned long)(dTex ? dTex.width : 0),
	                        (unsigned long)(dTex ? dTex.height : 0),
	                        sTex,
	                        c0.red,
	                        c0.green,
	                        c0.blue,
	                        c0.alpha,
	                        _renderPassDescriptor.depthAttachment.clearDepth,
	                        (unsigned)_renderPassDescriptor.stencilAttachment.clearStencil,
	                        ctx->state.caps.depth_test ? 1 : 0,
	                        ctx->state.var.depth_writemask ? 1 : 0,
	                        (unsigned)ctx->state.var.depth_func);
	        }
	    }
}

- (bool) finalizeRenderPassDescriptorLocked:(uint64_t)renderEncoderCall
                          traceRenderEncoder:(bool)traceRenderEncoder
{
	    _renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;

    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
        id<MTLTexture> c0Tex = _renderPassDescriptor.colorAttachments[0].texture;
        id<MTLTexture> dTex = _renderPassDescriptor.depthAttachment.texture;
        id<MTLTexture> sTex = _renderPassDescriptor.stencilAttachment.texture;
        MGLTraceNSLog(@"MGL TRACE renderpass.attach call=%llu fbo=%u drawBuf=0x%x rt=%lux%lu "
              "c0=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s stencil=%p fmt=%lu size=%lux%lu la/sa=%s/%s",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)ctx->state.draw_buffer,
              (unsigned long)_renderPassDescriptor.renderTargetWidth,
              (unsigned long)_renderPassDescriptor.renderTargetHeight,
              c0Tex,
              (unsigned long)(c0Tex ? c0Tex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(c0Tex ? c0Tex.usage : 0),
              (unsigned long)(c0Tex ? c0Tex.width : 0),
              (unsigned long)(c0Tex ? c0Tex.height : 0),
              mglLoadActionName(_renderPassDescriptor.colorAttachments[0].loadAction),
              mglStoreActionName(_renderPassDescriptor.colorAttachments[0].storeAction),
              dTex,
              (unsigned long)(dTex ? dTex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(dTex ? dTex.width : 0),
              (unsigned long)(dTex ? dTex.height : 0),
              mglLoadActionName(_renderPassDescriptor.depthAttachment.loadAction),
              mglStoreActionName(_renderPassDescriptor.depthAttachment.storeAction),
              sTex,
              (unsigned long)(sTex ? sTex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(sTex ? sTex.width : 0),
              (unsigned long)(sTex ? sTex.height : 0),
              mglLoadActionName(_renderPassDescriptor.stencilAttachment.loadAction),
              mglStoreActionName(_renderPassDescriptor.stencilAttachment.storeAction));
    }

    // create a render encoder from the renderpass descriptor
    // CRITICAL SAFETY: Validate inputs before creating render encoder
    if (!_renderPassDescriptor) {
        NSLog(@"MGL ERROR: Cannot create render encoder - render pass descriptor is NULL");
        [self recordGPUError];
        return false;
    }

    // Metal debug layer crashes if render pass has no output attachment.
    // Provide a tiny fallback color attachment for targetless/invalid passes.
    bool hasOutputAttachment = false;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (_renderPassDescriptor.colorAttachments[i].texture) {
            hasOutputAttachment = true;
            break;
        }
    }
    if (!hasOutputAttachment &&
        (_renderPassDescriptor.depthAttachment.texture || _renderPassDescriptor.stencilAttachment.texture)) {
        hasOutputAttachment = true;
    }

    if (!hasOutputAttachment) {
        if (!_fallbackRenderTargetTexture) {
            MTLTextureDescriptor *fbDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                   width:1
                                                                  height:1
                                                               mipmapped:NO];
            fbDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            fbDesc.storageMode = MTLStorageModeShared;
            _fallbackRenderTargetTexture = [_device newTextureWithDescriptor:fbDesc];
        }

        if (_fallbackRenderTargetTexture) {
            NSLog(@"MGL WARNING: Render pass had no attachments; binding 1x1 fallback color target");
            _renderPassDescriptor.colorAttachments[0].texture = _fallbackRenderTargetTexture;
            _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
            _renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
            _renderPassDescriptor.renderTargetWidth = 1;
            _renderPassDescriptor.renderTargetHeight = 1;
        } else {
            NSLog(@"MGL ERROR: Failed to allocate fallback render target texture");
            [self recordGPUError];
            return false;
        }
    }

    // Final guard: Metal will assert if a color attachment texture is missing RenderTarget usage.
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id<MTLTexture> attTex = _renderPassDescriptor.colorAttachments[i].texture;
        if (attTex && ((attTex.usage & MTLTextureUsageRenderTarget) == 0)) {
            NSLog(@"MGL WARNING: colorAttachment[%d] usage=0x%lx lacks RenderTarget; clearing attachment to avoid Metal assert",
                  i, (unsigned long)attTex.usage);
            _renderPassDescriptor.colorAttachments[i].texture = nil;
        }
    }

    // Default-framebuffer paths expect color attachment 0 specifically.
    // FBO draw-buffer mappings may intentionally leave slot 0 as GL_NONE.
    if (!ctx->state.framebuffer && !_renderPassDescriptor.colorAttachments[0].texture) {
        for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (_renderPassDescriptor.colorAttachments[i].texture) {
                NSLog(@"MGL WARNING: colorAttachment[0] missing; remapping colorAttachment[%d] -> [0]", i);
                _renderPassDescriptor.colorAttachments[0].texture = _renderPassDescriptor.colorAttachments[i].texture;
                _renderPassDescriptor.colorAttachments[0].loadAction = _renderPassDescriptor.colorAttachments[i].loadAction;
                _renderPassDescriptor.colorAttachments[0].storeAction = _renderPassDescriptor.colorAttachments[i].storeAction;
                break;
            }
        }
    }

    // Ultimate slot-0 fallback to keep draw path alive and avoid black frame.
    if (!hasOutputAttachment && !_renderPassDescriptor.colorAttachments[0].texture) {
        if (!_fallbackRenderTargetTexture) {
            MTLTextureDescriptor *fbDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                   width:1
                                                                  height:1
                                                               mipmapped:NO];
            fbDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            fbDesc.storageMode = MTLStorageModeShared;
            _fallbackRenderTargetTexture = [_device newTextureWithDescriptor:fbDesc];
        }
        if (_fallbackRenderTargetTexture) {
            NSLog(@"MGL WARNING: colorAttachment[0] unavailable; binding 1x1 fallback");
            _renderPassDescriptor.colorAttachments[0].texture = _fallbackRenderTargetTexture;
            _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
            _renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
            if (_renderPassDescriptor.renderTargetWidth == 0 || _renderPassDescriptor.renderTargetHeight == 0) {
                _renderPassDescriptor.renderTargetWidth = 1;
                _renderPassDescriptor.renderTargetHeight = 1;
            }
        } else {
            NSLog(@"MGL ERROR: Unable to allocate fallback colorAttachment[0] texture");
            [self recordGPUError];
            return false;
        }
    }

    // Ensure renderTargetWidth/Height are always coherent with the active attachments.
    {
        id<MTLTexture> sizeTex = _renderPassDescriptor.colorAttachments[0].texture;
        if (!sizeTex) {
            for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
                if (_renderPassDescriptor.colorAttachments[i].texture) {
                    sizeTex = _renderPassDescriptor.colorAttachments[i].texture;
                    break;
                }
            }
        }
        if (!sizeTex) {
            sizeTex = _renderPassDescriptor.depthAttachment.texture;
        }
        if (!sizeTex) {
            sizeTex = _renderPassDescriptor.stencilAttachment.texture;
        }

        if (sizeTex) {
            NSUInteger texWidth = sizeTex.width;
            NSUInteger texHeight = sizeTex.height;
            if (_renderPassDescriptor.renderTargetWidth == 0 ||
                _renderPassDescriptor.renderTargetHeight == 0 ||
                _renderPassDescriptor.renderTargetWidth > texWidth ||
                _renderPassDescriptor.renderTargetHeight > texHeight) {
                NSLog(@"MGL INFO: Normalizing renderTarget size from %lux%lu to %lux%lu",
                      (unsigned long)_renderPassDescriptor.renderTargetWidth,
                      (unsigned long)_renderPassDescriptor.renderTargetHeight,
                      (unsigned long)texWidth,
                      (unsigned long)texHeight);
                _renderPassDescriptor.renderTargetWidth = texWidth;
                _renderPassDescriptor.renderTargetHeight = texHeight;
            }
        }
    }
    return true;
}

- (bool) createRenderEncoderLocked:(uint64_t)renderEncoderCall
{
    // CRITICAL FIX: Validate command buffer state before creating render encoder
    if (!_currentCommandBuffer) {
        NSLog(@"MGL ERROR: Cannot create render encoder - command buffer is NULL");
        [self recordGPUError];
        return false;
    }

    // Check if command buffer already has an active encoder (Metal API violation)
    if (_currentRenderEncoder) {
        NSLog(@"MGL WARNING: Active render encoder detected - ending it before creating new one");
        @try {
            [_currentRenderEncoder endEncoding];
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: Exception ending existing encoder: %@", exception);
        }
        _currentRenderEncoder = nil;
    }

    // Validate command buffer status. If already committed/completed, rotate to a new buffer.
    MTLCommandBufferStatus bufferStatus = _currentCommandBuffer.status;
    if (bufferStatus >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL WARNING: Render encoder requested on finalized command buffer (status: %ld) - creating a fresh command buffer", (long)bufferStatus);
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer before creating render encoder");
            [self recordGPUError];
            return false;
        }

        if (!_currentCommandBuffer) {
            NSLog(@"MGL ERROR: newCommandBuffer returned but _currentCommandBuffer is NULL");
            [self recordGPUError];
            return false;
        }

        bufferStatus = _currentCommandBuffer.status;
        if (bufferStatus >= MTLCommandBufferStatusCommitted) {
            NSLog(@"MGL ERROR: Fresh command buffer is still finalized (status: %ld)", (long)bufferStatus);
            [self recordGPUError];
            return false;
        }
    }

    if (kMGLVerboseFrameLoopLogs) {
        NSLog(@"MGL DEBUG: About to create render encoder with descriptor and command buffer");
    }
	    {
	        static uint64_t s_renderPassPreCreateLogCount = 0;
	        uint64_t hit = ++s_renderPassPreCreateLogCount;
		        if (mglTraceLogIsEnabled() && (hit <= 128ull || (hit % 512ull) == 0ull)) {
	            mglLogRenderPassLifecycle("pre-create",
	                                      hit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
	                                      _renderPassFramebufferName,
	                                      _renderPassDrawBuffer,
	                                      _renderPassDrawBufferCount);
	            if (mglTraceLogIsEnabled()) {
	                id<MTLTexture> c0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
	                id<MTLTexture> depth = _renderPassDescriptor ? _renderPassDescriptor.depthAttachment.texture : nil;
	                mglTraceLog("RENDERPASS_PRE_CREATE hit=%llu call=%llu program=%u fbo=%u drawBuf=0x%x readBuf=0x%x "
	                            "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                            "c0=%p fmt=%lu size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s clearDepth=%.6f "
	                            "depthState(test=%d write=%d func=0x%x) pending(default=0x%x depth=0x%x)",
	                            (unsigned long long)hit,
	                            (unsigned long long)renderEncoderCall,
	                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
	                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
	                            (unsigned)(ctx ? ctx->state.draw_buffer : 0u),
	                            (unsigned)(ctx ? ctx->state.read_buffer : 0u),
	                            (int)(ctx ? ctx->state.viewport[0] : 0),
	                            (int)(ctx ? ctx->state.viewport[1] : 0),
	                            (int)(ctx ? ctx->state.viewport[2] : 0),
	                            (int)(ctx ? ctx->state.viewport[3] : 0),
	                            (ctx && ctx->state.caps.scissor_test) ? 1 : 0,
	                            (int)(ctx ? ctx->state.var.scissor_box[0] : 0),
	                            (int)(ctx ? ctx->state.var.scissor_box[1] : 0),
	                            (int)(ctx ? ctx->state.var.scissor_box[2] : 0),
	                            (int)(ctx ? ctx->state.var.scissor_box[3] : 0),
	                            c0,
	                            (unsigned long)(c0 ? c0.pixelFormat : MTLPixelFormatInvalid),
	                            (unsigned long)(c0 ? c0.width : 0),
	                            (unsigned long)(c0 ? c0.height : 0),
	                            mglLoadActionName(_renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare),
	                            mglStoreActionName(_renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare),
	                            depth,
	                            (unsigned long)(depth ? depth.pixelFormat : MTLPixelFormatInvalid),
	                            (unsigned long)(depth ? depth.width : 0),
	                            (unsigned long)(depth ? depth.height : 0),
	                            mglLoadActionName(_renderPassDescriptor ? _renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare),
	                            mglStoreActionName(_renderPassDescriptor ? _renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare),
	                            _renderPassDescriptor ? _renderPassDescriptor.depthAttachment.clearDepth : 0.0,
	                            (ctx && ctx->state.caps.depth_test) ? 1 : 0,
	                            (ctx && ctx->state.var.depth_writemask) ? 1 : 0,
	                            (unsigned)(ctx ? ctx->state.var.depth_func : 0u),
	                            (unsigned)(ctx ? ctx->state.default_fbo_clear_bitmask : 0u),
	                            (unsigned)(ctx && ctx->state.framebuffer ? ctx->state.framebuffer->depth.clear_bitmask : 0u));
	            }
	        }
	    }
        /* When a GL sample query (GL_SAMPLES_PASSED / GL_ANY_SAMPLES_PASSED)
         * is active, attach the visibility result buffer to the render pass
         * descriptor and zero it so the GPU accumulates a fresh count. */
        if (_sampleQueryActive && _renderPassDescriptor && _visibilityResultBuffer) {
            _renderPassDescriptor.visibilityResultBuffer = _visibilityResultBuffer;
            memset(_visibilityResultBuffer.contents, 0, _visibilityResultBuffer.length);
        }
        @try {
            _currentRenderEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor: _renderPassDescriptor];
            if (!_currentRenderEncoder) {
            NSLog(@"MGL ERROR: Failed to create render encoder - invalid render pass descriptor or command buffer");
            NSLog(@"MGL DEBUG: Command buffer: %@, Render pass descriptor: %@", _currentCommandBuffer, _renderPassDescriptor);
            [self recordGPUError];
            return false;
        }
        /* Enable visibility result mode on the encoder for all draws in this
         * pass when a sample query is active. MTLVisibilityResultModeBoolean
         * writes 1 to the buffer if any samples pass per-fragment tests. */
        if (_sampleQueryActive) {
            [_currentRenderEncoder setVisibilityResultMode:MTLVisibilityResultModeBoolean offset:0];
        }
        _renderPassFramebuffer = ctx ? ctx->state.framebuffer : NULL;
        mglTraceFragmentTextureTraceBindings("CLEAR",
                                             "new_render_encoder",
                                             _fragmentTextureTraceBindings,
                                             TEXTURE_UNITS,
                                             ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                             _pipelineProgramName);
        memset(_fragmentTextureTraceBindings, 0, sizeof(_fragmentTextureTraceBindings));
        _renderPassFramebufferName = _renderPassFramebuffer ? _renderPassFramebuffer->name : 0u;
        _renderPassDrawBuffer = ctx ? ctx->state.draw_buffer : 0u;
        _renderPassDrawBufferCount = ctx ? mglMetalDrawBufferCount(ctx) : 0;
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            _renderPassDrawBuffers[i] = (ctx && i < _renderPassDrawBufferCount)
                ? mglMetalDrawBufferAt(ctx, (GLuint)i)
                : GL_NONE;
        }
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Successfully created Metal render encoder");
        }
        [self recordGPUSuccess];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating render encoder: %@ - continuing with degraded functionality", exception);
        NSLog(@"MGL DEBUG: Exception details - name: %@, reason: %@", exception.name, exception.reason);
        [self recordGPUError];
        _currentRenderEncoder = NULL;
        return false;
    }
    _currentRenderEncoder.label = @"GL Render Encoder";
	    {
	        static uint64_t s_renderPassCreatedLogCount = 0;
	        uint64_t hit = ++s_renderPassCreatedLogCount;
		        if (mglTraceLogIsEnabled() && (hit <= 128ull || (hit % 512ull) == 0ull)) {
	            mglLogRenderPassLifecycle("created",
	                                      hit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
	                                      _renderPassFramebufferName,
	                                      _renderPassDrawBuffer,
	                                      _renderPassDrawBufferCount);
	            if (mglTraceLogIsEnabled()) {
	                id<MTLTexture> c0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
	                id<MTLTexture> depth = _renderPassDescriptor ? _renderPassDescriptor.depthAttachment.texture : nil;
	                mglTraceLog("RENDERPASS_CREATED hit=%llu call=%llu program=%u fbo=%u rpFbo=%u drawBuf=0x%x readBuf=0x%x "
	                            "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
	                            "c0=%p fmt=%lu size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s clearDepth=%.6f "
	                            "depthState(test=%d write=%d func=0x%x)",
	                            (unsigned long long)hit,
	                            (unsigned long long)renderEncoderCall,
	                            (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
	                            (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
	                            (unsigned)_renderPassFramebufferName,
	                            (unsigned)(ctx ? ctx->state.draw_buffer : 0u),
	                            (unsigned)(ctx ? ctx->state.read_buffer : 0u),
	                            (int)(ctx ? ctx->state.viewport[0] : 0),
	                            (int)(ctx ? ctx->state.viewport[1] : 0),
	                            (int)(ctx ? ctx->state.viewport[2] : 0),
	                            (int)(ctx ? ctx->state.viewport[3] : 0),
	                            (ctx && ctx->state.caps.scissor_test) ? 1 : 0,
	                            (int)(ctx ? ctx->state.var.scissor_box[0] : 0),
	                            (int)(ctx ? ctx->state.var.scissor_box[1] : 0),
	                            (int)(ctx ? ctx->state.var.scissor_box[2] : 0),
	                            (int)(ctx ? ctx->state.var.scissor_box[3] : 0),
	                            c0,
	                            (unsigned long)(c0 ? c0.pixelFormat : MTLPixelFormatInvalid),
	                            (unsigned long)(c0 ? c0.width : 0),
	                            (unsigned long)(c0 ? c0.height : 0),
	                            mglLoadActionName(_renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare),
	                            mglStoreActionName(_renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare),
	                            depth,
	                            (unsigned long)(depth ? depth.pixelFormat : MTLPixelFormatInvalid),
	                            (unsigned long)(depth ? depth.width : 0),
	                            (unsigned long)(depth ? depth.height : 0),
	                            mglLoadActionName(_renderPassDescriptor ? _renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare),
	                            mglStoreActionName(_renderPassDescriptor ? _renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare),
	                            _renderPassDescriptor ? _renderPassDescriptor.depthAttachment.clearDepth : 0.0,
	                            (ctx && ctx->state.caps.depth_test) ? 1 : 0,
	                            (ctx && ctx->state.var.depth_writemask) ? 1 : 0,
	                            (unsigned)(ctx ? ctx->state.var.depth_func : 0u));
	            }
	        }
	    }
    return true;
}

- (bool) newRenderEncoderLocked
{
    /* Stage 4 instrumentation: count every render encoder (re)creation. */
    MGL_PERF_INC(g_mglEncoderCreationsSinceSwap);
    // I can't remember why this is here...
    @autoreleasepool {
    /* Invalidate last-bound render encoder state — the new encoder must
     * re-issue all binds rather than skipping them via the dedup fast path.
     * Called here (in addition to endRenderEncodingLocked) so the cache is
     * also cleared on code paths that bypass endRenderEncodingLocked. */
    [self invalidateLastBoundState];

    static uint64_t s_newRenderEncoderCallCount = 0;
    uint64_t renderEncoderCall = ++s_newRenderEncoderCallCount;
    bool traceRenderEncoder = mglShouldTraceCall(renderEncoderCall) ||
                              (kMGLDiagnosticStateLogs && ((renderEncoderCall % 60ull) == 0ull));

    // AGX ERROR THROTTLING: Check if we should skip render encoder creation
    // BUT allow limited render encoder creation for essential functionality
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: Render encoder creation requested during GPU recovery - attempting essential creation");
        // Continue with essential render encoder creation even during recovery
    }

    // CRITICAL SAFETY: Check command buffer before creating render encoder
    if (!_currentCommandBuffer) {
        // Attempt recovery: create a new command buffer instead of failing immediately
        if ([self newCommandBufferLocked]) {
            // Successfully created - continue
        } else {
            NSLog(@"MGL ERROR: Cannot create render encoder - no command buffer available");
            [self recordGPUError];
            return false;
        }
    }

    // end encoding on current render encoder
    [self endRenderEncodingLocked];

    // grab the next drawable from CAMetalLayer
    if (_drawable == NULL)
    {
        if (!_layer) {
            NSLog(@"MGL ERROR: Cannot get drawable - no CAMetalLayer available");
            return false;
        }

        CGSize expectedDrawableSize = [self mglSyncLayerDrawableSizeFromView:"newRenderEncoder.nextDrawable"];
        _drawable = [_layer nextDrawable];

        // late init of gl scissor box on attachment to window system
        NSUInteger drawableWidth = (NSUInteger)MAX(1.0, expectedDrawableSize.width);
        NSUInteger drawableHeight = (NSUInteger)MAX(1.0, expectedDrawableSize.height);
        if (_drawable && _drawable.texture) {
            drawableWidth = (NSUInteger)_drawable.texture.width;
            drawableHeight = (NSUInteger)_drawable.texture.height;
        }

        if (!ctx->state.caps.scissor_test) {
            ctx->state.var.scissor_box[0] = 0;
            ctx->state.var.scissor_box[1] = 0;
        }
        ctx->state.var.scissor_box[2] = (GLint)drawableWidth;
        ctx->state.var.scissor_box[3] = (GLint)drawableHeight;
    }

    _renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
    if (!_renderPassDescriptor) {
        NSLog(@"MGL RENDERPASS ERROR: failed to allocate render pass descriptor");
        return false;
    }


    // Configure color/depth/stencil attachments based on FBO type
    if (ctx->state.framebuffer) {
        RETURN_FALSE_ON_FAILURE([self configureUserFBOAttachmentsLocked]);
    } else {
        RETURN_FALSE_ON_FAILURE([self configureDefaultFramebufferAttachmentsLocked]);
    }
    [self ensureTransientDepthForDefaultFramebufferLocked];

    // Capture clear state before load/store resolution for diagnostic logging
    GLuint fboColorClearCount = 0;
    GLbitfield fboColorClearMask = 0;
    GLbitfield fboColorAttachment0ClearMask = 0;

    Framebuffer *fbo = ctx->state.framebuffer;
    GLbitfield defaultClearMask = ctx->state.default_fbo_clear_bitmask;
    GLbitfield fboDepthClearMaskBefore = fbo ? fbo->depth.clear_bitmask : 0u;
    GLbitfield fboStencilClearMaskBefore = fbo ? fbo->stencil.clear_bitmask : 0u;

    if (fbo) {
        [self configureUserFBOLoadStoreActionsLocked:&fboColorClearCount
                                  fboColorClearMask:&fboColorClearMask
                     fboColorAttachment0ClearMask:&fboColorAttachment0ClearMask];
    } else {
        [self configureDefaultFramebufferLoadStoreActionsLocked];
    }

    [self logRenderPassClearResolveLocked:renderEncoderCall
                      traceRenderEncoder:traceRenderEncoder
                        fboColorClearCount:fboColorClearCount
                         fboColorClearMask:fboColorClearMask
            fboColorAttachment0ClearMask:fboColorAttachment0ClearMask
                 fboDepthClearMaskBefore:fboDepthClearMaskBefore
               fboStencilClearMaskBefore:fboStencilClearMaskBefore
                             defaultClearMask:defaultClearMask
                                         fbo:fbo];

    RETURN_FALSE_ON_FAILURE([self finalizeRenderPassDescriptorLocked:renderEncoderCall
                                                  traceRenderEncoder:traceRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self createRenderEncoderLocked:renderEncoderCall]);

    // apply all state that isn't included in a renderPassDescriptor into the render encoder
    [self updateCurrentRenderEncoder];

    // Only bind buffers when creating the encoder. Sampled textures depend on the
    // current GL program/MSL reflection and are rebound after the pipeline state is
    // selected for the draw.
    if (VAO())
    {
        if ([self bindVertexBuffersToCurrentRenderEncoder] == false)
        {
            DEBUG_PRINT("vertex buffer binding failed\n");
            [self recordGPUError];
            return false;
        }

        if ([self bindFragmentBuffersToCurrentRenderEncoder] == false)
        {
            DEBUG_PRINT("fragment buffer binding failed\n");
            [self recordGPUError];
            return false;
        }
    }

    // Record successful render encoder creation (final success)
    [self recordGPUSuccess];
    return true;
        
    } //     @autoreleasepool
}

- (bool) newCommandBuffer
{
    METAL_LOCK();
    bool result = [self newCommandBufferLocked];
    METAL_UNLOCK();
    return result;
}

- (bool) newCommandBufferLocked
{
    // CRITICAL FIX: Proper encoder cleanup BEFORE creating new command buffer
    // Metal API requires ending encoders before creating new command buffers

    // STEP 0: End any existing render encoder to prevent MTLReleaseAssertionFailure
    if (_currentRenderEncoder) {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Ending existing render encoder before creating new command buffer");
        }
        @try {
            [_currentRenderEncoder endEncoding];
            _currentRenderEncoder = nil;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: Exception ending render encoder: %@", exception);
            _currentRenderEncoder = nil; // Force clear even on exception
        }
    }

    // STEP 1: Clean up sync tracking list safely.
    // IMPORTANT: Do NOT dereference Sync* entries here. Sync objects are owned by GL sync lifecycle
    // and may already be deleted by glDeleteSync on other paths.
    SYNC_LOCK();
    if (_currentCommandBufferSyncList)
    {
        // Sync list access must use _syncListLock because mtlGetSync may append
        // while another thread rotates this context's command buffer.

        GLuint count = _currentCommandBufferSyncList->count;
        GLuint size = _currentCommandBufferSyncList->size;

        if (_currentCommandBufferSyncList->list == NULL || size == 0) {
            NSLog(@"MGL WARNING: Sync list storage invalid (list=%p size=%u), resetting", _currentCommandBufferSyncList->list, size);
            _currentCommandBufferSyncList->count = 0;
            SYNC_UNLOCK();
            goto create_new_command_buffer;
        }

        if (count > size) {
            NSLog(@"MGL WARNING: Sync list count overflow (count=%u size=%u), clamping", count, size);
            count = size;
            _currentCommandBufferSyncList->count = size;
        }

        for (GLuint i = 0; i < count; i++) {
            _currentCommandBufferSyncList->list[i] = NULL;
        }

        _currentCommandBufferSyncList->count = 0;
    }
    SYNC_UNLOCK();

create_new_command_buffer:
    // CRITICAL SAFETY: Validate command queue before creating buffer
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Cannot create command buffer - command queue is NULL");
        _currentCommandBuffer = NULL;
        return false;
    }

    // STEP 1: Create fresh command buffer FIRST with comprehensive AGX driver validation
    @try {
        // AGX DRIVER COMPATIBILITY: Validate command queue health before creating buffer
        if (!_commandQueue) {
            NSLog(@"MGL AGX ERROR: Command queue is NULL - recreating");
            [self resetMetalState];
            if (!_commandQueue) {
                NSLog(@"MGL AGX CRITICAL: Cannot recreate command queue");
                return false;
            }
        }

        // CRITICAL FIX: Validate _commandQueue before dereferencing to prevent NULL pointer crashes
        if (!_commandQueue) {
            NSLog(@"MGL AGX CRITICAL: _commandQueue is NULL - cannot create command buffer");
            [self recordGPUError];
            return false;
        }

        // Additional validation: Ensure _commandQueue is a valid Metal object
        @try {
            // Test if _commandQueue is valid by checking its class
            Class queueClass = [_commandQueue class];
            if (!queueClass) {
                NSLog(@"MGL AGX CRITICAL: _commandQueue is invalid (no class) - recreating");
                _commandQueue = [_device newCommandQueue];
                if (!_commandQueue) {
                    NSLog(@"MGL AGX CRITICAL: Failed to recreate command queue");
                    [self recordGPUError];
                    return false;
                }
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL AGX CRITICAL: _commandQueue validation exception: %@ - recreating", exception);
            [self recordGPUError];
            _commandQueue = [_device newCommandQueue];
            if (!_commandQueue) {
                NSLog(@"MGL AGX CRITICAL: Failed to recreate command queue after exception");
                [self recordGPUError];
                return false;
            }
        }

        _currentCommandBuffer = [_commandQueue commandBuffer];
        if (!_currentCommandBuffer) {
            NSLog(@"MGL AGX ERROR: Failed to create Metal command buffer - command queue may be in error state");
            [self recordGPUError];
            // Force command queue recreation
            [self resetMetalState];
            return false;
        }

        // AGX Driver Validation: Check if the command buffer is immediately invalid
        if (_currentCommandBuffer.error) {
            NSLog(@"MGL AGX WARNING: New command buffer has immediate error: %@", _currentCommandBuffer.error);
            [self recordGPUError];
            // Don't return false immediately - AGX sometimes creates error-state buffers that recover
        }

        // AGX DRIVER COMPATIBILITY: Enhanced validation to prevent rejections
        if (_currentCommandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"MGL AGX CRITICAL: Command buffer immediately in error state");
            [self recordGPUError];
            _currentCommandBuffer = nil; // Clear the problematic buffer
            [self resetMetalState]; // Force full reset
            return false;
        }

        // Additional AGX validation: Check for buffer properties that cause rejections
        if (_currentCommandBuffer.error) {
            NSLog(@"MGL AGX WARNING: Command buffer has immediate error: %@", _currentCommandBuffer.error);
            [self recordGPUError];
            _currentCommandBuffer = nil;
            [self resetMetalState];
            return false;
        }

        // Validate command queue health
        if (!_commandQueue) {
            NSLog(@"MGL AGX CRITICAL: Command queue became NULL");
            [self resetMetalState];
            return false;
        }

        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Successfully created new Metal command buffer (AGX validated)");
        }
        _mdiArgsScratchBuffer = nil;
        _mdiArgsScratchCapacity = 0;
        _mdiArgsScratchOffset = 0;
    } @catch (NSException *exception) {
        NSLog(@"MGL AGX ERROR: Exception creating command buffer: %@", exception);
        [self recordGPUError];
        _currentCommandBuffer = NULL;
        _mdiArgsScratchBuffer = nil;
        _mdiArgsScratchCapacity = 0;
        _mdiArgsScratchOffset = 0;

        // AGX DRIVER COMPATIBILITY: Force reset on exception to clear driver state
        [self resetMetalState];
        return false;
    }

    // STEP 2: Now handle pending event waits on the FRESH command buffer
    if (_currentEvent)
    {
        if (!_currentSyncName) {
            NSLog(@"MGL WARNING: dropping pending shared-event wait with no sync name");
            _currentEvent = NULL;
            return true;
        }

        if (kMGLDisableSharedEventSync) {
            NSLog(@"MGL INFO: Shared event wait disabled (debug no-op), skipping wait encode event=%p syncName=%u",
                  _currentEvent, _currentSyncName);
            _currentEvent = NULL;
            _currentSyncName = 0;
            return true;
        }

        // SAFELY ENCODE: Event wait functionality on the new command buffer
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Encoding event wait on fresh command buffer");
        }

        // CRITICAL SAFETY: Cache event and sync values to prevent race conditions
        id<MTLEvent> cachedEvent = _currentEvent;
        GLuint cachedSyncName = _currentSyncName;

        // COMPREHENSIVE EVENT VALIDATION: Validate Metal event pointer
        if (!cachedEvent) {
            NSLog(@"MGL ERROR: Cannot encode event wait - cached event is NULL");
            _currentEvent = NULL;
            _currentSyncName = 0;
            return false;
        }

        // Validate event pointer looks like a valid object address
        uintptr_t eventPtr = (uintptr_t)cachedEvent;
        if (eventPtr == 0x10 || eventPtr == 0x30 || eventPtr == 0x1000) {
            NSLog(@"MGL CRITICAL ERROR: Known corrupted event pointer pattern detected: 0x%lx", eventPtr);
            NSLog(@"MGL CRITICAL ERROR: Skipping event wait to prevent crash");
            _currentEvent = NULL;
            _currentSyncName = 0;
            return false;
        }

        if (eventPtr < 0x1000 || (eventPtr & 0x7) != 0) {
            NSLog(@"MGL ERROR: Suspicious event pointer value: %p", cachedEvent);
            NSLog(@"MGL INFO: Skipping event wait for safety");
            _currentEvent = NULL;
            _currentSyncName = 0;
            return false;
        }

        // ADDITIONAL SAFETY: Validate command buffer is still valid before encoding
        if (!_currentCommandBuffer) {
            NSLog(@"MGL ERROR: Command buffer became NULL before event wait encoding");
            _currentEvent = NULL;
            _currentSyncName = 0;
            return false;
        }

        @try {
            NSLog(@"MGL INFO: Encoding safe event wait: event=%p, syncName=%u, cmdbuf=%p", cachedEvent, cachedSyncName, _currentCommandBuffer);

            // Use conservative approach: only encode if everything looks perfect
            [_currentCommandBuffer encodeWaitForEvent:cachedEvent value:cachedSyncName];

            NSLog(@"MGL SUCCESS: Event wait encoded successfully on fresh command buffer");
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Event wait failed - %@: %@", exception.name, exception.reason);
            NSLog(@"MGL INFO: Continuing without event wait to maintain stability");
            // Continue without event wait - system remains stable
        }

        _currentEvent = NULL;
        _currentSyncName = 0;
    }

    return true;
}

- (bool)ensureWritableCommandBuffer:(const char *)reason
{
    METAL_LOCK();
    bool result = [self ensureWritableCommandBufferLocked:reason];
    METAL_UNLOCK();
    return result;
}

- (bool)ensureWritableCommandBufferLocked:(const char *)reason
{
    if (!_currentCommandBuffer) {
        if (kMGLDiagnosticStateLogs) {
            MGLTraceNSLog(@"MGL INFO: %s requested with NULL command buffer, creating one", reason ? reason : "operation");
        }
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to create command buffer for %s", reason ? reason : "operation");
            return false;
        }
    }

    MTLCommandBufferStatus status = _currentCommandBuffer.status;
    if (status >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL INFO: %s requested on finalized command buffer (status: %ld), rotating", reason ? reason : "operation", (long)status);
        [self endRenderEncodingLocked];
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer for %s", reason ? reason : "operation");
            return false;
        }

        if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted) {
            NSLog(@"MGL ERROR: Unable to obtain writable command buffer for %s", reason ? reason : "operation");
            return false;
        }
    }

    return true;
}

- (bool) newCommandBufferAndRenderEncoder
{
    // AGGRESSIVE MEMORY SAFETY: Validate fundamental Metal objects before use
    if (!_device) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - No device available");
        return false;
    }

    if (!_commandQueue) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - No command queue available");
        return false;
    }

    // Validate device pointer lower bound only (high canonical addresses are valid on macOS)
    uintptr_t device_addr = (uintptr_t)_device;
    if (device_addr < 0x1000) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - Invalid device pointer: 0x%lx", device_addr);
        return false;
    }

    @try {
        if ([self newCommandBuffer] == false) {
            NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - newCommandBuffer failed");
            return false;
        }

        if ([self newRenderEncoder] == false) {
            NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - newRenderEncoder failed");
            return false;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - Metal operation failed: %@", exception);
        return false;
    }

    return true;
}

#pragma mark pipeline descriptor
-(MTLRenderPipelineDescriptor *)generatePipelineDescriptor
{
    if (!ctx) {
        NSLog(@"MGL PIPELINE DESC fail: context is NULL");
        return nil;
    }

    Program *vertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    GLuint renderProgramKey = mglCurrentRenderProgramKey(ctx);
    GLuint vertexProgramName = vertexProgram ? vertexProgram->name : 0u;
    GLuint fragmentProgramName = fragmentProgram ? fragmentProgram->name : 0u;
    BOOL rasterizerDiscard = ctx->state.caps.rasterizer_discard ? YES : NO;

    if (!vertexProgram || (!fragmentProgram && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing stage program key=%u vs=%p fs=%p current=%u pipeline=%u",
              (unsigned)renderProgramKey,
              vertexProgram,
              fragmentProgram,
              (unsigned)ctx->state.program_name,
              (unsigned)ctx->state.var.program_pipeline_binding);
        return nil;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC begin key=%u vsProgram=%u fsProgram=%u",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName);
    }

    if ([self bindMTLProgram:vertexProgram] == false) {
        NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for VS program=%u",
              (unsigned)vertexProgramName);
        return nil;
    }
    if (fragmentProgram &&
        fragmentProgram != vertexProgram &&
        [self bindMTLProgram:fragmentProgram] == false) {
        NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for FS program=%u",
              (unsigned)fragmentProgramName);
        return nil;
    }

    Shader *vertex_shader = vertexProgram->shader_slots[_VERTEX_SHADER];
    Shader *fragment_shader = fragmentProgram ? fragmentProgram->shader_slots[_FRAGMENT_SHADER] : NULL;
    if (!vertex_shader || (!fragment_shader && !rasterizerDiscard)) {
        NSLog(@"MGL PIPELINE DESC fail: missing shaders key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)",
              (unsigned)renderProgramKey,
              (unsigned)vertexProgramName,
              (unsigned)fragmentProgramName,
              vertex_shader,
              fragment_shader);
        return nil;
    }

	    void *vertexFunctionPtr = vertex_shader->mtl_data.function;
	    if (ctx->state.var.clip_origin == GL_UPPER_LEFT) {
	        if (ctx->state.var.clip_depth_mode == GL_ZERO_TO_ONE &&
	            vertex_shader->mtl_data.upper_left_zero_to_one_function) {
	            vertexFunctionPtr = vertex_shader->mtl_data.upper_left_zero_to_one_function;
	        } else if (ctx->state.var.clip_depth_mode != GL_ZERO_TO_ONE &&
	                   vertex_shader->mtl_data.upper_left_function) {
	            vertexFunctionPtr = vertex_shader->mtl_data.upper_left_function;
	        }
	    } else if (ctx->state.var.clip_depth_mode == GL_ZERO_TO_ONE &&
	               vertex_shader->mtl_data.zero_to_one_function) {
	        vertexFunctionPtr = vertex_shader->mtl_data.zero_to_one_function;
	    }

	    id<MTLFunction> vertexFunction = (__bridge id<MTLFunction>)vertexFunctionPtr;
	    id<MTLFunction> fragmentFunction = fragment_shader ? (__bridge id<MTLFunction>)(fragment_shader->mtl_data.function) : nil;
    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC vs=%@ fs=%@",
              vertexFunction ? vertexFunction.name : @"(null)",
              fragmentFunction ? fragmentFunction.name : @"(null)");
    }
	    if (!vertexFunction || (!fragmentFunction && !rasterizerDiscard)) {
	        NSLog(@"MGL PIPELINE DESC fail: missing MTLFunction key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)",
	              (unsigned)renderProgramKey,
	              (unsigned)vertexProgramName,
	              (unsigned)fragmentProgramName,
	              vertexFunction,
	              fragmentFunction);
	        return nil;
	    }

	    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    if (!pipelineStateDescriptor) {
        NSLog(@"MGL PIPELINE DESC fail: descriptor allocation failed for key=%u",
              (unsigned)renderProgramKey);
        return nil;
    }
    pipelineStateDescriptor.label = @"GLSL Pipeline";
    pipelineStateDescriptor.vertexFunction = vertexFunction;
    pipelineStateDescriptor.fragmentFunction = fragmentFunction;
    /* GL_RASTERIZER_DISCARD: When rasterizer discard is active, the
     * fragment function is nil.  Metal requires rasterizationEnabled to
     * match the vertex function's return type:
     *   - vertex returns void  (no stage outputs)  -> rasterizationEnabled = NO
     *   - vertex returns struct (has stage outputs) -> rasterizationEnabled = YES
     * SPIRV-Cross generates a void return type when the vertex shader has
     * no varying outputs (e.g. SSBO-only vertex shaders), and a struct
     * return type when it has varying outputs. */
    if (rasterizerDiscard) {
        GLuint vsOutputCount = vertexProgram->spirv_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES].count;
        pipelineStateDescriptor.rasterizationEnabled = (vsOutputCount > 0) ? YES : NO;
    } else {
        pipelineStateDescriptor.rasterizationEnabled = YES;
    }

    if (ctx->state.framebuffer) {
        Framebuffer *fbo = ctx->state.framebuffer;

        for (int i = 0; i < STATE(max_color_attachments); i++) {
            if (fbo->color_attachments[i].texture) {
                Texture *tex = [self framebufferAttachmentTexture:&fbo->color_attachments[i]];
                if (tex && ![self bindMTLTexture:tex]) {
                    NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for color attachment %d tex=%u",
                          i, tex->name);
                    return nil;
                }
                if (tex && tex->mtl_data) {
                    pipelineStateDescriptor.colorAttachments[i].pixelFormat = mtlPixelFormatForGLTex(tex);
                } else {
                    pipelineStateDescriptor.colorAttachments[i].pixelFormat = MTLPixelFormatInvalid;
                }
            }

            if ((fbo->color_attachment_bitfield >> (i + 1)) == 0) {
                break;
            }
        }

        if (fbo->depth.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->depth];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for depth tex=%u", tex->name);
                return nil;
            }
            if (tex && tex->mtl_data) {
                MTLPixelFormat depthFormat = mtlPixelFormatForGLTex(tex);
                if (depthFormat == MTLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid depth texture format, falling back to Depth32Float");
                    depthFormat = MTLPixelFormatDepth32Float;
                }
                pipelineStateDescriptor.depthAttachmentPixelFormat = depthFormat;
            } else {
                pipelineStateDescriptor.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
            }
        }

        if (fbo->stencil.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->stencil];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for stencil tex=%u", tex->name);
                return nil;
            }
            if (tex && tex->mtl_data) {
                MTLPixelFormat stencilFormat = mtlPixelFormatForGLTex(tex);
                if (stencilFormat == MTLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid stencil texture format, falling back to Stencil8");
                    stencilFormat = MTLPixelFormatStencil8;
                }
                pipelineStateDescriptor.stencilAttachmentPixelFormat = stencilFormat;
            } else {
                pipelineStateDescriptor.stencilAttachmentPixelFormat = MTLPixelFormatInvalid;
            }
        }
    } else {
        MTLPixelFormat preferredColor0 = MTLPixelFormatInvalid;
        if (_renderPassDescriptor && _renderPassDescriptor.colorAttachments[0].texture) {
            preferredColor0 = _renderPassDescriptor.colorAttachments[0].texture.pixelFormat;
        } else if (_drawable && _drawable.texture) {
            preferredColor0 = _drawable.texture.pixelFormat;
        } else {
            preferredColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = preferredColor0;

        if (ctx->depth_format.format) {
            MTLPixelFormat depthFormat = ctx->depth_format.mtl_pixel_format;
            if (depthFormat == MTLPixelFormatInvalid) {
                depthFormat = MTLPixelFormatDepth32Float;
            }
            pipelineStateDescriptor.depthAttachmentPixelFormat = depthFormat;
        }

        if (ctx->stencil_format.format) {
            MTLPixelFormat stencilFormat = ctx->stencil_format.mtl_pixel_format;
            if (stencilFormat == MTLPixelFormatInvalid ||
                stencilFormat == MTLPixelFormatDepth32Float_Stencil8) {
                stencilFormat = MTLPixelFormatStencil8;
            }
            pipelineStateDescriptor.stencilAttachmentPixelFormat = stencilFormat;
        }
    }

    if (_renderPassDescriptor) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id<MTLTexture> rpColor = _renderPassDescriptor.colorAttachments[i].texture;
            if (rpColor) {
                pipelineStateDescriptor.colorAttachments[i].pixelFormat = rpColor.pixelFormat;
            }
        }

        id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
        id<MTLTexture> rpStencil = _renderPassDescriptor.stencilAttachment.texture;
        pipelineStateDescriptor.depthAttachmentPixelFormat =
            rpDepth ? rpDepth.pixelFormat : MTLPixelFormatInvalid;
        pipelineStateDescriptor.stencilAttachmentPixelFormat =
            rpStencil ? rpStencil.pixelFormat : MTLPixelFormatInvalid;
    }

    BOOL color0IsIntentionallyDisabled =
        ctx->state.framebuffer &&
        mglMetalDrawBufferAt(ctx, 0u) == GL_NONE;

    if (!color0IsIntentionallyDisabled &&
        (pipelineStateDescriptor.colorAttachments[0].pixelFormat == MTLPixelFormatInvalid ||
         pipelineStateDescriptor.colorAttachments[0].pixelFormat == 0)) {
        MTLPixelFormat fallbackColor0 = MTLPixelFormatInvalid;
        if (_renderPassDescriptor && _renderPassDescriptor.colorAttachments[0].texture) {
            fallbackColor0 = _renderPassDescriptor.colorAttachments[0].texture.pixelFormat;
        } else if (_drawable && _drawable.texture) {
            fallbackColor0 = _drawable.texture.pixelFormat;
        } else {
            fallbackColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        if (fallbackColor0 == MTLPixelFormatInvalid || fallbackColor0 == 0) {
            fallbackColor0 = MTLPixelFormatBGRA8Unorm;
        }
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL PIPELINE DESC missing color pixel format, fallback pixelFormat=%lu",
                  (unsigned long)fallbackColor0);
        }
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = fallbackColor0;
    }

    NSUInteger resolvedSampleCount = 1;
    if (_renderPassDescriptor) {
        id<MTLTexture> rpColor0 = _renderPassDescriptor.colorAttachments[0].texture;
        id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
        id<MTLTexture> rpStencil = _renderPassDescriptor.stencilAttachment.texture;
        if (rpColor0 && rpColor0.sampleCount > 0) {
            resolvedSampleCount = rpColor0.sampleCount;
        } else if (rpDepth && rpDepth.sampleCount > 0) {
            resolvedSampleCount = rpDepth.sampleCount;
        } else if (rpStencil && rpStencil.sampleCount > 0) {
            resolvedSampleCount = rpStencil.sampleCount;
        }
    }
    if (resolvedSampleCount == 0) {
        resolvedSampleCount = 1;
    }
    /* MTLRenderPipelineDescriptor.rasterSampleCount defaults to 1, so the
     * previous "only set when 0" check never overrode it. Always align the
     * pipeline's sample count with the actual render-pass attachment so
     * multisample (2D_MULTISAMPLE / 2D_MULTISAMPLE_ARRAY) targets don't
     * silently mismatch and produce empty draws. */
    pipelineStateDescriptor.rasterSampleCount = resolvedSampleCount;
    if (pipelineStateDescriptor.rasterSampleCount == 0) {
        pipelineStateDescriptor.rasterSampleCount = 1;
    }
    mglNormalizePipelineDepthStencilFormats(pipelineStateDescriptor, "generate");
    mglEnableIndirectCommandBuffersForPipeline(pipelineStateDescriptor);

    /* GL_COLOR_LOGIC_OP: Metal on Apple Silicon does NOT expose
     * MTLLogicOperation or logicOpEnabled on MTLRenderPipelineColorAttachmentDescriptor
     * (verified against MacOSX SDK headers). Logic op would require fragment-shader
     * emulation (reading framebuffer, applying bitwise op), which is not implemented.
     * State tracking (logic_op_mode) is preserved for query correctness. */
    NSUInteger activeColorAttachmentCount = 0;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (pipelineStateDescriptor.colorAttachments[i].pixelFormat != MTLPixelFormatInvalid &&
            pipelineStateDescriptor.colorAttachments[i].pixelFormat != 0) {
            activeColorAttachmentCount++;
        }
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC colorAttachmentCount=%lu depthFormat=%lu stencilFormat=%lu sampleCount=%lu",
              (unsigned long)activeColorAttachmentCount,
              (unsigned long)pipelineStateDescriptor.depthAttachmentPixelFormat,
              (unsigned long)pipelineStateDescriptor.stencilAttachmentPixelFormat,
              (unsigned long)pipelineStateDescriptor.rasterSampleCount);
        NSLog(@"MGL PIPELINE DESC renderTarget[0]=%lu",
              (unsigned long)pipelineStateDescriptor.colorAttachments[0].pixelFormat);
    }

    return pipelineStateDescriptor;
}

#pragma mark vertex descriptor
- (MTLVertexDescriptor *)generateVertexDescriptor
{
    MTLVertexDescriptor *vertexDescriptor = [[MTLVertexDescriptor alloc] init];
    if (!vertexDescriptor) {
        NSLog(@"MGL VERTEX ERROR: failed to allocate MTLVertexDescriptor");
        return nil;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    GLuint activeProgramName = activeProgram ? activeProgram->name : (ctx ? mglCurrentRenderProgramKey(ctx) : 0);
    GLuint maxAttribs;

    if (!vao) {
        NSLog(@"MGL PIPELINE DESC fail: cannot build vertex descriptor without a valid VAO");
        return nil;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL VERTEX DESC begin program=%u vao=%p enabledMask=0x%x",
              (unsigned)activeProgramName, vao, vao->enabled_attribs);
    }

    [vertexDescriptor reset]; // ??? debug
    maxAttribs = MAX_ATTRIBS;

    // Get the vertex shader MSL source to check which attributes are actually used.
    // SPIRV-Cross may report attributes in the reflection that are not used in the MSL
    // (e.g., when the GLSL declares an attribute but doesn't use it in the shader body).
    const char *vsMslStr = NULL;
    if (activeProgram) {
        vsMslStr = activeProgram->spirv[_VERTEX_SHADER].msl_str;
    }

    // we can bind a new vertex descriptor without creating a new renderbuffer
    bool attribsEnabledByApp = (vao->enabled_attribs != 0u);
    for (GLuint i = 0; i < maxAttribs; i++)
    {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, i)) {
            continue;
        }
        // Skip attributes that are in the reflection but not in the MSL source.
        // SPIRV-Cross optimizes away unused attributes from the MSL input struct,
        // but still reports them in the reflection. Configuring a vertex descriptor
        // entry for an attribute the shader doesn't use can cause Metal to silently
        // produce no rasterization output.
        if (vsMslStr) {
            char attrPattern[32];
            snprintf(attrPattern, sizeof(attrPattern), "[[attribute(%u)]]", i);
            if (!strstr(vsMslStr, attrPattern)) {
                continue;
            }
        }
        BOOL usesCurrentValue = mglRendererVertexAttribUsesCurrentValue(vao, i);
        MGLResolvedVertexAttribBinding resolved = {0};
        bool hasAttribBinding = mglRendererResolveVertexAttribBinding(ctx,
                                                                      vao,
                                                                      i,
                                                                      __FUNCTION__,
                                                                      &resolved);
        // When enabled_attribs tracking is empty but the program uses this attribute,
        // validate the buffer and proceed (Sodium DSA path compatibility).
        if (!attribsEnabledByApp && !hasAttribBinding) {
            continue;
        }

        {
            MTLVertexFormat format;
            Buffer *attribBuffer = hasAttribBinding ? resolved.buffer : NULL;

            if (!usesCurrentValue && !attribBuffer)
            {
                NSLog(@"MGL PIPELINE DESC fail: attrib %u enabled but buffer is invalid", i);
                return NULL;
            }

            GLboolean normalized = vao->attrib[i].normalized;
            if (!normalized &&
                vao->attrib[i].type == GL_UNSIGNED_BYTE &&
                vao->attrib[i].size == 4 &&
                mglRendererVertexAttribIsColorInput(activeProgram, i)) {
                normalized = GL_TRUE;
            }

            /* Determine whether this attrib will be CPU-converted before
             * binding. Converted buffers are reborn starting at the original
             * binding_offset, so the vertex descriptor's attribute offset must
             * NOT include binding_offset for them (only relativeoffset).
             * Non-converted attribs bind the original buffer at offset 0, so
             * their attribute offset must include binding_offset. */
            bool needsConversion = false;
            if (vao->attrib[i].type == GL_DOUBLE) {
                needsConversion = true;
            } else if (vao->attrib[i].integer == 0 &&
                       (vao->attrib[i].type == GL_INT ||
                        vao->attrib[i].type == GL_UNSIGNED_INT)) {
                needsConversion = true;
            } else if (vao->attrib[i].integer == 1) {
                SpirvResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, i);
                GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
                if (mglIntegerAttribNeedsConversion(vao->attrib[i].type,
                                                    shaderGlType,
                                                    vao->attrib[i].size,
                                                    NULL)) {
                    needsConversion = true;
                }
            }

            if (vao->attrib[i].type == GL_DOUBLE) {
                format = mglDoubleVertexAttribFloatFormat(vao->attrib[i].size);
            } else if (vao->attrib[i].integer == 0 &&
                       (vao->attrib[i].type == GL_INT ||
                        vao->attrib[i].type == GL_UNSIGNED_INT)) {
                /* Metal's 32-bit integer vertex formats (Int/UInt) cannot feed
                 * float shader inputs; glVertexAttribFormat (non-integer) with
                 * GL_INT/GL_UNSIGNED_INT requires int->float conversion. Use a
                 * float format here and convert the data on the CPU side in
                 * bindVertexBuffersToCurrentRenderEncoder (like GL_DOUBLE). */
                format = mglDoubleVertexAttribFloatFormat(vao->attrib[i].size);
            } else if (vao->attrib[i].integer == 1) {
                /* glVertexAttribIFormat path: Metal only allows 32-bit Int
                 * formats for int shader inputs and UInt formats for uint
                 * inputs. 8/16-bit signed formats sign-extend to int (and
                 * zero-extend to uint), but unsigned source formats cannot
                 * feed int inputs and signed sources cannot feed uint inputs.
                 * When the source signedness is incompatible with the shader's
                 * declared type, convert the data to the shader's 32-bit
                 * integer type on the CPU side in
                 * bindVertexBuffersToCurrentRenderEncoder. */
                MTLVertexFormat convertedFormat = MTLVertexFormatInvalid;
                SpirvResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, i);
                GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
                if (mglIntegerAttribNeedsConversion(vao->attrib[i].type,
                                                    shaderGlType,
                                                    vao->attrib[i].size,
                                                    &convertedFormat) &&
                    convertedFormat != MTLVertexFormatInvalid) {
                    format = convertedFormat;
                } else {
                    format = glTypeSizeToMtlType(vao->attrib[i].type,
                                                 vao->attrib[i].size,
                                                 normalized);
                }
            } else {
                format = glTypeSizeToMtlType(vao->attrib[i].type,
                                             vao->attrib[i].size,
                                             normalized);
            }

            if (format == MTLVertexFormatInvalid)
            {
                NSLog(@"MGL PIPELINE DESC fail: unable to map attrib %u type/size/normalize to MTL format", i);
                return nil;
            }

            int mapped_buffer_index;

            mapped_buffer_index = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, i, __FUNCTION__);
            if (mapped_buffer_index < 0 || mapped_buffer_index >= (int)kMGLMaxMetalVertexBufferCount) {
                NSLog(@"MGL ERROR: Invalid vertex buffer index %d for attribute %d (max valid=%lu)",
                      mapped_buffer_index, i, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return NULL;
            }

            vertexDescriptor.attributes[i].bufferIndex = mapped_buffer_index;
            /* When multiple attributes share a Metal buffer slot (because they
             * use the same VBO/stride/divisor), the per-attribute binding_offset
             * must be folded into the vertex descriptor's attribute offset.
             * The buffer itself is bound at offset 0 in
             * bindVertexBuffersToCurrentRenderEncoder.
             *
             * EXCEPTION: CPU-converted attribs (GL_DOUBLE, GL_INT→float,
             * integer signedness mismatch) produce a fresh buffer that already
             * starts at binding_offset. For those, the attribute offset must be
             * just relativeoffset, otherwise the shader would read past the
             * start of the converted data. */
            if (usesCurrentValue) {
                vertexDescriptor.attributes[i].offset = 0u;
            } else if (needsConversion) {
                vertexDescriptor.attributes[i].offset = (NSUInteger)resolved.relativeoffset;
            } else {
                vertexDescriptor.attributes[i].offset = (NSUInteger)(resolved.binding_offset + resolved.relativeoffset);
            }
            vertexDescriptor.attributes[i].format = format;

            if (usesCurrentValue) {
                vertexDescriptor.layouts[mapped_buffer_index].stride = 16u;
            } else if (vao->attrib[i].type == GL_DOUBLE) {
                NSUInteger doubleStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)(vao->attrib[i].size * sizeof(GLdouble));
                vertexDescriptor.layouts[mapped_buffer_index].stride = mglAlignVertexStrideForMetal(doubleStride);
            } else if (vao->attrib[i].integer == 0 &&
                       (vao->attrib[i].type == GL_INT ||
                        vao->attrib[i].type == GL_UNSIGNED_INT)) {
                NSUInteger intStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)(vao->attrib[i].size * sizeof(GLint));
                vertexDescriptor.layouts[mapped_buffer_index].stride = mglAlignVertexStrideForMetal(intStride);
            } else if (vao->attrib[i].integer == 1) {
                /* Integer attribs that need CPU conversion (unsigned source
                 * feeding int shader input, or signed source feeding uint
                 * input) are reborn as 32-bit Int/UInt buffers, so the layout
                 * stride must match the converted stride (componentCount * 4).
                 * Directly-compatible integer attribs keep their source stride. */
                MTLVertexFormat convertedFormat = MTLVertexFormatInvalid;
                SpirvResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, i);
                GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
                if (mglIntegerAttribNeedsConversion(vao->attrib[i].type,
                                                    shaderGlType,
                                                    vao->attrib[i].size,
                                                    &convertedFormat) &&
                    convertedFormat != MTLVertexFormatInvalid) {
                    NSUInteger convStride = mglAlignVertexStrideForMetal(
                        (NSUInteger)vao->attrib[i].size * sizeof(GLint));
                    vertexDescriptor.layouts[mapped_buffer_index].stride = convStride;
                } else if (vertexDescriptor.layouts[mapped_buffer_index].stride == 0) {
                    vertexDescriptor.layouts[mapped_buffer_index].stride = resolved.stride;
                }
            } else if (vertexDescriptor.layouts[mapped_buffer_index].stride == 0) {
                vertexDescriptor.layouts[mapped_buffer_index].stride = resolved.stride;
            }

            if (!usesCurrentValue && resolved.divisor)
            {
                vertexDescriptor.layouts[mapped_buffer_index].stepRate = resolved.divisor;
                vertexDescriptor.layouts[mapped_buffer_index].stepFunction = MTLVertexStepFunctionPerInstance;
            }
            else
            {
	            vertexDescriptor.layouts[mapped_buffer_index].stepRate = 1;
	            vertexDescriptor.layouts[mapped_buffer_index].stepFunction = MTLVertexStepFunctionPerVertex;
	        }

            static uint64_t s_traceFileVertexDescriptorAttribLogs = 0;
            if (mglProgramNeedsTraceLog(activeProgram) &&
                mglShouldLogTraceFileBindingForProgram(activeProgram, &s_traceFileVertexDescriptorAttribLogs)) {
                SpirvResource *resource = mglRendererProgramVertexAttribResource(activeProgram, i);
                mglTraceLog("VATTR_DESC program=%u attrib=%u resource=%s loc=%u metalSlot=%d glBuffer=%u bindingIndex=%u bindingOffset=%lld relOffset=%lld stride=%u size=%u type=0x%x normalized=%u/%u divisor=%u table=%d format=%lu(%s)",
                            (unsigned)activeProgramName,
                            (unsigned)i,
                            resource && resource->name ? resource->name : "(unknown)",
                            resource ? (unsigned)resource->location : 0xffffffffu,
                            mapped_buffer_index,
                            usesCurrentValue ? 0u : (unsigned)attribBuffer->name,
                            usesCurrentValue ? i : (unsigned)resolved.binding_index,
                            usesCurrentValue ? 0ll : (long long)resolved.binding_offset,
                            usesCurrentValue ? 0ll : (long long)resolved.relativeoffset,
                            usesCurrentValue ? 16u : (unsigned)resolved.stride,
                            (unsigned)vao->attrib[i].size,
                            (unsigned)vao->attrib[i].type,
                            (unsigned)vao->attrib[i].normalized,
                            (unsigned)normalized,
                            usesCurrentValue ? 0u : (unsigned)resolved.divisor,
                            usesCurrentValue ? 0 : (resolved.uses_binding_table ? 1 : 0),
                            (unsigned long)format,
                            mglVertexFormatName(format));
            }

	        if (kMGLVerbosePipelineLogs) {
	            NSLog(@"MGL VERTEX DESC attrib=%u enabled=%u glBuffer=%u metalIndex=%d bindingOffset=%lld offset=0x%llx stride=%u size=%u type=0x%x normalized=%u divisor=%u table=%d format=%lu(%s)",
	                  i,
	                  usesCurrentValue ? 0u : 1u,
                      usesCurrentValue ? 0u : attribBuffer->name,
                      mapped_buffer_index,
                      usesCurrentValue ? 0ll : (long long)resolved.binding_offset,
                      usesCurrentValue ? 0ull : (unsigned long long)(uintptr_t)resolved.relativeoffset,
                      usesCurrentValue ? 16u : (unsigned)resolved.stride,
                      (unsigned)vao->attrib[i].size,
                      (unsigned)vao->attrib[i].type,
                      (unsigned)normalized,
                      usesCurrentValue ? 0u : (unsigned)resolved.divisor,
                      usesCurrentValue ? 0 : (resolved.uses_binding_table ? 1 : 0),
                      (unsigned long)format,
                      mglVertexFormatName(format));
            }

            if (vao->attrib[i].type == GL_UNSIGNED_BYTE &&
                vao->attrib[i].size == 4 &&
                vao->attrib[i].normalized == GL_FALSE &&
                !normalized) {
                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL VERTEX DESC note: attrib %u uses UBYTE4 non-normalized (format=%lu)",
                          i, (unsigned long)format);
                }
            }
        }

    }

    // clear all dirty bits as they have been translated into a vertex descriptor
    vao->dirty_bits = 0;

    return vertexDescriptor;
}

- (void) updateBlendStateCache
{
    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        if (!mglIsValidGLBlendFactor(ctx->state.var.blend_src_rgb[i])) {
            mglLogRenderStateRepair("blend_src_rgb", ctx->state.var.blend_src_rgb[i], GL_ONE);
            ctx->state.var.blend_src_rgb[i] = GL_ONE;
        }
        if (!mglIsValidGLBlendFactor(ctx->state.var.blend_src_alpha[i])) {
            mglLogRenderStateRepair("blend_src_alpha", ctx->state.var.blend_src_alpha[i], GL_ONE);
            ctx->state.var.blend_src_alpha[i] = GL_ONE;
        }
        if (!mglIsValidGLBlendFactor(ctx->state.var.blend_dst_rgb[i])) {
            mglLogRenderStateRepair("blend_dst_rgb", ctx->state.var.blend_dst_rgb[i], GL_ZERO);
            ctx->state.var.blend_dst_rgb[i] = GL_ZERO;
        }
        if (!mglIsValidGLBlendFactor(ctx->state.var.blend_dst_alpha[i])) {
            mglLogRenderStateRepair("blend_dst_alpha", ctx->state.var.blend_dst_alpha[i], GL_ZERO);
            ctx->state.var.blend_dst_alpha[i] = GL_ZERO;
        }
        if (!mglIsValidGLBlendEquation(ctx->state.var.blend_equation_rgb[i])) {
            mglLogRenderStateRepair("blend_equation_rgb", ctx->state.var.blend_equation_rgb[i], GL_FUNC_ADD);
            ctx->state.var.blend_equation_rgb[i] = GL_FUNC_ADD;
        }
        if (!mglIsValidGLBlendEquation(ctx->state.var.blend_equation_alpha[i])) {
            mglLogRenderStateRepair("blend_equation_alpha", ctx->state.var.blend_equation_alpha[i], GL_FUNC_ADD);
            ctx->state.var.blend_equation_alpha[i] = GL_FUNC_ADD;
        }

        _src_blend_rgb_factor[i] = [self blendFactorFromGL:ctx->state.var.blend_src_rgb[i]];
        _src_blend_alpha_factor[i] = [self blendFactorFromGL:ctx->state.var.blend_src_alpha[i]];

        _dst_blend_rgb_factor[i] = [self blendFactorFromGL:ctx->state.var.blend_dst_rgb[i]];
        _dst_blend_alpha_factor[i] = [self blendFactorFromGL:ctx->state.var.blend_dst_alpha[i]];

        _rgb_blend_operation[i] = [self blendOperationFromGL: ctx->state.var.blend_equation_rgb[i]];
        _alpha_blend_operation[i] = [self blendOperationFromGL: ctx->state.var.blend_equation_alpha[i]];

        if (!ctx->state.caps.use_color_mask[i]) {
            _color_mask[i] = MTLColorWriteMaskAll;
        } else {
            _color_mask[i] = MTLColorWriteMaskNone;

            if (ctx->state.var.color_writemask[i][0])
                _color_mask[i] |= MTLColorWriteMaskRed;

            if (ctx->state.var.color_writemask[i][1])
                _color_mask[i] |= MTLColorWriteMaskGreen;

            if (ctx->state.var.color_writemask[i][2])
                _color_mask[i] |= MTLColorWriteMaskBlue;

            if (ctx->state.var.color_writemask[i][3])
                _color_mask[i] |= MTLColorWriteMaskAlpha;
        }
    }
}

-(void)bindBlendStateToPipelineStateDescriptor:(MTLRenderPipelineDescriptor *)pipelineStateDescriptor
{
    pipelineStateDescriptor.alphaToCoverageEnabled = ctx->state.caps.sample_alpha_to_coverage ? YES : NO;
    pipelineStateDescriptor.alphaToOneEnabled = ctx->state.caps.sample_alpha_to_one ? YES : NO;

    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        if (pipelineStateDescriptor.colorAttachments[i].pixelFormat != MTLPixelFormatInvalid)
        {
            if (mglMetalDrawBufferAt(ctx, (GLuint)i) == GL_NONE) {
                pipelineStateDescriptor.colorAttachments[i].blendingEnabled = NO;
                pipelineStateDescriptor.colorAttachments[i].writeMask = 0;
                continue;
            }

            pipelineStateDescriptor.colorAttachments[i].blendingEnabled =
                ctx->state.caps.blendi[i] ? true : false;

            pipelineStateDescriptor.colorAttachments[i].sourceRGBBlendFactor = _src_blend_rgb_factor[i];
            pipelineStateDescriptor.colorAttachments[i].destinationRGBBlendFactor = _dst_blend_rgb_factor[i];
            pipelineStateDescriptor.colorAttachments[i].sourceAlphaBlendFactor = _src_blend_alpha_factor[i];
            pipelineStateDescriptor.colorAttachments[i].destinationAlphaBlendFactor = _dst_blend_alpha_factor[i];

            pipelineStateDescriptor.colorAttachments[i].rgbBlendOperation = _rgb_blend_operation[i];
            pipelineStateDescriptor.colorAttachments[i].alphaBlendOperation = _alpha_blend_operation[i];

            pipelineStateDescriptor.colorAttachments[i].writeMask = _color_mask[i];
        }
    }
}

-(bool)bindFramebufferAttachmentTextures
{
    Framebuffer *fbo;

    // MEMORY SAFETY: Validate context and framebuffer
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in bindFramebufferAttachmentTextures");
        return false;
    }

    // Validate context pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t ctx_addr = (uintptr_t)ctx;
    if (ctx_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected in bindFramebufferAttachmentTextures: 0x%lx", ctx_addr);
        return false;
    }

    fbo = ctx->state.framebuffer;

    // MEMORY SAFETY: Validate framebuffer pointer
    if (!fbo) {
        NSLog(@"MGL ERROR: NULL framebuffer detected in bindFramebufferAttachmentTextures");
        return false;
    }

    // Validate framebuffer pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t fbo_addr = (uintptr_t)fbo;
    if (fbo_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid framebuffer pointer detected in bindFramebufferAttachmentTextures: 0x%lx", fbo_addr);
        return false;
    }

    for (int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        if (fbo->color_attachments[i].texture)
        {
            bool isDrawBuffer = true;
            if (fbo->color_attachments[i].textarget == GL_RENDERBUFFER && fbo->color_attachments[i].buf.rbo) {
                isDrawBuffer = fbo->color_attachments[i].buf.rbo->is_draw_buffer;
            }

            if ([self bindFramebufferTexture: &fbo->color_attachments[i] isDrawBuffer:isDrawBuffer] == false)
            {
                DEBUG_PRINT("Failed Framebuffer Attachment\n");
                return false;
            }
        }

        // early out
        if ((fbo->color_attachment_bitfield >> (i+1)) == 0)
            break;
    }

    // depth attachment
    if (fbo->depth.texture)
    {
        if ([self bindFramebufferTexture: &fbo->depth isDrawBuffer: true] == false)
        {
            DEBUG_PRINT("Failed Framebuffer Attachment\n");
            return false;
        }
    }

    // stencil attachment
    if (fbo->stencil.texture)
    {
        if ([self bindFramebufferTexture: &fbo->stencil isDrawBuffer: true] == false)
        {
            DEBUG_PRINT("Failed Framebuffer Attachment\n");
            return false;
        }
    }

    return true;
}

- (void)updateGLSampledCopiesForEndedRenderPassFramebuffer:(Framebuffer *)fbo
                                                  drawCount:(GLsizei)drawCount
                                               drawBuffers:(const GLenum *)drawBuffers
                                                    reason:(const char *)reason
{
    (void)drawCount;
    (void)drawBuffers;

    if (!ctx || !fbo) {
        return;
    }

    /* Early-out: skip the per-attachment copy loop entirely when no texture
     * in this FBO is a sampled render target.  The old code unconditionally
     * iterated all color attachments on every endRenderPass and created a
     * Y-flipped copy for each (~313 copies/frame, most never sampled).  The
     * copy is only needed when the texture will be sampled by a non-yflip
     * shader in a subsequent draw, which we can't know here — but we CAN skip
     * textures that were never written (rtVer==0) or never flagged as RT.
     *
     * Iterate the actual FBO color attachments rather than the draw-buffer
     * snapshot.  MC 1.21.11's render abstraction creates transient FBOs such
     * as the GUI item atlas where the GL draw-buffer state can be incomplete
     * by the time the Metal encoder ends, but the attachment itself is still
     * the texture that was rendered and will be sampled immediately.
     *
     * NOTE: do NOT skip non-zero attachment levels here.  MC 1.21.11's
     * terrain atlas is a mipmapped RT whose mip 1-4 are written by separate
     * FBOs (one per mip level).  Skipping them left the Y-flip copy stale
     * after those passes ended, so terrain sampling mip>0 fell back to the
     * un-flipped Metal RT and rendered stripes.  The per-level blit inside
     * updateGLSampledRenderTargetCopyForTexture handles non-zero levels
     * correctly. */
    bool anySampledRT = false;
    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }
        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        Texture *tex = [self framebufferAttachmentTexture:attachment];
        if (tex && tex->mtl_data && tex->is_render_target &&
            tex->mtl_render_target_write_version != 0u) {
            anySampledRT = true;
            break;
        }
    }
    if (!anySampledRT) {
        return;
    }

    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];

        Texture *tex = [self framebufferAttachmentTexture:attachment];
        if (!tex || !tex->mtl_data) {
            continue;
        }

        id<MTLTexture> source = (__bridge id<MTLTexture>)(tex->mtl_data);
        if (![self textureCanUseGLSampledRenderTargetCopy:tex source:source]) {
            continue;
        }

        /* Y-Flip Subsystem: if the RT was rendered by a program whose VS had
         * Y-flip injection, the Metal texture already holds GL-bottom-origin
         * data.  No Y-flipped copy is needed — sampling consumers will use
         * the original via mglDecideYFlipForSampledRT.
         *
         * Defensive hardening: only release the copy when it is STALE
         * (version mismatch).  If a matching copy already exists, keep it —
         * a future sampler bind that defensively distrusts the authority
         * (e.g. after an IR binding remap or program-detection change) can
         * still use the copy instead of falling back to the un-flipped Metal
         * texture.  Releasing a matching copy saves a tiny amount of VRAM
         * but creates a fragile coupling: any change that makes the authority
         * wrong will immediately flip GUI items upside-down. */
        if (mglRTWriteAuthorityIsCurrentAndUsesOriginal(tex)) {
            if (tex->mtl_gl_sampled_data &&
                tex->mtl_gl_sampled_write_version != tex->mtl_render_target_write_version) {
                [self releaseGLSampledRenderTargetCopyForTexture:tex];
                if (mglTraceLogIsEnabled()) {
                    mglTraceLog("RT_SAMPLE_COPY_SKIP_INJECTED_RENDER tex=%u label=\"%s\" reason=render_yflip_injected_stale_released",
                                (unsigned)tex->name,
                                mglTraceTextureLabel(tex));
                }
            }
            continue;
        }

        [self updateGLSampledRenderTargetCopyForTexture:tex
                                                 source:source
                                                 reason:reason ? reason : "end_render_pass"];
    }
}

- (void) endRenderEncoding
{
    METAL_LOCK();
    [self endRenderEncodingLocked];
    METAL_UNLOCK();
}

- (void) endRenderEncodingLocked
{
    /* Invalidate last-bound render encoder state — the next encoder must
     * re-issue all binds rather than skipping them via the dedup fast path. */
    [self invalidateLastBoundState];

    if (_currentRenderEncoder)
    {
        Framebuffer *endedFramebuffer = _renderPassFramebuffer;
        GLsizei endedDrawBufferCount = _renderPassDrawBufferCount;
        GLenum endedDrawBuffers[MAX_COLOR_ATTACHMENTS];
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            endedDrawBuffers[i] = _renderPassDrawBuffers[i];
        }

        static uint64_t s_renderPassEndLogCount = 0;
        uint64_t hit = ++s_renderPassEndLogCount;
        if (hit <= 128ull || (hit % 1024ull) == 0ull) {
            mglLogRenderPassLifecycle("end",
                                      hit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
                                      _renderPassFramebufferName,
                                      _renderPassDrawBuffer,
                                      _renderPassDrawBufferCount);
        }
        @try {
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL DEBUG: Ending render encoder");
            }
            [_currentRenderEncoder endEncoding];
            _currentRenderEncoder = NULL;
            mglTraceFragmentTextureTraceBindings("CLEAR",
                                                 "end_render_encoding",
                                                 _fragmentTextureTraceBindings,
                                                 TEXTURE_UNITS,
                                                 ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                                 _pipelineProgramName);
            memset(_fragmentTextureTraceBindings, 0, sizeof(_fragmentTextureTraceBindings));
            _renderPassFramebuffer = NULL;
            _renderPassFramebufferName = 0;
            _renderPassDrawBuffer = 0;
            _renderPassDrawBufferCount = 0;
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                _renderPassDrawBuffers[i] = GL_NONE;
            }
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL DEBUG: Render encoder ended successfully");
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception ending render encoder: %@ - ignoring", exception.reason);
            // Force clear the encoder even if ending failed
            _currentRenderEncoder = NULL;
            mglTraceFragmentTextureTraceBindings("CLEAR",
                                                 "end_render_encoding_exception",
                                                 _fragmentTextureTraceBindings,
                                                 TEXTURE_UNITS,
                                                 ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                                 _pipelineProgramName);
            memset(_fragmentTextureTraceBindings, 0, sizeof(_fragmentTextureTraceBindings));
            _renderPassFramebuffer = NULL;
            _renderPassFramebufferName = 0;
            _renderPassDrawBuffer = 0;
            _renderPassDrawBufferCount = 0;
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                _renderPassDrawBuffers[i] = GL_NONE;
            }
        }

        if (endedFramebuffer) {
            [self updateGLSampledCopiesForEndedRenderPassFramebuffer:endedFramebuffer
                                                            drawCount:endedDrawBufferCount
                                                         drawBuffers:endedDrawBuffers
                                                              reason:"end_render_pass"];
        }
    }
}

- (BOOL)currentRenderPassUsesTexture:(id<MTLTexture>)texture
{
    if (!texture || !_currentRenderEncoder || !_renderPassDescriptor) {
        return NO;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (_renderPassDescriptor.colorAttachments[i].texture == texture) {
            return YES;
        }
    }
    if (_renderPassDescriptor.depthAttachment.texture == texture ||
        _renderPassDescriptor.stencilAttachment.texture == texture) {
        return YES;
    }

    return NO;
}

/*
 * synchronizeRenderPassForTextureReadback:reason: — heaviest synchronization boundary (GPU write visibility guarantee before CPU readback)
 *
 * Trigger condition: invoked when the texture to be read back is exactly the render target of the current render pass (color/depth/stencil attachment).
 * Guarantee semantics: endRenderEncoding closes the open render encoder → commitCommandBufferWithAGXRecovery:
 *           commits the current CB → waitUntilCompleted blocks until GPU completion → newCommandBuffer creates a new CB.
 *           Ensures that before CPU readback, all GPU rendering writes encoded to that texture have completed and are visible to the CPU.
 * Degradation: if the texture is not the current render target, returns YES directly (no sync needed); if the CB is already finalized, only rotates.
 */
- (BOOL)synchronizeRenderPassForTextureReadback:(id<MTLTexture>)texture
                                         reason:(const char *)reason
{
    BOOL usesTexture = [self currentRenderPassUsesTexture:texture];
    if (!usesTexture) {
        return YES;
    }

    [self endRenderEncoding];

    if (!_currentCommandBuffer) {
        BOOL ok = [self newCommandBuffer];
        return ok;
    }

    if (_currentCommandBuffer.status != MTLCommandBufferStatusNotEnqueued) {
        BOOL ok = [self newCommandBuffer];
        return ok;
    }

    id<MTLCommandBuffer> commandBufferToCommit = _currentCommandBuffer;
    _currentCommandBuffer = nil;

    @try {
        [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
        [commandBufferToCommit waitUntilCompleted];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: failed to synchronize render pass for texture readback (%s): %@",
              reason ? reason : "texture_readback",
              exception.reason);
        [self recordGPUError];
        [self newCommandBuffer];
        return NO;
    }

    if (commandBufferToCommit.error) {
        NSLog(@"MGL ERROR: render pass texture readback sync failed (%s): %@",
              reason ? reason : "texture_readback",
              commandBufferToCommit.error);
        [self recordGPUError];
        [self newCommandBuffer];
        return NO;
    }

    return [self newCommandBuffer];
}

// ULTIMATE FAILSAFE: Emergency Metal state reset to recover from corruption
- (void) emergencyResetMetalState
{
    NSLog(@"MGL CRITICAL: Performing emergency Metal state reset");

    @try {
        // Force cleanup of all Metal objects
        [self endRenderEncoding];

        _currentCommandBuffer = NULL;
        _currentRenderEncoder = NULL;
        _drawable = NULL;

        // Re-initialize basic Metal objects
        if (_device && _commandQueue) {
            NSLog(@"MGL CRITICAL: Re-creating Metal command buffer");
            _currentCommandBuffer = [_commandQueue commandBuffer];

            if (!_currentCommandBuffer) {
                NSLog(@"MGL CRITICAL: Failed to create new command buffer during recovery");
            }
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: Emergency Metal reset failed: %@", exception);
    }
}

- (bool) processGLState: (bool) draw_command
{
    METAL_LOCK();
    bool result = [self processGLStateLocked:draw_command];
    METAL_UNLOCK();
    return result;
}

- (bool) processGLStateLocked: (bool) draw_command
{
    static uint64_t s_processGLStateCallCount = 0;
    static double s_processGLStateLastCallTime = 0.0;
    static uint64_t s_processGLStateLastCallCount = 0;
    uint64_t processCall = ++s_processGLStateCallCount;
    double processStartSeconds = mglNowSeconds();
    bool traceProcess = mglShouldTraceCall(processCall);
    mglLogLoopHeartbeat("processGLState.loop",
                        processCall,
                        processStartSeconds,
                        &s_processGLStateLastCallTime,
                        &s_processGLStateLastCallCount,
                        0.25);
    if (traceProcess) {
        MGLTraceNSLog(@"MGL TRACE processGLState.begin call=%llu draw=%d",
              (unsigned long long)processCall, draw_command ? 1 : 0);
        mglLogStateSnapshot("processGLState.enter",
                            ctx,
                            _currentCommandBuffer,
                            _currentRenderEncoder,
                            _renderPassDescriptor,
                            _drawable);
    }
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in processGLState");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    if (draw_command) {
        /*
         * This flag is derived from the current draw's final fragment sampler
         * binding.  Clear it before any early render-state refresh so the
         * previous draw cannot disable culling while DIRTY_VAO/FBO is handled.
         */
        _currentDrawUsesRTSampledCopy = NO;
        MGL_FRAME_INC(g_mglProcessDrawCallsSinceSwap);
    }

    uintptr_t earlyCtxAddr = (uintptr_t)ctx;
    if (earlyCtxAddr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected: 0x%lx", earlyCtxAddr);
        return false;
    }

    // REMOVED: Thread synchronization was causing deadlocks
    // The issue is not thread contention but Metal object corruption

    // ULTIMATE FAILSAFE: Metal state corruption detection and recovery
    static int corruption_recovery_count = 0;
    static int max_recovery_attempts = 3;

    // Check for corrupted Metal objects that might cause crashes.
    // Only reject NULL / obviously invalid low addresses.
    if (!_device || !_commandQueue || ((uintptr_t)_device < 0x1000) || ((uintptr_t)_commandQueue < 0x1000)) {
        NSLog(@"MGL CRITICAL: Metal state corruption detected in processGLState!");
        NSLog(@"MGL CRITICAL: device=0x%lx, queue=0x%lx", (uintptr_t)_device, (uintptr_t)_commandQueue);

        if (corruption_recovery_count < max_recovery_attempts) {
            NSLog(@"MGL CRITICAL: Attempting Metal state recovery (%d/%d)", corruption_recovery_count + 1, max_recovery_attempts);

            // Force a complete Metal state reset
            @try {
                [self emergencyResetMetalState];
                corruption_recovery_count++;

                // Re-check after recovery
                if (!_device || !_commandQueue) {
                    NSLog(@"MGL CRITICAL: Metal recovery failed, aborting operation");
                    return false;
                }
            } @catch (NSException *exception) {
                NSLog(@"MGL CRITICAL: Metal recovery failed: %@", exception);
                return false;
            }
        } else {
            NSLog(@"MGL CRITICAL: Maximum recovery attempts exceeded, permanently disabling Metal operations");
            return false;
        }
    }

    //logDirtyBits(ctx);

    if (!draw_command) {
        [self endRenderPassIfFramebufferChangedForNonDraw:processCall];
    }

    // since a clear is embedded into a render encoder
    if (VAO() == NULL)
    {
        if (draw_command)
        {
            NSLog(@"Error: No VAO defined for ctx\n");

            // quietly return if we are not in a draw command with no vao defined
            // like a clear or init call
            return false;
        }

        // for a clear flush sequence...
        if (ctx->state.dirty_bits & DIRTY_STATE)
        {
            // end encoding on current render encoder
            [self endRenderEncodingLocked];

            // Use GPU throttling to prevent crashes when creating new render encoder
            if (![self validateMetalObjects]) {
                NSLog(@"MGL WARNING: GPU throttling active - deferring render encoder creation");
                ctx->state.dirty_bits &= ~DIRTY_STATE;
                return true;
            }

            @try {
                [self newRenderEncoderLocked];
            } @catch (NSException *exception) {
                NSLog(@"MGL ERROR: Render encoder creation failed: %@", exception);
            }

            // Clear the dirty bit to prevent repeated attempts
            ctx->state.dirty_bits &= ~DIRTY_STATE;
        }

        return true;
    }

    // only draw commands need a functioning render encoder
    // this can mess up a transition between compute and rendering on a flush
    // so just return
    // we may have to create a blank render encoder to safely run compute and
    // rendering correctly
    if (draw_command == false)
    {
        return true;
    }

    // MEMORY SAFETY: Validate context before use
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in processGLState");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    // Validate context pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t ctx_addr = (uintptr_t)ctx;
    if (ctx_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected: 0x%lx", ctx_addr);
        return false;
    }

    // Early circuit-breaker: if a program is currently quarantined due to repeated
    // vertex/fragment interface mismatch, skip draw before creating/rotating buffers.
    GLuint blockedProgramKey = mglCurrentRenderProgramKey(ctx);
    if (blockedProgramKey != 0u &&
        _interfaceMismatchBlockedProgram != 0 &&
        blockedProgramKey == _interfaceMismatchBlockedProgram)
    {
        CFTimeInterval now = CFAbsoluteTimeGetCurrent();
        if (now < _interfaceMismatchBlockedUntil) {
            static uint64_t s_quarantineSkipCount = 0;
            s_quarantineSkipCount++;
            if (s_quarantineSkipCount <= 16 || (s_quarantineSkipCount % 1000) == 0) {
                double remaining = _interfaceMismatchBlockedUntil - now;
                if (remaining < 0.0) remaining = 0.0;
                NSLog(@"MGL WARNING: Program %u quarantined due to interface mismatch (%.2fs remaining), skipping draw",
                      (unsigned)_interfaceMismatchBlockedProgram, remaining);
            }
            return false;
        }
    }

    // Keep command buffer lifecycle healthy: if the active one is already finalized,
    // rotate to a fresh buffer before any state processing.
    // Stage 5.3 Step 5: skip this in parallel-encode mode — the caller owns
    // the command buffer and sub-encoder lifecycle.
    if (!_parallelEncodeActive && _currentCommandBuffer && _currentRenderEncoder == NULL) {
        MTLCommandBufferStatus preStatus = _currentCommandBuffer.status;
        if (preStatus >= MTLCommandBufferStatusCommitted) {
            NSLog(@"MGL INFO: processGLState rotating finalized command buffer (status: %ld)", (long)preStatus);
            if (![self newCommandBufferLocked]) {
                NSLog(@"MGL ERROR: processGLState failed to create a fresh command buffer");
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.new_cb_rotate",
                                        ctx,
                                        _currentCommandBuffer,
                                        _currentRenderEncoder,
                                        _renderPassDescriptor,
                                        _drawable);
                }
                return false;
            }
        }
    } else if (!_currentCommandBuffer) {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: processGLState found NULL command buffer, creating one");
        }
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: processGLState could not create initial command buffer");
            if (traceProcess) {
                mglLogStateSnapshot("processGLState.fail.new_cb_initial",
                                    ctx,
                                    _currentCommandBuffer,
                                    _currentRenderEncoder,
                                    _renderPassDescriptor,
                                    _drawable);
            }
            return false;
        }
    }

    RETURN_FALSE_ON_FAILURE([self processDirtyStateDomainsLocked:draw_command]);

    // Ensure a render encoder exists for draw commands.
    // Stage 5.3 Step 5: skip nil-encoder recovery in parallel-encode mode —
    // the sub-encoder is managed by encodeBatchForParallelWorker.
    if (!_parallelEncodeActive && !_currentRenderEncoder) {
        static uint64_t s_nilEncoderRecoveryCount = 0;
        uint64_t nilHit = ++s_nilEncoderRecoveryCount;
        NSLog(@"MGL WARNING: processGLState - current render encoder is nil, attempting recovery hit=%llu",
              (unsigned long long)nilHit);
        if (nilHit <= 128ull || (nilHit % 512ull) == 0ull) {
            mglLogRenderPassLifecycle("nil-encoder-before-recovery",
                                      nilHit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
                                      _renderPassFramebufferName,
                                      _renderPassDrawBuffer,
                                      _renderPassDrawBufferCount);
        }
        RETURN_FALSE_ON_FAILURE([self newRenderEncoderLocked]);
        if (nilHit <= 128ull || (nilHit % 512ull) == 0ull) {
            mglLogRenderPassLifecycle("nil-encoder-after-recovery",
                                      nilHit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
                                      _renderPassFramebufferName,
                                      _renderPassDrawBuffer,
                                      _renderPassDrawBufferCount);
        }
    }

    if (draw_command) {
        /* Stage 5.3 Step 5: skip FBO-mismatch rebuild in parallel-encode
         * mode — encodeBatchForParallelWorker already syncs
         * _renderPassFramebuffer* ivars and clears DIRTY_FBO, so a mismatch
         * here would be a false positive that destroys the sub-encoder. */
        if (!_parallelEncodeActive) {
            RETURN_FALSE_ON_FAILURE([self ensureCurrentRenderPassMatchesFramebufferForDraw]);
        }
        [self updateCurrentRenderEncoder];
    }

    if (draw_command && kMGLVerbosePipelineLogs) {
        static uint64_t s_drawPipelineLookupCount = 0;
        s_drawPipelineLookupCount++;
        if (s_drawPipelineLookupCount <= 256ull || (s_drawPipelineLookupCount % 1000ull) == 0ull) {
            Program *lookupProgram = mglResolveProgramFromState(ctx);
            Program *lookupVertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            Program *lookupFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
            GLuint lookupProgramName = mglCurrentRenderProgramKey(ctx);
            Framebuffer *lookupFBO = ctx->state.framebuffer;
            GLuint lookupFBOName = lookupFBO ? lookupFBO->name : 0;
            fprintf(stderr, "MGL Draw current program key=%u mono=%p vs=%u fs=%u\n",
                    (unsigned)lookupProgramName,
                    (void *)lookupProgram,
                    lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
                    lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u);
            NSLog(@"MGL DRAW pipeline lookup result=%p key=%u vs=%u fs=%u vao=%p fbo=%u",
                  _pipelineState,
                  (unsigned)lookupProgramName,
                  lookupVertexProgram ? (unsigned)lookupVertexProgram->name : 0u,
                  lookupFragmentProgram ? (unsigned)lookupFragmentProgram->name : 0u,
                  ctx->state.vao,
                  (unsigned)lookupFBOName);
        }
    }

    if (!_pipelineState) {
        static uint64_t nil_pipeline_count = 0;
        nil_pipeline_count++;
        if (nil_pipeline_count <= 8 || (nil_pipeline_count % 1000) == 0) {
            MGLTraceNSLog(@"MGL DRAW SKIP: pipelineState is nil, forcing rebuild (occurrence=%llu)",
                          (unsigned long long)nil_pipeline_count);
        }
        // Force rebuild on next state processing pass.
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_pipeline",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    RETURN_FALSE_ON_FAILURE([self validateRenderPassAttachmentsAndPipelineFormatsLocked:traceProcess]);

    @try {
        if (!_lastBoundValid || _lastPipelineState != _pipelineState) {
            [_currentRenderEncoder setRenderPipelineState:_pipelineState];
            _lastPipelineState = _pipelineState;
            MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
        } else {
            MGL_PERF_INC(g_mglSetRenderPipelineStateSkipsSinceSwap);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: processGLState - setRenderPipelineState failed: %@", exception.reason);
        // Force pipeline/state retranslation on next draw instead of crashing this frame.
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.set_pipeline",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    // Resource Sync domain (Stage 3.4): stability rebind before draw. The logic was moved to
    // syncResourceBindingsForContext:, only the dispatch remains here.
    RETURN_FALSE_ON_FAILURE([self syncResourceBindingsForContext:ctx]);

    Program *fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    BOOL useFragCoordParams;
    if (_mslCacheEnabled && fragmentProgram && fragmentProgram->mslCacheValid) {
        useFragCoordParams = (fragmentProgram->usesFragCoordParams == GL_TRUE);
    } else {
        const char *fragmentMSL = fragmentProgram ? fragmentProgram->spirv[_FRAGMENT_SHADER].msl_str : NULL;
        useFragCoordParams = (fragmentMSL && strstr(fragmentMSL, kMGLFragCoordParamsMSLName));
    }
    if (useFragCoordParams) {
        NSUInteger passHeight = _renderPassDescriptor ? _renderPassDescriptor.renderTargetHeight : 0;
        if (passHeight == 0 && _renderPassDescriptor) {
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS && passHeight == 0; i++) {
                id<MTLTexture> color = _renderPassDescriptor.colorAttachments[i].texture;
                passHeight = color ? color.height : 0;
            }
            if (passHeight == 0 && _renderPassDescriptor.depthAttachment.texture) {
                passHeight = _renderPassDescriptor.depthAttachment.texture.height;
            }
            if (passHeight == 0 && _renderPassDescriptor.stencilAttachment.texture) {
                passHeight = _renderPassDescriptor.stencilAttachment.texture.height;
            }
        }

        vector_float4 fragCoordParams = {
            (float)passHeight,
            ctx->state.var.clip_origin == GL_LOWER_LEFT ? 1.0f : 0.0f,
            0.0f,
            0.0f
        };
        [_currentRenderEncoder setFragmentBytes:&fragCoordParams
                                         length:sizeof(fragCoordParams)
                                        atIndex:kMGLFragCoordParamsBufferIndex];
        [self invalidateLastBoundFragmentBufferAtIndex:kMGLFragCoordParamsBufferIndex];
    }

    if (draw_command &&
        mglFragmentTextureTraceBindingsUseRTSampledCopy(_fragmentTextureTraceBindings, TEXTURE_UNITS)) {
        _currentDrawUsesRTSampledCopy = YES;
        [self updateCurrentRenderEncoder];
    }

    double processElapsedMs = (mglNowSeconds() - processStartSeconds) * 1000.0;
    if (traceProcess) {
        MGLTraceNSLog(@"MGL TRACE processGLState.end call=%llu draw=%d elapsed=%.3fms",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedMs);
        mglLogStateSnapshot("processGLState.exit.ok",
                            ctx,
                            _currentCommandBuffer,
                            _currentRenderEncoder,
                            _renderPassDescriptor,
                            _drawable);
    } else if (processElapsedMs >= 25.0) {
        MGLTraceNSLog(@"MGL TRACE processGLState.slow call=%llu draw=%d elapsed=%.3fms",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedMs);
    }
    return true;
}
/*
 * Dirty state domain processing extracted from processGLStateLocked:.
 * Handles all dirty-bits dispatch: DIRTY_FBO, DIRTY_STATE, DIRTY_PROGRAM/
 * VAO/BUFFER_BASE_STATE, DIRTY_TEX, DIRTY_VAO/BUFFER/RENDER_STATE, and the
 * pipeline sync call. Returns false on failure (caller should skip this
 * draw), true on success.
 */
- (bool)processDirtyStateDomainsLocked:(bool)draw_command
{
    bool deferredBufferMapForPipelineBuild = false;
    if (ctx->state.dirty_bits)
    {
        // FBO binding/attachment changes alter the Metal render pass itself. They must
        // be handled even when no generic DIRTY_STATE bit is present; otherwise the
        // current render encoder can keep drawing into an old attachment while GL state
        // already points at a different FBO. RenderPass Sync domain (Stage 3.2).
        if (ctx->state.dirty_bits & DIRTY_FBO)
        {
            RETURN_FALSE_ON_FAILURE([self syncRenderPassStateForContext:ctx]);
        }

        // dirty state covers all rendering attachments and general state
        if (ctx->state.dirty_bits & DIRTY_STATE)
        {
            if (ctx->state.dirty_bits & DIRTY_FBO)
            {
                // MEMORY SAFETY: Add comprehensive validation to prevent use-after-free crashes
                Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyStateFBO");
                if (framebuffer)
                {
                    if (framebuffer->dirty_bits & DIRTY_FBO_BINDING)
                    {
                        RETURN_FALSE_ON_FAILURE([self bindFramebufferAttachmentTextures]);

                        // Additional validation after binding
                        framebuffer = mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyStateFBO.afterBind");
                        if (framebuffer) {
                            framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
                        }
                    }
                }

                // dirty FBO state can't be cleared just yet its needed below
            }

            ctx->state.dirty_bits &= ~DIRTY_STATE;
        }

        // check for dirty program and vao
        // leave program / vao state dirty, buffers need to be mapped before used below
        // dirty program causes buffers to be remapped
        // dirty vao causes attributes to be remapped to new buffers
        // dirty buffer base causes buffers to be remapped to new indexes
        if (ctx->state.dirty_bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_BUFFER_BASE_STATE))
        {
            // Avoid mapping draw buffers against a nil pipeline during startup/rebuild.
            // We'll map again after a valid pipeline is bound.
            bool deferBufferMapForNilPipeline =
                (draw_command &&
                 _pipelineState == nil &&
                 (ctx->state.dirty_bits & DIRTY_PROGRAM));

            if (deferBufferMapForNilPipeline) {
                deferredBufferMapForPipelineBuild = true;
                static uint64_t s_deferredMapCount = 0;
                s_deferredMapCount++;
                if (s_deferredMapCount <= 16 || (s_deferredMapCount % 1000ull) == 0ull) {
                    MGLTraceNSLog(@"MGL DRAW SKIP: pipelineState is nil (deferring buffer mapping, occurrence=%llu)",
                                  (unsigned long long)s_deferredMapCount);
                }
            } else {
                // programs are now compiled before execution, we shouldn't get here
                //assert(ctx->state.program->mtl_data); //

                // figure out vertex shader uniforms / buffer mappings
                RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
            }

            ctx->state.dirty_bits &= ~DIRTY_BUFFER_BASE_STATE;
        }

        // Texture object uploads can be prepared before pipeline selection, but
        // sampled-resource binding must wait until after setRenderPipelineState()
        // so it uses the current program's sampler reflection.
        if (ctx->state.dirty_bits & (DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER))
        {
            RETURN_FALSE_ON_FAILURE([self bindActiveTexturesToMTL]);

            // textures / active textures and samplers are all handled in bindActiveTexturesToMTL
            ctx->state.dirty_bits &= ~(DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER);
        }

        // A dirty VAO changes vertex buffer bindings and may require a new
        // pipeline descriptor, but it does not change the render-pass
        // attachments. Keep the current encoder alive so GL draw ordering and
        // depth/load-store continuity are preserved across HUD/hand/UI passes.
        if (ctx->state.dirty_bits & DIRTY_VAO)
        {
            // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list]);
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list]);

            if (!_currentRenderEncoder) {
                RETURN_FALSE_ON_FAILURE([self newRenderEncoderLocked]);
            }

            [self updateCurrentRenderEncoder];

            // clear dirty render state
            ctx->state.dirty_bits &= ~DIRTY_RENDER_STATE;
        }
        else if (ctx->state.dirty_bits & DIRTY_BUFFER)
        {
            // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list]);
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list]);

            ctx->state.dirty_bits &= ~DIRTY_BUFFER;
        }
        else if (ctx->state.dirty_bits & DIRTY_RENDER_STATE)
        {
            if (_currentRenderEncoder == NULL)
            {
                RETURN_FALSE_ON_FAILURE([self newRenderEncoderLocked]);
            }

            // a dirty render state may just be something like alpha changes which don't require a new renderbuffer

            // updateCurrentRenderEncoder will update the renderstate outside of creating a new one
            [self updateCurrentRenderEncoder];

            ctx->state.dirty_bits &= ~DIRTY_RENDER_STATE;
        }

        // new pipeline / vertex / renderbuffer and pipelinestate descriptor, should probably make this a single dirty bit
        // Pipeline Sync domain (Stage 3.3): when program/VAO/FBO/alpha/render-state changes,
        // rebuild or reuse the PSO. The logic was moved entirely to syncPipelineStateWithDeferredBufferMap:,
        // only the dispatch remains here; deferredBufferMap is passed as a value parameter (not read after the block).
        if (ctx->state.dirty_bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE))
        {
            RETURN_FALSE_ON_FAILURE([self syncPipelineStateWithDeferredBufferMap:deferredBufferMapForPipelineBuild]);
        }

        //if (ctx->state.dirty_bits)
        //    logDirtyBits(ctx);

        // Unconditionally clear all dirty bits after processing.
        // All relevant state has been applied to Metal encoders above; any
        // remaining bits (e.g. DIRTY_DRAWABLE set at init, or bits accumulated
        // via |= in the defer path without DIRTY_ALL_BIT) are stale and would
        // cause false-positive rebinds on the next draw.
        ctx->state.dirty_bits = 0;
    }
    else // if (ctx->state.dirty_bits)
    {
        // buffer data can be changed but the bindings remain in place.. so we need to update the data if this is the case
        // like a uniform or buffer sub data call
        
        if( [self checkForDirtyBufferData: &ctx->state.vertex_buffer_map_list])
        {
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list]);

            RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder]);
        }
        
        if( [self checkForDirtyBufferData: &ctx->state.fragment_buffer_map_list])
        {
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list]);

            RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder]);
        }
    }
    return true;
}

/*
 * Render pass descriptor and pipeline format validation extracted from
 * processGLStateLocked:. Validates render-pass attachments and checks
 * pipeline/pass color, depth, and stencil format compatibility. Returns
 * false to skip the draw on validation failure, true to continue.
 */
- (bool)validateRenderPassAttachmentsAndPipelineFormatsLocked:(BOOL)traceProcess
{
    // Guard against invalid render pass state before binding pipeline.
    // Metal debug validation can abort the process if the encoder/render pass is incompatible.
    if (!_renderPassDescriptor) {
        NSLog(@"MGL ERROR: processGLState - renderPassDescriptor is nil before pipeline bind");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_rpd",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }
    BOOL passHasAnyAttachment = NO;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id<MTLTexture> colorAttachment = _renderPassDescriptor.colorAttachments[i].texture;
        if (colorAttachment) {
            passHasAnyAttachment = YES;
            if ((colorAttachment.usage & MTLTextureUsageRenderTarget) == 0) {
                NSLog(@"MGL WARNING: processGLState - color attachment %d missing RenderTarget usage (usage=0x%lx); skipping draw",
                      i,
                      (unsigned long)colorAttachment.usage);
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.color_usage",
                                        ctx,
                                        _currentCommandBuffer,
                                        _currentRenderEncoder,
                                        _renderPassDescriptor,
                                        _drawable);
                }
                return false;
            }
        }
    }
    if (_renderPassDescriptor.depthAttachment.texture ||
        _renderPassDescriptor.stencilAttachment.texture) {
        passHasAnyAttachment = YES;
    }

    if (!passHasAnyAttachment) {
        NSLog(@"MGL WARNING: processGLState - render pass has no attachments, skipping draw to avoid Metal assert");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.no_attachments",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    MTLPixelFormat currentColor0Format = MTLPixelFormatInvalid;
    MTLPixelFormat currentDepthFormat = MTLPixelFormatInvalid;
    MTLPixelFormat currentStencilFormat = MTLPixelFormatInvalid;

    id<MTLTexture> rpColor0 = _renderPassDescriptor.colorAttachments[0].texture;
    id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
    id<MTLTexture> rpStencil = _renderPassDescriptor.stencilAttachment.texture;
    if (rpColor0) {
        currentColor0Format = rpColor0.pixelFormat;
    }
    if (rpDepth) {
        currentDepthFormat = rpDepth.pixelFormat;
    }
    if (rpStencil) {
        currentStencilFormat = rpStencil.pixelFormat;
    }

    // IMPORTANT:
    // Never mutate depth/stencil attachments here to "fit" an existing pipeline.
    // The active Metal render encoder was already created with a render-pass descriptor,
    // and changing attachments after encoder creation does not make that encoder compatible.
    // We must instead reject mismatched pipeline/pass combinations and rebuild safely.

    if (_pipelineColor0Format != MTLPixelFormatInvalid &&
        currentColor0Format != MTLPixelFormatInvalid &&
        _pipelineColor0Format != currentColor0Format) {
        static uint64_t s_colorFormatMismatchCount = 0;
        s_colorFormatMismatchCount++;
	        if (s_colorFormatMismatchCount <= 16 || (s_colorFormatMismatchCount % 250) == 0) {
	            NSLog(@"MGL WARNING: Pipeline/pass color format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	                  (unsigned long)_pipelineColor0Format, (unsigned long)currentColor0Format);
	        }
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass color format mismatch"];
	        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }

    if (_pipelineDepthFormat != currentDepthFormat) {
        BOOL pipelineHasDepth = (_pipelineDepthFormat != MTLPixelFormatInvalid);
        BOOL passHasDepth = (currentDepthFormat != MTLPixelFormatInvalid);
        if (!pipelineHasDepth && !passHasDepth) {
            goto depth_format_ok;
	        }
	        NSLog(@"MGL WARNING: Pipeline/pass depth format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	              (unsigned long)_pipelineDepthFormat, (unsigned long)currentDepthFormat);
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass depth format mismatch"];
	        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }
depth_format_ok:;

    if (_pipelineStencilFormat != currentStencilFormat) {
        BOOL pipelineHasStencil = (_pipelineStencilFormat != MTLPixelFormatInvalid);
        BOOL passHasStencil = (currentStencilFormat != MTLPixelFormatInvalid);
        if (!pipelineHasStencil && !passHasStencil) {
            goto stencil_format_ok;
	        }
	        NSLog(@"MGL WARNING: Pipeline/pass stencil format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
	              (unsigned long)_pipelineStencilFormat, (unsigned long)currentStencilFormat);
	        [self invalidateCurrentPipelineStateForReason:@"pipeline/pass stencil format mismatch"];
	        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
	        return false;
	    }
stencil_format_ok:;
    return true;
}


/*
 * Pipeline Sync domain (Stage 3.3). PSO build/reuse logic moved verbatim from processGLStateLocked:
 * generates pipeline+vertex descriptor, queries/builds PSO cache, interface-mismatch
 * circuit breaker, failure fallback chain. Only operates on Metal pipeline state, state is read via ctx (same as before the move).
 * deferredBufferMap is passed in by the caller (deferred buffer mapping flag for nil pipeline).
 * Returns false to indicate this draw should be skipped (equivalent to the original inline return false semantics).
 */
- (bool)syncPipelineStateWithDeferredBufferMap:(bool)deferredBufferMapForPipelineBuild
{
            GLMState *state = MGL_STATE(ctx);
            /* Force a rebind of the pipeline state on the next setRenderPipelineState
             * call. Dirty program/VAO/FBO/render-state may rebuild or reuse the
             * pipeline, but the encoder still needs the binding re-issued.
             *
             * Task 5 gated fast path: when MGL_PSO_DEDUP is enabled (default ON)
             * and the render
             * encoder is unchanged (_lastBoundValid == YES) and the resolved
             * pipeline state pointer is identical to the previously bound
             * state (_pipelineState == _lastPipelineState), the nil assignment
             * is skipped. This allows the dedup check in
             * processGLStateLocked:'s setRenderPipelineState: path to
             * recognize the encoder already has the correct PSO bound and
             * skip the redundant MTL call. If any condition is false, the
             * original conservative nil assignment executes. */
            if (_psoDedupEnabled && _lastBoundValid && (_pipelineState == _lastPipelineState)) {
                MGL_PERF_INC(g_mglPSODedupHitsSinceSwap);
            } else {
                _lastPipelineState = nil;
                MGL_PERF_INC(g_mglPSODedupMissesSinceSwap);
            }
            static CFTimeInterval s_pipelineRetryAfter = 0.0;
            static CFTimeInterval s_interfaceMismatchRetryAfter = 0.0;
            static GLuint s_interfaceMismatchProgramName = 0;
            static MTLPixelFormat s_interfaceMismatchColor0Format = MTLPixelFormatInvalid;
            static MTLPixelFormat s_interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
            static MTLPixelFormat s_interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
            static uint32_t s_interfaceMismatchStreak = 0;
            static GLuint s_programMismatchProgramName = 0;
            static CFTimeInterval s_programMismatchRetryAfter = 0.0;
            static uint32_t s_programMismatchStreak = 0;
            CFTimeInterval now = CFAbsoluteTimeGetCurrent();
            bool skipPipelineBuild = false;
            Program *currentProgram = mglResolveProgramFromState(ctx);
            Program *currentVertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            Program *currentFragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
            GLuint currentProgramName = mglCurrentRenderProgramKey(ctx);
            VertexArray *currentVAO = state->vao;
            Framebuffer *currentFBO = mglRendererGetValidatedFramebuffer(ctx, "processGLState.currentFBO");
            GLuint currentFBOName = currentFBO ? currentFBO->name : 0;

            // Program-level breaker (independent of render-pass signature) to avoid
            // mismatch storms where color/depth/stencil signatures keep changing.
            if (_pipelineState != nil &&
                currentProgramName != 0 &&
                currentProgramName == s_programMismatchProgramName &&
                now < s_programMismatchRetryAfter) {
                static uint64_t s_programMismatchSkipCount = 0;
                s_programMismatchSkipCount++;
                if (s_programMismatchSkipCount <= 16 || (s_programMismatchSkipCount % 1000ull) == 0ull) {
                    double remaining = s_programMismatchRetryAfter - now;
                    if (remaining < 0.0) remaining = 0.0;
                    NSLog(@"MGL WARNING: Program-level mismatch breaker active (program=%u, %.2fs remaining), skipping draw",
                          (unsigned)currentProgramName,
                          remaining);
                }
                state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

	            if (now < s_pipelineRetryAfter) {
	                BOOL retryAppliesToCurrentProgram =
	                    (currentProgramName != 0 &&
	                     (currentProgramName == s_interfaceMismatchProgramName ||
	                      currentProgramName == s_programMismatchProgramName ||
	                      currentProgramName == _interfaceMismatchBlockedProgram));

	                if (retryAppliesToCurrentProgram) {
	                    if (_pipelineState) {
		                    state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	                    // Keep existing pipeline, but do not early-return before setRenderPipelineState.
		                    skipPipelineBuild = true;
	                    } else {
	                        s_pipelineRetryAfter = 0.0;
	                        s_programMismatchRetryAfter = 0.0;
	                        s_interfaceMismatchRetryAfter = 0.0;
	                    }
		                } else {
	                    static uint64_t s_retryBypassCount = 0;
	                    s_retryBypassCount++;
	                    if (s_retryBypassCount <= 16 || (s_retryBypassCount % 1000ull) == 0ull) {
	                        NSLog(@"MGL PIPELINE RETRY bypass global retry for unrelated program=%u mismatchProgram=%u blockedProgram=%u",
	                              (unsigned)currentProgramName,
	                              (unsigned)s_interfaceMismatchProgramName,
	                              (unsigned)_interfaceMismatchBlockedProgram);
	                    }
	                }
	            }

            if (!skipPipelineBuild) {
            // create pipeline descriptor
            MTLRenderPipelineDescriptor *pipelineStateDescriptor;

	            pipelineStateDescriptor = [self generatePipelineDescriptor];
	            if (!pipelineStateDescriptor) {
	                NSLog(@"MGL PIPELINE CREATE fail error=generatePipelineDescriptor returned nil");
	                [self invalidateCurrentPipelineStateForReason:@"pipeline descriptor failure"];
	                s_pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
                state->dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
                return false;
            }

            MTLPixelFormat builtColor0Format = pipelineStateDescriptor.colorAttachments[0].pixelFormat;
            MTLPixelFormat builtDepthFormat = pipelineStateDescriptor.depthAttachmentPixelFormat;
            MTLPixelFormat builtStencilFormat = pipelineStateDescriptor.stencilAttachmentPixelFormat;

            // Circuit breaker for repeated VS/FS interface mismatch.
            if (now < s_interfaceMismatchRetryAfter &&
                currentProgramName == s_interfaceMismatchProgramName &&
                builtColor0Format == s_interfaceMismatchColor0Format &&
                builtDepthFormat == s_interfaceMismatchDepthFormat &&
                builtStencilFormat == s_interfaceMismatchStencilFormat) {
                state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

            // create vertex descriptor
            MTLVertexDescriptor *vertexDescriptor;

            vertexDescriptor = [self generateVertexDescriptor];
	            if (!vertexDescriptor) {
	                NSLog(@"MGL PIPELINE CREATE fail error=generateVertexDescriptor returned nil");
	                [self invalidateCurrentPipelineStateForReason:@"vertex descriptor failure"];
	                s_pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
                state->dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
                return false;
            }

            [self updateBlendStateCache];
            state->dirty_bits &= ~DIRTY_ALPHA_STATE;
            [self bindBlendStateToPipelineStateDescriptor:pipelineStateDescriptor];

            if (kMGLVerbosePipelineLogs) {
                MTLRenderPipelineColorAttachmentDescriptor *ca0 = pipelineStateDescriptor.colorAttachments[0];
                NSLog(@"MGL PIPELINE DESC c0 state program=%u fmt=%lu writeMask=0x%x blend=%d srcRGB=%lu dstRGB=%lu srcA=%lu dstA=%lu opRGB=%lu opA=%lu",
                      (unsigned)currentProgramName,
                      (unsigned long)ca0.pixelFormat,
                      (unsigned)ca0.writeMask,
                      ca0.blendingEnabled ? 1 : 0,
                      (unsigned long)ca0.sourceRGBBlendFactor,
                      (unsigned long)ca0.destinationRGBBlendFactor,
                      (unsigned long)ca0.sourceAlphaBlendFactor,
                      (unsigned long)ca0.destinationAlphaBlendFactor,
                      (unsigned long)ca0.rgbBlendOperation,
                      (unsigned long)ca0.alphaBlendOperation);
            }

	            pipelineStateDescriptor.vertexDescriptor = vertexDescriptor;
	            NSString *pipelineCacheKey = nil;
	            bool pipelineResolvedFromCache = false;

	            if (!pipelineResolvedFromCache && _pipelineStateCache && currentProgramName != 0) {
	                uint64_t pipelineSig = mglPipelineDescriptorSignature(pipelineStateDescriptor);
                uint64_t vertexSig = mglVertexDescriptorSignature(vertexDescriptor);
	                pipelineCacheKey = [NSString stringWithFormat:@"%u:%x:%x:%016llx:%016llx",
		                                    (unsigned)currentProgramName,
		                                    (unsigned)state->var.clip_origin,
		                                    (unsigned)state->var.clip_depth_mode,
		                                    (unsigned long long)pipelineSig,
		                                    (unsigned long long)vertexSig];
	                id<MTLRenderPipelineState> cachedPipeline = [_pipelineStateCache objectForKey:pipelineCacheKey];
	                if (cachedPipeline) {
	                    static uint64_t s_pipelineCacheHitCount = 0;
                    s_pipelineCacheHitCount++;
                    MGL_PERF_INC(g_mglPipelineCacheHitsSinceSwap);
	                    if (kMGLVerbosePipelineLogs &&
                            (s_pipelineCacheHitCount <= 128ull || (s_pipelineCacheHitCount % 1000ull) == 0ull)) {
	                        NSLog(@"MGL PIPELINE CACHE hit program=%u vao=%p fbo=%u key=%@",
	                              (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName, pipelineCacheKey);
	                    }

	                    _pipelineState = cachedPipeline;
	                    pipelineResolvedFromCache = true;
	                    _pipelineColor0Format = builtColor0Format;
	                    _pipelineDepthFormat = builtDepthFormat;
	                    _pipelineStencilFormat = builtStencilFormat;
	                    _pipelineProgramName = currentProgramName;

	                    // Mirror successful compile-side breaker resets.
	                    s_interfaceMismatchStreak = 0;
	                    s_interfaceMismatchProgramName = 0;
	                    s_interfaceMismatchColor0Format = MTLPixelFormatInvalid;
	                    s_interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
	                    s_interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
	                    s_interfaceMismatchRetryAfter = 0.0;
	                    if (s_programMismatchProgramName == currentProgramName) {
	                        s_programMismatchProgramName = 0;
	                        s_programMismatchRetryAfter = 0.0;
	                        s_programMismatchStreak = 0u;
	                    }
	                    if (_interfaceMismatchBlockedProgram == currentProgramName) {
	                        _interfaceMismatchBlockedProgram = 0;
	                        _interfaceMismatchBlockedUntil = 0.0;
	                        _interfaceMismatchBlockedStreak = 0u;
	                    }
	                }
	            }

	            // PROPER AGX VIRTUALIZATION COMPATIBILITY: Fix root cause while maintaining Metal functionality
	            if (!pipelineResolvedFromCache) {
            MGL_PERF_INC(g_mglPipelineCacheMissesSinceSwap);
            NSError *error;
	            id<MTLRenderPipelineState> previousPipelineState = _pipelineState;
	            bool pipelineReusedPrevious = false;

            @try {
                static uint64_t s_pipelineCreateBeginCount = 0;
                s_pipelineCreateBeginCount++;
                if (kMGLVerbosePipelineLogs &&
                    (s_pipelineCreateBeginCount <= 128ull || (s_pipelineCreateBeginCount % 500ull) == 0ull)) {
                    NSLog(@"MGL PIPELINE CREATE begin program=%u vao=%p fbo=%u",
                          (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName);
                }

                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL INFO: Creating Metal pipeline state with AGX virtualization compatibility...");
                }

                // ROOT CAUSE FIX: The issue is with async shader compilation in virtualized environments
                // Force synchronous pipeline creation to avoid completion queue crashes
                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL INFO: Using synchronous pipeline creation to prevent virtualization crashes");
                }

                // PROPER FIX: Disable async compilation that causes completion queue crashes
                if (kMGLVerbosePipelineLogs &&
                    MGLCapabilityHasBug(&_capability, MGL_BUG_ASYNC_SHADER_COMPILE_IN_VM)) {
                    NSLog(@"MGL INFO: AGX virtualization detected - using safe synchronous compilation");
                }

                _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];

                if (!_pipelineState) {
                    NSLog(@"MGL PIPELINE CREATE fail error=%@", error);
                    NSLog(@"MGL ERROR: Pipeline creation failed: %@", error);

                    NSString *errDesc = error.localizedDescription ?: @"";
                    NSString *errDomain = error.domain ?: @"";
                    BOOL isInterfaceMismatch = ((error.code == 3 && [errDomain hasPrefix:@"AGXMetal"]) ||
                                                [errDesc containsString:@"mismatching vertex shader output"] ||
                                                [errDesc containsString:@"not written by vertex shader"]);

	                    if (isInterfaceMismatch) {
	                        mglWriteProgramMSLDump(currentVertexProgram, errDesc);
	                        if (currentFragmentProgram && currentFragmentProgram != currentVertexProgram) {
	                            mglWriteProgramMSLDump(currentFragmentProgram, errDesc);
	                        } else if (!currentVertexProgram) {
	                            mglWriteProgramMSLDump(currentProgram, errDesc);
	                        }
	                        BOOL sameProgram = (_pipelineProgramName != 0 && _pipelineProgramName == currentProgramName);
                        BOOL colorCompatible = (_pipelineColor0Format == MTLPixelFormatInvalid ||
                                                builtColor0Format == MTLPixelFormatInvalid ||
                                                _pipelineColor0Format == builtColor0Format);
                        BOOL depthCompatible = (_pipelineDepthFormat == MTLPixelFormatInvalid ||
                                                builtDepthFormat == MTLPixelFormatInvalid ||
                                                _pipelineDepthFormat == builtDepthFormat);
                        BOOL stencilCompatible = (_pipelineStencilFormat == MTLPixelFormatInvalid ||
                                                  builtStencilFormat == MTLPixelFormatInvalid ||
                                                  _pipelineStencilFormat == builtStencilFormat);

                        if (previousPipelineState && sameProgram && colorCompatible && depthCompatible && stencilCompatible) {
                            NSLog(@"MGL WARNING: Interface mismatch for program %u; reusing previous compatible pipeline once",
                                  (unsigned)currentProgramName);
                            _pipelineState = previousPipelineState;
                            pipelineReusedPrevious = true;
                        } else {
                            BOOL sameMismatchSignature =
                                (currentProgramName == s_interfaceMismatchProgramName &&
                                 builtColor0Format == s_interfaceMismatchColor0Format &&
                                 builtDepthFormat == s_interfaceMismatchDepthFormat &&
                                 builtStencilFormat == s_interfaceMismatchStencilFormat);
                            if (sameMismatchSignature) {
                                if (s_interfaceMismatchStreak < UINT32_MAX) {
                                    s_interfaceMismatchStreak++;
                                }
                            } else {
                                s_interfaceMismatchStreak = 1;
                                s_interfaceMismatchProgramName = currentProgramName;
                                s_interfaceMismatchColor0Format = builtColor0Format;
                                s_interfaceMismatchDepthFormat = builtDepthFormat;
                                s_interfaceMismatchStencilFormat = builtStencilFormat;
                            }

                            // Exponential backoff: 0.10, 0.20, 0.40, 0.80, 1.60, capped at 2.00 sec.
                            uint32_t cappedShift = (s_interfaceMismatchStreak > 5u) ? 4u : (s_interfaceMismatchStreak - 1u);
                            double retryDelay = 0.10 * (double)(1u << cappedShift);
                            if (retryDelay > 2.0) {
                                retryDelay = 2.0;
                            }
                            s_interfaceMismatchRetryAfter = now + retryDelay;

                            if (s_interfaceMismatchStreak <= 5u || (s_interfaceMismatchStreak % 200u) == 0u) {
                                NSLog(@"MGL WARNING: Interface mismatch (program=%u, streak=%u), throttling retries for %.2fs",
                                      (unsigned)currentProgramName,
                                      (unsigned)s_interfaceMismatchStreak,
                                      retryDelay);
                            }

                            // Program-level breaker update (ignores attachment signature).
                            if (s_programMismatchProgramName == currentProgramName) {
                                if (s_programMismatchStreak < UINT32_MAX) {
                                    s_programMismatchStreak++;
                                }
                            } else {
                                s_programMismatchProgramName = currentProgramName;
                                s_programMismatchStreak = 1u;
                            }
                            double programDelay = 0.25 * (double)(1u << ((s_programMismatchStreak > 6u) ? 6u : (s_programMismatchStreak - 1u)));
                            if (programDelay > 20.0) {
                                programDelay = 20.0;
                            }
                            s_programMismatchRetryAfter = now + programDelay;
                            if (s_programMismatchStreak <= 8u || (s_programMismatchStreak % 64u) == 0u) {
                                NSLog(@"MGL WARNING: Program %u mismatch breaker set for %.2fs (streak=%u)",
                                      (unsigned)currentProgramName,
                                      programDelay,
                                      (unsigned)s_programMismatchStreak);
                            }

                            // Global quarantine for this program to prevent command-buffer storm.
                            if (_interfaceMismatchBlockedProgram == currentProgramName) {
                                if (_interfaceMismatchBlockedStreak < UINT32_MAX) {
                                    _interfaceMismatchBlockedStreak++;
                                }
                            } else {
                                _interfaceMismatchBlockedProgram = currentProgramName;
                                _interfaceMismatchBlockedStreak = 1u;
                            }
                            // Use a stronger quarantine window than compile retry backoff.
                            // This prevents pathological draw loops from repeatedly re-entering
                            // pipeline compilation and overwhelming AGX command submission.
                            double quarantineDelay = retryDelay * 8.0;
                            if (quarantineDelay < 1.00) quarantineDelay = 1.00;
                            if (quarantineDelay > 15.00) quarantineDelay = 15.00;
                            _interfaceMismatchBlockedUntil = now + quarantineDelay;
                            if (_interfaceMismatchBlockedStreak <= 6u || (_interfaceMismatchBlockedStreak % 64u) == 0u) {
                                NSLog(@"MGL WARNING: Program %u quarantined for %.2fs after interface mismatch (streak=%u)",
                                      (unsigned)currentProgramName,
                                      quarantineDelay,
                                      (unsigned)_interfaceMismatchBlockedStreak);
                            }

	                            [self invalidateCurrentPipelineStateForReason:@"interface mismatch pipeline failure"];
	                            s_pipelineRetryAfter = (_interfaceMismatchBlockedUntil > s_interfaceMismatchRetryAfter)
	                                ? _interfaceMismatchBlockedUntil
	                                : s_interfaceMismatchRetryAfter;
                            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                            return false;
                        }
                    }

		                    if (!skipPipelineBuild && !_pipelineState) {
		                        [self invalidateCurrentPipelineStateForReason:@"pipeline creation failure"];
	                        // Avoid destructive global recovery during shader/pipeline compile errors.
	                        // These are usually content/interface issues, not GPU-state corruption.

                        // AGX VIRTUALIZATION FALLBACK: Try with minimal descriptor
                        @try {
                            NSLog(@"MGL INFO: VIRTUALIZED AGX - Trying simplified compilation fallback...");

                            // Simplify the descriptor to avoid complex shader compilation issues
                            MTLRenderPipelineDescriptor *simpleDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
                            simpleDescriptor.colorAttachments[0].pixelFormat = pipelineStateDescriptor.colorAttachments[0].pixelFormat;
                            simpleDescriptor.depthAttachmentPixelFormat = pipelineStateDescriptor.depthAttachmentPixelFormat;
                            simpleDescriptor.stencilAttachmentPixelFormat = pipelineStateDescriptor.stencilAttachmentPixelFormat;
                            simpleDescriptor.vertexDescriptor = pipelineStateDescriptor.vertexDescriptor;
                            simpleDescriptor.vertexFunction = pipelineStateDescriptor.vertexFunction;
                            simpleDescriptor.fragmentFunction = pipelineStateDescriptor.fragmentFunction;
                            simpleDescriptor.rasterizationEnabled = pipelineStateDescriptor.rasterizationEnabled;
                            mglNormalizePipelineDepthStencilFormats(simpleDescriptor, "simple-fallback");
                            mglEnableIndirectCommandBuffersForPipeline(simpleDescriptor);

                            _pipelineState = [_device newRenderPipelineStateWithDescriptor:simpleDescriptor error:&error];
                            if (_pipelineState) {
                                builtColor0Format = simpleDescriptor.colorAttachments[0].pixelFormat;
                                builtDepthFormat = simpleDescriptor.depthAttachmentPixelFormat;
                                builtStencilFormat = simpleDescriptor.stencilAttachmentPixelFormat;
                            }
                        } @catch (NSException *innerException) {
                            NSLog(@"MGL ERROR: VIRTUALIZED AGX - Simplified compilation also failed: %@", innerException);
                        }
                    }
                }

            } @catch (NSException *exception) {
                NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Metal pipeline creation crashed: %@", exception);
                NSLog(@"MGL CRITICAL: Exception name: %@", [exception name]);
                NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);

                // VIRTUALIZED AGX ULTIMATE FALLBACK: Create minimal safe pipeline
                NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating ultimate fallback pipeline for virtualization safety");

                @try {
                    MTLRenderPipelineDescriptor *safeDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
                    MTLPixelFormat safeColor0Format = pipelineStateDescriptor.colorAttachments[0].pixelFormat;
                    if (_renderPassDescriptor && _renderPassDescriptor.colorAttachments[0].texture) {
                        safeColor0Format = _renderPassDescriptor.colorAttachments[0].texture.pixelFormat;
                    } else if (_drawable && _drawable.texture) {
                        safeColor0Format = _drawable.texture.pixelFormat;
                    }
                    if (safeColor0Format == MTLPixelFormatInvalid) {
                        safeColor0Format = MTLPixelFormatBGRA8Unorm;
                    }
                    safeDescriptor.colorAttachments[0].pixelFormat = safeColor0Format;
                    safeDescriptor.depthAttachmentPixelFormat = pipelineStateDescriptor.depthAttachmentPixelFormat;
                    safeDescriptor.stencilAttachmentPixelFormat = pipelineStateDescriptor.stencilAttachmentPixelFormat;
                    safeDescriptor.colorAttachments[0].blendingEnabled = NO;
                    mglNormalizePipelineDepthStencilFormats(safeDescriptor, "safe-fallback");
                    mglEnableIndirectCommandBuffersForPipeline(safeDescriptor);

                    // Use hardcoded minimal shaders that are guaranteed to work in virtualization
                    NSString *safeVertexShader = @"#include <metal_stdlib>\nusing namespace metal;\nvertex float4 main(uint vid [[vertex_id]]) { return float4(0.0, 0.0, 0.0, 1.0); }";
                    NSString *safeFragmentShader = @"#include <metal_stdlib>\nusing namespace metal;\nfragment float4 main() { return float4(0.0, 0.0, 0.0, 1.0); }";

                    NSError *libraryError;
                    id<MTLLibrary> vertLibrary = [self newMetalLibraryWithSource:safeVertexShader
                                                                          options:nil
                                                                            label:@"MGL safe vertex fallback"
                                                                            error:&libraryError];
                    id<MTLLibrary> fragLibrary = [self newMetalLibraryWithSource:safeFragmentShader
                                                                          options:nil
                                                                            label:@"MGL safe fragment fallback"
                                                                            error:&libraryError];

                    if (vertLibrary && fragLibrary) {
                        safeDescriptor.vertexFunction = [vertLibrary newFunctionWithName:@"main"];
                        safeDescriptor.fragmentFunction = [fragLibrary newFunctionWithName:@"main"];

                        _pipelineState = [_device newRenderPipelineStateWithDescriptor:safeDescriptor error:&error];
                        if (_pipelineState) {
                            builtColor0Format = safeDescriptor.colorAttachments[0].pixelFormat;
                            builtDepthFormat = safeDescriptor.depthAttachmentPixelFormat;
                            builtStencilFormat = safeDescriptor.stencilAttachmentPixelFormat;
                            NSLog(@"MGL INFO: VIRTUALIZED AGX - Safe fallback pipeline created successfully");
                        }
                    }
                } @catch (NSException *fallbackException) {
                    NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Even fallback pipeline failed: %@", fallbackException);
                }

	                if (!_pipelineState) {
	                    NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - All pipeline creation attempts failed, disabling rendering");
	                    [self invalidateCurrentPipelineStateForReason:@"all pipeline fallbacks failed"];
	                    s_pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
	                    state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	                    return false;
                }
            }

            // Pipeline State creation could fail if the pipeline descriptor isn't set up properly.
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode.)
		            if (!_pipelineState) {
		                NSLog(@"MGL ERROR: Failed to create pipeline state: %@", error);
		                NSLog(@"MGL WARNING: Skipping draw for this pipeline build failure; will retry later");
	                [self invalidateCurrentPipelineStateForReason:@"pipeline state is nil after creation"];
	                s_pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
	                state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	                return false;
            } else {
                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL PIPELINE CREATE success pipeline=%p", _pipelineState);
                    NSLog(@"MGL INFO: Pipeline state created successfully");
                }
                // Clear interface-mismatch breaker after a successful compile path.
                s_interfaceMismatchStreak = 0;
                s_interfaceMismatchProgramName = 0;
                s_interfaceMismatchColor0Format = MTLPixelFormatInvalid;
                s_interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
                s_interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
                s_interfaceMismatchRetryAfter = 0.0;
                if (!pipelineReusedPrevious) {
                    _pipelineColor0Format = builtColor0Format;
                    _pipelineDepthFormat = builtDepthFormat;
                    _pipelineStencilFormat = builtStencilFormat;
                    _pipelineProgramName = currentProgramName;
                }
                if (s_programMismatchProgramName == currentProgramName) {
                    s_programMismatchProgramName = 0;
                    s_programMismatchRetryAfter = 0.0;
                    s_programMismatchStreak = 0u;
                }
	                if (_interfaceMismatchBlockedProgram == currentProgramName) {
	                    _interfaceMismatchBlockedProgram = 0;
	                    _interfaceMismatchBlockedUntil = 0.0;
	                    _interfaceMismatchBlockedStreak = 0u;
	                }

                [self insertPipelineIntoCacheWithKey:pipelineCacheKey];
	            }
	            }

                if (deferredBufferMapForPipelineBuild && _pipelineState != nil) {
                    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
                    deferredBufferMapForPipelineBuild = false;
                }

	            state->dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	            }

    return true;
}
/*
 * Pipeline cache insertion with LRU eviction, extracted from
 * syncPipelineStateWithDeferredBufferMap:.
 */
- (void)insertPipelineIntoCacheWithKey:(NSString *)pipelineCacheKey
{
    if (_pipelineStateCache) {
        /* LRU eviction: remove the oldest 25% of cached entries instead
         * of clearing the whole cache. NSMutableDictionary preserves
         * insertion order, so the first keys are the oldest inserts.
         * This is an approximation of true LRU — a real access-ordered
         * LRU would require rebuilding the dictionary on each hit,
         * which is too expensive for this hot path. */
        if (_pipelineStateCache.count >= 256) {
            NSArray *keys = _pipelineStateCache.allKeys;
            NSUInteger evictCount = keys.count / 4;  /* evict 25% */
            MGL_PERF_ADD(g_mglPipelineCacheEvictionsSinceSwap, evictCount);
            for (NSUInteger i = 0; i < evictCount; i++) {
                [_pipelineStateCache removeObjectForKey:keys[i]];
            }
        }
        if (pipelineCacheKey) {
            [_pipelineStateCache setObject:_pipelineState forKey:pipelineCacheKey];
        }
    }
}


/* Bind spvBufferSizeConstants for runtime-sized SSBO arrays in vertex/fragment
 * stages.  SPIRV-Cross emits code that reads uint32 byte-sizes from a
 * constant uint* buffer at MGL_BUFFER_SIZE_BUFFER_INDEX when a shader uses
 * .length() on unsized SSBO arrays.  The render encoder has separate buffer
 * tables for vertex and fragment, so we bind a size buffer for each stage
 * that needs it. */
- (bool) bindBufferSizeConstantsForRenderEncoder
{
    if (!_currentRenderEncoder) {
        return true;
    }

    static __strong id<MTLBuffer> s_vertexSizeBuffer = nil;
    static __strong id<MTLBuffer> s_fragmentSizeBuffer = nil;

    Program *vertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (vertexProgram && vertexProgram->spirv[_VERTEX_SHADER].needs_buffer_size_buffer)
    {
        uint32_t sizeConstants[31];
        memset(sizeConstants, 0, sizeof(sizeConstants));

        for (int i = 0; i < ctx->state.vertex_buffer_map_list.count; i++)
        {
            BufferMap *map = &ctx->state.vertex_buffer_map_list.buffers[i];
            if (!map->buf)
                continue;
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= 31 || metalSlot == MGL_BUFFER_SIZE_BUFFER_INDEX)
                continue;
            GLsizeiptr visibleSize = map->size > 0 ? map->size : (map->buf->size - map->offset);
            if (visibleSize < 0) visibleSize = 0;
            sizeConstants[metalSlot] = (uint32_t)visibleSize;
        }

        s_vertexSizeBuffer = [_device newBufferWithBytes:sizeConstants
                                                   length:sizeof(sizeConstants)
                                                  options:MTLResourceStorageModeShared];
        if (s_vertexSizeBuffer) {
            [_currentRenderEncoder setVertexBuffer:s_vertexSizeBuffer offset:0 atIndex:MGL_BUFFER_SIZE_BUFFER_INDEX];
            [self recordLastBoundVertexBuffer:s_vertexSizeBuffer
                                       offset:0
                                      atIndex:MGL_BUFFER_SIZE_BUFFER_INDEX];
            MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        }
    }

    Program *fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    if (fragmentProgram && fragmentProgram->spirv[_FRAGMENT_SHADER].needs_buffer_size_buffer)
    {
        uint32_t sizeConstants[31];
        memset(sizeConstants, 0, sizeof(sizeConstants));

        for (int i = 0; i < ctx->state.fragment_buffer_map_list.count; i++)
        {
            BufferMap *map = &ctx->state.fragment_buffer_map_list.buffers[i];
            if (!map->buf)
                continue;
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= 31 || metalSlot == MGL_BUFFER_SIZE_BUFFER_INDEX)
                continue;
            GLsizeiptr visibleSize = map->size > 0 ? map->size : (map->buf->size - map->offset);
            if (visibleSize < 0) visibleSize = 0;
            sizeConstants[metalSlot] = (uint32_t)visibleSize;
        }

        s_fragmentSizeBuffer = [_device newBufferWithBytes:sizeConstants
                                                     length:sizeof(sizeConstants)
                                                    options:MTLResourceStorageModeShared];
        if (s_fragmentSizeBuffer) {
            [_currentRenderEncoder setFragmentBuffer:s_fragmentSizeBuffer offset:0 atIndex:MGL_BUFFER_SIZE_BUFFER_INDEX];
            [self recordLastBoundFragmentBuffer:s_fragmentSizeBuffer
                                         offset:0
                                        atIndex:MGL_BUFFER_SIZE_BUFFER_INDEX];
            MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
        }
    }

    return true;
}

-(void) flushCommandBuffer: (bool) finish
{
    METAL_LOCK();
    [self flushCommandBufferLocked:finish];
    METAL_UNLOCK();
}

-(void) flushCommandBufferLocked: (bool) finish
{
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or queue is NULL in flushCommandBuffer");
        return;
    }

    [self flushDrawBuffer:ctx];

    if (![self processGLStateLocked: false]) {
        NSLog(@"MGL WARNING: processGLState failed in flushCommandBuffer, continuing with cleanup");
    }

    [self endRenderEncodingLocked];

    if (![self ensureWritableCommandBufferLocked:"flushCommandBuffer"]) {
        NSLog(@"MGL ERROR: Unable to obtain writable command buffer in flushCommandBuffer");
        return;
    }

    if (!_currentCommandBuffer) {
        NSLog(@"MGL WARNING: No current command buffer in flushCommandBuffer");
        return;
    }

    MTLCommandBufferStatus currentStatus = _currentCommandBuffer.status;
    if (currentStatus != MTLCommandBufferStatusNotEnqueued) {
        NSLog(@"MGL INFO: flushCommandBuffer found finalized buffer (status=%ld), rotating", (long)currentStatus);
        if (![self newCommandBufferLocked]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer in flushCommandBuffer");
        }
        return;
    }

    if (_currentCommandBuffer.error) {
        NSLog(@"MGL ERROR: Command buffer has error before commit: %@", _currentCommandBuffer.error);
        [self cleanupCommandBuffer];
        return;
    }

    if (![self validateMetalObjects]) {
        NSLog(@"MGL WARNING: GPU throttling active - skipping command buffer commit");
        [self cleanupCommandBuffer];
        return;
    }

    id<MTLCommandBuffer> commandBufferToCommit = _currentCommandBuffer;
    _currentCommandBuffer = nil;

    @try {
        [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
        if (finish && commandBufferToCommit.status != MTLCommandBufferStatusNotEnqueued) {
            [commandBufferToCommit waitUntilCompleted];
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Command buffer commit failed in flushCommandBuffer: %@", exception);
        [self recordGPUError];
        [self cleanupCommandBuffer];
    }

    if (!finish) {
        [self newCommandBufferLocked];
    }
}

- (bool)syncRenderPassStateForContext:(GLMContext)glm_ctx
{
    GLMState *state = MGL_STATE(glm_ctx);
    Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(glm_ctx, "processGLState.dirtyFBO");
    BOOL framebufferBindingDirty = framebuffer && (framebuffer->dirty_bits & DIRTY_FBO_BINDING);
    if (_currentRenderEncoder &&
        !framebufferBindingDirty &&
        [self currentRenderPassMatchesCurrentFramebuffer]) {
        state->dirty_bits &= ~DIRTY_FBO;
        return true;
    }

    if (framebuffer && framebufferBindingDirty)
    {
        RETURN_FALSE_ON_FAILURE([self bindFramebufferAttachmentTextures]);
        framebuffer = mglRendererGetValidatedFramebuffer(glm_ctx, "processGLState.dirtyFBO.afterBind");
        if (framebuffer) {
            framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
        }
    }

    /* Stage 4 instrumentation: an FBO change forced a real encoder rotation
     * (the "already matches" fast path above returned early without counting).
     * newRenderEncoderLocked also bumps g_mglEncoderCreationsSinceSwap, so
     * fboRot <= new always holds; new-minus-fboRot is non-FBO creation. */
    /* Stage 4.3: encoder open/close is owned by the RenderPass Manager
     * facade (rotateRenderEncoderForCurrentFramebufferLocked), not by this
     * Sync unit directly. The Sync layer only decides that a rotation is
     * needed and delegates the lifecycle transition. */
    RETURN_FALSE_ON_FAILURE([self rotateRenderEncoderForCurrentFramebufferLocked]);
    return true;
}

/*
 * Stage 4.3 — RenderPass Manager facade: single owner of the FBO-driven
 * encoder rotation (the primary open/close transition). Ends the current
 * render encoder and opens a fresh one against the now-bound framebuffer's
 * render-pass descriptor. Called by the RenderPass Sync unit
 * (syncRenderPassStateForContext:) when currentRenderPassMatchesCurrentFramebuffer
 * is false; the "already matches" fast path returns without rotating. The
 * recovery/nil-encoder paths elsewhere (processGLStateLocked, texture upload,
 * blit) still call newRenderEncoderLocked directly and are documented as
 * out-of-scope for this facade — they are safety nets, not the primary
 * lifecycle transition, and lifting them is deferred until a future pass
 * proves the facade is sufficient under Minecraft workloads.
 */
- (bool)rotateRenderEncoderForCurrentFramebufferLocked
{
    MGL_PERF_INC(g_mglEncoderFBORotationsSinceSwap);
    [self endRenderEncodingLocked];
    RETURN_FALSE_ON_FAILURE([self newRenderEncoderLocked]);
    return true;
}

- (BOOL)prepareRenderPassIfFBOChanged:(MGLDrawBatch *)batch
                              context:(GLMContext)glm_ctx
                          replayError:(GLenum *)replayError
{
    if (!(glm_ctx->state.dirty_bits & DIRTY_FBO))
        return YES;

    /* Orchestrator-driven FBO rotation (Stage 3.1) delegates to the shared
     * RenderPass Sync unit (Stage 3.2), surfacing any GL error as replayError
     * so the batch is skipped rather than drawn against a stale pass. */
    if (![self syncRenderPassStateForContext:glm_ctx]) {
        if (glm_ctx->state.error != GL_NO_ERROR)
            *replayError = glm_ctx->state.error;
        return NO;
    }
    return YES;
}

@end
