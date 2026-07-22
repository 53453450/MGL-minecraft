#import "MGLBindingSync.h"

static inline void MGLMarkTextureSlot(uint64_t mask[2], NSUInteger index)
{
    if (index < 64) {
        mask[0] |= 1ULL << index;
    } else if (index < 128) {
        mask[1] |= 1ULL << (index - 64);
    }
}

@implementation MGLBindingSync

- (instancetype)init
{
    self = [super init];
    if (self) [self invalidate];
    return self;
}

- (const MGLBindingDedupState *)state
{
    return &_state;
}

- (void)invalidate
{
    for (int index = 0; index < kMGLMaxBufferSlots; index++) {
        _state.lastBoundVertexBuffers[index].buffer = nil;
        _state.lastBoundVertexBuffers[index].offset = 0;
        _state.lastBoundFragmentBuffers[index].buffer = nil;
        _state.lastBoundFragmentBuffers[index].offset = 0;
    }
    for (int index = 0; index < TEXTURE_UNITS; index++) {
        _state.lastBoundVertexTextures[index] = nil;
        _state.lastBoundFragmentTextures[index] = nil;
        _state.lastBoundVertexSamplers[index] = nil;
        _state.lastBoundFragmentSamplers[index] = nil;
    }
    _state.lastBoundVertexBufferMask = 0;
    _state.lastBoundFragmentBufferMask = 0;
    _state.lastBoundTextureSlotMask[0] = 0;
    _state.lastBoundTextureSlotMask[1] = 0;
    _state.lastPipelineState = nil;
    _state.lastDepthStencilState = nil;
    _state.lastViewport = (MTLViewport){0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    _state.lastScissorRect = (MTLScissorRect){0, 0, 0, 0};
    _state.lastCullMode = MTLCullModeNone;
    _state.lastFrontFacingWinding = MTLWindingClockwise;
    _state.lastTriangleFillMode = MTLTriangleFillModeFill;
    _state.lastDepthBias = 0;
    _state.lastDepthBiasClamp = 0;
    _state.lastDepthSlopeScale = 0;
    _state.lastBoundValid = NO;
}

- (void)copyStateTo:(MGLBindingDedupState *)destination
{
    if (!destination) return;

    static uint64_t sparseIterationCount = 0;
    static uint64_t slotIterationCount = 0;
    BOOL traceSparse = getenv("MGL_TRACE_SPARSE_BINDING") != NULL;

    uint32_t vertexMask = _state.lastBoundVertexBufferMask;
    destination->lastBoundVertexBufferMask = vertexMask;
    while (vertexMask) {
        int index = __builtin_ctz(vertexMask);
        destination->lastBoundVertexBuffers[index] = _state.lastBoundVertexBuffers[index];
        vertexMask &= ~(1U << index);
        if (traceSparse) slotIterationCount++;
    }

    uint32_t fragmentMask = _state.lastBoundFragmentBufferMask;
    destination->lastBoundFragmentBufferMask = fragmentMask;
    while (fragmentMask) {
        int index = __builtin_ctz(fragmentMask);
        destination->lastBoundFragmentBuffers[index] = _state.lastBoundFragmentBuffers[index];
        fragmentMask &= ~(1U << index);
        if (traceSparse) slotIterationCount++;
    }

    for (NSUInteger half = 0; half < 2; half++) {
        uint64_t textureMask = _state.lastBoundTextureSlotMask[half];
        destination->lastBoundTextureSlotMask[half] = textureMask;
        while (textureMask) {
            NSUInteger index = (NSUInteger)__builtin_ctzll(textureMask) + half * 64;
            destination->lastBoundVertexTextures[index] = _state.lastBoundVertexTextures[index];
            destination->lastBoundFragmentTextures[index] = _state.lastBoundFragmentTextures[index];
            destination->lastBoundVertexSamplers[index] = _state.lastBoundVertexSamplers[index];
            destination->lastBoundFragmentSamplers[index] = _state.lastBoundFragmentSamplers[index];
            textureMask &= textureMask - 1;
            if (traceSparse) slotIterationCount++;
        }
    }

    if (traceSparse) {
        sparseIterationCount++;
        if ((sparseIterationCount % 1000) == 0) {
            NSLog(@"MGL SPARSE: %llu calls, %llu slots (avg %.1f slots/call, baseline would be ~192)",
                  sparseIterationCount,
                  slotIterationCount,
                  (double)slotIterationCount / sparseIterationCount);
        }
    }

    destination->lastPipelineState = _state.lastPipelineState;
    destination->lastDepthStencilState = _state.lastDepthStencilState;
    destination->lastViewport = _state.lastViewport;
    destination->lastScissorRect = _state.lastScissorRect;
    destination->lastCullMode = _state.lastCullMode;
    destination->lastFrontFacingWinding = _state.lastFrontFacingWinding;
    destination->lastTriangleFillMode = _state.lastTriangleFillMode;
    destination->lastDepthBias = _state.lastDepthBias;
    destination->lastDepthBiasClamp = _state.lastDepthBiasClamp;
    destination->lastDepthSlopeScale = _state.lastDepthSlopeScale;
    destination->lastBoundValid = _state.lastBoundValid;
}

- (void)restoreStateFrom:(const MGLBindingDedupState *)source
{
    if (!source) return;

    uint32_t staleVertexMask = _state.lastBoundVertexBufferMask &
                               ~source->lastBoundVertexBufferMask;
    while (staleVertexMask) {
        int index = __builtin_ctz(staleVertexMask);
        _state.lastBoundVertexBuffers[index].buffer = nil;
        _state.lastBoundVertexBuffers[index].offset = 0;
        staleVertexMask &= ~(1U << index);
    }
    uint32_t vertexMask = source->lastBoundVertexBufferMask;
    while (vertexMask) {
        int index = __builtin_ctz(vertexMask);
        _state.lastBoundVertexBuffers[index] = source->lastBoundVertexBuffers[index];
        vertexMask &= ~(1U << index);
    }
    _state.lastBoundVertexBufferMask = source->lastBoundVertexBufferMask;

    uint32_t staleFragmentMask = _state.lastBoundFragmentBufferMask &
                                 ~source->lastBoundFragmentBufferMask;
    while (staleFragmentMask) {
        int index = __builtin_ctz(staleFragmentMask);
        _state.lastBoundFragmentBuffers[index].buffer = nil;
        _state.lastBoundFragmentBuffers[index].offset = 0;
        staleFragmentMask &= ~(1U << index);
    }
    uint32_t fragmentMask = source->lastBoundFragmentBufferMask;
    while (fragmentMask) {
        int index = __builtin_ctz(fragmentMask);
        _state.lastBoundFragmentBuffers[index] = source->lastBoundFragmentBuffers[index];
        fragmentMask &= ~(1U << index);
    }
    _state.lastBoundFragmentBufferMask = source->lastBoundFragmentBufferMask;

    for (NSUInteger half = 0; half < 2; half++) {
        uint64_t staleTextureMask = _state.lastBoundTextureSlotMask[half] &
                                    ~source->lastBoundTextureSlotMask[half];
        while (staleTextureMask) {
            NSUInteger index = (NSUInteger)__builtin_ctzll(staleTextureMask) + half * 64;
            _state.lastBoundVertexTextures[index] = nil;
            _state.lastBoundFragmentTextures[index] = nil;
            _state.lastBoundVertexSamplers[index] = nil;
            _state.lastBoundFragmentSamplers[index] = nil;
            staleTextureMask &= staleTextureMask - 1;
        }

        uint64_t textureMask = source->lastBoundTextureSlotMask[half];
        while (textureMask) {
            NSUInteger index = (NSUInteger)__builtin_ctzll(textureMask) + half * 64;
            _state.lastBoundVertexTextures[index] = source->lastBoundVertexTextures[index];
            _state.lastBoundFragmentTextures[index] = source->lastBoundFragmentTextures[index];
            _state.lastBoundVertexSamplers[index] = source->lastBoundVertexSamplers[index];
            _state.lastBoundFragmentSamplers[index] = source->lastBoundFragmentSamplers[index];
            textureMask &= textureMask - 1;
        }
        _state.lastBoundTextureSlotMask[half] = source->lastBoundTextureSlotMask[half];
    }

    _state.lastPipelineState = source->lastPipelineState;
    _state.lastDepthStencilState = source->lastDepthStencilState;
    _state.lastViewport = source->lastViewport;
    _state.lastScissorRect = source->lastScissorRect;
    _state.lastCullMode = source->lastCullMode;
    _state.lastFrontFacingWinding = source->lastFrontFacingWinding;
    _state.lastTriangleFillMode = source->lastTriangleFillMode;
    _state.lastDepthBias = source->lastDepthBias;
    _state.lastDepthBiasClamp = source->lastDepthBiasClamp;
    _state.lastDepthSlopeScale = source->lastDepthSlopeScale;
    _state.lastBoundValid = source->lastBoundValid;
}

- (void)recordVertexBuffer:(id<MTLBuffer>)buffer
                    offset:(NSUInteger)offset
                   atIndex:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) return;
    _state.lastBoundVertexBuffers[index].buffer = buffer;
    _state.lastBoundVertexBuffers[index].offset = offset;
    _state.lastBoundVertexBufferMask |= 1U << index;
}

- (void)recordFragmentBuffer:(id<MTLBuffer>)buffer
                      offset:(NSUInteger)offset
                     atIndex:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) return;
    _state.lastBoundFragmentBuffers[index].buffer = buffer;
    _state.lastBoundFragmentBuffers[index].offset = offset;
    _state.lastBoundFragmentBufferMask |= 1U << index;
}

- (void)invalidateVertexBufferAtIndex:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) return;
    _state.lastBoundVertexBuffers[index].buffer = nil;
    _state.lastBoundVertexBuffers[index].offset = (NSUInteger)-1;
    _state.lastBoundVertexBufferMask |= 1U << index;
}

- (void)invalidateFragmentBufferAtIndex:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) return;
    _state.lastBoundFragmentBuffers[index].buffer = nil;
    _state.lastBoundFragmentBuffers[index].offset = (NSUInteger)-1;
    _state.lastBoundFragmentBufferMask |= 1U << index;
}

- (void)setVertexTextureIfNeeded:(id<MTLTexture>)texture
                          atIndex:(NSUInteger)index
                          encoder:(id<MTLRenderCommandEncoder>)encoder
{
    if (!encoder || index >= TEXTURE_UNITS) return;
    MGLMarkTextureSlot(_state.lastBoundTextureSlotMask, index);
    if (!_state.lastBoundValid || _state.lastBoundVertexTextures[index] != texture) {
        [encoder setVertexTexture:texture atIndex:index];
        _state.lastBoundVertexTextures[index] = texture;
    }
}

- (void)setFragmentTextureIfNeeded:(id<MTLTexture>)texture
                            atIndex:(NSUInteger)index
                            encoder:(id<MTLRenderCommandEncoder>)encoder
{
    if (!encoder || index >= TEXTURE_UNITS) return;
    MGLMarkTextureSlot(_state.lastBoundTextureSlotMask, index);
    if (!_state.lastBoundValid || _state.lastBoundFragmentTextures[index] != texture) {
        [encoder setFragmentTexture:texture atIndex:index];
        _state.lastBoundFragmentTextures[index] = texture;
    }
}

- (void)setVertexSamplerIfNeeded:(id<MTLSamplerState>)sampler
                         atIndex:(NSUInteger)index
                         encoder:(id<MTLRenderCommandEncoder>)encoder
{
    if (!encoder || index >= TEXTURE_UNITS) return;
    MGLMarkTextureSlot(_state.lastBoundTextureSlotMask, index);
    if (!_state.lastBoundValid || _state.lastBoundVertexSamplers[index] != sampler) {
        [encoder setVertexSamplerState:sampler atIndex:index];
        _state.lastBoundVertexSamplers[index] = sampler;
    }
}

- (void)setFragmentSamplerIfNeeded:(id<MTLSamplerState>)sampler
                           atIndex:(NSUInteger)index
                           encoder:(id<MTLRenderCommandEncoder>)encoder
{
    if (!encoder || index >= TEXTURE_UNITS) return;
    MGLMarkTextureSlot(_state.lastBoundTextureSlotMask, index);
    if (!_state.lastBoundValid || _state.lastBoundFragmentSamplers[index] != sampler) {
        [encoder setFragmentSamplerState:sampler atIndex:index];
        _state.lastBoundFragmentSamplers[index] = sampler;
    }
}

- (void)setViewportIfNeeded:(MTLViewport)viewport
                     encoder:(id<MTLRenderCommandEncoder>)encoder
{
    if (!encoder) return;
    MTLViewport last = _state.lastViewport;
    if (!_state.lastBoundValid || last.originX != viewport.originX ||
        last.originY != viewport.originY || last.width != viewport.width ||
        last.height != viewport.height || last.znear != viewport.znear ||
        last.zfar != viewport.zfar) {
        [encoder setViewport:viewport];
        _state.lastViewport = viewport;
    }
}

- (void)setScissorRectIfNeeded:(MTLScissorRect)rect
                        encoder:(id<MTLRenderCommandEncoder>)encoder
{
    if (!encoder) return;
    MTLScissorRect last = _state.lastScissorRect;
    if (!_state.lastBoundValid || last.x != rect.x || last.y != rect.y ||
        last.width != rect.width || last.height != rect.height) {
        [encoder setScissorRect:rect];
        _state.lastScissorRect = rect;
    }
}

- (void)setTriangleFillModeIfNeeded:(MTLTriangleFillMode)mode
                             encoder:(id<MTLRenderCommandEncoder>)encoder
{
    if (!encoder) return;
    if (!_state.lastBoundValid || _state.lastTriangleFillMode != mode) {
        [encoder setTriangleFillMode:mode];
        _state.lastTriangleFillMode = mode;
    }
}

- (void)updateVertexBufferSlot:(NSUInteger)index
                        buffer:(id<MTLBuffer>)buffer
                        offset:(NSUInteger)offset
{
    if (index >= kMGLMaxBufferSlots) return;
    _state.lastBoundVertexBuffers[index].buffer = buffer;
    _state.lastBoundVertexBuffers[index].offset = offset;
}

- (void)updateFragmentBufferSlot:(NSUInteger)index
                          buffer:(id<MTLBuffer>)buffer
                          offset:(NSUInteger)offset
{
    if (index >= kMGLMaxBufferSlots) return;
    _state.lastBoundFragmentBuffers[index].buffer = buffer;
    _state.lastBoundFragmentBuffers[index].offset = offset;
}

- (void)clearVertexBufferSlot:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) return;
    _state.lastBoundVertexBuffers[index].buffer = nil;
    _state.lastBoundVertexBuffers[index].offset = 0;
}

- (void)clearFragmentBufferSlot:(NSUInteger)index
{
    if (index >= kMGLMaxBufferSlots) return;
    _state.lastBoundFragmentBuffers[index].buffer = nil;
    _state.lastBoundFragmentBuffers[index].offset = 0;
}

- (void)orVertexBufferMask:(uint32_t)mask
{
    _state.lastBoundVertexBufferMask |= mask;
}

- (void)orFragmentBufferMask:(uint32_t)mask
{
    _state.lastBoundFragmentBufferMask |= mask;
}

- (void)setLastPipelineState:(id<MTLRenderPipelineState>)pipelineState
{
    _state.lastPipelineState = pipelineState;
}

- (void)setLastDepthStencilState:(id<MTLDepthStencilState>)depthStencilState
{
    _state.lastDepthStencilState = depthStencilState;
}

- (void)setLastCullMode:(MTLCullMode)cullMode
{
    _state.lastCullMode = cullMode;
}

- (void)setLastFrontFacingWinding:(MTLWinding)winding
{
    _state.lastFrontFacingWinding = winding;
}

- (void)setLastDepthBias:(float)bias
                   clamp:(float)clamp
              slopeScale:(float)slopeScale
{
    _state.lastDepthBias = bias;
    _state.lastDepthBiasClamp = clamp;
    _state.lastDepthSlopeScale = slopeScale;
}

- (void)setBoundValid:(BOOL)valid
{
    _state.lastBoundValid = valid;
}

@end
