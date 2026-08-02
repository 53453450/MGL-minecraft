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
    _state.lastBlendColorRed = 0.0f;
    _state.lastBlendColorGreen = 0.0f;
    _state.lastBlendColorBlue = 0.0f;
    _state.lastBlendColorAlpha = 0.0f;
    _state.lastBoundValid = NO;
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

- (void)setLastBlendColorRed:(float)r green:(float)g blue:(float)b alpha:(float)a
{
    _state.lastBlendColorRed = r;
    _state.lastBlendColorGreen = g;
    _state.lastBlendColorBlue = b;
    _state.lastBlendColorAlpha = a;
}

- (void)setBoundValid:(BOOL)valid
{
    _state.lastBoundValid = valid;
}

@end
