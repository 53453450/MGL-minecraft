#ifndef MGLBindingSync_h
#define MGLBindingSync_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "glm_context.h"

#ifndef kMGLMaxBufferSlots
#define kMGLMaxBufferSlots 31
#endif

typedef struct {
    id<MTLBuffer> __strong _Nullable buffer;
    NSUInteger offset;
} MGLLastBoundBuffer;

typedef struct MGLBindingDedupState_t {
    MGLLastBoundBuffer lastBoundVertexBuffers[kMGLMaxBufferSlots];
    MGLLastBoundBuffer lastBoundFragmentBuffers[kMGLMaxBufferSlots];
    id<MTLTexture> __strong _Nullable lastBoundVertexTextures[TEXTURE_UNITS];
    id<MTLTexture> __strong _Nullable lastBoundFragmentTextures[TEXTURE_UNITS];
    id<MTLSamplerState> __strong _Nullable lastBoundVertexSamplers[TEXTURE_UNITS];
    id<MTLSamplerState> __strong _Nullable lastBoundFragmentSamplers[TEXTURE_UNITS];
    uint32_t lastBoundVertexBufferMask;
    uint32_t lastBoundFragmentBufferMask;
    uint64_t lastBoundTextureSlotMask[2];
    id<MTLRenderPipelineState> __strong _Nullable lastPipelineState;
    id<MTLDepthStencilState> __strong _Nullable lastDepthStencilState;
    MTLViewport lastViewport;
    MTLScissorRect lastScissorRect;
    MTLCullMode lastCullMode;
    MTLWinding lastFrontFacingWinding;
    MTLTriangleFillMode lastTriangleFillMode;
    float lastDepthBias;
    float lastDepthBiasClamp;
    float lastDepthSlopeScale;
    float lastBlendColorRed;
    float lastBlendColorGreen;
    float lastBlendColorBlue;
    float lastBlendColorAlpha;
    BOOL lastBoundValid;
} MGLBindingDedupState;

NS_ASSUME_NONNULL_BEGIN

@interface MGLBindingSync : NSObject {
@private
    MGLBindingDedupState _state;
}

@property(nonatomic, readonly) const MGLBindingDedupState *state;

- (void)invalidate;

- (void)recordVertexBuffer:(nullable id<MTLBuffer>)buffer
                    offset:(NSUInteger)offset
                   atIndex:(NSUInteger)index;
- (void)recordFragmentBuffer:(nullable id<MTLBuffer>)buffer
                      offset:(NSUInteger)offset
                     atIndex:(NSUInteger)index;
- (void)invalidateVertexBufferAtIndex:(NSUInteger)index;
- (void)invalidateFragmentBufferAtIndex:(NSUInteger)index;

- (void)setVertexTextureIfNeeded:(nullable id<MTLTexture>)texture
                          atIndex:(NSUInteger)index
                          encoder:(nullable id<MTLRenderCommandEncoder>)encoder;
- (void)setFragmentTextureIfNeeded:(nullable id<MTLTexture>)texture
                            atIndex:(NSUInteger)index
                            encoder:(nullable id<MTLRenderCommandEncoder>)encoder;
- (void)setVertexSamplerIfNeeded:(nullable id<MTLSamplerState>)sampler
                         atIndex:(NSUInteger)index
                         encoder:(nullable id<MTLRenderCommandEncoder>)encoder;
- (void)setFragmentSamplerIfNeeded:(nullable id<MTLSamplerState>)sampler
                           atIndex:(NSUInteger)index
                           encoder:(nullable id<MTLRenderCommandEncoder>)encoder;
- (void)setViewportIfNeeded:(MTLViewport)viewport
                     encoder:(nullable id<MTLRenderCommandEncoder>)encoder;
- (void)setScissorRectIfNeeded:(MTLScissorRect)rect
                        encoder:(nullable id<MTLRenderCommandEncoder>)encoder;
- (void)setTriangleFillModeIfNeeded:(MTLTriangleFillMode)mode
                             encoder:(nullable id<MTLRenderCommandEncoder>)encoder;

/* Low-level dedup-state mutators used by the encoder hot path.  Callers keep
 * their own encoder calls, dedup comparisons, and perf counters; these only
 * write the tracked state so the manager owns all writes to _state (no raw
 * writable state pointer escapes). */

/* Record a slot's resolved buffer+offset WITHOUT touching the presence mask.
 * Distinct from recordVertexBuffer:offset:atIndex:, which also sets the mask. */
- (void)updateVertexBufferSlot:(NSUInteger)index
                        buffer:(nullable id<MTLBuffer>)buffer
                        offset:(NSUInteger)offset;
- (void)updateFragmentBufferSlot:(NSUInteger)index
                          buffer:(nullable id<MTLBuffer>)buffer
                          offset:(NSUInteger)offset;

/* Clear a slot to (nil, 0) WITHOUT touching the presence mask.  Distinct from
 * invalidateVertexBufferAtIndex:, which writes offset -1 and sets the mask. */
- (void)clearVertexBufferSlot:(NSUInteger)index;
- (void)clearFragmentBufferSlot:(NSUInteger)index;

- (void)orVertexBufferMask:(uint32_t)mask;
- (void)orFragmentBufferMask:(uint32_t)mask;

- (void)setLastPipelineState:(nullable id<MTLRenderPipelineState>)pipelineState;
- (void)setLastDepthStencilState:(nullable id<MTLDepthStencilState>)depthStencilState;
- (void)setLastCullMode:(MTLCullMode)cullMode;
- (void)setLastFrontFacingWinding:(MTLWinding)winding;
- (void)setLastDepthBias:(float)bias
                   clamp:(float)clamp
              slopeScale:(float)slopeScale;
- (void)setLastBlendColorRed:(float)r green:(float)g blue:(float)b alpha:(float)a;
- (void)setBoundValid:(BOOL)valid;

@end

NS_ASSUME_NONNULL_END

#endif /* MGLBindingSync_h */
