/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MGLRenderer+Blit_Private.h
 * MGL
 *
 * Private method declarations, types, and C helpers for the Blit category
 * (MGLRenderer+Blit.m).  Imports MGLRenderer_Private.h for ivar access
 * and shared types.
 */

#ifndef MGLRenderer_Blit_Private_h
#define MGLRenderer_Blit_Private_h

#import "MGLRenderer_Private.h"

/* === Shader parameter structs ===
 * Used by blit/copy/resolve pipelines in MGLRenderer+Blit.m and MGLRenderer.m.
 * The Metal shader string mirrors these layouts — keep field order in sync. */
typedef struct MGLScaledBlitParams_t {
    vector_float4 uvRect; // xy=min, zw=max in normalized Metal texture coordinates.
    float forceOpaqueAlpha;
    vector_float3 _padding;
} MGLScaledBlitParams;

typedef struct MGLMSAAIntegerResolveParams_t {
    vector_uint2 srcOrigin;
    vector_uint2 dstOrigin;
    vector_uint2 size;
    vector_uint2 _padding;
} MGLMSAAIntegerResolveParams;

typedef struct MGLClearRectParams_t {
    vector_float4 color;
    float depth;
    vector_float3 _padding;
} MGLClearRectParams;

/* === Diagnostic constant === */
static const BOOL kMGLSwapPresentDiagnostics = NO;

/* === C functions defined in MGLRenderer.m, used by MGLRenderer+Blit.m === */

/* GL internal format → CPU (format, type) mapping — used by mtlCopyImageSubData
 * for format-converting readback. */
GLboolean mglGetCPUFormatTypeForInternalFormat(GLenum internalformat,
                                               GLenum *outFormat,
                                               GLenum *outType);

/* RT Metal-fill marker — inline because it's small and called from
 * both MGLRenderer.m and MGLRenderer+Blit.m / MGLRenderer+Texture.m. */
static inline void mglMarkTextureLevelMetalFilled(Texture *tex, GLuint level, size_t uploadSize)
{
    TextureLevel *texLevel = mglTextureAttachmentLevel(tex, level);
    if (!texLevel) {
        return;
    }

    texLevel->ever_written = GL_TRUE;
    texLevel->has_initialized_data = GL_TRUE;
    texLevel->suspicious_zero_upload = GL_FALSE;
    texLevel->last_init_source = kTexMetalFill;
    texLevel->last_upload_size = uploadSize;
    texLevel->last_src_ptr = NULL;
    texLevel->last_src_hash = 0ull;

    if (tex->is_render_target) {
        tex->mtl_render_target_write_version++;
    }
}

/* === RT-write marker — used by Blit.m, Texture.m, Draw.m, RenderPass.m ===
 * The impl lives in MGLRenderer.m (non-static); the macro is here so category
 * files can call mglMarkTextureLevelRenderTargetWritten(tex, level). */
void mglMarkTextureLevelRenderTargetWrittenImpl(Texture *tex,
                                                GLuint level,
                                                const char *caller,
                                                int line);

#define mglMarkTextureLevelRenderTargetWritten(tex, level) \
    mglMarkTextureLevelRenderTargetWrittenImpl((tex), (level), __func__, __LINE__)

@interface MGLRenderer ()

// === Blit pipeline caches ===
- (id<MTLRenderPipelineState>)scaledBlitPipelineForPixelFormat:(MTLPixelFormat)pixelFormat;
- (id<MTLSamplerState>)scaledBlitSamplerForFilter:(GLuint)filter;
- (id<MTLRenderPipelineState>)clearRectPipelineForColorFormat:(MTLPixelFormat)colorFormat
                                                  depthFormat:(MTLPixelFormat)depthFormat
                                                  writesColor:(BOOL)writesColor
                                                  writesDepth:(BOOL)writesDepth;
- (id<MTLDepthStencilState>)clearRectDepthState;

// === Multisample resolve ===
- (id<MTLTexture>)resolvedReadbackTextureForMultisampleTexture:(id<MTLTexture>)sourceTexture
                                                   sourceLevel:(NSUInteger)sourceLevel
                                                   sourceSlice:(NSUInteger)sourceSlice
                                               sourceDepthPlane:(NSUInteger)sourceDepthPlane
                                                        reason:(const char *)reason;
- (id<MTLTexture>)depthFloatTextureForDepthStencilReadback:(id<MTLTexture>)sourceTexture
                                                    reason:(const char *)reason;

// === GL sampled render target copy management ===
- (BOOL)textureCanUseGLSampledRenderTargetCopy:(Texture *)tex
                                        source:(id<MTLTexture>)source;
- (void)releaseGLSampledRenderTargetCopyForTexture:(Texture *)tex;
- (BOOL)updateGLSampledRenderTargetCopyForTexture:(Texture *)tex
                                           source:(id<MTLTexture>)source
                                           reason:(const char *)reason;
- (id<MTLTexture>)freshGLSampledRenderTargetCopyForSampling:(Texture *)tex
                                                     source:(id<MTLTexture>)source
                                                      stage:(const char *)stage
                                                    program:(GLuint)programName
                                                    binding:(GLuint)binding
                                                       unit:(GLuint)unit
                                               expectedType:(MTLTextureType)expectedType
                                               expectedKind:(MGLTextureDataKind)expectedKind;

@end

#endif /* MGLRenderer_Blit_Private_h */
