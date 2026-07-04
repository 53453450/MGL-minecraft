/*
 * mgl_msl_compat.h
 * MGL
 *
 * MSL Post-Processing Subsystem.
 *
 * Bridges the semantic gap between GLSL/SPIR-V shader semantics and Metal
 * Shading Language (MSL) semantics.  Covers several spec-compliance areas:
 *
 *   - MSL struct size computation: GL per-vertex input/output struct sizes
 *     must be parsed from the MSL source because SPIRV-Cross reflection
 *     only reports user-defined variables, not built-in or padding fields.
 *   - MSL texture type / data-kind inference: GL sampler binding must match
 *     the Metal texture type (2D/3D/cube/array/buffer/MS) and data kind
 *     (float/int/uint/depth) declared in the MSL source.
 *   - MSL named-argument lookup: GL resource binding must verify that the
 *     MSL source actually declares the expected [[buffer(N)]]/[[texture(N)]]
 *     /[[sampler(N)]] argument, because SPIRV-Cross reflection can list
 *     stale resources that were optimized out of the MSL.
 *   - Stale resource skip gating: combine the above to skip binding
 *     resources that the MSL no longer references.
 *   - Vertex clip-space variant generation: GL_UPPER_LEFT / GL_ZERO_TO_ONE
 *     clip origin and depth range require MSL source patching because
 *     Metal only supports the GL_LOWER_LEFT / -1..1 convention natively.
 *   - Shader stage naming: map internal stage enum to string for logging.
 *
 * This module is pure specification-compliance machinery: every OpenGL
 * program that uses the corresponding GL shader features needs these
 * translations when running on Metal, regardless of application.
 */

#ifndef MGL_MSL_COMPAT_H
#define MGL_MSL_COMPAT_H

#include "glm_context.h"
#include "mgl_texture_compat.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* === Shader stage naming === */
const char *mglShaderStageName(int stage);

/* === MSL struct size computation === */

/* Compute the size of an MSL struct whose name ends with `suffix` by parsing
 * the MSL source.  Returns 0 if no matching struct is found. */
NSUInteger mglComputeMSLStructSizeBySuffix(const char *msl,
                                           const char *suffix,
                                           size_t suffixLen);

/* Convenience: compute the size of the per-vertex output struct
 * (struct <entry>_out). */
NSUInteger mglComputeMSLOutputStructSize(const char *msl);

/* === MSL texture type / data-kind inference === */

/* Infer the Metal texture type (MTLTextureType2D/3D/Cube/...) declared at
 * the given [[texture(N)]] binding in the MSL source.  Returns 0 if not
 * found. */
MTLTextureType mglExpectedTextureTypeFromMSL(const char *msl, GLuint binding);

/* Infer the Metal texture data kind (float/sint/uint/depth) declared at
 * the given [[texture(N)]] binding in the MSL source. */
MGLTextureDataKind mglExpectedTextureDataKindFromMSL(const char *msl, GLuint binding);

/* === MSL named-argument lookup === */

bool mglMSLArgumentIdentifierChar(char c);

/* Returns true if the MSL source for `program`'s `stage` declares a
 * parameter named `name` with the attribute [[attributeKind(metalBinding)]].
 * Returns true (vacuously) if the MSL source is missing — callers treat
 * missing MSL as "do not skip". */
bool mglStageMSLHasNamedArgument(Program *program,
                                 int stage,
                                 const char *name,
                                 const char *attributeKind,
                                 GLuint metalBinding);

bool mglStageMSLHasNamedBufferArgument(Program *program,
                                       int stage,
                                       const char *name,
                                       GLuint metalBinding);

bool mglStageMSLHasNamedTextureArgument(Program *program,
                                        int stage,
                                        const char *name,
                                        GLuint metalBinding);

bool mglStageMSLHasNamedSamplerArgument(Program *program,
                                        int stage,
                                        const char *name,
                                        GLuint metalBinding);

/* Returns true if the MSL source for `program`'s `stage` declares any
 * parameter with the attribute [[attributeKind(metalBinding)]]. */
bool mglStageMSLHasArgumentAtBinding(Program *program,
                                     int stage,
                                     const char *attributeKind,
                                     GLuint metalBinding);

/* === Stale resource skip gating === */

bool mglShouldSkipStageBufferResource(Program *program,
                                      int stage,
                                      int resourceType,
                                      const SpirvResource *resource);

bool mglShouldSkipStageTextureResource(Program *program,
                                       int stage,
                                       int resourceType,
                                       const SpirvResource *resource);

bool mglShouldSkipStageSamplerResource(Program *program,
                                       int stage,
                                       int resourceType,
                                       const SpirvResource *resource);

/* === Vertex clip-space variant generation ===
 *
 * GL_UPPER_LEFT clip origin and GL_ZERO_TO_ONE depth range are not native
 * to Metal.  These helpers generate patched MSL source and entry-point
 * names for the clip-origin / depth-range variants required by the
 * gl_ClipOrigin / gl_DepthRange state. */

#ifdef __OBJC__
NSString *mglVertexClipVariantMSLSource(Program *program,
                                        Shader *shader,
                                        BOOL keepDepthFixup,
                                        BOOL flipY,
                                        NSString *entrySuffix);

NSString *mglVertexClipVariantEntryName(Shader *shader, NSString *entrySuffix);

NSString *mglZeroToOneVertexMSLSource(Program *program, Shader *shader);
NSString *mglZeroToOneVertexEntryName(Shader *shader);

NSString *mglUpperLeftVertexMSLSource(Program *program, Shader *shader, BOOL zeroToOneDepth);
NSString *mglUpperLeftVertexEntryName(Shader *shader, BOOL zeroToOneDepth);
#endif

#ifdef __cplusplus
}
#endif

#endif /* MGL_MSL_COMPAT_H */
