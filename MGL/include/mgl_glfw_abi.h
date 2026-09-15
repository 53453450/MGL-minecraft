/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holder.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_glfw_abi.h — the MGL-side shared C declarations for out-of-tree
 * windowing hosts (the GLFW fork).
 *
 * The host used to hand-write these declarations (mgl_context.m:28-37).
 * C does not mangle names, so a signature drift between the two artifacts
 * still linked and corrupted the stack silently — strictly worse than the
 * selector crash it sat next to.  This header is the single source both
 * sides compile against:
 *
 *   - Consumers (GLFW) include it instead of hand-written declarations.
 *   - MGL includes it next to the definitions (glm_context.c) so a
 *     signature change without updating this header fails to compile.
 *
 * GLenum / GLboolean / GLuint are intentionally NOT typedef'd here: pulling
 * glcorearb.h would redefine GL_VERSION / GL_EXTENSIONS and break the
 * consumer's constant set.  Both supported environments provide compatible
 * definitions, and the consumer's spellings are used as-is:
 *   - MGL TUs:      MGL/include/GL/glcorearb.h  (unsigned int / unsigned char)
 *   - GLFW TU:      Apple OpenGL.framework gltypes.h (uint32_t / uint8_t),
 *                   reached via the QuartzCore/AppKit imports
 * The two environments are ABI-identical (GLenum 32-bit, GLboolean 8-bit).
 */

#ifndef MGL_GLFW_ABI_H
#define MGL_GLFW_ABI_H

#include <stdint.h> /* uint32_t (mtlPixelFormatForGLFormatType) */

/* Same forward declaration glm_context.h and the GLFW fork's MGLContext.h
 * use, guarded by the shared macro so any include order works. */
#ifndef __GLM_CONTEXT_
#define __GLM_CONTEXT_
typedef struct GLMContextRec_t *GLMContext;
#endif

#ifdef __cplusplus
extern "C" {
#endif

GLMContext createGLMContext(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type);

void MGLsetDefaultFramebufferSRGBCapable(GLMContext ctx, GLboolean capable);

void MGLsetCurrentContext(GLMContext ctx);
GLMContext MGLgetCurrentContext(void);
void MGLswapBuffers(GLMContext ctx);
void destroyGLMContext(GLMContext ctx);

/* Canonical return type is uint32_t (pixel_utils.h); the GLFW facade's
 * historical `GLenum` spelling is ABI-identical. */
uint32_t mtlPixelFormatForGLFormatType(GLenum gl_format, GLenum gl_type);

#ifdef __cplusplus
}
#endif

#endif /* MGL_GLFW_ABI_H */
