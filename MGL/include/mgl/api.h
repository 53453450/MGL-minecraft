/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl/api.h — cross-layer GL entry surface with opaque context (ARCH-015).
 *
 * Must not include the full GLMContextRec. Callers that need Rec fields
 * include the context runtime header in an L4/L2 implementation TU.
 * Full mgl* prototypes remain in mgl.h until entry points migrate here.
 */

#ifndef MGL_API_H
#define MGL_API_H

#include "glm_context_fwd.h"
#include "gltypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Context lifecycle used by L5 / platform without Rec layout. */
GLMContext createGLMContext(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type);
void MGLsetCurrentContext(GLMContext ctx);
GLMContext MGLgetCurrentContext(void);

/* First-bind validation for DSA scopes; implemented by the L4 VAO owner. */
GLboolean mglVertexArrayNameMayBind(GLMContext ctx, GLuint array);

#ifdef __cplusplus
}
#endif

#endif /* MGL_API_H */
