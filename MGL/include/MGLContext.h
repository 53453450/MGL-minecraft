/*
  MGLContext.h
  mgl

  Created by Michael Larson on 12/17/21.
*/

#ifndef MGLContext_h
#define MGLContext_h

// probably a reason for this... can't remember
#ifndef __GLM_CONTEXT_
#define __GLM_CONTEXT_
typedef struct GLMContextRec_t *GLMContext;
#endif

// Guarded so external consumers that also pull in glm_context.h (via
// MGLRenderer.h) don't see a redefinition. Internal MGL sources include
// glm_context.h directly and never MGLContext.h, so they are unaffected.
#include "mgl_context_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

GLMContext createGLMContext(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type);

GLuint sizeForFormatType(GLenum format, GLenum type);
GLuint bicountForFormatType(GLenum format, GLenum type, GLenum component);

GLMContext MGLgetCurrentContext(void);
void MGLsetCurrentContext(GLMContext ctx);
void destroyGLMContext(GLMContext ctx);

// MGLswapBuffers can take NULL for the ctx, in this case it will use the current ctx
void MGLswapBuffers(GLMContext ctx);

// MGLget can take NULL for the ctx, in this case it will use the current ctx
void MGLget(GLMContext ctx, GLenum param, GLuint *data);

#ifdef __cplusplus
};
#endif

#endif
