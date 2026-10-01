//
//  vertex_arrays.h
//  MGL
//
//  Created by Michael Larson on 10/17/23.
//

#ifndef vertex_arrays_h
#define vertex_arrays_h

#include "glm_context.h"

VertexArray *newVAO(GLMContext ctx, GLuint vao);

/* Current VAO, or NULL for VAO 0.  An invalid current pointer is reset to
 * VAO 0 (throttled warning naming caller) and NULL is returned. */
VertexArray *mglGetSafeCurrentVAO(GLMContext ctx, const char *caller);

#endif /* vertex_arrays_h */
