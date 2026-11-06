/* SPDX-License-Identifier: LGPL-3.0-only */

#include "glm_context.h"

/* Reflection tests do not construct a GL context or execute uniform seeding. */
GLint mglGetUniformLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
    (void)ctx;
    (void)program;
    (void)name;
    return -1;
}

#define STUB_UNIFORM_SCALAR(name, type)                         \
    void name(GLMContext ctx, GLint location, type value)        \
    {                                                            \
        (void)ctx; (void)location; (void)value;                   \
    }
#define STUB_UNIFORM_VECTOR(name, type)                         \
    void name(GLMContext ctx, GLint location, GLsizei count,     \
              const type *value)                                 \
    {                                                            \
        (void)ctx; (void)location; (void)count; (void)value;      \
    }

STUB_UNIFORM_SCALAR(mglUniform1f, GLfloat)
STUB_UNIFORM_SCALAR(mglUniform1i, GLint)
STUB_UNIFORM_SCALAR(mglUniform1ui, GLuint)
STUB_UNIFORM_VECTOR(mglUniform1fv, GLfloat)
STUB_UNIFORM_VECTOR(mglUniform1iv, GLint)
STUB_UNIFORM_VECTOR(mglUniform1uiv, GLuint)
STUB_UNIFORM_VECTOR(mglUniform2fv, GLfloat)
STUB_UNIFORM_VECTOR(mglUniform2iv, GLint)
STUB_UNIFORM_VECTOR(mglUniform2uiv, GLuint)
STUB_UNIFORM_VECTOR(mglUniform3fv, GLfloat)
STUB_UNIFORM_VECTOR(mglUniform3iv, GLint)
STUB_UNIFORM_VECTOR(mglUniform3uiv, GLuint)
STUB_UNIFORM_VECTOR(mglUniform4fv, GLfloat)
STUB_UNIFORM_VECTOR(mglUniform4iv, GLint)
STUB_UNIFORM_VECTOR(mglUniform4uiv, GLuint)

#undef STUB_UNIFORM_SCALAR
#undef STUB_UNIFORM_VECTOR
