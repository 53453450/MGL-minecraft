/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
 * non_core_unimplemented.c
 * MGL
 *
 */

#include "mgl.h"

/*
 * Helper macros for unimplemented legacy GL functions.
 * Each emits a one-time warning and sets GL_INVALID_OPERATION
 * instead of crashing via assert(0).
 */
#define MGL_UNIMPLEMENTED() do {                                              \
    static int _once = 0;                                                     \
    if (!_once) { _once = 1;                                                  \
      fprintf(stderr, "MGL: unimplemented %s called\n", __FUNCTION__);        \
    }                                                                         \
    ERROR_RETURN(GL_INVALID_OPERATION);                                       \
    return;                                                                   \
} while(0)

#define MGL_UNIMPLEMENTED_RETURN(val) do {                                    \
    static int _once = 0;                                                     \
    if (!_once) { _once = 1;                                                  \
      fprintf(stderr, "MGL: unimplemented %s called\n", __FUNCTION__);        \
    }                                                                         \
    ERROR_RETURN_VALUE(GL_INVALID_OPERATION, (val));                          \
} while(0)

/* Legacy GL attribute-group mask bits not present in the core profile header.
 * Defined locally so glPushAttrib / glPopAttrib can test them. */
#ifndef GL_VIEWPORT_BIT
#define GL_VIEWPORT_BIT             0x00000800
#endif
#ifndef GL_TRANSFORM_BIT
#define GL_TRANSFORM_BIT            0x00001000
#endif
#ifndef GL_ENABLE_BIT
#define GL_ENABLE_BIT               0x00002000
#endif
#ifndef GL_CLIENT_PIXEL_STORE_BIT
#define GL_CLIENT_PIXEL_STORE_BIT   0x00000001
#endif
#ifndef GL_CLIENT_VERTEX_ARRAY_BIT
#define GL_CLIENT_VERTEX_ARRAY_BIT  0x00000002
#endif

void mglArrayElement(GLMContext ctx, GLint i)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglBegin(GLMContext ctx, GLenum mode)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNewList(GLMContext ctx, GLuint list, GLenum mode)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEndList(GLMContext ctx)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglCallList(GLMContext ctx, GLuint list)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglCallLists(GLMContext ctx, GLsizei n, GLenum type, const void *lists)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglDeleteLists(GLMContext ctx, GLuint list, GLsizei range)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

GLuint mglGenLists(GLMContext ctx, GLsizei range)
{
    GLuint ret = 0;

    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED_RETURN(ret);
}

void mglListBase(GLMContext ctx, GLuint base)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglBitmap(GLMContext ctx, GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove, const GLubyte *bitmap)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3b(GLMContext ctx, GLbyte red, GLbyte green, GLbyte blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3bv(GLMContext ctx, const GLbyte *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3d(GLMContext ctx, GLdouble red, GLdouble green, GLdouble blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3f(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3i(GLMContext ctx, GLint red, GLint green, GLint blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3s(GLMContext ctx, GLshort red, GLshort green, GLshort blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3ub(GLMContext ctx, GLubyte red, GLubyte green, GLubyte blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3ubv(GLMContext ctx, const GLubyte *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3ui(GLMContext ctx, GLuint red, GLuint green, GLuint blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3uiv(GLMContext ctx, const GLuint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3us(GLMContext ctx, GLushort red, GLushort green, GLushort blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor3usv(GLMContext ctx, const GLushort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4b(GLMContext ctx, GLbyte red, GLbyte green, GLbyte blue, GLbyte alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4bv(GLMContext ctx, const GLbyte *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4d(GLMContext ctx, GLdouble red, GLdouble green, GLdouble blue, GLdouble alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4f(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4i(GLMContext ctx, GLint red, GLint green, GLint blue, GLint alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4s(GLMContext ctx, GLshort red, GLshort green, GLshort blue, GLshort alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4ub(GLMContext ctx, GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4ubv(GLMContext ctx, const GLubyte *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4ui(GLMContext ctx, GLuint red, GLuint green, GLuint blue, GLuint alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4uiv(GLMContext ctx, const GLuint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4us(GLMContext ctx, GLushort red, GLushort green, GLushort blue, GLushort alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColor4usv(GLMContext ctx, const GLushort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEnd(GLMContext ctx)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEdgeFlag(GLMContext ctx, GLboolean flag)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEdgeFlagv(GLMContext ctx, const GLboolean *flag)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexd(GLMContext ctx, GLdouble c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexdv(GLMContext ctx, const GLdouble *c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexf(GLMContext ctx, GLfloat c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexfv(GLMContext ctx, const GLfloat *c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexi(GLMContext ctx, GLint c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexiv(GLMContext ctx, const GLint *c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexs(GLMContext ctx, GLshort c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexsv(GLMContext ctx, const GLshort *c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3b(GLMContext ctx, GLbyte nx, GLbyte ny, GLbyte nz)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3bv(GLMContext ctx, const GLbyte *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3d(GLMContext ctx, GLdouble nx, GLdouble ny, GLdouble nz)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3f(GLMContext ctx, GLfloat nx, GLfloat ny, GLfloat nz)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3i(GLMContext ctx, GLint nx, GLint ny, GLint nz)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3s(GLMContext ctx, GLshort nx, GLshort ny, GLshort nz)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormal3sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos2d(GLMContext ctx, GLdouble x, GLdouble y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos2dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos2f(GLMContext ctx, GLfloat x, GLfloat y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos2fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos2i(GLMContext ctx, GLint x, GLint y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos2iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos2s(GLMContext ctx, GLshort x, GLshort y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos2sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos3d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos3dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos3f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos3fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos3i(GLMContext ctx, GLint x, GLint y, GLint z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos3iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos3s(GLMContext ctx, GLshort x, GLshort y, GLshort z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos3sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos4d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos4dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos4f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos4fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos4i(GLMContext ctx, GLint x, GLint y, GLint z, GLint w)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos4iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos4s(GLMContext ctx, GLshort x, GLshort y, GLshort z, GLshort w)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRasterPos4sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRectd(GLMContext ctx, GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRectdv(GLMContext ctx, const GLdouble *v1, const GLdouble *v2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRectf(GLMContext ctx, GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRectfv(GLMContext ctx, const GLfloat *v1, const GLfloat *v2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRecti(GLMContext ctx, GLint x1, GLint y1, GLint x2, GLint y2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRectiv(GLMContext ctx, const GLint *v1, const GLint *v2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRects(GLMContext ctx, GLshort x1, GLshort y1, GLshort x2, GLshort y2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRectsv(GLMContext ctx, const GLshort *v1, const GLshort *v2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord1d(GLMContext ctx, GLdouble s)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord1dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord1f(GLMContext ctx, GLfloat s)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord1fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord1i(GLMContext ctx, GLint s)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord1iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord1s(GLMContext ctx, GLshort s)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord1sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord2d(GLMContext ctx, GLdouble s, GLdouble t)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord2dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord2f(GLMContext ctx, GLfloat s, GLfloat t)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord2fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord2i(GLMContext ctx, GLint s, GLint t)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord2iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord2s(GLMContext ctx, GLshort s, GLshort t)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord2sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord3d(GLMContext ctx, GLdouble s, GLdouble t, GLdouble r)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord3dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord3f(GLMContext ctx, GLfloat s, GLfloat t, GLfloat r)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord3fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord3i(GLMContext ctx, GLint s, GLint t, GLint r)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord3iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord3s(GLMContext ctx, GLshort s, GLshort t, GLshort r)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord3sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord4d(GLMContext ctx, GLdouble s, GLdouble t, GLdouble r, GLdouble q)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord4dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord4f(GLMContext ctx, GLfloat s, GLfloat t, GLfloat r, GLfloat q)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord4fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord4i(GLMContext ctx, GLint s, GLint t, GLint r, GLint q)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord4iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord4s(GLMContext ctx, GLshort s, GLshort t, GLshort r, GLshort q)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoord4sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex2d(GLMContext ctx, GLdouble x, GLdouble y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex2dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex2f(GLMContext ctx, GLfloat x, GLfloat y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex2fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex2i(GLMContext ctx, GLint x, GLint y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex2iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex2s(GLMContext ctx, GLshort x, GLshort y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex2sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex3d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex3dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex3f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex3fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex3i(GLMContext ctx, GLint x, GLint y, GLint z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex3iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex3s(GLMContext ctx, GLshort x, GLshort y, GLshort z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex3sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex4d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex4dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex4f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex4fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex4i(GLMContext ctx, GLint x, GLint y, GLint z, GLint w)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex4iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex4s(GLMContext ctx, GLshort x, GLshort y, GLshort z, GLshort w)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertex4sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglClipPlane(GLMContext ctx, GLenum plane, const GLdouble *equation)
{
    if (!ctx || !equation)
        return;

    if (plane < GL_CLIP_DISTANCE0 ||
        plane > GL_CLIP_DISTANCE0 + MAX_CLIP_DISTANCES - 1) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    GLuint index = (GLuint)(plane - GL_CLIP_DISTANCE0);
    GLuint limit = ctx->state.var.max_clip_distances;
    if (limit == 0 || limit > MAX_CLIP_DISTANCES)
        limit = MAX_CLIP_DISTANCES;
    if (index >= limit) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    /* GL 1.1 semantics: the equation is transformed to eye coordinates by
     * the inverse-transpose of the current modelview.  MGL's fixed-function
     * matrix stack is unimplemented (the matrices are app-set uniforms), so
     * the effective transform is identity and the equation is stored as
     * given.  A shader deriving clip distances must dot the plane against
     * gl_ClipVertex in the same (eye) space. */
    memcpy(ctx->state.var.clip_planes[index], equation,
           sizeof(ctx->state.var.clip_planes[index]));
    mglMarkStateDirtyBits(&ctx->state, DIRTY_STATE | DIRTY_RENDER_STATE);
}

void mglColorMaterial(GLMContext ctx, GLenum face, GLenum mode)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogf(GLMContext ctx, GLenum pname, GLfloat param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogi(GLMContext ctx, GLenum pname, GLint param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogiv(GLMContext ctx, GLenum pname, const GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLightf(GLMContext ctx, GLenum light, GLenum pname, GLfloat param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLightfv(GLMContext ctx, GLenum light, GLenum pname, const GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLighti(GLMContext ctx, GLenum light, GLenum pname, GLint param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLightiv(GLMContext ctx, GLenum light, GLenum pname, const GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLightModelf(GLMContext ctx, GLenum pname, GLfloat param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLightModelfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLightModeli(GLMContext ctx, GLenum pname, GLint param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLightModeliv(GLMContext ctx, GLenum pname, const GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLineStipple(GLMContext ctx, GLint factor, GLushort pattern)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMaterialf(GLMContext ctx, GLenum face, GLenum pname, GLfloat param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMaterialfv(GLMContext ctx, GLenum face, GLenum pname, const GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMateriali(GLMContext ctx, GLenum face, GLenum pname, GLint param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMaterialiv(GLMContext ctx, GLenum face, GLenum pname, const GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPolygonStipple(GLMContext ctx, const GLubyte *mask)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglShadeModel(GLMContext ctx, GLenum mode)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexEnvf(GLMContext ctx, GLenum target, GLenum pname, GLfloat param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexEnvfv(GLMContext ctx, GLenum target, GLenum pname, const GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexEnvi(GLMContext ctx, GLenum target, GLenum pname, GLint param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexEnviv(GLMContext ctx, GLenum target, GLenum pname, const GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexGend(GLMContext ctx, GLenum coord, GLenum pname, GLdouble param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexGendv(GLMContext ctx, GLenum coord, GLenum pname, const GLdouble *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexGenf(GLMContext ctx, GLenum coord, GLenum pname, GLfloat param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexGenfv(GLMContext ctx, GLenum coord, GLenum pname, const GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexGeni(GLMContext ctx, GLenum coord, GLenum pname, GLint param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexGeniv(GLMContext ctx, GLenum coord, GLenum pname, const GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFeedbackBuffer(GLMContext ctx, GLsizei size, GLenum type, GLfloat *buffer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSelectBuffer(GLMContext ctx, GLsizei size, GLuint *buffer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

GLint  mglRenderMode(GLMContext ctx, GLenum mode)
{
    GLint ret = -1;

    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED_RETURN(ret);
}

void mglPixelMapfv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLfloat *values)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPixelMapuiv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLuint *values)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPixelMapusv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLushort *values)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglCopyPixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum type)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglDrawPixels(GLMContext ctx, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetClipPlane(GLMContext ctx, GLenum plane, GLdouble *equation)
{
    if (!ctx || !equation)
        return;

    if (plane < GL_CLIP_DISTANCE0 ||
        plane > GL_CLIP_DISTANCE0 + MAX_CLIP_DISTANCES - 1) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    GLuint index = (GLuint)(plane - GL_CLIP_DISTANCE0);
    GLuint limit = ctx->state.var.max_clip_distances;
    if (limit == 0 || limit > MAX_CLIP_DISTANCES)
        limit = MAX_CLIP_DISTANCES;
    if (index >= limit) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    memcpy(equation, ctx->state.var.clip_planes[index],
           sizeof(ctx->state.var.clip_planes[index]));
}

void mglGetLightfv(GLMContext ctx, GLenum light, GLenum pname, GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetLightiv(GLMContext ctx, GLenum light, GLenum pname, GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetMapdv(GLMContext ctx, GLenum target, GLenum query, GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetMapfv(GLMContext ctx, GLenum target, GLenum query, GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetMapiv(GLMContext ctx, GLenum target, GLenum query, GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetMaterialfv(GLMContext ctx, GLenum face, GLenum pname, GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetMaterialiv(GLMContext ctx, GLenum face, GLenum pname, GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetPixelMapfv(GLMContext ctx, GLenum map, GLfloat *values)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetPixelMapuiv(GLMContext ctx, GLenum map, GLuint *values)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetPixelMapusv(GLMContext ctx, GLenum map, GLushort *values)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetPolygonStipple(GLMContext ctx, GLubyte *mask)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetTexEnvfv(GLMContext ctx, GLenum target, GLenum pname, GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetTexEnviv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetTexGendv(GLMContext ctx, GLenum coord, GLenum pname, GLdouble *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetTexGenfv(GLMContext ctx, GLenum coord, GLenum pname, GLfloat *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetTexGeniv(GLMContext ctx, GLenum coord, GLenum pname, GLint *params)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

GLboolean mglIsList(GLMContext ctx, GLuint list)
{
    GLboolean ret = 0;

    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED_RETURN(ret);
}

void mglFrustum(GLMContext ctx, GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLoadIdentity(GLMContext ctx)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLoadMatrixf(GLMContext ctx, const GLfloat *m)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLoadMatrixd(GLMContext ctx, const GLdouble *m)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMatrixMode(GLMContext ctx, GLenum mode)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultMatrixf(GLMContext ctx, const GLfloat *m)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultMatrixd(GLMContext ctx, const GLdouble *m)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglOrtho(GLMContext ctx, GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPopMatrix(GLMContext ctx)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPushMatrix(GLMContext ctx)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRotated(GLMContext ctx, GLdouble angle, GLdouble x, GLdouble y, GLdouble z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglRotatef(GLMContext ctx, GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglScaled(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglScalef(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTranslated(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTranslatef(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}


void mglInitNames(GLMContext ctx)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLoadName(GLMContext ctx, GLuint name)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPassThrough(GLMContext ctx, GLfloat token)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPopName(GLMContext ctx)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPushName(GLMContext ctx, GLuint name)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglClearAccum(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglClearIndex(GLMContext ctx, GLfloat c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexMask(GLMContext ctx, GLuint mask)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglAccum(GLMContext ctx, GLenum op, GLfloat value)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPushAttrib(GLMContext ctx, GLbitfield mask)
{
    if (ctx->state.var.attrib_stack_depth >= MGL_ATTRIB_STACK_DEPTH)
    {
        ERROR_RETURN(GL_STACK_OVERFLOW);
        return;
    }

    MGLAttribStackEntry *entry = &ctx->state.var.attrib_stack[ctx->state.var.attrib_stack_depth];
    memset(entry, 0, sizeof(*entry));
    entry->mask = mask;

    if (mask & GL_COLOR_BUFFER_BIT)
    {
        memcpy(entry->blend_color, ctx->state.var.blend_color, sizeof(entry->blend_color));
        memcpy(entry->blend_src_rgb, ctx->state.var.blend_src_rgb, sizeof(entry->blend_src_rgb));
        memcpy(entry->blend_src_alpha, ctx->state.var.blend_src_alpha, sizeof(entry->blend_src_alpha));
        memcpy(entry->blend_dst_rgb, ctx->state.var.blend_dst_rgb, sizeof(entry->blend_dst_rgb));
        memcpy(entry->blend_dst_alpha, ctx->state.var.blend_dst_alpha, sizeof(entry->blend_dst_alpha));
        memcpy(entry->blend_equation_rgb, ctx->state.var.blend_equation_rgb, sizeof(entry->blend_equation_rgb));
        memcpy(entry->blend_equation_alpha, ctx->state.var.blend_equation_alpha, sizeof(entry->blend_equation_alpha));
        memcpy(entry->color_writemask, ctx->state.var.color_writemask, sizeof(entry->color_writemask));
        memcpy(entry->use_color_mask, ctx->state.caps.use_color_mask, sizeof(entry->use_color_mask));
        entry->blend_enabled = ctx->state.caps.blend;
        memcpy(entry->blendi, ctx->state.caps.blendi, sizeof(entry->blendi));
        entry->logic_op_enabled = ctx->state.caps.color_logic_op;
        entry->logic_op_mode = ctx->state.var.logic_op_mode;
    }

    if (mask & GL_DEPTH_BUFFER_BIT)
    {
        entry->depth_test_enabled = ctx->state.caps.depth_test;
        entry->depth_writemask = ctx->state.var.depth_writemask;
        entry->depth_func = ctx->state.var.depth_func;
        entry->depth_near = ctx->state.var.depth_range[0];
        entry->depth_far = ctx->state.var.depth_range[1];
    }

    if (mask & GL_STENCIL_BUFFER_BIT)
    {
        entry->stencil_test_enabled = ctx->state.caps.stencil_test;
        entry->stencil_writemask = ctx->state.var.stencil_writemask;
        entry->stencil_back_writemask = ctx->state.var.stencil_back_writemask;
    }

    if (mask & GL_ENABLE_BIT)
    {
        /* Save all relevant enable states (some overlap with buffer bits). */
        entry->depth_test_enabled = ctx->state.caps.depth_test;
        entry->stencil_test_enabled = ctx->state.caps.stencil_test;
        entry->blend_enabled = ctx->state.caps.blend;
        memcpy(entry->blendi, ctx->state.caps.blendi, sizeof(entry->blendi));
        entry->color_logic_op_enabled = ctx->state.caps.color_logic_op;
        entry->cull_face_enabled = ctx->state.caps.cull_face;
        entry->dither_enabled = ctx->state.caps.dither;
        entry->polygon_offset_fill_enabled = ctx->state.caps.polygon_offset_fill;
        entry->sample_alpha_to_coverage_enabled = ctx->state.caps.sample_alpha_to_coverage;
        entry->sample_coverage_enabled = ctx->state.caps.sample_coverage;
        entry->scissor_test_enabled = ctx->state.caps.scissor_test;
    }

    if (mask & GL_TRANSFORM_BIT)
    {
        entry->clip_origin = ctx->state.var.clip_origin;
        entry->clip_depth_mode = ctx->state.var.clip_depth_mode;
    }

    if (mask & GL_VIEWPORT_BIT)
    {
        memcpy(entry->viewport, ctx->state.viewport, sizeof(entry->viewport));
        entry->depth_range[0] = ctx->state.var.depth_range[0];
        entry->depth_range[1] = ctx->state.var.depth_range[1];
        memcpy(entry->scissor_box, ctx->state.var.scissor_box, sizeof(entry->scissor_box));
    }

    ctx->state.var.attrib_stack_depth++;
}

void mglPopAttrib(GLMContext ctx)
{
    if (ctx->state.var.attrib_stack_depth == 0)
    {
        ERROR_RETURN(GL_STACK_UNDERFLOW);
        return;
    }

    ctx->state.var.attrib_stack_depth--;
    MGLAttribStackEntry *entry = &ctx->state.var.attrib_stack[ctx->state.var.attrib_stack_depth];
    GLbitfield mask = entry->mask;

    if (mask & GL_COLOR_BUFFER_BIT)
    {
        mglBlendColor(ctx, entry->blend_color[0], entry->blend_color[1],
                      entry->blend_color[2], entry->blend_color[3]);
        for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
        {
            mglBlendFuncSeparatei(ctx, i,
                                  entry->blend_src_rgb[i], entry->blend_dst_rgb[i],
                                  entry->blend_src_alpha[i], entry->blend_dst_alpha[i]);
            mglBlendEquationSeparatei(ctx, i,
                                     entry->blend_equation_rgb[i],
                                     entry->blend_equation_alpha[i]);
            mglColorMaski(ctx, i,
                          entry->color_writemask[i][0], entry->color_writemask[i][1],
                          entry->color_writemask[i][2], entry->color_writemask[i][3]);
        }
        /* Restore per-attachment blend enables; caps.blend is recomputed
         * automatically by mglRecomputeGlobalBlendEnable. */
        for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
        {
            if (entry->blendi[i])
                mglEnablei(ctx, GL_BLEND, i);
            else
                mglDisablei(ctx, GL_BLEND, i);
        }
        if (entry->logic_op_enabled)
            mglEnable(ctx, GL_COLOR_LOGIC_OP);
        else
            mglDisable(ctx, GL_COLOR_LOGIC_OP);
        mglLogicOp(ctx, entry->logic_op_mode);
    }

    if (mask & GL_DEPTH_BUFFER_BIT)
    {
        if (entry->depth_test_enabled)
            mglEnable(ctx, GL_DEPTH_TEST);
        else
            mglDisable(ctx, GL_DEPTH_TEST);
        mglDepthMask(ctx, entry->depth_writemask);
        mglDepthFunc(ctx, entry->depth_func);
        mglDepthRange(ctx, entry->depth_near, entry->depth_far);
    }

    if (mask & GL_STENCIL_BUFFER_BIT)
    {
        if (entry->stencil_test_enabled)
            mglEnable(ctx, GL_STENCIL_TEST);
        else
            mglDisable(ctx, GL_STENCIL_TEST);
        mglStencilMaskSeparate(ctx, GL_FRONT, entry->stencil_writemask);
        mglStencilMaskSeparate(ctx, GL_BACK, entry->stencil_back_writemask);
    }

    if (mask & GL_ENABLE_BIT)
    {
        if (entry->depth_test_enabled)
            mglEnable(ctx, GL_DEPTH_TEST);
        else
            mglDisable(ctx, GL_DEPTH_TEST);
        if (entry->stencil_test_enabled)
            mglEnable(ctx, GL_STENCIL_TEST);
        else
            mglDisable(ctx, GL_STENCIL_TEST);
        for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
        {
            if (entry->blendi[i])
                mglEnablei(ctx, GL_BLEND, i);
            else
                mglDisablei(ctx, GL_BLEND, i);
        }
        if (entry->color_logic_op_enabled)
            mglEnable(ctx, GL_COLOR_LOGIC_OP);
        else
            mglDisable(ctx, GL_COLOR_LOGIC_OP);
        if (entry->cull_face_enabled)
            mglEnable(ctx, GL_CULL_FACE);
        else
            mglDisable(ctx, GL_CULL_FACE);
        if (entry->dither_enabled)
            mglEnable(ctx, GL_DITHER);
        else
            mglDisable(ctx, GL_DITHER);
        if (entry->polygon_offset_fill_enabled)
            mglEnable(ctx, GL_POLYGON_OFFSET_FILL);
        else
            mglDisable(ctx, GL_POLYGON_OFFSET_FILL);
        if (entry->sample_alpha_to_coverage_enabled)
            mglEnable(ctx, GL_SAMPLE_ALPHA_TO_COVERAGE);
        else
            mglDisable(ctx, GL_SAMPLE_ALPHA_TO_COVERAGE);
        if (entry->sample_coverage_enabled)
            mglEnable(ctx, GL_SAMPLE_COVERAGE);
        else
            mglDisable(ctx, GL_SAMPLE_COVERAGE);
        if (entry->scissor_test_enabled)
            mglEnable(ctx, GL_SCISSOR_TEST);
        else
            mglDisable(ctx, GL_SCISSOR_TEST);
    }

    if (mask & GL_TRANSFORM_BIT)
    {
        mglClipControl(ctx, entry->clip_origin, entry->clip_depth_mode);
    }

    if (mask & GL_VIEWPORT_BIT)
    {
        mglViewport(ctx, entry->viewport[0], entry->viewport[1],
                    entry->viewport[2], entry->viewport[3]);
        mglDepthRange(ctx, entry->depth_range[0], entry->depth_range[1]);
        mglScissor(ctx, entry->scissor_box[0], entry->scissor_box[1],
                   entry->scissor_box[2], entry->scissor_box[3]);
    }
}

void mglMap1d(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble *points)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMap1f(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint stride, GLint order, const GLfloat *points)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMap2d(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder, GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble *points)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMap2f(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder, GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, const GLfloat *points)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMapGrid1d(GLMContext ctx, GLint un, GLdouble u1, GLdouble u2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMapGrid1f(GLMContext ctx, GLint un, GLfloat u1, GLfloat u2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMapGrid2d(GLMContext ctx, GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMapGrid2f(GLMContext ctx, GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1, GLfloat v2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalCoord1d(GLMContext ctx, GLdouble u)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalCoord1dv(GLMContext ctx, const GLdouble *u)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalCoord1f(GLMContext ctx, GLfloat u)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalCoord1fv(GLMContext ctx, const GLfloat *u)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalCoord2d(GLMContext ctx, GLdouble u, GLdouble v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalCoord2dv(GLMContext ctx, const GLdouble *u)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalCoord2f(GLMContext ctx, GLfloat u, GLfloat v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalCoord2fv(GLMContext ctx, const GLfloat *u)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalMesh1(GLMContext ctx, GLenum mode, GLint i1, GLint i2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalPoint1(GLMContext ctx, GLint i)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalMesh2(GLMContext ctx, GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEvalPoint2(GLMContext ctx, GLint i, GLint j)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglAlphaFunc(GLMContext ctx, GLenum func, GLfloat ref)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPixelZoom(GLMContext ctx, GLfloat xfactor, GLfloat yfactor)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPixelTransferf(GLMContext ctx, GLenum pname, GLfloat param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPixelTransferi(GLMContext ctx, GLenum pname, GLint param)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetnPolygonStipple(GLMContext ctx, GLsizei bufSize, GLubyte *pattern)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetnColorTable(GLMContext ctx, GLenum target, GLenum format, GLenum type, GLsizei bufSize, void *table)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetnConvolutionFilter(GLMContext ctx, GLenum target, GLenum format, GLenum type, GLsizei bufSize, void *image)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetnSeparableFilter(GLMContext ctx, GLenum target, GLenum format, GLenum type, GLsizei rowBufSize, void *row, GLsizei columnBufSize, void *column, void *span)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetnHistogram(GLMContext ctx, GLenum target, GLboolean reset, GLenum format, GLenum type, GLsizei bufSize, void *values)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglGetnMinmax(GLMContext ctx, GLenum target, GLboolean reset, GLenum format, GLenum type, GLsizei bufSize, void *values)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglColorPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglDisableClientState(GLMContext ctx, GLenum array)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEdgeFlagPointer(GLMContext ctx, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglEnableClientState(GLMContext ctx, GLenum array)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglInterleavedArrays(GLMContext ctx, GLenum format, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglNormalPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglVertexPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogCoordf(GLMContext ctx, GLfloat coord)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogCoordfv(GLMContext ctx, const GLfloat *coord)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogCoordd(GLMContext ctx, GLdouble coord)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogCoorddv(GLMContext ctx, const GLdouble *coord)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglFogCoordPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3b(GLMContext ctx, GLbyte red, GLbyte green, GLbyte blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3bv(GLMContext ctx, const GLbyte *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3d(GLMContext ctx, GLdouble red, GLdouble green, GLdouble blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3f(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3i(GLMContext ctx, GLint red, GLint green, GLint blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3s(GLMContext ctx, GLshort red, GLshort green, GLshort blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3ub(GLMContext ctx, GLubyte red, GLubyte green, GLubyte blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3ubv(GLMContext ctx, const GLubyte *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3ui(GLMContext ctx, GLuint red, GLuint green, GLuint blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3uiv(GLMContext ctx, const GLuint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3us(GLMContext ctx, GLushort red, GLushort green, GLushort blue)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColor3usv(GLMContext ctx, const GLushort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglSecondaryColorPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos2d(GLMContext ctx, GLdouble x, GLdouble y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos2dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos2f(GLMContext ctx, GLfloat x, GLfloat y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos2fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos2i(GLMContext ctx, GLint x, GLint y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos2iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos2s(GLMContext ctx, GLshort x, GLshort y)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos2sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos3d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos3dv(GLMContext ctx, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos3f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos3fv(GLMContext ctx, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos3i(GLMContext ctx, GLint x, GLint y, GLint z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos3iv(GLMContext ctx, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos3s(GLMContext ctx, GLshort x, GLshort y, GLshort z)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglWindowPos3sv(GLMContext ctx, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordP1ui(GLMContext ctx, GLenum type, GLuint coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordP1uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordP2ui(GLMContext ctx, GLenum type, GLuint coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordP2uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordP3ui(GLMContext ctx, GLenum type, GLuint coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordP3uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordP4ui(GLMContext ctx, GLenum type, GLuint coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglTexCoordP4uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoordP1ui(GLMContext ctx, GLenum texture, GLenum type, GLuint coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoordP1uiv(GLMContext ctx, GLenum texture, GLenum type, const GLuint *coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoordP2ui(GLMContext ctx, GLenum texture, GLenum type, GLuint coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoordP2uiv(GLMContext ctx, GLenum texture, GLenum type, const GLuint *coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoordP3ui(GLMContext ctx, GLenum texture, GLenum type, GLuint coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoordP3uiv(GLMContext ctx, GLenum texture, GLenum type, const GLuint *coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoordP4ui(GLMContext ctx, GLenum texture, GLenum type, GLuint coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoordP4uiv(GLMContext ctx, GLenum texture, GLenum type, const GLuint *coords)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord1d(GLMContext ctx, GLenum target, GLdouble s)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord1dv(GLMContext ctx, GLenum target, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord1f(GLMContext ctx, GLenum target, GLfloat s)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord1fv(GLMContext ctx, GLenum target, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord1i(GLMContext ctx, GLenum target, GLint s)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord1iv(GLMContext ctx, GLenum target, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord1s(GLMContext ctx, GLenum target, GLshort s)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord1sv(GLMContext ctx, GLenum target, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord2d(GLMContext ctx, GLenum target, GLdouble s, GLdouble t)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord2dv(GLMContext ctx, GLenum target, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord2f(GLMContext ctx, GLenum target, GLfloat s, GLfloat t)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord2fv(GLMContext ctx, GLenum target, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord2i(GLMContext ctx, GLenum target, GLint s, GLint t)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord2iv(GLMContext ctx, GLenum target, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord2s(GLMContext ctx, GLenum target, GLshort s, GLshort t)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord2sv(GLMContext ctx, GLenum target, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord3d(GLMContext ctx, GLenum target, GLdouble s, GLdouble t, GLdouble r)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord3dv(GLMContext ctx, GLenum target, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord3f(GLMContext ctx, GLenum target, GLfloat s, GLfloat t, GLfloat r)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord3fv(GLMContext ctx, GLenum target, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord3i(GLMContext ctx, GLenum target, GLint s, GLint t, GLint r)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord3iv(GLMContext ctx, GLenum target, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord3s(GLMContext ctx, GLenum target, GLshort s, GLshort t, GLshort r)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord3sv(GLMContext ctx, GLenum target, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord4d(GLMContext ctx, GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord4dv(GLMContext ctx, GLenum target, const GLdouble *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord4f(GLMContext ctx, GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord4fv(GLMContext ctx, GLenum target, const GLfloat *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord4i(GLMContext ctx, GLenum target, GLint s, GLint t, GLint r, GLint q)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord4iv(GLMContext ctx, GLenum target, const GLint *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord4s(GLMContext ctx, GLenum target, GLshort s, GLshort t, GLshort r, GLshort q)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultiTexCoord4sv(GLMContext ctx, GLenum target, const GLshort *v)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLoadTransposeMatrixf(GLMContext ctx, const GLfloat *m)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglLoadTransposeMatrixd(GLMContext ctx, const GLdouble *m)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultTransposeMatrixf(GLMContext ctx, const GLfloat *m)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglMultTransposeMatrixd(GLMContext ctx, const GLdouble *m)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexub(GLMContext ctx, GLubyte c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglIndexubv(GLMContext ctx, const GLubyte *c)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}

void mglPushClientAttrib(GLMContext ctx, GLbitfield mask)
{
    if (ctx->state.var.client_attrib_stack_depth >= MGL_ATTRIB_STACK_DEPTH)
    {
        ERROR_RETURN(GL_STACK_OVERFLOW);
        return;
    }

    MGLClientAttribStackEntry *entry = &ctx->state.var.client_attrib_stack[ctx->state.var.client_attrib_stack_depth];
    memset(entry, 0, sizeof(*entry));
    entry->mask = mask;

    if (mask & GL_CLIENT_VERTEX_ARRAY_BIT)
    {
        entry->vertex_array_binding = ctx->state.var.vertex_array_binding;
    }

    if (mask & GL_CLIENT_PIXEL_STORE_BIT)
    {
        entry->pixel_pack_buffer_binding = ctx->state.var.pixel_pack_buffer_binding;
        entry->pixel_unpack_buffer_binding = ctx->state.var.pixel_unpack_buffer_binding;
    }

    ctx->state.var.client_attrib_stack_depth++;
}

void mglPopClientAttrib(GLMContext ctx)
{
    if (ctx->state.var.client_attrib_stack_depth == 0)
    {
        ERROR_RETURN(GL_STACK_UNDERFLOW);
        return;
    }

    ctx->state.var.client_attrib_stack_depth--;
    MGLClientAttribStackEntry *entry = &ctx->state.var.client_attrib_stack[ctx->state.var.client_attrib_stack_depth];
    GLbitfield mask = entry->mask;

    if (mask & GL_CLIENT_VERTEX_ARRAY_BIT)
    {
        mglBindVertexArray(ctx, entry->vertex_array_binding);
    }

    if (mask & GL_CLIENT_PIXEL_STORE_BIT)
    {
        mglBindBuffer(ctx, GL_PIXEL_PACK_BUFFER, entry->pixel_pack_buffer_binding);
        mglBindBuffer(ctx, GL_PIXEL_UNPACK_BUFFER, entry->pixel_unpack_buffer_binding);
    }
}

GLboolean mglAreTexturesResident(GLMContext ctx, GLsizei n, const GLuint *textures, GLboolean *residences)
{
    GLboolean ret = 0;

    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED_RETURN(ret);
}

void mglPrioritizeTextures(GLMContext ctx, GLsizei n, const GLuint *textures, const GLfloat *priorities)
{
    // Unimplemented function — deprecated GL function
    MGL_UNIMPLEMENTED();
}


