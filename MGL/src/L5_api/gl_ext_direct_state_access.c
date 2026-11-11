/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Selector-free EXT_direct_state_access entry points.  The forwarding scopes
 * implement the extension's object-creation and selector-restoration rules;
 * the actual GL operation remains in the existing core or legacy entry point.
 */

#include "glcorearb.h"
#include <stddef.h>
#include "mgl/api.h"
#include "mgl_types_texture.h"

#ifndef GL_CLIENT_ACTIVE_TEXTURE
#define GL_CLIENT_ACTIVE_TEXTURE 0x84E1
#endif
#ifndef GL_MATRIX_MODE
#define GL_MATRIX_MODE 0x0BA0
#endif
#ifndef GL_MODELVIEW
#define GL_MODELVIEW 0x1700
#endif
#ifndef GL_TEXTURE_COORD_ARRAY
#define GL_TEXTURE_COORD_ARRAY 0x8078
#endif
#ifndef GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING 0x889A
#endif
#ifndef GL_TEXTURE_COORD_ARRAY_POINTER
#define GL_TEXTURE_COORD_ARRAY_POINTER 0x8092
#endif
#ifndef GL_TEXTURE_COORD_ARRAY_SIZE
#define GL_TEXTURE_COORD_ARRAY_SIZE 0x8088
#endif
#ifndef GL_TEXTURE_COORD_ARRAY_STRIDE
#define GL_TEXTURE_COORD_ARRAY_STRIDE 0x808A
#endif
#ifndef GL_TEXTURE_COORD_ARRAY_TYPE
#define GL_TEXTURE_COORD_ARRAY_TYPE 0x8089
#endif

extern void glClientActiveTexture(GLenum texture);
extern void glColorPointer(GLint size, GLenum type, GLsizei stride, const void *pointer);
extern void glDisableClientState(GLenum array);
extern void glEdgeFlagPointer(GLsizei stride, const void *pointer);
extern void glEnableClientState(GLenum array);
extern void glFogCoordPointer(GLenum type, GLsizei stride, const void *pointer);
extern void glFrustum(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
extern void glGetTexEnvfv(GLenum target, GLenum pname, GLfloat *params);
extern void glGetTexEnviv(GLenum target, GLenum pname, GLint *params);
extern void glGetTexGendv(GLenum coord, GLenum pname, GLdouble *params);
extern void glGetTexGenfv(GLenum coord, GLenum pname, GLfloat *params);
extern void glGetTexGeniv(GLenum coord, GLenum pname, GLint *params);
extern void glIndexPointer(GLenum type, GLsizei stride, const void *pointer);
extern void glLoadIdentity();
extern void glLoadMatrixd(const GLdouble *m);
extern void glLoadMatrixf(const GLfloat *m);
extern void glLoadTransposeMatrixd(const GLdouble *m);
extern void glLoadTransposeMatrixf(const GLfloat *m);
extern void glMatrixMode(GLenum mode);
extern void glMultMatrixd(const GLdouble *m);
extern void glMultMatrixf(const GLfloat *m);
extern void glMultTransposeMatrixd(const GLdouble *m);
extern void glMultTransposeMatrixf(const GLfloat *m);
extern void glNormalPointer(GLenum type, GLsizei stride, const void *pointer);
extern void glOrtho(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
extern void glPopMatrix();
extern void glPushClientAttrib(GLbitfield mask);
extern void glPushMatrix();
extern void glRotated(GLdouble angle, GLdouble x, GLdouble y, GLdouble z);
extern void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
extern void glScaled(GLdouble x, GLdouble y, GLdouble z);
extern void glScalef(GLfloat x, GLfloat y, GLfloat z);
extern void glSecondaryColorPointer(GLint size, GLenum type, GLsizei stride, const void *pointer);
extern void glTexCoordPointer(GLint size, GLenum type, GLsizei stride, const void *pointer);
extern void glTexEnvf(GLenum target, GLenum pname, GLfloat param);
extern void glTexEnvfv(GLenum target, GLenum pname, const GLfloat *params);
extern void glTexEnvi(GLenum target, GLenum pname, GLint param);
extern void glTexEnviv(GLenum target, GLenum pname, const GLint *params);
extern void glTexGend(GLenum coord, GLenum pname, GLdouble param);
extern void glTexGendv(GLenum coord, GLenum pname, const GLdouble *params);
extern void glTexGenf(GLenum coord, GLenum pname, GLfloat param);
extern void glTexGenfv(GLenum coord, GLenum pname, const GLfloat *params);
extern void glTexGeni(GLenum coord, GLenum pname, GLint param);
extern void glTexGeniv(GLenum coord, GLenum pname, const GLint *params);
extern void glTranslated(GLdouble x, GLdouble y, GLdouble z);
extern void glTranslatef(GLfloat x, GLfloat y, GLfloat z);
extern void glVertexPointer(GLint size, GLenum type, GLsizei stride, const void *pointer);


#define GET_CONTEXT()   (mgl_lazy_init(), MGLgetCurrentContext())

extern void mgl_lazy_init(void);
extern void mglDispatchError(GLMContext ctx, const char *func, GLenum error);
extern Buffer *getBuffer(GLMContext ctx, GLenum target, GLuint buffer);
extern Texture *findTexture(GLMContext ctx, GLuint texture);

static void extError(const char *caller, GLenum error)
{
    GLMContext ctx = GET_CONTEXT();
    if (ctx)
        mglDispatchError(ctx, caller, error);
}

static void extUnsupported(const char *caller)
{
    /* These optional ARB/NV program and renderbuffer-texture interfaces have
     * no underlying implementation.  Fail explicitly instead of leaving a
     * NULL function pointer in extension loaders. */
    extError(caller, GL_INVALID_OPERATION);
}

typedef struct {
    GLint saved;
    GLboolean changed;
} ExtUnitScope;

static GLboolean extUnitBegin(ExtUnitScope *s, GLenum texunit,
                              const char *caller)
{
    GLint limit = 0;
    s->saved = GL_TEXTURE0;
    s->changed = GL_FALSE;
    if (!texunit)
        return GL_TRUE;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &limit);
    if (texunit < GL_TEXTURE0 ||
        texunit >= GL_TEXTURE0 + (GLenum)limit) {
        extError(caller, GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    glGetIntegerv(GL_ACTIVE_TEXTURE, &s->saved);
    if ((GLenum)s->saved != texunit) {
        s->changed = GL_TRUE;
        glActiveTexture(texunit);
    }
    return GL_TRUE;
}

static void extUnitEnd(ExtUnitScope *s)
{
    if (s->changed)
        glActiveTexture((GLenum)s->saved);
}

static GLboolean extClientUnitBegin(ExtUnitScope *s, GLenum texunit,
                                    const char *caller)
{
    GLint limit = 0;
    s->saved = GL_TEXTURE0;
    s->changed = GL_FALSE;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &limit);
    if (texunit < GL_TEXTURE0 ||
        texunit >= GL_TEXTURE0 + (GLenum)limit) {
        extError(caller, GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    glGetIntegerv(GL_CLIENT_ACTIVE_TEXTURE, &s->saved);
    if ((GLenum)s->saved != texunit) {
        s->changed = GL_TRUE;
        glClientActiveTexture(texunit);
    }
    return GL_TRUE;
}

static void extClientUnitEnd(ExtUnitScope *s)
{
    if (s->changed)
        glClientActiveTexture((GLenum)s->saved);
}

static GLenum extObjectTarget(GLenum target)
{
    switch (target) {
    case GL_TEXTURE_1D:
    case GL_TEXTURE_2D:
    case GL_TEXTURE_3D:
    case GL_TEXTURE_CUBE_MAP:
    case GL_TEXTURE_RECTANGLE:
    case GL_TEXTURE_1D_ARRAY:
    case GL_TEXTURE_2D_ARRAY:
    case GL_TEXTURE_CUBE_MAP_ARRAY:
    case GL_TEXTURE_BUFFER:
    case GL_TEXTURE_2D_MULTISAMPLE:
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        return target;
    case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        return GL_TEXTURE_CUBE_MAP;
    default:
        return 0;
    }
}

static GLenum extBindingPname(GLenum object_target)
{
    switch (object_target) {
    case GL_TEXTURE_1D: return GL_TEXTURE_BINDING_1D;
    case GL_TEXTURE_2D: return GL_TEXTURE_BINDING_2D;
    case GL_TEXTURE_3D: return GL_TEXTURE_BINDING_3D;
    case GL_TEXTURE_CUBE_MAP: return GL_TEXTURE_BINDING_CUBE_MAP;
    case GL_TEXTURE_RECTANGLE: return GL_TEXTURE_BINDING_RECTANGLE;
    case GL_TEXTURE_1D_ARRAY: return GL_TEXTURE_BINDING_1D_ARRAY;
    case GL_TEXTURE_2D_ARRAY: return GL_TEXTURE_BINDING_2D_ARRAY;
    case GL_TEXTURE_CUBE_MAP_ARRAY: return GL_TEXTURE_BINDING_CUBE_MAP_ARRAY;
    case GL_TEXTURE_BUFFER: return GL_TEXTURE_BINDING_BUFFER;
    case GL_TEXTURE_2D_MULTISAMPLE: return GL_TEXTURE_BINDING_2D_MULTISAMPLE;
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        return GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY;
    default: return 0;
    }
}

static GLboolean extProxyTarget(GLenum target)
{
    switch (target) {
    case GL_PROXY_TEXTURE_1D:
    case GL_PROXY_TEXTURE_2D:
    case GL_PROXY_TEXTURE_3D:
    case GL_PROXY_TEXTURE_CUBE_MAP:
    case GL_PROXY_TEXTURE_RECTANGLE:
        return GL_TRUE;
    default:
        return GL_FALSE;
    }
}

typedef struct {
    ExtUnitScope unit;
    GLenum object_target;
    GLint previous;
    GLboolean bound;
} ExtTextureScope;

static void extTextureEnd(ExtTextureScope *s);

static GLboolean extTextureBegin(ExtTextureScope *s, GLuint texture,
                                 GLenum target, GLenum texunit,
                                 const char *caller)
{
    s->object_target = extObjectTarget(target);
    s->previous = 0;
    s->bound = GL_FALSE;
    if (!extUnitBegin(&s->unit, texunit, caller))
        return GL_FALSE;
    if (extProxyTarget(target)) {
        if (texture) {
            extError(caller, GL_INVALID_OPERATION);
            extUnitEnd(&s->unit);
            return GL_FALSE;
        }
        return GL_TRUE;
    }
    if (!s->object_target) {
        extError(caller, GL_INVALID_ENUM);
        extUnitEnd(&s->unit);
        return GL_FALSE;
    }
    if (texture) {
        GLMContext ctx = GET_CONTEXT();
        Texture *existing = ctx ? findTexture(ctx, texture) : NULL;
        if (existing && existing->target != s->object_target) {
            extError(caller, GL_INVALID_OPERATION);
            extUnitEnd(&s->unit);
            return GL_FALSE;
        }
    }
    glGetIntegerv(extBindingPname(s->object_target), &s->previous);
    glBindTexture(s->object_target, texture);
    s->bound = GL_TRUE;
    if (texture) {
        GLMContext ctx = GET_CONTEXT();
        Texture *resolved = ctx ? findTexture(ctx, texture) : NULL;
        if (!resolved || resolved->target != s->object_target) {
            extTextureEnd(s);
            return GL_FALSE;
        }
    }
    return GL_TRUE;
}

static void extTextureEnd(ExtTextureScope *s)
{
    if (s->bound)
        glBindTexture(s->object_target, (GLuint)s->previous);
    extUnitEnd(&s->unit);
}

static GLboolean extBuffer(GLuint buffer, const char *caller)
{
    GLMContext ctx = GET_CONTEXT();
    if (!buffer) {
        extError(caller, GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    if (!ctx || !getBuffer(ctx, 0, buffer)) {
        if (ctx)
            mglDispatchError(ctx, caller, GL_OUT_OF_MEMORY);
        return GL_FALSE;
    }
    return GL_TRUE;
}

typedef struct {
    GLint read;
    GLint draw;
    GLenum target;
} ExtFramebufferScope;

static GLboolean extFramebufferBegin(ExtFramebufferScope *s, GLuint framebuffer,
                                     GLenum target, const char *caller)
{
    if (target != GL_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER &&
        target != GL_DRAW_FRAMEBUFFER) {
        extError(caller, GL_INVALID_ENUM);
        return GL_FALSE;
    }
    s->target = target;
    s->read = 0;
    s->draw = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s->read);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s->draw);
    glBindFramebuffer(target, framebuffer);
    return GL_TRUE;
}

static void extFramebufferEnd(ExtFramebufferScope *s)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)s->read);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)s->draw);
}

typedef struct {
    GLint previous;
} ExtRenderbufferScope;

static void extRenderbufferBegin(ExtRenderbufferScope *s, GLuint renderbuffer)
{
    s->previous = 0;
    glGetIntegerv(GL_RENDERBUFFER_BINDING, &s->previous);
    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
}

static void extRenderbufferEnd(ExtRenderbufferScope *s)
{
    glBindRenderbuffer(GL_RENDERBUFFER, (GLuint)s->previous);
}

typedef struct {
    GLMContext ctx;
    GLuint previous_vao;
    GLint previous_buffer;
    GLboolean with_buffer;
} ExtVAOScope;

static GLboolean extVAOBegin(ExtVAOScope *s, GLuint vaobj,
                             GLboolean with_buffer, GLuint buffer,
                             const char *caller)
{
    s->ctx = GET_CONTEXT();
    s->previous_vao = 0;
    s->previous_buffer = 0;
    s->with_buffer = with_buffer;
    if (!mglVertexArrayNameMayBind(s->ctx, vaobj)) {
        extError(caller, GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, (GLint *)&s->previous_vao);
    glBindVertexArray(vaobj);
    if (!glIsVertexArray(vaobj)) {
        glBindVertexArray(s->previous_vao);
        return GL_FALSE;
    }
    if (s->with_buffer) {
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &s->previous_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
    }
    return GL_TRUE;
}

static void extVAOEnd(ExtVAOScope *s)
{
    if (s->with_buffer)
        glBindBuffer(GL_ARRAY_BUFFER, (GLuint)s->previous_buffer);
    glBindVertexArray(s->previous_vao);
}

typedef struct {
    GLint mode;
    GLint active;
    GLboolean texture;
} ExtMatrixScope;

static GLboolean extMatrixBegin(ExtMatrixScope *s, GLenum mode)
{
    s->mode = GL_MODELVIEW;
    s->active = GL_TEXTURE0;
    s->texture = mode >= GL_TEXTURE0 && mode <= GL_TEXTURE31;
    glGetIntegerv(GL_MATRIX_MODE, &s->mode);
    if (s->texture) {
        glGetIntegerv(GL_ACTIVE_TEXTURE, &s->active);
        glMatrixMode(GL_TEXTURE);
        glActiveTexture(mode);
    } else {
        glMatrixMode(mode);
    }
    return GL_TRUE;
}

static void extMatrixEnd(ExtMatrixScope *s)
{
    if (s->texture)
        glActiveTexture((GLenum)s->active);
    glMatrixMode((GLenum)s->mode);
}

void glMatrixLoadfEXT(GLenum mode, const GLfloat *m)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glLoadMatrixf(m);
    extMatrixEnd(&s);
}

void glMatrixLoaddEXT(GLenum mode, const GLdouble *m)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glLoadMatrixd(m);
    extMatrixEnd(&s);
}

void glMatrixMultfEXT(GLenum mode, const GLfloat *m)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glMultMatrixf(m);
    extMatrixEnd(&s);
}

void glMatrixMultdEXT(GLenum mode, const GLdouble *m)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glMultMatrixd(m);
    extMatrixEnd(&s);
}

void glMatrixLoadIdentityEXT(GLenum mode)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glLoadIdentity();
    extMatrixEnd(&s);
}

void glMatrixRotatefEXT(GLenum mode, GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glRotatef(angle, x, y, z);
    extMatrixEnd(&s);
}

void glMatrixRotatedEXT(GLenum mode, GLdouble angle, GLdouble x, GLdouble y, GLdouble z)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glRotated(angle, x, y, z);
    extMatrixEnd(&s);
}

void glMatrixScalefEXT(GLenum mode, GLfloat x, GLfloat y, GLfloat z)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glScalef(x, y, z);
    extMatrixEnd(&s);
}

void glMatrixScaledEXT(GLenum mode, GLdouble x, GLdouble y, GLdouble z)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glScaled(x, y, z);
    extMatrixEnd(&s);
}

void glMatrixTranslatefEXT(GLenum mode, GLfloat x, GLfloat y, GLfloat z)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glTranslatef(x, y, z);
    extMatrixEnd(&s);
}

void glMatrixTranslatedEXT(GLenum mode, GLdouble x, GLdouble y, GLdouble z)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glTranslated(x, y, z);
    extMatrixEnd(&s);
}

void glMatrixFrustumEXT(GLenum mode, GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glFrustum(left, right, bottom, top, zNear, zFar);
    extMatrixEnd(&s);
}

void glMatrixOrthoEXT(GLenum mode, GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glOrtho(left, right, bottom, top, zNear, zFar);
    extMatrixEnd(&s);
}

void glMatrixPopEXT(GLenum mode)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glPopMatrix();
    extMatrixEnd(&s);
}

void glMatrixPushEXT(GLenum mode)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glPushMatrix();
    extMatrixEnd(&s);
}

void glClientAttribDefaultEXT(GLbitfield mask)
{
    extUnsupported(__FUNCTION__);
}

void glPushClientAttribDefaultEXT(GLbitfield mask)
{
    glPushClientAttrib(mask);
    glClientAttribDefaultEXT(mask);
}

void glTextureParameterfEXT(GLuint texture, GLenum target, GLenum pname, GLfloat param)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexParameterf(target, pname, param);
    extTextureEnd(&s);
}

void glTextureParameterfvEXT(GLuint texture, GLenum target, GLenum pname, const GLfloat *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexParameterfv(target, pname, params);
    extTextureEnd(&s);
}

void glTextureParameteriEXT(GLuint texture, GLenum target, GLenum pname, GLint param)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexParameteri(target, pname, param);
    extTextureEnd(&s);
}

void glTextureParameterivEXT(GLuint texture, GLenum target, GLenum pname, const GLint *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexParameteriv(target, pname, params);
    extTextureEnd(&s);
}

void glTextureImage1DEXT(GLuint texture, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexImage1D(target, level, internalformat, width, border, format, type, pixels);
    extTextureEnd(&s);
}

void glTextureImage2DEXT(GLuint texture, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
    extTextureEnd(&s);
}

void glTextureSubImage1DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexSubImage1D(target, level, xoffset, width, format, type, pixels);
    extTextureEnd(&s);
}

void glTextureSubImage2DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
    extTextureEnd(&s);
}

void glCopyTextureImage1DEXT(GLuint texture, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCopyTexImage1D(target, level, internalformat, x, y, width, border);
    extTextureEnd(&s);
}

void glCopyTextureImage2DEXT(GLuint texture, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCopyTexImage2D(target, level, internalformat, x, y, width, height, border);
    extTextureEnd(&s);
}

void glCopyTextureSubImage1DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCopyTexSubImage1D(target, level, xoffset, x, y, width);
    extTextureEnd(&s);
}

void glCopyTextureSubImage2DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height);
    extTextureEnd(&s);
}

void glGetTextureImageEXT(GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, void *pixels)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGetTexImage(target, level, format, type, pixels);
    extTextureEnd(&s);
}

void glGetTextureParameterfvEXT(GLuint texture, GLenum target, GLenum pname, GLfloat *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGetTexParameterfv(target, pname, params);
    extTextureEnd(&s);
}

void glGetTextureParameterivEXT(GLuint texture, GLenum target, GLenum pname, GLint *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGetTexParameteriv(target, pname, params);
    extTextureEnd(&s);
}

void glGetTextureLevelParameterfvEXT(GLuint texture, GLenum target, GLint level, GLenum pname, GLfloat *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGetTexLevelParameterfv(target, level, pname, params);
    extTextureEnd(&s);
}

void glGetTextureLevelParameterivEXT(GLuint texture, GLenum target, GLint level, GLenum pname, GLint *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGetTexLevelParameteriv(target, level, pname, params);
    extTextureEnd(&s);
}

void glTextureImage3DEXT(GLuint texture, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexImage3D(target, level, internalformat, width, height, depth, border, format, type, pixels);
    extTextureEnd(&s);
}

void glTextureSubImage3DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
    extTextureEnd(&s);
}

void glCopyTextureSubImage3DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCopyTexSubImage3D(target, level, xoffset, yoffset, zoffset, x, y, width, height);
    extTextureEnd(&s);
}

void glBindMultiTextureEXT(GLenum texunit, GLenum target, GLuint texture)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glBindTexture(target, texture);
    extUnitEnd(&s);
}

void glMultiTexCoordPointerEXT(GLenum texunit, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    ExtUnitScope s;
    if (!(extClientUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexCoordPointer(size, type, stride, pointer);
    extClientUnitEnd(&s);
}

void glMultiTexEnvfEXT(GLenum texunit, GLenum target, GLenum pname, GLfloat param)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexEnvf(target, pname, param);
    extUnitEnd(&s);
}

void glMultiTexEnvfvEXT(GLenum texunit, GLenum target, GLenum pname, const GLfloat *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexEnvfv(target, pname, params);
    extUnitEnd(&s);
}

void glMultiTexEnviEXT(GLenum texunit, GLenum target, GLenum pname, GLint param)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexEnvi(target, pname, param);
    extUnitEnd(&s);
}

void glMultiTexEnvivEXT(GLenum texunit, GLenum target, GLenum pname, const GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexEnviv(target, pname, params);
    extUnitEnd(&s);
}

void glMultiTexGendEXT(GLenum texunit, GLenum coord, GLenum pname, GLdouble param)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexGend(coord, pname, param);
    extUnitEnd(&s);
}

void glMultiTexGendvEXT(GLenum texunit, GLenum coord, GLenum pname, const GLdouble *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexGendv(coord, pname, params);
    extUnitEnd(&s);
}

void glMultiTexGenfEXT(GLenum texunit, GLenum coord, GLenum pname, GLfloat param)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexGenf(coord, pname, param);
    extUnitEnd(&s);
}

void glMultiTexGenfvEXT(GLenum texunit, GLenum coord, GLenum pname, const GLfloat *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexGenfv(coord, pname, params);
    extUnitEnd(&s);
}

void glMultiTexGeniEXT(GLenum texunit, GLenum coord, GLenum pname, GLint param)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexGeni(coord, pname, param);
    extUnitEnd(&s);
}

void glMultiTexGenivEXT(GLenum texunit, GLenum coord, GLenum pname, const GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexGeniv(coord, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexEnvfvEXT(GLenum texunit, GLenum target, GLenum pname, GLfloat *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexEnvfv(target, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexEnvivEXT(GLenum texunit, GLenum target, GLenum pname, GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexEnviv(target, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexGendvEXT(GLenum texunit, GLenum coord, GLenum pname, GLdouble *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexGendv(coord, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexGenfvEXT(GLenum texunit, GLenum coord, GLenum pname, GLfloat *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexGenfv(coord, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexGenivEXT(GLenum texunit, GLenum coord, GLenum pname, GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexGeniv(coord, pname, params);
    extUnitEnd(&s);
}

void glMultiTexParameteriEXT(GLenum texunit, GLenum target, GLenum pname, GLint param)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexParameteri(target, pname, param);
    extUnitEnd(&s);
}

void glMultiTexParameterivEXT(GLenum texunit, GLenum target, GLenum pname, const GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexParameteriv(target, pname, params);
    extUnitEnd(&s);
}

void glMultiTexParameterfEXT(GLenum texunit, GLenum target, GLenum pname, GLfloat param)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexParameterf(target, pname, param);
    extUnitEnd(&s);
}

void glMultiTexParameterfvEXT(GLenum texunit, GLenum target, GLenum pname, const GLfloat *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexParameterfv(target, pname, params);
    extUnitEnd(&s);
}

void glMultiTexImage1DEXT(GLenum texunit, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexImage1D(target, level, internalformat, width, border, format, type, pixels);
    extUnitEnd(&s);
}

void glMultiTexImage2DEXT(GLenum texunit, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
    extUnitEnd(&s);
}

void glMultiTexSubImage1DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexSubImage1D(target, level, xoffset, width, format, type, pixels);
    extUnitEnd(&s);
}

void glMultiTexSubImage2DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
    extUnitEnd(&s);
}

void glCopyMultiTexImage1DEXT(GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCopyTexImage1D(target, level, internalformat, x, y, width, border);
    extUnitEnd(&s);
}

void glCopyMultiTexImage2DEXT(GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCopyTexImage2D(target, level, internalformat, x, y, width, height, border);
    extUnitEnd(&s);
}

void glCopyMultiTexSubImage1DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCopyTexSubImage1D(target, level, xoffset, x, y, width);
    extUnitEnd(&s);
}

void glCopyMultiTexSubImage2DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height);
    extUnitEnd(&s);
}

void glGetMultiTexImageEXT(GLenum texunit, GLenum target, GLint level, GLenum format, GLenum type, void *pixels)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexImage(target, level, format, type, pixels);
    extUnitEnd(&s);
}

void glGetMultiTexParameterfvEXT(GLenum texunit, GLenum target, GLenum pname, GLfloat *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexParameterfv(target, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexParameterivEXT(GLenum texunit, GLenum target, GLenum pname, GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexParameteriv(target, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexLevelParameterfvEXT(GLenum texunit, GLenum target, GLint level, GLenum pname, GLfloat *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexLevelParameterfv(target, level, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexLevelParameterivEXT(GLenum texunit, GLenum target, GLint level, GLenum pname, GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexLevelParameteriv(target, level, pname, params);
    extUnitEnd(&s);
}

void glMultiTexImage3DEXT(GLenum texunit, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexImage3D(target, level, internalformat, width, height, depth, border, format, type, pixels);
    extUnitEnd(&s);
}

void glMultiTexSubImage3DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
    extUnitEnd(&s);
}

void glCopyMultiTexSubImage3DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCopyTexSubImage3D(target, level, xoffset, yoffset, zoffset, x, y, width, height);
    extUnitEnd(&s);
}

void glEnableClientStateIndexedEXT(GLenum array, GLuint index)
{
    ExtUnitScope s;
    if (!(extClientUnitBegin(&s, GL_TEXTURE0 + index, __FUNCTION__)))
        return;
    glEnableClientState(array);
    extClientUnitEnd(&s);
}

void glDisableClientStateIndexedEXT(GLenum array, GLuint index)
{
    ExtUnitScope s;
    if (!(extClientUnitBegin(&s, GL_TEXTURE0 + index, __FUNCTION__)))
        return;
    glDisableClientState(array);
    extClientUnitEnd(&s);
}

void glGetPointerIndexedvEXT(GLenum target, GLuint index, void **data)
{
    ExtUnitScope s;
    if (!(extClientUnitBegin(&s, GL_TEXTURE0 + index, __FUNCTION__)))
        return;
    glGetPointerv(target, data);
    extClientUnitEnd(&s);
}

void glCompressedTextureImage3DEXT(GLuint texture, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *bits)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCompressedTexImage3D(target, level, internalformat, width, height, depth, border, imageSize, bits);
    extTextureEnd(&s);
}

void glCompressedTextureImage2DEXT(GLuint texture, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *bits)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, bits);
    extTextureEnd(&s);
}

void glCompressedTextureImage1DEXT(GLuint texture, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *bits)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCompressedTexImage1D(target, level, internalformat, width, border, imageSize, bits);
    extTextureEnd(&s);
}

void glCompressedTextureSubImage3DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *bits)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, bits);
    extTextureEnd(&s);
}

void glCompressedTextureSubImage2DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *bits)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, imageSize, bits);
    extTextureEnd(&s);
}

void glCompressedTextureSubImage1DEXT(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *bits)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glCompressedTexSubImage1D(target, level, xoffset, width, format, imageSize, bits);
    extTextureEnd(&s);
}

void glGetCompressedTextureImageEXT(GLuint texture, GLenum target, GLint lod, void *img)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGetCompressedTexImage(target, lod, img);
    extTextureEnd(&s);
}

void glCompressedMultiTexImage3DEXT(GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *bits)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCompressedTexImage3D(target, level, internalformat, width, height, depth, border, imageSize, bits);
    extUnitEnd(&s);
}

void glCompressedMultiTexImage2DEXT(GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *bits)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, bits);
    extUnitEnd(&s);
}

void glCompressedMultiTexImage1DEXT(GLenum texunit, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *bits)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCompressedTexImage1D(target, level, internalformat, width, border, imageSize, bits);
    extUnitEnd(&s);
}

void glCompressedMultiTexSubImage3DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *bits)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, bits);
    extUnitEnd(&s);
}

void glCompressedMultiTexSubImage2DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *bits)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, imageSize, bits);
    extUnitEnd(&s);
}

void glCompressedMultiTexSubImage1DEXT(GLenum texunit, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *bits)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glCompressedTexSubImage1D(target, level, xoffset, width, format, imageSize, bits);
    extUnitEnd(&s);
}

void glGetCompressedMultiTexImageEXT(GLenum texunit, GLenum target, GLint lod, void *img)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetCompressedTexImage(target, lod, img);
    extUnitEnd(&s);
}

void glMatrixLoadTransposefEXT(GLenum mode, const GLfloat *m)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glLoadTransposeMatrixf(m);
    extMatrixEnd(&s);
}

void glMatrixLoadTransposedEXT(GLenum mode, const GLdouble *m)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glLoadTransposeMatrixd(m);
    extMatrixEnd(&s);
}

void glMatrixMultTransposefEXT(GLenum mode, const GLfloat *m)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glMultTransposeMatrixf(m);
    extMatrixEnd(&s);
}

void glMatrixMultTransposedEXT(GLenum mode, const GLdouble *m)
{
    ExtMatrixScope s;
    if (!(extMatrixBegin(&s, mode)))
        return;
    glMultTransposeMatrixd(m);
    extMatrixEnd(&s);
}

void glNamedBufferDataEXT(GLuint buffer, GLsizeiptr size, const void *data, GLenum usage)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glNamedBufferData(buffer, size, data, usage);
}

void glNamedBufferSubDataEXT(GLuint buffer, GLintptr offset, GLsizeiptr size, const void *data)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glNamedBufferSubData(buffer, offset, size, data);
}

void * glMapNamedBufferEXT(GLuint buffer, GLenum access)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return NULL;
    return glMapNamedBuffer(buffer, access);
}

GLboolean glUnmapNamedBufferEXT(GLuint buffer)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return GL_FALSE;
    return glUnmapNamedBuffer(buffer);
}

void glGetNamedBufferParameterivEXT(GLuint buffer, GLenum pname, GLint *params)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glGetNamedBufferParameteriv(buffer, pname, params);
}

void glGetNamedBufferPointervEXT(GLuint buffer, GLenum pname, void **params)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glGetNamedBufferPointerv(buffer, pname, params);
}

void glGetNamedBufferSubDataEXT(GLuint buffer, GLintptr offset, GLsizeiptr size, void *data)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glGetNamedBufferSubData(buffer, offset, size, data);
}

void glProgramUniform1fEXT(GLuint program, GLint location, GLfloat v0)
{
    glProgramUniform1f(program, location, v0);
}

void glProgramUniform2fEXT(GLuint program, GLint location, GLfloat v0, GLfloat v1)
{
    glProgramUniform2f(program, location, v0, v1);
}

void glProgramUniform3fEXT(GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2)
{
    glProgramUniform3f(program, location, v0, v1, v2);
}

void glProgramUniform4fEXT(GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
    glProgramUniform4f(program, location, v0, v1, v2, v3);
}

void glProgramUniform1iEXT(GLuint program, GLint location, GLint v0)
{
    glProgramUniform1i(program, location, v0);
}

void glProgramUniform2iEXT(GLuint program, GLint location, GLint v0, GLint v1)
{
    glProgramUniform2i(program, location, v0, v1);
}

void glProgramUniform3iEXT(GLuint program, GLint location, GLint v0, GLint v1, GLint v2)
{
    glProgramUniform3i(program, location, v0, v1, v2);
}

void glProgramUniform4iEXT(GLuint program, GLint location, GLint v0, GLint v1, GLint v2, GLint v3)
{
    glProgramUniform4i(program, location, v0, v1, v2, v3);
}

void glProgramUniform1fvEXT(GLuint program, GLint location, GLsizei count, const GLfloat *value)
{
    glProgramUniform1fv(program, location, count, value);
}

void glProgramUniform2fvEXT(GLuint program, GLint location, GLsizei count, const GLfloat *value)
{
    glProgramUniform2fv(program, location, count, value);
}

void glProgramUniform3fvEXT(GLuint program, GLint location, GLsizei count, const GLfloat *value)
{
    glProgramUniform3fv(program, location, count, value);
}

void glProgramUniform4fvEXT(GLuint program, GLint location, GLsizei count, const GLfloat *value)
{
    glProgramUniform4fv(program, location, count, value);
}

void glProgramUniform1ivEXT(GLuint program, GLint location, GLsizei count, const GLint *value)
{
    glProgramUniform1iv(program, location, count, value);
}

void glProgramUniform2ivEXT(GLuint program, GLint location, GLsizei count, const GLint *value)
{
    glProgramUniform2iv(program, location, count, value);
}

void glProgramUniform3ivEXT(GLuint program, GLint location, GLsizei count, const GLint *value)
{
    glProgramUniform3iv(program, location, count, value);
}

void glProgramUniform4ivEXT(GLuint program, GLint location, GLsizei count, const GLint *value)
{
    glProgramUniform4iv(program, location, count, value);
}

void glProgramUniformMatrix2fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix2fv(program, location, count, transpose, value);
}

void glProgramUniformMatrix3fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix3fv(program, location, count, transpose, value);
}

void glProgramUniformMatrix4fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix4fv(program, location, count, transpose, value);
}

void glProgramUniformMatrix2x3fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix2x3fv(program, location, count, transpose, value);
}

void glProgramUniformMatrix3x2fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix3x2fv(program, location, count, transpose, value);
}

void glProgramUniformMatrix2x4fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix2x4fv(program, location, count, transpose, value);
}

void glProgramUniformMatrix4x2fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix4x2fv(program, location, count, transpose, value);
}

void glProgramUniformMatrix3x4fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix3x4fv(program, location, count, transpose, value);
}

void glProgramUniformMatrix4x3fvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    glProgramUniformMatrix4x3fv(program, location, count, transpose, value);
}

void glTextureBufferEXT(GLuint texture, GLenum target, GLenum internalformat, GLuint buffer)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexBuffer(target, internalformat, buffer);
    extTextureEnd(&s);
}

void glMultiTexBufferEXT(GLenum texunit, GLenum target, GLenum internalformat, GLuint buffer)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexBuffer(target, internalformat, buffer);
    extUnitEnd(&s);
}

void glTextureParameterIivEXT(GLuint texture, GLenum target, GLenum pname, const GLint *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexParameterIiv(target, pname, params);
    extTextureEnd(&s);
}

void glTextureParameterIuivEXT(GLuint texture, GLenum target, GLenum pname, const GLuint *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexParameterIuiv(target, pname, params);
    extTextureEnd(&s);
}

void glGetTextureParameterIivEXT(GLuint texture, GLenum target, GLenum pname, GLint *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGetTexParameterIiv(target, pname, params);
    extTextureEnd(&s);
}

void glGetTextureParameterIuivEXT(GLuint texture, GLenum target, GLenum pname, GLuint *params)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGetTexParameterIuiv(target, pname, params);
    extTextureEnd(&s);
}

void glMultiTexParameterIivEXT(GLenum texunit, GLenum target, GLenum pname, const GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexParameterIiv(target, pname, params);
    extUnitEnd(&s);
}

void glMultiTexParameterIuivEXT(GLenum texunit, GLenum target, GLenum pname, const GLuint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glTexParameterIuiv(target, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexParameterIivEXT(GLenum texunit, GLenum target, GLenum pname, GLint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexParameterIiv(target, pname, params);
    extUnitEnd(&s);
}

void glGetMultiTexParameterIuivEXT(GLenum texunit, GLenum target, GLenum pname, GLuint *params)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGetTexParameterIuiv(target, pname, params);
    extUnitEnd(&s);
}

void glProgramUniform1uiEXT(GLuint program, GLint location, GLuint v0)
{
    glProgramUniform1ui(program, location, v0);
}

void glProgramUniform2uiEXT(GLuint program, GLint location, GLuint v0, GLuint v1)
{
    glProgramUniform2ui(program, location, v0, v1);
}

void glProgramUniform3uiEXT(GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2)
{
    glProgramUniform3ui(program, location, v0, v1, v2);
}

void glProgramUniform4uiEXT(GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3)
{
    glProgramUniform4ui(program, location, v0, v1, v2, v3);
}

void glProgramUniform1uivEXT(GLuint program, GLint location, GLsizei count, const GLuint *value)
{
    glProgramUniform1uiv(program, location, count, value);
}

void glProgramUniform2uivEXT(GLuint program, GLint location, GLsizei count, const GLuint *value)
{
    glProgramUniform2uiv(program, location, count, value);
}

void glProgramUniform3uivEXT(GLuint program, GLint location, GLsizei count, const GLuint *value)
{
    glProgramUniform3uiv(program, location, count, value);
}

void glProgramUniform4uivEXT(GLuint program, GLint location, GLsizei count, const GLuint *value)
{
    glProgramUniform4uiv(program, location, count, value);
}

void glNamedProgramLocalParameters4fvEXT(GLuint program, GLenum target, GLuint index, GLsizei count, const GLfloat *params)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParameterI4iEXT(GLuint program, GLenum target, GLuint index, GLint x, GLint y, GLint z, GLint w)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParameterI4ivEXT(GLuint program, GLenum target, GLuint index, const GLint *params)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParametersI4ivEXT(GLuint program, GLenum target, GLuint index, GLsizei count, const GLint *params)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParameterI4uiEXT(GLuint program, GLenum target, GLuint index, GLuint x, GLuint y, GLuint z, GLuint w)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParameterI4uivEXT(GLuint program, GLenum target, GLuint index, const GLuint *params)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParametersI4uivEXT(GLuint program, GLenum target, GLuint index, GLsizei count, const GLuint *params)
{
    extUnsupported(__FUNCTION__);
}

void glGetNamedProgramLocalParameterIivEXT(GLuint program, GLenum target, GLuint index, GLint *params)
{
    extUnsupported(__FUNCTION__);
}

void glGetNamedProgramLocalParameterIuivEXT(GLuint program, GLenum target, GLuint index, GLuint *params)
{
    extUnsupported(__FUNCTION__);
}

void glEnableClientStateiEXT(GLenum array, GLuint index)
{
    ExtUnitScope s;
    if (!(extClientUnitBegin(&s, GL_TEXTURE0 + index, __FUNCTION__)))
        return;
    glEnableClientState(array);
    extClientUnitEnd(&s);
}

void glDisableClientStateiEXT(GLenum array, GLuint index)
{
    ExtUnitScope s;
    if (!(extClientUnitBegin(&s, GL_TEXTURE0 + index, __FUNCTION__)))
        return;
    glDisableClientState(array);
    extClientUnitEnd(&s);
}

void glGetPointeri_vEXT(GLenum pname, GLuint index, void **params)
{
    ExtUnitScope s;
    if (!(extClientUnitBegin(&s, GL_TEXTURE0 + index, __FUNCTION__)))
        return;
    glGetPointerv(pname, params);
    extClientUnitEnd(&s);
}

void glNamedProgramStringEXT(GLuint program, GLenum target, GLenum format, GLsizei len, const void *string)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParameter4dEXT(GLuint program, GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParameter4dvEXT(GLuint program, GLenum target, GLuint index, const GLdouble *params)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParameter4fEXT(GLuint program, GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    extUnsupported(__FUNCTION__);
}

void glNamedProgramLocalParameter4fvEXT(GLuint program, GLenum target, GLuint index, const GLfloat *params)
{
    extUnsupported(__FUNCTION__);
}

void glGetNamedProgramLocalParameterdvEXT(GLuint program, GLenum target, GLuint index, GLdouble *params)
{
    extUnsupported(__FUNCTION__);
}

void glGetNamedProgramLocalParameterfvEXT(GLuint program, GLenum target, GLuint index, GLfloat *params)
{
    extUnsupported(__FUNCTION__);
}

void glGetNamedProgramivEXT(GLuint program, GLenum target, GLenum pname, GLint *params)
{
    extUnsupported(__FUNCTION__);
}

void glGetNamedProgramStringEXT(GLuint program, GLenum target, GLenum pname, void *string)
{
    extUnsupported(__FUNCTION__);
}

void glNamedRenderbufferStorageEXT(GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height)
{
    ExtRenderbufferScope s;
    if (!((extRenderbufferBegin(&s, renderbuffer), GL_TRUE)))
        return;
    glRenderbufferStorage(GL_RENDERBUFFER, internalformat, width, height);
    extRenderbufferEnd(&s);
}

void glGetNamedRenderbufferParameterivEXT(GLuint renderbuffer, GLenum pname, GLint *params)
{
    ExtRenderbufferScope s;
    if (!((extRenderbufferBegin(&s, renderbuffer), GL_TRUE)))
        return;
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, pname, params);
    extRenderbufferEnd(&s);
}

void glNamedRenderbufferStorageMultisampleEXT(GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height)
{
    ExtRenderbufferScope s;
    if (!((extRenderbufferBegin(&s, renderbuffer), GL_TRUE)))
        return;
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, internalformat, width, height);
    extRenderbufferEnd(&s);
}

void glNamedRenderbufferStorageMultisampleCoverageEXT(GLuint renderbuffer, GLsizei coverageSamples, GLsizei colorSamples, GLenum internalformat, GLsizei width, GLsizei height)
{
    extUnsupported(__FUNCTION__);
}

GLenum glCheckNamedFramebufferStatusEXT(GLuint framebuffer, GLenum target)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, target, __FUNCTION__)))
        return 0;
    GLenum result = glCheckFramebufferStatus(target);
    extFramebufferEnd(&s);
    return result;
}

void glNamedFramebufferTexture1DEXT(GLuint framebuffer, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glFramebufferTexture1D(GL_FRAMEBUFFER, attachment, textarget, texture, level);
    extFramebufferEnd(&s);
}

void glNamedFramebufferTexture2DEXT(GLuint framebuffer, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, textarget, texture, level);
    extFramebufferEnd(&s);
}

void glNamedFramebufferTexture3DEXT(GLuint framebuffer, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glFramebufferTexture3D(GL_FRAMEBUFFER, attachment, textarget, texture, level, zoffset);
    extFramebufferEnd(&s);
}

void glNamedFramebufferRenderbufferEXT(GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, renderbuffertarget, renderbuffer);
    extFramebufferEnd(&s);
}

void glGetNamedFramebufferAttachmentParameterivEXT(GLuint framebuffer, GLenum attachment, GLenum pname, GLint *params)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, pname, params);
    extFramebufferEnd(&s);
}

void glGenerateTextureMipmapEXT(GLuint texture, GLenum target)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glGenerateMipmap(target);
    extTextureEnd(&s);
}

void glGenerateMultiTexMipmapEXT(GLenum texunit, GLenum target)
{
    ExtUnitScope s;
    if (!(extUnitBegin(&s, texunit, __FUNCTION__)))
        return;
    glGenerateMipmap(target);
    extUnitEnd(&s);
}

void glFramebufferDrawBufferEXT(GLuint framebuffer, GLenum mode)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glDrawBuffer(mode);
    extFramebufferEnd(&s);
}

void glFramebufferDrawBuffersEXT(GLuint framebuffer, GLsizei n, const GLenum *bufs)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glDrawBuffers(n, bufs);
    extFramebufferEnd(&s);
}

void glFramebufferReadBufferEXT(GLuint framebuffer, GLenum mode)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glReadBuffer(mode);
    extFramebufferEnd(&s);
}

void glGetFramebufferParameterivEXT(GLuint framebuffer, GLenum pname, GLint *params)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glGetIntegerv(pname, params);
    extFramebufferEnd(&s);
}

void glNamedCopyBufferSubDataEXT(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    if (!extBuffer(readBuffer, __FUNCTION__) ||
        !extBuffer(writeBuffer, __FUNCTION__))
        return;
    glCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset, size);
}

void glNamedFramebufferTextureEXT(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glFramebufferTexture(GL_FRAMEBUFFER, attachment, texture, level);
    extFramebufferEnd(&s);
}

void glNamedFramebufferTextureLayerEXT(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLint layer)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texture, level, layer);
    extFramebufferEnd(&s);
}

void glNamedFramebufferTextureFaceEXT(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLenum face)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, face, texture, level);
    extFramebufferEnd(&s);
}

void glTextureRenderbufferEXT(GLuint texture, GLenum target, GLuint renderbuffer)
{
    extUnsupported(__FUNCTION__);
}

void glMultiTexRenderbufferEXT(GLenum texunit, GLenum target, GLuint renderbuffer)
{
    extUnsupported(__FUNCTION__);
}

void glVertexArrayVertexOffsetEXT(GLuint vaobj, GLuint buffer, GLint size, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glVertexPointer(size, type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayColorOffsetEXT(GLuint vaobj, GLuint buffer, GLint size, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glColorPointer(size, type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayEdgeFlagOffsetEXT(GLuint vaobj, GLuint buffer, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glEdgeFlagPointer(stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayIndexOffsetEXT(GLuint vaobj, GLuint buffer, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glIndexPointer(type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayNormalOffsetEXT(GLuint vaobj, GLuint buffer, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glNormalPointer(type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayTexCoordOffsetEXT(GLuint vaobj, GLuint buffer, GLint size, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glTexCoordPointer(size, type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayMultiTexCoordOffsetEXT(GLuint vaobj, GLuint buffer, GLenum texunit, GLint size, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__))
        return;
    ExtUnitScope unit;
    if (extClientUnitBegin(&unit, texunit, __FUNCTION__)) {
        glTexCoordPointer(size, type, stride, (const void *)(uintptr_t)offset);
        extClientUnitEnd(&unit);
    }
    extVAOEnd(&s);
}

void glVertexArrayFogCoordOffsetEXT(GLuint vaobj, GLuint buffer, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glFogCoordPointer(type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArraySecondaryColorOffsetEXT(GLuint vaobj, GLuint buffer, GLint size, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glSecondaryColorPointer(size, type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayVertexAttribOffsetEXT(GLuint vaobj, GLuint buffer, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glVertexAttribPointer(index, size, type, normalized, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayVertexAttribIOffsetEXT(GLuint vaobj, GLuint buffer, GLuint index, GLint size, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glVertexAttribIPointer(index, size, type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glEnableVertexArrayEXT(GLuint vaobj, GLenum array)
{
    ExtVAOScope s;
    if (!extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__))
        return;
    if (array >= GL_TEXTURE0 && array <= GL_TEXTURE31) {
        ExtUnitScope unit;
        if (extClientUnitBegin(&unit, array, __FUNCTION__)) {
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);
            extClientUnitEnd(&unit);
        }
    } else {
        glEnableClientState(array);
    }
    extVAOEnd(&s);
}

void glDisableVertexArrayEXT(GLuint vaobj, GLenum array)
{
    ExtVAOScope s;
    if (!extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__))
        return;
    if (array >= GL_TEXTURE0 && array <= GL_TEXTURE31) {
        ExtUnitScope unit;
        if (extClientUnitBegin(&unit, array, __FUNCTION__)) {
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            extClientUnitEnd(&unit);
        }
    } else {
        glDisableClientState(array);
    }
    extVAOEnd(&s);
}

void glEnableVertexArrayAttribEXT(GLuint vaobj, GLuint index)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glEnableVertexAttribArray(index);
    extVAOEnd(&s);
}

void glDisableVertexArrayAttribEXT(GLuint vaobj, GLuint index)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glDisableVertexAttribArray(index);
    extVAOEnd(&s);
}

void glGetVertexArrayIntegervEXT(GLuint vaobj, GLenum pname, GLint *param)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glGetIntegerv(pname, param);
    extVAOEnd(&s);
}

void glGetVertexArrayPointervEXT(GLuint vaobj, GLenum pname, void **param)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glGetPointerv(pname, param);
    extVAOEnd(&s);
}

void glGetVertexArrayIntegeri_vEXT(GLuint vaobj, GLuint index, GLenum pname, GLint *param)
{
    ExtVAOScope s;
    if (!extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__))
        return;
    if (pname == GL_TEXTURE_COORD_ARRAY ||
        pname == GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING ||
        pname == GL_TEXTURE_COORD_ARRAY_SIZE ||
        pname == GL_TEXTURE_COORD_ARRAY_STRIDE ||
        pname == GL_TEXTURE_COORD_ARRAY_TYPE) {
        ExtUnitScope unit;
        if (extClientUnitBegin(&unit, GL_TEXTURE0 + index, __FUNCTION__)) {
            glGetIntegerv(pname, param);
            extClientUnitEnd(&unit);
        }
    } else {
        glGetVertexAttribiv(index, pname, param);
    }
    extVAOEnd(&s);
}

void glGetVertexArrayPointeri_vEXT(GLuint vaobj, GLuint index, GLenum pname, void **param)
{
    ExtVAOScope s;
    if (!extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__))
        return;
    if (pname == GL_TEXTURE_COORD_ARRAY_POINTER) {
        ExtUnitScope unit;
        if (extClientUnitBegin(&unit, GL_TEXTURE0 + index, __FUNCTION__)) {
            glGetPointerv(pname, param);
            extClientUnitEnd(&unit);
        }
    } else {
        glGetVertexAttribPointerv(index, pname, param);
    }
    extVAOEnd(&s);
}

void * glMapNamedBufferRangeEXT(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return NULL;
    return glMapNamedBufferRange(buffer, offset, length, access);
}

void glFlushMappedNamedBufferRangeEXT(GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glFlushMappedNamedBufferRange(buffer, offset, length);
}

void glNamedBufferStorageEXT(GLuint buffer, GLsizeiptr size, const void *data, GLbitfield flags)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glNamedBufferStorage(buffer, size, data, flags);
}

void glClearNamedBufferDataEXT(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void *data)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glClearNamedBufferData(buffer, internalformat, format, type, data);
}

void glClearNamedBufferSubDataEXT(GLuint buffer, GLenum internalformat, GLsizeiptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    if (!extBuffer(buffer, __FUNCTION__))
        return;
    glClearNamedBufferSubData(buffer, internalformat, offset, size, format, type, data);
}

void glNamedFramebufferParameteriEXT(GLuint framebuffer, GLenum pname, GLint param)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glFramebufferParameteri(GL_FRAMEBUFFER, pname, param);
    extFramebufferEnd(&s);
}

void glGetNamedFramebufferParameterivEXT(GLuint framebuffer, GLenum pname, GLint *params)
{
    ExtFramebufferScope s;
    if (!(extFramebufferBegin(&s, framebuffer, GL_FRAMEBUFFER, __FUNCTION__)))
        return;
    glGetFramebufferParameteriv(GL_FRAMEBUFFER, pname, params);
    extFramebufferEnd(&s);
}

void glProgramUniform1dEXT(GLuint program, GLint location, GLdouble x)
{
    glProgramUniform1d(program, location, x);
}

void glProgramUniform2dEXT(GLuint program, GLint location, GLdouble x, GLdouble y)
{
    glProgramUniform2d(program, location, x, y);
}

void glProgramUniform3dEXT(GLuint program, GLint location, GLdouble x, GLdouble y, GLdouble z)
{
    glProgramUniform3d(program, location, x, y, z);
}

void glProgramUniform4dEXT(GLuint program, GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    glProgramUniform4d(program, location, x, y, z, w);
}

void glProgramUniform1dvEXT(GLuint program, GLint location, GLsizei count, const GLdouble *value)
{
    glProgramUniform1dv(program, location, count, value);
}

void glProgramUniform2dvEXT(GLuint program, GLint location, GLsizei count, const GLdouble *value)
{
    glProgramUniform2dv(program, location, count, value);
}

void glProgramUniform3dvEXT(GLuint program, GLint location, GLsizei count, const GLdouble *value)
{
    glProgramUniform3dv(program, location, count, value);
}

void glProgramUniform4dvEXT(GLuint program, GLint location, GLsizei count, const GLdouble *value)
{
    glProgramUniform4dv(program, location, count, value);
}

void glProgramUniformMatrix2dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix2dv(program, location, count, transpose, value);
}

void glProgramUniformMatrix3dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix3dv(program, location, count, transpose, value);
}

void glProgramUniformMatrix4dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix4dv(program, location, count, transpose, value);
}

void glProgramUniformMatrix2x3dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix2x3dv(program, location, count, transpose, value);
}

void glProgramUniformMatrix2x4dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix2x4dv(program, location, count, transpose, value);
}

void glProgramUniformMatrix3x2dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix3x2dv(program, location, count, transpose, value);
}

void glProgramUniformMatrix3x4dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix3x4dv(program, location, count, transpose, value);
}

void glProgramUniformMatrix4x2dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix4x2dv(program, location, count, transpose, value);
}

void glProgramUniformMatrix4x3dvEXT(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    glProgramUniformMatrix4x3dv(program, location, count, transpose, value);
}

void glTextureBufferRangeEXT(GLuint texture, GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTexBufferRange(target, internalformat, buffer, offset, size);
    extTextureEnd(&s);
}

void glTextureStorage1DEXT(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTextureStorage1D(texture, levels, internalformat, width);
    extTextureEnd(&s);
}

void glTextureStorage2DEXT(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTextureStorage2D(texture, levels, internalformat, width, height);
    extTextureEnd(&s);
}

void glTextureStorage3DEXT(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTextureStorage3D(texture, levels, internalformat, width, height, depth);
    extTextureEnd(&s);
}

void glTextureStorage2DMultisampleEXT(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTextureStorage2DMultisample(texture, samples, internalformat, width, height, fixedsamplelocations);
    extTextureEnd(&s);
}

void glTextureStorage3DMultisampleEXT(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    ExtTextureScope s;
    if (!(extTextureBegin(&s, texture, target, 0, __FUNCTION__)))
        return;
    glTextureStorage3DMultisample(texture, samples, internalformat, width, height, depth, fixedsamplelocations);
    extTextureEnd(&s);
}

void glVertexArrayBindVertexBufferEXT(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glBindVertexBuffer(bindingindex, buffer, offset, stride);
    extVAOEnd(&s);
}

void glVertexArrayVertexAttribFormatEXT(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glVertexAttribFormat(attribindex, size, type, normalized, relativeoffset);
    extVAOEnd(&s);
}

void glVertexArrayVertexAttribIFormatEXT(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glVertexAttribIFormat(attribindex, size, type, relativeoffset);
    extVAOEnd(&s);
}

void glVertexArrayVertexAttribLFormatEXT(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glVertexAttribLFormat(attribindex, size, type, relativeoffset);
    extVAOEnd(&s);
}

void glVertexArrayVertexAttribBindingEXT(GLuint vaobj, GLuint attribindex, GLuint bindingindex)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glVertexAttribBinding(attribindex, bindingindex);
    extVAOEnd(&s);
}

void glVertexArrayVertexBindingDivisorEXT(GLuint vaobj, GLuint bindingindex, GLuint divisor)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glVertexBindingDivisor(bindingindex, divisor);
    extVAOEnd(&s);
}

void glVertexArrayVertexAttribLOffsetEXT(GLuint vaobj, GLuint buffer, GLuint index, GLint size, GLenum type, GLsizei stride, GLintptr offset)
{
    if (buffer && offset < 0) {
        extError(__FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_TRUE, buffer, __FUNCTION__)))
        return;
    glVertexAttribLPointer(index, size, type, stride, (const void *)(uintptr_t)offset);
    extVAOEnd(&s);
}

void glVertexArrayVertexAttribDivisorEXT(GLuint vaobj, GLuint index, GLuint divisor)
{
    ExtVAOScope s;
    if (!(extVAOBegin(&s, vaobj, GL_FALSE, 0, __FUNCTION__)))
        return;
    glVertexAttribDivisor(index, divisor);
    extVAOEnd(&s);
}

