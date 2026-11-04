/*
 * SPDX-License-Identifier: LGPL-3.0-only
 * Internal FE texture helpers shared across textures split TUs (Phase 7).
 */
#ifndef MGL_TEXTURES_INTERNAL_H
#define MGL_TEXTURES_INTERNAL_H
#include "glm_context.h"
#include <stdbool.h>
#include <stdint.h>

#ifndef MGL_VERBOSE_TEXTURE_BIND_LOGS
#define MGL_VERBOSE_TEXTURE_BIND_LOGS 0
bool mglTextureStorageMultisampleMetadata(GLMContext ctx, Texture *tex,
                                          GLenum target, GLsizei samples,
                                          GLenum internalformat, GLsizei width,
                                          GLsizei height, GLsizei depth,
                                          GLboolean fixedsamplelocations,
                                          GLboolean proxy);

GLint mglCompressedInternalFormatToSizedUncompressed(GLint internalformat);
GLint mglTexLevelCanonicalInternalFormat(GLint internalformat);


int mglTryTexBufferLevelParameter(const Texture *tex, GLint level, GLenum pname,
                                  GLint *params, GLenum *err);
GLenum mglTextureQueryInternalFormat(const Texture *tex);
GLint mglTexLevelComponentBits(GLint internalformat, GLenum pname);
GLint mglTextureQueryCompressedImageSize(const Texture *tex, GLint level);
bool getParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname,
              GLint *iparam, GLfloat *fparam);
void *getBufferData(GLMContext ctx, Buffer *ptr);
Buffer *findBuffer(GLMContext ctx, GLuint buffer);
GLsizei mglSafeMaxTextureSize(GLMContext ctx);
GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);

GLint mglTexLevelComponentType(GLint internalformat, GLenum pname);

#endif

#ifndef MGL_VERBOSE_TEXTURE_UPLOAD_LOGS
#define MGL_VERBOSE_TEXTURE_UPLOAD_LOGS 0
bool mglTextureStorageMultisampleMetadata(GLMContext ctx, Texture *tex,
                                          GLenum target, GLsizei samples,
                                          GLenum internalformat, GLsizei width,
                                          GLsizei height, GLsizei depth,
                                          GLboolean fixedsamplelocations,
                                          GLboolean proxy);

GLint mglCompressedInternalFormatToSizedUncompressed(GLint internalformat);
GLint mglTexLevelCanonicalInternalFormat(GLint internalformat);


int mglTryTexBufferLevelParameter(const Texture *tex, GLint level, GLenum pname,
                                  GLint *params, GLenum *err);
GLenum mglTextureQueryInternalFormat(const Texture *tex);
GLint mglTexLevelComponentBits(GLint internalformat, GLenum pname);
GLint mglTextureQueryCompressedImageSize(const Texture *tex, GLint level);
bool getParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname,
              GLint *iparam, GLfloat *fparam);
void *getBufferData(GLMContext ctx, Buffer *ptr);
Buffer *findBuffer(GLMContext ctx, GLuint buffer);
GLsizei mglSafeMaxTextureSize(GLMContext ctx);
GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);

GLint mglTexLevelComponentType(GLint internalformat, GLenum pname);

#endif

void mglResetImageUnit(ImageUnit *iu);
GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);
bool mglIsTextureObjectTarget(GLenum target);
GLenum mglCanonicalTextureObjectTarget(GLenum target);
Texture *currentTexture(GLMContext ctx, GLuint index);
Texture *newTexObj(GLMContext ctx, GLenum target);
Texture *newTexture(GLMContext ctx, GLenum target, GLuint texture);
Texture *getTexture(GLMContext ctx, GLenum target, GLuint texture);
int isTexture(GLMContext ctx, GLuint texture);
Texture *findTexture(GLMContext ctx, GLuint texture);
GLboolean mglTextureUnitHasAnyBinding(GLMContext ctx, GLuint unit);
Texture *mglChooseTextureUnitActiveBinding(GLMContext ctx, GLuint unit);
void mglUpdateTextureUnitActiveMask(GLMContext ctx, GLuint unit);
Texture *getTex(GLMContext ctx, GLuint name, GLenum target);
bool checkInternalFormatForMetal(GLMContext ctx, GLuint internalformat);
void mglGenTextures(GLMContext ctx, GLsizei n, GLuint *textures);
void mglCreateTextures(GLMContext ctx, GLenum target, GLsizei n, GLuint *textures);
void mglBindTexture(GLMContext ctx, GLenum target, GLuint texture);
GLboolean mglTextureTargetUsesImageLayerParameter(GLenum target);
GLboolean mglIsLegalImageUnitFormat(GLenum format);
void mglBindImageTexture(GLMContext ctx, GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum internalformat);
void mglDetachTextureFromFramebuffers(GLuint name, void *data, void *user);
void mglDeleteTextures(GLMContext ctx, GLsizei n, const GLuint *textures);
GLboolean mglIsTexture(GLMContext ctx, GLuint texture);
void mglInvalidateTexImage(GLMContext ctx, GLuint texture, GLint level);
void mglInvalidateTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth);
void mglBindImageTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures);
void mglClientActiveTexture(GLMContext ctx, GLenum texture);
void mglActiveTexture(GLMContext ctx, GLenum texture);
void mglBindTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures);
void mglBindTextureUnit(GLMContext ctx, GLuint unit, GLuint texture);
GLuint mglTextureTargetMaxLevels(GLenum target,
                                        GLuint width,
                                        GLuint height,
                                        GLuint depth);
bool mglTextureRecordedLevelsMatchBase(Texture *tex,
                                              GLsizei width,
                                              GLsizei height,
                                              GLsizei depth);
void generateMipmaps(GLMContext ctx, GLuint texture, GLenum target);
void mglGenerateMipmap(GLMContext ctx, GLenum target);
void mglGenerateTextureMipmap(GLMContext ctx, GLuint texture);
void mglInvalidateTextureBaseLevelView(GLMContext ctx, Texture *tex);
void invalidateTexture(GLMContext ctx, Texture *tex);
void initBaseTexLevel(GLMContext ctx, Texture *tex, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth);
bool ensureTextureLevelCapacity(GLMContext ctx, Texture *tex, GLuint required_levels);
GLuint mglMultisampleCountLimit(GLMContext ctx);
GLenum mglMultisampleCountValidationError(GLMContext ctx, GLsizei samples);
GLuint mglPromoteMultisampleCount(GLMContext ctx, GLsizei samples);
bool checkTexLevelParams(GLMContext ctx, Texture *tex, GLint level, GLuint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type);
bool verifyInternalFormatAndFormatType(GLMContext ctx, GLint internalformat, GLenum format, GLenum type);
void unpackTexture(GLMContext ctx, Texture *tex, GLuint face, GLuint level, void *src_data, void *dst_data, size_t src_pitch, size_t src_image_pitch, size_t pixel_size, size_t xoffset, size_t yoffset, size_t zoffset, size_t width, size_t height, size_t depth);
bool createTextureLevel(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLboolean is_array, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels, GLboolean proxy);
void mglTexImage1D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels);
void mglTexImage2D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
void mglTexImage2DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
bool mglInternalFormatIsRGTC(GLenum internalformat);
void mglTexImage3D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
void mglTexImage3DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
bool texSubImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels);
void texSubImage1D(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels);
void mglTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels);
void mglTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels);
bool texSubImage2D(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
void mglTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
void mglTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
bool texSubImage3D(GLMContext ctx, Texture *tex, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);
void mglTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);
void mglTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);
int mglSparseStandardPageSize(GLenum internalformat, GLint *x, GLint *y, GLint *z);
void mglSparseCommitReset(Texture *tex);
int mglSparseCommitAppend(Texture *tex, GLint level,
                                 GLint x, GLint y, GLint z,
                                 GLint w, GLint h, GLint d);
int mglSparseTexelCommitted(const Texture *tex, GLint level,
                                   GLint x, GLint y, GLint z);
void mglSparseZeroUncommitted(Texture *tex, GLuint face, GLint level,
                                     GLint xoffset, GLint yoffset, GLint zoffset,
                                     GLsizei width, GLsizei height, GLsizei depth);
int mglSparseCommitSubtract(Texture *tex, GLint level,
                                   GLint x, GLint y, GLint z,
                                   GLint w, GLint h, GLint d);
bool mglSparseTexStorageAllowed(GLMContext ctx, Texture *tex,
                                      GLenum internalformat,
                                      GLsizei levels, GLsizei width,
                                      GLsizei height, GLsizei depth);
void texStorage(GLMContext ctx, Texture *tex, GLuint faces, GLsizei levels, GLboolean is_array, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean proxy);
void mglTexStorage1D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
void mglTextureStorage1D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width);
void mglTexStorage2D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
void mglCommitSparsePages(GLMContext ctx, Texture *tex, GLint level,
                                 GLint xoffset, GLint yoffset, GLint zoffset,
                                 GLsizei width, GLsizei height, GLsizei depth,
                                 GLboolean commit);
int mglSparsePageIntersectsCommit(const Texture *tex, GLint level,
                                         GLint x, GLint y, GLint z,
                                         GLint w, GLint h, GLint d);
void mglSparseScrubTextureLevel(Texture *tex, GLint level, int clear);
void mglSparseScrubBoundImages(GLMContext ctx);
void mglTexPageCommitmentARB(GLMContext ctx, GLenum target, GLint level,
                             GLint xoffset, GLint yoffset, GLint zoffset,
                             GLsizei width, GLsizei height, GLsizei depth,
                             GLboolean commit);
void mglTexturePageCommitmentEXT(GLMContext ctx, GLuint texture, GLint level,
                                 GLint xoffset, GLint yoffset, GLint zoffset,
                                 GLsizei width, GLsizei height, GLsizei depth,
                                 GLboolean commit);
void mglTextureStorage2D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
void mglTextureStorage2DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
void mglTexStorage3D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
void mglTextureStorage3D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
void mglTextureStorage3DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
void mglClearTexImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, const void *data);
void mglClearTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data);
void mglCompressedTexImage3D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data);
void mglCompressedTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data);
void mglCompressedTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data);
void mglCompressedTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data);
void mglCompressedTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data);
void mglCompressedTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data);
void mglCopyTexImageCommon(GLMContext ctx, GLenum target, GLuint face, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
void mglCopyTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border);
void mglCopyTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
void mglCopyTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
void mglCopyTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
void mglCopyTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
void mglCopyTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
void mglCopyTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
void mglCopyTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
bool mglResolvePackBufferDst(GLMContext ctx, const char *api,
                                    void **pixels, Buffer **pack_buffer_out);
bool mglCheckPackBufferRange(GLMContext ctx, const Buffer *pack_buffer,
                                    uintptr_t offset, size_t bytes, GLenum type);
void mglApplyPackSwapBytes(GLMContext ctx, uint8_t *dst, size_t pitch,
                                  size_t row_bytes, size_t rows, GLenum type);
void mglGetTexImageImpl(GLMContext ctx, Texture *tex, GLenum target,
                               GLint level, GLenum format, GLenum type,
                               GLsizei bufSize, void *pixels);
void mglGetTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, void *pixels);
void mglGetnTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels);
void mglGetTextureImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels);
void mglGetTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void *pixels);
void mglGetCompressedTexImage(GLMContext ctx, GLenum target, GLint level, void *img);
void mglGetnCompressedTexImage(GLMContext ctx, GLenum target, GLint lod, GLsizei bufSize, void *pixels);
void mglGetCompressedTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void *pixels);
GLenum mglTextureViewClass(GLenum internalformat);
bool mglTextureViewParameter(const Texture *tex, GLenum pname, GLint *out);
bool mglTextureFormatSamplesStencil(const Texture *tex);
bool mglTextureSamplingComplete(const Texture *tex, GLenum min_filter, GLenum mag_filter);
void mglTextureViewFamilyWritten(Texture *writer);
bool mglTextureViewTargetCompatible(GLenum orig, GLenum target);
bool mglTextureViewUnlink(GLMContext ctx, Texture *tex);
void mglTextureView(GLMContext ctx, GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers);
GLboolean mglTexBufferInternalFormatValid(GLenum internalformat);
void mglTextureBuffer(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer);
void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size);
void mglTextureBufferRangeImpl(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer,
                                      GLintptr offset, GLsizeiptr size, bool whole_buffer);
void mglCompressedTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data);
void mglCompressedTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data);
void mglGetCompressedTextureImage(GLMContext ctx, GLuint texture, GLint level, GLsizei bufSize, void *pixels);
void mglGetTextureLevelParameteriv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLint *params);
void mglGetTextureLevelParameterfv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLfloat *params);
bool mglTextureParameterGetiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *params);
bool mglTextureParameterGetfv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLfloat *params);
bool mglTextureParameterGetIiv(TextureParameter *tex_params, GLenum pname, GLint *params);
bool mglTextureParameterGetIuiv(TextureParameter *tex_params, GLenum pname, GLuint *params);
bool mglTextureParameterGetTarget(GLMContext ctx, Texture *tex, GLenum pname, GLfloat *fparams,
                                         GLint *iparams, GLuint *uiparams);
void mglGetTextureParameterfv(GLMContext ctx, GLuint texture, GLenum pname, GLfloat *params);
void mglGetTextureParameterIiv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params);
void mglGetTextureParameterIuiv(GLMContext ctx, GLuint texture, GLenum pname, GLuint *params);
void mglGetTextureParameteriv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params);
bool mglGetTexParameterITargetValid(GLenum target);
void mglGetTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params);
void mglGetTexParameterIuiv(GLMContext ctx, GLenum target, GLenum pname, GLuint *params);
void mglSampleCoverage(GLMContext ctx, GLfloat value, GLboolean invert);


bool mglTexLevelInternalFormatCompressed(GLint internalformat);
void mglTextureTargetLevelDimensions(GLenum target, GLuint base_width,
                                     GLuint base_height, GLuint base_depth,
                                     GLuint level, GLuint *out_width,
                                     GLuint *out_height, GLuint *out_depth);

bool mglTextureStorageMultisampleMetadata(GLMContext ctx, Texture *tex,
                                          GLenum target, GLsizei samples,
                                          GLenum internalformat, GLsizei width,
                                          GLsizei height, GLsizei depth,
                                          GLboolean fixedsamplelocations,
                                          GLboolean proxy);

GLint mglCompressedInternalFormatToSizedUncompressed(GLint internalformat);
GLint mglTexLevelCanonicalInternalFormat(GLint internalformat);


int mglTryTexBufferLevelParameter(const Texture *tex, GLint level, GLenum pname,
                                  GLint *params, GLenum *err);
GLenum mglTextureQueryInternalFormat(const Texture *tex);
GLint mglTexLevelComponentBits(GLint internalformat, GLenum pname);
GLint mglTextureQueryCompressedImageSize(const Texture *tex, GLint level);
bool getParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname,
              GLint *iparam, GLfloat *fparam);
void *getBufferData(GLMContext ctx, Buffer *ptr);
Buffer *findBuffer(GLMContext ctx, GLuint buffer);
GLsizei mglSafeMaxTextureSize(GLMContext ctx);
GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);

GLint mglTexLevelComponentType(GLint internalformat, GLenum pname);

#endif
