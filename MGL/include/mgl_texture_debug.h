/*
 * mgl_texture_debug.h
 * MGL
 *
 * Debug / utility helpers extracted from textures.c as part of the
 * God Object decomposition (Task 8).  These functions provide byte-dump
 * diagnostics, sampling hash/zero probes, texture rect helpers, overflow
 * checked size arithmetic, zero-CPU upload resource tagging and texture
 * unit state tracing.
 */

#ifndef mgl_texture_debug_h
#define mgl_texture_debug_h

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <mach/vm_types.h>
#include <mach/mach_time.h>

#include "glm_context.h"

/* ---------------------------------------------------------------------------
 * Inline helpers (kept inline because they are used on hot paths).
 * ------------------------------------------------------------------------- */

/* High-resolution monotonic timestamp in milliseconds. */
static inline double mglTextureNowMs(void)
{
    static mach_timebase_info_data_t s_timebase = {0, 0};
    if (s_timebase.denom == 0) {
        (void)mach_timebase_info(&s_timebase);
    }

    uint64_t t = mach_absolute_time();
    long double ns = (long double)t * (long double)s_timebase.numer / (long double)s_timebase.denom;
    return (double)(ns / 1000000.0L);
}

/* Texture name accessor (NULL-safe) used by trace call sites. */
static inline GLuint mglTraceTextureNameC(Texture *tex)
{
    return tex ? tex->name : 0u;
}

/* ---------------------------------------------------------------------------
 * Diagnostics and utility functions.
 * ------------------------------------------------------------------------- */

void mglDumpBytesToStderr(const char *label,
                          const uint8_t *bytes,
                          size_t length,
                          size_t base_offset);

void mglDumpByteWindowToStderr(const char *label,
                               const uint8_t *bytes,
                               size_t total_length,
                               size_t requested_offset,
                               size_t window_length);

void mglDumpTextureUploadRowSamples(const char *prefix,
                                    const uint8_t *bytes,
                                    size_t total_length,
                                    size_t pitch,
                                    size_t row_bytes,
                                    GLsizei width,
                                    GLsizei height,
                                    GLsizei depth,
                                    size_t pixel_size);

void mglDumpTextureUploadSamples(Texture *tex,
                                 GLuint face,
                                 GLint level,
                                 const uint8_t *src,
                                 size_t src_total,
                                 size_t src_pitch,
                                 const uint8_t *dst,
                                 size_t dst_total,
                                 size_t dst_pitch,
                                 size_t pixel_size,
                                 GLsizei width,
                                 GLsizei height,
                                 GLsizei depth);

const char *mglTextureInitSourceName(GLuint source);

const char *mglBufferInitSourceName(MGLBufferInitSource source);

void mglDumpNativeBacktraceToStderr(const char *tag, size_t max_frames);

void mglRequestJavaThreadDumpForZeroCpuUpload(Texture *tex,
                                              GLuint face,
                                              GLint level,
                                              GLsizei width,
                                              GLsizei height,
                                              GLsizei depth,
                                              uint64_t warning_id);

void mglDumpTexSubImageZeroCpuResourceTag(GLMContext ctx,
                                          Texture *tex,
                                          TextureLevel *lvl,
                                          GLuint face,
                                          GLint level,
                                          GLint xoffset,
                                          GLint yoffset,
                                          GLint zoffset,
                                          GLsizei width,
                                          GLsizei height,
                                          GLsizei depth,
                                          GLenum format,
                                          GLenum type,
                                          const void *pixels_raw,
                                          const uint8_t *resolved_src,
                                          Buffer *resolved_unpack_buf,
                                          size_t required_bytes,
                                          size_t compact_upload_bytes,
                                          size_t src_pitch,
                                          size_t compact_upload_row_bytes,
                                          size_t pixel_size,
                                          uint64_t warning_id);

uint64_t mglHashBytesSampled(const void *data, size_t len);

bool mglLooksAllZero(const uint8_t *bytes, size_t len);

bool mglLooksAllZeroSampled(const uint8_t *bytes, size_t len);

bool mglTextureRectByteRange(TextureLevel *level,
                             size_t pixel_size,
                             size_t xoffset,
                             size_t yoffset,
                             size_t zoffset,
                             size_t width,
                             size_t height,
                             size_t depth,
                             size_t *byte_offset_out,
                             size_t *byte_span_out);

uint64_t mglHashTextureRect(const uint8_t *base,
                            size_t dst_pitch,
                            size_t dst_image_pitch,
                            size_t row_bytes,
                            size_t height,
                            size_t depth);

bool mglTextureRectLooksAllZero(const uint8_t *base,
                                size_t dst_pitch,
                                size_t dst_image_pitch,
                                size_t row_bytes,
                                size_t height,
                                size_t depth);

bool mglShouldTraceTextureUpload(Texture *tex,
                                 GLuint unpack_name,
                                 GLsizei width,
                                 GLsizei height,
                                 GLsizei depth,
                                 size_t required_bytes);

bool mglMulSizeT(size_t a, size_t b, size_t *out);

bool mglAddSizeT(size_t a, size_t b, size_t *out);

void mglTraceTextureUnitState(GLMContext ctx,
                              const char *api,
                              GLuint unit,
                              GLenum target,
                              GLuint texture,
                              Texture *bound);

size_t page_size_align(size_t size);

#endif /* mgl_texture_debug_h */
