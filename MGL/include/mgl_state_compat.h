/*
 * mgl_state_compat.h
 * MGL
 *
 * GL State Compatibility Subsystem: pure helpers for translating GL enum
 * state (compare functions, winding, blend equations/factors) to Metal
 * equivalents, validating GL enums, and rate-limited logging of state
 * repairs.  All functions here are pure (no self/ivar dependency) and may
 * be called from any translation unit.
 */

#ifndef MGL_STATE_COMPAT_H
#define MGL_STATE_COMPAT_H

#include "glcorearb.h"
#include "mgl_render_values.h"
#include <stdint.h>
#include <objc/objc.h>  /* BOOL */

#ifdef __cplusplus
extern "C" {
#endif

/* Floating-point near-equality helper used by state comparison paths. */
BOOL mglNearlyEqual(double a, double b);

/* Translate a GL compare function (GL_NEVER..GL_ALWAYS) to the matching
 * MTLCompareFunction.  On unrecognized enums, logs a rate-limited warning
 * and returns `fallback`.  `label` is a short human-readable tag for the
 * log line (may be NULL). */
uint32_t mglMTLCompareFunctionForGL(GLenum func,
                                    uint32_t fallback,
                                    const char *label);

/* Translate GL front-face winding (GL_CW/GL_CCW) to MTLWinding.  On
 * unrecognized enums, logs a rate-limited warning and returns
 * MTLWindingCounterClockwise (GL default). */
uint32_t mglMTLWindingForGL(GLenum frontFace);

/* Enum validators (return YES for valid GL enum, NO otherwise). */
BOOL mglIsValidGLCompareFunction(GLenum func);
BOOL mglIsValidGLBlendEquation(GLenum op);
BOOL mglIsValidGLBlendFactor(GLenum factor);

/* Rate-limited logging for state repair events.  `field` is a short tag
 * naming the state field (may be NULL).  The first 64 hits plus every
 * 512th subsequent hit are logged to avoid log flooding. */
void mglLogRenderStateRepair(const char *field, GLenum value, GLenum fallback);

/* Rate-limited gating for "small base binding" diagnostic logs.  Returns
 * YES when the (program, stage, resourceType, binding, glName, rangeSize,
 * reflectedSize) tuple should emit a log line.  Deduplicates by tuple key
 * with a 128-entry static table; each unique key logs up to 4 times then
 * every 1024th hit. */
BOOL mglShouldLogSmallBaseBinding(GLuint programName,
                                  int stage,
                                  int resourceType,
                                  GLuint binding,
                                  GLuint glName,
                                  GLsizeiptr rangeSize,
                                  uint64_t reflectedSize);

#ifdef __cplusplus
}
#endif

#endif /* MGL_STATE_COMPAT_H */
