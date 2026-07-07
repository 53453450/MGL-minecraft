/*
 * mgl_state_log.h
 * MGL
 *
 * GL State Logging Subsystem: pure helpers for formatting dirty-bit masks
 * and other GL state into human-readable strings for diagnostic logging.
 *
 * All functions here are pure (no self/ivar/global dependency beyond the
 * std C library) and may be called from any translation unit.
 */

#ifndef MGL_STATE_LOG_H
#define MGL_STATE_LOG_H

#include "glm_context.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Appends "|name" (or "name" when *first is true) to the NUL-terminated
 * string in `dst` (capacity `dstSize`).  Updates *first to false on a
 * successful append.  No-op if dst/name is NULL or the buffer is full. */
void mglAppendFlagName(char *dst, size_t dstSize, const char *name, bool *first);

/* Formats a dirty-bits mask into a pipe-separated string of flag names
 * ("VAO|STATE|TEX").  Writes "none" for 0 and "0x%x" if no known bits
 * match.  Always NUL-terminates (truncating if necessary). */
void mglFormatDirtyBits(uint32_t bits, char *dst, size_t dstSize);

#ifdef __cplusplus
}
#endif

#endif /* MGL_STATE_LOG_H */
