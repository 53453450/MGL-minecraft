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
 * string in `dst` (capacity `dstSize`).  *used tracks the current logical
 * length of dst (including NUL) to avoid O(n^2) strlen on repeated appends;
 * the caller initializes it to strlen(dst)+1 (or 1 for an empty string).
 * Updates *first to false and *used to the new length on a successful
 * append.  No-op if dst/name is NULL or the buffer is full. */
void mglAppendFlagName(char *dst, size_t dstSize, size_t *used,
                       const char *name, bool *first);

/* Formats a dirty-bits mask into a pipe-separated string of flag names
 * ("VAO|STATE|TEX").  Writes "none" for 0 and "0x%x" if no known bits
 * match.  Always NUL-terminates (truncating if necessary). */
void mglFormatDirtyBits(uint32_t bits, char *dst, size_t dstSize);

#ifdef __cplusplus
}
#endif

#endif /* MGL_STATE_LOG_H */
