/*
 * mgl_byte_hash.h
 * MGL
 *
 * Byte Hash / Format / Dump Subsystem: pure FNV-1a hashing and hex-dump
 * helpers used by trace logging, vertex attribute comparison, and
 * pipeline signature computation.
 *
 * All functions here are pure: no self/ivar dependency, no renderer state.
 * They operate solely on byte buffers and produce uint64_t hashes or
 * formatted string output.
 *
 * Dependencies: standard C library only, plus mgl_trace_log for ObjC dumps.
 */

#ifndef MGL_BYTE_HASH_H
#define MGL_BYTE_HASH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Single FNV-1a hash step.  Used by pipeline/vertex-descriptor signature
 * computation in MGLRenderer.m.  Kept as static inline for hot-path
 * callers that accumulate hashes in loops. */
static inline uint64_t mglHashStepU64(uint64_t hash, uint64_t value)
{
    /* 64-bit FNV-1a */
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

/* FNV-1a hash of a byte buffer, sampling head and tail (up to 1024 bytes
 * each) for large buffers.  Returns 0 for NULL/empty input.  Used by
 * trace logging to fingerprint buffer/texture upload payloads. */
uint64_t mglTraceHashBytes(const void *data, size_t len);

/* Formats up to 8 bytes of `data` as a colon-separated hex string into
 * `out` (e.g. "ab:cd:ef:01...").  Writes "-" for NULL/empty input. */
void mglTraceFormatBytes(const void *data, size_t len, char *out, size_t outSize);

/* Dumps a byte buffer to the trace log as labeled hex+ascii rows (16 bytes/row).
 * ObjC only (uses NSString label). */
#ifdef __OBJC__
void mglDumpBytesToLog(NSString *label,
                       const uint8_t *bytes,
                       size_t length,
                       size_t baseOffset);
#endif

/* Full FNV-1a hash over an entire byte buffer (no head/tail sampling).
 * Used by vertex attribute byte comparison. */
uint64_t mglHashVertexBytesFNV1a(const uint8_t *bytes, size_t length);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BYTE_HASH_H */
