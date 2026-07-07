/*
 * mgl_byte_hash.m
 * MGL
 *
 * Implementation of the Byte Hash / Format / Dump Subsystem.
 * See mgl_byte_hash.h for the API contract.
 *
 * Pure FNV-1a hashing and hex formatting helpers extracted from
 * MGLRenderer.m.  No renderer state dependency.
 */

#import "mgl_byte_hash.h"
#import "mgl_trace_log.h"

#include <stdio.h>
#include <string.h>

uint64_t mglTraceHashBytes(const void *data, size_t len)
{
    if (!data || len == 0) {
        return 0ull;
    }

    const uint8_t *bytes = (const uint8_t *)data;
    size_t head = len < 1024 ? len : 1024;
    uint64_t hash = 1469598103934665603ull;

    for (size_t i = 0; i < head; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ull;
    }

    if (len > head) {
        const uint8_t *tail = bytes + (len - head);
        for (size_t i = 0; i < head; i++) {
            hash ^= (uint64_t)tail[i];
            hash *= 1099511628211ull;
        }
    }

    hash ^= (uint64_t)len;
    hash *= 1099511628211ull;
    return hash;
}

void mglTraceFormatBytes(const void *data, size_t len, char *out, size_t outSize)
{
    if (!out || outSize == 0) {
        return;
    }

    if (!data || len == 0) {
        snprintf(out, outSize, "-");
        return;
    }

    const uint8_t *bytes = (const uint8_t *)data;
    size_t sample = len < 8 ? len : 8;
    size_t used = 0;

    for (size_t i = 0; i < sample && used + 3 < outSize; i++) {
        int wrote = snprintf(out + used, outSize - used, "%02x", bytes[i]);
        if (wrote <= 0) {
            break;
        }
        used += (size_t)wrote;
        if (i + 1 < sample && used + 2 < outSize) {
            out[used++] = ':';
            out[used] = '\0';
        }
    }

    if (len > sample && used + 4 < outSize) {
        snprintf(out + used, outSize - used, "...");
    }
}

void mglDumpBytesToLog(NSString *label,
                       const uint8_t *bytes,
                       size_t length,
                       size_t baseOffset)
{
    if (!bytes || length == 0) {
        MGLTraceNSLog(@"MGL DUMP %@ empty", label ?: @"(null)");
        return;
    }

    const size_t row = 16u;
    for (size_t off = 0; off < length; off += row) {
        size_t n = MIN(row, length - off);
        char hex[3 * row + 1];
        char ascii[row + 1];
        size_t hp = 0;

        for (size_t i = 0; i < n; i++) {
            uint8_t b = bytes[off + i];
            int wrote = snprintf(hex + hp, sizeof(hex) - hp, "%02x", b);
            if (wrote <= 0) {
                break;
            }
            hp += (size_t)wrote;
            if (i + 1 < n && hp + 1 < sizeof(hex)) {
                hex[hp++] = ' ';
            }
            ascii[i] = (b >= 32u && b <= 126u) ? (char)b : '.';
        }
        hex[hp] = '\0';
        ascii[n] = '\0';

        MGLTraceNSLog(@"MGL DUMP %@ +0x%zx: %-47s |%s|",
                      label ?: @"(null)",
                      baseOffset + off,
                      hex,
                      ascii);
    }
}

uint64_t mglHashVertexBytesFNV1a(const uint8_t *bytes, size_t length)
{
    uint64_t hash = 1469598103934665603ull;
    if (!bytes) {
        return hash;
    }
    for (size_t i = 0; i < length; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}
