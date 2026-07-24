/*
 * mgl_state_compat.m
 * MGL
 *
 * Implementation of the GL State Compatibility Subsystem.
 * See mgl_state_compat.h for the API contract.
 */

#import "mgl_state_compat.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>

BOOL mglNearlyEqual(double a, double b)
{
    return fabs(a - b) <= 0.00001;
}

MTLCompareFunction mglMTLCompareFunctionForGL(GLenum func,
                                               MTLCompareFunction fallback,
                                               const char *label)
{
    switch (func) {
        case GL_NEVER: return MTLCompareFunctionNever;
        case GL_LESS: return MTLCompareFunctionLess;
        case GL_EQUAL: return MTLCompareFunctionEqual;
        case GL_LEQUAL: return MTLCompareFunctionLessEqual;
        case GL_GREATER: return MTLCompareFunctionGreater;
        case GL_NOTEQUAL: return MTLCompareFunctionNotEqual;
        case GL_GEQUAL: return MTLCompareFunctionGreaterEqual;
        case GL_ALWAYS: return MTLCompareFunctionAlways;
        default: {
            static uint64_t s_badCompareFunctionCount = 0;
            uint64_t hit = ++s_badCompareFunctionCount;

            if (hit <= 32 || (hit % 256) == 0) {
                NSLog(@"MGL WARNING: invalid %s compare func=0x%x, fallback=%lu hit=%llu",
                      label ? label : "unknown",
                      func,
                      (unsigned long)fallback,
                      (unsigned long long)hit);
            }

            return fallback;
        }
    }
}

MTLWinding mglMTLWindingForGL(GLenum frontFace)
{
    switch (frontFace) {
        case GL_CW:
            return MTLWindingClockwise;
        case GL_CCW:
            return MTLWindingCounterClockwise;
        default: {
            static uint64_t s_badFrontFaceCount = 0;
            uint64_t hit = ++s_badFrontFaceCount;

            if (hit <= 32 || (hit % 256) == 0) {
                NSLog(@"MGL WARNING: invalid front face enum=0x%x, fallback=GL_CCW hit=%llu",
                      frontFace,
                      (unsigned long long)hit);
            }

            return MTLWindingCounterClockwise;
        }
    }
}

BOOL mglIsValidGLCompareFunction(GLenum func)
{
    switch (func) {
        case GL_NEVER:
        case GL_LESS:
        case GL_EQUAL:
        case GL_LEQUAL:
        case GL_GREATER:
        case GL_NOTEQUAL:
        case GL_GEQUAL:
        case GL_ALWAYS:
            return YES;
        default:
            return NO;
    }
}

BOOL mglIsValidGLBlendEquation(GLenum op)
{
    switch (op) {
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN:
        case GL_MAX:
            return YES;
        default:
            return NO;
    }
}

BOOL mglIsValidGLBlendFactor(GLenum factor)
{
    switch (factor) {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
        case GL_SRC1_COLOR:
        case GL_ONE_MINUS_SRC1_COLOR:
        case GL_SRC1_ALPHA:
        case GL_ONE_MINUS_SRC1_ALPHA:
            return YES;
        default:
            return NO;
    }
}

void mglLogRenderStateRepair(const char *field, GLenum value, GLenum fallback)
{
    static uint64_t s_stateRepairCount = 0;
    uint64_t hit = ++s_stateRepairCount;

    if (hit <= 64 || (hit % 512) == 0) {
        NSLog(@"MGL WARNING: repairing invalid render state %s=0x%x -> 0x%x hit=%llu",
              field ? field : "unknown",
              value,
              fallback,
              (unsigned long long)hit);
    }
}

BOOL mglShouldLogSmallBaseBinding(GLuint programName,
                                   int stage,
                                   int resourceType,
                                   GLuint binding,
                                   GLuint glName,
                                   GLsizeiptr rangeSize,
                                   NSUInteger reflectedSize)
{
    typedef struct MGLSmallBaseBindingLogKey_t {
        GLuint programName;
        int stage;
        int resourceType;
        GLuint binding;
        GLuint glName;
        GLsizeiptr rangeSize;
        NSUInteger reflectedSize;
        uint64_t hits;  /* 0 == empty slot; >=1 == occupied */
    } MGLSmallBaseBindingLogKey;

    /* Open-addressing hash table with linear probing. The table size is a
     * power of two so modulo reduces to a bitmask; the empty-slot sentinel is
     * hits==0, which is impossible for an occupied entry because hits starts
     * at 1 and only increments. */
    enum { kMGLSmallBaseLogTableSize = 128 };
    static MGLSmallBaseBindingLogKey s_keys[kMGLSmallBaseLogTableSize];
    static uint32_t s_keyCount = 0;
    static uint64_t s_overflowHits = 0;

    /* Mix the seven key fields into a 64-bit hash.  Each multiplier is a
     * distinct odd constant (Fibonacci/golden-ratio derived) so that
     * correlated inputs (e.g. binding==glName) still disperse. */
    uint64_t h = (uint64_t)programName
               ^ ((uint64_t)(uint32_t)stage * 0x9E3779B97F4A7C15ULL)
               ^ ((uint64_t)(uint32_t)resourceType * 0xC2B2AE3D27D4EB4FULL)
               ^ ((uint64_t)binding * 0x165667B19E3779F9ULL)
               ^ ((uint64_t)glName * 0x9E3779B185EBCA87ULL)
               ^ ((uint64_t)rangeSize * 0xD1B54A32D192ED03ULL)
               ^ (reflectedSize * 0xA24BAED4963EE407ULL);
    uint32_t startIdx = (uint32_t)(h & (kMGLSmallBaseLogTableSize - 1u));

    /* Probe for a matching key or an empty slot.  Remember the first
     * empty slot seen so we can insert without re-probing. */
    int32_t emptySlotIdx = -1;
    for (uint32_t probe = 0; probe < kMGLSmallBaseLogTableSize; probe++) {
        uint32_t idx = (startIdx + probe) & (kMGLSmallBaseLogTableSize - 1u);
        MGLSmallBaseBindingLogKey *key = &s_keys[idx];

        if (key->hits == 0u) {
            if (emptySlotIdx < 0) {
                emptySlotIdx = (int32_t)idx;
            }
            break;
        }

        if (key->programName == programName &&
            key->stage == stage &&
            key->resourceType == resourceType &&
            key->binding == binding &&
            key->glName == glName &&
            key->rangeSize == rangeSize &&
            key->reflectedSize == reflectedSize) {
            key->hits++;
            return key->hits <= 1u || (key->hits % 4096u) == 0u;
        }
    }

    /* Key not found — try to insert into the empty slot we recorded. */
    if (s_keyCount < kMGLSmallBaseLogTableSize && emptySlotIdx >= 0) {
        if (s_keyCount >= 32u && (s_keyCount % 16u) != 0u) {
            return NO;
        }
        MGLSmallBaseBindingLogKey *key = &s_keys[emptySlotIdx];
        key->programName = programName;
        key->stage = stage;
        key->resourceType = resourceType;
        key->binding = binding;
        key->glName = glName;
        key->rangeSize = rangeSize;
        key->reflectedSize = reflectedSize;
        key->hits = 1u;
        s_keyCount++;
        return YES;
    }

    s_overflowHits++;
    return s_overflowHits <= 8u || (s_overflowHits % 2048u) == 0u;
}
