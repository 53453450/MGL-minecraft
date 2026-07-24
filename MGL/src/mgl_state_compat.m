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
        uint64_t hits;
    } MGLSmallBaseBindingLogKey;

    static MGLSmallBaseBindingLogKey s_keys[128];
    static uint32_t s_keyCount = 0;
    static uint64_t s_overflowHits = 0;

    for (uint32_t i = 0; i < s_keyCount; i++) {
        MGLSmallBaseBindingLogKey *key = &s_keys[i];
        if (key->programName == programName &&
            key->stage == stage &&
            key->resourceType == resourceType &&
            key->binding == binding &&
            key->glName == glName &&
            key->rangeSize == rangeSize &&
            key->reflectedSize == reflectedSize) {
            key->hits++;
            return key->hits <= 1 || (key->hits % 4096) == 0;
        }
    }

    if (s_keyCount < (uint32_t)(sizeof(s_keys) / sizeof(s_keys[0]))) {
        if (s_keyCount >= 32 && (s_keyCount % 16) != 0) {
            return NO;
        }
        s_keys[s_keyCount++] = (MGLSmallBaseBindingLogKey){
            .programName = programName,
            .stage = stage,
            .resourceType = resourceType,
            .binding = binding,
            .glName = glName,
            .rangeSize = rangeSize,
            .reflectedSize = reflectedSize,
            .hits = 1
        };
        return YES;
    }

    s_overflowHits++;
    return s_overflowHits <= 8 || (s_overflowHits % 2048) == 0;
}
