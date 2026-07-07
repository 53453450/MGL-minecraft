/*
 * mgl_msl_compiler.h
 * MGL
 *
 * MSL Compiler Subsystem.
 *
 * Compiles MSL source strings into Metal libraries and resolves entry-point
 * functions.  Bridges the MTL4 compiler fast path (macOS 26+) with the
 * classic MTLDevice fallback.  Also handles R32UI function-constant
 * specialization for SPIRV-Cross's linear-texture-alignment emulation.
 *
 * All functions are pure (take device/compiler as params, no self/ivar).
 *
 * Dependencies: Metal.framework (id<MTLDevice>, id<MTLLibrary>,
 * id<MTLFunction>, id<MTL4Compiler>, MTLCompileOptions) +
 * Foundation.framework (NSString, NSError) + glm_context.h (for
 * mglEnvFlagEnabled — declared in MGLRenderer.m, made available via extern).
 */

#ifndef MGL_MSL_COMPILER_H
#define MGL_MSL_COMPILER_H

#include "glcorearb.h"

#include <objc/objc.h>   /* BOOL */

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __OBJC__

/* Compiles MSL source into a Metal library.
 *
 * `device` is the MTLDevice (required, for fallback path).
 * `mtl4Compiler` is the MTL4 compiler (nullable, for fast path on macOS 26+).
 * `source` is the MSL source string (required).
 * `options` is the compile options (nullable → default options).
 * `label` is a debug label (nullable).
 * `error` is the error output (nullable).
 *
 * Returns nil on failure (error set).  Refuses to compile MSL containing
 * EmitVertex/EndPrimitive (geometry-shader emission — Metal cannot compile
 * these directly). */
id<MTLLibrary> mglCompileMSL(id<MTLDevice> device,
                              id mtl4Compiler,  /* id<MTL4Compiler>, untyped to avoid MTL4 header dep in .h */
                              NSString *source,
                              MTLCompileOptions *options,
                              NSString *label,
                              NSError **error);

/* Resolves an entry-point function from a compiled library.
 *
 * Handles R32UI function-constant specialization when `source` contains
 * SPIRV-Cross's `spvLinearTextureAlignmentOverride` + `[[function_constant(65535)]]`.
 *
 * `library` is the compiled Metal library (required).
 * `entryName` is the function entry point name (required).
 * `source` is the original MSL source (nullable, used for R32UI detection).
 * `label` is a debug label (nullable). */
id<MTLFunction> mglNewFunctionFromLibrary(id<MTLLibrary> library,
                                           NSString *entryName,
                                           const char *source,
                                           NSString *label);

#endif /* __OBJC__ */

#ifdef __cplusplus
}
#endif

#endif /* MGL_MSL_COMPILER_H */
