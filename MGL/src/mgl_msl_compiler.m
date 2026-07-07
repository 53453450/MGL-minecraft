/*
 * mgl_msl_compiler.m
 * MGL
 *
 * Implementation of the MSL Compiler Subsystem.
 * See mgl_msl_compiler.h for the API contract.
 */

#import "mgl_msl_compiler.h"

#include <string.h>

/* mglEnvFlagEnabled is defined in MGLRenderer.m; declare it extern here so
 * the MTL4 compiler fast-path gate can query MGL_DISABLE_MTL4_COMPILER. */
extern BOOL mglEnvFlagEnabled(const char *name);

/* MTL4 compiler availability — match MGLRenderer.m's guard so the
 * MTL4LibraryDescriptor type is available when the fast path is compiled. */
#if __has_include(<Metal/MTL4Compiler.h>) && __has_include(<Metal/MTL4LibraryDescriptor.h>)
#import <Metal/MTL4Compiler.h>
#import <Metal/MTL4LibraryDescriptor.h>
#define MGL_HAS_MTL4_COMPILER 1
#else
#define MGL_HAS_MTL4_COMPILER 0
#endif

id<MTLLibrary> mglCompileMSL(id<MTLDevice> device,
                              id mtl4Compiler,
                              NSString *source,
                              MTLCompileOptions *options,
                              NSString *label,
                              NSError **error)
{
    if (!source) {
        return nil;
    }

    if ([source rangeOfString:@"EmitVertex"].location != NSNotFound ||
        [source rangeOfString:@"EndPrimitive"].location != NSNotFound) {
        static uint64_t s_unsupportedGeometryMSLSkipCount = 0;
        uint64_t hit = ++s_unsupportedGeometryMSLSkipCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL WARNING: Refusing to compile geometry-shader MSL with unsupported Metal emission semantics label=%@ hit=%llu",
                  label ?: @"shader",
                  (unsigned long long)hit);
        }
        if (error) {
            NSString *description =
                @"Geometry shader MSL contains EmitVertex/EndPrimitive, which Metal cannot compile directly";
            *error = [NSError errorWithDomain:@"MGLRenderer"
                                         code:GL_INVALID_OPERATION
                                     userInfo:@{NSLocalizedDescriptionKey: description}];
        }
        return nil;
    }

#if MGL_HAS_MTL4_COMPILER
    if (mtl4Compiler && !mglEnvFlagEnabled("MGL_DISABLE_MTL4_COMPILER")) {
        if (@available(macOS 26.0, *)) {
            MTL4LibraryDescriptor *descriptor = [[MTL4LibraryDescriptor alloc] init];
            descriptor.source = source;
            descriptor.options = options;
            descriptor.name = label;

            id<MTLLibrary> library = [mtl4Compiler newLibraryWithDescriptor:descriptor error:error];
            if (library) {
                return library;
            }

            static uint64_t s_mtl4LibraryFallbackCount = 0;
            uint64_t hit = ++s_mtl4LibraryFallbackCount;
            if (hit <= 8ull || (hit % 256ull) == 0ull) {
                NSError *compileError = error ? *error : nil;
                NSLog(@"MGL WARNING: Metal 4 library compile failed label=%@ hit=%llu, falling back to MTLDevice: %@",
                      label ?: @"shader",
                      (unsigned long long)hit,
                      compileError.localizedDescription ?: compileError);
            }
        }
    }
#endif

    return [device newLibraryWithSource:source options:options error:error];
}

id<MTLFunction> mglNewFunctionFromLibrary(id<MTLLibrary> library,
                                           NSString *entryName,
                                           const char *source,
                                           NSString *label)
{
    if (!library || !entryName) {
        return nil;
    }

    /*
     * SPIRV-Cross emits this function constant when it emulates R32UI linear
     * texture atomics. Metal requires such functions to be explicitly
     * specialized before they can be used in a render pipeline.
     */
    BOOL needsR32UI = (source &&
                       strstr(source, "spvLinearTextureAlignmentOverride") &&
                       strstr(source, "[[function_constant(65535)]]"));

    if (needsR32UI) {
        MTLFunctionConstantValues *values = [[MTLFunctionConstantValues alloc] init];

        uint32_t alignment = 4u;
        [values setConstantValue:&alignment type:MTLDataTypeUInt atIndex:65535];

        __autoreleasing NSError *error = nil;
        id<MTLFunction> function = [library newFunctionWithName:entryName
                                                 constantValues:values
                                                          error:&error];
        if (!function) {
            NSLog(@"MGL ERROR: Failed to specialize %@ with function constants: %@",
                  label ?: entryName,
                  error.localizedDescription ?: error);
        }
        return function;
    }

    return [library newFunctionWithName:entryName];
}
