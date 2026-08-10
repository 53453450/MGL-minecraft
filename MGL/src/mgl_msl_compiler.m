/*
 * mgl_msl_compiler.m
 * MGL
 *
 * Implementation of the MSL Compiler Subsystem.
 * See mgl_msl_compiler.h for the API contract.
 */

#import "mgl_msl_compiler.h"
#import "mgl_frame_activity.h"
#import "mgl_metal_ref.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"

#include <string.h>
#include <stdatomic.h>
#include <mach/mach_time.h>
#include <objc/runtime.h>

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

static NSCache<NSString *, id<MTLLibrary>> *mglMSLLibraryCache(void)
{
    static NSCache<NSString *, id<MTLLibrary>> *cache = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        cache = [[NSCache alloc] init];
        cache.name = @"MGL MSL Library Cache";
        cache.countLimit = 512;
    });
    return cache;
}

static NSString *mglMSLCompileCacheKey(id<MTLDevice> device,
                                       NSString *source,
                                       MTLCompileOptions *options)
{
    if (!source) {
        return nil;
    }
    if (options) {
        return nil;
    }

    NSMutableString *key = [NSMutableString stringWithFormat:@"%p\n", device];
    [key appendString:@"options=nil\n"];
    [key appendString:source];
    return key;
}

static NSError *mglMSLMetalCppError(const char *message)
{
    NSString *description = message && message[0]
        ? [NSString stringWithUTF8String:message]
        : @"Metal-cpp library compilation failed";
    return [NSError errorWithDomain:@"MGLMSLCompiler"
                               code:1
                           userInfo:@{NSLocalizedDescriptionKey: description}];
}

static id<MTLLibrary> mglCompileMSLWithMetalCpp(
    id compiler,
    NSString *source,
    MTLCompileOptions *options,
    NSString *label,
    NSError **error)
{
    void *library = NULL;
    char message[1024] = {0};
    if (mglRenderCppCompileLibrary(
            compiler ? (__bridge void *)compiler : NULL,
            (__bridge void *)source,
            options ? (__bridge void *)options : NULL,
            label.UTF8String,
            &library, message, sizeof(message)) != 0 || !library) {
        if (error) {
            *error = mglMSLMetalCppError(message);
        }
        return nil;
    }
    if (error) {
        *error = nil;
    }
    return (__bridge_transfer id<MTLLibrary>)library;
}

static id<MTLFunction> mglCreateFunctionWithMetalCpp(
    id<MTLLibrary> library,
    NSString *entryName,
    MTLFunctionConstantValues *values,
    NSError **error)
{
    void *function = NULL;
    char message[1024] = {0};
    if (mglRenderCppCreateFunction(
            (__bridge void *)library, entryName.UTF8String,
            values ? (__bridge void *)values : NULL,
            &function, message, sizeof(message)) != 0 || !function) {
        if (error) *error = mglMSLMetalCppError(message);
        return nil;
    }
    if (error) *error = nil;
    return (__bridge_transfer id<MTLFunction>)function;
}

static NSMutableDictionary<NSString *, id<MTLFunction>> *mglFunctionCacheForLibrary(id<MTLLibrary> library)
{
    static const char kMGLFunctionCacheAssociationKey = 0;
    if (!library) {
        return nil;
    }

    @synchronized (library) {
        NSMutableDictionary<NSString *, id<MTLFunction>> *cache =
            objc_getAssociatedObject(library, &kMGLFunctionCacheAssociationKey);
        if (!cache) {
            cache = [[NSMutableDictionary alloc] initWithCapacity:4];
            objc_setAssociatedObject(library,
                                     &kMGLFunctionCacheAssociationKey,
                                     cache,
                                     OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        }
        return cache;
    }
}

static NSString *mglFunctionCacheKey(NSString *entryName, BOOL needsR32UI)
{
    if (!entryName) {
        return nil;
    }
    return [NSString stringWithFormat:@"%@:%@", needsR32UI ? @"r32ui" : @"plain", entryName];
}

static id<MTLLibrary> mglCompileMSLWithTiming(id<MTLDevice> device,
                                             NSString *source,
                                             MTLCompileOptions *options,
                                             NSString *label,
                                             NSError **error)
{
    if (mglPerfSummaryEnabled()) {
        /* mach_timebase_info is constant for the process lifetime. */
        static mach_timebase_info_data_t s_tb;
        static dispatch_once_t s_tbOnce;
        dispatch_once(&s_tbOnce, ^{ mach_timebase_info(&s_tb); });

        uint64_t compile_start = mach_absolute_time();
        id<MTLLibrary> library =
            mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
                    mglRenderCppGetDevice()
                ? mglCompileMSLWithMetalCpp(nil, source, options, label,
                                            error)
                : [device newLibraryWithSource:source
                                       options:options
                                         error:error];
        uint64_t compile_end = mach_absolute_time();
        double elapsed = (double)(compile_end - compile_start) * s_tb.numer / s_tb.denom / 1e9;
        MGL_FRAME_ADD(g_mglShaderCompileTimeSinceSwap, elapsed);
        MGL_FRAME_INC(g_mglShaderCompilesSinceSwap);
        if (library) {
            mglMetalCountCreate(MGLMetalKindLibrary);
        }
        return library;
    }

    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice()) {
        return mglCompileMSLWithMetalCpp(nil, source, options, label, error);
    }
    return [device newLibraryWithSource:source options:options error:error];
}

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

    /* The Metal compiler APIs return autoreleased NSError objects. If we
     * write *error while inside the @autoreleasepool below, the pool drain
     * at scope exit releases the error and leaves the caller with a
     * dangling pointer (compileShader: crashes in [error localizedDescription]).
     * Capture the error into a strong __block reference instead and publish
     * it to *error only after the pool has been drained. */
    __block id<MTLLibrary> result = nil;
    __block NSError *capturedError = nil;

    @autoreleasepool {
        NSString *cacheKey = mglMSLCompileCacheKey(device, source, options);
        id<MTLLibrary> cachedLibrary = cacheKey ? [mglMSLLibraryCache() objectForKey:cacheKey] : nil;
        if (cachedLibrary) {
            result = cachedLibrary;
            capturedError = nil;
        } else if ([source rangeOfString:@"EmitVertex"].location != NSNotFound ||
                   [source rangeOfString:@"EndPrimitive"].location != NSNotFound) {
            static uint64_t s_unsupportedGeometryMSLSkipCount = 0;
            uint64_t hit = ++s_unsupportedGeometryMSLSkipCount;
            if (hit <= 16ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL WARNING: Refusing to compile geometry-shader MSL with unsupported Metal emission semantics label=%@ hit=%llu",
                      label ?: @"shader",
                      (unsigned long long)hit);
            }
            NSString *description =
                @"Geometry shader MSL contains EmitVertex/EndPrimitive, which Metal cannot compile directly";
            capturedError = [NSError errorWithDomain:@"MGLRenderer"
                                                 code:GL_INVALID_OPERATION
                                             userInfo:@{NSLocalizedDescriptionKey: description}];
        } else {
#if MGL_HAS_MTL4_COMPILER
            static _Atomic uint32_t s_mtl4FailCount = 0;
            const uint32_t kMTL4FallbackThreshold = 8u;
            if (mtl4Compiler &&
                !mglEnvFlagEnabled("MGL_DISABLE_MTL4_COMPILER") &&
                atomic_load_explicit(&s_mtl4FailCount, memory_order_relaxed) < kMTL4FallbackThreshold) {
                if (@available(macOS 26.0, *)) {
                    BOOL useMetalCpp =
                        mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
                        mglRenderCppGetDevice();
                    MTL4LibraryDescriptor *descriptor = nil;
                    if (!useMetalCpp) {
                        descriptor = [[MTL4LibraryDescriptor alloc] init];
                        descriptor.source = source;
                        descriptor.options = options;
                        descriptor.name = label;
                    }

                    id<MTLLibrary> library = nil;
                    if (mglPerfSummaryEnabled()) {
                        /* mach_timebase_info is constant for the process lifetime. */
                        static mach_timebase_info_data_t s_tb;
                        static dispatch_once_t s_tbOnce;
                        dispatch_once(&s_tbOnce, ^{ mach_timebase_info(&s_tb); });

                        uint64_t compile_start = mach_absolute_time();
                        if (useMetalCpp) {
                            library = mglCompileMSLWithMetalCpp(
                                mtl4Compiler, source, options, label,
                                &capturedError);
                        } else {
                            library = [mtl4Compiler
                                newLibraryWithDescriptor:descriptor
                                                  error:&capturedError];
                        }
                        uint64_t compile_end = mach_absolute_time();
                        double elapsed = (double)(compile_end - compile_start) * s_tb.numer / s_tb.denom / 1e9;
                        MGL_FRAME_ADD(g_mglShaderCompileTimeSinceSwap, elapsed);
                        MGL_FRAME_INC(g_mglShaderCompilesSinceSwap);
                    } else {
                        if (useMetalCpp) {
                            library = mglCompileMSLWithMetalCpp(
                                mtl4Compiler, source, options, label,
                                &capturedError);
                        } else {
                            library = [mtl4Compiler
                                newLibraryWithDescriptor:descriptor
                                                  error:&capturedError];
                        }
                    }
                    if (library) {
                        mglMetalCountCreate(MGLMetalKindLibrary);
                        capturedError = nil;
                        if (cacheKey) {
                            [mglMSLLibraryCache() setObject:library forKey:cacheKey];
                        }
                        result = library;
                    } else {
                        atomic_fetch_add_explicit(&s_mtl4FailCount, 1u, memory_order_relaxed);
                        static uint64_t s_mtl4LibraryFallbackCount = 0;
                        uint64_t hit = ++s_mtl4LibraryFallbackCount;
                        if (hit <= 8ull || (hit % 256ull) == 0ull) {
                            NSLog(@"MGL WARNING: Metal 4 library compile failed label=%@ hit=%llu, falling back to MTLDevice: %@",
                                  label ?: @"shader",
                                  (unsigned long long)hit,
                                  capturedError.localizedDescription ?: capturedError);
                        }
                    }
                }
            }
#endif
            if (!result) {
                capturedError = nil;
                result = mglCompileMSLWithTiming(
                    device, source, options, label, &capturedError);
                if (result && cacheKey) {
                    [mglMSLLibraryCache() setObject:result forKey:cacheKey];
                }
            }
        }
    }

    if (error) {
        *error = capturedError;
    }
    return result;
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
    NSString *cacheKey = mglFunctionCacheKey(entryName, needsR32UI);
    NSMutableDictionary<NSString *, id<MTLFunction>> *functionCache =
        mglFunctionCacheForLibrary(library);
    if (cacheKey && functionCache) {
        @synchronized (library) {
            id<MTLFunction> cachedFunction = functionCache[cacheKey];
            if (cachedFunction) {
                return cachedFunction;
            }
        }
    }

    id<MTLFunction> function = nil;
    if (needsR32UI) {
        MTLFunctionConstantValues *values = [[MTLFunctionConstantValues alloc] init];

        uint32_t alignment = 4u;
        [values setConstantValue:&alignment type:MTLDataTypeUInt atIndex:65535];

        __autoreleasing NSError *error = nil;
        if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
            mglRenderCppGetDevice()) {
            function = mglCreateFunctionWithMetalCpp(
                library, entryName, values, &error);
        } else {
            function = [library newFunctionWithName:entryName
                                     constantValues:values
                                              error:&error];
        }
        if (function) {
            mglMetalCountCreate(MGLMetalKindFunction);
        }
        if (!function) {
            NSLog(@"MGL ERROR: Failed to specialize %@ with function constants: %@",
                  label ?: entryName,
                  error.localizedDescription ?: error);
        }
    } else {
        if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
            mglRenderCppGetDevice()) {
            function = mglCreateFunctionWithMetalCpp(
                library, entryName, nil, nil);
        } else {
            function = [library newFunctionWithName:entryName];
        }
    }

    if (function && cacheKey && functionCache) {
        @synchronized (library) {
            functionCache[cacheKey] = function;
        }
    }
    return function;
}
