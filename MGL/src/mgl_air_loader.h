/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

//------------------------------------------------------------------------------------------------
// Pure C interface for loading AIR metallibs and creating pipeline states.
//
// Objective-C callers never see MTL::* types. Returned libraries and pipeline
// states are owned references and must be released with mglAirRelease. The
// loader keeps an additional reference only for cached render pipelines.
//------------------------------------------------------------------------------------------------
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "mgl_render_api_query.h" /* MGLRenderPipelineDescriptorState */

#ifdef __cplusplus
extern "C" {
#endif

/* device is an internal MTL::Device*. bytes contains a metallib image.
 * Returns 0 with an owned library on success. */
int mglAirLoadLibrary(const void* device, const unsigned char* bytes, size_t size,
                      void** library_out, char* err, size_t errcap);

/* Function pointers are the selected MTL::Function variants. Returns 0 with
 * an owned pipeline state on success. */
int mglAirCreateRenderPipeline(const void* device, void* vs_function, void* fs_function,
                               const MGLPipelineDescriptorState* desc, void** pso_out,
                               char* err, size_t errcap);

/* Creates a render pipeline with an optional borrowed binary archive. Complete
 * vertex/fragment pipelines query the archive before compiling. Vertex-only
 * capture or rasterizer-discard pipelines are not archived because Metal
 * rejects those records during serialization. */
int mglAirCreateRenderPipelineWithArchive(
    const void* device, void* vs_function, void* fs_function,
    const MGLRenderPipelineDescriptorState* desc, void* binary_archive,
    void** pso_out, char* err, size_t errcap);

/* Creates the "main" compute pipeline from a borrowed MTL::Library*. The
 * returned pipeline is owned and is not cached by this loader. */
int mglAirCreateComputePipeline(const void* device, void* library,
                                void** pso_out, char* err, size_t errcap);

/* Releases an owned object returned by this loader. */
void mglAirRelease(void* obj);

/* PERF-029: compile via Metal completion handler; caller blocks on a
 * semaphore until the PSO is ready. Returns 0 with an owned pipeline state. */
int mglAirCreateRenderPipelineStateAsyncWait(void* device, void* descriptor,
                                             void** pso_out, char* err,
                                             size_t errcap);

/* Release the loader-owned PSO cache. Called by the C++ renderer when its
 * final device user shuts down. Safe to call repeatedly. */
void mglAirLoaderShutdown(void);

#ifdef __cplusplus
} // extern "C"
#endif
