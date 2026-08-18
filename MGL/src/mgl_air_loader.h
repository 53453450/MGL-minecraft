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

#ifdef __cplusplus
extern "C" {
#endif

/* Value-state input for render-pipeline construction. Objective-C fills this
 * structure without exposing MTLRenderPipelineDescriptor. */
typedef struct MGLRenderPipelineDescriptorState {
    uint64_t vertex_program_instance;
    uint64_t vertex_program_generation;
    uint64_t fragment_program_instance;
    uint64_t fragment_program_generation;
    uint32_t color_count;
    uint32_t color_format[8];   /* MGLPixelFormat ABI values. */
    uint32_t depth_format;      /* MGLPixelFormatInvalid is zero. */
    uint32_t stencil_format;    /* MGLPixelFormat ABI value. */
    int      rasterization_enabled;
    int      icb_enabled;       /* indirect command buffers */
    int      alpha_to_coverage_enabled;
    int      alpha_to_one_enabled;
    uint32_t input_primitive_topology;
    /* AIR vertex layout. */
    uint32_t attrib_count;
    uint32_t attrib_format[32];
    uint32_t attrib_offset[32];
    uint32_t attrib_stride[32];
    uint32_t attrib_buffer_index[32];
    uint32_t attrib_step_function[32];
    uint32_t attrib_step_rate[32];
    uint32_t color_write_mask[8];
    uint32_t source_rgb_blend_factor[8];
    uint32_t destination_rgb_blend_factor[8];
    uint32_t source_alpha_blend_factor[8];
    uint32_t destination_alpha_blend_factor[8];
    uint32_t rgb_blend_operation[8];
    uint32_t alpha_blend_operation[8];
    uint32_t blending_enabled_mask;
    uint32_t raster_sample_count;
    uint32_t tessellation_partition_mode;
    uint32_t max_tessellation_factor;
    int      tessellation_factor_scale_enabled;
    uint32_t tessellation_factor_format;
    uint32_t tessellation_control_point_index_type;
    uint32_t tessellation_factor_step_function;
    uint32_t tessellation_output_winding_order;
} MGLRenderPipelineDescriptorState;

/* Compatibility alias retained for existing callers. */
typedef MGLRenderPipelineDescriptorState MGLPipelineDescriptorState;

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

/* Release the loader-owned PSO cache. Called by the C++ renderer when its
 * final device user shuts down. Safe to call repeatedly. */
void mglAirLoaderShutdown(void);

#ifdef __cplusplus
} // extern "C"
#endif
