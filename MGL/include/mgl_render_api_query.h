/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_QUERY_H
#define MGL_RENDER_API_QUERY_H

/* Declarations for the query slice of the renderer facade.
 * Value layouts are defined here; types of other subsystems that are only
 * passed by pointer are incomplete (mgl_render_fwd.h). */

#include "glcorearb.h"
#include "mgl_render_fwd.h"

/* Query vocabulary: the two result values, the compare functions, and the
 * "is this the samples-passed target" question. */
static inline uint32_t mglQuerySignaledValue(void) { return GL_SIGNALED; }
/* The EQUAL compare function, for callers that need the enum value itself. */
static inline uint32_t mglCompareFunctionEqualValue(void) { return GL_EQUAL; }
static inline uint32_t mglQueryUnsignaledValue(void) { return GL_UNSIGNALED; }

typedef enum MGLCompareFunction {
    MGL_COMPARE_INVALID = 0,
    MGL_COMPARE_NEVER,
    MGL_COMPARE_LESS,
    MGL_COMPARE_EQUAL,
    MGL_COMPARE_LEQUAL,
    MGL_COMPARE_GREATER,
    MGL_COMPARE_NOTEQUAL,
    MGL_COMPARE_GEQUAL,
    MGL_COMPARE_ALWAYS
} MGLCompareFunction;

static inline MGLCompareFunction mglCompareFunctionOf(uint32_t func)
{
    switch (func) {
        case GL_NEVER: return MGL_COMPARE_NEVER;
        case GL_LESS: return MGL_COMPARE_LESS;
        case GL_EQUAL: return MGL_COMPARE_EQUAL;
        case GL_LEQUAL: return MGL_COMPARE_LEQUAL;
        case GL_GREATER: return MGL_COMPARE_GREATER;
        case GL_NOTEQUAL: return MGL_COMPARE_NOTEQUAL;
        case GL_GEQUAL: return MGL_COMPARE_GEQUAL;
        case GL_ALWAYS: return MGL_COMPARE_ALWAYS;
        default: return MGL_COMPARE_INVALID;
    }
}

static inline bool mglQueryTargetIsSamplesPassed(uint32_t target)
{
    return target == GL_SAMPLES_PASSED;
}

static inline bool mglQueryTargetIsRenderbuffer(uint32_t target)
{
    return target == GL_RENDERBUFFER;
}

static inline bool mglInternalFormatIsRGBA8(uint32_t fmt)
{
    return fmt == GL_RGBA8;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Value-state input for render-pipeline construction (was mgl_air_loader.h).
 * Frame/Pass TUs include this header instead of Compiler air_loader (G4). */
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

/* Device capability snapshot produced by the Metal-cpp owner.  The C ABI
 * carries only integer/value state; the MTL::Device is used exclusively by
 * mgl_render.cpp while populating this record. */
typedef struct MGLRenderCapabilityState_t {
    uint32_t family;
    uint32_t is_virtualized;
    uint32_t supports8x_msaa;
    uint64_t max_sample_count;
    uint64_t max_texture_dimensions;
    uint32_t bug_msl_pipeline_rejection;
    uint64_t command_buffer_recovery_limit;
    uint64_t max_concurrent_command_buffers;
    uint64_t texture_alignment_bytes;
    uint32_t conservative_cpu_cache_mode;
} MGLRenderCapabilityState;

int mglRenderErrorIsNone(uint32_t error);

uint32_t mglRenderErrorNone(void);

uint32_t mglRenderErrorInvalidOperation(void);

uint32_t mglRenderErrorInvalidValue(void);

uint32_t mglRenderErrorOutOfMemory(void);

uint32_t mglRenderGLBoolean(int value);

int mglRenderSerializeBinaryArchive(void *binary_archive,
                                       void *url,
                                       char *err,
                                       size_t errcap);

int mglRenderSetVisibilityResultMode(void *render_encoder,
                                        uint32_t mode,
                                        uint64_t offset);

int mglRenderSetVisibilityResultModeForRenderEncoderOwner(MGLRenderEncoderOwner *render_encoder_owner, uint32_t mode, uint64_t offset);

int mglRenderSampleTimestamps(uint64_t *cpu_timestamp_out,
                                 uint64_t *gpu_timestamp_out);

typedef struct MGLDirtyDomainPlan {
    int has_dirty;
    int sync_render_pass;
    int bind_fbo_attachments;
    int remap_buffers;
    int defer_buffer_map;
    int bind_textures;
    int vao_path;
    int buffer_path;
    int render_state_path;
    int sync_pipeline;
    int dirty_buffer_data;
} MGLDirtyDomainPlan;

typedef struct MGLRenderFboMatchCacheState_t {
    uint32_t fbo_name;
    uint64_t generation;
    int result;
} MGLRenderFboMatchCacheState;

int mglRenderSetFboMatchCache(MGLRenderPassIdentityOwner *owner, const MGLRenderFboMatchCacheState *cache);

int mglRenderPendingEventPrepare(MGLPendingEventOwner *owner_handle, int sync_name, void **event_out);

int mglRenderPendingEventDetach(MGLPendingEventOwner *owner_handle, int *sync_name_out, void **event_out);


/* Renderer initialization state as a C ABI value, never a borrowed object. */
int mglRenderIsInitialized(void);

/* Query device capabilities through Metal-cpp and return a pure value-state
 * snapshot. The device pointer is borrowed for the duration of the call. */
int mglRenderQueryCapability(void *device,
                                MGLRenderCapabilityState *state_out);

void mglRenderGetSync(GLMContext glm_ctx, Sync *sync);

unsigned int mglRenderGetSyncStatus(GLMContext glm_ctx, Sync *sync);

uint64_t mglRenderGetGPUTimestamp(GLMContext glm_ctx);

void mglRenderBeginTimerQueryCallback(GLMContext glm_ctx);

uint64_t mglRenderEndTimerQueryCallback(GLMContext glm_ctx);

void mglRenderBeginSampleQueryCallback(GLMContext glm_ctx,
                                          unsigned int target);

uint64_t mglRenderEndSampleQueryCallback(GLMContext glm_ctx);

int mglRenderIsSmallRGBA8(uint32_t width, uint32_t height, uint32_t internalformat);

int mglRenderIsValidGLCompareFunction(uint32_t func);

int mglRenderIsValidGLBlendEquation(uint32_t op);

int mglRenderIsValidGLBlendFactor(uint32_t factor);

/* Return stable device identity data for platform-neutral cache naming. */
int mglRenderGetDeviceIdentity(const void *device,
                                  uint64_t *registry_id_out,
                                  char *name_out,
                                  size_t name_capacity);

int mglRenderGetOrCreateDepthStencilState(MGLPipelineCacheOwner *owner, const MGLRenderDepthStencilDescriptorState *descriptor, void **depth_stencil_state_out, int *created_out);

int mglRenderCreateQueryStateOwner(uint32_t visibility_slot_count, MGLQueryStateOwner **owner_out);

int mglRenderBeginSampleQuery(MGLQueryStateOwner *owner, uint32_t counting, const char *buffer_label, void **visibility_buffer_out);

void mglRenderEndSampleQuery(MGLQueryStateOwner *owner);

int mglRenderIsSampleQueryActive(MGLQueryStateOwner *owner, uint32_t *active_out);

int mglRenderAcquireSampleQuerySlot(MGLQueryStateOwner *owner, uint32_t *mode_out, uint64_t *offset_out);

int mglRenderGetSampleQueryResult(MGLQueryStateOwner *owner, uint64_t *result_out);

int mglRenderBeginTimerQuery(MGLQueryStateOwner *owner);

int mglRenderEndTimerQuery(MGLQueryStateOwner *owner, uint64_t *elapsed_out);

void mglRenderDestroyQueryStateOwner(MGLQueryStateOwner **owner);

/* Validate and encode a complete compute plan, then (when copy-backs or CPU
 * visibility require a boundary) encode the copy-back blit and perform the
 * owner submit/wait transaction before synchronizing GL CPU prefixes. */
int mglRenderExecuteComputeExecutionPlan(MGLCommandBufferOwner *command_buffer_owner, MGLCommandBufferRecoveryOwner *recovery_owner, const MGLRenderComputeExecutionPlan *plan, const MGLRenderCopyBackEntry *copy_backs, uint32_t copy_back_count, uint32_t require_cpu_visibility, MGLRenderComputeExecutionResult *result, char *err, size_t errcap);

int mglRenderIsEmulatedMSColorTexture(uint32_t target, int32_t samples);

/* True when the GL target's Metal backing is the emulated multisample store
 * (texture2d_array sample planes, 8-plane stride per logical layer).  Unlike
 * mglRenderIsEmulatedMSColorTexture this also accepts GL_RENDERBUFFER: a
 * multisample renderbuffer's backing texture carries the same plane layout,
 * but its GL target stays GL_RENDERBUFFER. */
int mglRenderIsEmulatedMSPlaneTarget(uint32_t target, int32_t samples);

int mglRenderPlanDirtyDomains(uint32_t dirty_bits, int draw_command,
                              int has_pipeline, int fbo_binding_dirty,
                              MGLDirtyDomainPlan *out);

int mglRenderGetFboMatchCache(MGLRenderPassIdentityOwner *owner, MGLRenderFboMatchCacheState *cache_out);

int mglRenderGetRenderTargetSizeOwner(MGLRenderPassStateOwner *owner, uint64_t *width_out, uint64_t *height_out);

#ifdef __cplusplus
}
#endif

#endif
