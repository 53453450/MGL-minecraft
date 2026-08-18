/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

//------------------------------------------------------------------------------------------------
// AIR metallib -> MTL::Library -> PSO implementation using metal-cpp.
//
// This TU does not define the metal-cpp implementation macros. They belong to
// mgl_render.cpp; this file only consumes the shared declarations.
//
// This loader creates render and compute PSOs, maintains the render-pipeline
// cache, and handles optional binary-archive lookups.
//------------------------------------------------------------------------------------------------
#include "mgl_metal.h"
#include "mgl_air_loader.h"
#include "mgl_env_flag.h"

#include <dispatch/dispatch.h>
#include <map>
#include <string>

namespace {

// The cache is process-lifetime storage. Explicit shutdown releases its Metal
// objects before clearing it, avoiding static-destruction ordering hazards.
using PSOCache = std::map<std::string, void*>;

PSOCache& psoCache() {
    static PSOCache* cache = new PSOCache();
    return *cache;
}

std::string pipelineKey(const void* vs, const void* fs,
                        const MGLRenderPipelineDescriptorState* d) {
    std::string key;
    key.reserve(sizeof(vs) + sizeof(fs) + sizeof(*d));
    key.append(reinterpret_cast<const char*>(&vs), sizeof(vs));
    key.append(reinterpret_cast<const char*>(&fs), sizeof(fs));
    key.append(reinterpret_cast<const char*>(d), sizeof(*d));
    return key;
}

void copyError(NS::Error* e, char* err, size_t errcap) {
    if (!err || errcap == 0) return;
    if (e && e->localizedDescription()) {
        const char* s = e->localizedDescription()->utf8String();
        if (s) {
            snprintf(err, errcap, "%s", s);
            return;
        }
    }
    snprintf(err, errcap, "unknown Metal error");
}

// MTL::PixelFormat packed depth-stencil predicate.
bool isPackedDepthStencil(uint32_t format) {
    return format == static_cast<uint32_t>(MTL::PixelFormatDepth24Unorm_Stencil8) ||
           format == static_cast<uint32_t>(MTL::PixelFormatDepth32Float_Stencil8);
}

// Normalizes depth and stencil formats when one attachment uses a packed
// format. Metal requires both attachments to reference the shared format.
void normalizeDepthStencilFormats(MGLRenderPipelineDescriptorState* desc) {
    uint32_t depth = desc->depth_format;
    uint32_t stencil = desc->stencil_format;
    if (depth == static_cast<uint32_t>(MTL::PixelFormatInvalid) ||
        stencil == static_cast<uint32_t>(MTL::PixelFormatInvalid) ||
        depth == stencil) {
        return;
    }
    const bool depthPacked = isPackedDepthStencil(depth);
    const bool stencilPacked = isPackedDepthStencil(stencil);
    if (!depthPacked && !stencilPacked) {
        return;
    }
    const uint32_t packed = stencilPacked ? stencil : depth;
    desc->depth_format = packed;
    desc->stencil_format = packed;
}

// Builds a render-pipeline descriptor from value-state. Valid color
// attachments receive their write-mask and blend state; indirect command
// buffers are enabled only when explicitly requested by the caller.
MTL::RenderPipelineDescriptor* buildRenderPipelineDescriptor(
    const MGLRenderPipelineDescriptorState* desc) {
    MTL::RenderPipelineDescriptor* rpd =
        MTL::RenderPipelineDescriptor::alloc()->init();
    if (!rpd) {
        return nullptr;
    }
    rpd->setLabel(
        NS::String::string("GLSL Pipeline", NS::UTF8StringEncoding));

    rpd->setRasterizationEnabled(desc->rasterization_enabled ? true : false);
    if (mgl_env_flag_enabled("MGL_ENABLE_ICB_PIPELINES")) {
        rpd->setSupportIndirectCommandBuffers(true);
    }
    rpd->setAlphaToCoverageEnabled(desc->alpha_to_coverage_enabled ? true : false);
    rpd->setAlphaToOneEnabled(desc->alpha_to_one_enabled ? true : false);
    rpd->setInputPrimitiveTopology(
        (MTL::PrimitiveTopologyClass)desc->input_primitive_topology);
    if (desc->raster_sample_count > 0) {
        rpd->setRasterSampleCount(desc->raster_sample_count);
    }

    for (uint32_t i = 0; i < desc->color_count && i < 8; i++) {
        MTL::RenderPipelineColorAttachmentDescriptor* ca =
            rpd->colorAttachments()->object(i);
        ca->setPixelFormat((MTL::PixelFormat)desc->color_format[i]);
        /* Untouched (invalid-format) attachments keep Metal defaults —
         * writeMask All, blending off — exactly like the ObjC descriptor
         * that never touched them. */
        if (desc->color_format[i] !=
            static_cast<uint32_t>(MTL::PixelFormatInvalid)) {
            ca->setWriteMask((MTL::ColorWriteMask)desc->color_write_mask[i]);
            if (desc->blending_enabled_mask & (1u << i)) {
                ca->setBlendingEnabled(true);
                ca->setSourceRGBBlendFactor(
                    (MTL::BlendFactor)desc->source_rgb_blend_factor[i]);
                ca->setDestinationRGBBlendFactor(
                    (MTL::BlendFactor)desc->destination_rgb_blend_factor[i]);
                ca->setSourceAlphaBlendFactor(
                    (MTL::BlendFactor)desc->source_alpha_blend_factor[i]);
                ca->setDestinationAlphaBlendFactor(
                    (MTL::BlendFactor)desc->destination_alpha_blend_factor[i]);
                ca->setRgbBlendOperation(
                    (MTL::BlendOperation)desc->rgb_blend_operation[i]);
                ca->setAlphaBlendOperation(
                    (MTL::BlendOperation)desc->alpha_blend_operation[i]);
            }
        }
    }

    rpd->setDepthAttachmentPixelFormat((MTL::PixelFormat)desc->depth_format);
    rpd->setStencilAttachmentPixelFormat((MTL::PixelFormat)desc->stencil_format);

    if (desc->attrib_count > 0) {
        MTL::VertexDescriptor* vd = MTL::VertexDescriptor::alloc()->init();
        for (uint32_t i = 0; i < desc->attrib_count && i < 32; i++) {
            const uint32_t bufIdx = desc->attrib_buffer_index[i];
            vd->attributes()->object(i)->setFormat(
                (MTL::VertexFormat)desc->attrib_format[i]);
            vd->attributes()->object(i)->setOffset(desc->attrib_offset[i]);
            vd->attributes()->object(i)->setBufferIndex(bufIdx);
            /* Only valid attributes write a layout. Unused attributes must not
             * overwrite an existing stride or step rate with zero. */
            if (desc->attrib_format[i] !=
                static_cast<uint32_t>(MTL::VertexFormatInvalid)) {
                vd->layouts()->object(bufIdx)->setStride(desc->attrib_stride[i]);
                vd->layouts()->object(bufIdx)->setStepFunction(
                    (MTL::VertexStepFunction)desc->attrib_step_function[i]);
                vd->layouts()->object(bufIdx)->setStepRate(desc->attrib_step_rate[i]);
            }
        }
        rpd->setVertexDescriptor(vd);
        vd->release();
    }

    rpd->setTessellationPartitionMode(
        (MTL::TessellationPartitionMode)desc->tessellation_partition_mode);
    /* Metal defaults maxTessellationFactor to 64. A zero state means unset;
     * skip it to avoid passing an invalid value to Metal. */
    if (desc->max_tessellation_factor > 0) {
        rpd->setMaxTessellationFactor(desc->max_tessellation_factor);
    }
    rpd->setTessellationFactorScaleEnabled(
        desc->tessellation_factor_scale_enabled ? true : false);
    rpd->setTessellationFactorFormat(
        (MTL::TessellationFactorFormat)desc->tessellation_factor_format);
    rpd->setTessellationControlPointIndexType(
        (MTL::TessellationControlPointIndexType)
            desc->tessellation_control_point_index_type);
    rpd->setTessellationFactorStepFunction(
        (MTL::TessellationFactorStepFunction)
            desc->tessellation_factor_step_function);
    rpd->setTessellationOutputWindingOrder(
        (MTL::Winding)desc->tessellation_output_winding_order);

    return rpd;
}

// Shared PSO creation. Complete pipelines query the archive first and add a
// newly compiled state only on a miss, keeping archive updates incremental.
int createRenderPipelineInternal(
    MTL::Device* dev, MTL::Function* vsFn, MTL::Function* fsFn,
    const MGLRenderPipelineDescriptorState* desc, MTL::BinaryArchive* archive,
    void** pso_out, char* err, size_t errcap) {
    if (pso_out) *pso_out = nullptr;
    if (!dev || !vsFn || !desc || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }

    MGLRenderPipelineDescriptorState state = *desc;
    normalizeDepthStencilFormats(&state);

    std::string key = pipelineKey(vsFn, fsFn, &state);
    PSOCache& cache = psoCache();
    auto it = cache.find(key);
    const bool archiveEligible = archive && vsFn && fsFn;
    if (it != cache.end() && !archiveEligible) {
        static_cast<MTL::RenderPipelineState*>(it->second)->retain();
        *pso_out = it->second;
        return 0;
    }

    MTL::RenderPipelineDescriptor* rpd =
        buildRenderPipelineDescriptor(&state);
    if (!rpd) {
        if (err && errcap) snprintf(err, errcap, "descriptor alloc failed");
        return -1;
    }
    rpd->setVertexFunction(vsFn);
    if (fsFn) {
        rpd->setFragmentFunction(fsFn);
    }
    /* Metal accepts incomplete render pipelines used by capture/discard
     * paths, but MTLBinaryArchive rejects either missing stage when it later
     * serializes.  Match the ObjC archive gate and keep both vertex-only and
     * fragment-only PSOs out of the archive. */
    if (archiveEligible) {
        rpd->setBinaryArchives(NS::Array::array(archive));
    }

    NS::Error* nsErr = nullptr;
    MTL::RenderPipelineState* pso = nullptr;
    if (archiveEligible) {
        pso = dev->newRenderPipelineState(
            rpd, MTL::PipelineOptionFailOnBinaryArchiveMiss,
            nullptr, &nsErr);
        if (pso) {
            if (it != cache.end()) {
                pso->release();
                static_cast<MTL::RenderPipelineState*>(it->second)->retain();
                *pso_out = it->second;
                rpd->release();
                return 0;
            }
        } else if (it != cache.end()) {
            /* The PSO already exists in this process, so only teach the
             * persistent archive about the miss; recompiling the same PSO is
             * unnecessary. */
            NS::Error* addErr = nullptr;
            if (!archive->addRenderPipelineFunctions(rpd, &addErr)) {
                char addMessage[512] = {0};
                copyError(addErr, addMessage, sizeof(addMessage));
                fprintf(stderr,
                        "MGL BINARY ARCHIVE: addRenderPipeline warning: %s\n",
                        addMessage[0] ? addMessage : "unknown error");
            }
            static_cast<MTL::RenderPipelineState*>(it->second)->retain();
            *pso_out = it->second;
            rpd->release();
            return 0;
        }
    }
    const bool archiveMiss = archiveEligible && !pso;
    if (!pso) {
        nsErr = nullptr;
        pso = dev->newRenderPipelineState(rpd, &nsErr);
    }
    if (!pso) {
        copyError(nsErr, err, errcap);
        rpd->release();
        return -1;
    }
    if (archiveMiss) {
        NS::Error* addErr = nullptr;
        if (!archive->addRenderPipelineFunctions(rpd, &addErr)) {
            char addMessage[512] = {0};
            copyError(addErr, addMessage, sizeof(addMessage));
            fprintf(stderr,
                    "MGL BINARY ARCHIVE: addRenderPipeline warning: %s\n",
                    addMessage[0] ? addMessage : "unknown error");
        }
    }
    rpd->release();

    pso->retain(); // The cache holds a long-lived reference.
    cache[key] = pso;
    *pso_out = pso; // The caller owns a reference released with mglAirRelease.
    return 0;
}

} // namespace

extern "C" {

int mglAirLoadLibrary(const void* device, const unsigned char* bytes, size_t size,
                      void** library_out, char* err, size_t errcap) {
    if (!device || !bytes || size == 0 || !library_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    *library_out = nullptr;
    MTL::Device* dev = static_cast<MTL::Device*>(const_cast<void*>(device));

    // Match the Objective-C path: dispatch_data_create -> newLibrary.
    dispatch_data_t data = dispatch_data_create(bytes, size, nullptr,
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (!data) {
        if (err && errcap) snprintf(err, errcap, "dispatch_data_create failed");
        return -1;
    }
    NS::Error* nsErr = nullptr;
    MTL::Library* lib = dev->newLibrary(data, &nsErr);
    dispatch_release(data);
    if (!lib) {
        copyError(nsErr, err, errcap);
        return -1;
    }
    *library_out = lib; // Owned by the caller.
    return 0;
}

int mglAirCreateRenderPipeline(const void* device, void* vs_function, void* fs_function,
                               const MGLPipelineDescriptorState* desc, void** pso_out,
                               char* err, size_t errcap) {
    if (!device || !vs_function || !desc || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    return createRenderPipelineInternal(
        static_cast<MTL::Device*>(const_cast<void*>(device)),
        static_cast<MTL::Function*>(vs_function),
        fs_function ? static_cast<MTL::Function*>(fs_function) : nullptr,
        desc, nullptr /* archive */, pso_out, err, errcap);
}

int mglAirCreateRenderPipelineWithArchive(
    const void* device, void* vs_function, void* fs_function,
    const MGLRenderPipelineDescriptorState* desc, void* binary_archive,
    void** pso_out, char* err, size_t errcap) {
    if (!device || !vs_function || !desc || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    return createRenderPipelineInternal(
        static_cast<MTL::Device*>(const_cast<void*>(device)),
        static_cast<MTL::Function*>(vs_function),
        fs_function ? static_cast<MTL::Function*>(fs_function) : nullptr,
        desc, static_cast<MTL::BinaryArchive*>(binary_archive),
        pso_out, err, errcap);
}

int mglAirCreateComputePipeline(const void* device, void* library,
                                void** pso_out, char* err, size_t errcap) {
    if (!device || !library || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    *pso_out = nullptr;
    MTL::Device* dev = static_cast<MTL::Device*>(const_cast<void*>(device));
    MTL::Library* lib = static_cast<MTL::Library*>(library);
    MTL::Function* fn = lib->newFunction(NS::String::string("main", NS::UTF8StringEncoding));
    if (!fn) {
        if (err && errcap) snprintf(err, errcap, "compute function 'main' not found");
        return -1;
    }
    NS::Error* nsErr = nullptr;
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &nsErr);
    fn->release();
    if (!pso) {
        copyError(nsErr, err, errcap);
        return -1;
    }
    *pso_out = pso; // Owned by the caller; compute PSOs are not cached here.
    return 0;
}

void mglAirRelease(void* obj) {
    if (obj) {
        static_cast<NS::Object*>(obj)->release();
    }
}

void mglAirLoaderShutdown(void) {
    PSOCache& cache = psoCache();
    for (auto &entry : cache) {
        if (entry.second) {
            static_cast<MTL::RenderPipelineState*>(entry.second)->release();
        }
    }
    cache.clear();
}

} // extern "C"
