/* SPDX-License-Identifier: LGPL-3.0-only */
#include "mgl_metal.h"
#include "mgl_render.h"
#include "mgl_render_pixel.h"
#include "mgl_byte_hash.h"
#include "mgl_vertex_format.h"
extern "C" {
#include "pixel_utils.h"
}
#include "mgl_program_resource.h"
#include "mgl_renderer_backend.h"
#include "mgl_air_loader.h"
#include "mgl_air_tess_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_program_reflection.h"
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"
#include "glm_context.h"
#include "mgl_render_internal.h"
#include "mgl_capability.h"
#include "mgl_sync.h"
#include "glm_limits.h"
#include "mgl_shader_abi.h"
#include "mgl_buffer_slots.h"
#include "mgl_buffer_plan.h"
#include "mgl_tess_domain.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <chrono>
#include <tuple>
#include <utility>
#include <vector>

#include <mach/mach.h>
#include <Block.h>
#include <objc/runtime.h>

void mglRenderBindBuffer(GLMContext glm_ctx, Buffer* buffer) {
    char error[256] = {};
    int result = mglRenderBindBufferStorage(
        buffer, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_BOUND) return;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp buffer bind failed buffer=%u: %s\n",
            buffer ? (unsigned)buffer->name : 0u,
            error[0] ? error : "unknown error");
}

int mglRenderFillVertexConversionFromAttribKind(
    int attrib_kind, uint32_t size, uint32_t type, int normalized,
    int dst_signed, MGLRenderVertexConversion *out) {
    if (!out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    switch (attrib_kind) {
        case MGL_ATTRIB_CONV_DOUBLE:
            if (size == 0u || size > 4u) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT;
            out->component_count = size;
            out->source_type = GL_DOUBLE;
            break;
        case MGL_ATTRIB_CONV_INT_TO_FLOAT:
            if (size == 0u || size > 4u ||
                (type != GL_INT && type != GL_UNSIGNED_INT)) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_INT_TO_FLOAT;
            out->component_count = size;
            out->source_type = type;
            out->normalized = normalized ? 1u : 0u;
            break;
        case MGL_ATTRIB_CONV_FIXED:
            if (size == 0u || size > 4u) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_FIXED_TO_FLOAT;
            out->component_count = size;
            out->source_type = GL_FIXED;
            break;
        case MGL_ATTRIB_CONV_UINT_1010102:
            out->kind = MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT;
            out->component_count = 4u;
            out->source_type = GL_UNSIGNED_INT_10_10_10_2;
            out->normalized = 1u;
            break;
        case MGL_ATTRIB_CONV_UINT_10F11F11F:
            out->kind = MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT;
            out->component_count = 3u;
            out->source_type = GL_UNSIGNED_INT_10F_11F_11F_REV;
            break;
        case MGL_ATTRIB_CONV_INTEGER_SIGN:
            if (size == 0u || size > 4u) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_INTEGER_TO_32;
            out->component_count = size;
            out->source_type = type;
            out->destination_signed = dst_signed ? 1u : 0u;
            break;
        default:
            return -1;
    }
    return 0;
}

void mglRenderBindProgram(GLMContext glm_ctx, Program* program) {
    (void)glm_ctx;
    int failedStage = -1;
    char error[256] = {};
    int result = mglRenderBindAIRProgram(
        program, &failedStage, error, sizeof(error));
    if (result == MGL_RENDER_AIR_PROGRAM_BOUND) return;
    fprintf(stderr,
            "MGL ERROR: Metal-cpp program bind failed program=%u "
            "stage=%d: %s\n",
            program ? (unsigned)program->name : 0u, failedStage,
            error[0]
                ? error
                : (result == MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE
                       ? "linked program has no AIR metallib"
                       : "unknown error"));
}

extern "C"
int mglRenderEncodeStageBindingCopyBacks(
    const MGLRenderCopyBackEntry* entries, uint32_t count,
    void* blit_encoder) {
    if (!entries && count) return -1;
    for (uint32_t i = 0; i < count; i++) {
        const MGLRenderCopyBackEntry& entry = entries[i];
        if (entry.length == 0) continue;
        MTL::Buffer* temporary =
            static_cast<MTL::Buffer*>(const_cast<void*>(entry.temporary));
        MTL::Buffer* destination =
            static_cast<MTL::Buffer*>(const_cast<void*>(entry.destination));
        if (!temporary || !destination ||
            entry.length > temporary->length() ||
            entry.destination_offset > destination->length() ||
            entry.length >
                destination->length() - entry.destination_offset) {
            return -1;
        }
        if (blit_encoder &&
            mglRenderBlitCopyBuffer(
                blit_encoder, const_cast<void*>(entry.temporary), 0,
                const_cast<void*>(entry.destination),
                entry.destination_offset, entry.length) != 0) {
            return -1;
        }
    }
    return 0;
}



uint32_t mglRenderAttribFormatOrFallback(uint32_t planned, uint32_t type,
                                         uint32_t size, int normalized) {
    return planned != 0u
               ? planned
               : mglRenderGLTypeSizeToVertexFormat(type, size, normalized);
}

int mglRenderAttribNeedsConversion(int long_attr, uint32_t type, int integer) {
    if (long_attr || type == GL_DOUBLE) {
        return 1;
    }
    if (!integer && (type == GL_INT || type == GL_UNSIGNED_INT)) {
        return 1;
    }
    return 0;
}

int mglRenderAttribNeedsConvertedMetalStream(uint32_t type, int integer) {
    if (mglRenderAttribNeedsConversion(0, type, integer)) {
        return 1;
    }
    return type == GL_FIXED || type == GL_UNSIGNED_INT_10_10_10_2 ||
                   type == GL_UNSIGNED_INT_10F_11F_11F_REV
               ? 1
               : 0;
}

int mglRenderAttribColorUByteNeedsNormalize(uint32_t type, uint32_t size,
                                            int already_norm) {
    return !already_norm && type == GL_UNSIGNED_BYTE && size == 4u ? 1 : 0;
}

uint32_t mglRenderAttribEffectiveNormalized(uint32_t already, int needs) {
    return needs ? GL_TRUE : already;
}

double mglRenderDecodeVertexAttribComponent(const uint8_t *src, uint32_t type,
                                            int normalized, uint32_t component) {
    if (!src) {
        return 0.0;
    }
    switch (type) {
    case GL_FLOAT: {
        float v = 0.0f;
        memcpy(&v, src + component * sizeof(float), sizeof(v));
        return (double)v;
    }
    case GL_UNSIGNED_BYTE: {
        uint8_t v = 0;
        memcpy(&v, src + component, sizeof(v));
        return normalized ? ((double)v / 255.0) : (double)v;
    }
    case GL_BYTE: {
        int8_t v = 0;
        memcpy(&v, src + component, sizeof(v));
        if (normalized) {
            double d = (double)v / 127.0;
            return d < -1.0 ? -1.0 : d;
        }
        return (double)v;
    }
    case GL_UNSIGNED_SHORT: {
        uint16_t v = 0;
        memcpy(&v, src + component * sizeof(uint16_t), sizeof(v));
        return normalized ? ((double)v / 65535.0) : (double)v;
    }
    case GL_SHORT: {
        int16_t v = 0;
        memcpy(&v, src + component * sizeof(int16_t), sizeof(v));
        if (normalized) {
            double d = (double)v / 32767.0;
            return d < -1.0 ? -1.0 : d;
        }
        return (double)v;
    }
    case GL_UNSIGNED_INT: {
        uint32_t v = 0;
        memcpy(&v, src + component * sizeof(uint32_t), sizeof(v));
        return normalized ? ((double)v / 4294967295.0) : (double)v;
    }
    case GL_INT: {
        int32_t v = 0;
        memcpy(&v, src + component * sizeof(int32_t), sizeof(v));
        if (normalized) {
            double d = (double)v / 2147483647.0;
            return d < -1.0 ? -1.0 : d;
        }
        return (double)v;
    }
    default:
        return 0.0;
    }
}

int mglRenderBufferMapIsBaseBinding(uint32_t attribute_mask) {
    return attribute_mask == 0u ? 1 : 0;
}

int mglRenderClientBindingInRange(uint32_t binding, uint32_t max) {
    return binding < max ? 1 : 0;
}

int mglRenderBufferBindingEmpty(int has_buf, uint32_t name) {
    return !has_buf && name == 0u ? 1 : 0;
}

int mglRenderBaseBindingTooSmall(int64_t range, uint64_t reflected) {
    return reflected > 0u && range > 0 && (uint64_t)range < reflected ? 1 : 0;
}

int mglRenderAttribOffsetsValid(int64_t binding_offset,
                                int64_t relativeoffset) {
    return binding_offset >= 0 && relativeoffset >= 0 ? 1 : 0;
}

uint32_t mglRenderBuildCurrentVertexAttribBytes(
    uint32_t type, uint32_t size, const int32_t current_i[4],
    const uint32_t current_u[4], const float current_f[4], uint8_t bytes[16]) {
    if (!bytes || !current_i || !current_u || !current_f) {
        return 0u;
    }
    memset(bytes, 0, 16);
    if (size == 0u || size > 4u) {
        size = 4u;
    }
    switch (type) {
        case GL_BYTE:
        case GL_SHORT:
        case GL_INT: {
            const size_t component_bytes =
                type == GL_BYTE ? sizeof(int8_t)
                                : type == GL_SHORT ? sizeof(int16_t)
                                                   : sizeof(int32_t);
            if (component_bytes == 0u || component_bytes * size > 16u) {
                return 0u;
            }
            for (uint32_t i = 0u; i < size; i++) {
                const int32_t value = current_i[i];
                if (type == GL_BYTE) {
                    const int8_t packed = (int8_t)value;
                    memcpy(bytes + i * component_bytes, &packed,
                           component_bytes);
                } else if (type == GL_SHORT) {
                    const int16_t packed = (int16_t)value;
                    memcpy(bytes + i * component_bytes, &packed,
                           component_bytes);
                } else {
                    const int32_t packed = value;
                    memcpy(bytes + i * component_bytes, &packed,
                           component_bytes);
                }
            }
            return 16u;
        }
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_UNSIGNED_INT: {
            const size_t component_bytes =
                type == GL_UNSIGNED_BYTE
                    ? sizeof(uint8_t)
                    : type == GL_UNSIGNED_SHORT ? sizeof(uint16_t)
                                                : sizeof(uint32_t);
            if (component_bytes == 0u || component_bytes * size > 16u) {
                return 0u;
            }
            for (uint32_t i = 0u; i < size; i++) {
                const uint32_t value = current_u[i];
                if (type == GL_UNSIGNED_BYTE) {
                    const uint8_t packed = (uint8_t)value;
                    memcpy(bytes + i * component_bytes, &packed,
                           component_bytes);
                } else if (type == GL_UNSIGNED_SHORT) {
                    const uint16_t packed = (uint16_t)value;
                    memcpy(bytes + i * component_bytes, &packed,
                           component_bytes);
                } else {
                    const uint32_t packed = value;
                    memcpy(bytes + i * component_bytes, &packed,
                           component_bytes);
                }
            }
            return 16u;
        }
        case GL_DOUBLE:
        case GL_FLOAT:
        default: {
            const float packed[4] = {current_f[0], current_f[1], current_f[2],
                                     current_f[3]};
            memcpy(bytes, packed, sizeof(packed));
            return (uint32_t)sizeof(packed);
        }
    }
}



extern "C" uint32_t mglRenderPlanVertexAttribOffset(
    int uses_current, int needs_conversion, int absolute_offsets,
    uint32_t attrib_index, uint32_t pool_stride, uint32_t relativeoffset,
    uint32_t binding_offset) {
    if (uses_current) {
        return attrib_index * pool_stride;
    }
    if (needs_conversion || absolute_offsets) {
        return relativeoffset;
    }
    return binding_offset + relativeoffset;
}

extern "C"
int mglRenderResolveVertexAttribBinding(
    uint32_t binding_index, int binding_has_buffer,
    int64_t binding_offset, uint32_t binding_stride,
    int64_t attrib_binding_offset, uint32_t attrib_stride,
    uint32_t binding_divisor, uint32_t attrib_divisor,
    MGLRenderVertexAttribResolve* out) {
    if (!out) return -1;
    if (binding_index < MGL_MAX_VERTEX_ATTRIB_BINDINGS &&
        binding_has_buffer) {
        out->use_binding_table = 1;
        out->binding_offset = binding_offset;
        out->stride = (binding_stride > 0) ? binding_stride : attrib_stride;
        out->divisor = binding_divisor;
    } else {
        out->use_binding_table = 0;
        out->binding_offset = attrib_binding_offset;
        out->stride = attrib_stride;
        out->divisor = attrib_divisor;
    }
    return 0;
}

uint32_t mglRenderComputePipelineMaxTotalThreads(void *pipeline) {
    if (!pipeline) return 0;
    return static_cast<uint32_t>(
        static_cast<MTL::ComputePipelineState *>(pipeline)
            ->maxTotalThreadsPerThreadgroup());
}

int mglRenderSetComputePipelineState(void* compute_encoder,
                                        void* pipeline_state) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    MTL::ComputePipelineState* pipeline =
        static_cast<MTL::ComputePipelineState*>(pipeline_state);
    if (!encoder || !pipeline) return -1;
    encoder->setComputePipelineState(pipeline);
    return 0;
}

int mglRenderEncodeComputeBindingSnapshot(
    void* compute_encoder,
    const MGLRenderComputeBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!compute_encoder || !snapshot) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (snapshot->op_count >
        MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {
        if (err && errcap) snprintf(err, errcap, "snapshot count overflow");
        return -1;
    }
    for (uint32_t i = 0; i < snapshot->op_count; i++) {
        const MGLRenderComputeBindingOp* op = &snapshot->ops[i];
        if (op->kind == 0) {
            /* kind 0: set buffer; NULL buffer clears the slot. */
            encoder->setBuffer(static_cast<MTL::Buffer*>(op->buffer),
                               static_cast<NS::UInteger>(op->offset),
                               op->index);
        } else if (op->kind == 1) {
            if (!op->bytes) {
                if (err && errcap) {
                    snprintf(err, errcap, "null compute bytes op %u", i);
                }
                return -1;
            }
            encoder->setBytes(op->bytes, op->length, op->index);
        } else if (op->kind == 2) {
            /* kind 2: set texture; NULL clears the slot. */
            encoder->setTexture(static_cast<MTL::Texture*>(op->buffer),
                                op->index);
        } else if (op->kind == 3) {
            /* kind 3: set sampler state; NULL clears the slot. */
            encoder->setSamplerState(
                static_cast<MTL::SamplerState*>(op->buffer), op->index);
        } else {
            if (err && errcap) {
                snprintf(err, errcap, "bad compute op kind %u", op->kind);
            }
            return -1;
        }
    }
    return 0;
}

int mglRenderAppendComputeBindingSnapshotToPlan(
    MGLRenderComputeExecutionPlan* plan,
    const MGLRenderComputeBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!plan || !snapshot) {
        if (err && errcap) snprintf(err, errcap, "bad compute plan args");
        return -1;
    }
    if (snapshot->op_count > MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS ||
        plan->binding_op_count > MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS ||
        snapshot->op_count >
            MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS -
                plan->binding_op_count) {
        if (err && errcap) snprintf(err, errcap, "compute execution op overflow");
        return -1;
    }
    for (uint32_t i = 0; i < snapshot->op_count; i++) {
        const MGLRenderComputeBindingOp* op = &snapshot->ops[i];
        if (op->kind > 3u || (op->kind == 1u && !op->bytes)) {
            if (err && errcap) {
                snprintf(err, errcap, "invalid compute binding op %u", i);
            }
            return -1;
        }
        plan->binding_ops[plan->binding_op_count++] = *op;
    }
    return 0;
}

extern "C" bool mglRenderIsCullDistanceAttribName(const char* name) {
    return name && std::strncmp(name, "culldistance_data", 17) == 0;
}

extern "C" const char* mglRenderVertexAttribName(const Program* program,
                                                 uint32_t attrib) {
    if (!program || attrib >= MAX_ATTRIBS) {
        return NULL;
    }
    const MGLShaderResourceList* vsInputs =
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
    if (vsInputs && vsInputs->list) {
        for (GLuint r = 0; r < vsInputs->count; r++) {
            const MGLShaderResource* res = &vsInputs->list[r];
            if (res->location == attrib) {
                return res->name ? res->name
                                 : program->attrib_location_names[attrib];
            }
            const GLuint span =
                mglAIRVaryingLocationSpan(res->gl_type, res->gl_array_size);
            if (span > 1u && attrib >= res->location &&
                attrib < res->location + span) {
                return res->name ? res->name
                                 : program->attrib_location_names[attrib];
            }
        }
    }
    return program->attrib_location_names[attrib];
}

extern "C" void mglRenderAccumulateCullDistanceAttrib(
    MGLRenderCullDistanceLayout* layout, void* mtl_buffer,
    int64_t binding_offset, uint32_t stride, int64_t relativeoffset) {
    if (!layout || !mtl_buffer) {
        return;
    }
    if (layout->culldist_size == 0u) {
        layout->mtl_buffer = mtl_buffer;
        layout->binding_offset = binding_offset;
        layout->stride = stride;
        layout->first_relative_offset = relativeoffset;
    }
    layout->culldist_size++;
}

extern "C" void mglRenderBindCullDistanceEmuSlots(MGLRenderEncoderOwner * encoder_owner, void* vertex_buffer, const MGLCullDistanceEmuParams* params) {
    if (!encoder_owner || !params) {
        return;
    }
    (void)mglRenderSetRenderBufferForOwner(
        encoder_owner, vertex_buffer, 0u, MGL_RENDER_BINDING_STAGE_VERTEX,
        kMGLCullDistanceVertexBufferIndex);
    (void)mglRenderSetRenderBytesForOwner(
        encoder_owner, params, sizeof(*params),
        MGL_RENDER_BINDING_STAGE_VERTEX, kMGLCullDistanceParamsBufferIndex);
}

int mglRenderEncodeBindingSnapshot(
    void* render_encoder,
    const MGLRenderBindingSnapshot* snapshot,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!render_encoder || !snapshot) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (snapshot->vertex_op_count >
            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS ||
        snapshot->fragment_op_count >
            MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS) {
        if (err && errcap) snprintf(err, errcap, "snapshot count overflow");
        return -1;
    }
    for (uint32_t i = 0; i < snapshot->vertex_op_count; i++) {
        const MGLRenderBindingOp* op = &snapshot->vertex_ops[i];
        if (op->kind == 0) {
            /* kind 0: set buffer; NULL buffer clears the slot (the ObjC
             * skip paths emit nil clears through the same op). */
            encoder->setVertexBuffer(
                static_cast<MTL::Buffer*>(op->buffer),
                static_cast<NS::UInteger>(op->offset), op->index);
        } else if (op->kind == 1) {
            if (!op->bytes) {
                if (err && errcap) {
                    snprintf(err, errcap, "null vertex bytes op %u", i);
                }
                return -1;
            }
            encoder->setVertexBytes(op->bytes, op->length, op->index);
        } else {
            if (err && errcap) {
                snprintf(err, errcap, "bad vertex op kind %u", op->kind);
            }
            return -1;
        }
    }
    for (uint32_t i = 0; i < snapshot->fragment_op_count; i++) {
        const MGLRenderBindingOp* op = &snapshot->fragment_ops[i];
        if (op->kind == 0) {
            encoder->setFragmentBuffer(
                static_cast<MTL::Buffer*>(op->buffer),
                static_cast<NS::UInteger>(op->offset), op->index);
        } else if (op->kind == 1) {
            if (!op->bytes) {
                if (err && errcap) {
                    snprintf(err, errcap, "null fragment bytes op %u", i);
                }
                return -1;
            }
            encoder->setFragmentBytes(op->bytes, op->length, op->index);
        } else {
            if (err && errcap) {
                snprintf(err, errcap, "bad fragment op kind %u", op->kind);
            }
            return -1;
        }
    }
    return 0;
}

int mglRenderEncodeResourceBindingSnapshot(MGLBindingState * binding_state, void* render_encoder, const MGLRenderResourceBindingSnapshot* snapshot, char* err, size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!binding_state || !render_encoder || !snapshot) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    if (snapshot->vertex_op_count >
            MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS ||
        snapshot->fragment_op_count >
            MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS) {
        if (err && errcap) snprintf(err, errcap, "snapshot count overflow");
        return -1;
    }

    const auto encodeStage = [&](const MGLRenderResourceBindingOp* ops,
                                 uint32_t count,
                                 uint32_t stage) -> int {
        for (uint32_t i = 0; i < count; ++i) {
            const MGLRenderResourceBindingOp& op = ops[i];
            int result = -1;
            if (op.kind == MGL_RENDER_RESOURCE_BINDING_TEXTURE) {
                result = mglRenderBindingSetTexture(
                    binding_state, render_encoder, op.resource, stage,
                    op.index);
            } else if (op.kind == MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
                result = mglRenderBindingSetSampler(
                    binding_state, render_encoder, op.resource, stage,
                    op.index);
            } else {
                if (err && errcap) {
                    snprintf(err, errcap, "bad resource op kind %u at %u",
                             op.kind, i);
                }
                return -1;
            }
            if (result < 0) {
                if (err && errcap) {
                    snprintf(err, errcap,
                             "resource op failed stage=%u kind=%u index=%u",
                             stage, op.kind, op.index);
                }
                return -1;
            }
        }
        return 0;
    };

    if (encodeStage(snapshot->vertex_ops, snapshot->vertex_op_count,
                    MGL_RENDER_BINDING_STAGE_VERTEX) != 0) {
        return -1;
    }
    return encodeStage(snapshot->fragment_ops, snapshot->fragment_op_count,
                       MGL_RENDER_BINDING_STAGE_FRAGMENT);
}

int mglRenderSetRenderPipelineState(void* render_encoder,
                                       void* pipeline_state) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::RenderPipelineState* pipeline =
        static_cast<MTL::RenderPipelineState*>(pipeline_state);
    if (!encoder || !pipeline) return -1;
    encoder->setRenderPipelineState(pipeline);
    return 0;
}

int mglRenderBindBufferStorage(Buffer* buffer,
                                  char* err,
                                  size_t errcap) {
    constexpr size_t kMaxSafeBufferSize =
        static_cast<size_t>(2) * 1024u * 1024u * 1024u;
    if (err && errcap) err[0] = '\0';
    if (!buffer) {
        if (err && errcap) snprintf(err, errcap, "null buffer");
        return MGL_RENDER_BUFFER_ERROR;
    }

    if (buffer->size <= 0 ||
        static_cast<size_t>(buffer->size) > kMaxSafeBufferSize) {
        if (err && errcap) {
            snprintf(err, errcap, "suspicious size=%zu",
                     static_cast<size_t>(buffer->size));
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_BUFFER_ERROR;
    }

    uint64_t options = static_cast<uint64_t>(MTL::ResourceStorageModeShared);
    if ((buffer->storage_flags & GL_MAP_READ_BIT) == 0) {
        options |= static_cast<uint64_t>(
            MTL::ResourceCPUCacheModeWriteCombined);
    }

    size_t allocationSize = static_cast<size_t>(buffer->size);
    const void* bytes = nullptr;
    if (buffer->data.buffer_data != 0) {
        allocationSize = buffer->data.buffer_size;
        if (allocationSize == 0 || allocationSize > kMaxSafeBufferSize) {
            allocationSize = static_cast<size_t>(buffer->size);
        }
        bytes = reinterpret_cast<const void*>(
            static_cast<uintptr_t>(buffer->data.buffer_data));
    }
    if (buffer->transient_batch_buffer && !bytes) {
        if (err && errcap) {
            snprintf(err, errcap, "transient buffer has no CPU backing");
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_BUFFER_ERROR;
    }
    if (buffer->transient_batch_buffer) {
        allocationSize = static_cast<size_t>(buffer->size);
    }

    const bool clientStorage =
        (buffer->storage_flags & GL_CLIENT_STORAGE_BIT) != 0;
    const bool persistentNoCopy =
        bytes &&
        (buffer->immutable_storage & BUFFER_IMMUTABLE_STORAGE_FLAG) != 0 &&
        (buffer->storage_flags & GL_MAP_PERSISTENT_BIT) != 0;
    const bool noCopy = clientStorage || persistentNoCopy;
    if (noCopy && !bytes) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "no-copy buffer has no CPU backing buffer=%u",
                     static_cast<unsigned>(buffer->name));
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_BUFFER_ERROR;
    }
    if (clientStorage) {
        allocationSize = static_cast<size_t>(buffer->size);
    }

    void* metalBuffer = nullptr;
    mglMetalCountCreate(mgl::kMetalKindBuffer);
    int result = noCopy
        ? mglRenderCreateBufferWithBytesNoCopy(
              bytes, allocationSize, options, nullptr, 1, &metalBuffer)
        : (bytes
            ? mglRenderCreateBufferWithBytes(
                  bytes, allocationSize, options, nullptr, &metalBuffer)
            : mglRenderCreateBuffer(
                  allocationSize, options, nullptr, &metalBuffer));
    if (result != 0 || !metalBuffer) {
        if (err && errcap) {
            snprintf(err, errcap, "Metal buffer creation failed size=%zu",
                     allocationSize);
        }
        buffer->data.mtl_data = nullptr;
        return MGL_RENDER_BUFFER_ERROR;
    }

    buffer->data.mtl_data = metalBuffer;
    buffer->data.mtl_owns_buffer_data = noCopy ? GL_TRUE : GL_FALSE;
    if (!bytes) buffer->data.buffer_data = 0;
    return MGL_RENDER_BUFFER_BOUND;
}

int mglRenderBindAIRProgram(Program* program,
                               int* failed_stage_out,
                               char* err,
                               size_t errcap) {
    if (failed_stage_out) *failed_stage_out = -1;
    if (err && errcap) err[0] = '\0';
    MTL::Device* device = mgl::renderer().device;
    if (!program || !device) {
        if (err && errcap) snprintf(err, errcap, "renderer is not initialized");
        return MGL_RENDER_AIR_PROGRAM_ERROR;
    }
    program->dirty_bits &= ~DIRTY_PROGRAM;

    bool hasAIRStage = false;
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; ++stage) {
        Shader* shader = program->shader_slots[stage];
        if (!shader) continue;
        if (stage == _GEOMETRY_SHADER) {
            if (program->gs_route != MGL_GS_ROUTE_COMPUTE) {
                if (failed_stage_out) *failed_stage_out = stage;
                if (err && errcap) {
                    snprintf(err, errcap,
                             "unsupported geometry shader route %u",
                             (unsigned)program->gs_route);
                }
                return MGL_RENDER_AIR_PROGRAM_ERROR;
            }
        }
        MGLShaderModule* spirv = &program->modules[stage];
        if (!spirv->metallib_bytes || spirv->metallib_size == 0u) {
            return MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE;
        }
        hasAIRStage = true;
    }
    if (!hasAIRStage) return MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE;

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; ++stage) {
        Shader* shader = program->shader_slots[stage];
        if (!shader) {
            continue;
        }
        MGLShaderModule* spirv = &program->modules[stage];
        if (!spirv->mtl_library || !spirv->mtl_function) {
            mgl::releaseBridgedObject(&spirv->mtl_function);
            mgl::releaseBridgedObject(&spirv->mtl_library);
            if (mgl::loadAIRMainFunction(
                    device, spirv->metallib_bytes, spirv->metallib_size,
                    &spirv->mtl_library, &spirv->mtl_function,
                    err, errcap) != 0) {
                if (failed_stage_out) *failed_stage_out = stage;
                return MGL_RENDER_AIR_PROGRAM_ERROR;
            }
        }

        if (stage == _VERTEX_SHADER &&
            spirv->metallib_tess_capture_bytes &&
            (!spirv->mtl_tess_capture_library ||
             !spirv->mtl_tess_capture_function)) {
            mgl::releaseBridgedObject(&spirv->mtl_tess_capture_function);
            mgl::releaseBridgedObject(&spirv->mtl_tess_capture_library);
            if (mgl::loadAIRMainFunction(
                    device, spirv->metallib_tess_capture_bytes,
                    spirv->metallib_tess_capture_size,
                    &spirv->mtl_tess_capture_library,
                    &spirv->mtl_tess_capture_function,
                    err, errcap) != 0) {
                if (failed_stage_out) *failed_stage_out = stage;
                return MGL_RENDER_AIR_PROGRAM_ERROR;
            }
        }
        if (stage == _VERTEX_SHADER &&
            spirv->metallib_cull_capture_bytes &&
            (!spirv->mtl_cull_capture_library ||
             !spirv->mtl_cull_capture_function)) {
            mgl::releaseBridgedObject(&spirv->mtl_cull_capture_function);
            mgl::releaseBridgedObject(&spirv->mtl_cull_capture_library);
            if (mgl::loadAIRMainFunction(
                    device, spirv->metallib_cull_capture_bytes,
                    spirv->metallib_cull_capture_size,
                    &spirv->mtl_cull_capture_library,
                    &spirv->mtl_cull_capture_function,
                    err, errcap) != 0) {
                if (failed_stage_out) *failed_stage_out = stage;
                return MGL_RENDER_AIR_PROGRAM_ERROR;
            }
        }
    }
    return MGL_RENDER_AIR_PROGRAM_BOUND;
}

int mglRenderCreatePipelineCacheOwner(int pso_dedup_enabled, int depth_stencil_cache_enabled, int binary_archive_enabled, MGLPipelineCacheOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    try {
        auto owner = std::make_unique<mgl::PipelineCacheOwner>();
        owner->psoDedupEnabled = pso_dedup_enabled != 0;
        owner->depthStencilCacheEnabled =
            depth_stencil_cache_enabled != 0;
        owner->binaryArchiveEnabled = binary_archive_enabled != 0;
        *owner_out = reinterpret_cast<MGLPipelineCacheOwner*>(owner.release());
        return 0;
    } catch (...) {
        return -1;
    }
}

void mglRenderDestroyPipelineCacheOwner(MGLPipelineCacheOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

void mglRenderResetPipelineCacheOwner(MGLPipelineCacheOwner * owner_handle) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->clearCaches();
}

int mglRenderGetPipelineCacheFlags(MGLPipelineCacheOwner * owner_handle, int* pso_dedup_enabled_out, int* depth_stencil_cache_enabled_out, int* binary_archive_enabled_out) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (pso_dedup_enabled_out) {
        *pso_dedup_enabled_out = owner->psoDedupEnabled ? 1 : 0;
    }
    if (depth_stencil_cache_enabled_out) {
        *depth_stencil_cache_enabled_out =
            owner->depthStencilCacheEnabled ? 1 : 0;
    }
    if (binary_archive_enabled_out) {
        *binary_archive_enabled_out = owner->binaryArchiveEnabled ? 1 : 0;
    }
    return 0;
}

void mglRenderDisablePipelineBinaryArchive(MGLPipelineCacheOwner * owner_handle) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->binaryArchiveEnabled = false;
    owner->clearBinaryArchive();
}

int mglRenderGetPipelineBinaryArchiveState(MGLPipelineCacheOwner * owner_handle, int* enabled_out, int* present_out) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (enabled_out) *enabled_out = 0;
    if (present_out) *present_out = 0;
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (enabled_out) *enabled_out = owner->binaryArchiveEnabled ? 1 : 0;
    if (present_out) *present_out = owner->binaryArchive ? 1 : 0;
    return 0;
}

int mglRenderLoadPipelineBinaryArchive(MGLPipelineCacheOwner * owner_handle, const char* cache_key, void* url, int archive_exists, int* reused_out, char* err, size_t errcap) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    auto* archiveURL = static_cast<NS::URL*>(url);
    if (reused_out) *reused_out = 0;
    if (err && errcap) err[0] = '\0';
    if (!owner || !cache_key || !cache_key[0] || !archiveURL) return -1;

    mgl::Renderer& renderer = mgl::renderer();
    std::scoped_lock lock(renderer.mutex, owner->mutex);
    if (!renderer.device || !owner->binaryArchiveEnabled) return -1;

    auto shared = renderer.binaryArchives.find(cache_key);
    if (shared != renderer.binaryArchives.end() && shared->second) {
        owner->clearBinaryArchive();
        owner->binaryArchive = shared->second;
        owner->binaryArchive->retain();
        owner->binaryArchiveKey = cache_key;
        if (reused_out) *reused_out = 1;
        return 0;
    }

    MTL::BinaryArchiveDescriptor* descriptor =
        MTL::BinaryArchiveDescriptor::alloc()->init();
    if (!descriptor) return -1;
    if (archive_exists) descriptor->setUrl(archiveURL);
    NS::Error* nsError = nullptr;
    MTL::BinaryArchive* archive =
        renderer.device->newBinaryArchive(descriptor, &nsError);
    descriptor->release();
    if (!archive) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    archive->setLabel(NS::String::string(
        "MGL Pipeline Binary Archive", NS::UTF8StringEncoding));

    renderer.binaryArchives.emplace(cache_key, archive);
    owner->clearBinaryArchive();
    owner->binaryArchive = archive;
    owner->binaryArchive->retain();
    owner->binaryArchiveKey = cache_key;
    return 0;
}

int mglRenderSerializePipelineBinaryArchive(MGLPipelineCacheOwner * owner_handle, void* url, char* err, size_t errcap) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    auto* archiveURL = static_cast<NS::URL*>(url);
    if (err && errcap) err[0] = '\0';
    if (!owner || !archiveURL) return -1;

    MTL::BinaryArchive* archive = nullptr;
    {
        std::lock_guard<std::mutex> lock(owner->mutex);
        if (!owner->binaryArchiveEnabled || !owner->binaryArchive) return -1;
        archive = owner->binaryArchive;
        archive->retain();
    }
    NS::Error* nsError = nullptr;
    const bool serialized = archive->serializeToURL(archiveURL, &nsError);
    archive->release();
    if (!serialized) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    return 0;
}

void mglRenderDiscardPipelineBinaryArchive(MGLPipelineCacheOwner * owner_handle, const char* cache_key) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return;
    mgl::Renderer& renderer = mgl::renderer();
    std::scoped_lock lock(renderer.mutex, owner->mutex);
    std::string key = cache_key && cache_key[0]
        ? std::string(cache_key) : owner->binaryArchiveKey;
    owner->clearBinaryArchive();
    auto shared = renderer.binaryArchives.find(key);
    if (shared != renderer.binaryArchives.end()) {
        if (shared->second) shared->second->release();
        renderer.binaryArchives.erase(shared);
    }
}

int mglRenderGetPipelineActiveState(MGLPipelineCacheOwner * owner_handle, MGLRenderPipelineActiveState* state_out) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    *state_out = owner->active;
    return 0;
}

int mglRenderInvalidatePipelineActiveState(MGLPipelineCacheOwner * owner_handle) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::PipelineCacheOwner::releaseObject(owner->active.pipeline_state);
    mgl::PipelineCacheOwner::releaseObject(owner->active.vertex_function);
    mgl::PipelineCacheOwner::releaseObject(owner->active.fragment_function);
    owner->active = {};
    owner->active.color0_format =
        static_cast<uint32_t>(MTL::PixelFormatInvalid);
    owner->active.depth_format =
        static_cast<uint32_t>(MTL::PixelFormatInvalid);
    owner->active.stencil_format =
        static_cast<uint32_t>(MTL::PixelFormatInvalid);
    return 0;
}

int mglRenderSetPipelineActiveObject(MGLPipelineCacheOwner * owner_handle, void* pipeline_state) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::PipelineCacheOwner::retainObject(pipeline_state);
    mgl::PipelineCacheOwner::releaseObject(owner->active.pipeline_state);
    owner->active.pipeline_state = pipeline_state;
    return 0;
}

int mglRenderActivatePipelineState(MGLPipelineCacheOwner * owner_handle, const MGLRenderPipelineActiveState* state) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::PipelineCacheOwner::retainObject(state->pipeline_state);
    mgl::PipelineCacheOwner::retainObject(state->vertex_function);
    mgl::PipelineCacheOwner::retainObject(state->fragment_function);
    mgl::PipelineCacheOwner::releaseObject(owner->active.pipeline_state);
    mgl::PipelineCacheOwner::releaseObject(owner->active.vertex_function);
    mgl::PipelineCacheOwner::releaseObject(owner->active.fragment_function);
    owner->active = *state;
    return 0;
}

int mglRenderSetPipelineBlendState(MGLPipelineCacheOwner * owner_handle, uint32_t attachment, const MGLRenderPipelineBlendState* state) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state ||
        attachment >= MGL_RENDER_PIPELINE_COLOR_ATTACHMENTS) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->blend[attachment] = *state;
    return 0;
}

int mglRenderGetPipelineBlendState(MGLPipelineCacheOwner * owner_handle, uint32_t attachment, MGLRenderPipelineBlendState* state_out) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state_out ||
        attachment >= MGL_RENDER_PIPELINE_COLOR_ATTACHMENTS) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    *state_out = owner->blend[attachment];
    return 0;
}

int mglRenderLookupPipeline(MGLPipelineCacheOwner * owner_handle, const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS], MGLRenderPipelineActiveState* state_out) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !key_words || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    auto found = owner->pipelineCache.find(key);
    if (found == owner->pipelineCache.end()) return 0;
    *state_out = {};
    state_out->pipeline_state = found->second->pipeline;
    state_out->vertex_function = found->second->vertexFunction;
    state_out->fragment_function = found->second->fragmentFunction;
    mgl::PipelineCacheOwner::touch(owner->pipelineCacheLRU, key);
    return 1;
}

int mglRenderStorePipeline(MGLPipelineCacheOwner * owner_handle, const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS], const MGLRenderPipelineActiveState* state, uint32_t* evicted_out) {
    if (evicted_out) *evicted_out = 0;
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !key_words || !state || !state->pipeline_state) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    try {
        auto entry = std::make_unique<mgl::PipelineCacheEntry>();
        entry->pipeline =
            static_cast<MTL::RenderPipelineState*>(state->pipeline_state);
        entry->vertexFunction =
            static_cast<MTL::Function*>(state->vertex_function);
        entry->fragmentFunction =
            static_cast<MTL::Function*>(state->fragment_function);
        entry->pipeline->retain();
        if (entry->vertexFunction) entry->vertexFunction->retain();
        if (entry->fragmentFunction) entry->fragmentFunction->retain();

        const bool replacing = owner->pipelineCache.find(key) !=
                               owner->pipelineCache.end();
        uint32_t removed = 0;
        if (!replacing && owner->pipelineCache.size() >= 256u) {
            const size_t target =
                std::max<size_t>(1u, owner->pipelineCache.size() / 4u);
            while (removed < target && !owner->pipelineCacheLRU.empty()) {
                const mgl::PipelineCacheKey oldest =
                    owner->pipelineCacheLRU.front();
                owner->pipelineCacheLRU.pop_front();
                removed += owner->pipelineCache.erase(oldest) ? 1u : 0u;
            }
        }
        owner->pipelineCache[key] = std::move(entry);
        mgl::PipelineCacheOwner::touch(owner->pipelineCacheLRU, key);
        if (evicted_out) *evicted_out = removed;
        return 0;
    } catch (...) {
        return -1;
    }
}

int mglRenderLookupPipelineDescriptorState(MGLPipelineCacheOwner * owner_handle, const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS], MGLRenderPipelineDescriptorState* state_out) {
    if (state_out) *state_out = {};
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !key_words || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    auto found = owner->descriptorCache.find(key);
    if (found == owner->descriptorCache.end()) return 0;
    *state_out = found->second->state;
    mgl::PipelineCacheOwner::touch(owner->descriptorCacheLRU, key);
    return 1;
}

int mglRenderStorePipelineDescriptorState(MGLPipelineCacheOwner * owner_handle, const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS], const MGLRenderPipelineDescriptorState* state) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !key_words || !state) return -1;
    std::unique_ptr<mgl::PipelineCacheDescriptorEntry> entry(
        new (std::nothrow) mgl::PipelineCacheDescriptorEntry());
    if (!entry) return -1;
    entry->state = *state;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const mgl::PipelineCacheKey key =
        mgl::PipelineCacheOwner::makeKey(key_words);
    try {
        owner->descriptorCache[key] = std::move(entry);
        mgl::PipelineCacheOwner::touch(owner->descriptorCacheLRU, key);
        while (owner->descriptorCache.size() > 128u &&
               !owner->descriptorCacheLRU.empty()) {
            const mgl::PipelineCacheKey oldest =
                owner->descriptorCacheLRU.front();
            owner->descriptorCacheLRU.pop_front();
            owner->descriptorCache.erase(oldest);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

int mglRenderCreateRenderPipelineFromState(
    void* vs_function,
    void* fs_function,
    const MGLRenderPipelineDescriptorState* state,
    void* binary_archive,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!vs_function || !state || !pipeline_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    return mglAirCreateRenderPipelineWithArchive(
        renderer.device, vs_function, fs_function, state, binary_archive,
        pipeline_out, err, errcap);
}

int mglRenderCreateRenderPipelineFromStateWithArchiveOwner(MGLPipelineCacheOwner * owner_handle, void* vs_function, void* fs_function, const MGLRenderPipelineDescriptorState* state, void** pipeline_out, char* err, size_t errcap) {
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    MTL::BinaryArchive* archive = nullptr;
    if (owner) {
        std::lock_guard<std::mutex> lock(owner->mutex);
        if (owner->binaryArchiveEnabled && owner->binaryArchive) {
            archive = owner->binaryArchive;
            archive->retain();
        }
    }
    int result = mglRenderCreateRenderPipelineFromState(
        vs_function, fs_function, state, archive, pipeline_out, err, errcap);
    if (archive) archive->release();
    return result;
}

int mglRenderCreateRenderPipelineState(
    void* render_pipeline_descriptor,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::RenderPipelineDescriptor* descriptor =
        static_cast<MTL::RenderPipelineDescriptor*>(
            render_pipeline_descriptor);
    if (!descriptor || !pipeline_out) return -1;

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    NS::Error* nsError = nullptr;
    MTL::RenderPipelineState* pipeline =
        renderer.device->newRenderPipelineState(descriptor, &nsError);
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    *pipeline_out = pipeline;
    return 0;
}

int mglRenderCreateRenderPipelineStateWithArchive(
    void* render_pipeline_descriptor,
    void* binary_archive,
    void** pipeline_out,
    int* archive_hit_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (archive_hit_out) *archive_hit_out = 0;
    if (err && errcap) err[0] = '\0';
    MTL::RenderPipelineDescriptor* descriptor =
        static_cast<MTL::RenderPipelineDescriptor*>(
            render_pipeline_descriptor);
    MTL::BinaryArchive* archive =
        static_cast<MTL::BinaryArchive*>(binary_archive);
    if (!descriptor || !pipeline_out) return -1;

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    const bool archiveEligible = archive && descriptor->vertexFunction() &&
                                 descriptor->fragmentFunction();
    MTL::RenderPipelineState* pipeline = nullptr;
    NS::Error* nsError = nullptr;
    if (archiveEligible) {
        descriptor->setBinaryArchives(NS::Array::array(archive));
        pipeline = renderer.device->newRenderPipelineState(
            descriptor, MTL::PipelineOptionFailOnBinaryArchiveMiss,
            nullptr, &nsError);
        if (pipeline && archive_hit_out) *archive_hit_out = 1;
    }

    const bool archiveMiss = archiveEligible && !pipeline;
    if (!pipeline) {
        nsError = nullptr;
        pipeline = renderer.device->newRenderPipelineState(
            descriptor, &nsError);
    }
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }

    if (archiveMiss) {
        NS::Error* addError = nullptr;
        if (!archive->addRenderPipelineFunctions(descriptor, &addError)) {
            char addMessage[512] = {0};
            mgl::copyError(addError, addMessage, sizeof(addMessage));
            fprintf(stderr,
                    "MGL BINARY ARCHIVE: addRenderPipeline warning: %s\n",
                    addMessage[0] ? addMessage : "unknown error");
        }
    }
    *pipeline_out = pipeline;
    return 0;
}

int mglRenderCreateRenderPipelineStateWithArchiveOwner(MGLPipelineCacheOwner * owner_handle, void* render_pipeline_descriptor, void** pipeline_out, int* archive_hit_out, char* err, size_t errcap) {
    if (archive_hit_out) *archive_hit_out = 0;
    auto* owner = reinterpret_cast<mgl::PipelineCacheOwner*>(static_cast<void*>(owner_handle));
    MTL::BinaryArchive* archive = nullptr;
    if (owner) {
        std::lock_guard<std::mutex> lock(owner->mutex);
        if (owner->binaryArchiveEnabled && owner->binaryArchive) {
            archive = owner->binaryArchive;
            archive->retain();
        }
    }
    int result = archive
        ? mglRenderCreateRenderPipelineStateWithArchive(
              render_pipeline_descriptor, archive, pipeline_out,
              archive_hit_out, err, errcap)
        : mglRenderCreateRenderPipelineState(
              render_pipeline_descriptor, pipeline_out, err, errcap);
    if (archive) archive->release();
    return result;
}

int mglRenderCreateComputePipelineState(void* function,
                                           void** pipeline_out,
                                           char* err,
                                           size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Function* computeFunction =
        static_cast<MTL::Function*>(function);
    if (!computeFunction || !pipeline_out) return -1;

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    NS::Error* nsError = nullptr;
    MTL::ComputePipelineState* pipeline =
        renderer.device->newComputePipelineState(computeFunction, &nsError);
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    *pipeline_out = pipeline;
    return 0;
}

int mglRenderGetOrCreateComputePipeline(
    void* function,
    uint64_t program_instance,
    uint64_t program_generation,
    uint32_t stage,
    int cache_enabled,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!function || !pipeline_out || program_instance == 0) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }

    mgl::ComputePipelineKey key = {
        reinterpret_cast<uintptr_t>(function), program_instance,
        program_generation, stage};
    if (cache_enabled) {
        auto found = renderer.computePipelines.find(key);
        if (found != renderer.computePipelines.end()) {
            if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
                fprintf(stderr,
                        "MGL METALCPP: compute PSO cache hit "
                        "program=%llu generation=%llu stage=%u function=%p\n",
                        static_cast<unsigned long long>(program_instance),
                        static_cast<unsigned long long>(program_generation),
                        stage, function);
            }
            found->second->retain();
            *pipeline_out = found->second;
            return 0;
        }
    }

    NS::Error* nsError = nullptr;
    MTL::ComputePipelineState* pipeline =
        renderer.device->newComputePipelineState(
            static_cast<MTL::Function*>(function), &nsError);
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    if (cache_enabled) {
        pipeline->retain();
        renderer.computePipelines.emplace(key, pipeline);
    }
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: compute PSO create "
                "program=%llu generation=%llu stage=%u function=%p cache=%d\n",
                static_cast<unsigned long long>(program_instance),
                static_cast<unsigned long long>(program_generation),
                stage, function, cache_enabled != 0);
    }
    *pipeline_out = pipeline;
    return 0;
}

void* mglRenderBindingCreate(uint32_t max_texture_slots) {
    if (max_texture_slots == 0 || max_texture_slots > 128) return nullptr;
    mgl::BindingState* state =
        new (std::nothrow) mgl::BindingState(max_texture_slots);
    if (!state) return nullptr;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    renderer.bindingStates.insert(state);
    return state;
}

void mglRenderBindingDestroy(MGLBindingState * binding_state) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state) return;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    auto found = renderer.bindingStates.find(state);
    if (found == renderer.bindingStates.end()) return;
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: binding dedup "
                "texture=%llu/%llu sampler=%llu/%llu "
                "viewport=%llu/%llu scissor=%llu/%llu fill=%llu/%llu\n",
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_VERTEX_TEXTURE] +
                    state->stats.emitted[MGL_RENDER_BINDING_FRAGMENT_TEXTURE]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_VERTEX_TEXTURE] +
                    state->stats.skipped[MGL_RENDER_BINDING_FRAGMENT_TEXTURE]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_VERTEX_SAMPLER] +
                    state->stats.emitted[MGL_RENDER_BINDING_FRAGMENT_SAMPLER]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_VERTEX_SAMPLER] +
                    state->stats.skipped[MGL_RENDER_BINDING_FRAGMENT_SAMPLER]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_VIEWPORT]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_VIEWPORT]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_SCISSOR]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_SCISSOR]),
                static_cast<unsigned long long>(
                    state->stats.emitted[MGL_RENDER_BINDING_TRIANGLE_FILL]),
                static_cast<unsigned long long>(
                    state->stats.skipped[MGL_RENDER_BINDING_TRIANGLE_FILL]));
    }
    renderer.bindingStates.erase(found);
    delete state;
}

void mglRenderBindingInvalidate(MGLBindingState * binding_state) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (state) state->invalidate();
}

void mglRenderBindingSetValid(MGLBindingState * binding_state, int valid) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (state) state->valid = valid != 0;
}

int mglRenderBindingGetValid(MGLBindingState * binding_state, uint32_t* valid_out) {
    if (valid_out) *valid_out = 0;
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || !valid_out) return -1;
    *valid_out = state->valid ? 1u : 0u;
    return 0;
}

int mglRenderBindingInvalidateVertexBuffer(MGLBindingState * binding_state, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || index >= state->vertexBuffers.size()) return -1;
    mgl::BindingState::replaceObject(
        state->vertexBuffers[index], static_cast<MTL::Buffer*>(nullptr));
    state->vertexBufferOffsets[index] = UINT64_MAX;
    state->vertexBufferMask |= 1U << index;
    return 0;
}

int mglRenderBindingInvalidateFragmentBuffer(MGLBindingState * binding_state, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || index >= state->fragmentBuffers.size()) return -1;
    mgl::BindingState::replaceObject(
        state->fragmentBuffers[index], static_cast<MTL::Buffer*>(nullptr));
    state->fragmentBufferOffsets[index] = UINT64_MAX;
    state->fragmentBufferMask |= 1U << index;
    return 0;
}

int mglRenderBindingGetBuffer(MGLBindingState * binding_state, uint32_t stage, uint32_t index, void** buffer_out, uint64_t* offset_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (offset_out) *offset_out = 0;
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || !buffer_out || !offset_out ||
        stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) {
        return -1;
    }
    const std::vector<MTL::Buffer*>& buffers =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexBuffers : state->fragmentBuffers;
    const std::vector<uint64_t>& offsets =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexBufferOffsets : state->fragmentBufferOffsets;
    if (index >= buffers.size()) return -1;
    *buffer_out = buffers[index];
    *offset_out = offsets[index];
    return 0;
}

void mglRenderBindingOrVertexBufferMask(MGLBindingState * binding_state, uint32_t mask) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (state) state->vertexBufferMask |= mask;
}

void mglRenderBindingOrFragmentBufferMask(MGLBindingState * binding_state, uint32_t mask) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (state) state->fragmentBufferMask |= mask;
}

void mglRenderBindingSetPipelineState(MGLBindingState * binding_state, void* pipeline_state) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (state) {
        mgl::BindingState::replaceObject(
            state->pipelineState,
            static_cast<MTL::RenderPipelineState*>(pipeline_state));
    }
}

void mglRenderBindingSetDepthStencilState(MGLBindingState * binding_state, void* depth_stencil_state) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (state) {
        mgl::BindingState::replaceObject(
            state->depthStencilState,
            static_cast<MTL::DepthStencilState*>(depth_stencil_state));
    }
}

int mglRenderBindingGetPipelineState(MGLBindingState * binding_state, void** pipeline_state_out) {
    if (pipeline_state_out) *pipeline_state_out = nullptr;
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || !pipeline_state_out) return -1;
    *pipeline_state_out = state->pipelineState;
    return 0;
}

int mglRenderBindingGetDepthStencilState(MGLBindingState * binding_state, void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || !depth_stencil_state_out) return -1;
    *depth_stencil_state_out = state->depthStencilState;
    return 0;
}

void mglRenderBindingSetCullMode(MGLBindingState * binding_state, uint32_t mode) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (state) state->lastCullMode = static_cast<MTL::CullMode>(mode);
}

void mglRenderBindingSetWinding(MGLBindingState * binding_state, uint32_t winding) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (state) state->lastWinding = static_cast<MTL::Winding>(winding);
}

void mglRenderBindingSetDepthBias(MGLBindingState * binding_state, float bias, float clamp, float slope_scale) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state) return;
    state->lastDepthBias = bias;
    state->lastDepthBiasClamp = clamp;
    state->lastDepthSlopeScale = slope_scale;
}

void mglRenderBindingSetBlendColor(MGLBindingState * binding_state, float red, float green, float blue, float alpha) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state) return;
    state->lastBlendColorRed = red;
    state->lastBlendColorGreen = green;
    state->lastBlendColorBlue = blue;
    state->lastBlendColorAlpha = alpha;
}

int mglRenderBindingSetPipelineIfNeeded(MGLBindingState * binding_state, void* render_encoder, void* pipeline_state) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::RenderPipelineState* pipeline =
        static_cast<MTL::RenderPipelineState*>(pipeline_state);
    if (!state || !encoder || !pipeline) return -1;
    const bool emitted = !state->valid || state->pipelineState != pipeline;
    if (emitted) {
        encoder->setRenderPipelineState(pipeline);
        mgl::BindingState::replaceObject(state->pipelineState, pipeline);
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetDepthStencilIfNeeded(MGLBindingState * binding_state, void* render_encoder, void* depth_stencil_state) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::DepthStencilState* depthStencil =
        static_cast<MTL::DepthStencilState*>(depth_stencil_state);
    if (!state || !encoder || !depthStencil) return -1;
    const bool emitted = !state->valid ||
                         state->depthStencilState != depthStencil;
    if (emitted) {
        encoder->setDepthStencilState(depthStencil);
        mgl::BindingState::replaceObject(state->depthStencilState, depthStencil);
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetCullIfNeeded(MGLBindingState * binding_state, void* render_encoder, uint32_t mode) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    MTL::CullMode cullMode = static_cast<MTL::CullMode>(mode);
    const bool emitted = !state->valid || state->lastCullMode != cullMode;
    if (emitted) {
        encoder->setCullMode(cullMode);
        state->lastCullMode = cullMode;
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetWindingIfNeeded(MGLBindingState * binding_state, void* render_encoder, uint32_t winding) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    MTL::Winding frontWinding = static_cast<MTL::Winding>(winding);
    const bool emitted = !state->valid || state->lastWinding != frontWinding;
    if (emitted) {
        encoder->setFrontFacingWinding(frontWinding);
        state->lastWinding = frontWinding;
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetDepthBiasIfNeeded(MGLBindingState * binding_state, void* render_encoder, float bias, float clamp, float slope_scale) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    const bool emitted = !state->valid || state->lastDepthBias != bias ||
                         state->lastDepthBiasClamp != clamp ||
                         state->lastDepthSlopeScale != slope_scale;
    if (emitted) {
        encoder->setDepthBias(bias, slope_scale, clamp);
        state->lastDepthBias = bias;
        state->lastDepthBiasClamp = clamp;
        state->lastDepthSlopeScale = slope_scale;
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetBlendColorIfNeeded(MGLBindingState * binding_state, void* render_encoder, float red, float green, float blue, float alpha) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    const bool emitted = !state->valid || state->lastBlendColorRed != red ||
                         state->lastBlendColorGreen != green ||
                         state->lastBlendColorBlue != blue ||
                         state->lastBlendColorAlpha != alpha;
    if (emitted) {
        encoder->setBlendColor(red, green, blue, alpha);
        state->lastBlendColorRed = red;
        state->lastBlendColorGreen = green;
        state->lastBlendColorBlue = blue;
        state->lastBlendColorAlpha = alpha;
    }
    return emitted ? 1 : 0;
}

int mglRenderBindingSetViewport(MGLBindingState * binding_state, void* render_encoder, double origin_x, double origin_y, double width, double height, double znear, double zfar) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    MTL::Viewport viewport = {origin_x, origin_y, width, height, znear, zfar};
    const bool emitted = !state->valid ||
                         !mgl::viewportEqual(state->viewport, viewport);
    if (emitted) {
        encoder->setViewport(viewport);
        state->viewport = viewport;
        state->viewports[0] = viewport;
        state->viewportCount = 1;
    }
    mgl::recordBindingResult(*state, MGL_RENDER_BINDING_VIEWPORT, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingSetViewports(MGLBindingState * binding_state, void* render_encoder, const double* viewports, uint64_t count) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || !viewports || count == 0u ||
        count > MGL_MAX_VIEWPORTS) {
        return -1;
    }
    MTL::Viewport vps[MGL_MAX_VIEWPORTS];
    for (uint64_t i = 0; i < count; i++) {
        vps[i] = {viewports[6 * i], viewports[6 * i + 1],
                  viewports[6 * i + 2], viewports[6 * i + 3],
                  viewports[6 * i + 4], viewports[6 * i + 5]};
    }
    bool same = state->valid && state->viewportCount == count;
    for (uint64_t i = 0; same && i < count; i++) {
        same = mgl::viewportEqual(state->viewports[i], vps[i]);
    }
    if (!same) {
        encoder->setViewports(vps, count);
        for (uint64_t i = 0; i < count; i++) {
            state->viewports[i] = vps[i];
        }
        state->viewportCount = count;
        state->viewport = vps[0];
    }
    mgl::recordBindingResult(*state, MGL_RENDER_BINDING_VIEWPORT, !same);
    return same ? 0 : 1;
}

int mglRenderBindingSetScissor(MGLBindingState * binding_state, void* render_encoder, uint64_t x, uint64_t y, uint64_t width, uint64_t height) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder) return -1;
    MTL::ScissorRect scissor = {
        static_cast<NS::UInteger>(x), static_cast<NS::UInteger>(y),
        static_cast<NS::UInteger>(width), static_cast<NS::UInteger>(height)};
    const bool emitted = !state->valid ||
                         !mgl::scissorEqual(state->scissor, scissor);
    if (emitted) {
        encoder->setScissorRect(scissor);
        state->scissor = scissor;
    }
    mgl::recordBindingResult(*state, MGL_RENDER_BINDING_SCISSOR, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingSetTriangleFill(MGLBindingState * binding_state, void* render_encoder, uint32_t mode) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || mode > MTL::TriangleFillModeLines) return -1;
    MTL::TriangleFillMode fillMode = static_cast<MTL::TriangleFillMode>(mode);
    const bool emitted = !state->valid || state->triangleFillMode != fillMode;
    if (emitted) {
        encoder->setTriangleFillMode(fillMode);
        state->triangleFillMode = fillMode;
    }
    mgl::recordBindingResult(
        *state, MGL_RENDER_BINDING_TRIANGLE_FILL, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingGetStats(MGLBindingState * binding_state, MGLRenderBindingStats* stats_out) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || !stats_out) return -1;
    memcpy(stats_out, &state->stats, sizeof(*stats_out));
    return 0;
}

extern "C"
uint32_t mglRenderVertexAttribComponentSize(uint64_t gl_type) {
    switch (gl_type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1u;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2u;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return 4u;
        case GL_DOUBLE:
            return 8u;
        default:
            return 0u;
    }
}

extern "C"
uint64_t mglRenderVertexAttribElementBytes(uint64_t gl_type, uint32_t size) {
    switch (gl_type) {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
            return 4u;
        default: {
            const uint32_t comp = mglRenderVertexAttribComponentSize(gl_type);
            if (comp == 0u || size == 0u) {
                return 0u;
            }
            return (uint64_t)comp * (uint64_t)size;
        }
    }
}

int mglGetOrCreateProgramComputePipeline(Program* program,
                                         int stage,
                                         void** pipeline_out,
                                         char* err,
                                         size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    const bool validStage = stage == _COMPUTE_SHADER ||
                            stage == _TESS_CONTROL_SHADER ||
                            stage == _TESS_EVALUATION_SHADER ||
                            stage == _GEOMETRY_SHADER;
    if (!program || !validStage || !pipeline_out) {
        if (err && errcap) snprintf(err, errcap, "invalid Program or shader stage");
        return -1;
    }
    MGLShaderModule* spirv = &program->modules[stage];
    /* TES render-vertex / compute dual emit: when the TES carries a separate
     * compute expansion kernel (metallib_bytes_tes_compute) for the
     * indexed-fallback route, build the compute pipeline from that kernel
     * rather than the render-vertex mtl_function.  The kernel function is
     * loaded lazily here because the program may have been bound (and its
     * stages loaded) before the kernel blob was published at link time. */
    const bool hasComputeVariant =
        (stage == _TESS_EVALUATION_SHADER) &&
        spirv->metallib_bytes_tes_compute != nullptr &&
        spirv->metallib_size_tes_compute > 0u;
    if (hasComputeVariant && !spirv->mtl_function_compute) {
        void *kernelLibrary = nullptr;
        void *loadedFunction = nullptr;
        if (mglRenderLoadAIRMainFunction(
                spirv->metallib_bytes_tes_compute,
                spirv->metallib_size_tes_compute,
                &kernelLibrary, &loadedFunction, err, errcap) == 0 &&
            kernelLibrary && loadedFunction) {
            spirv->mtl_function_compute = loadedFunction;
            mgl::releaseBridgedObject(&kernelLibrary);
        } else {
            if (kernelLibrary) mgl::releaseBridgedObject(&kernelLibrary);
            if (loadedFunction) mgl::releaseBridgedObject(&loadedFunction);
            return -1;
        }
    }
    void *kernelFunction =
        hasComputeVariant ? spirv->mtl_function_compute : spirv->mtl_function;
    if (!kernelFunction) {
        if (err && errcap) snprintf(err, errcap, "compiled compute function is unavailable");
        return -1;
    }
    return mglRenderGetOrCreateComputePipeline(
        kernelFunction,
        program->pipeline_cache_instance_id,
        program->pipeline_cache_generation,
        static_cast<uint32_t>(stage), 1, pipeline_out, err, errcap);
}

uint64_t mglRenderPipelineDescriptorSignature(const void *descriptor) {
    const MTL::RenderPipelineDescriptor *pipeline =
        static_cast<const MTL::RenderPipelineDescriptor *>(descriptor);
    uint64_t hash = 1469598103934665603ull;
    if (!pipeline) return hash;
    hash = mglRenderHashStepU64(hash, pipeline->rasterSampleCount());
    hash = mglRenderHashStepU64(hash, pipeline->isRasterizationEnabled());
    hash = mglRenderHashStepU64(hash, pipeline->isAlphaToCoverageEnabled());
    hash = mglRenderHashStepU64(hash, pipeline->isAlphaToOneEnabled());
    hash = mglRenderHashStepU64(hash, pipeline->depthAttachmentPixelFormat());
    hash = mglRenderHashStepU64(hash, pipeline->stencilAttachmentPixelFormat());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationPartitionMode());
    hash = mglRenderHashStepU64(hash, pipeline->maxTessellationFactor());
    hash = mglRenderHashStepU64(hash, pipeline->isTessellationFactorScaleEnabled());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationFactorFormat());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationControlPointIndexType());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationFactorStepFunction());
    hash = mglRenderHashStepU64(hash, pipeline->tessellationOutputWindingOrder());
    MTL::RenderPipelineColorAttachmentDescriptorArray *attachments = pipeline->colorAttachments();
    for (uint32_t i = 0; i < 8u; ++i) {
        MTL::RenderPipelineColorAttachmentDescriptor *attachment =
            attachments ? attachments->object(i) : nullptr;
        if (!attachment) continue;
        hash = mglRenderHashStepU64(hash, attachment->pixelFormat());
        hash = mglRenderHashStepU64(hash, attachment->isBlendingEnabled());
        hash = mglRenderHashStepU64(hash, attachment->sourceRGBBlendFactor());
        hash = mglRenderHashStepU64(hash, attachment->destinationRGBBlendFactor());
        hash = mglRenderHashStepU64(hash, attachment->rgbBlendOperation());
        hash = mglRenderHashStepU64(hash, attachment->sourceAlphaBlendFactor());
        hash = mglRenderHashStepU64(hash, attachment->destinationAlphaBlendFactor());
        hash = mglRenderHashStepU64(hash, attachment->alphaBlendOperation());
        hash = mglRenderHashStepU64(hash, attachment->writeMask());
    }
    return hash;
}

int mglRenderPlanVertexAttribSpan(int64_t binding_offset, int64_t relativeoffset,
                                  uint32_t type, uint32_t size,
                                  int64_t *offset_out, int64_t *span_out,
                                  int64_t *end_out) {
    if (relativeoffset < 0) {
        return MGL_ATTRIB_SPAN_NEGATIVE_RELATIVE;
    }
    const int64_t offset = binding_offset + relativeoffset;
    const uint64_t elem = mglRenderVertexAttribElementBytes(type, size);
    int64_t span = 0;
    if (elem > 0u) {
        if (elem > (uint64_t)INT64_MAX) {
            return MGL_ATTRIB_SPAN_OVERFLOW;
        }
        span = (int64_t)elem;
    }
    const int64_t end = offset + (span > 0 ? span : 1);
    if (offset_out) {
        *offset_out = offset;
    }
    if (span_out) {
        *span_out = span;
    }
    if (end_out) {
        *end_out = end;
    }
    return MGL_ATTRIB_SPAN_OK;
}

int mglRenderPlanAttribFetch(uint32_t gl_type, uint32_t size, uint32_t stride,
                             int64_t binding_offset, int64_t relativeoffset,
                             uint32_t divisor, uint64_t first_vertex,
                             uint64_t last_vertex, int64_t vbo_size,
                             MGLRenderAttribFetchPlan *out) {
    if (!out) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    const uint64_t elem = mglRenderVertexAttribElementBytes(gl_type, size);
    if (elem == 0u) {
        out->status = MGL_ATTRIB_FETCH_BAD_FORMAT;
        return 1;
    }
    const uint64_t use_stride = stride > 0u ? (uint64_t)stride : elem;
    if (binding_offset < 0 || relativeoffset < 0) {
        out->status = MGL_ATTRIB_FETCH_OVERFLOW;
        return 1;
    }
    if ((uint64_t)binding_offset > UINT64_MAX - (uint64_t)relativeoffset) {
        out->status = MGL_ATTRIB_FETCH_OVERFLOW;
        return 1;
    }
    const uint64_t rel = (uint64_t)binding_offset + (uint64_t)relativeoffset;
    if (use_stride == 0u) {
        out->status = MGL_ATTRIB_FETCH_BAD_FORMAT;
        return 1;
    }
    const uint64_t range_first = divisor != 0u ? 0u : first_vertex;
    const uint64_t range_last = divisor != 0u ? 0u : last_vertex;
    if (rel > UINT64_MAX - elem) {
        out->status = MGL_ATTRIB_FETCH_OVERFLOW;
        return 1;
    }
    if (range_last > (UINT64_MAX - rel - elem) / use_stride ||
        range_first > (UINT64_MAX - rel) / use_stride) {
        out->status = MGL_ATTRIB_FETCH_OVERFLOW;
        return 1;
    }
    const uint64_t byte_start = rel + range_first * use_stride;
    const uint64_t byte_end = rel + range_last * use_stride + elem;
    const uint64_t vbo = vbo_size > 0 ? (uint64_t)vbo_size : 0u;
    out->elem_bytes = elem;
    out->stride = use_stride;
    out->rel_offset = rel;
    out->byte_start = byte_start;
    out->byte_end = byte_end;
    if (byte_end > vbo) {
        out->status = MGL_ATTRIB_FETCH_OOB;
        return 1;
    }
    out->status = MGL_ATTRIB_FETCH_OK;
    return 1;
}

extern "C" void mglRenderPlanVertexAttribFormat(
    uint32_t type, uint32_t size, int integer, int normalized,
    int is_color_input, uint32_t shader_gl_type, uint32_t *format_out,
    int *needs_conversion_out, int *normalized_out, int *conversion_kind_out) {
    int norm = normalized;
    if (!norm && type == GL_UNSIGNED_BYTE && size == 4u && is_color_input) {
        norm = 1;
    }
    if (normalized_out) {
        *normalized_out = norm;
    }
    uint32_t format = 0u;
    int needs_conversion = 0;
    int kind = MGL_ATTRIB_CONV_NONE;
    if (type == GL_DOUBLE) {
        needs_conversion = 1;
        kind = MGL_ATTRIB_CONV_DOUBLE;
        format = mglRenderDoubleVertexAttribFloatFormat(size);
    } else if (integer == 0 &&
               (type == GL_INT || type == GL_UNSIGNED_INT)) {
        needs_conversion = 1;
        kind = MGL_ATTRIB_CONV_INT_TO_FLOAT;
        format = mglRenderDoubleVertexAttribFloatFormat(size);
    } else if (type == GL_FIXED) {
        needs_conversion = 1;
        kind = MGL_ATTRIB_CONV_FIXED;
        format = mglRenderDoubleVertexAttribFloatFormat(size);
    } else if (type == GL_UNSIGNED_INT_10_10_10_2) {
        needs_conversion = 1;
        kind = MGL_ATTRIB_CONV_UINT_1010102;
        format = mglRenderDoubleVertexAttribFloatFormat(4u);
    } else if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
        needs_conversion = 1;
        kind = MGL_ATTRIB_CONV_UINT_10F11F11F;
        format = mglRenderDoubleVertexAttribFloatFormat(3u);
    } else if (integer == 1) {
        const uint32_t converted = mglRenderIntegerAttribConversionFormat(
            type, shader_gl_type, size);
        if (converted != 0u && converted != MGLVertexFormatInvalid) {
            needs_conversion = 1;
            kind = MGL_ATTRIB_CONV_INTEGER_SIGN;
            format = converted;
        } else {
            format = mglRenderGLTypeSizeToVertexFormat(type, size, norm);
        }
    } else {
        format = mglRenderGLTypeSizeToVertexFormat(type, size, norm);
    }
    if (format_out) {
        *format_out = format;
    }
    if (needs_conversion_out) {
        *needs_conversion_out = needs_conversion;
    }
    if (conversion_kind_out) {
        *conversion_kind_out = kind;
    }
}

extern "C" uint32_t mglRenderPlanVertexAttribStride(
    uint32_t type, uint32_t size, int integer, int uses_current,
    int integer_converted, uint32_t resolved_stride,
    uint32_t existing_layout_stride) {
    if (uses_current) {
        return 16u;
    }
    if (type == GL_DOUBLE) {
        const uint64_t raw =
            resolved_stride > 0u ? resolved_stride
                                 : (uint64_t)size * sizeof(double);
        return (uint32_t)mglRenderAlignVertexStrideForMetal(raw);
    }
    if (integer == 0 && (type == GL_INT || type == GL_UNSIGNED_INT)) {
        const uint64_t raw =
            resolved_stride > 0u ? resolved_stride
                                 : (uint64_t)size * sizeof(int32_t);
        return (uint32_t)mglRenderAlignVertexStrideForMetal(raw);
    }
    if (type == GL_FIXED) {
        const uint64_t raw =
            resolved_stride > 0u ? resolved_stride
                                 : (uint64_t)size * sizeof(int32_t);
        const uint64_t min_bytes = (uint64_t)size * sizeof(float);
        return (uint32_t)mglRenderAlignVertexStrideForMetal(
            raw > min_bytes ? raw : min_bytes);
    }
    if (type == GL_UNSIGNED_INT_10_10_10_2) {
        const uint64_t raw =
            resolved_stride > 0u ? resolved_stride : sizeof(uint32_t);
        const uint64_t min_bytes = 4u * sizeof(float);
        return (uint32_t)mglRenderAlignVertexStrideForMetal(
            raw > min_bytes ? raw : min_bytes);
    }
    if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
        const uint64_t raw =
            resolved_stride > 0u ? resolved_stride : sizeof(uint32_t);
        const uint64_t min_bytes = 3u * sizeof(float);
        return (uint32_t)mglRenderAlignVertexStrideForMetal(
            raw > min_bytes ? raw : min_bytes);
    }
    if (integer == 1 && integer_converted) {
        return (uint32_t)mglRenderAlignVertexStrideForMetal(
            (uint64_t)size * sizeof(int32_t));
    }
    return existing_layout_stride == 0u ? resolved_stride
                                        : existing_layout_stride;
}

MTL::Library* loadAuxLibraryLocked(mgl::Renderer& renderer,
                                   const unsigned char* bytes,
                                   size_t size,
                                   uint64_t asset_hash,
                                   char* err,
                                   size_t errcap) {
    if (!bytes || size == 0) {
        if (err && errcap) {
            snprintf(err, errcap, "aux shader asset row is empty (table not built?)");
        }
        return nullptr;
    }
    auto found = renderer.auxLibraries.find(asset_hash);
    if (found != renderer.auxLibraries.end()) return found->second;
    const uint64_t computedHash = mglAuxAssetHash(bytes, size);
    if (asset_hash == 0) {
        /* Runtime-compiled blob (no committed table row): key the library
         * cache on the content hash instead of a table fingerprint. */
        asset_hash = computedHash;
        auto cached = renderer.auxLibraries.find(asset_hash);
        if (cached != renderer.auxLibraries.end()) return cached->second;
    } else if (computedHash != asset_hash) {
        if (err && errcap) {
            snprintf(err, errcap,
                     "aux shader asset hash mismatch (table 0x%016llx, computed 0x%016llx)",
                     static_cast<unsigned long long>(asset_hash),
                     static_cast<unsigned long long>(computedHash));
        }
        return nullptr;
    }
    // Same loading path as mglAirLoadLibrary: dispatch_data -> newLibrary.
    dispatch_data_t dispatchData = dispatch_data_create(
        bytes, size, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (!dispatchData) {
        if (err && errcap) {
            snprintf(err, errcap, "aux shader asset dispatch_data_create failed");
        }
        return nullptr;
    }
    NS::Error* nsError = nullptr;
    MTL::Library* library = renderer.device->newLibrary(dispatchData, &nsError);
#ifdef __OBJC__
    // -fobjc-arc builds (test_metalcpp_smoke) manage dispatch objects
    // automatically; the C++ lib build releases the temporary manually.
    (void)dispatchData;
#else
    dispatch_release(dispatchData);
#endif
    if (!library) {
        mgl::copyError(nsError, err, errcap);
        return nullptr;
    }
    library->retain();
    renderer.auxLibraries.emplace(asset_hash, library);
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: aux shader asset library loaded "
                "hash=0x%016llx bytes=%zu\n",
                static_cast<unsigned long long>(asset_hash), size);
    }
    return library;
}

MTL::Function* newAuxEntryFunction(MTL::Library* library,
                                   const char* entry,
                                   char* err,
                                   size_t errcap) {
    if (!entry) return nullptr;
    MTL::Function* function = library->newFunction(
        NS::String::string(entry, NS::UTF8StringEncoding));
    if (!function && err && errcap) {
        snprintf(err, errcap, "aux shader entry function '%s' not found",
                 entry);
    }
    return function;
}

int getOrCreateAuxComputePipelineLocked(mgl::Renderer& renderer,
                                        void* function,
                                        uint32_t kind,
                                        uint64_t variant,
                                        void** pipeline_out,
                                        char* err,
                                        size_t errcap) {
    mgl::AuxComputePipelineKey key = {kind, variant};
    auto found = renderer.auxComputePipelines.find(key);
    if (found != renderer.auxComputePipelines.end()) {
        if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
            fprintf(stderr,
                    "MGL METALCPP: aux compute PSO cache hit "
                    "kind=%u variant=%llu\n",
                    kind, static_cast<unsigned long long>(variant));
        }
        found->second->retain();
        *pipeline_out = found->second;
        return 0;
    }
    if (!function) return 1;

    NS::Error* nsError = nullptr;
    MTL::ComputePipelineState* pipeline =
        renderer.device->newComputePipelineState(
            static_cast<MTL::Function*>(function), &nsError);
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    pipeline->retain();
    renderer.auxComputePipelines.emplace(key, pipeline);
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: aux compute PSO create "
                "kind=%u variant=%llu function=%p\n",
                kind, static_cast<unsigned long long>(variant), function);
    }
    *pipeline_out = pipeline;
    return 0;
}

int mglRenderGetOrCreateAuxComputePipeline(
    void* function,
    uint32_t kind,
    uint64_t variant,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!pipeline_out || kind == 0) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    return getOrCreateAuxComputePipelineLocked(
        renderer, function, kind, variant, pipeline_out, err, errcap);
}

int mglRenderGetOrCreateAuxComputePipelineFromMetallib(
    const unsigned char* bytes,
    size_t size,
    uint64_t asset_hash,
    const char* entry_name,
    uint32_t kind,
    uint64_t variant,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!pipeline_out || kind == 0 || !entry_name) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    MTL::Library* library = loadAuxLibraryLocked(
        renderer, bytes, size, asset_hash, err, errcap);
    if (!library) return -1;
    MTL::Function* function =
        newAuxEntryFunction(library, entry_name, err, errcap);
    if (!function) return -1;
    int result = getOrCreateAuxComputePipelineLocked(
        renderer, function, kind, variant, pipeline_out, err, errcap);
    function->release();
    return result;
}

int recordBufferSlot(std::vector<MTL::Buffer*>& buffers,
                     std::vector<uint64_t>& offsets,
                     uint32_t& mask,
                     void* buffer,
                     uint64_t offset,
                     uint32_t index,
                     bool markPresent) {
    if (index >= buffers.size()) return -1;
    mgl::BindingState::replaceObject(
        buffers[index], static_cast<MTL::Buffer*>(buffer));
    offsets[index] = offset;
    if (markPresent) mask |= 1U << index;
    return 0;
}

int mglRenderBindingRecordVertexBuffer(MGLBindingState * binding_state, void* buffer, uint64_t offset, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    return state ? recordBufferSlot(state->vertexBuffers,
                                    state->vertexBufferOffsets,
                                    state->vertexBufferMask,
                                    buffer, offset, index, true) : -1;
}

int mglRenderBindingRecordFragmentBuffer(MGLBindingState * binding_state, void* buffer, uint64_t offset, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    return state ? recordBufferSlot(state->fragmentBuffers,
                                    state->fragmentBufferOffsets,
                                    state->fragmentBufferMask,
                                    buffer, offset, index, true) : -1;
}

int mglRenderBindingUpdateVertexBuffer(MGLBindingState * binding_state, void* buffer, uint64_t offset, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    return state ? recordBufferSlot(state->vertexBuffers,
                                    state->vertexBufferOffsets,
                                    state->vertexBufferMask,
                                    buffer, offset, index, false) : -1;
}

int mglRenderBindingUpdateFragmentBuffer(MGLBindingState * binding_state, void* buffer, uint64_t offset, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    return state ? recordBufferSlot(state->fragmentBuffers,
                                    state->fragmentBufferOffsets,
                                    state->fragmentBufferMask,
                                    buffer, offset, index, false) : -1;
}

int mglRenderEncodeBindingSnapshotForRenderEncoderOwner(MGLRenderEncoderOwner * render_encoder_owner, const MGLRenderBindingSnapshot* snapshot, char* err, size_t errcap) {
    return mglRenderEncodeBindingSnapshot(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        snapshot, err, errcap);
}

int mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, const MGLRenderResourceBindingSnapshot* snapshot, char* err, size_t errcap) {
    return mglRenderEncodeResourceBindingSnapshot(
        binding_state,
        mglRenderActiveRenderEncoder(render_encoder_owner),
        snapshot, err, errcap);
}

int mglRenderBindingSetPipelineIfNeededForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, void* pipeline_state) {
    return mglRenderBindingSetPipelineIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        pipeline_state);
}

int mglRenderBindingSetDepthStencilIfNeededForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, void* depth_stencil_state) {
    return mglRenderBindingSetDepthStencilIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        depth_stencil_state);
}

int mglRenderBindingSetCullIfNeededForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, uint32_t mode) {
    return mglRenderBindingSetCullIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        mode);
}

int mglRenderBindingSetWindingIfNeededForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, uint32_t winding) {
    return mglRenderBindingSetWindingIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        winding);
}

int mglRenderBindingSetBlendColorIfNeededForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, float red, float green, float blue, float alpha) {
    return mglRenderBindingSetBlendColorIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        red, green, blue, alpha);
}

int mglRenderBindingSetDepthBiasIfNeededForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, float depth_bias, float clamp, float slope_scale) {
    return mglRenderBindingSetDepthBiasIfNeeded(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        depth_bias, clamp, slope_scale);
}

int mglRenderBindingSetViewportForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, double origin_x, double origin_y, double width, double height, double znear, double zfar) {
    return mglRenderBindingSetViewport(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        origin_x, origin_y, width, height, znear, zfar);
}

int mglRenderBindingSetViewportsForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, const double* viewports, uint64_t count) {
    return mglRenderBindingSetViewports(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        viewports, count);
}

int mglRenderBindingSetScissorForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, uint64_t x, uint64_t y, uint64_t width, uint64_t height) {
    return mglRenderBindingSetScissor(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        x, y, width, height);
}

int mglRenderBindingSetTriangleFillForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, uint32_t mode) {
    return mglRenderBindingSetTriangleFill(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        mode);
}

int mglRenderSetRenderPipelineStateForOwner(MGLRenderEncoderOwner * render_encoder_owner, void* pipeline_state) {
    return mglRenderSetRenderPipelineState(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        pipeline_state);
}
