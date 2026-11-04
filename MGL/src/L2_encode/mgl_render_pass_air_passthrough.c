/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_pass_manager_ops.c — C twins of the MGLRenderPassManager methods
 * that only touch the command state.  The manager keeps calling its own
 * methods; C callers use these.
 */

#include "mgl_render_pass_sync_ops.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_pass_ops_shared.h"
#include "mgl_state_apply.h"
#include "mgl_rt_sync.h"   /* mglRenderTargetMarkGPUAuthoritative */
#include "mgl_render_encoder_ops.h"
#include "mgl_attachment_binding.h"   /* mglRendererBindFramebufferAttachmentTextures */
#include "mgl_buffer_map.h"           /* map / dirty base buffers */
#include "mgl_texture_bind.h"          /* mglRendererBindMTLTexture */
#include "mgl_draw_buffer.h"          /* Metal draw-buffer mapping */
#include "mgl_sync.h"                 /* MGLMetalAttachmentSubresource */
#include "mgl_renderer_backend.h"     /* default draw buffer attachments */
#include "mgl_renderer_binding_ops.h"  /* mglRendererBindActiveTexturesToMTL */
#include "mgl_stage_encode_drivers.h" /* stage encode bind drivers */
#include "mgl_frame_activity.h"     /* MGL_ENC_REASON_* */
#include "mgl_render_api_query.h"  /* MGLRenderPipelineDescriptorState */
#include "mgl_render_api_lifecycle.h" /* mglRenderLoadAIRMainFunction */
#include "mgl_byte_hash.h"         /* mglHashStepU64 */
#include "mgl_draw_tess.h"         /* mglTessRasterGLMode */
#include "mgl_vertex_layout.h"     /* mglRendererGenerateVertexDescriptorState */
#include "mgl_render_pass_manager.h" /* pass-manager transaction entries */
#include "mgl_render_pass_clear.h"   /* mglRenderPassPlanClearValues */
#include "mgl_program_resource.h"   /* mglProgramStageBuiltinMask */
#include "mgl_shader_abi.h"         /* mglAIRPerVertexStrideForResources, mglShaderCompileGLSL */
#include "mgl_buffer_slots.h"       /* mglProgramStageHasSamplers (shared with the link gate) */
#include "mgl_metal_ref.h"         /* mglReleaseMetalObjNoNull */
#include "mgl_draw_gs.h"           /* mglDrawGsPassthroughDeclType */
#include "mgl_render_pass_plan.h"    /* mglRenderProcessGLState*, plans */
#include "mgl_buffer_slots.h"        /* kMGL*BufferIndex */
#include "mgl_renderer_binding_ops.h"  /* mglRendererSyncResourceBindingsForContext */
#include "mgl_swap_diagnostics.h"     /* swap colour copy + diagnostics */
#include <dispatch/dispatch.h>       /* dispatch_async_f */
#include "mgl_trace_log.h"         /* mglTraceLog, kMGLDiagnosticStateLogs */
#include "mgl_gpu_recovery.h"      /* mglRendererRecordGPUError */
#include "mgl_pso_format_class.h"  /* mglRenderDefaultColorPixelFormat */
#include "mgl_render.h"           /* attachment kinds, MS plane adjust */
#include "mgl_texture_compat.h"   /* mglMetalTextureLevelDimension */

#include "mgl_pso_build_ops.h"    /* mglRenderPassSyncPipelineState */
#include "mgl_renderer_state_areas.h"
#include "mgl_renderer_binding_ops.h"      /* mglRendererBindingInvalidateLastBoundState */
#include "mgl_trace_strategy.h"         /* mglClearFragmentTraceBindingsForRenderer */

#include <stdio.h>
#include <stdarg.h>   /* va_list for the GLSL source builder */
#include <stdlib.h>   /* malloc / realloc / free */
#include <string.h>   /* memcpy / strlen */
#include <math.h>     /* lrintf */
#include "mgl_readback_policy.h"
#include "mgl_program_backend.h"
#include "mgl_texture_backend.h"
#include "mgl_raster_orientation_state.h"

uint32_t mglPdGeometryPassthroughLayerStride(GLMContext ctx);
uint64_t mglPdGeometryPassthroughCacheKey(const Program *program,                                                 uint32_t layerStride);


/* === AIR passthrough vertex builders (log 177) ========================= */

/* The .m built the source with NSMutableString/appendFormat:; the C twins use
 * this growable buffer instead (vsnprintf into the tail, doubling on demand). */
typedef struct MglPdSource_t {
    char *data;
    size_t len;
    size_t cap;
} MglPdSource;

static void mglPdSourceInit(MglPdSource *src)
{
    src->cap = 4096u;
    src->len = 0u;
    src->data = (char *)malloc(src->cap);
    if (src->data) src->data[0] = '\0';
}

static void mglPdSourceGrow(MglPdSource *src, size_t need)
{
    if (!src->data || need + 1u <= src->cap) return;
    size_t cap = src->cap;
    while (cap < need + 1u) cap *= 2u;
    char *grown = (char *)realloc(src->data, cap);
    if (!grown) return;
    src->data = grown;
    src->cap = cap;
}

/* appendString: */
static void mglPdSourceAppendRaw(MglPdSource *src, const char *text)
{
    if (!src->data || !text) return;
    const size_t n = strlen(text);
    mglPdSourceGrow(src, src->len + n);
    if (!src->data || src->len + n + 1u > src->cap) return;
    memcpy(src->data + src->len, text, n);
    src->len += n;
    src->data[src->len] = '\0';
}

/* appendFormat: */
static void mglPdSourceAppendF(MglPdSource *src, const char *fmt, ...)
{
    if (!src->data) return;
    va_list ap;
    va_start(ap, fmt);
    for (;;) {
        va_list ap2;
        va_copy(ap2, ap);
        const int n = vsnprintf(src->data + src->len, src->cap - src->len, fmt,
                                ap2);
        va_end(ap2);
        if (n < 0) break;
        if ((size_t)n + 1u <= src->cap - src->len) {
            src->len += (size_t)n;
            break;
        }
        mglPdSourceGrow(src, src->len + (size_t)n);
        if (src->len + (size_t)n + 1u > src->cap) break; /* allocation failed */
    }
    va_end(ap);
}

static void mglPdSourceFree(MglPdSource *src)
{
    free(src->data);
    src->data = NULL;
    src->len = 0u;
    src->cap = 0u;
}

/* The .m twin hands the two +1 handles to ARC, which releases them at scope
 * exit; the C twin takes the same +1 and releases it explicitly with the
 * shared helper (mgl_metal_ref.h). */
static bool mglPdLoadAIRMainFunction(const unsigned char *bytes, size_t size,
                                     void **libraryOut, void **functionOut,
                                     char *errorText, size_t errorCap)
{
    if (libraryOut) *libraryOut = NULL;
    if (functionOut) *functionOut = NULL;
    if (!bytes || size == 0u || !libraryOut || !functionOut) {
        if (errorText && errorCap) snprintf(errorText, errorCap, "bad args");
        return false;
    }
    void *libraryHandle = NULL;
    void *functionHandle = NULL;
    if (mglRenderLoadAIRMainFunction(bytes, size, &libraryHandle,
                                     &functionHandle, errorText,
                                     errorCap) != 0 ||
        !libraryHandle || !functionHandle) {
        return false;
    }
    *libraryOut = libraryHandle;
    *functionOut = functionHandle;
    return true;
}

/* Twins of the .m geometry-passthrough type mappers. */
static const char *mglPdGeometryPassthroughColumnSwizzle(unsigned rows)
{
    return mglRenderGLSLColumnSwizzle(rows);
}

static const char *mglPdGeometryPassthroughColumnType(unsigned rows)
{
    return mglRenderGLSLColumnType(rows);
}

static const char *mglPdGeometryPassthroughFloatType(GLenum type)
{
    return mglRenderGLSLIntegerAsFloatType((uint32_t)type);
}

static const char *mglPdGeometryPassthroughSwizzle(GLenum type)
{
    return mglRenderGLSLTypeSwizzle((uint32_t)type);
}

static const char *mglPdGeometryPassthroughType(GLenum type)
{
    return mglRenderGLSLTypeName((uint32_t)type);
}

static unsigned mglPdGeometryPassthroughMatrixCols(GLenum type)
{
    return (unsigned)mglRenderGLSLMatrixCols((uint32_t)type);
}

static unsigned mglPdGeometryPassthroughMatrixRows(GLenum type)
{
    return (unsigned)mglRenderGLSLMatrixRows((uint32_t)type);
}

static bool mglPdGeometryPassthroughNeedsFlat(GLenum type)
{
    return mglRenderGLSLNeedsFlat((uint32_t)type) != 0;
}

/* gl_PerVertex members stay in STAGE_OUTPUT with location UINT32_MAX so they
 * do not steal user slots, but they are not user varyings.  Emitting
 * layout(location=4294967295) out vec4 gl_Position overwrites the builtin
 * position load (slot wraparound) and breaks Metal PSO creation. */
static int mglPdPassthroughSkipBuiltinOutput(const MGLShaderResource *output)
{
    if (!output || !output->name)
        return 1;
    if (output->location >= 0x0fffffffu)
        return 1;
    if (strncmp(output->name, "gl_", 3) != 0)
        return 0;
    return strcmp(output->name, "gl_Position") == 0 ||
           strcmp(output->name, "gl_PointSize") == 0 ||
           strcmp(output->name, "gl_ClipDistance") == 0 ||
           strcmp(output->name, "gl_CullDistance") == 0 ||
           strcmp(output->name, "gl_Layer") == 0 ||
           strcmp(output->name, "gl_ViewportIndex") == 0;
}

/* Interface-block members keep query names like "BLOCK.value".  Those are not
 * GLSL identifiers; Metal still matches the FS by layout(location). */
static int mglPdPassthroughNameIsIdent(const char *name)
{
    unsigned char c;
    if (!name || !*name)
        return 0;
    c = (unsigned char)name[0];
    if (!(c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
        return 0;
    for (name++; *name; name++) {
        c = (unsigned char)*name;
        if (!(c == '_' || (c >= '0' && c <= '9') ||
              (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
            return 0;
    }
    return 1;
}

static const char *mglPdPassthroughIdent(const MGLShaderResource *output,
                                         unsigned index, char *buf, size_t cap)
{
    if (!output || !buf || cap < 20u)
        return NULL;
    if (mglPdPassthroughNameIsIdent(output->name))
        return output->name;
    if (snprintf(buf, cap, "mgl_ptvs_%u", index) < 0)
        return NULL;
    return buf;
}

static GLenum mglPdPassthroughDeclType(const MGLShaderResourceList *fsInputs,
                                       const MGLShaderResource *output)
{
    uint32_t decl = output->gl_type;
    for (GLuint fi = 0; fsInputs && fsInputs->list && fi < fsInputs->count;
         fi++) {
        const MGLShaderResource *in = &fsInputs->list[fi];
        decl = mglDrawGsPassthroughDeclType(
            decl, in->gl_type,
            output->name && in->name && strcmp(in->name, output->name) == 0
                ? 1
                : 0);
        if (decl != output->gl_type) {
            return (GLenum)decl;
        }
    }
    return (GLenum)decl;
}

/* -ensureAIRGeometryPassthroughFunctionForProgram:outputPrimitive:.  The
 * output-primitive argument was already unused in the .m body. */
int mglRenderPassEnsureAIRGeometryPassthroughFunctionForProgram(
    void *renderer, Program *program, uint32_t outputPrimitive)
{
    (void)outputPrimitive;
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;

    if (!program) return 0;
    const uint32_t layerStride = mglPdGeometryPassthroughLayerStride(ctx);
    const uint64_t passthroughKey =
        mglPdGeometryPassthroughCacheKey(program, layerStride);
    void *cachedFunction = NULL;
    if (mglRendererBackendGetPassthroughFunction(
            areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
            passthroughKey, &cachedFunction) == 1) {
        return 1;
    }
    (void)mglRendererBackendSetPassthroughFunction(
        areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY, NULL, NULL,
        0u);

    MGLShaderResourceList *outputs =
        &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES];
    const uint64_t recordStride = mglAIRPerVertexStrideForResources(outputs);
    const uint64_t vec4Stride = recordStride / 16u;
    MglPdSource src;
    mglPdSourceInit(&src);
    mglPdSourceAppendRaw(&src,
                         "#version 460 core\n"
                         "layout(std430, binding = 0) buffer MGLGSOutput {\n"
                         "    vec4 records[];\n"
                         "} mgl_gs_output;\n");
    if (layerStride > 1u) {
        /* Sample-loop plane offset; binding matches kMGLEmulatedMSPlaneBufferIndex. */
        mglPdSourceAppendF(
            &src,
            "layout(std140, binding = %u) uniform MGLEmulatedMSPlaneBlock {\n"
            "    int mgl_emulated_ms_plane;\n"
            "};\n",
            (unsigned)kMGLEmulatedMSPlaneBufferIndex);
    }
    /* The stage-out record stores every varying as a full vec4 slot, so the
     * reflected gl_type of a GS output is promoted to the record width.  The
     * passthrough VS must declare the interface with the *fragment shader's*
     * input type instead: Metal rejects a pipeline whose vertex output type
     * differs from the fragment input (e.g. record-promoted vec4 vs a declared
     * vec3). */
    const MGLShaderResourceList *fsInputs =
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES];

    /* Builtins never appear in the reflected output list (all gl_ builtins are
     * filtered during reflection), so ask the frontend for the exact per-stage
     * usage mask instead of scanning the GS source text.  gl_PointSize
     * forwarding matters because the pipeline builder rejects a vertex stage
     * writing point size on a Line/Triangle topology, while Points-topology
     * programs expect the real size. */
    const uint32_t gsBuiltins =
        mglProgramStageBuiltinMask(program, _GEOMETRY_SHADER);
    int hasPointSize = (gsBuiltins & MGL_AIR_BUILTIN_POINT_SIZE) != 0u;
    /* GS-written gl_PrimitiveID is parked at record offset 64 (vec4 slot 4,
     * component x; layout v2 / A04) and ferried to the fragment stage as a flat
     * float varying at the reserved location below. */
    int hasPrimitiveId = (gsBuiltins & MGL_AIR_BUILTIN_PRIMITIVE_ID) != 0u;
    int hasClipDistance = (gsBuiltins & MGL_AIR_BUILTIN_CLIP_DISTANCE) != 0u;
    int result = 0;
    if (hasPrimitiveId) {
        /* Float carrier: the GS kernel stores sitofp(id) and a flat int
         * stage_input that is actually read crashes Apple's AGX compiler (see
         * storeGeometryPrimitiveId). */
        mglPdSourceAppendF(
            &src, "layout(location = %u) flat out float mgl_primitive_id;\n",
            (unsigned)MGL_AIR_PRIMITIVE_ID_LOCATION);
    }
    if (hasClipDistance) {
        mglPdSourceAppendF(&src, "out float gl_ClipDistance[%u];\n",
                           (unsigned)MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT);
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;

        if (output->stream > 0) continue;
        if (mglPdPassthroughSkipBuiltinOutput(output)) continue;
        if (getenv("MGL_DUMP_AIR"))
            fprintf(stderr, "MGL PTVS varying: name=%s gl_type=0x%x loc=%u\n",
                    output->name ? output->name : "?",
                    (unsigned)output->gl_type, (unsigned)output->location);
        const GLenum declType = mglPdPassthroughDeclType(fsInputs, output);
        const unsigned matCols = mglPdGeometryPassthroughMatrixCols(declType);
        const unsigned matRows = mglPdGeometryPassthroughMatrixRows(declType);
        char identBuf[32];
        const char *ident =
            mglPdPassthroughIdent(output, (unsigned)i, identBuf, sizeof identBuf);
        if (matCols > 0u) {
            /* Metal rejects matrix stage-out attributes; emit one vector
             * output per column at consecutive locations (GL 4.6 4.4.1). */
            const char *colType = mglPdGeometryPassthroughColumnType(matRows);
            if (!colType || !ident) {
                fprintf(stderr,
                        "MGL GS ERROR: unsupported passthrough matrix type 0x%x\n",
                        (unsigned)output->gl_type);
                goto done;
            }
            for (unsigned c = 0; c < matCols; c++) {
                mglPdSourceAppendF(
                    &src, "layout(location = %u) out %s %s_c%u;\n",
                    (unsigned)(output->location + c), colType, ident, c);
            }
            continue;
        }
        /* Integer varyings ride as float carriers (the AIR backend pairs this
         * with an fptosi at the fragment entry; raw int attributes do not
         * survive the GS-expansion pipeline plumbing). */
        const char *type =
            mglPdGeometryPassthroughNeedsFlat(declType)
                ? mglPdGeometryPassthroughFloatType(declType)
                : mglPdGeometryPassthroughType(declType);
        if (!type || !ident) {
            fprintf(stderr,
                    "MGL GS ERROR: unsupported passthrough varying type 0x%x\n",
                    (unsigned)output->gl_type);
            goto done;
        }
        mglPdSourceAppendF(
            &src, "layout(location = %u) %sout %s %s;\n",
            (unsigned)output->location,
            mglPdGeometryPassthroughNeedsFlat(output->gl_type) ? "flat " : "",
            type, ident);
    }
    mglPdSourceAppendF(&src,
                       "void main() {\n"
                       "    int mgl_base = gl_VertexID * %lu;\n"
                       "    gl_Position = mgl_gs_output.records[mgl_base];\n",
                       (unsigned long)vec4Stride);
    if (hasPointSize) {
        /* Forward the kernel's point size (slot 1.x).  Only emitted when the GS
         * declared gl_PointSize -- the pipeline builder rejects a vertex stage
         * writing point size on a Line/Triangle topology.  Two-step load: the
         * frontend rejects a member access directly on an SSBO array element. */
        mglPdSourceAppendRaw(
            &src, "    vec4 mgl_point_size = mgl_gs_output.records[mgl_base + 1];\n"
                  "    gl_PointSize = mgl_point_size.x;\n");
    }
    if (hasPrimitiveId) {
        /* Two-step load: the frontend rejects a member access directly on an
         * SSBO array element.  The record already holds the float carrier, so
         * forward it unchanged.  Layout v2: primitive_id @64. */
        const unsigned primSlot =
            (unsigned)(MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET / 16u);
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_prim_vec = mgl_gs_output.records[mgl_base + %u];\n"
            "    mgl_primitive_id = mgl_prim_vec.x;\n",
            primSlot);
    }
    if (hasClipDistance) {
        /* Clip distances live at byte offset 64 (vec4 slots 4..5). */
        const unsigned clipSlot =
            (unsigned)(MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET / 16u);
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_clip0 = mgl_gs_output.records[mgl_base + %u];\n"
            "    vec4 mgl_clip1 = mgl_gs_output.records[mgl_base + %u];\n"
            "    gl_ClipDistance[0] = mgl_clip0.x;\n"
            "    gl_ClipDistance[1] = mgl_clip0.y;\n"
            "    gl_ClipDistance[2] = mgl_clip0.z;\n"
            "    gl_ClipDistance[3] = mgl_clip0.w;\n"
            "    gl_ClipDistance[4] = mgl_clip1.x;\n"
            "    gl_ClipDistance[5] = mgl_clip1.y;\n"
            "    gl_ClipDistance[6] = mgl_clip1.z;\n"
            "    gl_ClipDistance[7] = mgl_clip1.w;\n",
            clipSlot, clipSlot + 1u);
    }
    if (getenv("MGL_GS_PROBE")) {
        /* Pixel probe: R = vertex id, G = GPU-read position.y remapped, B =
         * GPU-read varying.r.  Renders the real positions so the geometry stays
         * identifiable. */
        mglPdSourceAppendRaw(
            &src, "    vec4 mgl_probe_pos = mgl_gs_output.records[mgl_base];\n"
                  "    vec4 mgl_probe_col = mgl_gs_output.records[mgl_base + 4];\n"
                  "    gl_Position = mgl_probe_pos;\n"
                  "    gs_fs_color = vec4(float(gl_VertexID) / 6.0,\n"
                  "                        mgl_probe_pos.y * 0.5 + 0.5,\n"
                  "                        abs(mgl_probe_col.r), 1.0);\n"
                  "    return;\n");
    }
    if (getenv("MGL_GS_PROBE_VID")) {
        /* Probe 2: ignore the SSBO entirely; geometry is derived from
         * gl_VertexID alone (six points spread horizontally at mid height).
         * Correct render => vertex ids / draw are sane and the defect is in the
         * SSBO read path. */
        mglPdSourceAppendRaw(
            &src, "    float mgl_vid = float(gl_VertexID);\n"
                  "    gl_Position = vec4(mgl_vid / 3.0 - 1.0, 0.25, 0.0, 1.0);\n"
                  "    gs_fs_color = vec4(mgl_vid / 6.0, 1.0, 0.0, 1.0);\n"
                  "    return;\n");
    }
    if (getenv("MGL_GS_PROBE_WAVE")) {
        /* Probe 3: oscilloscope.  Vertex x is fixed by vid; the polyline y
         * traces records[mgl_base].x as read on the GPU, and color carries
         * .y/.z/.w.  One render reconstructs every slot the passthrough VS
         * actually sees. */
        mglPdSourceAppendRaw(
            &src, "    float mgl_vid = float(gl_VertexID);\n"
                  "    vec4 mgl_p0 = mgl_gs_output.records[mgl_base];\n"
                  "    gl_Position = vec4(mgl_vid / 3.0 - 1.0, mgl_p0.x, 0.0, 1.0);\n"
                  "    gs_fs_color = vec4(mgl_p0.y * 0.5 + 0.5,\n"
                  "                       mgl_p0.z * 0.5 + 0.5,\n"
                  "                       mgl_p0.w * 0.5 + 0.5, 1.0);\n"
                  "    return;\n");
    }
    int fsNeedsLayer = 0;
    int fsNeedsViewport = 0;
    for (GLuint fi = 0; fsInputs && fsInputs->list && fi < fsInputs->count;
         fi++) {
        const MGLShaderResource *in = &fsInputs->list[fi];
        if (!in->name) continue;
        if (strcmp(in->name, "gl_Layer") == 0) fsNeedsLayer = 1;
        if (strcmp(in->name, "gl_ViewportIndex") == 0) fsNeedsViewport = 1;
    }
    {
        const uint32_t fsBuiltins =
            mglProgramStageBuiltinMask(program, _FRAGMENT_SHADER);
        if (!fsNeedsLayer && (fsBuiltins & MGL_AIR_BUILTIN_LAYER) != 0u)
            fsNeedsLayer = 1;
        if (!fsNeedsViewport &&
            (fsBuiltins & MGL_AIR_BUILTIN_VIEWPORT_INDEX) != 0u)
            fsNeedsViewport = 1;
    }
    if (getenv("MGL_PTVS_NO_SPECIALS")) {
        /* Diagnostic: omit the layer/viewport special outputs entirely so the
         * vertex return carries only position + user varyings. */
    } else if ((gsBuiltins &
                (MGL_AIR_BUILTIN_LAYER | MGL_AIR_BUILTIN_VIEWPORT_INDEX)) != 0u ||
               fsNeedsLayer || fsNeedsViewport) {

        /* Layout v2 (A04): layer @52 / viewport @56 share vec4 slot 3 as .y /
         * .z (stream occupies .w).  Emulated MS_ARRAY is flat =
         * layer * stride + sample_plane; the plane comes from the host. */
        const unsigned layerSlot =
            (unsigned)(MGL_AIR_PER_VERTEX_LAYER_OFFSET / 16u);
        if (layerStride > 1u) {
            mglPdSourceAppendF(
                &src,
                "    vec4 mgl_layer_vp = mgl_gs_output.records[mgl_base + %u];\n"
                "    gl_Layer = floatBitsToInt(mgl_layer_vp.y) * %u + "
                "mgl_emulated_ms_plane;\n"
                "    gl_ViewportIndex = floatBitsToInt(mgl_layer_vp.z);\n",
                layerSlot, (unsigned)layerStride);
        } else {
            mglPdSourceAppendF(
                &src,
                "    vec4 mgl_layer_vp = mgl_gs_output.records[mgl_base + %u];\n"
                "    gl_Layer = floatBitsToInt(mgl_layer_vp.y) * %u;\n"
                "    gl_ViewportIndex = floatBitsToInt(mgl_layer_vp.z);\n",
                layerSlot, (unsigned)layerStride);
        }
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        if (output->stream > 0) continue;
        if (mglPdPassthroughSkipBuiltinOutput(output)) continue;
        const GLenum declType = mglPdPassthroughDeclType(fsInputs, output);
        const unsigned matCols = mglPdGeometryPassthroughMatrixCols(declType);
        const unsigned matRows = mglPdGeometryPassthroughMatrixRows(declType);
        char identBuf[32];
        const char *ident =
            mglPdPassthroughIdent(output, (unsigned)i, identBuf, sizeof identBuf);
        if (matCols > 0u) {
            /* Stage-out stores one column per location slot (GL 4.6 4.4.1).
             * Forward each column as its own vector varying. */
            const char *colSwizzle =
                mglPdGeometryPassthroughColumnSwizzle(matRows);
            if (!colSwizzle || !ident) goto done;
            const unsigned baseSlot =
                (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location);
            for (unsigned c = 0; c < matCols; c++) {
                mglPdSourceAppendF(
                    &src,
                    "    vec4 mgl_slot_%u_%u = mgl_gs_output.records[mgl_base + %u];\n"
                    "    %s_c%u = mgl_slot_%u_%u%s;\n",
                    (unsigned)i, c, baseSlot + c, ident, c, (unsigned)i,
                    c, colSwizzle);
            }
            continue;
        }
        const char *swizzle = mglPdGeometryPassthroughSwizzle(declType);
        if (!swizzle || !ident) goto done;
        /* Integer records already hold SIToFP/UIToFP float carriers - forward
         * the float swizzle; do not floatBitsTo*. */
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_slot_%u = mgl_gs_output.records[mgl_base + %u];\n"
            "    %s = mgl_slot_%u%s;\n",
            (unsigned)i,
            (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
            ident, (unsigned)i, swizzle);
    }
    mglPdSourceAppendRaw(&src, "}\n");
    if (getenv("MGL_GS_DIAG")) {
        fprintf(stderr, "MGL GS DIAG passthrough VS source:\n%s\n", src.data);
    }
    {
        unsigned char *bytes = NULL;
        size_t size = 0u;
        char errorText[512] = {0};
        if (mglShaderCompileGLSL(src.data, MGL_STAGE_VERTEX, &bytes, &size,
                                 errorText, sizeof(errorText)) != 0 ||
            !bytes || size == 0u) {
            fprintf(stderr,
                    "MGL GS ERROR: failed to compile AIR passthrough vertex: %s\n",
                    errorText[0] ? errorText : "?");
            mglShaderFree(bytes);
            goto done;
        }
        if (getenv("MGL_DUMP_AIR")) {
            FILE *f = fopen("/tmp/poison_ptvs.air", "wb");
            if (f) {
                fwrite(bytes, 1, size, f);
                fclose(f);
                fprintf(stderr, "MGL DUMP: ptvs.air %zu bytes\n", size);
            }
        }
        void *library = NULL;
        void *function = NULL;
        const bool loaded = mglPdLoadAIRMainFunction(
            bytes, size, &library, &function, errorText, sizeof(errorText));
        mglShaderFree(bytes);
        if (!loaded || !library || !function) {
            fprintf(stderr,
                    "MGL GS ERROR: failed to load AIR passthrough vertex: %s\n",
                    errorText[0] ? errorText : "?");
            goto done;
        }
        result = mglRendererBackendSetPassthroughFunction(
                     areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
                     library, function, passthroughKey) == 0
                     ? 1
                     : 0;
        /* ARC released these at scope exit in the .m twin. */
        mglReleaseMetalObjNoNull(library);
        mglReleaseMetalObjNoNull(function);
    }
done:
    mglPdSourceFree(&src);
    return result;
}

/* -ensureAIRTessEvalPassthroughFunctionForProgram:. */
int mglRenderPassEnsureAIRTessEvalPassthroughFunctionForProgram(void *renderer,
                                                                Program *program)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);

    if (!program) return 0;
    void *cachedFunction = NULL;
    if (mglRendererBackendGetPassthroughFunction(
            areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
            mglProgramBackendView(program)->pipeline_cache_instance_id, &cachedFunction) == 1) {
        return 1;
    }
    (void)mglRendererBackendSetPassthroughFunction(
        areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION, NULL,
        NULL, 0u);

    MGLShaderResourceList *outputs = &program
                                          ->shader_resources_list
                                              [_TESS_EVALUATION_SHADER]
                                              [_STAGE_OUTPUT_RES];
    const uint64_t recordStride = mglAIRPerVertexStrideForResources(outputs);
    const uint64_t vec4Stride = recordStride / 16u;
    const int hasClipDistance =
        (mglProgramStageBuiltinMask(program, _TESS_EVALUATION_SHADER) &
         MGL_AIR_BUILTIN_CLIP_DISTANCE) != 0u;
    /* Isolines rasterize as lines; Metal rejects a vertex stage that writes
     * point size on a non-point topology. */
    const int writePointSize =
        mglTessWritePointSize((uint32_t)program->tess_gen_point_mode) != 0;
    MglPdSource src;
    mglPdSourceInit(&src);
    mglPdSourceAppendRaw(&src,
                         "#version 460 core\n"
                         "layout(std430, binding = 0) buffer MGLTESOutput {\n"
                         "    vec4 records[];\n"
                         "} mgl_tes_output;\n");
    int result = 0;
    if (hasClipDistance) {
        mglPdSourceAppendF(&src, "out float gl_ClipDistance[%u];\n",
                           (unsigned)MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT);
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        if (mglPdPassthroughSkipBuiltinOutput(output)) continue;
        /* Integer varyings are stored as float carriers in the TES record (same
         * ABI as GS expansion); declare float attributes and forward the
         * swizzle - FS converts with fptosi/fptoui. */
        const unsigned matCols = mglPdGeometryPassthroughMatrixCols(
            output->gl_type);
        const unsigned matRows = mglPdGeometryPassthroughMatrixRows(
            output->gl_type);
        char identBuf[32];
        const char *ident =
            mglPdPassthroughIdent(output, (unsigned)i, identBuf, sizeof identBuf);
        if (matCols > 0u) {
            const char *colType = mglPdGeometryPassthroughColumnType(matRows);
            if (!colType || !ident) {
                fprintf(stderr,
                        "MGL TESS ERROR: unsupported passthrough matrix type 0x%x\n",
                        (unsigned)output->gl_type);
                goto done;
            }
            for (unsigned c = 0; c < matCols; c++) {
                mglPdSourceAppendF(
                    &src, "layout(location = %u) out %s %s_c%u;\n",
                    (unsigned)(output->location + c), colType, ident, c);
            }
            continue;
        }
        const char *type =
            mglPdGeometryPassthroughNeedsFlat(output->gl_type)
                ? mglPdGeometryPassthroughFloatType(output->gl_type)
                : mglPdGeometryPassthroughType(output->gl_type);
        if (!type || !ident) {
            fprintf(stderr,
                    "MGL TESS ERROR: unsupported passthrough varying type 0x%x\n",
                    (unsigned)output->gl_type);
            goto done;
        }
        mglPdSourceAppendF(
            &src, "layout(location = %u) %sout %s %s;\n",
            (unsigned)output->location,
            mglPdGeometryPassthroughNeedsFlat(output->gl_type) ? "flat " : "",
            type, ident);
    }
    mglPdSourceAppendF(&src,
                       "void main() {\n"
                       "    int mgl_base = gl_VertexID * %lu;\n"
                       "    gl_Position = mgl_tes_output.records[mgl_base];\n",
                       (unsigned long)vec4Stride);
    if (writePointSize) {
        mglPdSourceAppendRaw(
            &src, "    vec4 mgl_point_size = mgl_tes_output.records[mgl_base + 1];\n"
                  "    gl_PointSize = mgl_point_size.x;\n");
    }
    if (hasClipDistance) {
        const unsigned clipSlot =
            (unsigned)(MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET / 16u);
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_clip0 = mgl_tes_output.records[mgl_base + %u];\n"
            "    vec4 mgl_clip1 = mgl_tes_output.records[mgl_base + %u];\n"
            "    gl_ClipDistance[0] = mgl_clip0.x;\n"
            "    gl_ClipDistance[1] = mgl_clip0.y;\n"
            "    gl_ClipDistance[2] = mgl_clip0.z;\n"
            "    gl_ClipDistance[3] = mgl_clip0.w;\n"
            "    gl_ClipDistance[4] = mgl_clip1.x;\n"
            "    gl_ClipDistance[5] = mgl_clip1.y;\n"
            "    gl_ClipDistance[6] = mgl_clip1.z;\n"
            "    gl_ClipDistance[7] = mgl_clip1.w;\n",
            clipSlot, clipSlot + 1u);
    }
    if (program->tess_cull_distance_count > 0u) {
        const int isolines =
            mglTessGenModeIsIsolines((uint32_t)program->tess_gen_mode) != 0;
        if (isolines) {
            /* Both endpoints of an isoline segment share the same v, so the
             * cull condition needs the partner record's distances.  The partner
             * record index is (gl_VertexID ^ 1) -- every patch span holds an
             * even item count. */
            mglPdSourceAppendF(
                &src,
                "    int mgl_partner = (gl_VertexID ^ 1) * %lu;\n"
                "    vec4 mgl_p0 = mgl_tes_output.records[mgl_partner + 1];\n"
                "    vec4 mgl_p1 = mgl_tes_output.records[mgl_partner + 2];\n"
                "    vec4 mgl_p2 = mgl_tes_output.records[mgl_partner + 3];\n",
                (unsigned long)vec4Stride);
        }
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_c0 = mgl_tes_output.records[mgl_base + 1];\n"
            "    vec4 mgl_c1 = mgl_tes_output.records[mgl_base + 2];\n"
            "    vec4 mgl_c2 = mgl_tes_output.records[mgl_base + 3];\n"
            "    bool mgl_culled = false\n"
            "%s"
            "    if (mgl_culled) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);\n",
            isolines ? "        || (mgl_c0.y < 0.0 && mgl_p0.y < 0.0)\n"
                       "        || (mgl_c0.z < 0.0 && mgl_p0.z < 0.0)\n"
                       "        || (mgl_c0.w < 0.0 && mgl_p0.w < 0.0)\n"
                       "        || (mgl_c1.x < 0.0 && mgl_p1.x < 0.0)\n"
                       "        || (mgl_c1.y < 0.0 && mgl_p1.y < 0.0)\n"
                       "        || (mgl_c1.z < 0.0 && mgl_p1.z < 0.0)\n"
                       "        || (mgl_c1.w < 0.0 && mgl_p1.w < 0.0)\n"
                       "        || (mgl_c2.x < 0.0 && mgl_p2.x < 0.0);\n"
                     : "        || mgl_c0.y < 0.0\n"
                       "        || mgl_c0.z < 0.0\n"
                       "        || mgl_c0.w < 0.0\n"
                       "        || mgl_c1.x < 0.0\n"
                       "        || mgl_c1.y < 0.0\n"
                       "        || mgl_c1.z < 0.0\n"
                       "        || mgl_c1.w < 0.0\n"
                       "        || mgl_c2.x < 0.0;\n");
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        if (mglPdPassthroughSkipBuiltinOutput(output)) continue;
        char identBuf[32];
        const char *ident =
            mglPdPassthroughIdent(output, (unsigned)i, identBuf, sizeof identBuf);
        const unsigned matCols = mglPdGeometryPassthroughMatrixCols(
            output->gl_type);
        const unsigned matRows = mglPdGeometryPassthroughMatrixRows(
            output->gl_type);
        if (matCols > 0u) {
            const char *colSwizzle =
                mglPdGeometryPassthroughColumnSwizzle(matRows);
            if (!colSwizzle || !ident) goto done;
            const unsigned baseSlot =
                (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location);
            for (unsigned c = 0; c < matCols; c++) {
                mglPdSourceAppendF(
                    &src,
                    "    vec4 mgl_slot_%u_%u = mgl_tes_output.records[mgl_base + %u];\n"
                    "    %s_c%u = mgl_slot_%u_%u%s;\n",
                    (unsigned)i, c, baseSlot + c, ident, c, (unsigned)i,
                    c, colSwizzle);
            }
            continue;
        }
        const char *swizzle = mglPdGeometryPassthroughSwizzle(output->gl_type);
        if (!swizzle || !ident) goto done;
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_slot_%u = mgl_tes_output.records[mgl_base + %u];\n"
            "    %s = mgl_slot_%u%s;\n",
            (unsigned)i,
            (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
            ident, (unsigned)i, swizzle);
    }
    mglPdSourceAppendRaw(&src, "}\n");
    if (getenv("MGL_GS_DIAG")) {
        fprintf(stderr, "MGL GS DIAG passthrough VS source:\n%s\n", src.data);
    }
    {
        unsigned char *bytes = NULL;
        size_t size = 0u;
        char errorText[512] = {0};
        if (mglShaderCompileGLSL(src.data, MGL_STAGE_VERTEX, &bytes, &size,
                                 errorText, sizeof(errorText)) != 0 ||
            !bytes || size == 0u) {
            fprintf(stderr,
                    "MGL TESS ERROR: failed to compile AIR TES passthrough vertex: %s\nSOURCE:\n%s\n",
                    errorText[0] ? errorText : "?", src.data);
            mglShaderFree(bytes);
            goto done;
        }
        void *library = NULL;
        void *function = NULL;
        const bool loaded = mglPdLoadAIRMainFunction(
            bytes, size, &library, &function, errorText, sizeof(errorText));
        mglShaderFree(bytes);
        if (!loaded || !library || !function) {
            fprintf(stderr,
                    "MGL TESS ERROR: failed to load AIR TES passthrough vertex: %s\n",
                    errorText[0] ? errorText : "?");
            goto done;
        }
        result =
            mglRendererBackendSetPassthroughFunction(
                areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
                library, function, mglProgramBackendView(program)->pipeline_cache_instance_id) == 0
                ? 1
                : 0;
        /* ARC released these at scope exit in the .m twin. */
        mglReleaseMetalObjNoNull(library);
        mglReleaseMetalObjNoNull(function);
    }
done:
    mglPdSourceFree(&src);
    return result;
}
