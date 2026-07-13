/*
 * msl_patch_pipeline.c
 * MGL
 *
 * Implementation of the MSL Patch Pipeline Subsystem.
 * See msl_patch_pipeline.h for the API contract.
 *
 * The pipeline owns the MSL string and runs registered patch steps in
 * order.  Before each step, the MSL is snapshotted (strdup).  If the step
 * returns GL_FALSE, the snapshot is restored (the failed step's changes
 * are discarded) and a warning is logged.  This gives per-step rollback
 * semantics that the individual patch functions do not all provide
 * themselves.
 */

#include "msl_patch_pipeline.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* P1-6: extern declaration — mglResolveProgramForStageFromState is defined
 * in MGLRenderer.m (declared in MGLRenderer+RenderPass_Private.h).  We
 * avoid pulling in that ObjC header here to keep this translation unit C-only. */
extern Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);

/* === Per-stage pipeline === */

GLboolean mslPipelineInit(MSLPatchPipeline *pipeline,
                          Program *program,
                          int stage,
                          char *initialMSL)
{
    if (!pipeline) {
        return GL_FALSE;
    }

    pipeline->steps = NULL;
    pipeline->count = 0;
    pipeline->capacity = 0;
    pipeline->msl = initialMSL;  /* take ownership */
    pipeline->ctx.program = program;
    pipeline->ctx.stage = stage;
    pipeline->failed_step = -1;

    return GL_TRUE;
}

GLboolean mslPipelineAddStep(MSLPatchPipeline *pipeline,
                             const char *name,
                             MSLPatchFn fn)
{
    if (!pipeline || !name || !fn) {
        return GL_FALSE;
    }

    if (pipeline->count >= pipeline->capacity) {
        int newCapacity = pipeline->capacity == 0 ? 8 : pipeline->capacity * 2;
        MSLPatchStep *newSteps = (MSLPatchStep *)realloc(
            pipeline->steps,
            (size_t)newCapacity * sizeof(MSLPatchStep));
        if (!newSteps) {
            return GL_FALSE;
        }
        pipeline->steps = newSteps;
        pipeline->capacity = newCapacity;
    }

    pipeline->steps[pipeline->count].name = name;
    pipeline->steps[pipeline->count].patch_fn = fn;
    pipeline->steps[pipeline->count].enabled = GL_TRUE;
    pipeline->count++;

    return GL_TRUE;
}

GLboolean mslPipelineRun(MSLPatchPipeline *pipeline)
{
    if (!pipeline) {
        return GL_FALSE;
    }

    GLboolean allOk = GL_TRUE;

    for (int i = 0; i < pipeline->count; i++) {
        if (!pipeline->steps[i].enabled) {
            continue;
        }

        /* Snapshot MSL before the step for rollback. */
        char *snapshot = NULL;
        if (pipeline->msl) {
            snapshot = strdup(pipeline->msl);
            if (!snapshot) {
                /* Can't snapshot — log and skip the step, keeping current MSL. */
                fprintf(stderr,
                        "MGL MSL PIPELINE: step '%s' skipped (snapshot alloc failed)\n",
                        pipeline->steps[i].name);
                if (pipeline->failed_step < 0) {
                    pipeline->failed_step = i;
                }
                allOk = GL_FALSE;
                continue;
            }
        }

        GLboolean ok = pipeline->steps[i].patch_fn(&pipeline->ctx, &pipeline->msl);

        if (!ok) {
            /* Step failed — roll back to snapshot. */
            fprintf(stderr,
                    "MGL MSL PIPELINE: step '%s' failed, rolling back to pre-step MSL\n",
                    pipeline->steps[i].name);
            if (pipeline->msl) {
                free(pipeline->msl);
            }
            pipeline->msl = snapshot;  /* restore pre-step MSL */
            snapshot = NULL;           /* pipeline owns it now */
            if (pipeline->failed_step < 0) {
                pipeline->failed_step = i;
            }
            allOk = GL_FALSE;
        } else if (pipeline->msl == NULL) {
            /* Step succeeded but nulled the MSL — treat as failure. */
            fprintf(stderr,
                    "MGL MSL PIPELINE: step '%s' left MSL NULL, rolling back\n",
                    pipeline->steps[i].name);
            pipeline->msl = snapshot;
            snapshot = NULL;
            if (pipeline->failed_step < 0) {
                pipeline->failed_step = i;
            }
            allOk = GL_FALSE;
        }

        if (snapshot) {
            free(snapshot);
        }
    }

    return allOk;
}

char *mslPipelineTakeResult(MSLPatchPipeline *pipeline)
{
    if (!pipeline) {
        return NULL;
    }

    char *result = pipeline->msl;
    pipeline->msl = NULL;
    return result;
}

void mslPipelineDestroy(MSLPatchPipeline *pipeline)
{
    if (!pipeline) {
        return;
    }

    if (pipeline->msl) {
        free(pipeline->msl);
        pipeline->msl = NULL;
    }

    if (pipeline->steps) {
        free(pipeline->steps);
        pipeline->steps = NULL;
    }

    pipeline->count = 0;
    pipeline->capacity = 0;
    pipeline->failed_step = -1;
}

/* === Post-link pipeline === */

GLboolean mslPipelinePostLinkInit(MSLPatchPipelinePostLink *pipeline,
                                  Program *program)
{
    if (!pipeline) {
        return GL_FALSE;
    }

    pipeline->steps = NULL;
    pipeline->count = 0;
    pipeline->capacity = 0;
    pipeline->program = program;
    pipeline->failed_step = -1;

    return GL_TRUE;
}

GLboolean mslPipelinePostLinkAddStep(MSLPatchPipelinePostLink *pipeline,
                                     const char *name,
                                     MSLPatchFnPostLink fn)
{
    if (!pipeline || !name || !fn) {
        return GL_FALSE;
    }

    if (pipeline->count >= pipeline->capacity) {
        int newCapacity = pipeline->capacity == 0 ? 8 : pipeline->capacity * 2;
        MSLPatchStepPostLink *newSteps = (MSLPatchStepPostLink *)realloc(
            pipeline->steps,
            (size_t)newCapacity * sizeof(MSLPatchStepPostLink));
        if (!newSteps) {
            return GL_FALSE;
        }
        pipeline->steps = newSteps;
        pipeline->capacity = newCapacity;
    }

    pipeline->steps[pipeline->count].name = name;
    pipeline->steps[pipeline->count].patch_fn = fn;
    pipeline->steps[pipeline->count].enabled = GL_TRUE;
    pipeline->count++;

    return GL_TRUE;
}

GLboolean mslPipelinePostLinkRun(MSLPatchPipelinePostLink *pipeline)
{
    if (!pipeline) {
        return GL_FALSE;
    }

    GLboolean allOk = GL_TRUE;

    for (int i = 0; i < pipeline->count; i++) {
        if (!pipeline->steps[i].enabled) {
            continue;
        }

        /* P2-10: Snapshot all stages' MSL strings before the step for
         * rollback.  Post-link patches may touch multiple stages' MSL;
         * if a step fails, restore the snapshots so subsequent steps see
         * the pre-failure MSL (matching per-stage pipeline semantics).
         * The cost is at most 6 strdup+free per step — negligible vs the
         * actual MSL patching work. */
        char *snapshots[_MAX_SHADER_TYPES];
        int snap_count = 0;
        Program *prog = pipeline->program;
        if (prog) {
            for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
                snapshots[s] = NULL;
                char *msl = prog->spirv[s].msl_str;
                if (msl) {
                    snapshots[s] = strdup(msl);
                    if (snapshots[s]) {
                        snap_count++;
                    }
                }
            }
        }

        GLboolean ok = pipeline->steps[i].patch_fn(pipeline->program);

        if (!ok) {
            fprintf(stderr,
                    "MGL MSL PIPELINE: post-link step '%s' failed, rolling back MSL\n",
                    pipeline->steps[i].name);
            if (pipeline->failed_step < 0) {
                pipeline->failed_step = i;
            }
            allOk = GL_FALSE;

            /* Restore MSL snapshots so the next step sees pre-failure MSL. */
            if (prog) {
                for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
                    if (snapshots[s]) {
                        free(prog->spirv[s].msl_str);
                        prog->spirv[s].msl_str = snapshots[s];
                        snapshots[s] = NULL;  /* transferred to program */
                    }
                }
            }
        }

        /* Free any snapshots not consumed by rollback. */
        for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
            if (snapshots[s]) {
                free(snapshots[s]);
                snapshots[s] = NULL;
            }
        }
    }

    return allOk;
}

void mslPipelinePostLinkDestroy(MSLPatchPipelinePostLink *pipeline)
{
    if (!pipeline) {
        return;
    }

    if (pipeline->steps) {
        free(pipeline->steps);
        pipeline->steps = NULL;
    }

    pipeline->count = 0;
    pipeline->capacity = 0;
    pipeline->failed_step = -1;
}

/* === TCS stage-in struct parsing (P1-6: migrated from MGLRenderer.m) === */

static bool mglTCSStageInParseAttributeMarker(const char *member, GLuint *outAttribute)
{
    if (!member || !outAttribute) {
        return false;
    }
    const char *marker = strstr(member, "/*mgl_attribute(");
    if (marker) {
        marker += strlen("/*mgl_attribute(");
        char *end = NULL;
        unsigned long value = strtoul(marker, &end, 10);
        if (end && end != marker && value < MAX_ATTRIBS) {
            *outAttribute = (GLuint)value;
            return true;
        }
    }

    const char *attr = strstr(member, "[[attribute(");
    if (attr) {
        attr += strlen("[[attribute(");
        char *end = NULL;
        unsigned long value = strtoul(attr, &end, 10);
        if (end && end != attr && value < MAX_ATTRIBS) {
            *outAttribute = (GLuint)value;
            return true;
        }
    }
    return false;
}

static void mglTCSStageInDescribeMember(const char *member,
                                        size_t *outSize,
                                        size_t *outAlign,
                                        size_t *outComponentBytes,
                                        GLuint *outComponents,
                                        MGLTCSStageInBaseType *outBaseType)
{
    size_t memberSize = 16u;
    size_t memberAlign = 16u;
    size_t componentBytes = 4u;
    GLuint components = 4u;
    MGLTCSStageInBaseType baseType = MGLTCSStageInBaseFloat;

    if (!member) {
        goto done;
    }

    if (strstr(member, "uint") || strstr(member, "uchar") || strstr(member, "ushort")) {
        baseType = MGLTCSStageInBaseUInt;
    } else if (strstr(member, "int") || strstr(member, "char") || strstr(member, "short")) {
        baseType = MGLTCSStageInBaseInt;
    }

    if      (strstr(member, "float4"))  { memberSize = 16; memberAlign = 16; componentBytes = 4; components = 4; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "float3"))  { memberSize = 16; memberAlign = 16; componentBytes = 4; components = 3; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "float2"))  { memberSize =  8; memberAlign =  8; componentBytes = 4; components = 2; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "float"))   { memberSize =  4; memberAlign =  4; componentBytes = 4; components = 1; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "double4")) { memberSize = 32; memberAlign = 32; componentBytes = 8; components = 4; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "double3")) { memberSize = 32; memberAlign = 32; componentBytes = 8; components = 3; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "double2")) { memberSize = 16; memberAlign = 16; componentBytes = 8; components = 2; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "double"))  { memberSize =  8; memberAlign =  8; componentBytes = 8; components = 1; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "half4"))   { memberSize =  8; memberAlign =  8; componentBytes = 2; components = 4; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "half3"))   { memberSize =  8; memberAlign =  8; componentBytes = 2; components = 3; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "half2"))   { memberSize =  4; memberAlign =  4; componentBytes = 2; components = 2; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "half"))    { memberSize =  2; memberAlign =  2; componentBytes = 2; components = 1; baseType = MGLTCSStageInBaseFloat; }
    else if (strstr(member, "4") && (strstr(member, "int") || strstr(member, "char"))) { memberSize = 16; memberAlign = 16; componentBytes = 4; components = 4; }
    else if (strstr(member, "3") && (strstr(member, "int") || strstr(member, "char"))) { memberSize = 16; memberAlign = 16; componentBytes = 4; components = 3; }
    else if (strstr(member, "2") && (strstr(member, "int") || strstr(member, "char"))) { memberSize =  8; memberAlign =  8; componentBytes = 4; components = 2; }
    else if (strstr(member, "int") || strstr(member, "uint") || strstr(member, "char") || strstr(member, "uchar")) { memberSize = 4; memberAlign = 4; componentBytes = 4; components = 1; }
    else if (strstr(member, "short4") || strstr(member, "ushort4")) { memberSize = 8; memberAlign = 8; componentBytes = 2; components = 4; }
    else if (strstr(member, "short3") || strstr(member, "ushort3")) { memberSize = 8; memberAlign = 8; componentBytes = 2; components = 3; }
    else if (strstr(member, "short2") || strstr(member, "ushort2")) { memberSize = 4; memberAlign = 4; componentBytes = 2; components = 2; }
    else if (strstr(member, "short")  || strstr(member, "ushort"))  { memberSize = 2; memberAlign = 2; componentBytes = 2; components = 1; }
    else if (strstr(member, "bool"))   { memberSize = 1; memberAlign = 1; componentBytes = 1; components = 1; baseType = MGLTCSStageInBaseUInt; }

done:
    if (outSize) *outSize = memberSize;
    if (outAlign) *outAlign = memberAlign;
    if (outComponentBytes) *outComponentBytes = componentBytes;
    if (outComponents) *outComponents = components;
    if (outBaseType) *outBaseType = baseType;
}

size_t mglParseTCSStageInMembers(const char *msl,
                                     MGLTCSStageInMember *members,
                                     size_t capacity,
                                     size_t *outStride)
{
    if (outStride) {
        *outStride = 0u;
    }
    if (!msl) {
        return 0u;
    }

    const char *cursor = msl;
    while ((cursor = strstr(cursor, "struct ")) != NULL) {
        cursor += 7;
        while (*cursor == ' ' || *cursor == '\t') {
            cursor++;
        }
        const char *nameStart = cursor;
        while (*cursor && *cursor != ' ' && *cursor != '\t' &&
               *cursor != '\n' && *cursor != '\r' && *cursor != '{') {
            cursor++;
        }
        size_t nameLen = (size_t)(cursor - nameStart);
        if (nameLen <= 3u || strncmp(nameStart + nameLen - 3u, "_in", 3u) != 0) {
            continue;
        }

        const char *braceStart = strchr(cursor, '{');
        if (!braceStart) {
            continue;
        }
        const char *braceEnd = braceStart + 1;
        int depth = 1;
        while (*braceEnd && depth > 0) {
            if (*braceEnd == '{') depth++;
            else if (*braceEnd == '}') depth--;
            braceEnd++;
        }
        if (depth != 0) {
            continue;
        }

        size_t running = 0u;
        size_t maxAlign = 1u;
        size_t memberCount = 0u;
        const char *p = braceStart + 1;
        while (p < braceEnd - 1) {
            const char *semi = p;
            while (semi < braceEnd - 1 && *semi != ';') {
                semi++;
            }
            if (semi >= braceEnd - 1) {
                break;
            }

            const char *mp = p;
            while (mp < semi && isspace((unsigned char)*mp)) {
                mp++;
            }
            size_t mlen = (size_t)(semi - mp);
            if (mlen > 0u) {
                char member[512];
                if (mlen >= sizeof(member)) {
                    mlen = sizeof(member) - 1u;
                }
                memcpy(member, mp, mlen);
                member[mlen] = '\0';

                size_t arrayCount = 1u;
                char *arr = strchr(member, '[');
                if (arr) {
                    *arr = '\0';
                    unsigned long cnt = strtoul(arr + 1, NULL, 10);
                    if (cnt > 0u) {
                        arrayCount = (size_t)cnt;
                    }
                }

                size_t memberSize = 0u;
                size_t memberAlign = 0u;
                size_t componentBytes = 0u;
                GLuint components = 0u;
                MGLTCSStageInBaseType baseType = MGLTCSStageInBaseFloat;
                mglTCSStageInDescribeMember(member,
                                            &memberSize,
                                            &memberAlign,
                                            &componentBytes,
                                            &components,
                                            &baseType);
                if (memberAlign == 0u) {
                    memberAlign = 1u;
                }
                if (memberAlign > maxAlign) {
                    maxAlign = memberAlign;
                }
                running = (running + memberAlign - 1u) & ~(memberAlign - 1u);

                GLuint attribute = 0u;
                if (mglTCSStageInParseAttributeMarker(mp, &attribute) && memberCount < capacity) {
                    members[memberCount].attribute = attribute;
                    members[memberCount].offset = running;
                    members[memberCount].size = memberSize * arrayCount;
                    members[memberCount].componentBytes = componentBytes;
                    members[memberCount].components = components;
                    members[memberCount].baseType = baseType;
                    memberCount++;
                }

                running += memberSize * arrayCount;
            }
            p = semi + 1;
        }

        running = (running + maxAlign - 1u) & ~(maxAlign - 1u);
        if (outStride) {
            *outStride = running;
        }
        return memberCount;
    }
    return 0u;
}

void mglWriteTCSStageInComponent(uint8_t *dst,
                                 const MGLTCSStageInMember *member,
                                 size_t component,
                                 double value)
{
    if (!dst || !member || component >= member->components) {
        return;
    }
    uint8_t *componentDst = dst + member->offset + (component * member->componentBytes);
    size_t copyBytes = (member->componentBytes < sizeof(int32_t))
                       ? member->componentBytes : sizeof(int32_t);
    switch (member->baseType) {
        case MGLTCSStageInBaseInt: {
            int32_t v = (int32_t)value;
            memcpy(componentDst, &v, copyBytes);
            break;
        }
        case MGLTCSStageInBaseUInt: {
            uint32_t v = (value < 0.0) ? 0u : (uint32_t)value;
            memcpy(componentDst, &v, copyBytes);
            break;
        }
        case MGLTCSStageInBaseFloat:
        default: {
            float v = (float)value;
            memcpy(componentDst, &v, copyBytes);
            break;
        }
    }
}

/* === Tessellation passthrough detection (P1-6: migrated from MGLRenderer.m) === */

static bool mglShaderSourceContainsAny(const char *src, const char *const *needles, size_t count)
{
    if (!src) {
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (needles[i] && strstr(src, needles[i])) {
            return true;
        }
    }
    return false;
}

bool mglTessControlUnitPassthroughForPatchSize(const char *tcs, GLuint patchVertices)
{
    const char *outer0[] = {
        "gl_TessLevelOuter[0] = 1.0",
        "gl_TessLevelOuter[0]=1.0"
    };
    const char *outer1[] = {
        "gl_TessLevelOuter[1] = 1.0",
        "gl_TessLevelOuter[1]=1.0"
    };
    const char *outer2[] = {
        "gl_TessLevelOuter[2] = 1.0",
        "gl_TessLevelOuter[2]=1.0"
    };
    const char *outer3[] = {
        "gl_TessLevelOuter[3] = 1.0",
        "gl_TessLevelOuter[3]=1.0"
    };
    const char *inner0[] = {
        "gl_TessLevelInner[0] = 1.0",
        "gl_TessLevelInner[0]=1.0"
    };
    const char *inner1[] = {
        "gl_TessLevelInner[1] = 1.0",
        "gl_TessLevelInner[1]=1.0"
    };
    const char *positionCopy[] = {
        "gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position",
        "gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position"
    };

    if (!tcs ||
        !mglShaderSourceContainsAny(tcs, outer0, sizeof(outer0) / sizeof(outer0[0])) ||
        !mglShaderSourceContainsAny(tcs, outer1, sizeof(outer1) / sizeof(outer1[0])) ||
        !mglShaderSourceContainsAny(tcs, positionCopy, sizeof(positionCopy) / sizeof(positionCopy[0]))) {
        return false;
    }

    if (patchVertices >= 3u &&
        (!mglShaderSourceContainsAny(tcs, inner0, sizeof(inner0) / sizeof(inner0[0])) ||
         !mglShaderSourceContainsAny(tcs, outer2, sizeof(outer2) / sizeof(outer2[0])))) {
        return false;
    }
    if (patchVertices >= 4u &&
        (!mglShaderSourceContainsAny(tcs, inner1, sizeof(inner1) / sizeof(inner1[0])) ||
         !mglShaderSourceContainsAny(tcs, outer3, sizeof(outer3) / sizeof(outer3[0])))) {
        return false;
    }

    switch (patchVertices) {
        case 1u:
            return strstr(tcs, "layout(vertices = 1) out") ||
                   strstr(tcs, "layout(vertices=1) out") ||
                   strstr(tcs, "layout (vertices = 1) out") ||
                   strstr(tcs, "layout (vertices=1) out");
        case 2u:
            return strstr(tcs, "layout(vertices = 2) out") ||
                   strstr(tcs, "layout(vertices=2) out") ||
                   strstr(tcs, "layout (vertices = 2) out") ||
                   strstr(tcs, "layout (vertices=2) out");
        case 3u:
            return strstr(tcs, "layout(vertices = 3) out") ||
                   strstr(tcs, "layout(vertices=3) out") ||
                   strstr(tcs, "layout (vertices = 3) out") ||
                   strstr(tcs, "layout (vertices=3) out");
        default:
            return false;
    }
}

bool mglTessEvalUnitPassthroughForPatchSize(const char *tes, GLuint patchVertices)
{
    if (!tes) {
        return false;
    }

    const char *sideEffectNeedles[] = {
        "imageStore",
        "atomic",
        "barrier(",
        "memoryBarrier",
        "texture(",
        "texelFetch",
        "discard"
    };
    if (mglShaderSourceContainsAny(tes, sideEffectNeedles,
                                   sizeof(sideEffectNeedles) / sizeof(sideEffectNeedles[0]))) {
        return false;
    }

    switch (patchVertices) {
        case 1u:
            return strstr(tes, "gl_Position = gl_in[0].gl_Position") &&
                   !strstr(tes, "gl_TessCoord");
        case 2u:
            return strstr(tes, "gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x)") ||
                   strstr(tes, "gl_Position=mix(gl_in[0].gl_Position,gl_in[1].gl_Position,gl_TessCoord.x)") ||
                   (strstr(tes, "gl_in[0].gl_Position * (1.0 - gl_TessCoord.x)") &&
                    strstr(tes, "gl_in[1].gl_Position * gl_TessCoord.x"));
        case 3u:
            return strstr(tes, "gl_in[0].gl_Position * gl_TessCoord.x") &&
                   strstr(tes, "gl_in[1].gl_Position * gl_TessCoord.y") &&
                   strstr(tes, "gl_in[2].gl_Position * gl_TessCoord.z") &&
                   strstr(tes, "gl_Position =");
        default:
            return false;
    }
}

bool mglTessellationShadersArePassthrough(Program *program, GLuint patchVertices)
{
    const char *tcs = (program && program->shader_slots[_TESS_CONTROL_SHADER])
        ? program->shader_slots[_TESS_CONTROL_SHADER]->src
        : NULL;
    const char *tes = (program && program->shader_slots[_TESS_EVALUATION_SHADER])
        ? program->shader_slots[_TESS_EVALUATION_SHADER]->src
        : NULL;
    if (!tcs || !tes) {
        return false;
    }

    return mglTessControlUnitPassthroughForPatchSize(tcs, patchVertices) &&
           mglTessEvalUnitPassthroughForPatchSize(tes, patchVertices) &&
           !strstr(tcs, "gl_PrimitiveID") &&
           !strstr(tes, "gl_PrimitiveID") &&
           !strstr(tcs, "gl_Layer") &&
           !strstr(tes, "gl_Layer") &&
           !strstr(tcs, "gl_ViewportIndex") &&
           !strstr(tes, "gl_ViewportIndex");
}

bool mglResolvePassthroughPatchModeForContext(GLMContext drawCtx,
                                              GLenum *mode,
                                              const char *label)
{
    if (!drawCtx || !mode || *mode != GL_PATCHES) {
        return false;
    }

    GLuint patchVertices = drawCtx->state.var.patch_vertices;
    GLenum passthroughMode = GL_PATCHES;
    switch (patchVertices) {
        case 1u: passthroughMode = GL_POINTS; break;
        case 2u: passthroughMode = GL_LINES; break;
        case 3u: passthroughMode = GL_TRIANGLES; break;
        default: return false;
    }

    Program *tessProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_CONTROL_SHADER);
    if (!tessProgram) {
        tessProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_EVALUATION_SHADER);
    }
    if (!mglTessellationShadersArePassthrough(tessProgram, patchVertices)) {
        return false;
    }

    static uint64_t s_passthroughTessDrawSkipCount = 0;
    uint64_t hit = ++s_passthroughTessDrawSkipCount;
    if (hit <= 16ull || (hit % 512ull) == 0ull) {
        fprintf(stderr,
                "MGL INFO: Drawing passthrough tessellation label=%s as primitive mode=0x%x hit=%llu\n",
                label ? label : "(unknown)",
                (unsigned)passthroughMode,
                (unsigned long long)hit);
    }
    *mode = passthroughMode;
    return true;
}
