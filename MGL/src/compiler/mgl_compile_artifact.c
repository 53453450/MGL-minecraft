/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * CompileArtifact owner helpers (ARCHITECTURE_AUDIT R2).
 */

#include "mgl_compile_artifact.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_air_reflect.h"
#include "mgl_shader_abi.h"
#include "mgl_buffer_slots.h"
#include "mgl_glsl_parser.h"
#include "mgl_glsl_ast.h"
#include "mgl_frontend_session.h"
#include "glm_limits.h"

void mglCompileArtifactInit(MGLCompileArtifact *art)
{
    if (!art) {
        return;
    }
    memset(art, 0, sizeof(*art));
    mglResourceLayoutPlanInitDefault(&art->layout);
}

void mglCompileArtifactDestroy(MGLCompileArtifact *art)
{
    if (!art) {
        return;
    }
    free(art->frontend.diagnostics);
    art->frontend.diagnostics = NULL;
    free(art->metallib_bytes);
    art->metallib_bytes = NULL;
    art->metallib_size = 0;
    mglGLSLTranslationUnitDestroy(art->tu);
    art->tu = NULL;
    mglAirReflectDestroy(art->resources);
    memset(art->resources, 0, sizeof(art->resources));
    art->complete = 0;
}

MGLCompileArtifact *mglCompileArtifactCreate(void)
{
    MGLCompileArtifact *art =
        (MGLCompileArtifact *)calloc(1, sizeof(MGLCompileArtifact));
    if (art)
        mglCompileArtifactInit(art);
    return art;
}

void mglCompileArtifactFree(MGLCompileArtifact *art)
{
    if (!art)
        return;
    mglCompileArtifactDestroy(art);
    free(art);
}

int mglCompileArtifactCanReuseAtLink(const MGLCompileArtifact *art,
                                     int air_stage,
                                     uint32_t air_flags,
                                     const void *iface_peers,
                                     const char *const *attrib_names)
{
    if (!art || !art->complete || !art->metallib_bytes ||
        art->metallib_size == 0 || art->frontend.stage != air_stage) {
        return 0;
    }
    if (air_flags != 0u || iface_peers != NULL)
        return 0;
    /* Vertex attrib location names force a fresh compile so reflection
     * locations match BindAttribLocation. */
    if (air_stage == (int)MGL_STAGE_VERTEX && attrib_names) {
        for (int i = 0; i < MAX_ATTRIBS; i++) {
            if (attrib_names[i] && attrib_names[i][0])
                return 0;
        }
    }
    return 1;
}

int mglCompileArtifactFromPlan(const MGLCompilePlan *plan,
                               MGLCompileArtifact *art_out,
                               char *err_buf, size_t err_cap)
{
    if (!plan) {
        return -1;
    }
    return mglCompileArtifactFromGLSLEx(
        plan->source, plan->stage, plan->attrib_names, plan->air_flags,
        plan->iface_peers, plan->tess_patch_vertices,
        plan->sparse_residency_buffer_index, art_out, err_buf, err_cap);
}

int mglCompileArtifactFromGLSLEx(const char *src, int stage,
                                 const char *const *attrib_names,
                                 uint32_t air_flags,
                                 const void *iface_peers,
                                 uint32_t tess_patch_vertices,
                                 uint32_t sparse_residency_buffer_index,
                                 MGLCompileArtifact *art_out,
                                 char *err_buf, size_t err_cap)
{
    if (!art_out) {
        return -1;
    }
    mglCompileArtifactDestroy(art_out);
    mglCompileArtifactInit(art_out);

    if (!src || stage < (int)MGL_STAGE_VERTEX ||
        stage > (int)MGL_STAGE_GEOMETRY) {
        if (err_buf && err_cap) {
            snprintf(err_buf, err_cap,
                     "CompileArtifact: invalid source or stage");
        }
        return -1;
    }

    /* Seed the TES patch size before codegen: mglAirCompileGLSLWithReflectInfoEx
     * treats stage_info.tess_patch_vertices as an input, reads it ahead of
     * reflection and restores it afterwards.  Leaving it 0 makes the backend
     * fall back to the GL default of 3, which silently mis-sizes the metallib
     * tessellation tag for any TCS that does not declare layout(vertices = 3)
     * -- a TCS with vertices = 1 then fails the native-interface check at
     * draw time and the whole draw is dropped. */
    if (tess_patch_vertices > 0u) {
        art_out->stage_info.tess_patch_vertices = tess_patch_vertices;
    }

    art_out->frontend.source = src;
    art_out->frontend.stage = stage;
    art_out->frontend.layout_version = art_out->layout.layout_version;

    unsigned char *bytes = NULL;
    size_t size = 0;
    char local_err[512];
    char *err = err_buf && err_cap ? err_buf : local_err;
    size_t cap = err_buf && err_cap ? err_cap : sizeof(local_err);
    err[0] = '\0';

    /* One compile path produces metallib + reflection + stage_info together.
     * Failure leaves art_out incomplete and destroys any partial lists. */
    if (mglAirCompileGLSLWithReflectInfoEx(
            src, stage, attrib_names, &bytes, &size, art_out->resources,
            &art_out->stage_info, air_flags,
            (const MGLShaderResourceList *)iface_peers, err, cap,
            &art_out->tu, sparse_residency_buffer_index) != 0) {
        free(bytes);
        mglCompileArtifactDestroy(art_out);
        if (err[0] && !art_out->frontend.diagnostics) {
            art_out->frontend.diagnostics = strdup(err);
        }
        return -1;
    }

    art_out->metallib_bytes = bytes;
    art_out->metallib_size = size;
    art_out->complete = 1;
    return 0;
}

int mglCompileArtifactFromGLSL(const char *src, int stage,
                               const char *const *attrib_names,
                               MGLCompileArtifact *art_out,
                               char *err_buf, size_t err_cap)
{
    return mglCompileArtifactFromGLSLEx(src, stage, attrib_names, 0u, NULL,
                                        0u, kMGLSparseResidencyDefaultBufferIndex,
                                        art_out, err_buf, err_cap);
}

int mglCompileArtifactParseGLSL(const char *src, int stage,
                                struct MGLTranslationUnit **tu_out,
                                char *err_buf, size_t err_cap)
{
    if (tu_out) {
        *tu_out = NULL;
    }
    if (!src || !tu_out) {
        if (err_buf && err_cap) {
            snprintf(err_buf, err_cap, "CompileArtifact parse: bad args");
        }
        return -1;
    }
    MGLFrontendSession sess;
    mglFrontendSessionInit(&sess);
    if (mglFrontendSessionBuild(&sess, src, stage, err_buf, err_cap) != 0) {
        mglFrontendSessionDestroy(&sess);
        return -1;
    }
    *tu_out = mglFrontendSessionStealTU(&sess);
    mglFrontendSessionDestroy(&sess);
    return *tu_out ? 0 : -1;
}

void mglCompileTranslationUnitDestroy(struct MGLTranslationUnit *tu)
{
    mglGLSLTranslationUnitDestroy(tu);
}

uint32_t mglCompileFrontendParseCount(void)
{
    return mglFrontendParseCount();
}

void mglCompileFrontendNoteReuse(void)
{
    mglFrontendNoteReuse();
}

int mglCompileTUHasNamedFunction(const struct MGLTranslationUnit *tu,
                                 const char *name)
{
    if (!tu || !name) {
        return 0;
    }
    for (uint32_t di = 0; di < tu->decl_count; di++) {
        const MGLDecl *d = tu->decls[di];
        if (d && d->name && d->body && strcmp(d->name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

void mglCompileTUGetComputeLocalSize(const struct MGLTranslationUnit *tu,
                                     int32_t *x, int32_t *y, int32_t *z)
{
    if (x) {
        *x = tu ? tu->layout_local_size_x : -1;
    }
    if (y) {
        *y = tu ? tu->layout_local_size_y : -1;
    }
    if (z) {
        *z = tu ? tu->layout_local_size_z : -1;
    }
}

int mglCompileTUBuiltinUsed(const struct MGLTranslationUnit *tu,
                            const char *name)
{
    return mglFrontendBuiltinUsed(NULL, tu, name);
}
