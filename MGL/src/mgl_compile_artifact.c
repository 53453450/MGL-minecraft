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
#include "mgl_glsl_parser.h"
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

int mglCompileArtifactFromGLSLEx(const char *src, int stage,
                                 const char *const *attrib_names,
                                 uint32_t air_flags,
                                 const void *iface_peers,
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
            &art_out->tu) != 0) {
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
                                        art_out, err_buf, err_cap);
}
