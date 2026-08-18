/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_legacy_compat.h
 * MGL
 *
 * Legacy GLSL Compatibility Subsystem.
 *
 * Translates pre-GLSL-3.30 (GLSL 1.10 ~ 1.50) source-level constructs to
 * core-profile-compatible forms so the frontend can parse them.
 *
 * Reference: GLSLangSpec.1.10.pdf ~ GLSLangSpec.1.50.pdf
 *
 * The frontend parses core-profile GLSL, so legacy
 * syntax must be rewritten at the source level before the frontend sees it.
 * This module covers the syntactic translation layer:
 *
 *   - attribute / varying keywords        -> in / out  (§4.3, §4.4)
 *   - gl_FragColor                        -> user out  (§7.2)
 *   - gl_TexCoord[n]                      -> user varying array  (§7.1)
 *   - gl_FragData[n]                      -> user MRT out array  (§7.2)
 *   - texture1D/2D/3D/Cube/2DRect         -> texture  (§8.8)
 *   - texture1DProj/2DProj/3DProj         -> textureProj  (§8.8)
 *   - VS attribute builtins (gl_Normal, gl_Color, gl_MultiTexCoord0..7, ...) (§7.1)
 *   - VS varying outputs (gl_FrontColor, gl_BackColor, gl_ClipVertex, ...)   (§7.1)
 *   - FS varying inputs (gl_Color, gl_SecondaryColor, gl_FogFragCoord)      (§7.2)
 *
 * Fixed-function matrix uniforms (gl_ModelViewProjectionMatrix, ftransform,
 * gl_Vertex, ...) are handled separately by mglRewriteLegacyGLSL() in
 * shaders.c; this module does not touch those identifiers.
 *
 * Shadow texture functions (texture1DShadow, texture2DShadow, ...) are NOT
 * translated here because GLSL 1.30+ texture() returns float for shadow
 * samplers whereas 1.10 texture*Shadow returns vec4 — a simple identifier
 * swap would cause type errors.  These are left for a future pass that
 * wraps the call in vec4().
 *
 * All functions are pure (no renderer/ivar dependency) and operate on a
 * mutable source buffer with capacity tracking.
 */

#ifndef MGL_LEGACY_COMPAT_H
#define MGL_LEGACY_COMPAT_H

#include "glcorearb.h"
#include <stddef.h>  /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Detection result: which legacy features appear in the source.
 * Zero-initialise with memset(&feat, 0, sizeof(feat)) before calling
 * mgl_legacy_detect(). */
typedef struct {
    /* Keywords */
    GLboolean has_attribute;     /* 'attribute' keyword (VS input)          */
    GLboolean has_varying;       /* 'varying' keyword (VS out / FS in)      */

    /* Fragment outputs */
    GLboolean has_gl_FragColor;  /* gl_FragColor builtin fragment output    */
    GLboolean has_gl_TexCoord;   /* gl_TexCoord[] builtin varying array     */
    GLboolean has_gl_FragData;   /* gl_FragData[] builtin MRT output array  */

    /* Legacy texture functions (§8.8) */
    GLboolean has_texture1D;       /* texture1D()        */
    GLboolean has_texture1DProj;   /* texture1DProj()    */
    GLboolean has_texture2D;       /* texture2D()        */
    GLboolean has_texture2DProj;   /* texture2DProj()    */
    GLboolean has_texture3D;       /* texture3D()        */
    GLboolean has_texture3DProj;   /* texture3DProj()    */
    GLboolean has_textureCube;     /* textureCube()      */
    GLboolean has_texture2DRect;   /* texture2DRect()    */

    /* Legacy builtin variables (§7.1, §7.2) — aggregate flag.
     * Covers: gl_Normal, gl_Color, gl_SecondaryColor, gl_FogCoord,
     * gl_MultiTexCoord0..7, gl_FrontColor, gl_BackColor,
     * gl_FrontSecondaryColor, gl_BackSecondaryColor, gl_ClipVertex,
     * gl_FogFragCoord.  Individual detection is done internally via
     * the builtin table; callers only need this aggregate. */
    GLboolean has_legacy_builtins;

    GLboolean has_gl_ClipVertex; /* gl_ClipVertex VS builtin output      */

    /* Legacy matrix built-in uniforms (§7.4): gl_ModelViewMatrix,
     * gl_ProjectionMatrix, gl_ModelViewProjectionMatrix, gl_TextureMatrix[],
     * gl_NormalMatrix + inverse/transpose variants.  Injected verbatim
     * (original gl_ names) so the GL-side uniform contract is unchanged. */
    GLboolean has_legacy_matrices;

    GLboolean has_ftransform;    /* ftransform() fixed-function vertex      */

    /* Aggregate: true if any legacy feature was detected.  Set by
     * mgl_legacy_detect() for caller convenience. */
    GLboolean needs_translation;
} mgl_legacy_features_t;

/* Scan source for legacy GLSL constructs (identifier-aware, skips comments
 * and string literals to avoid false positives).
 *
 *   src      - GLSL source code (NUL-terminated)
 *   features - output struct, zeroed by caller
 */
void mgl_legacy_detect(const char *src, mgl_legacy_features_t *features);

/* Translate legacy GLSL constructs in-place to core-profile-compatible forms.
 *
 * Operates on the mutable source buffer (must have room for growth; caller
 * should allocate at least src_len + 2048, matching initGLSLInput's
 * modified_src allocation).
 *
 *   src          - mutable GLSL source (NUL-terminated, will be modified)
 *   src_capacity - total capacity of src buffer
 *   shader_type  - GL_VERTEX_SHADER / GL_FRAGMENT_SHADER (determines
 *                  varying translation direction and builtin variable
 *                  direction)
 *   version      - original GLSL version number (e.g. 110, 120, 150)
 *   features     - result of mgl_legacy_detect() (may be NULL, in which
 *                  case this function performs its own detection)
 *
 * Returns: 1 if source was modified, 0 if no changes, -1 on error.
 */
int mgl_translate_legacy_glsl(char *src,
                              size_t src_capacity,
                              GLuint shader_type,
                              int version,
                              const mgl_legacy_features_t *features);

#ifdef __cplusplus
}
#endif

#endif /* MGL_LEGACY_COMPAT_H */
