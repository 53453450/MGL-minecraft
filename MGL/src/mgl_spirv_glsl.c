/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * SPIR-V → GLSL via SPIRV-Cross (C API). Used by SpecializeShader.
 */

#include "mgl_spirv_glsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "GL/glcorearb.h"

#if defined(MGL_HAVE_SPIRV_CROSS)
#include <spirv_cross_c.h>
#endif

int mglSpirvExecutionModelForGLType(unsigned gl_shader_type)
{
    switch (gl_shader_type) {
        case GL_VERTEX_SHADER:
            return 0; /* SpvExecutionModelVertex */
        case GL_TESS_CONTROL_SHADER:
            return 1; /* SpvExecutionModelTessellationControl */
        case GL_TESS_EVALUATION_SHADER:
            return 2; /* SpvExecutionModelTessellationEvaluation */
        case GL_GEOMETRY_SHADER:
            return 3; /* SpvExecutionModelGeometry */
        case GL_FRAGMENT_SHADER:
            return 4; /* SpvExecutionModelFragment */
        case GL_COMPUTE_SHADER:
            return 5; /* SpvExecutionModelGLCompute */
        default:
            return -1;
    }
}

int mglSpirvModuleHasNames(const uint32_t *words, size_t word_count)
{
    /* SPIR-V header is 5 words; OpName=5, OpMemberName=6. */
    size_t i;

    if (!words || word_count < 5u)
        return 0;
    i = 5u;
    while (i < word_count) {
        uint32_t insn = words[i];
        uint16_t opc = (uint16_t)(insn & 0xFFFFu);
        uint16_t len = (uint16_t)(insn >> 16);
        if (len == 0u || (size_t)len > word_count - i)
            break;
        if (opc == 5u || opc == 6u)
            return 1;
        i += (size_t)len;
    }
    return 0;
}

#if !defined(MGL_HAVE_SPIRV_CROSS)

int mglSpirvToGLSL(const uint32_t *words, size_t word_count,
                   unsigned gl_shader_type, const char *entry_point,
                   const unsigned *spec_ids, const unsigned *spec_values,
                   unsigned num_specs, char **glsl_out, char *err,
                   size_t err_len)
{
    (void)words;
    (void)word_count;
    (void)gl_shader_type;
    (void)entry_point;
    (void)spec_ids;
    (void)spec_values;
    (void)num_specs;
    if (glsl_out)
        *glsl_out = NULL;
    if (err && err_len)
        snprintf(err, err_len, "SPIR-V support not built (need spirv-cross)");
    return MGL_SPIRV_ERR_UNSUPPORTED;
}

#else /* MGL_HAVE_SPIRV_CROSS */

static void mgl_spirv_set_err(char *err, size_t err_len, const char *msg)
{
    if (!err || err_len == 0u)
        return;
    if (!msg)
        msg = "SPIR-V cross-compile failed";
    snprintf(err, err_len, "%s", msg);
}

int mglSpirvToGLSL(const uint32_t *words, size_t word_count,
                   unsigned gl_shader_type, const char *entry_point,
                   const unsigned *spec_ids, const unsigned *spec_values,
                   unsigned num_specs, char **glsl_out, char *err,
                   size_t err_len)
{
    spvc_context context = NULL;
    spvc_parsed_ir ir = NULL;
    spvc_compiler compiler = NULL;
    spvc_compiler_options options = NULL;
    const spvc_entry_point *entries = NULL;
    size_t num_entries = 0;
    const spvc_specialization_constant *specs = NULL;
    size_t num_module_specs = 0;
    const char *source = NULL;
    int model;
    size_t ei, si, mi;
    int found_entry = 0;
    char *out = NULL;

    if (glsl_out)
        *glsl_out = NULL;
    if (!words || word_count == 0u || !entry_point || !glsl_out) {
        mgl_spirv_set_err(err, err_len, "invalid SPIR-V cross-compile arguments");
        return MGL_SPIRV_ERR_PARSE;
    }
    if (num_specs > 0u && (!spec_ids || !spec_values)) {
        mgl_spirv_set_err(err, err_len, "invalid specialization constant arrays");
        return MGL_SPIRV_ERR_SPEC;
    }

    model = mglSpirvExecutionModelForGLType(gl_shader_type);
    if (model < 0) {
        mgl_spirv_set_err(err, err_len, "unsupported shader stage for SPIR-V");
        return MGL_SPIRV_ERR_PARSE;
    }

    if (spvc_context_create(&context) != SPVC_SUCCESS) {
        mgl_spirv_set_err(err, err_len, "spvc_context_create failed");
        return MGL_SPIRV_ERR_OOM;
    }

    if (spvc_context_parse_spirv(context, (const SpvId *)words, word_count, &ir) !=
        SPVC_SUCCESS) {
        mgl_spirv_set_err(err, err_len, spvc_context_get_last_error_string(context));
        spvc_context_destroy(context);
        return MGL_SPIRV_ERR_PARSE;
    }

    if (spvc_context_create_compiler(context, SPVC_BACKEND_GLSL, ir,
                                     SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                                     &compiler) != SPVC_SUCCESS) {
        mgl_spirv_set_err(err, err_len, spvc_context_get_last_error_string(context));
        spvc_context_destroy(context);
        return MGL_SPIRV_ERR_PARSE;
    }

    if (spvc_compiler_get_entry_points(compiler, &entries, &num_entries) !=
        SPVC_SUCCESS) {
        mgl_spirv_set_err(err, err_len, spvc_context_get_last_error_string(context));
        spvc_context_destroy(context);
        return MGL_SPIRV_ERR_PARSE;
    }

    for (ei = 0; ei < num_entries; ei++) {
        if (entries[ei].execution_model == (SpvExecutionModel)model &&
            entries[ei].name && strcmp(entries[ei].name, entry_point) == 0) {
            found_entry = 1;
            break;
        }
    }
    if (!found_entry) {
        mgl_spirv_set_err(err, err_len, "entry point not found for shader stage");
        spvc_context_destroy(context);
        return MGL_SPIRV_ERR_ENTRY;
    }

    if (spvc_compiler_get_specialization_constants(compiler, &specs,
                                                   &num_module_specs) !=
        SPVC_SUCCESS) {
        mgl_spirv_set_err(err, err_len, spvc_context_get_last_error_string(context));
        spvc_context_destroy(context);
        return MGL_SPIRV_ERR_PARSE;
    }

    for (si = 0; si < (size_t)num_specs; si++) {
        int found = 0;
        for (mi = 0; mi < num_module_specs; mi++) {
            if (specs[mi].constant_id == spec_ids[si]) {
                spvc_constant c =
                    spvc_compiler_get_constant_handle(compiler, specs[mi].id);
                if (!c) {
                    mgl_spirv_set_err(err, err_len,
                                      "specialization constant handle missing");
                    spvc_context_destroy(context);
                    return MGL_SPIRV_ERR_SPEC;
                }
                /* GL SpecializeShader values are 32-bit words; treat as u32. */
                spvc_constant_set_scalar_u32(c, 0, 0, spec_values[si]);
                found = 1;
                break;
            }
        }
        if (!found) {
            mgl_spirv_set_err(err, err_len,
                              "unknown specialization constant id");
            spvc_context_destroy(context);
            return MGL_SPIRV_ERR_SPEC;
        }
    }

    if (spvc_compiler_set_entry_point(compiler, entry_point,
                                      (SpvExecutionModel)model) != SPVC_SUCCESS) {
        mgl_spirv_set_err(err, err_len, spvc_context_get_last_error_string(context));
        spvc_context_destroy(context);
        return MGL_SPIRV_ERR_ENTRY;
    }

    /* Emit void main() so the existing GLSL frontend can consume the unit. */
    if (strcmp(entry_point, "main") != 0) {
        if (spvc_compiler_rename_entry_point(compiler, entry_point, "main",
                                            (SpvExecutionModel)model) !=
            SPVC_SUCCESS) {
            mgl_spirv_set_err(err, err_len,
                              spvc_context_get_last_error_string(context));
            spvc_context_destroy(context);
            return MGL_SPIRV_ERR_COMPILE;
        }
    }

    if (spvc_compiler_create_compiler_options(compiler, &options) != SPVC_SUCCESS ||
        spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_GLSL_VERSION,
                                       450) != SPVC_SUCCESS ||
        spvc_compiler_install_compiler_options(compiler, options) != SPVC_SUCCESS) {
        mgl_spirv_set_err(err, err_len, spvc_context_get_last_error_string(context));
        spvc_context_destroy(context);
        return MGL_SPIRV_ERR_COMPILE;
    }

    if (spvc_compiler_compile(compiler, &source) != SPVC_SUCCESS || !source) {
        mgl_spirv_set_err(err, err_len, spvc_context_get_last_error_string(context));
        spvc_context_destroy(context);
        return MGL_SPIRV_ERR_COMPILE;
    }

    out = strdup(source);
    spvc_context_destroy(context);
    if (!out) {
        mgl_spirv_set_err(err, err_len, "out of memory duplicating GLSL");
        return MGL_SPIRV_ERR_OOM;
    }
    *glsl_out = out;
    if (err && err_len)
        err[0] = '\0';
    return MGL_SPIRV_OK;
}

#endif /* MGL_HAVE_SPIRV_CROSS */
