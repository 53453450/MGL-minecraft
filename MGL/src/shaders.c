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
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * shaders.c
 * MGL
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "shaders.h"
#include "glm_context.h"
#include "mgl_glsl_parser.h"
#include "mgl_compile_artifact.h"
#include "mgl_metal_ref.h"
#include "mgl_shader_abi.h"
#include "mgl_glsl_ast.h"
#include "mgl_frontend_session.h"
#include "mgl_spirv_glsl.h"

GLboolean mglIsProgram(GLMContext ctx, GLuint program);

static void mglShaderClearSpirV(Shader *ptr)
{
    if (!ptr)
        return;
    free(ptr->spirv_words);
    ptr->spirv_words = NULL;
    ptr->spirv_word_count = 0;
    ptr->spir_v_binary = GL_FALSE;
    ptr->spir_v_specialized = GL_FALSE;
    ptr->spir_v_nameless = GL_FALSE;
    free((void *)ptr->entry_point);
    ptr->entry_point = NULL;
}

static void mglCompileShaderObject(GLMContext ctx, Shader *ptr);

 const char *getShaderTypeStr(GLuint type)
{
    static const char *types[] = {"VERTEX_SHADER", "FRAGMENT_SHADER",
        "GEOMETRY_SHADER", "TESS_CONTROL_SHADER", "TESS_EVALUATION_SHADER",
        "COMPUTE_SHADER", "MAX_SHADER_TYPES", NULL};

    if (type >= _MAX_SHADER_TYPES)
        return "UNKNOWN_SHADER";

    return types[type];
};

GLuint glShaderTypeToGLMType(GLuint type)
{
    switch(type) {
        case GL_VERTEX_SHADER: return _VERTEX_SHADER;
        case GL_FRAGMENT_SHADER: return _FRAGMENT_SHADER;
        case GL_GEOMETRY_SHADER: return _GEOMETRY_SHADER;
        case GL_TESS_CONTROL_SHADER: return _TESS_CONTROL_SHADER;
        case GL_TESS_EVALUATION_SHADER: return _TESS_EVALUATION_SHADER;
        case GL_COMPUTE_SHADER: return _COMPUTE_SHADER;
        default:
            // CRITICAL FIX: Handle unknown shader types gracefully instead of crashing
            fprintf(stderr, "MGL ERROR: Unknown shader type 0x%x, defaulting to vertex shader\n", type);
            return _VERTEX_SHADER;
    }
}

Shader *newShader(GLMContext ctx, GLenum type, GLuint shader)
{
    Shader *ptr;
    char shader_type_name[128];

    ptr = (Shader *)malloc(sizeof(Shader));
    // CRITICAL SECURITY FIX: Check malloc result instead of using assert()
    if (!ptr) {
        fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate memory for shader\n");
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return NULL;
    }

    bzero(ptr, sizeof(Shader));

    ptr->name = shader;
    ptr->type = type;
    ptr->glm_type = glShaderTypeToGLMType(type);

    snprintf(shader_type_name, sizeof(shader_type_name), "%s_%d", getShaderTypeStr(ptr->glm_type), shader);
    ptr->mtl_shader_type_name = strdup(shader_type_name);
    /* If strdup fails (OOM), ptr->mtl_shader_type_name stays NULL.  This is
     * safe: the field is only ever passed to free() in the shader teardown
     * path and is never dereferenced, so a NULL value is a harmless no-op. */

    return ptr;
}

Shader *getShader(GLMContext ctx, GLenum type, GLuint shader)
{
    Shader *ptr;

    ptr = (Shader *)searchHashTable(&STATE(shader_table), shader);

    if (!ptr)
    {
        ptr = newShader(ctx, type, shader);

        insertHashElement(&STATE(shader_table), shader, ptr);
    }

    return ptr;
}

int isShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;

    ptr = (Shader *)searchHashTable(&STATE(shader_table), shader);

    /* GL §7.1: IsShader is TRUE for existing shader objects, including
     * those flagged for deletion but still attached to a program. */
    return ptr ? 1 : 0;
}

Shader *findShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;

    ptr = (Shader *)searchHashTable(&STATE(shader_table), shader);

    return ptr;
}

GLuint mglCreateShader(GLMContext ctx, GLenum type)
{
    GLuint shader;

    switch(type)
    {
        case GL_VERTEX_SHADER:
        case GL_FRAGMENT_SHADER:
        case GL_GEOMETRY_SHADER:
        case GL_COMPUTE_SHADER:
        case GL_TESS_CONTROL_SHADER:
        case GL_TESS_EVALUATION_SHADER:
            break;

        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, 0);
    }

    shader = getNewName(&STATE(shader_table));
    /* GL §7.1: shader and program objects share one name space. */
    while (shader != 0 &&
           searchHashTable(&STATE(program_table), shader) != NULL) {
        shader = getNewName(&STATE(shader_table));
    }

    getShader(ctx, type, shader);

    return shader;
}

void mglShaderReplaceFrontendTU(Shader *ptr, struct MGLTranslationUnit *tu)
{
    if (!ptr) {
        mglGLSLTranslationUnitDestroy(tu);
        return;
    }
    if (ptr->frontend_tu && ptr->frontend_tu != tu)
        mglGLSLTranslationUnitDestroy(ptr->frontend_tu);
    ptr->frontend_tu = tu;
}

void mglFreeShader(GLMContext ctx, Shader *ptr)
{
    (void)ctx;
    free((void *)ptr->mtl_shader_type_name);
    free((void *)ptr->src);
    if (ptr->log) free(ptr->log);
    free(ptr->frontend_diagnostics);
    ptr->frontend_diagnostics = NULL;
    ptr->frontend_valid = GL_FALSE;
    mglCompileArtifactFree(ptr->cached_artifact);
    ptr->cached_artifact = NULL;
    mglGLSLTranslationUnitDestroy(ptr->frontend_tu);
    ptr->frontend_tu = NULL;
    mglShaderClearSpirV(ptr);

    free(ptr->debug_label);
    free(ptr);
}

void mglDeleteShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;

    /* OpenGL spec: A value of 0 for shader will be silently ignored. */
    if (shader == 0) {
        return;
    }

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    ptr->delete_status = GL_TRUE;

    if (ptr->refcount == 0)
    {
        deleteHashElement(&STATE(shader_table), shader);
        mglFreeShader(ctx, ptr);
    }
}

GLboolean mglIsShader(GLMContext ctx, GLuint shader)
{
    return isShader(ctx, shader);
}

void mglShaderSource(GLMContext ctx, GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length)
{
    size_t len;
    GLchar *src;
    Shader *ptr;

    ERROR_CHECK_RETURN(shader != 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    if (count>1)
    {
        // compute storage requirement
        len = 0;
        if (!length) {
            for(int i=0; i<count; i++)
            {
                len += strlen(string[i]);
            }
        }
        else {
            for(int i=0; i<count; i++)
            {
                len += length[i];
            }
        }   
        ERROR_CHECK_RETURN(len, GL_INVALID_VALUE);

        // allocate storage
        src = (GLchar *)malloc(len+1); // +1 for NULL
        ERROR_CHECK_RETURN(src, GL_OUT_OF_MEMORY);

        if (!length) {        
            // string[i] are null-terminated
            *src = 0;
            for(int i=0; i<count; ++i)
            {
                strlcat(src, string[i], len+1);
            }
            if (strlen(src) != (size_t)len) {
                fprintf(stderr,
                        "MGL WARNING: shader source length mismatch expected=%zu actual=%zu\n",
                        (size_t)len,
                        strlen(src));
            }
        } else {
            // CRITICAL SECURITY FIX: Prevent buffer overflow in shader source concatenation
            // string[i] may not be null-terminated - we must validate bounds carefully
            size_t cum_len = 0;
            for(int i=0; i<count; ++i)
            {
                // CRITICAL: Check if adding this string would exceed buffer bounds
                if (cum_len + length[i] > (size_t)len) {
                    // SECURITY: Truncate safely instead of overflowing buffer
                    fprintf(stderr, "MGL SECURITY ERROR: Shader source concatenation would overflow buffer, truncating safely\n");
                    // Copy only what fits
                    size_t safe_copy_len = ((size_t)len > cum_len) ? ((size_t)len - cum_len) : 0;
                    if (safe_copy_len > 0) {
                        strncpy(&src[cum_len], string[i], safe_copy_len);
                    }
                    cum_len = len; // Force termination at end
                    break;
                }

                // CRITICAL: Validate source pointer and length before copy
                if (!string[i]) {
                    fprintf(stderr, "MGL SECURITY ERROR: NULL string pointer in shader source concatenation\n");
                    continue; // Skip this string
                }

                strncpy(&src[cum_len], string[i], length[i]);
                cum_len += length[i];
            }
            // CRITICAL: Ensure null termination regardless of truncation
            src[cum_len < (size_t)len ? cum_len : (size_t)len] = '\0';
        }
    }
    else
    {
        ERROR_CHECK_RETURN(string, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN(string[0], GL_INVALID_VALUE);

        /* Honor length[0] when provided (including for a single string).
         * CTS line_continuation NON_NULL cases append ignored trailing
         * text and pass an explicit length that excludes it. */
        if (length && length[0] >= 0) {
            len = (size_t)length[0];
            ERROR_CHECK_RETURN(len, GL_INVALID_VALUE);
            src = (GLchar *)malloc(len + 1);
            ERROR_CHECK_RETURN(src, GL_OUT_OF_MEMORY);
            memcpy(src, string[0], len);
            src[len] = '\0';
        } else {
            src = strdup(string[0]);
            if (!src) {
                mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                return;
            }
            len = strlen(src);
            ERROR_CHECK_RETURN(len, GL_INVALID_VALUE);
        }
    }

    if (len > (size_t)8 * 1024u * 1024u) {
        free(src);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    /* ShaderSource replaces any prior SPIR-V binary (ARB_gl_spirv). */
    mglShaderClearSpirV(ptr);
    free((void *)ptr->src);
    ptr->src_len = len;
    ptr->src = src;
    ptr->dirty_bits |= DIRTY_SHADER;
    ptr->compile_success = GL_FALSE;
    ptr->frontend_valid = GL_FALSE;
    mglCompileArtifactFree(ptr->cached_artifact);
    ptr->cached_artifact = NULL;
    mglShaderReplaceFrontendTU(ptr, NULL);
}

static void mglCompileShaderObject(GLMContext ctx, Shader *ptr)
{
    ptr->compile_success = GL_FALSE;
    ptr->frontend_valid = GL_FALSE;
    free(ptr->frontend_diagnostics);
    ptr->frontend_diagnostics = NULL;
    mglCompileArtifactFree(ptr->cached_artifact);
    ptr->cached_artifact = NULL;
    mglShaderReplaceFrontendTU(ptr, NULL);
    if (ptr->log) {
        free(ptr->log);
        ptr->log = NULL;
    }

    int air_stage = -1;
    switch (ptr->type) {
        case GL_VERTEX_SHADER: air_stage = MGL_STAGE_VERTEX; break;
        case GL_FRAGMENT_SHADER: air_stage = MGL_STAGE_FRAGMENT; break;
        case GL_COMPUTE_SHADER: air_stage = MGL_STAGE_COMPUTE; break;
        case GL_TESS_CONTROL_SHADER: air_stage = MGL_STAGE_TESS_CONTROL; break;
        case GL_TESS_EVALUATION_SHADER: air_stage = MGL_STAGE_TESS_EVALUATION; break;
        case GL_GEOMETRY_SHADER: air_stage = MGL_STAGE_GEOMETRY; break;
        default: break;
    }

    char error_text[1024] = {0};
    if (air_stage < 0 || !ptr->src) {
        ptr->log = strdup("AIR shader compilation failed");
        ptr->frontend_diagnostics = ptr->log ? strdup(ptr->log) : NULL;
        ptr->frontend_stage = air_stage;
        return;
    }

    /* R2: compile into an owned CompileArtifact so link can reuse metallib +
     * reflection when no variant remapping is required. */
    MGLCompileArtifact *art = mglCompileArtifactCreate();
    if (!art) {
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return;
    }
    if (mglCompileArtifactFromGLSL(ptr->src, air_stage, NULL, art,
                                   error_text, sizeof(error_text)) != 0 ||
        !art->complete) {
        mglCompileArtifactFree(art);
        art = NULL;
        /* GLSL 4.60 §1.2.1: a compilation unit may call a function whose
         * definition lives in another shader object of the same stage.
         * Parse/sema success is enough for COMPILE_STATUS; AIR is emitted
         * at link after the units are merged. */
        MGLFrontendSession sess;
        mglFrontendSessionInit(&sess);
        char parse_err[1024] = {0};
        if (mglFrontendSessionBuild(&sess, ptr->src, air_stage, parse_err,
                                    sizeof(parse_err)) == 0) {
            ptr->compile_success = GL_TRUE;
            ptr->frontend_stage = air_stage;
            ptr->frontend_parse_generation = mglFrontendParseCount();
            ptr->frontend_valid = GL_TRUE;
            mglShaderReplaceFrontendTU(ptr, mglFrontendSessionStealTU(&sess));
            mglFrontendSessionDestroy(&sess);
            ptr->dirty_bits |= DIRTY_SHADER;
            return;
        }
        mglFrontendSessionDestroy(&sess);
        ptr->log = strdup(error_text[0]
            ? error_text
            : (parse_err[0] ? parse_err : "AIR shader compilation failed"));
        ptr->frontend_diagnostics = ptr->log ? strdup(ptr->log) : NULL;
        ptr->frontend_stage = air_stage;
        return;
    }

    ptr->compile_success = GL_TRUE;
    ptr->frontend_stage = air_stage;
    ptr->frontend_parse_generation = mglFrontendParseCount();
    ptr->frontend_valid = GL_TRUE;
    ptr->cached_artifact = art;
    mglShaderReplaceFrontendTU(ptr, art->tu);
    art->tu = NULL;
    ptr->dirty_bits |= DIRTY_SHADER;
}

void mglCompileShader(GLMContext ctx, GLuint shader)
{
    ERROR_CHECK_RETURN(shader != 0, GL_INVALID_VALUE);

    Shader *ptr = findShader(ctx, shader);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    /* ARB_gl_spirv: CompileShader on a SPIR-V binary shader is invalid. */
    if (ptr->spir_v_binary) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    mglCompileShaderObject(ctx, ptr);
}

void mglShaderBinary(GLMContext ctx, GLsizei count, const GLuint *shaders,
                     GLenum binaryFormat, const void *binary, GLsizei length)
{
    GLsizei i;
    const uint32_t *words;
    size_t word_count;
    GLboolean nameless;

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(length >= 0, GL_INVALID_VALUE);
    if (count == 0)
        return;
    ERROR_CHECK_RETURN(shaders != NULL, GL_INVALID_VALUE);

    if (binaryFormat != GL_SHADER_BINARY_FORMAT_SPIR_V &&
        binaryFormat != GL_SHADER_BINARY_FORMAT_SPIR_V_ARB) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (STATE(var).num_shader_binary_formats == 0) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (length == 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if ((length & 3) != 0 || !binary) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (i = 0; i < count; i++) {
        ERROR_CHECK_RETURN(shaders[i] != 0, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN(findShader(ctx, shaders[i]) != NULL, GL_INVALID_VALUE);
    }

    words = (const uint32_t *)binary;
    word_count = (size_t)length / sizeof(uint32_t);
    if (word_count < 5u || words[0] != 0x07230203u) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    nameless = mglSpirvModuleHasNames(words, word_count) ? GL_FALSE : GL_TRUE;

    for (i = 0; i < count; i++) {
        Shader *ptr = findShader(ctx, shaders[i]);
        uint32_t *copy;

        copy = (uint32_t *)malloc(word_count * sizeof(uint32_t));
        if (!copy) {
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return;
        }
        memcpy(copy, words, word_count * sizeof(uint32_t));

        free((void *)ptr->src);
        ptr->src = NULL;
        ptr->src_len = 0;
        mglShaderClearSpirV(ptr);
        if (ptr->log) {
            free(ptr->log);
            ptr->log = NULL;
        }
        free(ptr->frontend_diagnostics);
        ptr->frontend_diagnostics = NULL;
        mglCompileArtifactFree(ptr->cached_artifact);
        ptr->cached_artifact = NULL;
        mglShaderReplaceFrontendTU(ptr, NULL);

        ptr->spirv_words = copy;
        ptr->spirv_word_count = word_count;
        ptr->spir_v_binary = GL_TRUE;
        ptr->spir_v_specialized = GL_FALSE;
        ptr->spir_v_nameless = nameless;
        ptr->compile_success = GL_FALSE;
        ptr->frontend_valid = GL_FALSE;
        ptr->dirty_bits |= DIRTY_SHADER;
    }
}

void mglSpecializeShader(GLMContext ctx, GLuint shader, const GLchar *pEntryPoint,
                         GLuint numSpecializationConstants,
                         const GLuint *pConstantIndex,
                         const GLuint *pConstantValue)
{
    Shader *ptr;
    char *glsl = NULL;
    char err[1024] = {0};
    int rc;

    if (shader == 0 || findShader(ctx, shader) == NULL) {
        /* Not a shader name: program names are INVALID_OPERATION. */
        if (shader != 0 && mglIsProgram(ctx, shader)) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    ptr = findShader(ctx, shader);
    if (!ptr->spir_v_binary) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (ptr->spir_v_specialized) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!pEntryPoint) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (numSpecializationConstants > 0u &&
        (!pConstantIndex || !pConstantValue)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    rc = mglSpirvToGLSL(ptr->spirv_words, ptr->spirv_word_count, ptr->type,
                        pEntryPoint, pConstantIndex, pConstantValue,
                        numSpecializationConstants, &glsl, err, sizeof(err));
    if (rc == MGL_SPIRV_ERR_ENTRY || rc == MGL_SPIRV_ERR_SPEC) {
        free(glsl);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (rc == MGL_SPIRV_ERR_OOM) {
        free(glsl);
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return;
    }
    if (rc != MGL_SPIRV_OK || !glsl) {
        free(glsl);
        if (ptr->log)
            free(ptr->log);
        ptr->log = strdup(err[0] ? err : "SpecializeShader failed");
        ptr->compile_success = GL_FALSE;
        return;
    }

    free((void *)ptr->src);
    ptr->src = glsl;
    ptr->src_len = strlen(glsl);
    free((void *)ptr->entry_point);
    ptr->entry_point = strdup(pEntryPoint);

    mglCompileShaderObject(ctx, ptr);
    if (ptr->compile_success)
        ptr->spir_v_specialized = GL_TRUE;
}

void mglGetShaderiv(GLMContext ctx, GLuint shader, GLenum pname, GLint *params)
{
    Shader *ptr;

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    switch(pname)
    {
        case GL_SHADER_TYPE:
            switch(ptr->glm_type)
            {
                case _VERTEX_SHADER: *params = GL_VERTEX_SHADER; break;
                case _FRAGMENT_SHADER: *params = GL_FRAGMENT_SHADER; break;
                case _GEOMETRY_SHADER: *params = GL_GEOMETRY_SHADER; break;
                case _COMPUTE_SHADER: *params = GL_COMPUTE_SHADER; break;
                case _TESS_CONTROL_SHADER: *params = GL_TESS_CONTROL_SHADER; break;
                case _TESS_EVALUATION_SHADER: *params = GL_TESS_EVALUATION_SHADER; break;
                default:
                    // CRITICAL FIX: Handle unknown shader types gracefully instead of crashing
                    fprintf(stderr, "MGL ERROR: Unknown internal shader type %d, defaulting to vertex\n", ptr->glm_type);
                    *params = GL_VERTEX_SHADER;
            }
            break;

        case GL_DELETE_STATUS:
            *params = ptr->delete_status ? GL_TRUE : GL_FALSE;
            break;

        case GL_COMPILE_STATUS:
            *params = ptr->compile_success ? GL_TRUE : GL_FALSE;
            break;

        case GL_INFO_LOG_LENGTH:
            *params = ptr->log ? (GLint)strlen(ptr->log) : 0;
            break;

        case GL_SHADER_SOURCE_LENGTH:
            *params = (GLint)ptr->src_len;
            break;

        case GL_SPIR_V_BINARY: /* == GL_SPIR_V_BINARY_ARB */
            *params = ptr->spir_v_binary ? GL_TRUE : GL_FALSE;
            break;

        case GL_COMPLETION_STATUS_KHR: /* GL_ARB/KHR_parallel_shader_compile */
            /* MGL compiles shaders synchronously, so every shader is always
             * complete by the time this query is issued. */
            *params = GL_TRUE;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

void mglGetShaderInfoLog(GLMContext ctx, GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    Shader *ptr = findShader(ctx, shader);
    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    const char *src = ptr->log ? ptr->log : "";
    size_t n = strlen(src);
    if (length) {
        *length = (GLsizei)n;
    }
    if (!infoLog || bufSize <= 0) {
        return;
    }
    if (n >= (size_t)bufSize) {
        n = (size_t)bufSize - 1u;
    }
    if (n > 0u) {
        memcpy(infoLog, src, n);
    }
    infoLog[n] = '\0';
}

void mglGetShaderSource(GLMContext ctx, GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source)
{
    Shader *ptr;

    ptr = findShader(ctx, shader);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);

    if (ptr->src)
    {
        if (length)
        {
            *length = (GLsizei)ptr->src_len;
        }

        if (source)
        {
            if (bufSize >= (GLsizei)ptr->src_len)
            {
                memcpy(source, ptr->src, ptr->src_len);
            }
        }
    }

}
