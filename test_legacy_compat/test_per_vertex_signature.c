/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * gl_PerVertex signature harness (O7.4 残条).
 *
 * Behavioural oracle for mglProgramPerVertexSignature() and the two
 * compatibility checks built on it (mglProgramPipelinePerVertexCompatible for
 * separable programs, mglLinkedProgramPerVertexCompatible for a linked one).
 *
 * The signature used to be recovered by scanning the GLSL text: find
 * "gl_PerVertex", take the braced region after it, and look for whole-word
 * member tokens.  That answered for comments and for any textual look-alike,
 * and it needed the raw source to stay alive.  The fixture below pins the
 * declaration-based semantics the TU gives us: only a real `gl_PerVertex`
 * block redeclaration produces a signature, its members are read from the
 * block's member list (instance name irrelevant), and a stage without such a
 * redeclaration is skipped by the comparison.
 *
 * Build: see `make test-per-vertex-signature`.
 */

#include "mgl_program_reflection.h"
#include "mgl_glsl_parser.h"
#include "mgl_types_program.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run;
static int tests_passed;

#define CHECK(cond, label)                                       \
    do {                                                         \
        tests_run++;                                             \
        if (cond) {                                              \
            tests_passed++;                                      \
            printf("  [PASS] %s\n", (label));                    \
        } else {                                                 \
            printf("  [FAIL] %s\n", (label));                    \
        }                                                        \
    } while (0)

/* Each install gets its own Shader + TU: two programs in one comparison must
 * not share a shader slot (that would make their signatures trivially equal). */
#define FIXTURE_MAX 8
typedef struct Fixture {
    Shader shader;
    MGLTranslationUnit *tu;
} Fixture;

static Fixture g_fixtures[FIXTURE_MAX];
static int g_fixture_count;

/* Install `src` as the shader for `stage` of `program`, compiling the TU the
 * way glCompileShader does (frontend_tu is what the query reads). */
static void install(Program *program, int stage, const char *src)
{
    if (g_fixture_count >= FIXTURE_MAX) {
        fprintf(stderr, "FATAL: fixture pool exhausted\n");
        exit(2);
    }
    Fixture *f = &g_fixtures[g_fixture_count++];
    memset(&f->shader, 0, sizeof(f->shader));
    f->shader.src = src;
    f->tu = NULL;
    if (src) {
        f->tu = mglGLSLParse(src, strlen(src));
        if (!f->tu || f->tu->error) {
            fprintf(stderr, "FATAL: fixture did not parse: %s\n",
                    f->tu && f->tu->error ? f->tu->error : "no TU");
            exit(2);
        }
        f->shader.frontend_tu = f->tu;
    }
    memset(program, 0, sizeof(*program));
    program->shader_slots[stage] = &f->shader;
    program->attached_shader_mask = 1u << stage;
}

static void release(void)
{
    for (int i = 0; i < g_fixture_count; i++) {
        mglGLSLTranslationUnitDestroy(g_fixtures[i].tu);
        g_fixtures[i].tu = NULL;
    }
    g_fixture_count = 0;
}

/* ---- fixtures ---- */

static const char *kNoRedecl =
    "#version 450\n"
    "layout(location = 0) in vec4 p;\n"
    "void main() { gl_Position = p; }\n";

static const char *kPositionOnly =
    "#version 450\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "void main() { gl_Position = vec4(1.0); }\n";

static const char *kFull =
    "#version 450\n"
    "out gl_PerVertex {\n"
    "    vec4 gl_Position;\n"
    "    float gl_PointSize;\n"
    "    float gl_ClipDistance[4];\n"
    "};\n"
    "void main() { gl_Position = vec4(1.0); }\n";

static const char *kCull =
    "#version 450\n"
    "out gl_PerVertex { vec4 gl_Position; float gl_CullDistance[2]; };\n"
    "void main() { gl_Position = vec4(1.0); }\n";

static const char *kNamedInstance =
    "#version 450\n"
    "out gl_PerVertex { vec4 gl_Position; float gl_PointSize; } vs_out;\n"
    "void main() { vs_out.gl_Position = vec4(1.0); }\n";

/* The old text scan answered "declared" for these: the name appears, and a
 * braced region follows.  A declaration-based query must not. */
static const char *kCommentOnly =
    "#version 450\n"
    "/* out gl_PerVertex { vec4 gl_Position; float gl_PointSize; }; */\n"
    "void main() { gl_Position = vec4(1.0); }\n";

static const char *kLookalikeStruct =
    "#version 450\n"
    "struct gl_PerVertexLike { vec4 gl_Position; float gl_PointSize; };\n"
    "void main() { gl_Position = vec4(1.0); }\n";

int main(void)
{
    printf("gl_PerVertex signature harness (O7.4)\n");

    Program prog_a, prog_b;
    unsigned sig = 0xffffffffu;

    /* ---- a stage without a redeclaration is not a reference ---- */
    install(&prog_a, _VERTEX_SHADER, kNoRedecl);
    CHECK(mglProgramPerVertexSignature(&prog_a, _VERTEX_SHADER, &sig) ==
              GL_FALSE,
          "no redeclaration -> no signature");
    CHECK(sig == 0u, "no redeclaration leaves the signature cleared");
    release();

    /* ---- members come from the declared block ---- */
    install(&prog_a, _VERTEX_SHADER, kPositionOnly);
    CHECK(mglProgramPerVertexSignature(&prog_a, _VERTEX_SHADER, &sig) == GL_TRUE,
          "gl_Position-only redeclaration is a signature");
    CHECK(sig == (1u << 0), "gl_Position-only signature has exactly bit 0");
    release();

    install(&prog_a, _VERTEX_SHADER, kFull);
    CHECK(mglProgramPerVertexSignature(&prog_a, _VERTEX_SHADER, &sig) == GL_TRUE,
          "full redeclaration is a signature");
    CHECK(sig == ((1u << 0) | (1u << 1) | (1u << 2)),
          "full redeclaration sets Position/PointSize/ClipDistance");
    release();

    install(&prog_a, _VERTEX_SHADER, kCull);
    CHECK(mglProgramPerVertexSignature(&prog_a, _VERTEX_SHADER, &sig) == GL_TRUE,
          "gl_CullDistance redeclaration is a signature");
    CHECK(sig == ((1u << 0) | (1u << 3)),
          "gl_CullDistance sets bit 3");
    release();

    install(&prog_a, _VERTEX_SHADER, kNamedInstance);
    CHECK(mglProgramPerVertexSignature(&prog_a, _VERTEX_SHADER, &sig) == GL_TRUE,
          "a named block instance is still a redeclaration");
    CHECK(sig == ((1u << 0) | (1u << 1)),
          "named instance reports its declared members");
    release();

    /* ---- text look-alikes are not declarations ---- */
    install(&prog_a, _VERTEX_SHADER, kCommentOnly);
    CHECK(mglProgramPerVertexSignature(&prog_a, _VERTEX_SHADER, &sig) ==
              GL_FALSE,
          "a commented-out block is not a redeclaration");
    release();

    install(&prog_a, _VERTEX_SHADER, kLookalikeStruct);
    CHECK(mglProgramPerVertexSignature(&prog_a, _VERTEX_SHADER, &sig) ==
              GL_FALSE,
          "a similarly named struct is not a redeclaration");
    release();

    /* ---- degenerate inputs ---- */
    CHECK(mglProgramPerVertexSignature(NULL, _VERTEX_SHADER, &sig) == GL_FALSE,
          "NULL program -> no signature");
    install(&prog_a, _VERTEX_SHADER, kFull);
    CHECK(mglProgramPerVertexSignature(&prog_a, _VERTEX_SHADER, NULL) ==
              GL_FALSE,
          "NULL signature out -> false");
    CHECK(mglProgramPerVertexSignature(&prog_a, _MAX_SHADER_TYPES, &sig) ==
              GL_FALSE,
          "out-of-range stage -> false");
    /* A stage that was never attached has no shader at all. */
    {
        Program empty;
        memset(&empty, 0, sizeof(empty));
        CHECK(mglProgramPerVertexSignature(&empty, _VERTEX_SHADER, &sig) ==
                  GL_FALSE,
              "missing stage -> false");
    }
    release();

    /* ---- pipeline compatibility compares the signatures ---- */
    install(&prog_a, _VERTEX_SHADER, kFull);   /* Position + PointSize + Clip */
    install(&prog_b, _GEOMETRY_SHADER, kPositionOnly); /* Position only */
    {
        Program *stages[_MAX_SHADER_TYPES] = {0};
        stages[_VERTEX_SHADER] = &prog_a;
        stages[_GEOMETRY_SHADER] = &prog_b;
        CHECK(mglProgramPipelinePerVertexCompatible(stages) == GL_FALSE,
              "mismatched redeclarations make the pipeline incompatible");
    }
    release();

    install(&prog_a, _VERTEX_SHADER, kPositionOnly);
    install(&prog_b, _GEOMETRY_SHADER, kPositionOnly);
    {
        Program *stages[_MAX_SHADER_TYPES] = {0};
        stages[_VERTEX_SHADER] = &prog_a;
        stages[_GEOMETRY_SHADER] = &prog_b;
        CHECK(mglProgramPipelinePerVertexCompatible(stages) == GL_TRUE,
              "matching redeclarations are compatible");
    }
    release();

    install(&prog_a, _VERTEX_SHADER, kPositionOnly);
    install(&prog_b, _GEOMETRY_SHADER, kNoRedecl);
    {
        Program *stages[_MAX_SHADER_TYPES] = {0};
        stages[_VERTEX_SHADER] = &prog_a;
        stages[_GEOMETRY_SHADER] = &prog_b;
        CHECK(mglProgramPipelinePerVertexCompatible(stages) == GL_TRUE,
              "a stage without a redeclaration is skipped");
        CHECK(mglProgramPipelinePerVertexCompatible(NULL) == GL_TRUE,
              "no stages at all is compatible");
    }
    release();

    /* A linked program carries every attached stage of the same Program. */
    {
        Program linked;
        memset(&linked, 0, sizeof(linked));
        static Shader vs_shader, fs_shader;
        static MGLTranslationUnit *vs_tu, *fs_tu;
        vs_tu = mglGLSLParse(kFull, strlen(kFull));
        fs_tu = mglGLSLParse(kPositionOnly, strlen(kPositionOnly));
        memset(&vs_shader, 0, sizeof(vs_shader));
        memset(&fs_shader, 0, sizeof(fs_shader));
        vs_shader.src = kFull;
        vs_shader.frontend_tu = vs_tu;
        fs_shader.src = kPositionOnly;
        fs_shader.frontend_tu = fs_tu;
        linked.shader_slots[_VERTEX_SHADER] = &vs_shader;
        linked.shader_slots[_FRAGMENT_SHADER] = &fs_shader;
        linked.attached_shader_mask =
            (1u << _VERTEX_SHADER) | (1u << _FRAGMENT_SHADER);
        CHECK(mglLinkedProgramPerVertexCompatible(&linked) == GL_FALSE,
              "a linked program with mismatched stages is incompatible");

        /* Same source in both stages: compatible. */
        fs_shader.src = kFull;
        fs_shader.frontend_tu = vs_tu;
        CHECK(mglLinkedProgramPerVertexCompatible(&linked) == GL_TRUE,
              "a linked program with matching stages is compatible");
        CHECK(mglLinkedProgramPerVertexCompatible(NULL) == GL_TRUE,
              "NULL program is compatible");
        mglGLSLTranslationUnitDestroy(vs_tu);
        mglGLSLTranslationUnitDestroy(fs_tu);
    }

    printf("test_per_vertex_signature: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
