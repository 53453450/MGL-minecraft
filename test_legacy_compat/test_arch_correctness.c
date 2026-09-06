/*
 * Architecture-audit correctness probes (A02/A03/A13).
 * These lock in the CPU-visible counterexamples from
 * docs/ARCHITECTURE_AUDIT_2026-09-05.md.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

#include "draw_command.h"
#include "glm_context.h"
#include "hash_table.h"
#include "mgl_types_program.h"
#include "mgl_glsl_parser.h"
#include "mgl_air_tess_abi.h"
#include "mgl_air_gs_abi.h"
#include "mgl_draw_gs.h"
#include "mgl_draw_tess.h"
#include "mgl_render.h"
#include "mgl_shader_abi.h"

static volatile sig_atomic_t g_got_segv;
static jmp_buf g_jb;

static void on_segv(int sig)
{
    (void)sig;
    g_got_segv = 1;
    longjmp(g_jb, 1);
}

static int fail_count;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        fail_count++;
    } else {
        fprintf(stderr, "PASS: %s\n", msg);
    }
}

static GLMContext make_ctx(void)
{
    return createGLMContext(GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                            GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
}

static void test_a13_sync_null(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "A13 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }

    g_got_segv = 0;
    signal(SIGSEGV, on_segv);
    if (setjmp(g_jb) == 0) {
        glDeleteSync(NULL);
    }
    signal(SIGSEGV, SIG_DFL);
    GLenum err = glGetError();
    expect(!g_got_segv && err == GL_INVALID_VALUE,
           "A13 DeleteSync(NULL) -> INVALID_VALUE, no crash");

    while (glGetError() != GL_NO_ERROR) {
    }
    GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (sync) {
        glDeleteSync(sync);
        while (glGetError() != GL_NO_ERROR) {
        }
        glDeleteSync(sync);
        err = glGetError();
        expect(err == GL_INVALID_VALUE, "A13 double DeleteSync -> INVALID_VALUE");
    } else {
        fprintf(stderr, "SKIP: A13 double-delete (FenceSync NULL)\n");
    }

    destroyGLMContext(ctx);
}

static void test_a03_scissor_key(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "A03 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);

    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 32, 32);
    MGLStateKey key_a;
    MGLStateKey key_b;
    memset(&key_a, 0, sizeof(key_a));
    memset(&key_b, 0, sizeof(key_b));
    mglComputeStateKey(ctx, GL_TRIANGLES, false, &key_a);
    glScissor(65536, 0, 32, 32);
    mglComputeStateKey(ctx, GL_TRIANGLES, false, &key_b);
    GLenum err = glGetError();
    expect(err == GL_NO_ERROR && !mglStateKeysEqual(&key_a, &key_b) &&
               key_a.scissor[0] != key_b.scissor[0],
           "A03 scissor (0,*) vs (65536,*) produce distinct keys");

    destroyGLMContext(ctx);
}

static void test_a02_query_isolation(void)
{
    GLMContext a = make_ctx();
    GLMContext b = make_ctx();
    expect(a != NULL && b != NULL, "A02 create two contexts");
    if (!a || !b)
        return;

    MGLsetCurrentContext(a);
    while (glGetError() != GL_NO_ERROR) {
    }
    GLuint q = 0;
    glCreateQueries(GL_PRIMITIVES_GENERATED, 1, &q);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    expect(glGetError() == GL_NO_ERROR, "A02 context A begin query");

    MGLsetCurrentContext(b);
    while (glGetError() != GL_NO_ERROR) {
    }
    GLint current = -1;
    glGetQueryiv(GL_PRIMITIVES_GENERATED, GL_CURRENT_QUERY, &current);
    GLboolean sees = glIsQuery(q);
    expect(current == 0 && sees == GL_FALSE,
           "A02 context B does not see A's active query/object");

    GLuint qb = 0;
    glCreateQueries(GL_PRIMITIVES_GENERATED, 1, &qb);
    glBeginQuery(GL_PRIMITIVES_GENERATED, qb);
    GLint cur_b = 0;
    glGetQueryiv(GL_PRIMITIVES_GENERATED, GL_CURRENT_QUERY, &cur_b);
    expect(glGetError() == GL_NO_ERROR && (GLuint)cur_b == qb,
           "A02 context B can begin same target independently");

    MGLsetCurrentContext(a);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    destroyGLMContext(a);

    MGLsetCurrentContext(b);
    GLint cur_after = 0;
    glGetQueryiv(GL_PRIMITIVES_GENERATED, GL_CURRENT_QUERY, &cur_after);
    expect((GLuint)cur_after == qb && glIsQuery(qb),
           "A02 destroying A leaves B query intact");
    glEndQuery(GL_PRIMITIVES_GENERATED);
    destroyGLMContext(b);
}

static void test_a17_error_contracts(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "A17 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }

    glBindSampler(0xffffffffu, 0);
    expect(glGetError() == GL_INVALID_VALUE,
           "A17 BindSampler out-of-range unit -> INVALID_VALUE");

    while (glGetError() != GL_NO_ERROR) {
    }
    GLuint bad = glCreateShader(0xDEAD);
    expect(bad == 0 && glGetError() == GL_INVALID_ENUM,
           "A17 CreateShader invalid type -> 0 + INVALID_ENUM");

    destroyGLMContext(ctx);
}

static void test_a16_failed_relink_keeps_modules(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "A16 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }

    static const char *kVS =
        "#version 450\n"
        "void main() { gl_Position = vec4(0.0); }\n";
    static const char *kFS =
        "#version 450\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = vec4(1.0); }\n";
    static const char *kBadFS =
        "#version 450\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = undeclared_identifier; }\n";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint prog = glCreateProgram();
    glShaderSource(vs, 1, &kVS, NULL);
    glCompileShader(vs);
    glShaderSource(fs, 1, &kFS, NULL);
    glCompileShader(fs);
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    expect(linked == GL_TRUE, "A16 initial link succeeds");

    Program *pptr = (Program *)searchHashTable(&ctx->state.program_table, prog);
    expect(pptr != NULL && pptr->modules[_VERTEX_SHADER].metallib_bytes != NULL,
           "A16 initial VS metallib present");
    size_t vs_size = pptr ? pptr->modules[_VERTEX_SHADER].metallib_size : 0;
    unsigned char *vs_bytes = pptr ? pptr->modules[_VERTEX_SHADER].metallib_bytes : NULL;

    glUseProgram(prog);
    glShaderSource(fs, 1, &kBadFS, NULL);
    glCompileShader(fs);
    glLinkProgram(prog);
    GLint relinked = 1;
    glGetProgramiv(prog, GL_LINK_STATUS, &relinked);
    expect(relinked == GL_FALSE, "A16 failed relink reports LINK_STATUS false");
    expect(pptr && pptr->modules[_VERTEX_SHADER].metallib_bytes == vs_bytes &&
               pptr->modules[_VERTEX_SHADER].metallib_size == vs_size &&
               vs_size > 0,
           "A16 failed relink keeps prior VS metallib bytes");

    destroyGLMContext(ctx);
}

static void test_r3_draw_state(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "R3 createGLMContext");
    if (!ctx)
        return;
    expect(ctx->active_state == &ctx->state,
           "R3 active_state defaults to live state");

    MGLsetCurrentContext(ctx);
    glViewport(1, 2, 3, 4);
    glEnable(GL_SCISSOR_TEST);
    glScissor(5, 6, 7, 8);

    MGLStateKey key;
    memset(&key, 0, sizeof(key));
    mglComputeStateKey(ctx, GL_TRIANGLES, true, &key);

    MGLDrawState ds;
    mglDrawStateFromKey(&ds, &key, 1u);
    expect(ds.valid && ds.uses_elements &&
               ds.viewport[0] == 1 && ds.viewport[1] == 2 &&
               ds.scissor[0] == 5 && ds.scissor_enabled,
           "R3 DrawState captures indexed viewport/scissor");

    /* Simulate independent replay workspace storage without shallow-copying
     * HashTables (those aliases would dangle after destroy). */
    GLint live_vp0 = ctx->state.viewport[0];
    memset(&ctx->replay_state, 0, sizeof(ctx->replay_state));
    ctx->replay_state.viewport[0] = 99;
    ctx->replay_state.viewport[1] = ctx->state.viewport[1];
    expect(ctx->state.viewport[0] == live_vp0,
           "R3 replay_state storage is distinct from live viewport");
    expect(ctx->active_state == &ctx->state,
           "R3 active_state defaults to live outside flush");
    /* Redirect pattern used by flushDrawBufferLocked. */
    ctx->active_state = &ctx->replay_state;
    expect(ctx->active_state->viewport[0] == 99 &&
               ctx->state.viewport[0] == live_vp0,
           "R3 redirect reads replay without mutating live");
    ctx->active_state = &ctx->state;

    destroyGLMContext(ctx);
}

static void test_a18_plain_uniform_cleared_on_relink(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "A18 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }

    static const char *kVS =
        "#version 450\n"
        "uniform float u_scale;\n"
        "void main() { gl_Position = vec4(u_scale); }\n";
    static const char *kFS =
        "#version 450\n"
        "layout(location=0) out vec4 color;\n"
        "uniform float u_scale;\n"
        "void main() { color = vec4(u_scale); }\n";
    static const char *kVS2 =
        "#version 450\n"
        "uniform float u_scale;\n"
        "void main() { gl_Position = vec4(u_scale * 2.0); }\n";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vs, 1, &kVS, NULL);
    glCompileShader(vs);
    glShaderSource(fs, 1, &kFS, NULL);
    glCompileShader(fs);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    expect(linked == GL_TRUE, "A18 initial link succeeds");
    glUseProgram(prog);
    GLint loc = glGetUniformLocation(prog, "u_scale");
    expect(loc >= 0, "A18 u_scale location");
    glUniform1f(loc, 0.25f);

    Program *pptr = (Program *)searchHashTable(&ctx->state.program_table, prog);
    expect(pptr != NULL, "A18 program table lookup");
    int had_active = 0;
    if (pptr) {
        for (GLuint w = 0; w < 2; w++) {
            if (pptr->plain_uniform_active_mask[w])
                had_active = 1;
        }
    }
    expect(had_active, "A18 plain uniform active after upload");

    /* Relink with a different VS; storage must reset (mask cleared). */
    GLuint vs2 = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs2, 1, &kVS2, NULL);
    glCompileShader(vs2);
    glDetachShader(prog, vs);
    glAttachShader(prog, vs2);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    expect(linked == GL_TRUE, "A18 relink succeeds");
    /* SeedUniformInitializers may re-create slots for GLSL defaults; the
     * prior 0.25 upload must not survive. */
    glUseProgram(prog);
    loc = glGetUniformLocation(prog, "u_scale");
    expect(loc >= 0, "A18 u_scale location after relink");
    GLfloat got = 0.25f;
    glGetUniformfv(prog, loc, &got);
    expect(got != 0.25f,
           "A18 relink clears prior plain-uniform 0.25 upload");

    destroyGLMContext(ctx);
}


static void test_a18_use_program_unlinked_strict(void)
{
    unsetenv("MGL_COMPAT_PROGRAM_ERRORS");
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "A18 strict createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }
    GLuint prog = glCreateProgram();
    glUseProgram(prog);
    expect(glGetError() == GL_INVALID_OPERATION,
           "A18 UseProgram(unlinked) default -> INVALID_OPERATION");
    destroyGLMContext(ctx);
}

static void test_r2_frontend_parse_count(void)
{
    uint32_t before = mglFrontendParseCount();
    MGLTranslationUnit *tu = mglGLSLParse("#version 450\nvoid main(){}\n", 28);
    expect(tu != NULL, "R2 parse returns TU");
    if (tu)
        mglGLSLTranslationUnitDestroy(tu);
    expect(mglFrontendParseCount() > before, "R2 parse count increments");

    /* CompileShader caches CompileArtifact; LinkProgram adopts FS cache
     * (no iface peers) and notes frontend reuse. */
    uint32_t reuse_before = mglFrontendReuseCount();
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "R2 reuse createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    const char *vs =
        "#version 450\nvoid main() { gl_Position = vec4(0.0); }\n";
    const char *fs =
        "#version 450\nout vec4 color;\nvoid main() { color = vec4(1.0); }\n";
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(v);
    glCompileShader(f);
    GLint ok = 0;
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    expect(ok == GL_TRUE, "R2 VS compile");
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    expect(ok == GL_TRUE, "R2 FS compile");
    GLuint prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    expect(ok == GL_TRUE, "R2 link");
    expect(mglFrontendReuseCount() > reuse_before,
           "R2 link reuses CompileShader artifact");
    destroyGLMContext(ctx);
}

static void test_r4_fake_executor(void)
{
    MGLDrawState ds;
    MGLStateKey key;
    memset(&key, 0, sizeof(key));
    key.program_name = 1;
    key.vao_name = 2;
    key.viewport[0] = 0;
    key.viewport[2] = 8;
    key.scissor[2] = 8;
    mglDrawStateFromKey(&ds, &key, 1u);

    void *ex = mglFakeDrawExecutorCreate();
    expect(ex != NULL, "R4 fake executor create");
    if (!ex)
        return;
    const MGLDrawExecutorVTable *vt = mglFakeDrawExecutorVTable(ex);
    expect(vt && vt->encode_indexed && vt->destroy, "R4 fake vtable");

    MGLBufferHandle vb = { .obj = (void *)0x1, .generation = 1 };
    MGLBufferHandle ib = { .obj = (void *)0x2, .generation = 1 };
    expect(mglHandleIsLive(vb.obj, vb.generation, 1) &&
               mglHandleIsLive(ib.obj, ib.generation, 1),
           "R4 typed handles report live for matching generation");
    expect(!mglHandleIsLive(vb.obj, vb.generation, 2),
           "R4 typed handles reject stale generation");
    expect(ds.valid && ds.uses_elements, "R4 DrawState ready for indexed encode");
    expect(vt->encode_indexed(ex, &ds, vb, ib, 3u) == 0 &&
               mglFakeDrawExecutorEncodeCount(ex) == 1u,
           "R4 fake executor encodes indexed DrawState once");
    vt->destroy(ex);
}

static void test_xfb_draw_fail_closed(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "F01 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }
    glDrawTransformFeedback(GL_TRIANGLES, 1);
    expect(glGetError() == GL_INVALID_OPERATION,
           "F01 DrawTransformFeedback -> INVALID_OPERATION");
    destroyGLMContext(ctx);
}

static void test_vertex_attrib_defaults(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "F18 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }
    glVertexAttrib1f(0, 3.0f);
    GLfloat v[4] = {0, 0, 0, 0};
    glGetVertexAttribfv(0, GL_CURRENT_VERTEX_ATTRIB, v);
    expect(v[0] == 3.0f && v[1] == 0.0f && v[2] == 0.0f && v[3] == 1.0f,
           "F18 VertexAttrib1f defaults (x,0,0,1)");
    destroyGLMContext(ctx);
}

static void test_debug_message_log(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "F17 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_OTHER, 7,
                         GL_DEBUG_SEVERITY_NOTIFICATION, -1, "hello-mgl");
    GLenum src = 0, type = 0, sev = 0;
    GLuint id = 0;
    GLsizei len = 0;
    char buf[64];
    GLuint n = glGetDebugMessageLog(1, (GLsizei)sizeof(buf), &src, &type, &id,
                                    &sev, &len, buf);
    expect(n == 1 && id == 7 && strcmp(buf, "hello-mgl") == 0,
           "F17 DebugMessageInsert/GetDebugMessageLog round-trip");
    destroyGLMContext(ctx);
}

static void test_f04_msaa_query_and_fbo(void)
{
    GLMContext ctx = make_ctx();
    expect(ctx != NULL, "F04 createGLMContext");
    if (!ctx)
        return;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }

    GLint max_samples = 0;
    GLint max_image_samples = 0;
    glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
    glGetIntegerv(GL_MAX_IMAGE_SAMPLES, &max_image_samples);
    expect(max_samples == 4 && max_image_samples == 4,
           "F04 MAX_SAMPLES == MAX_IMAGE_SAMPLES == 4");

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, 8, 8, GL_TRUE);
    expect(glGetError() == GL_NO_ERROR, "F04 TexImage2DMultisample");

    GLint samples = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D_MULTISAMPLE, 0, GL_TEXTURE_SAMPLES,
                             &samples);
    expect(samples == 4, "F04 TEXTURE_SAMPLES reports GL sample count");

    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D_MULTISAMPLE, tex, 0);
    expect(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
           "F04 MS color FBO is complete");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    destroyGLMContext(ctx);
}

static void test_f16_tess_texture_and_per_patch_plan(void)
{
    Program tes;
    memset(&tes, 0, sizeof(tes));
    tes.tess_gen_mode = GL_ISOLINES;

    uint8_t factors[2u * MGL_AIR_TESS_FACTOR_RECORD_BYTES];
    memset(factors, 0, sizeof(factors));
    /* Patch 0 stays discarded (outer 0). Patch 1 matches the isolines
     * item-count fixture: edges {1,2,...} → 4 items. */
    const uint16_t live[6] = {0x3C00, 0x4000, 0x4200, 0x4400, 0x3800, 0x3800};
    memcpy(factors + MGL_AIR_TESS_FACTOR_RECORD_BYTES, live, sizeof(live));

    expect(mglTessEvalItemsPerPatch(&tes, factors) == 0u,
           "F16 discarded patch items == 0");
    expect(mglTessEvalItemsPerPatch(
               &tes, factors + MGL_AIR_TESS_FACTOR_RECORD_BYTES) == 4u,
           "F16 isolines live patch items == 4");
    expect(mglTessEvalItemsPerInstance(&tes, factors, 2u) == 4u,
           "F16 items per instance skips discarded patches");

    MGLRenderComputeExecutionPlan plan;
    memset(&plan, 0, sizeof(plan));
    char dummy_glin[4];
    MGLTessEvalPerPatchDispatchSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.gl_in_buffer = dummy_glin;
    spec.patch_count = 2u;
    spec.instance_count = 2u;
    spec.items_per_instance = 4u;
    spec.gl_in_vertices = 3u;
    void *keep = NULL;
    expect(mglTessAppendEvalPerPatchDispatches(&plan, &tes, factors, &spec,
                                               &keep) != false,
           "F16 TES per-patch plan succeeds");
    expect(plan.dispatch_op_count == 2u,
           "F16 TES dispatches once per live patch per instance");
    free(keep);

    MGLTessTextureBind binds[8];
    expect(mglTessCollectTextureBinds(NULL, &tes, _TESS_EVALUATION_SHADER,
                                      binds, 8u) == 0u,
           "F16 empty TES program has no texture binds");

    uint8_t stage_in[MGL_AIR_PER_VERTEX_STRIDE];
    memset(stage_in, 0xFF, sizeof(stage_in));
    expect(mglTessInitStageInDefaults(stage_in, 1u, MGL_AIR_PER_VERTEX_STRIDE) !=
               false,
           "F16 stage-in defaults succeed");
    {
        float point_size = 0.0f;
        float cull0 = 0.0f;
        memcpy(&point_size, stage_in + MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET,
               sizeof(point_size));
        memcpy(&cull0, stage_in + MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET,
               sizeof(cull0));
        expect(point_size == 1.0f, "F16 stage-in point size defaults to 1");
        expect(cull0 == 1.0f, "F16 stage-in cull distance defaults to 1");
    }

    {
        const uint32_t sparse[4] = {10u, 20u, 30u, 40u};
        const uint32_t gather[2] = {3u, 0u};
        uint32_t continuous[2] = {0u, 0u};
        expect(mglTessCompactSparseCapture(sparse, 0u, 4u, sizeof(uint32_t),
                                           gather, 2u, 1u, continuous,
                                           sizeof(continuous)) != false,
               "F16 sparse compact succeeds");
        expect(continuous[0] == 40u && continuous[1] == 10u,
               "F16 sparse compact gathers by index");
    }

    {
        uint32_t loc_map[32];
        memset(loc_map, 0xFF, sizeof(loc_map));
        mglDrawGsFillLocationMap(NULL, NULL, NULL, loc_map);
        expect(loc_map[0] == 0u && loc_map[31] == 0u,
               "F16 empty GS location map is identity fallback");
    }

    {
        uint32_t counts[MGL_AIR_GS_COUNTS_RECORD_WORDS * 2u];
        memset(counts, 0, sizeof(counts));
        mglDrawGsPresetCounts(counts, 2u);
        expect(counts[1] == 1u &&
                   counts[MGL_AIR_GS_COUNTS_RECORD_WORDS + 1u] == 1u,
               "F16 GS counts preset instance_count=1");
    }

    {
        MGLAIRGSXFBScatterParams scatter;
        expect(mglDrawGsFillXFBScatterParams(NULL, &scatter) == 0u,
               "F16 empty GS has no XFB buffers");
        expect(scatter.buffer_stream[0] == MGL_AIR_GS_XFB_NO_STREAM,
               "F16 empty XFB buffer_stream is NO_STREAM");
        expect(scatter.field_count == 0u, "F16 empty XFB has no fields");
    }

    {
        MGLRenderComputeExecutionPlan gs_plan;
        memset(&gs_plan, 0, sizeof(gs_plan));
        char input[1], output[1], counts[1], meta[1], gparams[4];
        expect(mglDrawGsAppendCoreBindings(&gs_plan, input, 8u, output, counts,
                                           counts, NULL, meta, counts, gparams,
                                           (uint32_t)sizeof(gparams)) != false,
               "F16 GS core bindings succeed without XFB");
        expect(gs_plan.binding_op_count == 7u,
               "F16 GS core bindings skip optional XFB slot");
        expect(gs_plan.binding_ops[0].index == MGL_AIR_GS_SLOT_INPUT &&
                   gs_plan.binding_ops[0].offset == 8u,
               "F16 GS input slot and offset");
        expect(gs_plan.binding_ops[6].kind == 1u &&
                   gs_plan.binding_ops[6].index == MGL_AIR_GS_SLOT_GATHER_PARAMS,
               "F16 GS gather params are bytes at slot 25");
    }

    {
        Program gs;
        memset(&gs, 0, sizeof(gs));
        gs.geometry_output_type = GL_POINTS;
        gs.geometry_vertices_out = 1u;
        gs.geometry_invocations = 2u;
        MGLGsComputeLayout layout;
        expect(mglDrawGsComputeLayout(&gs, 3u, 4u, GL_POINTS, &layout) != false,
               "F16 GS layout succeeds");
        expect(layout.work_item_count == 24u, "F16 GS work items = prims*inst*inv");
        expect(layout.records_per_primitive == 3u,
               "F16 GS points layout is 2 headers + 1 expanded");
        expect(layout.expanded_vertices == 1u, "F16 GS points expand 1 vertex");
        expect(layout.output_stride == MGL_AIR_PER_VERTEX_STRIDE,
               "F16 GS empty program uses default stride");
    }

    {
        uint32_t vis[8] = {3u, 0u, 0u, 0u, 5u, 0u, 0u, 0u};
        uint32_t offsets[8];
        memset(offsets, 0xFF, sizeof(offsets));
        mglDrawGsExclusivePrefixSum(vis, offsets, 2u, 1u);
        expect(offsets[0] == 0u && offsets[4] == 3u,
               "F16 GS XFB prefix sum is exclusive per buffer");
    }

    {
        MGLRenderComputeExecutionPlan scatter;
        char pipeline, params[4], vis[1], offsets[1], stage[1], xfb[1], written[1];
        expect(mglDrawGsFillXFBScatterPlan(&scatter, &pipeline, params, 4u, vis,
                                           offsets, stage, xfb, written, 3u) !=
                   false,
               "F16 GS XFB scatter plan succeeds");
        expect(scatter.binding_op_count == 6u,
               "F16 GS XFB scatter has params + 5 buffers");
        expect(scatter.dispatch.groups_x == 3u,
               "F16 GS XFB scatter dispatches one group per work item");
        expect(scatter.barrier_scope == MGL_RENDER_COMPUTE_BARRIER_BUFFERS,
               "F16 GS XFB scatter requests a buffer barrier");
    }

    {
        uint8_t rec[MGL_AIR_PER_VERTEX_STRIDE];
        expect(mglTessInitStageInDefaults(rec, 1u, MGL_AIR_PER_VERTEX_STRIDE) !=
                   false,
               "F16 pack setup defaults");
        MGLTessStageInMember member;
        memset(&member, 0, sizeof(member));
        member.size = 16u;
        member.component_bytes = 4u;
        member.components = 4u;
        member.base_type = MGL_TESS_STAGE_IN_FLOAT;
        MGLTessStageInAttribSrc src;
        memset(&src, 0, sizeof(src));
        src.use_current = 1u;
        src.current_valid = 1u;
        src.type = GL_FLOAT;
        src.attrib_size = 4u;
        const float current[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        memcpy(src.current, current, sizeof(current));
        expect(mglTessPackStageInRecords(rec, 1u, MGL_AIR_PER_VERTEX_STRIDE, 0,
                                         1, NULL, 0, false, 0u, 0, 0u, &member,
                                         1u, &src) != false,
               "F16 TCS stage-in pack succeeds");
        float pos[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        memcpy(pos, rec, sizeof(pos));
        expect(pos[0] == 1.0f && pos[3] == 4.0f,
               "F16 TCS stage-in pack writes current attrib");
    }

    {
        MGLRenderCullDistancePrimitive prims[4];
        uint32_t n = 0u;
        expect(mglRenderFillCullDistanceArrayPrimitives(
                   GL_TRIANGLE_STRIP, 5, 4u, prims, 4u, &n) == 0,
               "F16 strip cull split succeeds");
        expect(n == 2u, "F16 strip count=4 yields 2 triangles");
        expect(prims[0].vertices[0] == 5u && prims[0].vertices[1] == 6u &&
                   prims[0].vertices[2] == 7u,
               "F16 strip prim 0 is consecutive first+p");
        expect(prims[1].vertices[0] == 6u && prims[1].vertices[1] == 7u &&
                   prims[1].vertices[2] == 8u,
               "F16 strip prim 1 is consecutive first+p");

        expect(mglRenderFillCullDistanceArrayPrimitives(
                   GL_TRIANGLE_FAN, 5, 4u, prims, 4u, &n) == 0,
               "F16 fan cull split succeeds");
        expect(n == 2u && prims[0].vertices[0] == 5u &&
                   prims[0].vertices[1] == 6u && prims[0].vertices[2] == 7u &&
                   prims[1].vertices[0] == 5u && prims[1].vertices[1] == 7u &&
                   prims[1].vertices[2] == 8u,
               "F16 fan prims share first vertex");

        expect(mglRenderFillCullDistanceArrayPrimitives(
                   GL_LINE_STRIP, 2, 3u, prims, 4u, &n) == 0,
               "F16 line-strip cull split succeeds");
        expect(n == 2u && prims[0].vertex_count == 0u &&
                   prims[0].index_count == 0u && prims[0].vertices[0] == 2u &&
                   prims[1].vertices[0] == 3u,
               "F16 line-strip uses array start, no index buffer");

        expect(mglRenderFillCullDistanceArrayPrimitives(
                   GL_TRIANGLES, 0, 3u, prims, 4u, &n) == 1,
               "F16 triangles are not a cull array split");
    }

    {
        const uint32_t src[4] = {1u, 0xFFFFFFFFu, 2u, 0xFFFFFFFFu};
        uint32_t dst[4] = {0u, 0u, 0u, 0u};
        expect(mglTessSanitizeRestartIndices(dst, src, 4u, GL_UNSIGNED_INT,
                                             0xFFFFFFFFu) != false,
               "F16 restart sanitize succeeds");
        expect(dst[0] == 1u && dst[1] == 0u && dst[2] == 2u && dst[3] == 0u,
               "F16 restart indices become vertex 0");
    }

    {
        expect(mglRenderIsCullDistanceAttribName("culldistance_data") &&
                   mglRenderIsCullDistanceAttribName("culldistance_data[0]") &&
                   !mglRenderIsCullDistanceAttribName("position"),
               "F16 cull attrib name prefix");

        Program prog;
        memset(&prog, 0, sizeof(prog));
        MGLShaderResource res;
        memset(&res, 0, sizeof(res));
        res.name = "culldistance_data";
        res.location = 1u;
        res.gl_type = GL_FLOAT;
        res.gl_array_size = 1;
        prog.shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES].list = &res;
        prog.shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES].count = 1u;
        uint32_t attribs[4];
        expect(mglRenderCollectCullDistanceAttribs(&prog, attribs, 4u) == 1u &&
                   attribs[0] == 1u,
               "F16 collect culldistance_data at location 1");
        res.name = "position";
        expect(mglRenderCollectCullDistanceAttribs(&prog, attribs, 4u) == 0u,
               "F16 collect skips non-cull attrib");
        res.name = NULL;
        prog.attrib_location_names[1] = (char *)"culldistance_data[2]";
        expect(mglRenderCollectCullDistanceAttribs(&prog, attribs, 4u) == 1u &&
                   attribs[0] == 1u,
               "F16 collect falls back to attrib_location_names");

        MGLRenderCullDistanceLayout layout;
        memset(&layout, 0, sizeof(layout));
        char buf_a, buf_b;
        mglRenderAccumulateCullDistanceAttrib(&layout, &buf_a, 16, 32u, 4);
        mglRenderAccumulateCullDistanceAttrib(&layout, &buf_a, 99, 32u, 8);
        mglRenderAccumulateCullDistanceAttrib(&layout, &buf_b, 0, 16u, 0);
        expect(layout.culldist_size == 3u && layout.mtl_buffer == &buf_a &&
                   layout.stride == 32u &&
                   mglRenderCullDistanceLayoutOffset(&layout) == 20u,
               "F16 cull layout keeps first buffer and offset");

        const uint32_t verts[3] = {5u, 6u, 7u};
        MGLCullDistanceEmuParams params;
        mglRenderFillCullDistanceEmuParams(3u, 9u, verts, 3u, 20u, 32u, 2u, 1u,
                                           4u, &params);
        expect(params.prim_vertex_count == 3u && params.first_vertex == 9u &&
                   params.explicit_vertex_count == 3u &&
                   params.explicit_vertices[0] == 5u &&
                   params.culldist_size == 2u && params.first_instance == 1u &&
                   params.instance_stride == 4u,
               "F16 cull emu params fill");

        uint64_t bytes = 0u;
        expect(mglRenderCullDistanceCaptureBytes(0u, 3u, 1u, &bytes) == 0 &&
                   bytes == 96u,
               "F16 cull capture bytes for count=3");
        expect(mglRenderCullDistanceCaptureBytes(0u, 0u, 1u, &bytes) == -1,
               "F16 cull capture bytes reject count=0");

        uint32_t cap[3] = {9u, 9u, 9u};
        mglTessFillCaptureParams(4u, 8u, 2u, cap);
        expect(cap[0] == 4u && cap[1] == 8u && cap[2] == 2u,
               "F16 tess capture params are first/stride/base");
    }
}

int main(void)
{
    fail_count = 0;
    test_a13_sync_null();
    test_a03_scissor_key();
    test_a02_query_isolation();
    test_a17_error_contracts();
    test_a16_failed_relink_keeps_modules();
    test_a18_plain_uniform_cleared_on_relink();
    test_a18_use_program_unlinked_strict();
    test_r2_frontend_parse_count();
    test_r3_draw_state();
    test_r4_fake_executor();
    test_xfb_draw_fail_closed();
    test_vertex_attrib_defaults();
    test_debug_message_log();
    test_f04_msaa_query_and_fbo();
    test_f16_tess_texture_and_per_patch_plan();
    if (fail_count) {
        fprintf(stderr, "arch-correctness: %d failure(s)\n", fail_count);
        return 1;
    }
    fprintf(stderr, "arch-correctness: all probes passed\n");
    return 0;
}
