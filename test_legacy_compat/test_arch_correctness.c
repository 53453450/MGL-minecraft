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
    if (fail_count) {
        fprintf(stderr, "arch-correctness: %d failure(s)\n", fail_count);
        return 1;
    }
    fprintf(stderr, "arch-correctness: all probes passed\n");
    return 0;
}
