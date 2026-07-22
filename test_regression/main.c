/*
 * test_regression/main.c — MGL draw pipeline regression suite (Stage 0.1)
 *
 * Non-interactive, headless, FBO-offscreen. Covers the scenarios required by
 * docs/RENDERER_EVOLUTION_TODO.md §0.1:
 *   array / element / instanced / multidraw / indirect
 *   + FBO switch + transform feedback + conditional render
 *
 * Each test:
 *   1. Builds a small FBO (RGBA8 + DEPTH24)
 *   2. Renders one frame
 *   3. glReadPixels into a buffer
 *   4. Writes a TGA to the output dir
 *   5. (Caller compares against golden)
 *
 * Build:  make test-regression
 * Run:    build/test_regression [test_name|all]
 *
 * Exit code: 0 if all PASS, 1 if any FAIL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mach/mach.h>
#include <mach/task_info.h>

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

#include "draw_command.h"
#include "glm_context.h"
#include "MGLRenderer.h"

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

#define REG_W 128
#define REG_H 128
#define MAX_TESTS 29
#define SOAK_ITERATIONS 100000u
#define SOAK_SAMPLE_INTERVAL 4096u
#define SOAK_DEFAULT_GROWTH_LIMIT_MB 64u
#define TEST_RESULT_SKIP 77

typedef struct {
    uint64_t resident_bytes;
    uint64_t footprint_bytes;
} ProcessMemorySample;

static int sample_process_memory(ProcessMemorySample *sample)
{
    if (!sample) return -1;

    mach_task_basic_info_data_t basic = {0};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t result = task_info(mach_task_self(),
                                     MACH_TASK_BASIC_INFO,
                                     (task_info_t)&basic,
                                     &basic_count);
    if (result != KERN_SUCCESS) return -1;

    task_vm_info_data_t vm = {0};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    result = task_info(mach_task_self(),
                       TASK_VM_INFO,
                       (task_info_t)&vm,
                       &vm_count);
    if (result != KERN_SUCCESS) return -1;

    sample->resident_bytes = (uint64_t)basic.resident_size;
    sample->footprint_bytes = (uint64_t)vm.phys_footprint;
    return 0;
}

static uint64_t positive_growth(uint64_t current, uint64_t baseline)
{
    return current > baseline ? current - baseline : 0u;
}

static uint64_t soak_growth_limit_bytes(void)
{
    const char *value = getenv("MGL_SOAK_RSS_LIMIT_MB");
    if (value && value[0] != '\0') {
        char *end = NULL;
        unsigned long long limit_mb = strtoull(value, &end, 10);
        if (end != value && *end == '\0' && limit_mb > 0u &&
            limit_mb <= (UINT64_MAX / (1024u * 1024u))) {
            return (uint64_t)limit_mb * 1024u * 1024u;
        }
    }
    return (uint64_t)SOAK_DEFAULT_GROWTH_LIMIT_MB * 1024u * 1024u;
}

static int soak_should_checkpoint(uint32_t completed)
{
    return completed == 1u || completed == 16u || completed == 256u ||
           completed == 1024u ||
           completed % SOAK_SAMPLE_INTERVAL == 0u ||
           completed == SOAK_ITERATIONS;
}

static int soak_memory_hard_limit_exceeded(
    const char *name,
    const ProcessMemorySample *baseline,
    const ProcessMemorySample *current,
    uint64_t hard_limit)
{
    uint64_t rss_growth = positive_growth(current->resident_bytes,
                                          baseline->resident_bytes);
    uint64_t footprint_growth = positive_growth(current->footprint_bytes,
                                                baseline->footprint_bytes);
    if (rss_growth <= hard_limit && footprint_growth <= hard_limit) return 0;

    fprintf(stderr,
            "%s: hard memory limit exceeded (rss=%.1f MiB footprint=%.1f MiB "
            "limit=%.1f MiB)\n",
            name,
            (double)rss_growth / (1024.0 * 1024.0),
            (double)footprint_growth / (1024.0 * 1024.0),
            (double)hard_limit / (1024.0 * 1024.0));
    return 1;
}

static int verify_soak_memory_growth(
    const char *name,
    const ProcessMemorySample *baseline,
    const ProcessMemorySample *midpoint,
    const ProcessMemorySample *final,
    uint64_t limit)
{
    uint64_t rss_growth = positive_growth(final->resident_bytes,
                                          baseline->resident_bytes);
    uint64_t footprint_growth = positive_growth(final->footprint_bytes,
                                                baseline->footprint_bytes);
    uint64_t rss_tail = positive_growth(final->resident_bytes,
                                        midpoint->resident_bytes);
    uint64_t footprint_tail = positive_growth(final->footprint_bytes,
                                              midpoint->footprint_bytes);
    uint64_t tail_limit = limit / 2u;

    fprintf(stderr,
            "%s: rss +%.1f MiB (tail +%.1f), footprint +%.1f MiB "
            "(tail +%.1f)\n",
            name,
            (double)rss_growth / (1024.0 * 1024.0),
            (double)rss_tail / (1024.0 * 1024.0),
            (double)footprint_growth / (1024.0 * 1024.0),
            (double)footprint_tail / (1024.0 * 1024.0));

    return rss_growth > limit || footprint_growth > limit ||
           rss_tail > tail_limit || footprint_tail > tail_limit;
}

/* ------------------------------------------------------------------ */
/* TGA writer (uncompressed BGRA-top-left, 3 or 4 channel)            */
/* ------------------------------------------------------------------ */

static int write_tga(const char *path, int w, int h, const unsigned char *rgba)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    unsigned char header[18] = {0};
    header[2]  = 2;          /* uncompressed true-color */
    header[12] = w & 0xFF;
    header[13] = (w >> 8) & 0xFF;
    header[14] = h & 0xFF;
    header[15] = (h >> 8) & 0xFF;
    header[16] = 24;         /* 24 bpp (BGR) */
    header[17] = 0x20;       /* top-left origin */
    fwrite(header, 1, 18, fp);

    /* RGBA -> BGR */
    for (int i = 0; i < w * h; i++) {
        unsigned char bgr[3] = {
            rgba[i * 4 + 2],  /* B */
            rgba[i * 4 + 1],  /* G */
            rgba[i * 4 + 0],  /* R */
        };
        fwrite(bgr, 1, 3, fp);
    }
    fclose(fp);
    return 0;
}

/* ------------------------------------------------------------------ */
/* File compare (byte-for-byte)                                       */
/* ------------------------------------------------------------------ */

static int files_equal(const char *a, const char *b)
{
    FILE *fa = fopen(a, "rb");
    FILE *fb = fopen(b, "rb");
    if (!fa || !fb) {
        if (fa) fclose(fa);
        if (fb) fclose(fb);
        return 0;
    }
    int eq = 1;
    char ba[4096], bb[4096];
    while (1) {
        size_t na = fread(ba, 1, sizeof(ba), fa);
        size_t nb = fread(bb, 1, sizeof(bb), fb);
        if (na != nb) { eq = 0; break; }
        if (na == 0) break;
        if (memcmp(ba, bb, na) != 0) { eq = 0; break; }
    }
    fclose(fa);
    fclose(fb);
    return eq;
}

/* ------------------------------------------------------------------ */
/* GL state reset (Stage 5.3 prerequisite — context isolation)        */
/* ------------------------------------------------------------------ */

/* Number of texture units to unbind in resetGLState.  The suite never
 * exceeds unit 0, but we sweep a few extra for robustness against future
 * tests and to catch residual binds from prior test failures. */
#define REG_RESET_TEXTURE_UNITS 4

/* Reset all mutable GL state to a known-clean baseline so each test starts
 * from the same conditions regardless of what the previous test (or a
 * previous failed early-return) left bound / enabled.
 *
 * This is the core of the context-isolation fix: without it, a prior test's
 * residual bound program / enabled cap / texture binding leaks into the next
 * test's first draw, and under ASan pressure the leaked state causes
 * nondeterministic processGLStateLocked paths → flaky golden mismatches.
 *
 * Every call here uses GL name 0 (unbind / default) — no dependency on any
 * object still existing. */
static void resetGLState(void)
{
    /* --- Bind targets to 0 (unbind everything) --- */
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    /* Indexed UBO / TF / SSBO binding 0 — clear residual indexed binds
     * that glBindBuffer(0) does not touch. */
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);

    /* --- Disable all mutable caps --- */
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_RASTERIZER_DISCARD);

    /* --- Depth state defaults --- */
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    /* --- Color / stencil mask defaults --- */
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glStencilMask(0xFF);

    /* --- Blend / stencil func defaults --- */
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glStencilFunc(GL_ALWAYS, 0, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

    /* --- Texture units: unbind all, set active to 0 --- */
    for (int i = REG_RESET_TEXTURE_UNITS - 1; i >= 0; i--) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        glBindTexture(GL_TEXTURE_3D, 0);
        glBindSampler(i, 0);
    }
    glActiveTexture(GL_TEXTURE0);

    /* --- Viewport / scissor defaults --- */
    glViewport(0, 0, REG_W, REG_H);
    glDisable(GL_SCISSOR_TEST);

    /* --- Pixel store defaults --- */
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
}

/* ------------------------------------------------------------------ */
/* GL helpers                                                         */
/* ------------------------------------------------------------------ */

static GLuint compile_shader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "  [shader compile FAIL] %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint link_program(const char *vs_src, const char *fs_src)
{
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    if (!vs) return 0;
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!fs) { glDeleteShader(vs); return 0; }
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "  [program link FAIL] %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

/* Create an FBO with color texture + depth renderbuffer. Returns fbo id;
 * color texture id via out_tex. */
static GLuint make_fbo(int w, int h, GLuint *out_tex)
{
    GLuint fbo, tex, rbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "  [FBO incomplete: 0x%x]\n", st);
        return 0;
    }
    if (out_tex) *out_tex = tex;
    return fbo;
}

static void clear_color(float r, float g, float b)
{
    glClearColor(r, g, b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

/* ------------------------------------------------------------------ */
/* Shared shaders                                                     */
/* ------------------------------------------------------------------ */

/* Basic: position (vec2) + color (vec3), optional instanced offset */
static const char *VS_BASIC =
    "#version 330 core\n"
    "layout(location = 0) in vec2 a_pos;\n"
    "layout(location = 1) in vec3 a_color;\n"
    "#ifdef INSTANCED\n"
    "layout(location = 2) in vec2 a_inst_offset;\n"
    "#endif\n"
    "out vec3 v_color;\n"
    "uniform vec2 u_offset;\n"
    "uniform float u_scale;\n"
    "void main() {\n"
    "  vec2 p = a_pos * u_scale + u_offset;\n"
    "#ifdef INSTANCED\n"
    "  p += a_inst_offset;\n"
    "#endif\n"
    "  gl_Position = vec4(p, 0.0, 1.0);\n"
    "  v_color = a_color;\n"
    "}\n";

static const char *FS_BASIC =
    "#version 330 core\n"
    "in vec3 v_color;\n"
    "out vec4 frag;\n"
    "void main() { frag = vec4(v_color, 1.0); }\n";

/* XFB pass-through: just pass position through, capture it */
static const char *VS_XFB =
    "#version 330 core\n"
    "layout(location = 0) in vec2 a_pos;\n"
    "out vec2 tf_pos;\n"
    "void main() {\n"
    "  tf_pos = a_pos * 0.5;\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

static const char *FS_XFB =
    "#version 330 core\n"
    "out vec4 frag;\n"
    "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";

/* Read back XFB: use captured positions */
static const char *VS_XFB_DRAW =
    "#version 330 core\n"
    "layout(location = 0) in vec2 a_pos;\n"
    "out vec3 v_color;\n"
    "void main() {\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "  v_color = vec3(1.0, 0.5, 0.0);\n"  /* orange = XFB result */
    "}\n";

static const char *FS_SOLID =
    "#version 330 core\n"
    "out vec4 frag;\n"
    "uniform vec4 u_color;\n"
    "void main() { frag = u_color; }\n";

/* ------------------------------------------------------------------ */
/* Geometry                                                           */
/* ------------------------------------------------------------------ */

/* Single triangle, NDC, red */
static const float TRI_VERTS[] = {
    -0.6f, -0.6f,
     0.6f, -0.6f,
     0.0f,  0.6f,
};
static const float TRI_COLORS[] = {
    1.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
};

/* Quad as two triangles (6 verts) for element draw */
static const float QUAD_VERTS[] = {
    -0.6f, -0.6f,
     0.6f, -0.6f,
     0.6f,  0.6f,
    -0.6f,  0.6f,
};
static const float QUAD_COLORS[] = {
    0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f,
};
static const unsigned short QUAD_INDICES[] = { 0, 1, 2, 0, 2, 3 };

static GLuint make_vbo(const void *data, size_t sz)
{
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sz, data, GL_STATIC_DRAW);
    return b;
}

/* Bind a pos2-only VAO (location 0, vec2) from `verts`. Caller owns *out_vao
 * and *out_vbo and must glDelete them. Used by the Stage 4.2 DontCare tests. */
static void make_pos2_vao(const void *verts, size_t sz, GLuint *out_vao, GLuint *out_vbo)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(verts, sz);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    if (out_vao) *out_vao = vao;
    if (out_vbo) *out_vbo = vbo;
}

static GLuint make_rgba8_texture(const unsigned char rgba[4])
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return texture;
}

static GLuint make_sampler_test_program(int texture_count)
{
    static const char *vs =
        "#version 330 core\n"
        "layout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p,0.0,1.0); }\n";
    static const char *fs_one =
        "#version 330 core\n"
        "uniform sampler2D u_tex; out vec4 frag;\n"
        "void main(){ frag=texture(u_tex,vec2(-1.0,0.5)); }\n";
    static const char *fs_two =
        "#version 330 core\n"
        "uniform sampler2D u_tex0; uniform sampler2D u_tex1; out vec4 frag;\n"
        "void main(){ frag=(texture(u_tex0,vec2(0.5)) + "
        "texture(u_tex1,vec2(0.5))) * 0.5; }\n";
    return link_program(vs, texture_count == 2 ? fs_two : fs_one);
}

static void make_sampler_switch_vao(GLuint *out_vao, GLuint *out_vbo)
{
    static const float verts[] = {
        -0.95f, -0.80f,  -0.35f, -0.80f,  -0.65f, 0.60f,
        -0.30f, -0.80f,   0.30f, -0.80f,   0.00f, 0.60f,
         0.35f, -0.80f,   0.95f, -0.80f,   0.65f, 0.60f,
    };
    make_pos2_vao(verts, sizeof(verts), out_vao, out_vbo);
}

static int verify_sampler_switch_pixels(const unsigned char *pixels,
                                        const char *test_name)
{
    static const int xs[3] = { 22, 64, 106 };
    static const unsigned char expected[3][3] = {
        { 255, 255, 255 },
        { 255,   0,   0 },
        { 255, 255, 255 },
    };
    const int y = 51;
    for (int i = 0; i < 3; i++) {
        const unsigned char *actual = &pixels[(y * REG_W + xs[i]) * 4];
        for (int c = 0; c < 3; c++) {
            int delta = (int)actual[c] - (int)expected[i][c];
            if (delta < -2 || delta > 2) {
                fprintf(stderr,
                        "%s: pixel %d expected rgb=(%u,%u,%u), got (%u,%u,%u)\n",
                        test_name, i,
                        expected[i][0], expected[i][1], expected[i][2],
                        actual[0], actual[1], actual[2]);
                return 1;
            }
        }
    }
    return 0;
}

static int command_buffer_counts_equal(const MGLCommandBuffer *cb,
                                       uint32_t total_commands,
                                       uint32_t batch_count,
                                       uint16_t key_count,
                                       uint16_t set_count)
{
    return cb &&
           cb->total_commands == total_commands &&
           cb->batch_count == batch_count &&
           cb->sampler_snapshot_key_count == key_count &&
           cb->sampler_snapshot_set_count == set_count;
}

/* ------------------------------------------------------------------ */
/* Test cases                                                         */
/* Each returns 0 on success, nonzero on GL error.                    */
/* Fills `pixels` with REG_W*REG_H*4 RGBA bytes.                     */
/* ------------------------------------------------------------------ */

typedef int (*test_fn)(unsigned char *pixels, const char *out_path);

/* ---- 1. glDrawArrays ---- */
static int test_draw_arrays(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(VS_BASIC, FS_BASIC);
    if (!prog) return 2;
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 1.0f);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_p = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    GLuint vbo_c = make_vbo(TRI_COLORS, sizeof(TRI_COLORS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_p);
    glDeleteBuffers(1, &vbo_c);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 2. glDrawElements ---- */
static int test_draw_elements(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(VS_BASIC, FS_BASIC);
    if (!prog) return 2;
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 1.0f);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_p = make_vbo(QUAD_VERTS, sizeof(QUAD_VERTS));
    GLuint vbo_c = make_vbo(QUAD_COLORS, sizeof(QUAD_COLORS));
    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(QUAD_INDICES), QUAD_INDICES, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_p);
    glDeleteBuffers(1, &vbo_c);
    glDeleteBuffers(1, &ibo);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 3. glDrawArraysInstanced ---- */
static int test_draw_arrays_instanced(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(
        "#version 330 core\n"
        "#define INSTANCED\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "layout(location = 1) in vec3 a_color;\n"
        "layout(location = 2) in vec2 a_inst_offset;\n"
        "out vec3 v_color;\n"
        "uniform float u_scale;\n"
        "void main() {\n"
        "  vec2 p = a_pos * u_scale + a_inst_offset;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  v_color = a_color;\n"
        "}\n",
        FS_BASIC);
    if (!prog) return 2;
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 0.3f);

    /* 3 instances, each offset horizontally */
    float inst_offsets[] = {
        -0.5f, 0.0f,
         0.0f, 0.0f,
         0.5f, 0.0f,
    };
    float colors[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_p = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    GLuint vbo_c = make_vbo(colors, sizeof(colors));
    GLuint vbo_i = make_vbo(inst_offsets, sizeof(inst_offsets));

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_i);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glVertexAttribDivisor(2, 1);

    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_p);
    glDeleteBuffers(1, &vbo_c);
    glDeleteBuffers(1, &vbo_i);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 4. glMultiDrawElements ---- */
static int test_multi_draw_elements(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(VS_BASIC, FS_BASIC);
    if (!prog) return 2;
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 0.3f);

    /* Two quads side by side via two index ranges in one element buffer.
     * We build one big index buffer holding two quads (12 indices) and issue
     * a multi-draw with count[0]=6, count[1]=6. */
    float verts[] = {
        /* left quad */
        -0.9f, -0.6f,
        -0.2f, -0.6f,
        -0.2f,  0.6f,
        -0.9f,  0.6f,
        /* right quad */
         0.2f, -0.6f,
         0.9f, -0.6f,
         0.9f,  0.6f,
         0.2f,  0.6f,
    };
    float colors[] = {
        1.0f, 0.0f, 0.0f,  /* red */
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,  /* green */
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    unsigned short indices[] = {
        0, 1, 2, 0, 2, 3,   /* left quad */
        4, 5, 6, 4, 6, 7,   /* right quad */
    };

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_p = make_vbo(verts, sizeof(verts));
    GLuint vbo_c = make_vbo(colors, sizeof(colors));
    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    GLsizei counts[] = { 6, 6 };
    const void *ptrs[] = { (void *)0, (void *)(sizeof(unsigned short) * 6) };

    glMultiDrawElements(GL_TRIANGLES, counts, GL_UNSIGNED_SHORT, ptrs, 2);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_p);
    glDeleteBuffers(1, &vbo_c);
    glDeleteBuffers(1, &ibo);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 5. glDrawArraysIndirect ---- */
static int test_draw_arrays_indirect(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(VS_BASIC, FS_BASIC);
    if (!prog) return 2;
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 1.0f);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_p = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    GLuint vbo_c = make_vbo(TRI_COLORS, sizeof(TRI_COLORS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    /* DrawArraysIndirectCommand: {count, primCount, first, baseInstance} */
    struct {
        GLuint count;
        GLuint primCount;
        GLuint first;
        GLuint baseInstance;
    } cmd = { 3, 1, 0, 0 };

    GLuint cmd_buf;
    glGenBuffers(1, &cmd_buf);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cmd_buf);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmd), &cmd, GL_STATIC_DRAW);

    glDrawArraysIndirect(GL_TRIANGLES, 0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_p);
    glDeleteBuffers(1, &vbo_c);
    glDeleteBuffers(1, &cmd_buf);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 6. FBO switch ----
 * Render red to FBO A, then bind FBO B and render green over it.  Verify
 * the final FBO B has green in the center (proves encoder rotation + FBO
 * switch works). */
static int test_fbo_switch(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fboA = 0, fboB = 0, texA = 0, texB = 0;
    fboA = make_fbo(REG_W, REG_H, &texA);
    fboB = make_fbo(REG_W, REG_H, &texB);
    if (!fboA || !fboB) return 1;

    GLuint progRed = link_program(
        "#version 330 core\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n",
        "#version 330 core\n"
        "out vec4 frag;\n"
        "void main() { frag = vec4(1.0, 0.0, 0.0, 1.0); }\n");
    GLuint progGreen = link_program(
        "#version 330 core\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n",
        "#version 330 core\n"
        "out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n");
    if (!progRed || !progGreen) return 2;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* Initialize both targets before queuing the cross-FBO draw sequence. */
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    clear_color(0.0f, 0.0f, 0.0f);
    glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    clear_color(0.05f, 0.05f, 0.1f);

    /* Queue A then B without an intermediate clear/finish. Deferred replay
     * must rotate render encoders while retaining one command buffer. */
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    glUseProgram(progRed);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    glUseProgram(progGreen);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(progRed);
    glDeleteProgram(progGreen);
    glDeleteFramebuffers(1, &fboA);
    glDeleteFramebuffers(1, &fboB);
    glDeleteTextures(1, &texA);
    glDeleteTextures(1, &texB);
    return 0;
}

/* ---- 7. Transform Feedback ----
 * Run a vertex shader that scales positions by 0.5, capture to a TF buffer,
 * then draw the captured buffer with a solid color.  The captured triangle
 * is smaller than the original, so the image is distinct. */
static int test_transform_feedback(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    /* Pass 1: XFB capture */
    GLuint prog_xfb = glCreateProgram();
    GLuint vs = compile_shader(GL_VERTEX_SHADER, VS_XFB);
    if (!vs) return 2;
    glAttachShader(prog_xfb, vs);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS_XFB);
    if (!fs) return 2;
    glAttachShader(prog_xfb, fs);

    const char *tf_varying = "tf_pos";
    glTransformFeedbackVaryings(prog_xfb, 1, &tf_varying, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog_xfb);
    GLint ok = 0;
    glGetProgramiv(prog_xfb, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog_xfb, sizeof(log), NULL, log);
        fprintf(stderr, "  [XFB link FAIL] %s\n", log);
        return 3;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    GLuint tf_buf;
    glGenBuffers(1, &tf_buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tf_buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(TRI_VERTS), NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tf_buf);

    glUseProgram(prog_xfb);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    glFinish();

    /* Pass 2: draw captured positions */
    GLuint prog_draw = link_program(VS_XFB_DRAW, FS_BASIC);
    if (!prog_draw) return 4;
    glUseProgram(prog_draw);

    GLuint vao2;
    glGenVertexArrays(1, &vao2);
    glBindVertexArray(vao2);
    glBindBuffer(GL_ARRAY_BUFFER, tf_buf);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteVertexArrays(1, &vao2);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &tf_buf);
    glDeleteProgram(prog_xfb);
    glDeleteProgram(prog_draw);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 8. Conditional Render ----
 * Create an occlusion query, draw a triangle (visible), end query.
 * Then begin conditional render and draw a second triangle.  Since the
 * query result is non-zero, the conditional draw SHOULD execute.
 * Image: center should show the conditional (green) triangle. */
static int test_conditional_render(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(VS_BASIC, FS_BASIC);
    if (!prog) return 2;
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 1.0f);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_p = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    GLuint vbo_c = make_vbo(TRI_COLORS, sizeof(TRI_COLORS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    /* Query pass: draw the red triangle, count samples passed. */
    GLuint q;
    glGenQueries(1, &q);
    glBeginQuery(GL_SAMPLES_PASSED, q);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndQuery(GL_SAMPLES_PASSED);

    /* Wait for result so conditional render is guaranteed available. */
    GLint avail = 0;
    while (!avail) {
        glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE, &avail);
    }
    GLuint result = 0;
    glGetQueryObjectuiv(q, GL_QUERY_RESULT, &result);
    if (result == 0) {
        fprintf(stderr, "conditional: query returned 0 samples\n");
        glDeleteQueries(1, &q);
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo_p);
        glDeleteBuffers(1, &vbo_c);
        glDeleteProgram(prog);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
        return 3;
    }

    /* A draw that produces no fragments must report zero. This guards both
     * query ordering and the real Metal visibility-result path. */
    GLuint q_zero;
    glGenQueries(1, &q_zero);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 0, 0);
    glBeginQuery(GL_SAMPLES_PASSED, q_zero);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndQuery(GL_SAMPLES_PASSED);
    glDisable(GL_SCISSOR_TEST);
    GLuint zero_result = 1;
    glGetQueryObjectuiv(q_zero, GL_QUERY_RESULT, &zero_result);
    glDeleteQueries(1, &q_zero);
    if (zero_result != 0) {
        fprintf(stderr, "conditional: occluded query returned %u samples\n",
                zero_result);
        glDeleteQueries(1, &q);
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo_p);
        glDeleteBuffers(1, &vbo_c);
        glDeleteProgram(prog);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
        return 4;
    }

    /* Conditional draw: should execute (query != 0). Use green triangle offset. */
    float green[] = { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
    GLuint vbo_g = make_vbo(green, sizeof(green));
    glBindBuffer(GL_ARRAY_BUFFER, vbo_g);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBeginConditionalRender(q, GL_QUERY_WAIT);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.2f, -0.2f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndConditionalRender();
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteBuffers(1, &vbo_g);
    glDeleteQueries(1, &q);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_p);
    glDeleteBuffers(1, &vbo_c);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 9. UBO Range Switch ----
 * Queue three draws that read red, green, then red aligned slices of one UBO.
 * The middle range also changes size, so deferred replay must retain both the
 * offset and range before restoring the batch's base range. */
static int test_ubo_range_switch(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(
        "#version 330 core\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n",
        "#version 330 core\n"
        "layout(std140) uniform DrawColor { vec4 color; };\n"
        "out vec4 frag;\n"
        "void main() { frag = color; }\n");
    if (!prog) return 2;
    glUseProgram(prog);

    GLuint block = glGetUniformBlockIndex(prog, "DrawColor");
    if (block == GL_INVALID_INDEX) return 3;
    glUniformBlockBinding(prog, block, 0);

    static const float verts[] = {
        -0.9f, -0.6f,  -0.1f, -0.6f,  -0.5f, 0.6f,
         0.1f, -0.6f,   0.9f, -0.6f,   0.5f, 0.6f,
        -0.25f, 0.2f,   0.25f, 0.2f,    0.0f, 0.85f,
    };
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(verts, sizeof(verts));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    GLint alignment = 1;
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &alignment);
    size_t stride = alignment > 16 ? (size_t)alignment : 16u;
    size_t ubo_size = stride + 32u;
    unsigned char *ubo_data = (unsigned char *)calloc(1, ubo_size);
    if (!ubo_data) return 4;
    static const float red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    static const float green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
    memcpy(ubo_data, red, sizeof(red));
    memcpy(ubo_data + stride, green, sizeof(green));

    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, (GLsizeiptr)ubo_size, ubo_data, GL_STATIC_DRAW);
    free(ubo_data);

    glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, 0, 16);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, (GLintptr)stride, 32);
    glDrawArrays(GL_TRIANGLES, 3, 3);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, 0, 16);
    glDrawArrays(GL_TRIANGLES, 6, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteBuffers(1, &ubo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 10. Per-draw Vertex Binding Switch ----
 * Queue A -> B -> A indexed draws by mutating binding 0 on the same VAO.
 * This matches Minecraft/Sodium's arena-VBO path: deferred replay must keep
 * an immutable buffer/offset override for each command. */
static int test_vao_binding_switch(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 p;\n"
        "layout(location=1) in vec4 c;\n"
        "out vec4 color;\n"
        "void main(){ color=c; gl_Position=vec4(p + vec2(float(gl_VertexID) * 0.0),0.0,1.0); }\n",
        "#version 330 core\n"
        "in vec4 color; out vec4 frag;\n"
        "void main(){ frag=color; }\n");
    if (!prog) return 2;
    glUseProgram(prog);

    typedef struct {
        float x, y;
        uint8_t r, g, b, a;
    } PackedVertex;
    static const PackedVertex verts_a[] = {
        {9.0f, 9.0f, 9, 9, 9, 9}, /* non-zero batch base skips this vertex */
        {-0.9f, -0.6f, 255, 0, 0, 255}, {-0.1f, -0.6f, 255, 0, 0, 255},
        {-0.5f,  0.2f, 255, 0, 0, 255}, {-0.25f, 0.25f, 255, 0, 0, 255},
        { 0.25f, 0.25f, 255, 0, 0, 255}, { 0.0f,  0.85f, 255, 0, 0, 255},
    };
    static const PackedVertex verts_b[] = {
        {9.0f, 9.0f, 9, 9, 9, 9}, /* base offset */
        {9.0f, 9.0f, 9, 9, 9, 9}, /* per-draw offset delta */
        {0.1f, -0.6f, 0, 255, 0, 255}, {0.9f, -0.6f, 0, 255, 0, 255},
        {0.5f,  0.2f, 0, 255, 0, 255}, {0.1f,  0.3f, 0, 255, 0, 255},
        {0.9f,  0.3f, 0, 255, 0, 255}, {0.5f,  0.9f, 0, 255, 0, 255},
    };
    static const unsigned short indices[] = { 0, 1, 2, 3, 4, 5 };

    GLuint vao, ebos[2];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, ebos);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(unsigned short), indices,
                 GL_STATIC_DRAW);

    GLuint vbos[2];
    glGenBuffers(2, vbos);
    const void *vertex_data[] = { verts_a, verts_b };
    const size_t vertex_bytes[] = { sizeof(verts_a), sizeof(verts_b) };
    for (int i = 0; i < 2; i++) {
        glBindBuffer(GL_ARRAY_BUFFER, vbos[i]);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vertex_bytes[i],
                     vertex_data[i], GL_STATIC_DRAW);
    }
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribFormat(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, 2 * sizeof(float));
    glVertexAttribBinding(0, 0);
    glVertexAttribBinding(1, 0);

    glBindVertexBuffer(0, vbos[0], sizeof(PackedVertex), sizeof(PackedVertex));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[0]);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, 0);
    glBindVertexBuffer(0, vbos[1], 2 * sizeof(PackedVertex), sizeof(PackedVertex));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[1]);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, 0);
    glBindVertexBuffer(0, vbos[0], sizeof(PackedVertex), sizeof(PackedVertex));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[0]);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT,
                   (void *)(3 * sizeof(unsigned short)));

    /* Format/enable mutations remain VAO hazards and must flush. */
    glDisableVertexAttribArray(1);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteBuffers(2, vbos);
    glDeleteBuffers(2, ebos);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* Exercise the AGX workarounds for padded 3D uploads and non-zero-origin
 * glCopyImageSubData. The internal authority flag forces the copy source
 * through Metal readback instead of the otherwise-valid CPU fast path. */
static int test_agx_3d_texture_workarounds(unsigned char *pixels,
                                           const char *out_path)
{
    (void)out_path;
    int rc = 0;
    GLuint fbo = 0, color_tex = 0, program = 0, vao = 0, vbo = 0;
    GLuint textures[2] = {0, 0};

    fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.05f, 0.05f, 0.05f);

    program = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p,0.0,1.0); }\n",
        "#version 330 core\n"
        "uniform sampler3D u_tex; out vec4 frag;\n"
        "void main(){ frag=texture(u_tex,vec3(0.75)); }\n");
    if (!program) {
        rc = 2;
        goto cleanup;
    }
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_tex"), 0);

    static const float verts[] = {
        -0.95f, -0.80f,  -0.05f, -0.80f,  -0.50f, 0.80f,
         0.05f, -0.80f,   0.95f, -0.80f,   0.50f, 0.80f,
    };
    make_pos2_vao(verts, sizeof(verts), &vao, &vbo);

    enum {
        texture_width = 2,
        texture_height = 2,
        texture_depth = 2,
        unpack_row_length = 3,
        unpack_image_height = 3,
        unpack_texels = unpack_row_length * unpack_image_height * texture_depth
    };
    unsigned char padded_source[unpack_texels * 4];
    unsigned char empty_destination[texture_width * texture_height *
                                    texture_depth * 4];
    memset(padded_source, 0, sizeof(padded_source));
    memset(empty_destination, 0, sizeof(empty_destination));

    /* The sampled corner is red; a different texel in the second depth plane
     * is blue and becomes the copy source below. */
    size_t sampled_texel =
        ((1u * unpack_image_height + 1u) * unpack_row_length + 1u) * 4u;
    padded_source[sampled_texel + 0] = 255;
    padded_source[sampled_texel + 3] = 255;
    size_t copied_texel =
        ((1u * unpack_image_height + 0u) * unpack_row_length + 0u) * 4u;
    padded_source[copied_texel + 2] = 255;
    padded_source[copied_texel + 3] = 255;

    glGenTextures(2, textures);
    glBindTexture(GL_TEXTURE_3D, textures[0]);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, unpack_row_length);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, unpack_image_height);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8,
                 texture_width, texture_height, texture_depth, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, padded_source);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0);

    glBindTexture(GL_TEXTURE_3D, textures[1]);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8,
                 texture_width, texture_height, texture_depth, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, empty_destination);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0);

    glBindTexture(GL_TEXTURE_3D, textures[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "agx_3d_texture_workarounds: padded upload/draw failed\n");
        rc = 3;
        goto cleanup;
    }

    GLMContext glm_ctx = MGLgetCurrentContext();
    Texture *internal_source = glm_ctx
        ? (Texture *)searchHashTable(&glm_ctx->active_state->texture_table,
                                     textures[0])
        : NULL;
    if (!internal_source || !internal_source->faces[0].levels) {
        fprintf(stderr, "agx_3d_texture_workarounds: source texture lookup failed\n");
        rc = 4;
        goto cleanup;
    }
    internal_source->metal_data_authoritative = GL_TRUE;
    internal_source->faces[0].levels[0].metal_data_authoritative = GL_TRUE;

    glCopyImageSubData(textures[0], GL_TEXTURE_3D, 0, 0, 0, 1,
                       textures[1], GL_TEXTURE_3D, 0, 1, 1, 1,
                       1, 1, 1);
    GLenum copy_error = glGetError();
    if (copy_error != GL_NO_ERROR) {
        fprintf(stderr,
                "agx_3d_texture_workarounds: 3D copy failed (error=0x%x)\n",
                copy_error);
        rc = 5;
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_3D, textures[1]);
    glDrawArrays(GL_TRIANGLES, 3, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "agx_3d_texture_workarounds: destination draw/read failed\n");
        rc = 6;
        goto cleanup;
    }

    static const unsigned char expected[2][3] = {
        {255, 0, 0},
        {0, 0, 255},
    };
    static const int xs[2] = {REG_W / 4, 3 * REG_W / 4};
    for (int i = 0; i < 2; i++) {
        const unsigned char *actual =
            &pixels[((REG_H / 2) * REG_W + xs[i]) * 4];
        for (int c = 0; c < 3; c++) {
            int delta = (int)actual[c] - (int)expected[i][c];
            if (delta < -2 || delta > 2) {
                fprintf(stderr,
                        "agx_3d_texture_workarounds: pixel %d expected "
                        "rgb=(%u,%u,%u), got (%u,%u,%u)\n",
                        i, expected[i][0], expected[i][1], expected[i][2],
                        actual[0], actual[1], actual[2]);
                rc = 7;
                goto cleanup;
            }
        }
    }

cleanup:
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
    if (textures[0] || textures[1]) glDeleteTextures(2, textures);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color_tex) glDeleteTextures(1, &color_tex);
    return rc;
}

/* ---- 11. Texture + Vertex Binding Switch ----
 * Queue A -> B -> A sampled draws while advancing one VBO binding per draw.
 * B is first uploaded during deferred replay, so encoder restoration must not
 * lose that draw's dynamic vertex offset. Updating B afterwards must still
 * flush those reads, so the queued B draw remains green while the post-update
 * B draw observes blue. */
static int test_texture_binding_switch(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, color_tex;
    fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p,0.0,1.0); }\n",
        "#version 330 core\n"
        "uniform sampler2D u_tex; out vec4 frag;\n"
        "void main(){ frag=texture(u_tex,vec2(0.5)); }\n");
    if (!prog) return 2;
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "u_tex"), 0);

    static const float verts[] = {
        -0.95f, -0.90f,  -0.05f, -0.90f,  -0.50f, -0.10f,
         0.05f, -0.90f,   0.95f, -0.90f,   0.50f, -0.10f,
        -0.95f,  0.10f,  -0.05f,  0.10f,  -0.50f,  0.90f,
         0.05f,  0.10f,   0.95f,  0.10f,   0.50f,  0.90f,
    };
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(verts, sizeof(verts));
    glEnableVertexAttribArray(0);
    glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(0, 0);

    enum { switch_tex_size = 16, switch_tex_bytes = 16 * 16 * 4 };
    unsigned char red[switch_tex_bytes];
    unsigned char green[switch_tex_bytes];
    unsigned char blue[switch_tex_bytes];
    for (size_t i = 0; i < switch_tex_bytes; i += 4) {
        red[i + 0] = 255; red[i + 1] = 0;   red[i + 2] = 0;   red[i + 3] = 255;
        green[i + 0] = 0; green[i + 1] = 255; green[i + 2] = 0; green[i + 3] = 255;
        blue[i + 0] = 0; blue[i + 1] = 0;  blue[i + 2] = 255; blue[i + 3] = 255;
    }
    GLuint textures[2];
    glGenTextures(2, textures);
    const unsigned char *initial[] = { red, green };
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, textures[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     switch_tex_size, switch_tex_size, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, initial[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    /* Keep B dirty until deferred replay without flushing any queued draw. */
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    switch_tex_size, switch_tex_size,
                    GL_RGBA, GL_UNSIGNED_BYTE, green);

    const GLsizei vertex_stride = 2 * (GLsizei)sizeof(float);
    glBindVertexBuffer(0, vbo, 0, vertex_stride);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexBuffer(0, vbo, 3 * vertex_stride, vertex_stride);
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexBuffer(0, vbo, 6 * vertex_stride, vertex_stride);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexBuffer(0, vbo, 9 * vertex_stride, vertex_stride);
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    switch_tex_size, switch_tex_size,
                    GL_RGBA, GL_UNSIGNED_BYTE, blue);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteTextures(2, textures);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_tex);
    return 0;
}

/* ---- 12. Per-draw texture sampler parameters ----
 * Queue CLAMP_TO_BORDER -> CLAMP_TO_EDGE -> CLAMP_TO_BORDER on one texture.
 * The three draws must retain white -> red -> white sampler state without a
 * parameter mutation flushing the earlier draws. */
static int test_texture_parameter_switch(unsigned char *pixels,
                                         const char *out_path)
{
    (void)out_path;
    int rc = 0;
    GLuint color_tex = 0;
    GLuint fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.05f, 0.05f, 0.05f);

    GLuint program = make_sampler_test_program(1);
    if (!program) return 2;
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_tex"), 0);

    GLuint vao = 0, vbo = 0;
    make_sampler_switch_vao(&vao, &vbo);

    static const unsigned char red[4] = { 255, 0, 0, 255 };
    GLuint sampled_tex = make_rgba8_texture(red);
    static const GLfloat white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, white);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glDrawArrays(GL_TRIANGLES, 3, 3);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glDrawArrays(GL_TRIANGLES, 6, 3);

    GLMContext ctx = MGLgetCurrentContext();
    if (mglSamplerSnapshotEnabled() &&
        (!ctx || ctx->draw_command_buffer.total_commands != 3u)) {
        fprintf(stderr,
                "texture_parameter_switch: parameter mutation flushed queued draws "
                "(pending=%u)\n",
                ctx ? ctx->draw_command_buffer.total_commands : 0u);
        rc = 3;
    }

    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (rc == 0 && verify_sampler_switch_pixels(
            pixels, "texture_parameter_switch") != 0) {
        rc = 4;
    }

    glDeleteTextures(1, &sampled_tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_tex);
    return rc;
}

/* ---- 13. Per-draw sampler-object parameters ----
 * Mutate one bound sampler object through the same A -> B -> A sequence. The
 * command snapshot must contain sampler values, not the mutable GL object. */
static int test_sampler_parameter_switch(unsigned char *pixels,
                                         const char *out_path)
{
    (void)out_path;
    int rc = 0;
    GLuint color_tex = 0;
    GLuint fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.05f, 0.05f, 0.05f);

    GLuint program = make_sampler_test_program(1);
    if (!program) return 2;
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_tex"), 0);

    GLuint vao = 0, vbo = 0;
    make_sampler_switch_vao(&vao, &vbo);

    static const unsigned char red[4] = { 255, 0, 0, 255 };
    GLuint sampled_tex = make_rgba8_texture(red);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    GLuint sampler = 0;
    glGenSamplers(1, &sampler);
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    static const GLfloat white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glSamplerParameterfv(sampler, GL_TEXTURE_BORDER_COLOR, white);
    glBindSampler(0, sampler);

    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glDrawArrays(GL_TRIANGLES, 3, 3);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glDrawArrays(GL_TRIANGLES, 6, 3);

    GLMContext ctx = MGLgetCurrentContext();
    if (mglSamplerSnapshotEnabled() &&
        (!ctx || ctx->draw_command_buffer.total_commands != 3u)) {
        fprintf(stderr,
                "sampler_parameter_switch: parameter mutation flushed queued draws "
                "(pending=%u)\n",
                ctx ? ctx->draw_command_buffer.total_commands : 0u);
        rc = 3;
    }

    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (rc == 0 && verify_sampler_switch_pixels(
            pixels, "sampler_parameter_switch") != 0) {
        rc = 4;
    }

    glBindSampler(0, 0);
    glDeleteSamplers(1, &sampler);
    glDeleteTextures(1, &sampled_tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_tex);
    return rc;
}

/* ---- 14. Same-value sampler setters ----
 * Repeating an already-current texture and sampler value is a no-op: it must
 * neither flush the queued draw nor allocate another snapshot key/set. */
static int test_sampler_same_value_no_flush(unsigned char *pixels,
                                            const char *out_path)
{
    (void)out_path;
    int rc = 0;
    GLuint color_tex = 0;
    GLuint fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);

    GLuint program = make_sampler_test_program(1);
    if (!program) return 2;
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_tex"), 0);

    GLuint vao = 0, vbo = 0;
    make_sampler_switch_vao(&vao, &vbo);
    static const unsigned char red[4] = { 255, 0, 0, 255 };
    GLuint sampled_tex = make_rgba8_texture(red);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    GLuint sampler = 0;
    glGenSamplers(1, &sampler);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glBindSampler(0, sampler);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    GLMContext ctx = MGLgetCurrentContext();
    if (!ctx || ctx->draw_command_buffer.total_commands != 1u) {
        rc = 3;
    } else {
        MGLCommandBuffer *cb = &ctx->draw_command_buffer;
        uint32_t total_commands = cb->total_commands;
        uint32_t batch_count = cb->batch_count;
        uint16_t key_count = cb->sampler_snapshot_key_count;
        uint16_t set_count = cb->sampler_snapshot_set_count;

        for (int i = 0; i < 1000; i++) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        }

        Texture *bound_texture =
            ctx->active_state->texture_units[0].textures[_TEXTURE_2D];
        Sampler *bound_sampler = ctx->active_state->texture_samplers[0];
        GLint texture_wrap = bound_texture ? bound_texture->params.wrap_s : 0;
        GLint sampler_wrap = bound_sampler ? bound_sampler->params.wrap_s : 0;
        GLenum error = glGetError();
        if (error != GL_NO_ERROR ||
            texture_wrap != GL_CLAMP_TO_EDGE ||
            sampler_wrap != GL_CLAMP_TO_EDGE ||
            !command_buffer_counts_equal(cb, total_commands, batch_count,
                                         key_count, set_count)) {
            fprintf(stderr,
                    "sampler_same_value_no_flush: error=0x%x tex=0x%x sampler=0x%x "
                    "pending=%u/%u sets=%u/%u\n",
                    error, texture_wrap, sampler_wrap,
                    cb->total_commands, total_commands,
                    cb->sampler_snapshot_set_count, set_count);
            rc = 4;
        }
    }

    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindSampler(0, 0);
    glDeleteSamplers(1, &sampler);
    glDeleteTextures(1, &sampled_tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_tex);
    return rc;
}

/* ---- 15. Invalid sampler setters ----
 * Validation happens against a candidate copy. Invalid values must leave both
 * objects unchanged and must not flush or mutate the snapshot pools. */
static int test_sampler_invalid_no_flush(unsigned char *pixels,
                                         const char *out_path)
{
    (void)out_path;
    int rc = 0;
    GLuint color_tex = 0;
    GLuint fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);

    GLuint program = make_sampler_test_program(1);
    if (!program) return 2;
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_tex"), 0);

    GLuint vao = 0, vbo = 0;
    make_sampler_switch_vao(&vao, &vbo);
    static const unsigned char red[4] = { 255, 0, 0, 255 };
    GLuint sampled_tex = make_rgba8_texture(red);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD, 0.75f);

    GLuint sampler = 0;
    glGenSamplers(1, &sampler);
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameterf(sampler, GL_TEXTURE_MIN_LOD, 0.75f);
    glBindSampler(0, sampler);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    GLMContext ctx = MGLgetCurrentContext();
    if (!ctx || ctx->draw_command_buffer.total_commands != 1u) {
        rc = 3;
    } else {
        MGLCommandBuffer *cb = &ctx->draw_command_buffer;
        uint32_t total_commands = cb->total_commands;
        uint32_t batch_count = cb->batch_count;
        uint16_t key_count = cb->sampler_snapshot_key_count;
        uint16_t set_count = cb->sampler_snapshot_set_count;

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x7fffffff);
        GLenum texture_error = glGetError();

        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, 0x7fffffff);
        GLenum sampler_error = glGetError();

        GLint queried_texture_wrap = -1;
        GLint queried_sampler_wrap = -1;
        GLfloat queried_texture_lod = -1.0f;
        GLfloat queried_sampler_lod = -1.0f;
        glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                            &queried_texture_wrap);
        glGetSamplerParameteriv(sampler, GL_TEXTURE_WRAP_S,
                                &queried_sampler_wrap);
        glGetTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD,
                            &queried_texture_lod);
        glGetSamplerParameterfv(sampler, GL_TEXTURE_MIN_LOD,
                                &queried_sampler_lod);
        GLenum query_error = glGetError();

        Texture *bound_texture =
            ctx->active_state->texture_units[0].textures[_TEXTURE_2D];
        Sampler *bound_sampler = ctx->active_state->texture_samplers[0];
        GLint texture_wrap = bound_texture ? bound_texture->params.wrap_s : 0;
        GLint sampler_wrap = bound_sampler ? bound_sampler->params.wrap_s : 0;

        if (texture_error != GL_INVALID_ENUM ||
            sampler_error != GL_INVALID_ENUM || query_error != GL_NO_ERROR ||
            texture_wrap != GL_CLAMP_TO_EDGE ||
            sampler_wrap != GL_CLAMP_TO_EDGE ||
            queried_texture_wrap != GL_CLAMP_TO_EDGE ||
            queried_sampler_wrap != GL_CLAMP_TO_EDGE ||
            queried_texture_lod != 0.75f || queried_sampler_lod != 0.75f ||
            !command_buffer_counts_equal(cb, total_commands, batch_count,
                                         key_count, set_count)) {
            fprintf(stderr,
                    "sampler_invalid_no_flush: errors=0x%x/0x%x/0x%x "
                    "tex=0x%x/0x%x sampler=0x%x/0x%x lod=%.2f/%.2f "
                    "pending=%u/%u sets=%u/%u\n",
                    texture_error, sampler_error, query_error,
                    texture_wrap, queried_texture_wrap,
                    sampler_wrap, queried_sampler_wrap,
                    queried_texture_lod, queried_sampler_lod,
                    cb->total_commands, total_commands,
                    cb->sampler_snapshot_set_count, set_count);
            rc = 4;
        }
    }

    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindSampler(0, 0);
    glDeleteSamplers(1, &sampler);
    glDeleteTextures(1, &sampled_tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_tex);
    return rc;
}

/* ---- 16. Snapshot-set pool overflow ----
 * Seventeen values across two active samplers provide 257 unique pairs while
 * using only 17 unique keys. Draw 257 must flush the full 256-set command
 * buffer and retry capture into an empty buffer. */
static int test_sampler_snapshot_overflow(unsigned char *pixels,
                                          const char *out_path)
{
    (void)out_path;
    if (!mglSamplerSnapshotEnabled()) {
        memset(pixels, 0, REG_W * REG_H * 4);
        return 0;
    }

    int rc = 0;
    GLuint color_tex = 0;
    GLuint fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);

    GLuint program = make_sampler_test_program(2);
    if (!program) return 2;
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_tex0"), 0);
    glUniform1i(glGetUniformLocation(program, "u_tex1"), 1);

    static const float fullscreen[] = {
        -1.0f, -1.0f,  3.0f, -1.0f,  -1.0f, 3.0f,
    };
    GLuint vao = 0, vbo = 0;
    make_pos2_vao(fullscreen, sizeof(fullscreen), &vao, &vbo);

    static const unsigned char red[4] = { 255, 0, 0, 255 };
    GLuint textures[2] = { 0, 0 };
    for (int unit = 0; unit < 2; unit++) {
        glActiveTexture(GL_TEXTURE0 + unit);
        textures[unit] = make_rgba8_texture(red);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    }

    GLMContext ctx = MGLgetCurrentContext();
    if (!ctx) {
        rc = 3;
    } else {
        MGLCommandBuffer *cb = &ctx->draw_command_buffer;
        for (int draw = 0; draw <= MGL_MAX_SAMPLER_SNAPSHOT_SETS; draw++) {
            int first_value = draw / 16;
            int second_value = draw % 16;
            glActiveTexture(GL_TEXTURE0);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD,
                            (GLfloat)first_value * 0.01f);
            glActiveTexture(GL_TEXTURE1);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD,
                            (GLfloat)second_value * 0.01f);

            glDrawArrays(GL_TRIANGLES, 0, 3);

            if (draw == MGL_MAX_SAMPLER_SNAPSHOT_SETS - 1 &&
                (cb->sampler_snapshot_set_count !=
                     MGL_MAX_SAMPLER_SNAPSHOT_SETS ||
                 cb->total_commands != MGL_MAX_SAMPLER_SNAPSHOT_SETS ||
                 cb->sampler_snapshot_key_count != 16u)) {
                fprintf(stderr,
                        "sampler_snapshot_overflow: pool did not fill as expected "
                        "(commands=%u sets=%u keys=%u)\n",
                        cb->total_commands, cb->sampler_snapshot_set_count,
                        cb->sampler_snapshot_key_count);
                rc = 4;
                break;
            }
            if (draw == MGL_MAX_SAMPLER_SNAPSHOT_SETS &&
                (cb->total_commands != 1u ||
                 cb->sampler_snapshot_set_count != 1u ||
                 cb->sampler_snapshot_key_count != 2u ||
                 cb->sampler_snapshot_incomplete)) {
                fprintf(stderr,
                        "sampler_snapshot_overflow: retry did not reset the pool "
                        "(commands=%u sets=%u keys=%u incomplete=%d)\n",
                        cb->total_commands, cb->sampler_snapshot_set_count,
                        cb->sampler_snapshot_key_count,
                        cb->sampler_snapshot_incomplete ? 1 : 0);
                rc = 5;
            }
        }
    }

    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    const unsigned char *center = &pixels[((REG_H / 2) * REG_W + REG_W / 2) * 4];
    if (rc == 0 &&
        (center[0] < 253 || center[1] > 2 || center[2] > 2)) {
        fprintf(stderr,
                "sampler_snapshot_overflow: rendered center is (%u,%u,%u)\n",
                center[0], center[1], center[2]);
        rc = 6;
    }

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(2, textures);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_tex);
    return rc;
}

/* ---- 9. Program switch (PSO cache + rebind) ----
 * Draw with program A (vertex-color), switch to program B (solid uniform
 * color), then switch back to A. Exercises the pipeline cache lookup, the
 * _lastPipelineState rebind path, and PSO reuse when A is re-bound. */
static int test_program_switch(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint progA = link_program(VS_BASIC, FS_BASIC);   /* vertex color */
    GLuint progB = link_program(VS_BASIC, FS_SOLID);   /* uniform color */
    if (!progA || !progB) return 2;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_p = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    GLuint vbo_c = make_vbo(TRI_COLORS, sizeof(TRI_COLORS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    /* A: red vertex-color triangle, left */
    glUseProgram(progA);
    glUniform1f(glGetUniformLocation(progA, "u_scale"), 0.5f);
    glUniform2f(glGetUniformLocation(progA, "u_offset"), -0.35f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* B: solid green triangle, right */
    glUseProgram(progB);
    glUniform1f(glGetUniformLocation(progB, "u_scale"), 0.5f);
    glUniform2f(glGetUniformLocation(progB, "u_offset"), 0.35f, 0.0f);
    glUniform4f(glGetUniformLocation(progB, "u_color"), 0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* A again: PSO cache should have A's pipeline; draw top */
    glUseProgram(progA);
    glUniform1f(glGetUniformLocation(progA, "u_scale"), 0.4f);
    glUniform2f(glGetUniformLocation(progA, "u_offset"), 0.0f, 0.4f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_p);
    glDeleteBuffers(1, &vbo_c);
    glDeleteProgram(progA);
    glDeleteProgram(progB);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 10. Blend state ----
 * Draw an opaque blue background quad, then a semi-transparent red triangle
 * with GL_SRC_ALPHA / GL_ONE_MINUS_SRC_ALPHA. The overlap must show a
 * purple blend, proving the pipeline descriptor's blend attachment state is
 * applied (DIRTY_ALPHA_STATE -> bindBlendStateToPipelineStateDescriptor). */
static int test_blend(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);

    GLuint prog = link_program(
        "#version 330 core\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "uniform vec2 u_offset;\n"
        "uniform float u_scale;\n"
        "void main() { gl_Position = vec4(a_pos * u_scale + u_offset, 0.0, 1.0); }\n",
        FS_SOLID);
    if (!prog) return 2;
    glUseProgram(prog);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_q = make_vbo(QUAD_VERTS, sizeof(QUAD_VERTS));
    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(QUAD_INDICES), QUAD_INDICES, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_q);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* Opaque blue quad, blending off */
    glDisable(GL_BLEND);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 1.0f);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 0.0f, 0.0f, 1.0f, 1.0f);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

    /* Semi-transparent red triangle over it, blending on */
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    GLuint vbo_t = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glBindBuffer(GL_ARRAY_BUFFER, vbo_t);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 1.0f, 0.0f, 0.0f, 0.5f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_q);
    glDeleteBuffers(1, &vbo_t);
    glDeleteBuffers(1, &ibo);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- 11. Depth test ----
 * With GL_DEPTH_TEST + GL_LESS, draw a far red triangle then a near green
 * triangle at overlapping XY. The near one must win in the overlap, and a
 * third far-behind blue triangle must be occluded. Exercises the
 * depth-stencil state binding in updateCurrentRenderEncoder. */
static const char *VS_DEPTH =
    "#version 330 core\n"
    "layout(location = 0) in vec2 a_pos;\n"
    "uniform vec2 u_offset;\n"
    "uniform float u_scale;\n"
    "uniform float u_depth;\n"
    "void main() { gl_Position = vec4(a_pos * u_scale + u_offset, u_depth, 1.0); }\n";

/* Minimal depth-occlusion probe with NO per-draw uniforms: two programs each
 * hardcode gl_Position.z and frag color, same XY geometry. Isolates depth
 * testing from the uniform-layout confound seen in test_depth_test. */
static GLuint make_fbo_depth_tex(int w, int h, GLuint *out_tex, GLuint *out_depth);

/* Shared cross-stage uniform diagnostic: the SAME uniform name (u_tint) is
 * declared in BOTH vertex and fragment. A shared uniform must resolve to ONE
 * GL location and ONE plain_uniform_buffers slot, so writing it once feeds
 * both stages. If pass 1 hands the two stages different locations, the vertex
 * and fragment read different data. Render: vertex nudges X by u_tint.x,
 * fragment colors by u_tint -> a single teal-ish triangle offset right.
 * Gate: guards that a same-named cross-stage uniform stays on ONE location /
 * plain_uniform_buffers slot (the sameName branch of the location fix). */
static int test_shared_uniform(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 a_pos;\n"
        "uniform vec4 u_tint;\n"
        "void main(){ gl_Position=vec4(a_pos*0.5 + vec2(u_tint.x,0.0),0.0,1.0); }\n",
        "#version 330 core\n"
        "uniform vec4 u_tint;\n"
        "out vec4 f; void main(){ f=vec4(0.0,u_tint.g,u_tint.b,1.0); }\n");
    if (!prog) return 2;
    glUseProgram(prog);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* x=0.4 shifts right; g=1,b=1 -> teal fragment. One write, both stages. */
    glUniform4f(glGetUniformLocation(prog, "u_tint"), 0.4f, 1.0f, 1.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* Cross-stage uniform-location collision gate. The vertex shader (VS_DEPTH)
 * has u_offset/u_scale/u_depth; the fragment shader (FS_SOLID) has u_color.
 * SPIR-V numbers default-block uniforms per stage, so u_offset (vertex) and
 * u_color (fragment) both reflect location 0. Before the link-time fix they
 * shared plain_uniform_buffers[0], so writing u_color clobbered u_offset and
 * the two triangles landed at different XY. Correct behavior: u_offset stays
 * (0,0) for both draws, so the two triangles fully overlap at center — draw 2
 * (green) covers draw 1 (red). Golden must show green-only at center, no red. */
static int test_uniform_alias(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(VS_DEPTH, FS_SOLID);   /* vec2 u_offset; float u_scale; float u_depth */
    if (!prog) return 2;
    glUseProgram(prog);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* Fix offset + scale ONCE. */
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 0.4f);

    /* Draw 1: red, u_depth=0.0 */
    glUniform1f(glGetUniformLocation(prog, "u_depth"), 0.0f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 1.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Draw 2: green, ONLY u_depth changes (0.0 -> 0.9). Offset untouched.
     * Must land on top of the red triangle (same XY). */
    glUniform1f(glGetUniformLocation(prog, "u_depth"), 0.9f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

__attribute__((unused))
static int test_depth_probe(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex, dtex;
    fbo = make_fbo_depth_tex(REG_W, REG_H, &tex, &dtex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    /* Far red at z=0.6 */
    GLuint progFar = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 a_pos;\n"
        "void main(){ gl_Position=vec4(a_pos*0.7,0.6,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(1.0,0.0,0.0,1.0); }\n");
    /* Near green at z=-0.2 */
    GLuint progNear = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 a_pos;\n"
        "void main(){ gl_Position=vec4(a_pos*0.7,-0.2,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(0.0,1.0,0.0,1.0); }\n");
    if (!progFar || !progNear) return 2;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    /* Draw far red first, then near green. With working depth, green wins
     * the (identical) overlap. Without depth, green wins anyway (drawn last),
     * so to distinguish we ALSO draw a far blue LAST: */
    glUseProgram(progFar);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUseProgram(progNear);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Far blue drawn LAST at z=0.9: must be occluded by near green if depth
     * works; if depth is off, blue (last) overwrites everything. */
    GLuint progFarLast = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 a_pos;\n"
        "void main(){ gl_Position=vec4(a_pos*0.7,0.9,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(0.0,0.0,1.0,1.0); }\n");
    if (!progFarLast) return 3;
    glUseProgram(progFarLast);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glDisable(GL_DEPTH_TEST);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(progFar);
    glDeleteProgram(progNear);
    glDeleteProgram(progFarLast);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &dtex);
    return 0;
}

/* ---- Stencil probe (probe style: NO per-draw uniforms) ----
 * Pass 1: small triangle (scale baked into progMask), stencil ALWAYS->REPLACE
 * ref=1, colorMask off — writes the stencil mask only.
 * Pass 2: full-size green quad (scale baked into progFill), stencil EQUAL 1 —
 * only passes inside the mask triangle. Result: green clipped to the triangle.
 * Two separate programs, each with hardcoded scale/color, so no same-program
 * multi-draw uniform changes (avoids the deferred-uniform bug). */
static int test_stencil_probe(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo = 0, tex = 0, rbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, REG_W, REG_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, REG_W, REG_H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return 1;

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    /* progMask: small white triangle (scale 0.5 baked in) */
    GLuint progMask = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 a_pos;\n"
        "void main(){ gl_Position=vec4(a_pos*0.5,0.0,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(1.0,1.0,1.0,1.0); }\n");
    /* progFill: full-size green quad (scale 1.0 baked in) */
    GLuint progFill = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 a_pos;\n"
        "void main(){ gl_Position=vec4(a_pos,0.0,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(0.0,1.0,0.0,1.0); }\n");
    if (!progMask || !progFill) return 2;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_t = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    GLuint vbo_q = make_vbo(QUAD_VERTS, sizeof(QUAD_VERTS));
    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(QUAD_INDICES), QUAD_INDICES, GL_STATIC_DRAW);

    glEnable(GL_STENCIL_TEST);

    /* Pass 1: write stencil=1 in the small triangle, color off. */
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glStencilMask(0xFF);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glUseProgram(progMask);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_t);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Pass 2: green quad only where stencil==1. */
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilMask(0x00);
    glUseProgram(progFill);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_q);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

    glDisable(GL_STENCIL_TEST);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_t);
    glDeleteBuffers(1, &vbo_q);
    glDeleteBuffers(1, &ibo);
    glDeleteProgram(progMask);
    glDeleteProgram(progFill);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteRenderbuffers(1, &rbo);
    return 0;
}

/* FBO with color texture + DEPTH texture (matches Minecraft's glFramebufferTexture2D
 * depth usage, unlike make_fbo which uses a depth renderbuffer). */
static GLuint make_fbo_depth_tex(int w, int h, GLuint *out_tex, GLuint *out_depth)
{
    GLuint fbo, tex, dtex;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    glGenTextures(1, &dtex);
    glBindTexture(GL_TEXTURE_2D, dtex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dtex, 0);

    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "  [depth-tex FBO incomplete: 0x%x]\n", st);
        return 0;
    }
    if (out_tex) *out_tex = tex;
    if (out_depth) *out_depth = dtex;
    return fbo;
}

__attribute__((unused))
static int test_depth_test(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex, dtex;
    fbo = make_fbo_depth_tex(REG_W, REG_H, &tex, &dtex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);

    GLuint prog = link_program(VS_DEPTH, FS_SOLID);
    if (!prog) return 2;
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 0.7f);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    /* Far red at z=0.5 */
    glUniform1f(glGetUniformLocation(prog, "u_depth"), 0.5f);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 1.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Near green at z=-0.5 (should win the overlap) */
    glUniform1f(glGetUniformLocation(prog, "u_depth"), -0.5f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Behind blue at z=0.9 (should be fully occluded) */
    glUniform1f(glGetUniformLocation(prog, "u_depth"), 0.9f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 0.0f, 0.0f, 1.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glDisable(GL_DEPTH_TEST);
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &dtex);
    return 0;
}

/* ---- 12. Stencil test ----
 * Write a stencil mask by drawing a small triangle with stencil always-replace
 * (ref=1), then draw a large quad that only passes where stencil==1. The
 * result is the large quad clipped to the mask triangle. Exercises the
 * stencil portion of depth-stencil state and per-draw stencil ref. */
__attribute__((unused))
static int test_stencil(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo = 0, tex = 0, rbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, REG_W, REG_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    /* Depth24 + Stencil8 packed */
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, REG_W, REG_H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return 1;

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    GLuint prog = link_program(
        "#version 330 core\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "uniform float u_scale;\n"
        "void main() { gl_Position = vec4(a_pos * u_scale, 0.0, 1.0); }\n",
        FS_SOLID);
    if (!prog) return 2;
    glUseProgram(prog);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo_t = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    GLuint vbo_q = make_vbo(QUAD_VERTS, sizeof(QUAD_VERTS));
    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(QUAD_INDICES), QUAD_INDICES, GL_STATIC_DRAW);

    glEnable(GL_STENCIL_TEST);

    /* Pass 1: write stencil=1 in the small triangle, don't touch color. */
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glStencilMask(0xFF);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_t);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 0.5f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 1.0f, 1.0f, 1.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Pass 2: draw a big green quad only where stencil==1. */
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilMask(0x00);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_q);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 1.0f);
    glUniform4f(glGetUniformLocation(prog, "u_color"), 0.0f, 1.0f, 0.0f, 1.0f);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

    glDisable(GL_STENCIL_TEST);
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_t);
    glDeleteBuffers(1, &vbo_q);
    glDeleteBuffers(1, &ibo);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteRenderbuffers(1, &rbo);
    return 0;
}

/* ---- Multi-pass resume (Stage 4.2 DontCare safety net) ----
 * Pass 1: draw a RED triangle (left) into FBO A.
 * Then bind FBO B and draw into it — this forces the render encoder to rotate
 * away from A.
 * Pass 2: bind A again WITHOUT clearing and draw a GREEN triangle (right).
 * A's pass-2 load must preserve pass-1 content. If DontCare inference wrongly
 * fires on the RESUME of A (not its first use this frame), pass 1's red
 * triangle is discarded. Correct result: A contains BOTH red (left) and green
 * (right). Guards that DontCare only applies to a frame's first use of an
 * attachment, never to a resume. */
static int test_multipass_resume(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fboA, texA, fboB, texB;
    fboA = make_fbo(REG_W, REG_H, &texA);
    fboB = make_fbo(REG_W, REG_H, &texB);
    if (!fboA || !fboB) return 1;

    /* Two hardcoded-color programs, position shifted by a baked constant. */
    GLuint progRedLeft = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p*0.4 + vec2(-0.4,0.0),0.0,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(1.0,0.0,0.0,1.0); }\n");
    GLuint progGreenRight = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p*0.4 + vec2(0.4,0.0),0.0,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(0.0,1.0,0.0,1.0); }\n");
    GLuint progBlue = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p*0.4,0.0,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(0.0,0.0,1.0,1.0); }\n");
    if (!progRedLeft || !progGreenRight || !progBlue) return 2;

    GLuint vao, vbo;
    make_pos2_vao(TRI_VERTS, sizeof(TRI_VERTS), &vao, &vbo);

    /* Pass 1: A cleared once, red-left drawn. */
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    clear_color(0.1f, 0.1f, 0.1f);
    glUseProgram(progRedLeft);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Detour: B forces the encoder to rotate off A. */
    glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    clear_color(0.0f, 0.0f, 0.0f);
    glUseProgram(progBlue);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Pass 2: resume A WITHOUT clearing; add green-right. Red must survive. */
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    glUseProgram(progGreenRight);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(progRedLeft);
    glDeleteProgram(progGreenRight);
    glDeleteProgram(progBlue);
    glDeleteFramebuffers(1, &fboA);
    glDeleteFramebuffers(1, &fboB);
    glDeleteTextures(1, &texA);
    glDeleteTextures(1, &texB);
    return 0;
}

/* ---- DontCare positive case (Stage 4.2) ----
 * Fresh FBO, NO glClear, draw a FULLSCREEN quad covering every pixel. On the
 * attachment's first use this frame with the DontCare flag on, loadAction is
 * DontCare (skip loading prior tile contents); the fullscreen draw then fully
 * defines every pixel, so the result is deterministic and identical whether
 * the load was DontCare or Load. Proves DontCare fires AND yields correct
 * output. Golden: solid magenta. */
static int test_dontcare_fullscreen(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    /* Intentionally NO clear — DontCare/Load must be masked by full coverage.
     * Force full-FBO coverage explicitly: the context is shared across suite
     * tests, so a prior test's viewport/scissor could otherwise leave pixels
     * uncovered, which under DontCare would be undefined (nondeterministic). */
    glViewport(0, 0, REG_W, REG_H);
    glDisable(GL_SCISSOR_TEST);

    GLuint prog = link_program(
        "#version 330 core\n"
        "layout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p,0.0,1.0); }\n",
        "#version 330 core\n"
        "out vec4 f; void main(){ f=vec4(1.0,0.0,1.0,1.0); }\n");
    if (!prog) return 2;
    glUseProgram(prog);

    /* Two triangles covering the whole NDC square. */
    static const float FULL[] = {
        -1.0f,-1.0f,  1.0f,-1.0f,  1.0f, 1.0f,
        -1.0f,-1.0f,  1.0f, 1.0f, -1.0f, 1.0f,
    };
    GLuint vao, vbo;
    make_pos2_vao(FULL, sizeof(FULL), &vao, &vbo);

    glDrawArrays(GL_TRIANGLES, 0, 6);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- Multi-batch same-FBO (Stage 5.1 parallel-group scaffold) ----
 * One FBO, three DIFFERENT programs (hardcoded colors), three non-overlapping
 * triangles. Different programs -> different state keys -> three separate
 * batches in one flush, all targeting the same FBO (same MTLRenderPassDescriptor).
 * This is the "parallel group" candidate Stage 5 would encode concurrently.
 * Gate: all three triangles must render (proves batch grouping + same-FBO
 * continuity hold across the deferred replay loop). */
static int test_multibatch_same_fbo(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fbo, tex;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.1f, 0.1f, 0.1f);
    glViewport(0, 0, REG_W, REG_H);
    glDisable(GL_SCISSOR_TEST);

    /* Three programs differ only by baked color + X offset -> distinct keys. */
    GLuint pR = link_program(
        "#version 330 core\nlayout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p*0.3 + vec2(-0.55,0.0),0.0,1.0); }\n",
        "#version 330 core\nout vec4 f; void main(){ f=vec4(1.0,0.0,0.0,1.0); }\n");
    GLuint pG = link_program(
        "#version 330 core\nlayout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p*0.3 + vec2(0.0,0.0),0.0,1.0); }\n",
        "#version 330 core\nout vec4 f; void main(){ f=vec4(0.0,1.0,0.0,1.0); }\n");
    GLuint pB = link_program(
        "#version 330 core\nlayout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p*0.3 + vec2(0.55,0.0),0.0,1.0); }\n",
        "#version 330 core\nout vec4 f; void main(){ f=vec4(0.0,0.0,1.0,1.0); }\n");
    if (!pR || !pG || !pB) return 2;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glUseProgram(pR); glDrawArrays(GL_TRIANGLES, 0, 3);
    glUseProgram(pG); glDrawArrays(GL_TRIANGLES, 0, 3);
    glUseProgram(pB); glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(pR); glDeleteProgram(pG); glDeleteProgram(pB);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

/* ---- Read-after-write hazard (Stage 5 safety net) ----
 * Render a red quad to FBO A's color texture T, then bind FBO B and draw a
 * fullscreen quad sampling T. The second draw READS a texture the first draw
 * WROTE within the same flush — the Hazard Tracker must flush A's batch before
 * B's so B samples the freshly-rendered red, not stale contents. This is the
 * exact render-to-texture-then-sample pattern (shadow map / post-process) that
 * parallel command recording must preserve. Gate: FBO B must be solid red. */
static int test_render_to_texture_sample(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    GLuint fboA, texT;
    fboA = make_fbo(REG_W, REG_H, &texT);
    if (!fboA) return 1;

    /* FBO B: default-configured separate color target. */
    GLuint fboB, texB;
    fboB = make_fbo(REG_W, REG_H, &texB);
    if (!fboB) return 1;

    /* Pass 1: draw a fullscreen RED quad to FBO A (writes texture T). */
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    clear_color(0.0f, 0.0f, 0.0f);
    glViewport(0, 0, REG_W, REG_H);
    glDisable(GL_SCISSOR_TEST);
    GLuint progFill = link_program(
        "#version 330 core\nlayout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p,0.0,1.0); }\n",
        "#version 330 core\nout vec4 f; void main(){ f=vec4(1.0,0.0,0.0,1.0); }\n");
    if (!progFill) return 2;
    static const float FULL[] = {
        -1.0f,-1.0f, 1.0f,-1.0f, 1.0f,1.0f,
        -1.0f,-1.0f, 1.0f,1.0f, -1.0f,1.0f,
    };
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(FULL, sizeof(FULL));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(progFill);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    /* Pass 2: FBO B samples T (just written) and draws it fullscreen.
     * The Hazard Tracker must flush A before B so the sample sees red. */
    glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    clear_color(0.0f, 0.0f, 0.0f);
    glViewport(0, 0, REG_W, REG_H);
    GLuint progSample = link_program(
        "#version 330 core\nlayout(location=0) in vec2 p;\n"
        "out vec2 v_uv;\n"
        "void main(){ v_uv=p*0.5+0.5; gl_Position=vec4(p,0.0,1.0); }\n",
        "#version 330 core\nin vec2 v_uv;\nuniform sampler2D u_tex;\n"
        "out vec4 f; void main(){ f=texture(u_tex, v_uv); }\n");
    if (!progSample) return 3;
    glUseProgram(progSample);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texT);
    glUniform1i(glGetUniformLocation(progSample, "u_tex"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(progFill); glDeleteProgram(progSample);
    glDeleteFramebuffers(1, &fboA); glDeleteFramebuffers(1, &fboB);
    glDeleteTextures(1, &texT); glDeleteTextures(1, &texB);
    return 0;
}

/* Explicit-only memory soak: 257 keys force continuous eviction from the
 * renderer's fixed 256-entry MTLSamplerState cache. */
static int test_sampler_cache_rss_soak(unsigned char *pixels,
                                       const char *out_path)
{
    (void)out_path;
    memset(pixels, 0, REG_W * REG_H * 4);
    if (!mglSamplerSnapshotEnabled()) {
        fprintf(stderr, "sampler_cache_rss_soak: snapshot path disabled; skipping\n");
        return TEST_RESULT_SKIP;
    }

    int rc = 0;
    GLuint fbo = 0, color_tex = 0, program = 0;
    GLuint vao = 0, vbo = 0, sampled_tex = 0;
    ProcessMemorySample baseline = {0}, midpoint = {0}, final = {0};
    int midpoint_set = 0;
    const uint64_t limit = soak_growth_limit_bytes();
    uint64_t hard_limit = limit <= UINT64_MAX / 4u ? limit * 4u : UINT64_MAX;
    const uint64_t minimum_hard_limit = 256u * 1024u * 1024u;
    if (hard_limit < minimum_hard_limit) hard_limit = minimum_hard_limit;

    fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 1, 1);

    program = make_sampler_test_program(1);
    if (!program) {
        rc = 2;
        goto cleanup;
    }
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_tex"), 0);

    static const float fullscreen[] = {
        -1.0f, -1.0f,  3.0f, -1.0f,  -1.0f, 3.0f,
    };
    make_pos2_vao(fullscreen, sizeof(fullscreen), &vao, &vbo);
    static const unsigned char red[4] = { 255, 0, 0, 255 };
    sampled_tex = make_rgba8_texture(red);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    for (uint32_t i = 0; i < 2u * 257u; i++) {
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD,
                        (GLfloat)(i % 257u) / 256.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    if (glGetError() != GL_NO_ERROR || sample_process_memory(&baseline) != 0) {
        fprintf(stderr, "sampler_cache_rss_soak: warmup or memory sample failed\n");
        rc = 3;
        goto cleanup;
    }

    for (uint32_t i = 0; i < SOAK_ITERATIONS; i++) {
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD,
                        (GLfloat)(i % 257u) / 256.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        if (soak_should_checkpoint(i + 1u)) {
            ProcessMemorySample current = {0};
            glFinish();
            if (glGetError() != GL_NO_ERROR ||
                sample_process_memory(&current) != 0) {
                fprintf(stderr,
                        "sampler_cache_rss_soak: checkpoint %u failed\n",
                        i + 1u);
                rc = 4;
                goto cleanup;
            }
            if (!midpoint_set && i + 1u >= SOAK_ITERATIONS / 2u) {
                midpoint = current;
                midpoint_set = 1;
            }
            if (soak_memory_hard_limit_exceeded(
                    "sampler_cache_rss_soak", &baseline, &current,
                    hard_limit)) {
                rc = 5;
                goto cleanup;
            }
            final = current;
        }
    }

    if (!midpoint_set) midpoint = baseline;
    if (verify_soak_memory_growth("sampler_cache_rss_soak",
                                  &baseline, &midpoint, &final, limit)) {
        rc = 6;
    }

cleanup:
    glDisable(GL_SCISSOR_TEST);
    glFinish();
    if (sampled_tex) glDeleteTextures(1, &sampled_tex);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color_tex) glDeleteTextures(1, &color_tex);
    return rc;
}

/* Explicit-only memory soak for Minecraft's persistent mapped arena update
 * pattern. The no-copy Metal backing must remain stable for every flush. */
static int test_persistent_map_rss_soak(unsigned char *pixels,
                                        const char *out_path)
{
    (void)out_path;
    memset(pixels, 0, REG_W * REG_H * 4);

    enum { BUFFER_BYTES = 1024 * 1024, UPDATE_BYTES = 64, UPDATE_START = 4096 };
    int rc = 0;
    GLuint fbo = 0, color_tex = 0, program = 0, vao = 0, buffer_name = 0;
    unsigned char *mapped = NULL;
    Buffer *internal = NULL;
    void *stable_mtl_data = NULL;
    ProcessMemorySample baseline = {0}, midpoint = {0}, final = {0};
    int midpoint_set = 0;
    const uint64_t limit = soak_growth_limit_bytes();
    uint64_t hard_limit = limit <= UINT64_MAX / 4u ? limit * 4u : UINT64_MAX;
    const uint64_t minimum_hard_limit = 256u * 1024u * 1024u;
    if (hard_limit < minimum_hard_limit) hard_limit = minimum_hard_limit;

    fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);

    program = link_program(
        "#version 330 core\nlayout(location=0) in vec2 p;\n"
        "void main(){ gl_Position=vec4(p,0.0,1.0); }\n",
        "#version 330 core\nout vec4 f;\n"
        "void main(){ f=vec4(1.0); }\n");
    if (!program) {
        rc = 2;
        goto cleanup;
    }
    glUseProgram(program);

    glGenBuffers(1, &buffer_name);
    glBindBuffer(GL_ARRAY_BUFFER, buffer_name);
    glBufferStorage(GL_ARRAY_BUFFER, BUFFER_BYTES, NULL,
                    GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
    mapped = (unsigned char *)glMapBufferRange(
        GL_ARRAY_BUFFER, 0, BUFFER_BYTES,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT |
            GL_MAP_FLUSH_EXPLICIT_BIT);
    if (!mapped || glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "persistent_map_rss_soak: persistent map failed\n");
        rc = 3;
        goto cleanup;
    }

    memset(mapped, 0, BUFFER_BYTES);
    static const float triangle[] = {
        -0.5f, -0.5f,  0.5f, -0.5f,  0.0f, 0.5f,
    };
    memcpy(mapped, triangle, sizeof(triangle));
    glFlushMappedBufferRange(GL_ARRAY_BUFFER, 0, BUFFER_BYTES);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(0, 0);
    glBindVertexBuffer(0, buffer_name, 0, 2 * sizeof(float));
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    GLMContext ctx = MGLgetCurrentContext();
    internal = ctx ? ctx->active_state->buffers[_ARRAY_BUFFER] : NULL;
    if (!internal || !internal->data.mtl_data ||
        !internal->data.mtl_owns_buffer_data ||
        internal->mapped_ptr != mapped) {
        fprintf(stderr,
                "persistent_map_rss_soak: no-copy backing not established "
                "(buffer=%p mtl=%p owns=%u mapped=%p expected=%p)\n",
                (void *)internal,
                internal ? internal->data.mtl_data : NULL,
                internal ? (unsigned)internal->data.mtl_owns_buffer_data : 0u,
                internal ? internal->mapped_ptr : NULL,
                mapped);
        rc = 4;
        goto cleanup;
    }
    stable_mtl_data = internal->data.mtl_data;
    if (glGetError() != GL_NO_ERROR || sample_process_memory(&baseline) != 0) {
        fprintf(stderr, "persistent_map_rss_soak: warmup or memory sample failed\n");
        rc = 5;
        goto cleanup;
    }

    const uint32_t update_slots =
        (BUFFER_BYTES - UPDATE_START) / UPDATE_BYTES;
    for (uint32_t i = 0; i < SOAK_ITERATIONS; i++) {
        size_t offset = UPDATE_START +
            (size_t)(i % update_slots) * UPDATE_BYTES;
        memcpy(mapped + offset, triangle, sizeof(triangle));
        memset(mapped + offset + sizeof(triangle), (int)(i & 0xffu),
               UPDATE_BYTES - sizeof(triangle));
        glFlushMappedBufferRange(GL_ARRAY_BUFFER,
                                 (GLintptr)offset, UPDATE_BYTES);
        glBindVertexBuffer(0, buffer_name, (GLintptr)offset,
                           2 * sizeof(float));
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (internal->data.mtl_data != stable_mtl_data ||
            !internal->data.mtl_owns_buffer_data) {
            fprintf(stderr,
                    "persistent_map_rss_soak: Metal backing changed at %u "
                    "(%p -> %p owns=%u)\n",
                    i + 1u, stable_mtl_data, internal->data.mtl_data,
                    (unsigned)internal->data.mtl_owns_buffer_data);
            rc = 6;
            goto cleanup;
        }

        if (soak_should_checkpoint(i + 1u)) {
            ProcessMemorySample current = {0};
            glFinish();
            if (glGetError() != GL_NO_ERROR ||
                sample_process_memory(&current) != 0) {
                fprintf(stderr,
                        "persistent_map_rss_soak: checkpoint %u failed\n",
                        i + 1u);
                rc = 7;
                goto cleanup;
            }
            if (!midpoint_set && i + 1u >= SOAK_ITERATIONS / 2u) {
                midpoint = current;
                midpoint_set = 1;
            }
            if (soak_memory_hard_limit_exceeded(
                    "persistent_map_rss_soak", &baseline, &current,
                    hard_limit)) {
                rc = 8;
                goto cleanup;
            }
            final = current;
        }
    }

    if (!midpoint_set) midpoint = baseline;
    if (verify_soak_memory_growth("persistent_map_rss_soak",
                                  &baseline, &midpoint, &final, limit)) {
        rc = 9;
    }

cleanup:
    glFinish();
    if (vao) {
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &vao);
    }
    if (buffer_name) {
        glBindBuffer(GL_ARRAY_BUFFER, buffer_name);
        if (mapped) {
            GLboolean unmapped = glUnmapBuffer(GL_ARRAY_BUFFER);
            GLenum cleanup_error = glGetError();
            if (rc == 0 && (!unmapped || cleanup_error != GL_NO_ERROR)) {
                fprintf(stderr,
                        "persistent_map_rss_soak: cleanup unmap failed "
                        "(result=%u error=0x%x)\n",
                        (unsigned)unmapped, cleanup_error);
                rc = 10;
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &buffer_name);
    }
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color_tex) glDeleteTextures(1, &color_tex);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Test registry                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;
    test_fn fn;
    int self_check;
    int explicit_only;
} TestCase;

#define GOLDEN_TEST(name, fn)              { name, fn, 0, 0 }
#define SELF_CHECK_TEST(name, fn)          { name, fn, 1, 0 }
#define EXPLICIT_SELF_CHECK_TEST(name, fn) { name, fn, 1, 1 }

static const TestCase TESTS[] = {
    GOLDEN_TEST("draw_arrays",            test_draw_arrays),
    GOLDEN_TEST("draw_elements",          test_draw_elements),
    GOLDEN_TEST("draw_arrays_instanced",  test_draw_arrays_instanced),
    GOLDEN_TEST("multi_draw_elements",    test_multi_draw_elements),
    GOLDEN_TEST("draw_arrays_indirect",   test_draw_arrays_indirect),
    GOLDEN_TEST("fbo_switch",             test_fbo_switch),
    GOLDEN_TEST("transform_feedback",     test_transform_feedback),
    GOLDEN_TEST("conditional_render",     test_conditional_render),
    GOLDEN_TEST("ubo_range_switch",       test_ubo_range_switch),
    GOLDEN_TEST("vao_binding_switch",     test_vao_binding_switch),
    SELF_CHECK_TEST("agx_3d_texture_workarounds",
                    test_agx_3d_texture_workarounds),
    GOLDEN_TEST("texture_binding_switch", test_texture_binding_switch),
    GOLDEN_TEST("texture_parameter_switch", test_texture_parameter_switch),
    GOLDEN_TEST("sampler_parameter_switch", test_sampler_parameter_switch),
    SELF_CHECK_TEST("sampler_same_value_no_flush",
                    test_sampler_same_value_no_flush),
    SELF_CHECK_TEST("sampler_invalid_no_flush",
                    test_sampler_invalid_no_flush),
    SELF_CHECK_TEST("sampler_snapshot_overflow",
                    test_sampler_snapshot_overflow),
    EXPLICIT_SELF_CHECK_TEST("sampler_cache_rss_soak",
                             test_sampler_cache_rss_soak),
    EXPLICIT_SELF_CHECK_TEST("persistent_map_rss_soak",
                             test_persistent_map_rss_soak),
    GOLDEN_TEST("program_switch",         test_program_switch),
    GOLDEN_TEST("blend",                  test_blend),
    GOLDEN_TEST("depth_test",             test_depth_probe),
    GOLDEN_TEST("stencil",                test_stencil_probe),
    GOLDEN_TEST("uniform_alias",          test_uniform_alias),
    GOLDEN_TEST("shared_uniform",         test_shared_uniform),
    GOLDEN_TEST("multipass_resume",       test_multipass_resume),
    GOLDEN_TEST("dontcare_fullscreen",    test_dontcare_fullscreen),
    GOLDEN_TEST("multibatch_same_fbo",    test_multibatch_same_fbo),
    GOLDEN_TEST("rtt_sample",             test_render_to_texture_sample),
    /* depth_test/stencil use probe-style fns (test_depth_probe /
     * test_stencil_probe): hardcoded per-program values.
     * uniform_alias gates the cross-stage uniform-location fix (program.c
     * mglAssignPlainUniformLocations): a vertex and a fragment uniform must
     * not share a location / plain_uniform_buffers slot.
     *
     * NOT registered (kept as __attribute__((unused)) diagnostics):
     *  - test_depth_test / test_stencil: original versions used the
     *    same-program-multi-draw-with-changing-uniforms pattern; superseded by
     *    the probe versions above. Kept for reference. */
};
#undef GOLDEN_TEST
#undef SELF_CHECK_TEST
#undef EXPLICIT_SELF_CHECK_TEST

static const int NUM_TESTS = (int)(sizeof(TESTS) / sizeof(TESTS[0]));
_Static_assert(sizeof(TESTS) / sizeof(TESTS[0]) == MAX_TESTS,
               "MAX_TESTS must match the test registry");

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    const char *golden_dir = "MGL_Golden_Images";
    const char *out_dir = "/tmp/MGL_Regression";
    const char *only = NULL;
    int update = 0;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--update") == 0 || strcmp(argv[i], "-u") == 0) {
            update = 1;
        } else if (strcmp(argv[i], "--golden-dir") == 0 && i + 1 < argc) {
            golden_dir = argv[++i];
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "all") == 0) {
            /* run all */
        } else {
            only = argv[i];
        }
    }

    if (only) {
        int found = 0;
        for (int i = 0; i < NUM_TESTS; i++) {
            if (strcmp(only, TESTS[i].name) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Unknown regression test: %s\n", only);
            return 2;
        }
    }

    /* Ensure output dir exists */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", out_dir);
    system(cmd);

    /* Create headless MGL context */
    GLMContext glm_ctx = createGLMContext(
        GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
        GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
    if (!glm_ctx) {
        fprintf(stderr, "FATAL: createGLMContext failed\n");
        return 1;
    }
    void *renderer = CppCreateMGLRendererHeadless(glm_ctx);
    if (!renderer) {
        fprintf(stderr, "FATAL: CppCreateMGLRendererHeadless failed\n");
        return 1;
    }
    MGLsetCurrentContext(glm_ctx);

    fprintf(stderr, "MGL regression suite — %d tests\n", NUM_TESTS);
    fprintf(stderr, "  golden: %s\n", golden_dir);
    fprintf(stderr, "  out:    %s\n", out_dir);
    fprintf(stderr, "  mode:   %s\n\n", update ? "UPDATE GOLDEN" : "COMPARE");

    unsigned char *pixels = (unsigned char *)malloc(REG_W * REG_H * 4);
    int n_pass = 0, n_fail = 0, n_skip = 0;

    for (int i = 0; i < NUM_TESTS; i++) {
        const TestCase *t = &TESTS[i];
        if (only && strcmp(only, t->name) != 0) {
            n_skip++;
            continue;
        }
        if (t->explicit_only && !only) {
            n_skip++;
            continue;
        }

        char out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s/Reg_%s.tga", out_dir, t->name);

        fprintf(stderr, "[%02d/%02d] %-24s ... ", i + 1, NUM_TESTS, t->name);
        fflush(stderr);

        memset(pixels, 0, REG_W * REG_H * 4);

        /* Reset all GL state to a clean baseline so this test starts
         * independent of the previous test's residual binds / caps.
         * (Stage 5.3 prerequisite — context isolation.) */
        resetGLState();

        /* Drain any lingering GL errors from resetGLState or prior cleanup. */
        while (glGetError() != GL_NO_ERROR) { /* drain */ }

        int rc = t->fn(pixels, out_path);

        /* Safety-net glFinish: ensures all GPU work from this test is
         * complete before the next test starts.  Most tests already call
         * glFinish before glReadPixels, but a test that returns early
         * (rc != 0) may leave pending GPU work. */
        glFinish();

        /* drain any lingering GL errors for cleanliness (always runs,
         * even on early-return failure) */
        GLenum e;
        while ((e = glGetError()) != GL_NO_ERROR) {
            /* only warn; some drivers leave harmless errors */
        }

        if (rc == TEST_RESULT_SKIP) {
            fprintf(stderr, "SKIP\n");
            n_skip++;
            continue;
        }

        if (rc != 0) {
            fprintf(stderr, "ERROR (rc=%d)\n", rc);
            n_fail++;
            continue;
        }

        if (t->self_check) {
            fprintf(stderr, "PASS (self-check)\n");
            n_pass++;
            continue;
        }

        if (write_tga(out_path, REG_W, REG_H, pixels) != 0) {
            fprintf(stderr, "WRITE FAIL\n");
            n_fail++;
            continue;
        }

        if (update) {
            char gpath[1100];
            snprintf(gpath, sizeof(gpath), "%s/Reg_%s.tga", golden_dir, t->name);
            char cp[2200];
            snprintf(cp, sizeof(cp), "cp '%s' '%s'", out_path, gpath);
            if (system(cp) == 0) {
                fprintf(stderr, "GOLDEN UPDATED\n");
                n_pass++;
            } else {
                fprintf(stderr, "CP FAIL\n");
                n_fail++;
            }
        } else {
            char gpath[1100];
            snprintf(gpath, sizeof(gpath), "%s/Reg_%s.tga", golden_dir, t->name);
            if (files_equal(out_path, gpath)) {
                fprintf(stderr, "PASS\n");
                n_pass++;
            } else {
                fprintf(stderr, "FAIL (mismatch vs %s)\n", gpath);
                n_fail++;
            }
        }
    }

    free(pixels);

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  PASS: %d   FAIL: %d   SKIP: %d   /   %d\n",
            n_pass, n_fail, n_skip, NUM_TESTS);
    fprintf(stderr, "========================================\n");

    return (n_fail == 0) ? 0 : 1;
}
