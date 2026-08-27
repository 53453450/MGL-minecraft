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

/* Legacy GL 1.1 clip-plane surface: glClipPlane/glGetClipPlane are not
 * declared by glcorearb.h, and GL_CLIP_PLANE0..5 share the GL_CLIP_DISTANCE
 * values (0x3000 + i). */
#ifndef GL_CLIP_PLANE0
#define GL_CLIP_PLANE0 GL_CLIP_DISTANCE0
#define GL_CLIP_PLANE1 GL_CLIP_DISTANCE1
#define GL_CLIP_PLANE2 GL_CLIP_DISTANCE2
#define GL_CLIP_PLANE3 GL_CLIP_DISTANCE3
#define GL_CLIP_PLANE4 GL_CLIP_DISTANCE4
#define GL_CLIP_PLANE5 GL_CLIP_DISTANCE5
#define GL_CLIP_PLANE6 GL_CLIP_DISTANCE6
#define GL_CLIP_PLANE7 GL_CLIP_DISTANCE7
#endif
GLAPI void APIENTRY glClipPlane(GLenum plane, const GLdouble *equation);
GLAPI void APIENTRY glGetClipPlane(GLenum plane, GLdouble *equation);

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

#define REG_W 128
#define REG_H 128
#define MAX_TESTS 84
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

static GLuint link_program_with_geometry(const char *vs_src,
                                         const char *gs_src,
                                         const char *fs_src)
{
    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs_src),
        compile_shader(GL_GEOMETRY_SHADER, gs_src),
        compile_shader(GL_FRAGMENT_SHADER, fs_src),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) {
        for (int i = 0; i < 3; i++) {
            if (shaders[i]) glDeleteShader(shaders[i]);
        }
        return 0;
    }
    GLuint program = glCreateProgram();
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "  [geometry program link FAIL] %s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

/* Link a program after installing an XFB varying list.  The API error is
 * returned separately because ARB_transform_feedback3 rejects special names
 * in GL_SEPARATE_ATTRIBS before link time. */
static int xfb_link_status(const char *vs_src,
                           const char *gs_src,
                           const char *fs_src,
                           GLsizei varying_count,
                           const char *const *varyings,
                           GLenum buffer_mode,
                           GLenum *out_api_error,
                           GLint *out_link_status)
{
    if (!vs_src || !fs_src || !out_api_error || !out_link_status)
        return 1;

    GLuint shaders[3] = {0, 0, 0};
    shaders[0] = compile_shader(GL_VERTEX_SHADER, vs_src);
    shaders[1] = gs_src ? compile_shader(GL_GEOMETRY_SHADER, gs_src) : 0;
    shaders[2] = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!shaders[0] || (gs_src && !shaders[1]) || !shaders[2]) {
        for (int i = 0; i < 3; i++) {
            if (shaders[i]) glDeleteShader(shaders[i]);
        }
        return 1;
    }

    GLuint program = glCreateProgram();
    if (!program) {
        for (int i = 0; i < 3; i++) {
            if (shaders[i]) glDeleteShader(shaders[i]);
        }
        return 1;
    }
    for (int i = 0; i < 3; i++) {
        if (shaders[i]) glAttachShader(program, shaders[i]);
    }
    while (glGetError() != GL_NO_ERROR) { }
    glTransformFeedbackVaryings(program, varying_count, varyings,
                                buffer_mode);
    *out_api_error = glGetError();
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, out_link_status);
    glDeleteProgram(program);
    for (int i = 0; i < 3; i++) {
        if (shaders[i]) glDeleteShader(shaders[i]);
    }
    while (glGetError() != GL_NO_ERROR) { }
    return 0;
}

static int geometry_program_link_status_with_ssbo_count(GLuint ssbo_count,
                                                        int use_runtime_length,
                                                        GLint *out_status)
{
    static const char *vs_src =
        "#version 460 core\n"
        "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    static const char *fs_src =
        "#version 460 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(1.0); }\n";
    char gs_src[8192];
    size_t used = 0u;
    int written = snprintf(gs_src, sizeof(gs_src),
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=1) out;\n");
    if (written < 0 || (size_t)written >= sizeof(gs_src)) {
        return 1;
    }
    used = (size_t)written;
    for (GLuint i = 0; i < ssbo_count; i++) {
        if (i == 0u && use_runtime_length) {
            written = snprintf(gs_src + used, sizeof(gs_src) - used,
                "layout(std430, binding=0) buffer B0 { uint prefix; "
                "float values[]; } b0;\n");
        } else {
            written = snprintf(gs_src + used, sizeof(gs_src) - used,
                "layout(std430, binding=%u) buffer B%u { uint value%u; } b%u;\n",
                i, i, i, i);
        }
        if (written < 0 || (size_t)written >= sizeof(gs_src) - used) {
            return 1;
        }
        used += (size_t)written;
    }
    written = snprintf(gs_src + used, sizeof(gs_src) - used,
        use_runtime_length
            ? "void main() { b0.prefix = uint(b0.values.length()); "
              "gl_Position = gl_in[0].gl_Position; EmitVertex(); "
              "EndPrimitive(); }\n"
            : "void main() { gl_Position = gl_in[0].gl_Position; "
              "EmitVertex(); EndPrimitive(); }\n");
    if (written < 0 || (size_t)written >= sizeof(gs_src) - used) {
        return 1;
    }

    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs_src),
        compile_shader(GL_GEOMETRY_SHADER, gs_src),
        compile_shader(GL_FRAGMENT_SHADER, fs_src),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) {
        for (int i = 0; i < 3; i++) {
            if (shaders[i]) glDeleteShader(shaders[i]);
        }
        return 1;
    }

    GLuint program = glCreateProgram();
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    glGetProgramiv(program, GL_LINK_STATUS, out_status);
    glDeleteProgram(program);
    return 0;
}

static int compute_program_link_status_with_ssbo_count(
    GLuint ssbo_count,
    int use_runtime_length,
    GLint *out_status)
{
    if (ssbo_count == 0u || !out_status) return 1;

    char cs_src[16384];
    size_t used = 0u;
    int written = snprintf(cs_src, sizeof(cs_src),
                           "#version 460 core\n"
                           "layout(local_size_x=1) in;\n");
    if (written < 0 || (size_t)written >= sizeof(cs_src)) return 1;
    used = (size_t)written;

    for (GLuint i = 0u; i < ssbo_count; i++) {
        if (i == 0u && use_runtime_length) {
            written = snprintf(cs_src + used, sizeof(cs_src) - used,
                "layout(std430, binding=0) buffer B0 { uint prefix; "
                "float values[]; } b0;\n");
        } else {
            written = snprintf(cs_src + used, sizeof(cs_src) - used,
                "layout(std430, binding=%u) buffer B%u { uint value%u; } b%u;\n",
                i, i, i, i);
        }
        if (written < 0 || (size_t)written >= sizeof(cs_src) - used) return 1;
        used += (size_t)written;
    }

    written = snprintf(cs_src + used, sizeof(cs_src) - used,
        use_runtime_length
            ? "void main() { b0.prefix = uint(b0.values.length()); }\n"
            : "void main() { b0.value0 = b0.value0; }\n");
    if (written < 0 || (size_t)written >= sizeof(cs_src) - used) return 1;

    GLuint shader = compile_shader(GL_COMPUTE_SHADER, cs_src);
    if (!shader) return 1;
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);
    glGetProgramiv(program, GL_LINK_STATUS, out_status);
    glDeleteProgram(program);
    return 0;
}

static GLuint link_compute_program(const char *cs_src)
{
    GLuint cs = compile_shader(GL_COMPUTE_SHADER, cs_src);
    if (!cs) return 0;
    GLuint program = glCreateProgram();
    glAttachShader(program, cs);
    glLinkProgram(program);
    glDeleteShader(cs);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "  [compute program link FAIL] %s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

static GLuint link_program_with_tessellation(const char *vs_src,
                                             const char *tcs_src,
                                             const char *tes_src,
                                             const char *fs_src)
{
    GLuint shaders[4] = {
        compile_shader(GL_VERTEX_SHADER, vs_src),
        compile_shader(GL_TESS_CONTROL_SHADER, tcs_src),
        compile_shader(GL_TESS_EVALUATION_SHADER, tes_src),
        compile_shader(GL_FRAGMENT_SHADER, fs_src),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2] || !shaders[3]) {
        for (int i = 0; i < 4; i++) {
            if (shaders[i]) glDeleteShader(shaders[i]);
        }
        return 0;
    }
    GLuint program = glCreateProgram();
    for (int i = 0; i < 4; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (int i = 0; i < 4; i++) glDeleteShader(shaders[i]);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "  [tessellation program link FAIL] %s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

/* TES-only program (no TCS): native indexed patch draws use the default
 * tessellation factors via glPatchParameterfv. */
static GLuint link_program_tess_eval_only(const char *vs_src,
                                          const char *tes_src,
                                          const char *fs_src)
{
    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs_src),
        compile_shader(GL_TESS_EVALUATION_SHADER, tes_src),
        compile_shader(GL_FRAGMENT_SHADER, fs_src),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) {
        for (int i = 0; i < 3; i++) {
            if (shaders[i]) glDeleteShader(shaders[i]);
        }
        return 0;
    }
    GLuint program = glCreateProgram();
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "  [TES-only program link FAIL] %s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
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

/* R32I integer color framebuffer (CTS limits tests render int outputs). */
static GLuint make_fbo_r32i(int w, int h, GLuint *out_tex)
{
    GLuint fbo, tex;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, w, h, 0, GL_RED_INTEGER,
                 GL_INT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tex, 0);
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "  [R32I FBO incomplete: 0x%x]\n", st);
        return 0;
    }
    if (out_tex) *out_tex = tex;
    return fbo;
}

/* Two-layer 2D array framebuffer for gl_Layer coverage; the draw target is
 * switched between layers with glFramebufferTextureLayer. */
static GLuint make_layer_fbo(int w, int h, GLuint *out_tex)
{
    GLuint fbo, tex, rbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, w, h, 2, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, 0);

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "  [layer FBO incomplete: 0x%x]\n", st);
        return 0;
    }
    if (out_tex) *out_tex = tex;
    return fbo;
}

static void drain_gl_errors(void)
{
    while (glGetError() != GL_NO_ERROR) { }
}

static int expect_single_gl_error(const char *label, GLenum expected)
{
    GLenum actual = glGetError();
    GLenum extra = glGetError();
    if (actual == expected && extra == GL_NO_ERROR) {
        return 0;
    }

    fprintf(stderr,
            "%s: error=0x%x extra=0x%x expected=0x%x\n",
            label, actual, extra, expected);
    drain_gl_errors();
    return 1;
}

static int expect_bound_texture_level_dimensions(const char *label,
                                                 GLenum target,
                                                 GLint level,
                                                 GLint expected_width,
                                                 GLint expected_height,
                                                 GLint expected_depth)
{
    GLint width = -1;
    GLint height = -1;
    GLint depth = -1;
    glGetTexLevelParameteriv(target, level, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(target, level, GL_TEXTURE_HEIGHT, &height);
    glGetTexLevelParameteriv(target, level, GL_TEXTURE_DEPTH, &depth);
    if (glGetError() != GL_NO_ERROR ||
        width != expected_width ||
        height != expected_height ||
        depth != expected_depth) {
        fprintf(stderr,
                "%s: level=%d size=%dx%dx%d expected=%dx%dx%d\n",
                label, level, width, height, depth,
                expected_width, expected_height, expected_depth);
        drain_gl_errors();
        return 1;
    }
    return 0;
}

static int test_texture_mip_dimensions(unsigned char *pixels,
                                       const char *out_path)
{
    (void)pixels;
    (void)out_path;
    GLuint textures[10] = {0u};
    int result = 1;

    glGenTextures(10, textures);

    glBindTexture(GL_TEXTURE_1D_ARRAY, textures[0]);
    glTexStorage2D(GL_TEXTURE_1D_ARRAY, 3, GL_RGBA8, 16, 5);
    if (expect_bound_texture_level_dimensions("texture_mip_dimensions: 1D array",
                                              GL_TEXTURE_1D_ARRAY, 2, 4, 5, 1)) {
        goto cleanup;
    }
    glBindTexture(GL_TEXTURE_3D, textures[1]);
    glTexStorage3D(GL_TEXTURE_3D, 3, GL_RGBA8, 16, 8, 4);
    if (expect_bound_texture_level_dimensions("texture_mip_dimensions: 3D",
                                              GL_TEXTURE_3D, 2, 4, 2, 1)) {
        goto cleanup;
    }
    {
        GLubyte data[4 * 2 * 4] = {0u};
        glTexSubImage3D(GL_TEXTURE_3D, 2, 0, 0, 0, 4, 2, 1,
                        GL_RGBA, GL_UNSIGNED_BYTE, data);
        if (expect_single_gl_error("texture_mip_dimensions: 3D last slice",
                                   GL_NO_ERROR)) {
            goto cleanup;
        }
        glTexSubImage3D(GL_TEXTURE_3D, 2, 0, 0, 1, 4, 2, 1,
                        GL_RGBA, GL_UNSIGNED_BYTE, data);
        if (expect_single_gl_error("texture_mip_dimensions: 3D overflow",
                                   GL_INVALID_VALUE)) {
            goto cleanup;
        }
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, textures[2]);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 3, GL_RGBA8, 16, 8, 5);
    if (expect_bound_texture_level_dimensions("texture_mip_dimensions: 2D array",
                                              GL_TEXTURE_2D_ARRAY, 2, 4, 2, 5)) {
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, textures[3]);
    glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 3, GL_RGBA8, 16, 16, 12);
    if (expect_bound_texture_level_dimensions("texture_mip_dimensions: cube array",
                                              GL_TEXTURE_CUBE_MAP_ARRAY, 2, 4, 4, 12)) {
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, textures[4]);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 4, 4, 2, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    if (expect_single_gl_error("texture_mip_dimensions: generate 2D array",
                               GL_NO_ERROR) ||
        expect_bound_texture_level_dimensions("texture_mip_dimensions: generated 2D array",
                                              GL_TEXTURE_2D_ARRAY, 1, 2, 2, 2)) {
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, textures[5]);
    glTexImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, GL_RGBA8, 4, 4, 6, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP_ARRAY);
    if (expect_single_gl_error("texture_mip_dimensions: generate cube array",
                               GL_NO_ERROR) ||
        expect_bound_texture_level_dimensions("texture_mip_dimensions: generated cube array",
                                              GL_TEXTURE_CUBE_MAP_ARRAY, 1, 2, 2, 6)) {
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_2D, textures[6]);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA8, 16, 16);
    GLint immutable_levels = 0;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_LEVELS,
                        &immutable_levels);
    if (glGetError() != GL_NO_ERROR || immutable_levels != 2) {
        fprintf(stderr,
                "texture_mip_dimensions: immutable levels before generate=%d\n",
                immutable_levels);
        goto cleanup;
    }
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_LEVELS,
                        &immutable_levels);
    if (glGetError() != GL_NO_ERROR || immutable_levels != 2) {
        fprintf(stderr,
                "texture_mip_dimensions: immutable levels after generate=%d\n",
                immutable_levels);
        goto cleanup;
    }

    drain_gl_errors();
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA8, 16, 16);
    if (expect_single_gl_error("texture_mip_dimensions: repeated storage",
                               GL_INVALID_OPERATION)) {
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_RECTANGLE, textures[7]);
    drain_gl_errors();
    glTexStorage2D(GL_TEXTURE_RECTANGLE, 2, GL_RGBA8, 16, 16);
    if (expect_single_gl_error("texture_mip_dimensions: rectangle levels",
                               GL_INVALID_OPERATION)) {
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_1D_ARRAY, textures[8]);
    glTexImage2D(GL_TEXTURE_1D_ARRAY, 0, GL_RGBA8, 4, 5, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_1D_ARRAY);
    if (expect_single_gl_error("texture_mip_dimensions: generate 1D array",
                               GL_NO_ERROR) ||
        expect_bound_texture_level_dimensions("texture_mip_dimensions: generated 1D array",
                                              GL_TEXTURE_1D_ARRAY, 1, 2, 5, 1)) {
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_2D, textures[9]);
    drain_gl_errors();
    glGenerateMipmap(GL_TEXTURE_2D);
    if (expect_single_gl_error("texture_mip_dimensions: undefined base level",
                               GL_INVALID_OPERATION)) {
        goto cleanup;
    }

    result = 0;

cleanup:
    glDeleteTextures(10, textures);
    drain_gl_errors();
    return result;
}

static int expect_texture_storage_unallocated(const char *label, GLenum target)
{
    GLint immutable = -1;
    GLint width = -1;
    GLint internalformat = -1;

    glGetTexParameteriv(target, GL_TEXTURE_IMMUTABLE_FORMAT, &immutable);
    glGetTexLevelParameteriv(target, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(target, 0, GL_TEXTURE_INTERNAL_FORMAT,
                             &internalformat);
    GLenum error = glGetError();
    if (error == GL_NO_ERROR && immutable == GL_FALSE && width == 0 &&
        internalformat == 0) {
        return 0;
    }

    fprintf(stderr,
            "%s: error=0x%x immutable=%d width=%d internalformat=0x%x\n",
            label, error, immutable, width, internalformat);
    drain_gl_errors();
    return 1;
}

static int expect_texture_storage_allocated(const char *label, GLenum target,
                                            GLint expected_width)
{
    GLint immutable = -1;
    GLint width = -1;
    GLint internalformat = -1;

    glGetTexParameteriv(target, GL_TEXTURE_IMMUTABLE_FORMAT, &immutable);
    glGetTexLevelParameteriv(target, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(target, 0, GL_TEXTURE_INTERNAL_FORMAT,
                             &internalformat);
    GLenum error = glGetError();
    if (error == GL_NO_ERROR && immutable == GL_TRUE &&
        width == expected_width && internalformat == GL_RGBA8) {
        return 0;
    }

    fprintf(stderr,
            "%s: error=0x%x immutable=%d width=%d internalformat=0x%x "
            "expectedWidth=%d\n",
            label, error, immutable, width, internalformat, expected_width);
    drain_gl_errors();
    return 1;
}

static int test_texture_storage_internalformat_validation(
    unsigned char *pixels, const char *out_path)
{
    (void)pixels;
    (void)out_path;

    enum StorageEntry {
        STORAGE_BOUND_1D,
        STORAGE_BOUND_2D,
        STORAGE_BOUND_3D,
        STORAGE_DSA_1D,
        STORAGE_DSA_2D,
        STORAGE_DSA_3D,
        STORAGE_DSA_2D_MS,
        STORAGE_DSA_2D_MS_ARRAY,
    };
    static const struct StorageCase {
        const char *label;
        enum StorageEntry entry;
        GLenum target;
        GLenum invalid_format;
    } cases[] = {
        {"bound 1D unsized", STORAGE_BOUND_1D, GL_TEXTURE_1D, GL_RGBA},
        {"bound 2D generic compressed", STORAGE_BOUND_2D, GL_TEXTURE_2D,
         GL_COMPRESSED_RGBA},
        {"DSA 1D unsized", STORAGE_DSA_1D, GL_TEXTURE_1D, GL_RED},
        {"DSA 2D generic compressed", STORAGE_DSA_2D, GL_TEXTURE_2D,
         GL_COMPRESSED_RGB},
        {"DSA 3D stencil1", STORAGE_DSA_3D, GL_TEXTURE_3D,
         GL_STENCIL_INDEX1},
        {"DSA 2D multisample unsized", STORAGE_DSA_2D_MS,
         GL_TEXTURE_2D_MULTISAMPLE, GL_RGBA},
        {"DSA 2D multisample array unsized", STORAGE_DSA_2D_MS_ARRAY,
         GL_TEXTURE_2D_MULTISAMPLE_ARRAY, GL_DEPTH_STENCIL},
    };
    int result = 1;

    for (size_t i = 0u; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const struct StorageCase *test = &cases[i];
        GLuint texture = 0u;
        char error_label[160];
        char state_label[160];
        char success_label[160];
        GLboolean dsa = test->entry >= STORAGE_DSA_1D;

        if (dsa) {
            glCreateTextures(test->target, 1, &texture);
        } else {
            glGenTextures(1, &texture);
            glBindTexture(test->target, texture);
        }
        if (!texture || glGetError() != GL_NO_ERROR) {
            fprintf(stderr,
                    "texture_storage_internalformat_validation: setup failed "
                    "for %s\n",
                    test->label);
            goto cleanup;
        }

        drain_gl_errors();
        switch (test->entry) {
            case STORAGE_BOUND_1D:
                glTexStorage1D(test->target, 1, test->invalid_format, 4);
                break;
            case STORAGE_BOUND_2D:
                glTexStorage2D(test->target, 1, test->invalid_format, 4, 4);
                break;
            case STORAGE_BOUND_3D:
                glTexStorage3D(test->target, 1, test->invalid_format, 4, 4, 4);
                break;
            case STORAGE_DSA_1D:
                glTextureStorage1D(texture, 1, test->invalid_format, 4);
                break;
            case STORAGE_DSA_2D:
                glTextureStorage2D(texture, 1, test->invalid_format, 4, 4);
                break;
            case STORAGE_DSA_3D:
                glTextureStorage3D(texture, 1, test->invalid_format, 4, 4, 4);
                break;
            case STORAGE_DSA_2D_MS:
                glTextureStorage2DMultisample(texture, 1,
                                              test->invalid_format, 4, 4,
                                              GL_TRUE);
                break;
            case STORAGE_DSA_2D_MS_ARRAY:
                glTextureStorage3DMultisample(texture, 1,
                                              test->invalid_format, 4, 4, 2,
                                              GL_TRUE);
                break;
        }

        snprintf(error_label, sizeof(error_label),
                 "texture_storage_internalformat_validation: %s error",
                 test->label);
        if (expect_single_gl_error(error_label, GL_INVALID_ENUM)) {
            glDeleteTextures(1, &texture);
            goto cleanup;
        }

        glBindTexture(test->target, texture);
        snprintf(state_label, sizeof(state_label),
                 "texture_storage_internalformat_validation: %s state",
                 test->label);
        if (expect_texture_storage_unallocated(state_label, test->target)) {
            glDeleteTextures(1, &texture);
            goto cleanup;
        }

        switch (test->entry) {
            case STORAGE_BOUND_1D:
            case STORAGE_DSA_1D:
                if (dsa) glTextureStorage1D(texture, 1, GL_RGBA8, 4);
                else glTexStorage1D(test->target, 1, GL_RGBA8, 4);
                break;
            case STORAGE_BOUND_2D:
            case STORAGE_DSA_2D:
                if (dsa) glTextureStorage2D(texture, 1, GL_RGBA8, 4, 4);
                else glTexStorage2D(test->target, 1, GL_RGBA8, 4, 4);
                break;
            case STORAGE_BOUND_3D:
            case STORAGE_DSA_3D:
                if (dsa) glTextureStorage3D(texture, 1, GL_RGBA8, 4, 4, 4);
                else glTexStorage3D(test->target, 1, GL_RGBA8, 4, 4, 4);
                break;
            case STORAGE_DSA_2D_MS:
                glTextureStorage2DMultisample(texture, 1, GL_RGBA8, 4, 4,
                                              GL_TRUE);
                break;
            case STORAGE_DSA_2D_MS_ARRAY:
                glTextureStorage3DMultisample(texture, 1, GL_RGBA8, 4, 4, 2,
                                              GL_TRUE);
                break;
        }
        if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr,
                    "texture_storage_internalformat_validation: positive "
                    "allocation failed for %s\n",
                    test->label);
            glDeleteTextures(1, &texture);
            goto cleanup;
        }

        snprintf(success_label, sizeof(success_label),
                 "texture_storage_internalformat_validation: %s success",
                 test->label);
        if (expect_texture_storage_allocated(success_label, test->target, 4)) {
            glDeleteTextures(1, &texture);
            goto cleanup;
        }
        glDeleteTextures(1, &texture);
    }

    drain_gl_errors();
    glTexStorage2DMultisample(GL_PROXY_TEXTURE_2D_MULTISAMPLE, 1, GL_RGBA,
                             4, 4, GL_TRUE);
    if (expect_single_gl_error(
            "texture_storage_internalformat_validation: proxy 2D MS",
            GL_INVALID_ENUM)) {
        goto cleanup;
    }
    glTexStorage3DMultisample(GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY, 1,
                             GL_COMPRESSED_RGBA, 4, 4, 2, GL_TRUE);
    if (expect_single_gl_error(
            "texture_storage_internalformat_validation: proxy 2D MS array",
            GL_INVALID_ENUM)) {
        goto cleanup;
    }

    {
        GLuint compressed = 0u;
        glGenTextures(1, &compressed);
        glBindTexture(GL_TEXTURE_2D, compressed);
        glTexStorage2D(GL_TEXTURE_2D, 1,
                       GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, 4, 4);
        if (expect_single_gl_error(
                "texture_storage_internalformat_validation: concrete compressed",
                GL_NO_ERROR)) {
            glDeleteTextures(1, &compressed);
            goto cleanup;
        }
        glDeleteTextures(1, &compressed);
    }

    /* GL_DEPTH_COMPONENT32 is a valid sized depth format and must be accepted
     * by glTexStorage*; only a genuinely invalid internal format may raise
     * GL_INVALID_ENUM.  (Historically the internal-format validation switch
     * omitted GL_DEPTH_COMPONENT32 and erroneously rejected it; this locks the
     * acceptance.)  A depth format is exercised on GL_TEXTURE_2D (Metal
     * requires depth/stencil pixel formats to use a 2D/cube target, not 3D). */
    {
        GLuint depth32 = 0u;
        glGenTextures(1, &depth32);
        glBindTexture(GL_TEXTURE_2D, depth32);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32, 4, 4);
        if (expect_single_gl_error(
                "texture_storage_internalformat_validation: depth32 storage",
                GL_NO_ERROR)) {
            glDeleteTextures(1, &depth32);
            goto cleanup;
        }
        GLint immutable = -1;
        GLint depth_width = -1;
        GLint depth_fmt = -1;
        glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_FORMAT,
                            &immutable);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH,
                                 &depth_width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT,
                                 &depth_fmt);
        if (glGetError() != GL_NO_ERROR || immutable != GL_TRUE ||
            depth_width != 4 || depth_fmt != (GLint)GL_DEPTH_COMPONENT32) {
            fprintf(stderr,
                    "texture_storage_internalformat_validation: depth32 state "
                    "immutable=%d width=%d fmt=0x%x\n",
                    immutable, depth_width, depth_fmt);
            drain_gl_errors();
            glDeleteTextures(1, &depth32);
            goto cleanup;
        }
        glDeleteTextures(1, &depth32);
    }

    result = 0;

cleanup:
    drain_gl_errors();
    return result;
}

struct FramebufferAttachmentState {
    GLint object;
    GLint level;
    GLint layer;
    GLint layered;
};

static int capture_framebuffer_attachment_state(
    const char *label,
    struct FramebufferAttachmentState *state)
{
    state->object = -1;
    state->level = -1;
    state->layer = -1;
    state->layered = -1;
    glGetFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &state->object);
    glGetFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &state->level);
    glGetFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER, &state->layer);
    glGetFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_LAYERED, &state->layered);
    GLenum error = glGetError();
    GLenum extra = glGetError();
    if (error != GL_NO_ERROR || extra != GL_NO_ERROR) {
        fprintf(stderr,
                "%s: attachment query error=0x%x extra=0x%x\n",
                label, error, extra);
        drain_gl_errors();
        return 1;
    }
    return 0;
}

static int expect_framebuffer_attachment_state(
    const char *label,
    const struct FramebufferAttachmentState *expected)
{
    struct FramebufferAttachmentState actual;
    if (capture_framebuffer_attachment_state(label, &actual)) {
        return 1;
    }
    if (actual.object != expected->object ||
        actual.level != expected->level ||
        actual.layer != expected->layer ||
        actual.layered != expected->layered) {
        fprintf(stderr,
                "%s: attachment=(object=%d level=%d layer=%d layered=%d) "
                "expected=(object=%d level=%d layer=%d layered=%d)\n",
                label,
                actual.object, actual.level, actual.layer, actual.layered,
                expected->object, expected->level, expected->layer,
                expected->layered);
        return 1;
    }
    return 0;
}

static GLint framebuffer_test_max_mip_level(GLint max_size)
{
    GLint level = 0;
    while (max_size > 1) {
        max_size >>= 1;
        level++;
    }
    return level;
}

static int expect_framebuffer_layer_status(const char *label,
                                           GLuint fbo,
                                           GLuint texture,
                                           GLint level,
                                           GLint layer,
                                           GLboolean named,
                                           GLenum expected_status)
{
    drain_gl_errors();
    if (named) {
        glNamedFramebufferTextureLayer(fbo, GL_COLOR_ATTACHMENT0,
                                       texture, level, layer);
    } else {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  texture, level, layer);
    }
    if (expect_single_gl_error(label, GL_NO_ERROR)) {
        return 1;
    }

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != expected_status) {
        fprintf(stderr,
                "%s: status=0x%x expected=0x%x\n",
                label, status, expected_status);
        return 1;
    }
    return 0;
}

static int test_framebuffer_texture_layer_validation(unsigned char *pixels,
                                                     const char *out_path)
{
    (void)pixels;
    (void)out_path;
    GLuint fbo = 0u;
    GLuint old_array = 0u;
    GLuint texture_2d = 0u;
    GLuint cube = 0u;
    GLuint unrealized = 0u;
    GLuint layered[4] = {0u};
    GLuint no_storage[5] = {0u};
    struct FramebufferAttachmentState preserved_state;
    int result = 1;

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    glGenTextures(1, &old_array);
    glBindTexture(GL_TEXTURE_2D_ARRAY, old_array);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 2, GL_RGBA8, 4, 4, 2);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, old_array, 1);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "framebuffer_texture_layer_validation: initial FBO incomplete\n");
        goto cleanup;
    }
    if (capture_framebuffer_attachment_state(
            "framebuffer_texture_layer_validation: initial attachment",
            &preserved_state)) {
        goto cleanup;
    }

    glGenTextures(1, &texture_2d);
    glBindTexture(GL_TEXTURE_2D, texture_2d);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    drain_gl_errors();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              texture_2d, 2, 1);
    if (expect_single_gl_error("framebuffer_texture_layer_validation: bound 2D",
                               GL_INVALID_OPERATION) ||
        expect_framebuffer_attachment_state(
            "framebuffer_texture_layer_validation: bound unchanged",
            &preserved_state)) {
        goto cleanup;
    }

    drain_gl_errors();
    glNamedFramebufferTextureLayer(fbo, GL_COLOR_ATTACHMENT0,
                                   texture_2d, 2, 1);
    if (expect_single_gl_error("framebuffer_texture_layer_validation: named 2D",
                               GL_INVALID_OPERATION) ||
        expect_framebuffer_attachment_state(
            "framebuffer_texture_layer_validation: named unchanged",
            &preserved_state)) {
        goto cleanup;
    }

    drain_gl_errors();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              old_array, 0, -1);
    if (expect_single_gl_error("framebuffer_texture_layer_validation: negative layer",
                               GL_INVALID_VALUE) ||
        expect_framebuffer_attachment_state(
            "framebuffer_texture_layer_validation: negative unchanged",
            &preserved_state)) {
        goto cleanup;
    }

    glGenTextures(1, &cube);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
    for (GLuint face = 0u; face < 6u; ++face) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA8,
                     4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
    drain_gl_errors();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              cube, 0, 6);
    if (expect_single_gl_error("framebuffer_texture_layer_validation: cube layer 6",
                               GL_INVALID_VALUE) ||
        expect_framebuffer_attachment_state(
            "framebuffer_texture_layer_validation: cube unchanged",
            &preserved_state)) {
        goto cleanup;
    }

    glGenTextures(1, &unrealized);
    drain_gl_errors();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              unrealized, 0, 0);
    if (expect_single_gl_error("framebuffer_texture_layer_validation: unrealized texture",
                               GL_INVALID_OPERATION) ||
        expect_framebuffer_attachment_state(
            "framebuffer_texture_layer_validation: unrealized unchanged",
            &preserved_state)) {
        goto cleanup;
    }

    drain_gl_errors();
    glFramebufferTextureLayer(GL_TEXTURE_2D, GL_COLOR_ATTACHMENT0,
                              old_array, 0, 0);
    if (expect_single_gl_error("framebuffer_texture_layer_validation: invalid framebuffer target",
                               GL_INVALID_ENUM) ||
        expect_framebuffer_attachment_state(
            "framebuffer_texture_layer_validation: target unchanged",
            &preserved_state)) {
        goto cleanup;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0u);
    drain_gl_errors();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              old_array, 0, 0);
    if (expect_single_gl_error("framebuffer_texture_layer_validation: default framebuffer",
                               GL_INVALID_OPERATION)) {
        goto cleanup;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    if (expect_framebuffer_attachment_state(
            "framebuffer_texture_layer_validation: default unchanged",
            &preserved_state)) {
        goto cleanup;
    }

    GLint max_texture_size = 0;
    GLint max_3d_texture_size = 0;
    GLint max_cube_map_texture_size = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_3d_texture_size);
    glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &max_cube_map_texture_size);
    if (glGetError() != GL_NO_ERROR ||
        max_texture_size <= 0 || max_3d_texture_size <= 0 ||
        max_cube_map_texture_size <= 0) {
        fprintf(stderr,
                "framebuffer_texture_layer_validation: invalid texture limits "
                "2D=%d 3D=%d cube=%d\n",
                max_texture_size, max_3d_texture_size,
                max_cube_map_texture_size);
        goto cleanup;
    }

    glGenTextures(5, no_storage);
    struct MipLimitCase {
        const char *label;
        GLenum target;
        GLint max_size;
        GLint legal_layer;
        GLboolean named;
    } mip_cases[] = {
        {"framebuffer_texture_layer_validation: no-storage 3D",
         GL_TEXTURE_3D, max_3d_texture_size, 1, GL_FALSE},
        {"framebuffer_texture_layer_validation: no-storage 1D array",
         GL_TEXTURE_1D_ARRAY, max_texture_size, 1, GL_TRUE},
        {"framebuffer_texture_layer_validation: no-storage 2D array",
         GL_TEXTURE_2D_ARRAY, max_texture_size, 1, GL_FALSE},
        {"framebuffer_texture_layer_validation: no-storage cube",
         GL_TEXTURE_CUBE_MAP, max_cube_map_texture_size, 2, GL_TRUE},
        {"framebuffer_texture_layer_validation: no-storage cube array",
         GL_TEXTURE_CUBE_MAP_ARRAY, max_cube_map_texture_size, 1, GL_FALSE},
    };

    for (size_t i = 0u; i < sizeof(mip_cases) / sizeof(mip_cases[0]); ++i) {
        char top_label[192];
        char overflow_label[192];
        char unchanged_label[192];
        struct FramebufferAttachmentState top_state;
        GLint top_level = framebuffer_test_max_mip_level(mip_cases[i].max_size);

        glBindTexture(mip_cases[i].target, no_storage[i]);
        drain_gl_errors();
        if (mip_cases[i].named) {
            glNamedFramebufferTextureLayer(fbo, GL_COLOR_ATTACHMENT0,
                                           no_storage[i], top_level,
                                           mip_cases[i].legal_layer);
        } else {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      no_storage[i], top_level,
                                      mip_cases[i].legal_layer);
        }
        snprintf(top_label, sizeof(top_label), "%s top", mip_cases[i].label);
        if (expect_single_gl_error(top_label, GL_NO_ERROR)) {
            goto cleanup;
        }
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
            GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
            fprintf(stderr, "%s: top level unexpectedly complete\n",
                    mip_cases[i].label);
            goto cleanup;
        }
        if (capture_framebuffer_attachment_state(top_label, &top_state) ||
            top_state.object != (GLint)no_storage[i] ||
            top_state.level != top_level ||
            top_state.layer != (mip_cases[i].target == GL_TEXTURE_CUBE_MAP
                                    ? 0 : mip_cases[i].legal_layer) ||
            top_state.layered != GL_FALSE) {
            fprintf(stderr,
                    "%s: top attachment=(object=%d level=%d layer=%d layered=%d)\n",
                    mip_cases[i].label,
                    top_state.object, top_state.level, top_state.layer,
                    top_state.layered);
            goto cleanup;
        }

        drain_gl_errors();
        if (mip_cases[i].named) {
            glNamedFramebufferTextureLayer(fbo, GL_COLOR_ATTACHMENT0,
                                           no_storage[i], top_level + 1,
                                           mip_cases[i].legal_layer + 1);
        } else {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      no_storage[i], top_level + 1,
                                      mip_cases[i].legal_layer + 1);
        }
        snprintf(overflow_label, sizeof(overflow_label), "%s top+1",
                 mip_cases[i].label);
        snprintf(unchanged_label, sizeof(unchanged_label), "%s unchanged",
                 mip_cases[i].label);
        if (expect_single_gl_error(overflow_label, GL_INVALID_VALUE) ||
            expect_framebuffer_attachment_state(unchanged_label, &top_state)) {
            goto cleanup;
        }

        if (mip_cases[i].target == GL_TEXTURE_CUBE_MAP) {
            glBindTexture(GL_TEXTURE_CUBE_MAP, no_storage[i]);
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X +
                             mip_cases[i].legal_layer,
                         top_level, GL_RGBA8, 1, 1, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            if (expect_single_gl_error(
                    "framebuffer_texture_layer_validation: cube layer preserved",
                    GL_NO_ERROR) ||
                glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
                    GL_FRAMEBUFFER_COMPLETE) {
                fprintf(stderr,
                        "framebuffer_texture_layer_validation: cube layer changed "
                        "after top+1 failure\n");
                goto cleanup;
            }
        }
    }

    drain_gl_errors();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              0u, -1, -1);
    if (expect_single_gl_error("framebuffer_texture_layer_validation: detach ignores level/layer",
                               GL_NO_ERROR)) {
        goto cleanup;
    }
    GLint object_type = -1;
    glGetFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &object_type);
    if (glGetError() != GL_NO_ERROR || object_type != GL_NONE) {
        fprintf(stderr,
                "framebuffer_texture_layer_validation: detach object type=0x%x\n",
                object_type);
        goto cleanup;
    }

    glGenTextures(4, layered);
    glBindTexture(GL_TEXTURE_3D, layered[0]);
    glTexStorage3D(GL_TEXTURE_3D, 2, GL_RGBA8, 4, 4, 4);

    glBindTexture(GL_TEXTURE_1D_ARRAY, layered[1]);
    glTexStorage2D(GL_TEXTURE_1D_ARRAY, 2, GL_RGBA8, 4, 2);

    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, layered[2]);
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 2,
                            GL_RGBA8, 4, 4, 2, GL_TRUE);

    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, layered[3]);
    glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 1, GL_RGBA8, 4, 4, 6);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr,
                "framebuffer_texture_layer_validation: layered texture setup failed\n");
        goto cleanup;
    }

    struct LayerCase {
        const char *label;
        GLuint texture;
        GLint level;
        GLint valid_layer;
        GLint missing_layer;
        GLboolean named;
    } cases[] = {
        {"framebuffer_texture_layer_validation: 3D", layered[0], 1, 1, 2, GL_FALSE},
        {"framebuffer_texture_layer_validation: 1D array", layered[1], 1, 1, 2, GL_FALSE},
        {"framebuffer_texture_layer_validation: 2D array DSA", old_array, 0, 1, 2, GL_TRUE},
        {"framebuffer_texture_layer_validation: 2D MS array", layered[2], 0, 1, 2, GL_FALSE},
        {"framebuffer_texture_layer_validation: cube array", layered[3], 0, 5, 6, GL_FALSE},
    };

    for (size_t i = 0u; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        char valid_label[160];
        char missing_label[160];
        snprintf(valid_label, sizeof(valid_label), "%s valid", cases[i].label);
        snprintf(missing_label, sizeof(missing_label), "%s missing", cases[i].label);
        if (expect_framebuffer_layer_status(valid_label, fbo,
                                            cases[i].texture,
                                            cases[i].level,
                                            cases[i].valid_layer,
                                            cases[i].named,
                                            GL_FRAMEBUFFER_COMPLETE) ||
            expect_framebuffer_layer_status(missing_label, fbo,
                                            cases[i].texture,
                                            cases[i].level,
                                            cases[i].missing_layer,
                                            cases[i].named,
                                            GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT)) {
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    glBindFramebuffer(GL_FRAMEBUFFER, 0u);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (old_array) glDeleteTextures(1, &old_array);
    if (texture_2d) glDeleteTextures(1, &texture_2d);
    if (cube) glDeleteTextures(1, &cube);
    if (unrealized) glDeleteTextures(1, &unrealized);
    glDeleteTextures(4, layered);
    glDeleteTextures(5, no_storage);
    drain_gl_errors();
    return result;
}

/* GL 4.6 section 9.4.2 layered completeness: populated layered color
 * attachments must have the same texture target.  Their layer counts may
 * differ; rendering is limited to the smallest attachment layer count. */
static int test_framebuffer_layer_targets(unsigned char *pixels,
                                          const char *out_path)
{
    (void)pixels;
    (void)out_path;
    GLuint fbo = 0u;
    GLuint array_textures[2] = {0u, 0u};
    GLuint texture_3d = 0u;
    int result = 1;

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(2, array_textures);
    for (GLuint i = 0u; i < 2u; ++i) {
        glBindTexture(GL_TEXTURE_2D_ARRAY, array_textures[i]);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 8, 8,
                     i == 0u ? 2 : 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    glGenTextures(1, &texture_3d);
    glBindTexture(GL_TEXTURE_3D, texture_3d);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, 8, 8, 4, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    /* Same target with different layer counts remains framebuffer-complete. */
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                         array_textures[0], 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                         array_textures[1], 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "framebuffer_layer_targets: same-target layered FBO status=0x%x\n",
                status);
        goto cleanup;
    }

    /* Both attachments are layered, but their texture targets differ. */
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texture_3d, 0);
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS) {
        fprintf(stderr,
                "framebuffer_layer_targets: target mismatch status=0x%x\n",
                status);
        goto cleanup;
    }

    /* Preserve the existing layered/non-layered attachment mismatch rule. */
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                              array_textures[1], 0, 0);
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS) {
        fprintf(stderr,
                "framebuffer_layer_targets: layered mix status=0x%x\n",
                status);
        goto cleanup;
    }

    result = 0;

cleanup:
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (array_textures[0] || array_textures[1]) {
        glDeleteTextures(2, array_textures);
    }
    if (texture_3d) glDeleteTextures(1, &texture_3d);
    return result;
}

/* glFramebufferTextureLayer addresses cube-map faces through layer 0..5.
 * Writing a non-zero face must not alias face 0 in the Metal slice mapping. */
static int test_framebuffer_cube_layer_slice(unsigned char *pixels,
                                             const char *out_path)
{
    (void)out_path;
    GLuint fbo = 0u;
    GLuint cube = 0u;
    GLuint sparse_cube = 0u;
    int result = 1;
    const int width = 8;
    const int height = 8;
    const unsigned char *center;

    glGenTextures(1, &cube);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
    for (GLuint face = 0u; face < 6u; ++face) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA8,
                     width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    /* A cube layer attachment references exactly the selected face for
     * completeness, even when the other five faces have no storage. */
    glGenTextures(1, &sparse_cube);
    glBindTexture(GL_TEXTURE_CUBE_MAP, sparse_cube);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA8,
                 width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              sparse_cube, 0, 3);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: sparse face-3 layer incomplete\n");
        goto cleanup;
    }
    {
        GLint layer = -1;
        glGetFramebufferAttachmentParameteriv(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER, &layer);
        if (layer != 0) {
            fprintf(stderr,
                    "framebuffer_cube_layer_slice: cube layer query=%d expected=0\n",
                    layer);
            goto cleanup;
        }
    }

    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              sparse_cube, 0, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: missing face-0 layer complete\n");
        goto cleanup;
    }

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, sparse_cube, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: sparse face-enum incomplete\n");
        goto cleanup;
    }

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_CUBE_MAP_POSITIVE_X, sparse_cube, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: missing face-enum complete\n");
        goto cleanup;
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, sparse_cube);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA8,
                 width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              sparse_cube, 0, 4);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: missing face-4 layer complete\n");
        goto cleanup;
    }

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, sparse_cube, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: sparse whole cube complete\n");
        goto cleanup;
    }

    for (GLuint face = 1u; face < 6u; ++face) {
        if (face == 3u) continue;
        GLsizei face_width = face == 5u ? width / 2 : width;
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA8,
                     face_width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: non-square whole cube complete\n");
        goto cleanup;
    }

    for (GLuint face = 0u; face < 6u; ++face) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA8,
                     width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: complete whole cube incomplete\n");
        goto cleanup;
    }
    {
        GLint layered = GL_FALSE;
        GLint layer = -1;
        glGetFramebufferAttachmentParameteriv(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_FRAMEBUFFER_ATTACHMENT_LAYERED, &layered);
        glGetFramebufferAttachmentParameteriv(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER, &layer);
        if (layered != GL_TRUE || layer != 0) {
            fprintf(stderr,
                    "framebuffer_cube_layer_slice: whole cube layered=%d layer=%d\n",
                    layered, layer);
            goto cleanup;
        }
    }

    /* Keep a whole-cube clear pending while changing the attachment so the
     * CPU fallback must materialize all six faces before consuming it. */
    {
        const GLfloat blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cube, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr,
                    "framebuffer_cube_layer_slice: fallback whole cube incomplete\n");
            goto cleanup;
        }
        glEnable(GL_SCISSOR_TEST);
        glScissor(0, 0, width, height);
        glClearBufferfv(GL_COLOR, 0, blue);
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  cube, 0, 0);
        glDisable(GL_SCISSOR_TEST);

        for (GLuint face = 0u; face < 6u; ++face) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      cube, 0, (GLint)face);
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
            center = &pixels[((height / 2) * width + width / 2) * 4];
            if (center[0] > 20u || center[1] > 20u || center[2] < 220u) {
                fprintf(stderr,
                        "framebuffer_cube_layer_slice: face %u fallback clear=(%u,%u,%u)\n",
                        face, center[0], center[1], center[2]);
                goto cleanup;
            }
        }
    }

    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cube, 0, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: face-0 FBO incomplete\n");
        goto cleanup;
    }
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cube, 0, 3);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: face-3 FBO incomplete\n");
        goto cleanup;
    }
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    center = &pixels[((height / 2) * width + width / 2) * 4];
    if (center[0] > 20u || center[1] < 220u || center[2] > 20u) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: face 3 not green (%u,%u,%u)\n",
                center[0], center[1], center[2]);
        goto cleanup;
    }

    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cube, 0, 0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    center = &pixels[((height / 2) * width + width / 2) * 4];
    if (center[0] < 220u || center[1] > 20u || center[2] > 20u) {
        fprintf(stderr,
                "framebuffer_cube_layer_slice: face 0 changed (%u,%u,%u)\n",
                center[0], center[1], center[2]);
        goto cleanup;
    }

    result = 0;

cleanup:
    glBindFramebuffer(GL_FRAMEBUFFER, 0u);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (sparse_cube) glDeleteTextures(1, &sparse_cube);
    if (cube) glDeleteTextures(1, &cube);
    return result;
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
/* P4.1e2 occlusion-query scissor probe.  GL_SAMPLES_PASSED on a 2D FBO:
 * a visible triangle must count > 0, and a draw clipped to a 0-size scissor
 * rect must count exactly 0.  The zero-scissor round is repeated several
 * times so encoder recreation between queries is exercised; this is the
 * guard for the conditional_render scissor-dedup regression (P4.1e). */
static int test_air_query_scissor_occluded(unsigned char *pixels,
                                            const char *out_path)
{
    (void)pixels;
    (void)out_path;
    GLuint fbo = 0u, tex = 0u, vao = 0u, vbo_p = 0u, vbo_c = 0u;
    GLuint q = 0u, q_zero[4] = {0u, 0u, 0u, 0u};
    int result = 1;

    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) goto cleanup;
    GLuint prog = link_program(VS_BASIC, FS_BASIC);
    if (!prog) goto cleanup;
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "u_scale"), 1.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    vbo_p = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    vbo_c = make_vbo(TRI_COLORS, sizeof(TRI_COLORS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    /* Visible triangle: samples > 0. */
    glGenQueries(1, &q);
    glBeginQuery(GL_SAMPLES_PASSED, q);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndQuery(GL_SAMPLES_PASSED);
    glFinish();
    GLuint visible = 0u;
    glGetQueryObjectuiv(q, GL_QUERY_RESULT, &visible);
    if (visible == 0u) {
        fprintf(stderr, "air_query_scissor_occluded: visible query returned 0\n");
        goto cleanup;
    }

    /* Repeated 0-size-scissor draws: every query must return exactly 0. */
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 0, 0);
    for (int round = 0; round < 4; ++round) {
        glGenQueries(1, &q_zero[round]);
        glBeginQuery(GL_SAMPLES_PASSED, q_zero[round]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndQuery(GL_SAMPLES_PASSED);
        glFinish();
        GLuint occluded = 1u;
        glGetQueryObjectuiv(q_zero[round], GL_QUERY_RESULT, &occluded);
        if (occluded != 0u) {
            fprintf(stderr,
                    "air_query_scissor_occluded: occluded round %d returned "
                    "%u samples (expected 0)\n", round, occluded);
            goto cleanup;
        }
    }
    glDisable(GL_SCISSOR_TEST);

    /* Verify the same framebuffer still renders (scissor fully cleared). */
    clear_color(0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUniform2f(glGetUniformLocation(prog, "u_offset"), 0.0f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    const int px = (int)((0.0f + 1.0f) * 0.5f * REG_W);
    const int py = (int)((0.0f + 1.0f) * 0.5f * REG_H);
    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
    if (c[0] < 200u || c[1] > 60u || c[2] > 60u) {
        fprintf(stderr,
                "air_query_scissor_occluded: post-scissor triangle not red "
                "(%u,%u,%u)\n", c[0], c[1], c[2]);
        goto cleanup;
    }

    result = 0;

cleanup:
    for (int i = 0; i < 4; ++i) {
        if (q_zero[i]) glDeleteQueries(1, &q_zero[i]);
    }
    if (q) glDeleteQueries(1, &q);
    if (vbo_c) glDeleteBuffers(1, &vbo_c);
    if (vbo_p) glDeleteBuffers(1, &vbo_p);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (prog) glDeleteProgram(prog);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (tex) glDeleteTextures(1, &tex);
    return result;
}

/* P4.1e2 layer-binding probe: a 2D-ARRAY color attachment bound through
 * glFramebufferTextureLayer at slice ∈ {0, 1}, with a program that has no
 * gl_Layer output.  GL 4.6 §9.4.2: the bound layer is the draw target.
 * The Metal render pass must remain non-layered, preserve the selected
 * attachment slice, and ignore render_target_array_index. */
static int test_air_renderpass_layer_slice(unsigned char *pixels,
                                           const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 330 core\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
    static const char *fs =
        "#version 330 core\n"
        "out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    GLuint lfbo = 0u, color = 0u, vao = 0u, vbo = 0u, prog = 0u;
    int result = 1;

    lfbo = make_layer_fbo(REG_W, REG_H, &color);
    if (!lfbo) goto cleanup;
    prog = link_program(vs, fs);
    if (!prog) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, lfbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    static const float tri[6] = { -0.6f, -0.6f, 0.6f, -0.6f, 0.0f, 0.6f };
    vbo = make_vbo(tri, sizeof(tri));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(prog);

    const int px = (int)((0.0f + 1.0f) * 0.5f * REG_W);
    const int py = (int)((0.0f + 1.0f) * 0.5f * REG_H);
    const int bgx = REG_W / 8;
    const int bgy = REG_H / 8;
    const unsigned char *c;

    /* Segment 1: slice 0 binding, no gl_Layer output → triangle on layer 0. */
    clear_color(0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    c = &pixels[(py * REG_W + px) * 4];
    if (c[0] > 20u || c[1] < 220u || c[2] > 20u) {
        fprintf(stderr,
                "air_renderpass_layer_slice: slice-0 draw not green on layer "
                "0 (%u,%u,%u)\n", c[0], c[1], c[2]);
        goto cleanup;
    }
    c = &pixels[(bgy * REG_W + bgx) * 4];
    if (c[0] > 20u || c[1] > 20u || c[2] > 20u) {
        fprintf(stderr,
                "air_renderpass_layer_slice: slice-0 background not black "
                "(%u,%u,%u)\n", c[0], c[1], c[2]);
        goto cleanup;
    }

    /* Segment 2: slice 1 binding, no gl_Layer output.  The clear and draw
     * must affect only layer 1. */
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color,
                              0, 1);
    clear_color(0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    c = &pixels[(py * REG_W + px) * 4];
    if (c[0] > 20u || c[1] < 220u || c[2] > 20u) {
        fprintf(stderr,
                "air_renderpass_layer_slice: slice-1 draw not green on layer "
                "1 (%u,%u,%u)\n",
                c[0], c[1], c[2]);
        goto cleanup;
    }
    c = &pixels[(bgy * REG_W + bgx) * 4];
    if (c[0] > 20u || c[1] > 20u || c[2] < 220u) {
        fprintf(stderr,
                "air_renderpass_layer_slice: slice-1 background not blue "
                "(%u,%u,%u)\n", c[0], c[1], c[2]);
        goto cleanup;
    }

    /* Layer 0 must retain the image produced by segment 1. */
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color,
                              0, 0);
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    c = &pixels[(py * REG_W + px) * 4];
    if (c[0] > 20u || c[1] < 220u || c[2] > 20u) {
        fprintf(stderr,
                "air_renderpass_layer_slice: slice-0 image changed at center "
                "(%u,%u,%u)\n", c[0], c[1], c[2]);
        goto cleanup;
    }
    c = &pixels[(bgy * REG_W + bgx) * 4];
    if (c[0] > 20u || c[1] > 20u || c[2] > 20u) {
        fprintf(stderr,
                "air_renderpass_layer_slice: slice-0 background changed "
                "(%u,%u,%u)\n", c[0], c[1], c[2]);
        goto cleanup;
    }

    result = 0;

cleanup:
    if (prog) glDeleteProgram(prog);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (lfbo) glDeleteFramebuffers(1, &lfbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

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

/* Exercise the complete AIR CullDistance product path: stage reflection
 * marks the program, the renderer locates culldistance_data, and the hidden
 * vertex buffers implement primitive-level all-negative culling. */
static int test_air_cull_distance(unsigned char *pixels,
                                  const char *out_path)
{
    (void)out_path;
    int rc = 0;
    GLuint fbo = 0, color_tex = 0, program = 0, vao = 0, ebo = 0;
    GLuint indirect_buffer = 0;
    GLuint buffers[3] = {0, 0, 0};

    fbo = make_fbo(REG_W, REG_H, &color_tex);
    if (!fbo) return 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    program = link_program(
        "#version 460 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=1) in float culldistance_data;\n"
        "void main(){\n"
        "  vec2 p=pos;\n"
        "  p.x += float(gl_InstanceID);\n"
        "  float d=culldistance_data;\n"
        "  if (gl_InstanceID == 1) d += 2.0;\n"
        "  if (gl_BaseInstance != 0) {\n"
        "    if (gl_BaseInstance != 5) d = -100.0;\n"
        "  }\n"
        "  gl_Position=vec4(p,0.0,1.0);\n"
        "  gl_CullDistance[0]=d;\n"
        "}\n",
        "#version 460 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main(){ frag=vec4(0.0,1.0,0.0,1.0); }\n");
    if (!program) {
        rc = 2;
        goto cleanup;
    }
    glUseProgram(program);

    static const float positions[] = {
         4.0f,  4.0f,
        -1.0f, -1.0f,
         3.0f, -1.0f,
        -1.0f,  3.0f,
    };
    static const float all_negative[] = {1.0f, -1.0f, -1.0f, -1.0f};
    static const float one_positive[] = {1.0f, -1.0f, 1.0f, -1.0f};

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(3, buffers);
    const void *data[3] = {positions, all_negative, one_positive};
    const GLsizeiptr sizes[3] = {
        (GLsizeiptr)sizeof(positions),
        (GLsizeiptr)sizeof(all_negative),
        (GLsizeiptr)sizeof(one_positive),
    };
    for (int i = 0; i < 3; i++) {
        glBindBuffer(GL_ARRAY_BUFFER, buffers[i]);
        glBufferData(GL_ARRAY_BUFFER, sizes[i], data[i], GL_STATIC_DRAW);
    }

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);

    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 1, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_cull_distance: all-negative draw failed\n");
        rc = 3;
        goto cleanup;
    }
    for (size_t i = 0; i < (size_t)REG_W * REG_H; i++) {
        if (pixels[i * 4 + 0] != 0 || pixels[i * 4 + 1] != 0 ||
            pixels[i * 4 + 2] != 0) {
            fprintf(stderr,
                    "air_cull_distance: all-negative primitive was visible\n");
            rc = 4;
            goto cleanup;
        }
    }

    clear_color(0.0f, 0.0f, 0.0f);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_TRIANGLES, 1, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_cull_distance: one-positive draw failed\n");
        rc = 5;
        goto cleanup;
    }
    {
        const unsigned char *center =
            &pixels[((REG_H / 2) * REG_W + REG_W / 2) * 4];
        if (center[0] > 2 || center[1] < 250 || center[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: one-positive primitive missing "
                    "(center=%u,%u,%u)\n",
                    center[0], center[1], center[2]);
            rc = 6;
        }
    }
    if (rc != 0) goto cleanup;

    static const float strip_positions[] = {
        -1.0f, -1.0f,
         0.0f, -1.0f,
        -1.0f,  1.0f,
         0.0f,  1.0f,
    };
    static const float shared_primitive_distances[] = {
        -1.0f, -1.0f, -1.0f, 1.0f,
    };
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(strip_positions), strip_positions,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(shared_primitive_distances),
                 shared_primitive_distances, GL_STATIC_DRAW);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled = &pixels[(32 * REG_W + 16) * 4];
        const unsigned char *visible = &pixels[(96 * REG_W + 48) * 4];
        if (culled[0] > 2 || culled[1] > 2 || culled[2] > 2 ||
            visible[0] > 2 || visible[1] < 250 || visible[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: triangle strip primitive split "
                    "failed (culled=%u,%u,%u visible=%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            rc = 7;
            goto cleanup;
        }
    }

    static const float fan_positions[] = {
         0.0f,  0.0f,
        -1.0f, -1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f,
    };
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(fan_positions), fan_positions,
                 GL_STATIC_DRAW);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled = &pixels[(32 * REG_W + 48) * 4];
        const unsigned char *visible = &pixels[(80 * REG_W + 112) * 4];
        if (culled[0] > 2 || culled[1] > 2 || culled[2] > 2 ||
            visible[0] > 2 || visible[1] < 250 || visible[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: triangle fan primitive split "
                    "failed (culled=%u,%u,%u visible=%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            rc = 8;
        }
    }
    if (rc != 0) goto cleanup;

    static const float indexed_positions[] = {
         0.0f, -1.0f,
         0.0f,  1.0f,
        -1.0f, -1.0f,
        -1.0f,  1.0f,
    };
    static const float indexed_distances[] = {
        -1.0f, 1.0f, -1.0f, -1.0f,
    };
    static const unsigned short indexed_strip[] = {2, 0, 3, 1};
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(indexed_positions),
                 indexed_positions, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(indexed_distances),
                 indexed_distances, GL_STATIC_DRAW);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indexed_strip),
                 indexed_strip, GL_STATIC_DRAW);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled = &pixels[(32 * REG_W + 16) * 4];
        const unsigned char *visible = &pixels[(96 * REG_W + 48) * 4];
        if (culled[0] > 2 || culled[1] > 2 || culled[2] > 2 ||
            visible[0] > 2 || visible[1] < 250 || visible[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: indexed triangle strip split "
                    "failed (culled=%u,%u,%u visible=%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            rc = 9;
        }
    }
    if (rc != 0) goto cleanup;

    static const float instanced_positions[] = {
        -0.9f, -0.6f,
        -0.1f, -0.6f,
        -0.5f,  0.6f,
    };
    static const float instanced_distances[] = {-1.0f, -1.0f, -1.0f};
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(instanced_positions),
                 instanced_positions, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(instanced_distances),
                 instanced_distances, GL_STATIC_DRAW);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 2);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled = &pixels[(64 * REG_W + 32) * 4];
        const unsigned char *visible = &pixels[(64 * REG_W + 96) * 4];
        if (culled[0] > 2 || culled[1] > 2 || culled[2] > 2 ||
            visible[0] > 2 || visible[1] < 250 || visible[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: gl_InstanceID capture split failed "
                    "(culled=%u,%u,%u visible=%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            rc = 10;
            goto cleanup;
        }
    }

    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 0, 3, 2, 5u);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled = &pixels[(64 * REG_W + 32) * 4];
        const unsigned char *visible = &pixels[(64 * REG_W + 96) * 4];
        if (culled[0] > 2 || culled[1] > 2 || culled[2] > 2 ||
            visible[0] > 2 || visible[1] < 250 || visible[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: base-instance capture split failed "
                    "(culled=%u,%u,%u visible=%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            rc = 11;
        }
    }
    if (rc != 0) goto cleanup;

    static const float multi_positions[] = {
        -0.9f, -0.6f,
        -0.1f, -0.6f,
        -0.5f,  0.6f,
         0.1f, -0.6f,
         0.9f, -0.6f,
         0.5f,  0.6f,
    };
    static const float multi_distances[] = {
        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
    };
    static const GLint multi_firsts[] = {0, 3};
    static const GLsizei multi_counts[] = {3, 3};
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(multi_positions), multi_positions,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(multi_distances), multi_distances,
                 GL_STATIC_DRAW);

    clear_color(0.0f, 0.0f, 0.0f);
    glMultiDrawArrays(GL_TRIANGLES, multi_firsts, multi_counts, 2);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled = &pixels[(64 * REG_W + 32) * 4];
        const unsigned char *visible = &pixels[(64 * REG_W + 96) * 4];
        if (culled[0] > 2 || culled[1] > 2 || culled[2] > 2 ||
            visible[0] > 2 || visible[1] < 250 || visible[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: multi-draw array split failed "
                    "(culled=%u,%u,%u visible=%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            rc = 12;
            goto cleanup;
        }
    }

    static const GLuint array_indirect_commands[] = {
        3u, 1u, 0u, 0u,
        3u, 1u, 3u, 0u,
    };
    glGenBuffers(1, &indirect_buffer);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buffer);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(array_indirect_commands),
                 array_indirect_commands, GL_STATIC_DRAW);
    clear_color(0.0f, 0.0f, 0.0f);
    glMultiDrawArraysIndirect(GL_TRIANGLES, 0, 2, 0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled = &pixels[(64 * REG_W + 32) * 4];
        const unsigned char *visible = &pixels[(64 * REG_W + 96) * 4];
        if (culled[0] > 2 || culled[1] > 2 || culled[2] > 2 ||
            visible[0] > 2 || visible[1] < 250 || visible[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: indirect array split failed "
                    "(culled=%u,%u,%u visible=%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            rc = 13;
            goto cleanup;
        }
    }

    static const unsigned short multi_indices[] = {0, 1, 2, 3, 4, 5};
    static const GLuint element_indirect_commands[] = {
        3u, 1u, 0u, 0u, 0u,
        3u, 1u, 3u, 0u, 0u,
    };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(multi_indices), multi_indices,
                 GL_STATIC_DRAW);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(element_indirect_commands),
                 element_indirect_commands, GL_STATIC_DRAW);
    clear_color(0.0f, 0.0f, 0.0f);
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, 0, 2, 0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled = &pixels[(64 * REG_W + 32) * 4];
        const unsigned char *visible = &pixels[(64 * REG_W + 96) * 4];
        if (culled[0] > 2 || culled[1] > 2 || culled[2] > 2 ||
            visible[0] > 2 || visible[1] < 250 || visible[2] > 2) {
            fprintf(stderr,
                    "air_cull_distance: indirect element split failed "
                    "(culled=%u,%u,%u visible=%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            rc = 14;
        }
    }

cleanup:
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    if (indirect_buffer) glDeleteBuffers(1, &indirect_buffer);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (buffers[0] || buffers[1] || buffers[2]) glDeleteBuffers(3, buffers);
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

/* ---- MSAA resolve + FBO switch (P4.4 coverage) ----
 * Render into a 4x multisample FBO, glBlitFramebuffer the MSAA color
 * attachment into a single-sample FBO (GL's multisample->single-sample
 * resolve path), and verify the resolved pixels.  Segment B re-renders the
 * MSAA FBO after the resolve blit and FBO switch, proving the multisample
 * target stays renderable. */
static int test_air_msaa_resolve(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    const int SAMPLES = 4;

    /* MSAA source FBO: 4x color + 4x depth renderbuffers. */
    GLuint msaaFbo = 0, msaaColor = 0, msaaDepth = 0;
    glGenFramebuffers(1, &msaaFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, msaaFbo);
    glGenRenderbuffers(1, &msaaColor);
    glBindRenderbuffer(GL_RENDERBUFFER, msaaColor);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, SAMPLES, GL_RGBA8,
                                     REG_W, REG_H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, msaaColor);
    glGenRenderbuffers(1, &msaaDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, msaaDepth);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, SAMPLES,
                                     GL_DEPTH_COMPONENT24, REG_W, REG_H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, msaaDepth);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "air_msaa_resolve: MSAA FBO incomplete\n");
        return 1;
    }

    /* Single-sample destination FBOs (resolved images). */
    GLuint dstA = 0, dstATex = 0, dstB = 0, dstBTex = 0;
    dstA = make_fbo(REG_W, REG_H, &dstATex);
    dstB = make_fbo(REG_W, REG_H, &dstBTex);
    if (!dstA || !dstB) return 1;

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

    glViewport(0, 0, REG_W, REG_H);
    glDisable(GL_SCISSOR_TEST);

    /* Segment A: red triangle into the MSAA FBO, then resolve-blit to dstA. */
    glBindFramebuffer(GL_FRAMEBUFFER, msaaFbo);
    clear_color(0.05f, 0.05f, 0.3f); /* dark blue background */
    glUseProgram(progRed);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, msaaFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dstA);
    glBlitFramebuffer(0, 0, REG_W, REG_H, 0, 0, REG_W, REG_H,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glFinish();
    glBindFramebuffer(GL_FRAMEBUFFER, dstA);
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Triangle (-0.6,-0.6) (0.6,-0.6) (0,0.6): interior probes. */
        static const float probes[3][2] = {
            { -0.35f, -0.35f }, { 0.35f, -0.35f }, { 0.0f, 0.35f },
        };
        for (int i = 0; i < 3; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] >= 200u && c[1] <= 60u && c[2] <= 60u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_msaa_resolve: seg A probe %d not red at (%d,%d)\n",
                        i, sx, sy);
                return 3;
            }
        }
        /* Exterior probe must stay the clear color (resolve did not smear). */
        const int ex = (int)((-0.8f + 1.0f) * 0.5f * REG_W);
        const int ey = (int)((0.0f + 1.0f) * 0.5f * REG_H);
        const unsigned char *c = &pixels[(ey * REG_W + ex) * 4];
        if (!(c[0] <= 60u && c[1] <= 60u && c[2] >= 50u)) {
            fprintf(stderr,
                    "air_msaa_resolve: seg A exterior not clear at (%d,%d) "
                    "rgb=(%u,%u,%u)\n",
                    ex, ey, c[0], c[1], c[2]);
            return 4;
        }
    }

    /* Segment B: re-render the MSAA FBO (green triangle) after the resolve
     * blit + FBO switch, resolve to dstB, verify green interior + clear
     * exterior.  Proves the multisample target stays renderable. */
    glBindFramebuffer(GL_FRAMEBUFFER, msaaFbo);
    clear_color(0.3f, 0.05f, 0.05f); /* dark red background */
    glUseProgram(progGreen);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, msaaFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dstB);
    glBlitFramebuffer(0, 0, REG_W, REG_H, 0, 0, REG_W, REG_H,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glFinish();
    glBindFramebuffer(GL_FRAMEBUFFER, dstB);
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        static const float probes[3][2] = {
            { -0.35f, -0.35f }, { 0.35f, -0.35f }, { 0.0f, 0.35f },
        };
        for (int i = 0; i < 3; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 60u && c[1] >= 200u && c[2] <= 60u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_msaa_resolve: seg B probe %d not green at (%d,%d)\n",
                        i, sx, sy);
                return 5;
            }
        }
        const int ex = (int)((-0.8f + 1.0f) * 0.5f * REG_W);
        const int ey = (int)((0.0f + 1.0f) * 0.5f * REG_H);
        const unsigned char *c = &pixels[(ey * REG_W + ex) * 4];
        if (!(c[0] >= 50u && c[1] <= 60u && c[2] <= 60u)) {
            fprintf(stderr,
                    "air_msaa_resolve: seg B exterior not clear at (%d,%d)\n",
                    ex, ey);
            return 6;
        }
    }

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(progRed);
    glDeleteProgram(progGreen);
    glDeleteFramebuffers(1, &msaaFbo);
    glDeleteRenderbuffers(1, &msaaColor);
    glDeleteRenderbuffers(1, &msaaDepth);
    glDeleteFramebuffers(1, &dstA);
    glDeleteFramebuffers(1, &dstB);
    glDeleteTextures(1, &dstATex);
    glDeleteTextures(1, &dstBTex);
    return 0;
}

/* ---- Legacy GLSL frontend wiring (item 753 follow-up) ----
 * Compile GLSL 1.10-style sources (attribute/varying/gl_FragColor) through
 * the product path: glShaderSource stores the raw source, mglAirCompileStage
 * feeds it to the AIR frontend, which must detect + translate pre-3.30
 * constructs BEFORE parsing (mgl_legacy_compat wiring in mgl_air_backend.cpp).
 * Segment A: legacy VS/FS with gl_FragColor -> red triangle interior.
 * Segment B: legacy texture2D() sampling a red 1x1 texture -> red interior. */
/* ---- Legacy clip planes (GL 1.1 glClipPlane / glGetClipPlane) ----
 * The legacy clip-plane GL surface: equations are stored as given
 * (MGL's fixed-function matrix stack is unimplemented, so the GL 1.1
 * eye-space transform is identity), glGetClipPlane returns them, and
 * GL_CLIP_PLANE0..5 share the GL_CLIP_DISTANCE0..5 values so
 * glEnable/glIsEnabled already route through the clip-distance caps.
 * The shader-side derivation (gl_ClipVertex -> clip distances) is
 * covered separately. */
static int test_gl_clip_planes(unsigned char *pixels, const char *out_path)
{
    (void)pixels;
    (void)out_path;

    const GLdouble eq0[4] = { 1.0, 0.0, 0.0, 0.5 };
    const GLdouble eq1[4] = { 0.0, 1.0, 0.0, -0.25 };
    const GLdouble eq5[4] = { 0.0, 0.0, 1.0, 2.0 };
    GLdouble got[4];

    /* Default state: planes are zero, disabled. */
    memset(got, 0x7f, sizeof(got));
    glGetClipPlane(GL_CLIP_PLANE0, got);
    if (got[0] != 0.0 || got[1] != 0.0 || got[2] != 0.0 || got[3] != 0.0) {
        fprintf(stderr, "gl_clip_planes: default plane0 not zero "
                "(%g,%g,%g,%g)\n", got[0], got[1], got[2], got[3]);
        return 1;
    }
    if (glIsEnabled(GL_CLIP_PLANE0) || glIsEnabled(GL_CLIP_PLANE5)) {
        fprintf(stderr, "gl_clip_planes: clip planes enabled by default\n");
        return 2;
    }

    /* Set/get roundtrip on planes 0, 1 and 5. */
    glClipPlane(GL_CLIP_PLANE0, eq0);
    glClipPlane(GL_CLIP_PLANE1, eq1);
    glClipPlane(GL_CLIP_PLANE5, eq5);
    memset(got, 0x7f, sizeof(got));
    glGetClipPlane(GL_CLIP_PLANE0, got);
    if (memcmp(got, eq0, sizeof(eq0)) != 0) {
        fprintf(stderr, "gl_clip_planes: plane0 mismatch "
                "(%g,%g,%g,%g)\n", got[0], got[1], got[2], got[3]);
        return 3;
    }
    memset(got, 0x7f, sizeof(got));
    glGetClipPlane(GL_CLIP_PLANE1, got);
    if (memcmp(got, eq1, sizeof(eq1)) != 0) {
        fprintf(stderr, "gl_clip_planes: plane1 mismatch "
                "(%g,%g,%g,%g)\n", got[0], got[1], got[2], got[3]);
        return 4;
    }
    memset(got, 0x7f, sizeof(got));
    glGetClipPlane(GL_CLIP_PLANE5, got);
    if (memcmp(got, eq5, sizeof(eq5)) != 0) {
        fprintf(stderr, "gl_clip_planes: plane5 mismatch "
                "(%g,%g,%g,%g)\n", got[0], got[1], got[2], got[3]);
        return 5;
    }

    /* glClipPlane must not disturb the other planes. */
    memset(got, 0x7f, sizeof(got));
    glGetClipPlane(GL_CLIP_PLANE2, got);
    if (got[0] != 0.0 || got[3] != 0.0) {
        fprintf(stderr, "gl_clip_planes: plane2 disturbed "
                "(%g,%g,%g,%g)\n", got[0], got[1], got[2], got[3]);
        return 6;
    }

    /* Enable/disable routing through the shared clip caps. */
    glEnable(GL_CLIP_PLANE1);
    if (!glIsEnabled(GL_CLIP_PLANE1)) {
        fprintf(stderr, "gl_clip_planes: enable plane1 not reflected\n");
        return 7;
    }
    glDisable(GL_CLIP_PLANE1);
    if (glIsEnabled(GL_CLIP_PLANE1)) {
        fprintf(stderr, "gl_clip_planes: disable plane1 not reflected\n");
        return 8;
    }

    /* Out-of-range plane is GL_INVALID_ENUM. */
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "gl_clip_planes: stray GL error 0x%x\n",
                (unsigned)err);
        return 9;
    }
    glClipPlane(0x3008, eq0);   /* beyond GL_CLIP_PLANE7/GL_CLIP_DISTANCE7 */
    err = glGetError();
    if (err != GL_INVALID_ENUM) {
        fprintf(stderr, "gl_clip_planes: out-of-range plane not rejected "
                "(0x%x)\n", (unsigned)err);
        return 10;
    }

    return 0;
}

/* ---- Legacy clip-plane shader derivation (gl_ClipVertex) ----
 * A GLSL 1.10 VS writing gl_ClipVertex gets a wrapper main injected by the
 * translator that derives gl_ClipDistance[i] = mix(1, dot(plane_i, clipVtx),
 * enabled_i) from the _mglClipPlane/_mglClipPlaneEnabled uniforms, which
 * mglDrawDispatch refreshes per draw from the glClipPlane + enable caps.
 * The clip vertex here is a constant eye-space point (identity modelview),
 * so plane (0,0,-1,0) gives distance -0.5 and clips everywhere. */
static int test_legacy_clip_vertex(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    const char *vs330 =
        "#version 330 core\n"
        "layout(location = 0) in vec2 a_pos;\n"
        "uniform vec4 _mglClipPlane[8];\n"
        "uniform float _mglClipPlaneEnabled[8];\n"
        "void main() {\n"
        "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
        "    gl_ClipDistance[0] = mix(1.0, dot(_mglClipPlane[0], vec4(0.0,0.0,0.5,1.0)), _mglClipPlaneEnabled[0]);\n"
        "    gl_ClipDistance[1] = mix(1.0, dot(_mglClipPlane[1], vec4(0.0,0.0,0.5,1.0)), _mglClipPlaneEnabled[1]);\n"
        "    gl_ClipDistance[2] = mix(1.0, dot(_mglClipPlane[2], vec4(0.0,0.0,0.5,1.0)), _mglClipPlaneEnabled[2]);\n"
        "    gl_ClipDistance[3] = mix(1.0, dot(_mglClipPlane[3], vec4(0.0,0.0,0.5,1.0)), _mglClipPlaneEnabled[3]);\n"
        "    gl_ClipDistance[4] = mix(1.0, dot(_mglClipPlane[4], vec4(0.0,0.0,0.5,1.0)), _mglClipPlaneEnabled[4]);\n"
        "    gl_ClipDistance[5] = mix(1.0, dot(_mglClipPlane[5], vec4(0.0,0.0,0.5,1.0)), _mglClipPlaneEnabled[5]);\n"
        "    gl_ClipDistance[6] = mix(1.0, dot(_mglClipPlane[6], vec4(0.0,0.0,0.5,1.0)), _mglClipPlaneEnabled[6]);\n"
        "    gl_ClipDistance[7] = mix(1.0, dot(_mglClipPlane[7], vec4(0.0,0.0,0.5,1.0)), _mglClipPlaneEnabled[7]);\n"
        "}\n";
    /* Solid red: clipped fragments leave the clear color. */
    const char *fs330 =
        "#version 330 core\n"
        "out vec4 f;\n"
        "void main() { f = vec4(1.0, 0.0, 0.0, 1.0); }\n";

    GLuint prog = link_program(vs330, fs330);
    if (!prog) {
        fprintf(stderr, "legacy_clip_vertex: link failed\n");
        return 1;
    }

    GLuint fbo = 0, tex = 0;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 2;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glViewport(0, 0, REG_W, REG_H);
    glDisable(GL_SCISSOR_TEST);
    glUseProgram(prog);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    const size_t mid = ((size_t)(REG_H / 2) * (size_t)REG_W + (size_t)(REG_W / 2)) * 4u;

    /* Draw 1: default state, no planes enabled — red everywhere. */
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (pixels[mid + 0] < 200u || pixels[mid + 1] >= 50u || pixels[mid + 2] >= 50u) {
        fprintf(stderr, "legacy_clip_vertex: draw1 default expected red, got "
                        "(%u,%u,%u,%u)\n",
                (unsigned)pixels[mid + 0], (unsigned)pixels[mid + 1],
                (unsigned)pixels[mid + 2], (unsigned)pixels[mid + 3]);
        return 3;
    }

    /* Draw 2: clip plane 0 (0,0,-1,0) enabled — eye-space clip vertex
     * (0,0,0.5,1) gives distance -0.5 < 0, the fragment is clipped and the
     * pixel stays black. */
    {
        const GLdouble eq[4] = { 0.0, 0.0, -1.0, 0.0 };
        glClipPlane(GL_CLIP_PLANE0, eq);
    }
    glEnable(GL_CLIP_PLANE0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (pixels[mid + 0] >= 50u || pixels[mid + 1] >= 50u || pixels[mid + 2] >= 50u) {
        fprintf(stderr, "legacy_clip_vertex: draw2 plane0=(0,0,-1,0) should "
                        "clip, got (%u,%u,%u,%u)\n",
                (unsigned)pixels[mid + 0], (unsigned)pixels[mid + 1],
                (unsigned)pixels[mid + 2], (unsigned)pixels[mid + 3]);
        return 4;
    }

    /* Draw 3: plane 0 disabled again — red restored. */
    glDisable(GL_CLIP_PLANE0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (pixels[mid + 0] < 200u || pixels[mid + 1] >= 50u || pixels[mid + 2] >= 50u) {
        fprintf(stderr, "legacy_clip_vertex: draw3 after disable expected red, "
                        "got (%u,%u,%u,%u)\n",
                (unsigned)pixels[mid + 0], (unsigned)pixels[mid + 1],
                (unsigned)pixels[mid + 2], (unsigned)pixels[mid + 3]);
        return 5;
    }

    /* Draw 4: plane 0 flipped to (0,0,1,0) — distance +0.5, no clip. */
    {
        const GLdouble eq[4] = { 0.0, 0.0, 1.0, 0.0 };
        glClipPlane(GL_CLIP_PLANE0, eq);
    }
    glEnable(GL_CLIP_PLANE0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (pixels[mid + 0] < 200u || pixels[mid + 1] >= 50u || pixels[mid + 2] >= 50u) {
        fprintf(stderr, "legacy_clip_vertex: draw4 positive plane should not "
                        "clip, got (%u,%u,%u,%u)\n",
                (unsigned)pixels[mid + 0], (unsigned)pixels[mid + 1],
                (unsigned)pixels[mid + 2], (unsigned)pixels[mid + 3]);
        return 6;
    }

    /* Draw 5: plane 5 (0,0,-1,0) enabled — clips via gl_ClipDistance[5]. */
    {
        const GLdouble eq[4] = { 0.0, 0.0, -1.0, 0.0 };
        glClipPlane(GL_CLIP_PLANE5, eq);
    }
    glEnable(GL_CLIP_PLANE5);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (pixels[mid + 0] >= 50u || pixels[mid + 1] >= 50u || pixels[mid + 2] >= 50u) {
        fprintf(stderr, "legacy_clip_vertex: draw5 plane5=(0,0,-1,0) should "
                        "clip, got (%u,%u,%u,%u)\n",
                (unsigned)pixels[mid + 0], (unsigned)pixels[mid + 1],
                (unsigned)pixels[mid + 2], (unsigned)pixels[mid + 3]);
        return 7;
    }

    /* Stray error: an out-of-range plane index must raise GL_INVALID_ENUM
     * and leave state untouched (plane 5 stays enabled and still clips). */
    {
        const GLdouble eq[4] = { 0.0, 0.0, -1.0, 0.0 };
        glClipPlane(0x3000 + 100, eq);
        if (glGetError() != GL_INVALID_ENUM) {
            fprintf(stderr, "legacy_clip_vertex: out-of-range glClipPlane "
                            "not rejected\n");
            return 8;
        }
    }

    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    return 0;
}

static int test_legacy_glsl_frontend(unsigned char *pixels, const char *out_path)
{
    (void)out_path;
    /* GLSL 1.10 vertex shader: attribute/varying style. */
    const char *vs110 =
        "#version 110\n"
        "attribute vec2 a_pos;\n"
        "varying vec2 v_uv;\n"
        "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); v_uv = a_pos; }\n";
    /* GLSL 1.10 fragment shader: gl_FragColor output. */
    const char *fs110 =
        "#version 110\n"
        "varying vec2 v_uv;\n"
        "void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }\n";

    GLuint progA = link_program(vs110, fs110);
    if (!progA) {
        fprintf(stderr, "legacy_glsl_frontend: link failed (segment A)\n");
        return 1;
    }
    /* GLSL 1.10 fragment shader using the legacy texture2D() builtin. */
    const char *fs110tex =
        "#version 110\n"
        "uniform sampler2D tex;\n"
        "varying vec2 v_uv;\n"
        "void main() { gl_FragColor = texture2D(tex, v_uv); }\n";
    GLuint progB = link_program(vs110, fs110tex);
    if (!progB) {
        fprintf(stderr, "legacy_glsl_frontend: link failed (segment B)\n");
        return 2;
    }

    GLuint fbo = 0, tex = 0;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) return 3;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = make_vbo(TRI_VERTS, sizeof(TRI_VERTS));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glViewport(0, 0, REG_W, REG_H);
    glDisable(GL_SCISSOR_TEST);

    /* Segment A: legacy gl_FragColor path. */
    clear_color(0.0f, 0.0f, 0.0f);
    glUseProgram(progA);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const int px[3][2] = {
            { REG_W/2, REG_H/2 }, { REG_W/2 - 8, REG_H/2 },
            { REG_W/2, REG_H/2 - 8 },
        };
        for (int i = 0; i < 3; i++) {
            const unsigned char *c =
                &pixels[(px[i][1] * REG_W + px[i][0]) * 4];
            if (c[0] < 200u || c[1] > 60u || c[2] > 60u) {
                fprintf(stderr,
                        "legacy_glsl_frontend: seg A probe %d not red at "
                        "(%d,%d) rgb=(%u,%u,%u)\n",
                        i, px[i][0], px[i][1], c[0], c[1], c[2]);
                return 4;
            }
        }
        const unsigned char *e =
            &pixels[((3 * REG_H / 4) * REG_W + REG_W / 4) * 4];
        if (e[0] > 20u || e[1] > 20u || e[2] > 20u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg A exterior not black "
                    "rgb=(%u,%u,%u)\n", e[0], e[1], e[2]);
            return 5;
        }
    }

    /* Segment B: legacy texture2D() path with a red 1x1 texture. */
    GLuint texUnit = 0, redTex = 0;
    glGenTextures(1, &redTex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, redTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    const unsigned char redPx[4] = { 255, 0, 0, 255 };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, redPx);
    glUseProgram(progB);
    glUniform1i(glGetUniformLocation(progB, "tex"), 0);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *c =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (c[0] < 200u || c[1] > 60u || c[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg B probe not red rgb=(%u,%u,%u)\n",
                    c[0], c[1], c[2]);
            return 6;
        }
    }
    /* Segment C: classic fixed-function-style GLSL 1.10 VS using the implicit
     * gl_Vertex attribute + built-in matrix uniforms.  The translator injects
     * the matrices with their ORIGINAL gl_ names, so the GL-side uniform
     * contract is unchanged: resolve "gl_ModelViewProjectionMatrix" directly
     * and set an identity matrix -> same red triangle. */
    {
        const char *vs110m =
            "#version 110\n"
            "attribute vec2 a_pos;\n"
            "void main() {\n"
            "    gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos, 0.0, 1.0);\n"
            "    gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        const char *fs110m =
            "#version 110\n"
            "void main() { gl_FragColor = gl_FrontColor; }\n";
        GLuint progC = link_program(vs110m, fs110m);
        if (!progC) {
            fprintf(stderr,
                    "legacy_glsl_frontend: link failed (segment C)\n");
            return 7;
        }
        GLint mvpLoc = glGetUniformLocation(progC, "gl_ModelViewProjectionMatrix");
        if (mvpLoc < 0) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg C uniform "
                    "gl_ModelViewProjectionMatrix not found\n");
            return 8;
        }
        const float identity[16] = {
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
        };
        glUseProgram(progC);
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, identity);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *c =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (c[0] < 200u || c[1] > 60u || c[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg C probe not red "
                    "rgb=(%u,%u,%u)\n", c[0], c[1], c[2]);
            return 9;
        }
    }

    /* Segment D: gl_Vertex end-to-end.  The translator injects
     * 'layout(location = 0) in vec4 gl_Vertex;' and the reflector now admits
     * explicitly-located gl_-prefixed stage inputs, so the implicit legacy
     * position attribute is bindable at the conventional slot 0: the harness
     * VAO's attrib-0 stream (TRI_VERTS) feeds gl_Vertex directly. */
    {
        const char *vs110v =
            "#version 110\n"
            "void main() {\n"
            "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
            "    gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        const char *fs110v =
            "#version 110\n"
            "void main() { gl_FragColor = gl_FrontColor; }\n";
        GLuint progD = link_program(vs110v, fs110v);
        if (!progD) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment D)\n");
            return 10;
        }
        GLint vLoc = glGetAttribLocation(progD, "gl_Vertex");
        if (vLoc != 0) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg D gl_Vertex location=%d "
                    "(expected 0)\n", (int)vLoc);
            return 11;
        }
        GLint mvpLocD = glGetUniformLocation(progD, "gl_ModelViewProjectionMatrix");
        if (mvpLocD < 0) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg D MVP uniform not found\n");
            return 12;
        }
        const float identityD[16] = {
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
        };
        glUseProgram(progD);
        glUniformMatrix4fv(mvpLocD, 1, GL_FALSE, identityD);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *d =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (d[0] < 200u || d[1] > 60u || d[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg D probe not red "
                    "rgb=(%u,%u,%u)\n", d[0], d[1], d[2]);
            return 13;
        }
    }

    /* Segment E: ftransform() end-to-end.  The translator expands
     * ftransform() to gl_ModelViewProjectionMatrix * gl_Vertex and the
     * source-guarded injection declares both, so a shader that only ever
     * calls ftransform() renders through the fixed-function transform. */
    {
        const char *vs110f =
            "#version 110\n"
            "void main() {\n"
            "    gl_Position = ftransform();\n"
            "    gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        const char *fs110f =
            "#version 110\n"
            "void main() { gl_FragColor = gl_FrontColor; }\n";
        GLuint progE = link_program(vs110f, fs110f);
        if (!progE) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment E)\n");
            return 14;
        }
        GLint vLocE = glGetAttribLocation(progE, "gl_Vertex");
        if (vLocE != 0) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg E gl_Vertex location=%d "
                    "(expected 0)\n", (int)vLocE);
            return 15;
        }
        GLint mvpLocE = glGetUniformLocation(progE, "gl_ModelViewProjectionMatrix");
        if (mvpLocE < 0) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg E MVP uniform not found\n");
            return 16;
        }
        const float identityE[16] = {
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
        };
        glUseProgram(progE);
        glUniformMatrix4fv(mvpLocE, 1, GL_FALSE, identityE);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *e =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (e[0] < 200u || e[1] > 60u || e[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg E probe not red "
                    "rgb=(%u,%u,%u)\n", e[0], e[1], e[2]);
            return 17;
        }
    }

    /* Segment F: classic legacy texture flow.  VS feeds the builtin varying
     * gl_TexCoord[0] from the implicit gl_MultiTexCoord0 attribute; the FS
     * samples texture2D() with gl_TexCoord[0].xy.  Exercises the
     * gl_MultiTexCoord0/gl_TexCoord renames + declarations in both stages
     * and the texture2D -> texture rewrite, end to end. */
    {
        const char *vs110t =
            "#version 110\n"
            "void main() {\n"
            "    gl_TexCoord[0] = gl_MultiTexCoord0;\n"
            "    gl_Position = ftransform();\n"
            "}\n";
        const char *fs110t =
            "#version 110\n"
            "uniform sampler2D u_tex;\n"
            "void main() {\n"
            "    gl_FragColor = texture2D(u_tex, gl_TexCoord[0].xy);\n"
            "}\n";
        GLuint progF = link_program(vs110t, fs110t);
        if (!progF) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment F)\n");
            return 18;
        }
        GLint texLocF = glGetUniformLocation(progF, "u_tex");
        /* Bind the red 1x1 texture; texcoords come from the legacy
         * fixed-function slot 8 (gl_MultiTexCoord0) — a tiny UV stream. */
        static const float uvsF[6] = {
            0.0f, 0.0f,  1.0f, 0.0f,  0.0f, 1.0f,
        };
        GLuint uvVBO = 0;
        glGenBuffers(1, &uvVBO);
        glBindBuffer(GL_ARRAY_BUFFER, uvVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uvsF), uvsF, GL_STATIC_DRAW);
        glEnableVertexAttribArray(8);
        glVertexAttribPointer(8, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, redTex);
        glUseProgram(progF);
        if (texLocF >= 0) glUniform1i(texLocF, 0);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *f =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (f[0] < 200u || f[1] > 60u || f[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg F probe not red "
                    "rgb=(%u,%u,%u)\n", f[0], f[1], f[2]);
            return 20;
        }
    }

    /* Segment G: legacy color chain.  The VS passes the gl_Color
     * attribute (fixed-function slot 3) through gl_FrontColor; the FS
     * colors directly from gl_FrontColor.  Exercises the gl_Color /
     * gl_FrontColor renames + declarations (VS out / FS in) and the
     * fixed-function attribute slot 3 end to end, including the
     * glGetAttribLocation("gl_Color") == 3 contract. */
    {
        const char *vs110c =
            "#version 110\n"
            "void main() {\n"
            "    gl_FrontColor = gl_Color;\n"
            "    gl_Position = ftransform();\n"
            "}\n";
        const char *fs110c =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragColor = gl_FrontColor;\n"
            "}\n";
        GLuint progG = link_program(vs110c, fs110c);
        if (!progG) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment G)\n");
            return 21;
        }
        GLint colorLoc = glGetAttribLocation(progG, "gl_Color");
        if (colorLoc != 3) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg G gl_Color at %d, want 3\n",
                    colorLoc);
            return 22;
        }
        /* Red color stream at fixed-function slot 3. */
        static const float colorsG[12] = {
            1.0f, 0.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f, 1.0f,
        };
        GLuint colorVBO = 0;
        glGenBuffers(1, &colorVBO);
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(colorsG), colorsG,
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, NULL);
        glUseProgram(progG);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *g =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (g[0] < 200u || g[1] > 60u || g[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg G probe not red "
                    "rgb=(%u,%u,%u)\n", g[0], g[1], g[2]);
            return 23;
        }
    }

    /* Segment H: legacy fog-coordinate chain.  The VS passes the gl_FogCoord
     * float attribute (fixed-function slot 5) through the gl_FogFragCoord
     * varying; the FS colors from gl_FogFragCoord.  Exercises the float
     * attribute / float varying paths and the slot-5 bound stream. */
    {
        const char *vs110f =
            "#version 110\n"
            "void main() {\n"
            "    gl_FogFragCoord = gl_FogCoord;\n"
            "    gl_Position = ftransform();\n"
            "}\n";
        const char *fs110f =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragColor = vec4(gl_FogFragCoord, 0.0, 0.0, 1.0);\n"
            "}\n";
        GLuint progH = link_program(vs110f, fs110f);
        if (!progH) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment H)\n");
            return 24;
        }
        GLint fogLoc = glGetAttribLocation(progH, "gl_FogCoord");
        if (fogLoc != 5) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg H gl_FogCoord at %d, want 5\n",
                    fogLoc);
            return 25;
        }
        /* Constant fog coord 1.0 (vec3 stride to exercise float layout). */
        static const float fogVals[3] = { 1.0f, 1.0f, 1.0f };
        GLuint fogVBO = 0;
        glGenBuffers(1, &fogVBO);
        glBindBuffer(GL_ARRAY_BUFFER, fogVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(fogVals), fogVals,
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 0, NULL);
        glUseProgram(progH);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *h =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (h[0] < 200u || h[1] > 60u || h[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg H probe not red "
                    "rgb=(%u,%u,%u)\n", h[0], h[1], h[2]);
            return 26;
        }
    }

    /* Segment J: gl_FragData[0]-only fragment shader.  The common legacy
     * single-buffer pattern must map to the scalar color output (the
     * translator rewrites index-0 writes to _mglFragColor) and render to
     * the single color attachment. */
    {
        const char *vs110d =
            "#version 110\n"
            "void main() {\n"
            "    gl_Position = ftransform();\n"
            "}\n";
        const char *fs110d =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragData[0] = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        GLuint progJ = link_program(vs110d, fs110d);
        if (!progJ) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment J)\n");
            return 27;
        }
        glUseProgram(progJ);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *j =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (j[0] < 200u || j[1] > 60u || j[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg J probe not red "
                    "rgb=(%u,%u,%u)\n", j[0], j[1], j[2]);
            return 28;
        }
    }

    /* Segment K: gl_FrontFacing (front/back flag).  The FS colors by the
     * facing flag; glFrontFace flips the convention so BOTH branches are
     * verified: CCW (front) -> red, CW (back) -> blue. */
    {
        const char *vs110ff =
            "#version 110\n"
            "void main() {\n"
            "    gl_Position = ftransform();\n"
            "}\n";
        const char *fs110ff =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragColor = gl_FrontFacing\n"
            "        ? vec4(1.0, 0.0, 0.0, 1.0)\n"
            "        : vec4(0.0, 0.0, 1.0, 1.0);\n"
            "}\n";
        GLuint progK = link_program(vs110ff, fs110ff);
        if (!progK) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment K)\n");
            return 29;
        }
        glUseProgram(progK);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *kf =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (kf[0] < 200u || kf[1] > 60u || kf[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg K front probe not red "
                    "rgb=(%u,%u,%u)\n", kf[0], kf[1], kf[2]);
            return 30;
        }
        /* Back-facing: same triangle, CW convention. */
        clear_color(0.0f, 0.0f, 0.0f);
        glFrontFace(GL_CW);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glFrontFace(GL_CCW);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *kb =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (kb[2] < 200u || kb[0] > 60u || kb[1] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg K back probe not blue "
                    "rgb=(%u,%u,%u)\n", kb[0], kb[1], kb[2]);
            return 31;
        }
    }

    /* Segment L: point sprite.  The VS writes gl_PointSize and positions a
     * point at the center; the FS colors by gl_PointCoord (the
     * point_coord fragment argument).  At the point center gl_PointCoord.x
     * is 0.5, so the probe must read red. */
    {
        const char *vs110ps =
            "#version 110\n"
            "void main() {\n"
            "    gl_PointSize = 96.0;\n"
            "    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        const char *fs110ps =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragColor = vec4(gl_PointCoord.x < 2.0 ? 1.0 : 0.0,\n"
            "                        0.0, 0.0, 1.0);\n"
            "}\n";
        GLuint progL = link_program(vs110ps, fs110ps);
        if (!progL) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment L)\n");
            return 32;
        }
        glUseProgram(progL);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_POINTS, 0, 1);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *l =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (l[0] < 200u || l[1] > 60u || l[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg L point probe not red "
                    "rgb=(%u,%u,%u)\n", l[0], l[1], l[2]);
            return 33;
        }
    }

    /* Segment M: gl_FragDepth.  Pass 1 writes depth 0.25 (red); pass 2
     * writes depth 0.75 (blue) with GL_DEPTH_TEST + GL_LEQUAL over the
     * same triangle.  If gl_FragDepth flows into the depth attachment,
     * pass 2 fails the test (0.75 > 0.25) and the probe stays red.  If
     * the depth write were ignored, both passes write the interpolated
     * depth (0.5) and pass 2 passes LEQUAL, turning the probe blue.
     * Uses the depth-TEXTURE FBO (make_fbo_depth_tex): the renderer's
     * depth path is texture-based (make_fbo's depth renderbuffer is not
     * attached; verified by the pure-z control below which fails there
     * and passes here). */
    {
        GLuint mfbo, mtex, mdtex;
        mfbo = make_fbo_depth_tex(REG_W, REG_H, &mtex, &mdtex);
        if (!mfbo) {
            fprintf(stderr, "legacy_glsl_frontend: seg M fbo failed\n");
            return 34;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, mfbo);
        const char *vs110d =
            "#version 110\n"
            "void main() { gl_Position = ftransform(); }\n";
        const char *fs110d1d =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "    gl_FragDepth = 0.25;\n"
            "}\n";
        const char *fs110d2d =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);\n"
            "    gl_FragDepth = 0.75;\n"
            "}\n";
        /* CONTROL: pure z positions (no gl_FragDepth) with this FBO.
         * Pass 1 z=-0.5 (depth 0.25), pass 2 z=+0.5 (depth 0.75). */
        const char *vs110dzc =
            "#version 110\n"
            "attribute vec2 a_pos;\n"
            "void main() { gl_Position = vec4(a_pos, -0.5, 1.0); }\n";
        const char *vs110dzc2 =
            "#version 110\n"
            "attribute vec2 a_pos;\n"
            "void main() { gl_Position = vec4(a_pos, 0.5, 1.0); }\n";
        const char *fs110dc =
            "#version 110\n"
            "void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }\n";
        const char *fs110dc2 =
            "#version 110\n"
            "void main() { gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0); }\n";
        GLuint progC1 = link_program(vs110dzc, fs110dc);
        GLuint progC2 = link_program(vs110dzc2, fs110dc2);
        if (!progC1 || !progC2) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (seg M control)\n");
            return 34;
        }
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glUseProgram(progC1);
        clear_color(0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glUseProgram(progC2);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *mc =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (mc[0] < 200u || mc[1] > 60u || mc[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg M z-control not red "
                    "rgb=(%u,%u,%u) — plumbing broken\n", mc[0], mc[1], mc[2]);
            return 35;
        }
        GLuint progM1 = link_program(vs110d, fs110d1d);
        GLuint progM2 = link_program(vs110d, fs110d2d);
        if (!progM1 || !progM2) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment M)\n");
            return 34;
        }
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glUseProgram(progM1);
        clear_color(0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glUseProgram(progM2);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *m =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (m[0] < 200u || m[1] > 60u || m[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg M depth-write probe not red "
                    "rgb=(%u,%u,%u)\n", m[0], m[1], m[2]);
            return 35;
        }
    }

    /* Segment N: two-sided lighting.  The VS emits gl_FrontColor (red)
     * and gl_BackColor (blue); the FS selects between them with
     * gl_FrontFacing (classic GLSL 1.10 pattern).  CCW (front) -> red;
     * glFrontFace(GL_CW) flips the convention -> blue. */
    {
        const char *vs110ts =
            "#version 110\n"
            "void main() {\n"
            "    gl_Position = ftransform();\n"
            "    gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "    gl_BackColor = vec4(0.0, 0.0, 1.0, 1.0);\n"
            "}\n";
        const char *fs110ts =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragColor = gl_FrontFacing ? gl_Color : gl_BackColor;\n"
            "}\n";
        GLuint progN = link_program(vs110ts, fs110ts);
        if (!progN) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment N)\n");
            return 36;
        }
        glUseProgram(progN);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *nf =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (nf[0] < 200u || nf[1] > 60u || nf[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg N front probe not red "
                    "rgb=(%u,%u,%u)\n", nf[0], nf[1], nf[2]);
            return 37;
        }
        clear_color(0.0f, 0.0f, 0.0f);
        glFrontFace(GL_CW);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glFrontFace(GL_CCW);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *nb =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        if (nb[2] < 200u || nb[0] > 60u || nb[1] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg N back probe not blue "
                    "rgb=(%u,%u,%u)\n", nb[0], nb[1], nb[2]);
            return 38;
        }
    }

    /* Segment O: per-fragment primitive/sample builtins.  A single
     * 6-vertex draw (two triangles side by side) gives gl_PrimitiveID
     * 0 and 1; with the default 1-sample target gl_SampleID is 0 and
     * gl_SamplePosition is (0.5, 0.5).  Left probe (pid 0) must read
     * red, right probe (pid 1) blue. */
    {
        static const float two_tri_verts[12] = {
            -0.6f, -0.6f,   0.0f, -0.6f,   -0.3f,  0.6f,
             0.0f, -0.6f,   0.6f, -0.6f,    0.3f,  0.6f,
        };
        GLuint vboO = make_vbo(two_tri_verts, sizeof two_tri_verts);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vboO);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        const char *vs110o =
            "#version 110\n"
            "attribute vec2 a_pos;\n"
            "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
        const char *fs110o =
            "#version 110\n"
            "void main() {\n"
            "    bool ok = gl_PrimitiveID == 0 && gl_SampleID == 0;\n"
            "    gl_FragColor = ok ? vec4(1.0, 0.0, 0.0, 1.0)\n"
            "                      : vec4(0.0, 0.0, 1.0, 1.0);\n"
            "}\n";
        GLuint progO = link_program(vs110o, fs110o);
        if (!progO) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment O)\n");
            return 39;
        }
        glUseProgram(progO);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *ol =
            &pixels[((REG_H/2) * REG_W + 44) * 4];
        const unsigned char *or_ =
            &pixels[((REG_H/2) * REG_W + 84) * 4];
        if (ol[0] < 200u || ol[1] > 60u || ol[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg O pid0 probe not red "
                    "rgb=(%u,%u,%u)\n", ol[0], ol[1], ol[2]);
            return 40;
        }
        if (or_[2] < 200u || or_[0] > 60u || or_[1] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg O pid1 probe not blue "
                    "rgb=(%u,%u,%u)\n", or_[0], or_[1], or_[2]);
            return 41;
        }
        /* Restore the harness triangle for any later segments. */
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    }

    /* Segment P: gl_ClipDistance.  P1 clips where y > 0 (clip_distance[0]
     * = -a_pos.y; Metal clips fragments with a negative clip distance):
     * the probe below center stays red, the one above center is clipped
     * to the clear color.  P2 (control) writes a constant positive clip
     * distance — nothing clipped, both probes red. */
    {
        const char *vs110p =
            "#version 330 core\n"
            "layout(location = 0) in vec2 a_pos;\n"
            "void main() {\n"
            "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
            "    gl_ClipDistance[0] = -a_pos.y;\n"
            "}\n";
        const char *vs110p2 =
            "#version 330 core\n"
            "layout(location = 0) in vec2 a_pos;\n"
            "void main() {\n"
            "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
            "    gl_ClipDistance[0] = 1.0;\n"
            "}\n";
        const char *fs110p =
            "#version 330 core\n"
            "out vec4 f; void main() { f = vec4(1.0, 0.0, 0.0, 1.0); }\n";
        GLuint progP1 = link_program(vs110p, fs110p);
        GLuint progP2 = link_program(vs110p2, fs110p);
        if (!progP1 || !progP2) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment P)\n");
            return 42;
        }
        glUseProgram(progP1);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *pbot =
            &pixels[((REG_H/4) * REG_W + REG_W/2) * 4];
        const unsigned char *ptop =
            &pixels[((3*REG_H/4) * REG_W + REG_W/2) * 4];
        if (pbot[0] < 200u || pbot[1] > 60u || pbot[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg P clipped probe not red "
                    "rgb=(%u,%u,%u)\n", pbot[0], pbot[1], pbot[2]);
            return 43;
        }
        if (ptop[0] > 60u || ptop[1] > 60u || ptop[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg P top probe not clipped "
                    "rgb=(%u,%u,%u)\n", ptop[0], ptop[1], ptop[2]);
            return 44;
        }
        glUseProgram(progP2);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *p2b =
            &pixels[((REG_H/4) * REG_W + REG_W/2) * 4];
        const unsigned char *p2t =
            &pixels[((3*REG_H/4) * REG_W + REG_W/2) * 4];
        if (p2b[0] < 200u || p2t[0] < 200u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg P control not red both "
                    "rgb=(%u,%u,%u)/(%u,%u,%u)\n",
                    p2b[0], p2b[1], p2b[2], p2t[0], p2t[1], p2t[2]);
            return 45;
        }
    }

    /* Segment Q: gl_FragData MRT.  An FBO with two color attachments +
     * glDrawBuffers(0|1): gl_FragData[0] -> red on attachment 0,
     * gl_FragData[1] -> green on attachment 1.  Read each attachment back
     * via glReadBuffer. */
    {
        GLuint qfbo, qtex0, qtex1, qrbo;
        glGenFramebuffers(1, &qfbo);
        glBindFramebuffer(GL_FRAMEBUFFER, qfbo);
        glGenTextures(1, &qtex0);
        glBindTexture(GL_TEXTURE_2D, qtex0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, REG_W, REG_H, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, qtex0, 0);
        glGenTextures(1, &qtex1);
        glBindTexture(GL_TEXTURE_2D, qtex1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, REG_W, REG_H, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                               GL_TEXTURE_2D, qtex1, 0);
        glGenRenderbuffers(1, &qrbo);
        glBindRenderbuffer(GL_RENDERBUFFER, qrbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                              REG_W, REG_H);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, qrbo);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "legacy_glsl_frontend: seg Q fbo incomplete\n");
            return 46;
        }
        const GLenum qbufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, qbufs);
        (void)qbufs;

        const char *vs110q =
            "#version 110\n"
            "void main() { gl_Position = ftransform(); }\n";
        const char *fs110q =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragData[0] = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "    gl_FragData[1] = vec4(0.0, 1.0, 0.0, 1.0);\n"
            "}\n";
        GLuint progQ = link_program(vs110q, fs110q);
        if (!progQ) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment Q)\n");
            return 47;
        }
        glUseProgram(progQ);
        clear_color(0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        unsigned char q0px[4], q1px[4];
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        memcpy(q0px, &pixels[((REG_H/2) * REG_W + REG_W/2) * 4], 4u);
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        memcpy(q1px, &pixels[((REG_H/2) * REG_W + REG_W/2) * 4], 4u);
        const unsigned char *q0 = q0px;
        const unsigned char *q1 = q1px;
        if (q0[0] < 200u || q0[1] > 60u || q0[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg Q attachment0 not red "
                    "rgb=(%u,%u,%u)\n", q0[0], q0[1], q0[2]);
            return 48;
        }
        if (q1[1] < 200u || q1[0] > 60u || q1[2] > 60u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg Q attachment1 not green "
                    "rgb=(%u,%u,%u)\n", q1[0], q1[1], q1[2]);
            return 49;
        }
        /* Restore the harness state for any later segments. */
        const GLenum oneBuf[1] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, oneBuf);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    }

    /* ---- Segment R: GLSL 1.10 built-in compile-time constants ---- */
    {
        const char *vs110r =
            "#version 110\n"
            "void main() { gl_Position = ftransform(); }\n";
        const char *fs110r =
            "#version 110\n"
            "void main() {\n"
            "    gl_FragColor = vec4(float(gl_MaxDrawBuffers) / 8.0,\n"
            "                        float(gl_MaxClipPlanes) / 8.0,\n"
            "                        float(gl_MaxTextureUnits) / 8.0, 1.0);\n"
            "}\n";
        GLuint progR = link_program(vs110r, fs110r);
        if (!progR) {
            fprintf(stderr, "legacy_glsl_frontend: link failed (segment R)\n");
            return 50;
        }
        glUseProgram(progR);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, REG_W, REG_H);
        clear_color(0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const unsigned char *rpx =
            &pixels[((REG_H/2) * REG_W + REG_W/2) * 4];
        /* (255, 191, 255) = (8/8, 6/8, 8/8) — the injected builtin
         * constants must fold to their literal values at runtime. */
        if (rpx[0] < 200u || rpx[1] < 150u || rpx[1] > 230u ||
            rpx[2] < 200u) {
            fprintf(stderr,
                    "legacy_glsl_frontend: seg R constants wrong "
                    "rgb=(%u,%u,%u)\n", rpx[0], rpx[1], rpx[2]);
            return 50;
        }
    }

    glDeleteTextures(1, &redTex);
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

static int test_air_geometry_varying(unsigned char *pixels,
                                     const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) out vec3 v_color;\n"
        "void main() {\n"
        "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "  v_color = vec3(0.0, 1.0, 0.0);\n"
        "}\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=6) out;\n"
        "layout(location=0) in vec3 v_color[];\n"
        "layout(location=0) out vec3 g_color;\n"
        "void main() {\n"
        "  g_color = v_color[0];\n"
        "  gl_CullDistance[0] = -1.0;\n"
        "  gl_Position = vec4(-0.9, -0.5, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.3, -0.5, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.6,  0.5, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "  gl_CullDistance[0] = 1.0;\n"
        "  gl_Position = vec4( 0.1, -0.5, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 0.7, -0.5, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 0.4,  0.5, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec3 g_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(g_color, 1.0); }\n";

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_geometry(vs, gs, fs);
    GLuint vao = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(program);
    glDrawArrays(GL_POINTS, 0, 1);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *culled =
            &pixels[((REG_H / 2) * REG_W + 26) * 4];
        const unsigned char *visible =
            &pixels[((REG_H / 2) * REG_W + 90) * 4];
        if (culled[0] > 20u || culled[1] > 20u || culled[2] > 20u ||
            visible[0] > 20u || visible[1] < 220u || visible[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_varying: expected culled black/visible "
                    "green, got (%u,%u,%u)/(%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visible[0], visible[1], visible[2]);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* GS writes gl_CullDistance per emitted vertex.  Primitive culling rule:
 * a primitive is discarded only when EVERY vertex's cull distance for the
 * same index is negative; any non-negative vertex keeps the primitive.
 * Covers the array path (glDrawArrays(GL_POINTS)) and the element path
 * (glDrawElements(GL_POINTS)) through the GS cull-distance capture in the
 * batch replay (direct + deferred). */
static int test_air_geometry_cull_distance(unsigned char *pixels,
                                           const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=1) in float cdsel;\n"
        "layout(location=0) out float v_cdsel;\n"
        "void main() {\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  v_cdsel = cdsel;\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) in float v_cdsel[];\n"
        "void main() {\n"
        "  float sel = v_cdsel[0];\n"
        "  float cd0 = (sel == 0.0) ? -1.0 : 1.0;\n"
        "  float cd1 = (sel == 0.0) ? -1.0 : ((sel == 1.0) ? 1.0 : -1.0);\n"
        "  float cd2 = (sel == 0.0) ? -1.0 : 1.0;\n"
        "  gl_CullDistance[0] = cd0;\n"
        "  gl_Position = vec4(gl_in[0].gl_Position.x - 0.28, -0.55, 0.0, 1.0); EmitVertex();\n"
        "  gl_CullDistance[0] = cd1;\n"
        "  gl_Position = vec4(gl_in[0].gl_Position.x + 0.28, -0.55, 0.0, 1.0); EmitVertex();\n"
        "  gl_CullDistance[0] = cd2;\n"
        "  gl_Position = vec4(gl_in[0].gl_Position.x, 0.5, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_geometry(vs, gs, fs);
    GLuint vao = 0u, vbo = 0u, selbo = 0u, ebo = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    /* Three input points: x = -0.6 (sel 0 → all-negative → culled),
     * x = 0.0 (sel 1 → all-positive → visible), x = 0.6 (sel 2 →
     * mixed +1/-1/+1 → visible because not ALL vertices are negative). */
    static const float positions[] = {
        -0.6f, 0.0f,
         0.0f, 0.0f,
         0.6f, 0.0f,
    };
    static const float sels[] = {0.0f, 1.0f, 2.0f};
    static const unsigned short elements[] = {0u, 1u, 2u};

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glGenBuffers(1, &selbo);
    glBindBuffer(GL_ARRAY_BUFFER, selbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sels), sels, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, selbo);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(program);

    glDrawArrays(GL_POINTS, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_geometry_cull_distance: arrays draw failed\n");
        goto cleanup;
    }
    {
const unsigned char *culled =
            &pixels[(51 * REG_W + 26) * 4];
        const unsigned char *visA =
            &pixels[(51 * REG_W + 64) * 4];
        const unsigned char *visB =
            &pixels[(51 * REG_W + 102) * 4];
        if (culled[0] > 20u || culled[1] > 20u || culled[2] > 20u ||
            visA[0] > 20u || visA[1] < 220u || visA[2] > 20u ||
            visB[0] > 20u || visB[1] < 220u || visB[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_cull_distance: arrays expected culled "
                    "black/visible green/visible green, got (%u,%u,%u)/"
                    "(%u,%u,%u)/(%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visA[0], visA[1], visA[2],
                    visB[0], visB[1], visB[2]);
            goto cleanup;
        }
    }

    /* Element path: same 3 points through an index buffer. */
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements,
                 GL_STATIC_DRAW);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawElements(GL_POINTS, 3, GL_UNSIGNED_SHORT, 0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_geometry_cull_distance: elements draw failed\n");
        goto cleanup;
    }
    {
const unsigned char *culled =
            &pixels[(51 * REG_W + 26) * 4];
        const unsigned char *visA =
            &pixels[(51 * REG_W + 64) * 4];
        const unsigned char *visB =
            &pixels[(51 * REG_W + 102) * 4];
        if (culled[0] > 20u || culled[1] > 20u || culled[2] > 20u ||
            visA[0] > 20u || visA[1] < 220u || visA[2] > 20u ||
            visB[0] > 20u || visB[1] < 220u || visB[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_cull_distance: elements expected culled "
                    "black/visible green/visible green, got (%u,%u,%u)/"
                    "(%u,%u,%u)/(%u,%u,%u)\n",
                    culled[0], culled[1], culled[2],
                    visA[0], visA[1], visA[2],
                    visB[0], visB[1], visB[2]);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (ebo) glDeleteBuffers(1, &ebo);
    if (selbo) glDeleteBuffers(1, &selbo);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_geometry_resources(unsigned char *pixels,
                                       const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 460 core\n"
        "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) out vec3 g_color;\n"
        "layout(std140, binding=0) uniform Params { vec4 tint; };\n"
        "layout(std430, binding=1) buffer Data { uint count; vec4 colors[]; } dataBuffer;\n"
        "layout(binding=0) uniform sampler2D sampleTex;\n"
        "layout(rgba8, binding=1) uniform image2D outputImage;\n"
        "void main() {\n"
        "  dataBuffer.count = uint(dataBuffer.colors.length());\n"
        "  imageStore(outputImage, ivec2(0), vec4(0.0, 1.0, 0.0, 1.0));\n"
        "  vec4 storageColor = dataBuffer.colors[0];\n"
        "  g_color = tint.rgb + storageColor.rgb + "
        "texture(sampleTex, vec2(0.5)).rgb;\n"
        "  gl_Position = vec4(-0.7, -0.6, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 0.7, -0.6, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 0.0,  0.7, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location=0) in vec3 g_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(g_color, 1.0); }\n";

    GLuint color = 0u, fbo = 0u, program = 0u, vao = 0u;
    GLuint ubo = 0u, ssbo = 0u, sampled = 0u, image = 0u;
    int result = 1;
    const float tint[4] = {0.0f, 0.25f, 0.0f, 0.0f};
    struct {
        uint32_t count;
        uint32_t padding[3];
        float color[4];
    } storage = {0u, {0u, 0u, 0u}, {0.0f, 0.25f, 0.0f, 0.0f}};
    const unsigned char sampledPixel[4] = {0u, 64u, 0u, 255u};
    unsigned char imagePixel[4] = {0u, 0u, 0u, 0u};

    fbo = make_fbo(REG_W, REG_H, &color);
    program = link_program_with_geometry(vs, gs, fs);
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(tint), tint, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(storage), &storage,
                 GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo);

    glGenTextures(1, &sampled);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sampled);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, sampledPixel);

    glGenTextures(1, &image);
    /* Keep the storage-image texture on a different sampler unit.  Image
     * bindings are independent from sampler bindings, but glBindTexture still
     * updates the currently active sampler unit; leaving GL_TEXTURE0 active
     * would replace the sampled texture used by the GS. */
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, imagePixel);
    glBindImageTexture(1, image, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
    glActiveTexture(GL_TEXTURE0);

    glUseProgram(program);
    GLint samplerLocation = glGetUniformLocation(program, "sampleTex");
    if (samplerLocation >= 0) glUniform1i(samplerLocation, 0);
    glDrawArrays(GL_POINTS, 0, 1);
    glFinish();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT);
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(storage), &storage);
    glBindTexture(GL_TEXTURE_2D, image);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagePixel);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_geometry_resources: GL operation failed\n");
        goto cleanup;
    }
    {
        const unsigned char *center =
            &pixels[((REG_H / 2) * REG_W + REG_W / 2) * 4];
        if (center[0] > 20u || center[1] < 180u || center[2] > 20u ||
            storage.count != 1u || imagePixel[0] > 20u ||
            imagePixel[1] < 220u || imagePixel[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_resources: expected green + SSBO length 1 "
                    "+ green image, got pixel=(%u,%u,%u) count=%u "
                    "image=(%u,%u,%u,%u)\n",
                    center[0], center[1], center[2], storage.count,
                    imagePixel[0], imagePixel[1], imagePixel[2], imagePixel[3]);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
    if (image) glDeleteTextures(1, &image);
    if (sampled) glDeleteTextures(1, &sampled);
    if (ssbo) glDeleteBuffers(1, &ssbo);
    if (ubo) glDeleteBuffers(1, &ubo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_geometry_ssbo_visibility(unsigned char *pixels,
                                             const char *out_path)
{
    /* GPU->GPU write visibility through glMemoryBarrier:
     * segment 1 = GS writes an SSBO, a LATER draw's GS reads it back
     * (shader-storage barrier); segment 2 = GS imageStore into a texture, a
     * later draw's GS samples it (texture-fetch barrier).  The reader emits
     * the barriered value at a DIFFERENT position than the writer, so a
     * stale/unordered read shows up as a wrong probe color. */
    (void)out_path;
    static const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
    static const char *gsWriter =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) out vec3 g_color;\n"
        "layout(std430, binding=0) buffer Data { vec4 color; } dataBuffer;\n"
        "layout(rgba8, binding=1) uniform image2D outputImage;\n"
        "void main() {\n"
        "  dataBuffer.color = vec4(0.0, 0.0, 1.0, 1.0);\n"
        "  imageStore(outputImage, ivec2(0), vec4(1.0, 0.0, 0.0, 1.0));\n"
        "  g_color = vec3(0.0, 1.0, 0.0);\n"
        "  gl_Position = vec4(-0.90, -0.55, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.30, -0.55, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.60,  0.50, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *gsReaderSSBO =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) out vec3 g_color;\n"
        "layout(std430, binding=0) buffer Data { vec4 color; } dataBuffer;\n"
        "void main() {\n"
        "  vec4 c = dataBuffer.color;\n"
        "  g_color = c.rgb;\n"
        "  gl_Position = vec4(0.30, -0.55, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(0.90, -0.55, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(0.60,  0.50, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *gsReaderTex =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) out vec3 g_color;\n"
        "layout(binding=0) uniform sampler2D resultTex;\n"
        "void main() {\n"
        "  g_color = texture(resultTex, vec2(0.5)).rgb;\n"
        "  gl_Position = vec4(0.30, -0.55, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(0.90, -0.55, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(0.60,  0.50, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location=0) in vec3 g_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(g_color, 1.0); }\n";

    GLuint color = 0u, fbo = 0u, vao = 0u, vbo = 0u;
    GLuint ssbo = 0u, image = 0u;
    GLuint writer = 0u, readerSSBO = 0u, readerTex = 0u;
    int result = 1;
    const float black[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float points[6] = {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    float ssboReadback[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    fbo = make_fbo(REG_W, REG_H, &color);
    writer = link_program_with_geometry(vs, gsWriter, fs);
    readerSSBO = link_program_with_geometry(vs, gsReaderSSBO, fs);
    readerTex = link_program_with_geometry(vs, gsReaderTex, fs);
    if (!fbo || !writer || !readerSSBO || !readerTex) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(black), black,
                 GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glGenTextures(1, &image);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, black);
    glBindImageTexture(1, image, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
    glActiveTexture(GL_TEXTURE0);

    /* Segment 1: GS SSBO write -> later GS read (same SSBO) across a
     * shader-storage barrier.  Writer emits green on the left, the value
     * carried in the SSBO is blue and must surface on the right. */
    clear_color(0.0f, 0.0f, 0.0f);
    glUseProgram(writer);
    glDrawArrays(GL_POINTS, 0, 3);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUseProgram(readerSSBO);
    glDrawArrays(GL_POINTS, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(ssboReadback),
                       ssboReadback);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_geometry_ssbo_visibility: seg1 GL failed\n");
        goto cleanup;
    }
    {
        const unsigned char *left = &pixels[(51 * REG_W + 25) * 4];
        const unsigned char *right = &pixels[(51 * REG_W + 102) * 4];
        if (left[0] > 20u || left[1] < 220u || left[2] > 20u ||
            right[0] > 20u || right[1] > 20u || right[2] < 220u ||
            ssboReadback[2] < 0.9f || ssboReadback[0] > 0.1f) {
            fprintf(stderr,
                    "air_geometry_ssbo_visibility: seg1 expected left green / "
                    "right blue (SSBO read), got left=(%u,%u,%u) "
                    "right=(%u,%u,%u) ssbo=(%.2f,%.2f,%.2f)\n",
                    left[0], left[1], left[2], right[0], right[1], right[2],
                    ssboReadback[0], ssboReadback[1], ssboReadback[2]);
            goto cleanup;
        }
    }

    /* Segment 2: GS imageStore -> later GS texture fetch across a
     * texture-fetch barrier.  The image (red) is sampled by the reader. */
    clear_color(0.0f, 0.0f, 0.0f);
    glUseProgram(writer);
    glDrawArrays(GL_POINTS, 0, 3);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, image);
    glUseProgram(readerTex);
    GLint texLocation = glGetUniformLocation(readerTex, "resultTex");
    if (texLocation >= 0) glUniform1i(texLocation, 0);
    glDrawArrays(GL_POINTS, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_geometry_ssbo_visibility: seg2 GL failed\n");
        goto cleanup;
    }
    {
        const unsigned char *left = &pixels[(51 * REG_W + 25) * 4];
        const unsigned char *right = &pixels[(51 * REG_W + 102) * 4];
        if (left[0] > 20u || left[1] < 220u || left[2] > 20u ||
            right[0] < 220u || right[1] > 20u || right[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_ssbo_visibility: seg2 expected left green / "
                    "right red (image fetch), got left=(%u,%u,%u) "
                    "right=(%u,%u,%u)\n",
                    left[0], left[1], left[2], right[0], right[1], right[2]);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (ssbo) glDeleteBuffers(1, &ssbo);
    if (image) glDeleteTextures(1, &image);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (writer) glDeleteProgram(writer);
    if (readerSSBO) glDeleteProgram(readerSSBO);
    if (readerTex) glDeleteProgram(readerTex);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* P4.5 compute dispatch plan: user glDispatchCompute through the C++ value-
 * state plan (gate-on) or the per-dims ObjC path (gate-off).  The legacy
 * frontend does not parse layout(local_size_*), so Program::local_workgroup_size
 * is always 0 and both paths resolve it to (1,1,1) — one thread per group.
 * Pass 1 dispatches 8 groups (8 invocations, data[i] = i*2+1); pass 2
 * dispatches 4 groups (4 invocations, data[i] = i+100).  SSBO readback
 * verifies both, and gate-on/gate-off must produce identical results. */
static int test_compute_dispatch_ssbo(unsigned char *pixels,
                                      const char *out_path)
{
    (void)out_path;
    memset(pixels, 0, REG_W * REG_H * 4);
    int result = 1;
    GLuint ssbo = 0u, program = 0u;
    int data[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    static const char *cs_odd =
        "#version 430 core\n"
        "layout(std430, binding = 0) buffer Out { int data[8]; };\n"
        "void main() {\n"
        "    uint i = gl_GlobalInvocationID.x;\n"
        "    if (i < 8u) data[i] = int(i) * 2 + 1;\n"
        "}\n";
    static const char *cs_shift =
        "#version 430 core\n"
        "layout(std430, binding = 0) buffer Out { int data[8]; };\n"
        "void main() {\n"
        "    uint i = gl_GlobalInvocationID.x;\n"
        "    if (i < 8u) data[i] = int(i) + 100;\n"
        "}\n";

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data,
                 GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    /* Pass 1: 8 groups x 1 thread = 8 invocations, data[i] = i*2+1. */
    program = link_compute_program(cs_odd);
    if (!program) {
        fprintf(stderr, "compute_dispatch_ssbo: pass 1 link failed\n");
        goto cleanup;
    }
    glUseProgram(program);
    glDispatchCompute(8, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();
    memset(data, 0, sizeof(data));
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(data), data);
    for (int i = 0; i < 8; i++) {
        if (data[i] != i * 2 + 1) {
            fprintf(stderr,
                    "compute_dispatch_ssbo: pass 1 data[%d]=%d want %d\n",
                    i, data[i], i * 2 + 1);
            goto cleanup;
        }
    }
    glDeleteProgram(program);
    program = 0;

    /* Pass 2: 4 groups x 1 thread = 4 invocations, data[i] = i+100. */
    program = link_compute_program(cs_shift);
    if (!program) {
        fprintf(stderr, "compute_dispatch_ssbo: pass 2 link failed\n");
        goto cleanup;
    }
    glUseProgram(program);
    memset(data, 0, sizeof(data));
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data,
                 GL_DYNAMIC_COPY);
    glDispatchCompute(4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();
    memset(data, 0, sizeof(data));
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(data), data);
    for (int i = 0; i < 4; i++) {
        if (data[i] != i + 100) {
            fprintf(stderr,
                    "compute_dispatch_ssbo: pass 2 data[%d]=%d want %d\n",
                    i, data[i], i + 100);
            goto cleanup;
        }
    }
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "compute_dispatch_ssbo: GL operation failed\n");
        goto cleanup;
    }
    result = 0;

cleanup:
    if (program) glDeleteProgram(program);
    if (ssbo) glDeleteBuffers(1, &ssbo);
    return result;
}

static int test_air_geometry_instancing(unsigned char *pixels,
                                        const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 460 core\n"
        "layout(location=0) out vec2 v_offset;\n"
        "void main() {\n"
        "  float x = gl_InstanceID == 0 ? -0.5 : 0.5;\n"
        "  if (gl_BaseInstance != 5) x = 4.0;\n"
        "  v_offset = vec2(x, 0.0);\n"
        "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) in vec2 v_offset[];\n"
        "void main() {\n"
        "  vec2 o = v_offset[0];\n"
        "  gl_Position = vec4(o + vec2(-0.3, -0.5), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(o + vec2( 0.3, -0.5), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(o + vec2( 0.0,  0.5), 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_geometry(vs, gs, fs);
    GLuint vao = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(program);
    glDrawArraysInstancedBaseInstance(GL_POINTS, 0, 1, 2, 5u);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *left =
            &pixels[((REG_H / 2) * REG_W + REG_W / 4) * 4];
        const unsigned char *right =
            &pixels[((REG_H / 2) * REG_W + 3 * REG_W / 4) * 4];
        if (left[0] > 20u || left[1] < 220u || left[2] > 20u ||
            right[0] > 20u || right[1] < 220u || right[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_instancing: expected two green instances, "
                    "got left=(%u,%u,%u) right=(%u,%u,%u)\n",
                    left[0], left[1], left[2],
                    right[0], right[1], right[2]);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* P0 negative test (docs/AIR_M3_CPP_TODO.md §3 P0): an unsupported GS draw
 * must surface GL_INVALID_OPERATION instead of silently dropping the draw.
 * Draw with GL_POINTS against a GS declared `layout(triangles) in;` — the
 * mode/topology mismatch must be reported.  A following matching
 * GL_TRIANGLES draw must NOT report an error. */
static int test_air_gs_unsupported(unsigned char *pixels,
                                   const char *out_path)
{
    (void)pixels;
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position; EmitVertex();\n"
        "  gl_Position = gl_in[1].gl_Position; EmitVertex();\n"
        "  gl_Position = gl_in[2].gl_Position; EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    GLuint program = link_program_with_geometry(vs, gs, fs);
    GLuint vao = 0u;
    int result = 1;
    if (!program) goto cleanup;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(program);

    /* Drain any prior errors, then issue the mismatched draw. */
    while (glGetError() != GL_NO_ERROR) { }

    glDrawArrays(GL_POINTS, 0, 3);
    if (glGetError() != GL_INVALID_OPERATION) {
        fprintf(stderr,
                "air_gs_unsupported: mode/topology mismatch did not raise "
                "GL_INVALID_OPERATION\n");
        goto cleanup;
    }

    /* The matching draw must be accepted without error. */
    while (glGetError() != GL_NO_ERROR) { }
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr,
                "air_gs_unsupported: matching GL_TRIANGLES draw raised an "
                "unexpected error\n");
        goto cleanup;
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    return result;
}

/* P1 regression (docs/AIR_M3_CPP_TODO.md §3 P1): direct indexed GS draws.
 * Covers points-in plain/base-vertex/restart draws plus triangles-in restart
 * in the middle of an incomplete input primitive. */
static int test_air_geometry_indexed(unsigned char *pixels,
                                     const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  gl_Position = vec4(p + vec2(-0.3, -0.4), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.3, -0.4), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.0,  0.4), 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *triGs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "void main() {\n"
        "  for (int i = 0; i < 3; ++i) {\n"
        "    gl_Position = gl_in[i].gl_Position; EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    static const float positions[8] = {
        -0.5f, -0.5f,  /* 0: lower-left  */
         0.5f, -0.5f,  /* 1: lower-right */
         0.0f,  0.5f,  /* 2: top-center  */
         0.7f,  0.7f,  /* 3: top-right   */
    };
    static const uint32_t indices[4] = {0u, 1u, 2u, 3u};
    static const uint32_t restartIndices[5] = {
        0u, 3u, 0xFFFFFFFFu, 1u, 2u,
    };
    static const float triPositions[10] = {
        -0.9f, -0.9f,  /* 0: discarded partial primitive */
        -0.9f, -0.5f,  /* 1: discarded partial primitive */
         0.2f, -0.3f,  /* 2: valid triangle */
         0.8f, -0.3f,  /* 3: valid triangle */
         0.5f,  0.5f,  /* 4: valid triangle */
    };
    static const uint32_t triRestartIndices[6] = {
        0u, 1u, 0xFFFFFFFFu, 2u, 3u, 4u,
    };
    /* Triangle centroid for input point p: (p.x, p.y - 1/15) because the GS
     * expands to p+(-0.3,-0.4),(0.3,-0.4),(0,0.4) — centroid (p.x, p.y-0.1333). */
    static const float expected[3][2] = {
        {-0.5f, -0.6333f}, /* point 0  */
        {0.5f, -0.6333f},  /* point 1  */
        {0.0f,  0.3667f},  /* point 2  */
    };

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_geometry(vs, gs, fs);
    GLuint triangleProgram = link_program_with_geometry(vs, triGs, fs);
    GLuint vao = 0u, vbo = 0u, ebo = 0u;
    int result = 1;
    if (!fbo || !program || !triangleProgram) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);
    glUseProgram(program);

    /* 1) Plain indexed draw: points 0..3, expect green at centroids 0/1/2. */
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawElements(GL_POINTS, 4, GL_UNSIGNED_INT, (void *)0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    for (int i = 0; i < 3; i++) {
        int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
        int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
        if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
        const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_indexed: plain indexed centroid %d "
                    "expected green, got (%u,%u,%u)\n",
                    i, px[0], px[1], px[2]);
            goto cleanup;
        }
    }

    /* 2) Base-vertex: indices 2..3 resolve to points 2..3 (top area). */
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawElementsBaseVertex(GL_POINTS, 2, GL_UNSIGNED_INT, (void *)0, 2);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        int sx = (int)((expected[2][0] + 1.0) * 0.5 * REG_W);
        int sy = (int)((expected[2][1] + 1.0) * 0.5 * REG_H);
        if (sx >= 0 && sx < REG_W && sy >= 0 && sy < REG_H) {
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                fprintf(stderr,
                        "air_geometry_indexed: base-vertex centroid expected "
                        "green, got (%u,%u,%u)\n",
                        px[0], px[1], px[2]);
                goto cleanup;
            }
        }
        /* The base-vertex draw must NOT cover point 0's centroid. */
        sx = (int)((expected[0][0] + 1.0) * 0.5 * REG_W);
        sy = (int)((expected[0][1] + 1.0) * 0.5 * REG_H);
        if (sx >= 0 && sx < REG_W && sy >= 0 && sy < REG_H) {
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[1] > 20u) {
                fprintf(stderr,
                        "air_geometry_indexed: base-vertex leaked into point 0 "
                        "region, got (%u,%u,%u)\n",
                        px[0], px[1], px[2]);
                goto cleanup;
            }
        }
    }

    /* 3) Primitive restart: [0, 3, restart, 1, 2] splits into [0,3] and
     * [1,2]; all four points expand (each index is one points-in primitive
     * per segment). */
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(restartIndices),
                 restartIndices, GL_STATIC_DRAW);
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFFFFFFu);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawElements(GL_POINTS, 5, GL_UNSIGNED_INT, (void *)0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glDisable(GL_PRIMITIVE_RESTART);
    for (int i = 0; i < 3; i++) {
        int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
        int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
        if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
        const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_indexed: restart centroid %d expected "
                    "green, got (%u,%u,%u)\n",
                    i, px[0], px[1], px[2]);
            goto cleanup;
        }
    }

    /* 4) Triangles-in restart inside an incomplete primitive.  [0,1] must
     * be discarded at the restart and only [2,3,4] may reach the GS.  The
     * old gather bug formed [0,1,2], producing the lower-left decoy. */
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triPositions), triPositions,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(triRestartIndices),
                 triRestartIndices, GL_STATIC_DRAW);
    glUseProgram(triangleProgram);
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFFFFFFu);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glDisable(GL_PRIMITIVE_RESTART);
    {
        const float expectedCenter[2] = {0.5f, -0.0333f};
        const float staleCenter[2] = {-0.5333f, -0.5667f};
        int sx = (int)((expectedCenter[0] + 1.0f) * 0.5f * REG_W);
        int sy = (int)((expectedCenter[1] + 1.0f) * 0.5f * REG_H);
        const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_indexed: mid-restart valid triangle missing "
                    "at (%d,%d), got (%u,%u,%u)\n",
                    sx, sy, px[0], px[1], px[2]);
            goto cleanup;
        }
        sx = (int)((staleCenter[0] + 1.0f) * 0.5f * REG_W);
        sy = (int)((staleCenter[1] + 1.0f) * 0.5f * REG_H);
        px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] > 20u || px[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_indexed: mid-restart leaked discarded "
                    "fragment at (%d,%d), got (%u,%u,%u)\n",
                    sx, sy, px[0], px[1], px[2]);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (triangleProgram) glDeleteProgram(triangleProgram);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* P1 regression: indirect GS draws (docs/AIR_M3_CPP_TODO.md §3 P1).
 * Same points-in GS as air_geometry_indexed; verifies the renderer reads
 * the indirect command back and routes it through the GS expansion. */
static int test_air_geometry_indirect(unsigned char *pixels,
                                      const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  gl_Position = vec4(p + vec2(-0.3, -0.4), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.3, -0.4), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.0,  0.4), 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const float positions[8] = {
        -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f, 0.7f, 0.7f,
    };
    static const uint32_t indices[4] = {0u, 1u, 2u, 3u};
    static const float expected[3][2] = {
        {-0.5f, -0.6333f}, {0.5f, -0.6333f}, {0.0f, 0.3667f},
    };
    struct {
        GLuint count;
        GLuint primCount;
        GLuint first;
        GLuint baseInstance;
    } arrayCmd = {4u, 1u, 0u, 0u};
    struct {
        GLuint count;
        GLuint primCount;
        GLuint first;
        GLint baseVertex;
        GLuint baseInstance;
    } elemCmd = {4u, 1u, 0u, 0, 0u};

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_geometry(vs, gs, fs);
    GLuint vao = 0u, vbo = 0u, ebo = 0u, cmdBuf = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);
    glUseProgram(program);

    /* 1) glDrawArraysIndirect → GS expansion. */
    clear_color(0.0f, 0.0f, 0.0f);
    glGenBuffers(1, &cmdBuf);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cmdBuf);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(arrayCmd), &arrayCmd,
                 GL_STATIC_DRAW);
    glDrawArraysIndirect(GL_POINTS, (void *)0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    for (int i = 0; i < 3; i++) {
        int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
        int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
        if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
        const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_indirect: arrays-indirect centroid %d "
                    "expected green, got (%u,%u,%u)\n",
                    i, px[0], px[1], px[2]);
            goto cleanup;
        }
    }

    /* 2) glDrawElementsIndirect → GS expansion. */
    clear_color(0.0f, 0.0f, 0.0f);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cmdBuf);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(elemCmd), &elemCmd,
                 GL_STATIC_DRAW);
    glDrawElementsIndirect(GL_POINTS, GL_UNSIGNED_INT, (void *)0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    for (int i = 0; i < 3; i++) {
        int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
        int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
        if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
        const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_indirect: elements-indirect centroid %d "
                    "expected green, got (%u,%u,%u)\n",
                    i, px[0], px[1], px[2]);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (cmdBuf) glDeleteBuffers(1, &cmdBuf);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* P1 regression: GS multi-draw (docs/AIR_M3_CPP_TODO.md §3 P1).
 * glMultiDrawElements splits the index stream into two sub-draws;
 * glMultiDrawArraysIndirect decodes two commands. */
static int test_air_geometry_multi_draw(unsigned char *pixels,
                                        const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  gl_Position = vec4(p + vec2(-0.3, -0.4), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.3, -0.4), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.0,  0.4), 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const float positions[8] = {
        -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f, 0.7f, 0.7f,
    };
    static const uint32_t indices[4] = {0u, 1u, 2u, 3u};
    static const float expected[3][2] = {
        {-0.5f, -0.6333f}, {0.5f, -0.6333f}, {0.0f, 0.3667f},
    };

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_geometry(vs, gs, fs);
    GLuint vao = 0u, vbo = 0u, ebo = 0u, cmdBuf = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);
    glUseProgram(program);

    /* 1) glMultiDrawElements: two sub-draws [0,1] and [2,3]. */
    {
        GLsizei counts[2] = {2, 2};
        const void *subIndices[2] = {
            (void *)0u, (void *)(2u * sizeof(uint32_t)),
        };
        clear_color(0.0f, 0.0f, 0.0f);
        glMultiDrawElements(GL_POINTS, counts, GL_UNSIGNED_INT, subIndices, 2);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        for (int i = 0; i < 3; i++) {
            int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
            int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
            if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                fprintf(stderr,
                        "air_geometry_multi_draw: multi-draw-elements centroid "
                        "%d expected green, got (%u,%u,%u)\n",
                        i, px[0], px[1], px[2]);
                goto cleanup;
            }
        }
    }

    /* 2) glMultiDrawArraysIndirect: commands [0,1] and [2,3]. */
    {
        struct {
            GLuint count;
            GLuint primCount;
            GLuint first;
            GLuint baseInstance;
        } cmds[2] = {
            {2u, 1u, 0u, 0u},
            {2u, 1u, 2u, 0u},
        };
        clear_color(0.0f, 0.0f, 0.0f);
        glGenBuffers(1, &cmdBuf);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cmdBuf);
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmds), cmds,
                     GL_STATIC_DRAW);
        glMultiDrawArraysIndirect(GL_POINTS, (void *)0, 2, 0);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        for (int i = 0; i < 3; i++) {
            int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
            int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
            if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                fprintf(stderr,
                        "air_geometry_multi_draw: multi-draw-arrays-indirect "
                        "centroid %d expected green, got (%u,%u,%u)\n",
                        i, px[0], px[1], px[2]);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (cmdBuf) glDeleteBuffers(1, &cmdBuf);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* P1 regression: instanced indexed GS with base-vertex and base-instance
 * (docs/AIR_M3_CPP_TODO.md §3 P1).  drawElementsInstancedBaseVertexBaseInstance
 * must expand each (instance x indexed vertex) through the GS. */
static int test_air_geometry_base_vertex_instance(unsigned char *pixels,
                                                  const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  gl_Position = vec4(p + vec2(-0.3, -0.4), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.3, -0.4), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.0,  0.4), 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    /* Indices 2/3 resolve directly to VBO[2]=(-0.3,-0.3), VBO[3]=(0.3,-0.3);
     * both instances draw the same two triangles (overlapping). */
    static const float positions[8] = {
        0.0f, 0.0f, 0.5f, 0.5f, -0.3f, -0.3f, 0.3f, -0.3f,
    };
    static const uint32_t indices[2] = {2u, 3u};
    static const float expected[2][2] = {
        {-0.3f, -0.4333f}, {0.3f, -0.4333f},
    };

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_geometry(vs, gs, fs);
    GLuint vao = 0u, vbo = 0u, ebo = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);
    glUseProgram(program);

    glDrawElementsInstancedBaseVertexBaseInstance(
        GL_POINTS, 2, GL_UNSIGNED_INT, (void *)0, 2, 0, 0u);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    for (int i = 0; i < 2; i++) {
        int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
        int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
        if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
        const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
            fprintf(stderr,
                    "air_geometry_base_vertex_instance: centroid %d expected "
                    "green, got (%u,%u,%u)\n",
                    i, px[0], px[1], px[2]);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_indexed(unsigned char *pixels,
                                         const char *out_path)
{
    (void)out_path;
    /* TES-only (no TCS): indexed native TES.  The EBO is deliberately
     * shuffled ([2,0,1] then [5,3,4]) so a wrong control-point stream
     * (array-order instead of gather) produces a visibly wrong triangle. */
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position * gl_TessCoord.x +\n"
        "                gl_in[1].gl_Position * gl_TessCoord.y +\n"
        "                gl_in[2].gl_Position * gl_TessCoord.z;\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    /* 8 vertices: triangle A = {0,1,2} around origin; triangle B = {3,4,5}
     * offset left; {6,7} are fill so the gather maxIndex covers the span. */
    static const float positions[16] = {
        -0.6f, -0.4f,  0.6f, -0.4f,  0.0f,  0.6f,
        -0.9f, -0.3f, -0.3f, -0.3f, -0.6f,  0.6f,
         1.0f,  1.0f, -1.0f, -1.0f,
    };
    /* Shuffled index stream: [2,0,1] + [5,3,4] = two triangles. */
    static const uint32_t indices[6] = {2u, 0u, 1u, 2u, 0u, 1u};

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_tess_eval_only(vs, tes, fs);
    GLuint vao = 0u, vbo = 0u, ebo = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    /* Draw 1: plain indexed.  Patch 0 = [2,0,1] (origin triangle).  The EBO
     * is shuffled ([2,0,1] instead of [0,1,2]) so a wrong control-point
     * stream (array-order instead of gather) produces a wrong triangle. */
    glDrawElements(GL_PATCHES, 3, GL_UNSIGNED_INT, (void *)0);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Triangle A centroid ≈ (0.0, -0.0667). */
        static const float expected[1][2] = {
            {0.0f, -0.0667f},
        };
        for (int i = 0; i < 1; i++) {
            int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
            int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
            if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                fprintf(stderr,
                        "air_tessellation_indexed: triangle %d centroid "
                        "expected green, got (%u,%u,%u)\n",
                        i, px[0], px[1], px[2]);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_instanced(unsigned char *pixels,
                                           const char *out_path)
{
    (void)out_path;
    /* TES-only (no TCS): instanced native TES.  The VS moves each instance
     * by gl_InstanceID on x, so instance 1 must render at +0.6 — a wrong
     * implementation (same patch data for every instance, or a capture
     * that ignores per-instance records) draws both triangles at the
     * instance-0 location. */
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position.x + float(gl_InstanceID) * 0.6,"
        "                      position.y, 0.0, 1.0);\n"
        "}\n";
    static const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position * gl_TessCoord.x +\n"
        "                gl_in[1].gl_Position * gl_TessCoord.y +\n"
        "                gl_in[2].gl_Position * gl_TessCoord.z;\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    /* One triangle; centroid ≈ (0.0, -0.133). */
    static const float positions[6] = {
        -0.4f, -0.4f,  0.4f, -0.4f,  0.0f,  0.4f,
    };

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_tess_eval_only(vs, tes, fs);
    GLuint vao = 0u, vbo = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    /* baseInstance=1: gl_InstanceID must stay 0/1 (instance_id - base), so
     * the instance offsets are the same as a plain instanced draw. */
    glDrawArraysInstancedBaseInstance(GL_PATCHES, 0, 3, 2, 1);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Instance 0 centroid ≈ (0.0, -0.133), instance 1 ≈ (0.6, -0.133). */
        static const float expected[2][2] = {
            {0.0f, -0.133f},
            {0.6f, -0.133f},
        };
        for (int i = 0; i < 2; i++) {
            int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
            int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
            if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                fprintf(stderr,
                        "air_tessellation_instanced: instance %d centroid "
                        "expected green, got (%u,%u,%u) at (%d,%d)\n",
                        i, px[0], px[1], px[2], sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_multipatch(unsigned char *pixels,
                                            const char *out_path)
{
    (void)out_path;
    /* TES-only (no TCS): two patches and two instances in one draw.  The
     * 80-byte capture records produce a 480-byte instance stride, while the
     * TES varying and gl_PrimitiveID distinguish every patch. */
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=0) out vec2 cp_position;\n"
        "void main() {\n"
        "  cp_position = position + vec2(0.0, float(gl_InstanceID) * 0.9);\n"
        "  gl_Position = vec4(cp_position, 0.0, 1.0);\n"
        "}\n";
    static const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(location=0) in vec2 cp_position[];\n"
        "void main() {\n"
        "  vec2 p = cp_position[0] * gl_TessCoord.x +\n"
        "           cp_position[1] * gl_TessCoord.y +\n"
        "           cp_position[2] * gl_TessCoord.z;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_Position.y += float(gl_PrimitiveID) * 0.3;\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *factor_vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *factor_tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    float level = 0.0;\n"
        "    if (gl_PrimitiveID == 0) level = 1.0;\n"
        "    gl_TessLevelOuter[0] = level;\n"
        "    gl_TessLevelOuter[1] = level;\n"
        "    gl_TessLevelOuter[2] = level;\n"
        "    gl_TessLevelInner[0] = level;\n"
        "  }\n"
        "}\n";
    static const char *factor_tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position * gl_TessCoord.x +\n"
        "                gl_in[1].gl_Position * gl_TessCoord.y +\n"
        "                gl_in[2].gl_Position * gl_TessCoord.z;\n"
        "}\n";
    static const float positions[12] = {
        -0.7f, -0.75f, -0.3f, -0.75f, -0.5f, -0.45f,
         0.3f, -0.75f,  0.7f, -0.75f,  0.5f, -0.45f,
    };

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_tess_eval_only(vs, tes, fs);
    GLuint factor_program = link_program_with_tessellation(
        factor_vs, factor_tcs, factor_tes, fs);
    GLuint vao = 0u, vbo = 0u;
    int result = 1;
    if (!fbo || !program || !factor_program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    glDrawArraysInstanced(GL_PATCHES, 0, 6, 2);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        static const float expected[4][2] = {
            {-0.5f, -0.65f},
            { 0.5f, -0.35f},
            {-0.5f,  0.25f},
            { 0.5f,  0.55f},
        };
        for (int i = 0; i < 4; i++) {
            const int sx = (int)((expected[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((expected[i][1] + 1.0f) * 0.5f * REG_H);
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                fprintf(stderr,
                        "air_tessellation_multipatch: probe %d centroid "
                        "expected green, got (%u,%u,%u) at (%d,%d)\n",
                        i, px[0], px[1], px[2], sx, sy);
                goto cleanup;
            }
        }
    }

    /* TCS factor addressing: patch 0 has level 1 and must render; patch 1
     * has level 0 and must be discarded.  Resetting patchStart to zero would
     * incorrectly reuse patch 0's factor record and render both patches. */
    clear_color(0.0f, 0.0f, 0.0f);
    glUseProgram(factor_program);
    glDrawArrays(GL_PATCHES, 0, 6);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        static const float probes[2][2] = {
            {-0.5f, -0.65f},
            { 0.5f, -0.65f},
        };
        for (int i = 0; i < 2; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            const int is_green = px[0] <= 20u && px[1] >= 220u && px[2] <= 20u;
            if ((i == 0 && !is_green) || (i == 1 && is_green)) {
                fprintf(stderr,
                        "air_tessellation_multipatch: factor probe %d got "
                        "(%u,%u,%u) at (%d,%d)\n",
                        i, px[0], px[1], px[2], sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (program) glDeleteProgram(program);
    if (factor_program) glDeleteProgram(factor_program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_indirect(unsigned char *pixels,
                                          const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";
    static const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position * gl_TessCoord.x +\n"
        "                gl_in[1].gl_Position * gl_TessCoord.y +\n"
        "                gl_in[2].gl_Position * gl_TessCoord.z;\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 fragColor;\n"
        "void main() {\n"
        "  fragColor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n";

    /* Triangles: A = {(-0.4,-0.4),(0.4,-0.4),(0,0.4)} → centroid
     * ≈ (0.0,-0.133); B = A shifted by +0.6 on x → centroid ≈ (0.6,-0.133). */
    static const float verts[6][2] = {
        {-0.4f, -0.4f}, {0.4f, -0.4f}, {0.0f, 0.4f},
        {0.2f, -0.4f},  {1.0f, -0.4f}, {0.6f, 0.4f},
    };
    static const GLuint indices[6] = {0, 1, 2, 3, 4, 5};

    GLuint vao = 0, vbo = 0, ebo = 0, cmd_buf = 0, program = 0;
    GLuint fbo = 0, color = 0;
    int result = -1;

    program = link_program_tess_eval_only(vs, tes, fs);
    if (!program) goto cleanup;
    glUseProgram(program);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    vbo = make_vbo(verts, sizeof(verts));
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glGenBuffers(1, &cmd_buf);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cmd_buf);

    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, REG_W, REG_H);

    /* 1) drawArraysIndirect: one patch (3 vertices). */
    {
        /* DrawArraysIndirectCommand: {count, primCount, first, baseInstance} */
        GLuint cmd[4] = {3u, 1u, 0u, 0u};
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmd), cmd, GL_STATIC_DRAW);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArraysIndirect(GL_PATCHES, 0);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        int sx = (int)((1.0 - 0.133) * 0.5 * REG_W);
        int sy = (int)((1.0 - 0.133) * 0.5 * REG_H);
        const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
            fprintf(stderr,
                    "air_tessellation_indirect: drawArraysIndirect centroid "
                    "expected green, got (%u,%u,%u) at (%d,%d)\n",
                    px[0], px[1], px[2], sx, sy);
            goto cleanup;
        }
    }

    /* 2) drawElementsIndirect: one patch (3 indices). */
    {
        /* DrawElementsIndirectCommand:
         * {count, primCount, firstIndex, baseVertex, baseInstance} */
        GLuint cmd[5] = {3u, 1u, 0u, 0u, 0u};
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmd), cmd, GL_STATIC_DRAW);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_INT, 0);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        int sx = (int)((1.0 - 0.133) * 0.5 * REG_W);
        int sy = (int)((1.0 - 0.133) * 0.5 * REG_H);
        const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
        if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
            fprintf(stderr,
                    "air_tessellation_indirect: drawElementsIndirect centroid "
                    "expected green, got (%u,%u,%u) at (%d,%d)\n",
                    px[0], px[1], px[2], sx, sy);
            goto cleanup;
        }
    }

    /* 3) multiDrawArraysIndirect: two patches (A first=0, B first=3). */
    {
        /* DrawArraysIndirectCommand: {count, primCount, first, baseInstance} */
        GLuint cmd[8] = {3u, 1u, 0u, 0u, 3u, 1u, 3u, 0u};
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmd), cmd, GL_STATIC_DRAW);
        glClear(GL_COLOR_BUFFER_BIT);
        glMultiDrawArraysIndirect(GL_PATCHES, 0, 2, 0);
        glFinish();
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        static const float expected[2][2] = {
            {0.0f, -0.133f},
            {0.6f, -0.133f},
        };
        for (int i = 0; i < 2; i++) {
            int sx = (int)((expected[i][0] + 1.0) * 0.5 * REG_W);
            int sy = (int)((expected[i][1] + 1.0) * 0.5 * REG_H);
            if (sx < 0 || sx >= REG_W || sy < 0 || sy >= REG_H) goto cleanup;
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                fprintf(stderr,
                        "air_tessellation_indirect: multiDrawArraysIndirect "
                        "patch %d centroid expected green, got (%u,%u,%u) "
                        "at (%d,%d)\n",
                        i, px[0], px[1], px[2], sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (cmd_buf) glDeleteBuffers(1, &cmd_buf);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_varying(unsigned char *pixels,
                                         const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=0) out vec3 v_control;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  v_control = vec3(1.0);\n"
        "}\n";
    static const char *tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "layout(location=0) in vec3 v_control[];\n"
        "layout(location=0) out vec3 tc_control[];\n"
        "layout(location=1) patch out vec3 patch_color;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  tc_control[gl_InvocationID] = v_control[gl_InvocationID];\n"
        "  if (gl_InvocationID == 0) {\n"
        "    patch_color = vec3(0.0, 1.0, 0.0);\n"
        "    gl_TessLevelOuter[0] = 1.0;\n"
        "    gl_TessLevelOuter[1] = 1.0;\n"
        "    gl_TessLevelOuter[2] = 1.0;\n"
        "    gl_TessLevelInner[0] = 1.0;\n"
        "  }\n"
        "}\n";
    static const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(location=0) in vec3 tc_control[];\n"
        "layout(location=1) patch in vec3 patch_color;\n"
        "layout(location=0) out vec3 te_color;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position * gl_TessCoord.x + "
        "gl_in[1].gl_Position * gl_TessCoord.y + "
        "gl_in[2].gl_Position * gl_TessCoord.z;\n"
        "  te_color = patch_color * tc_control[0];\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec3 te_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(te_color, 1.0); }\n";

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_tessellation(vs, tcs, tes, fs);
    GLuint vao = 0u, vbo = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    make_pos2_vao(TRI_VERTS, sizeof(TRI_VERTS), &vao, &vbo);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *center =
            &pixels[((REG_H / 2) * REG_W + REG_W / 2) * 4];
        if (center[0] > 20u || center[1] < 220u || center[2] > 20u) {
            fprintf(stderr,
                    "air_tessellation_varying: center expected green, got "
                    "(%u,%u,%u,%u)\n",
                    center[0], center[1], center[2], center[3]);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* TCS per-patch input/output: TCS reads the full patch control-point stream
 * (per-patch input) and derives a patch-qualified varying, which the TES
 * consumes via `patch in`.  Also covers a subdivided triangle (outer=3,
 * inner=2, fractional_odd spacing, ccw winding) across two patches, each
 * routed to a different patch color — verifying per-patch output does not
 * leak across patches. */
static int test_air_tessellation_patch_varying(unsigned char *pixels,
                                               const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=0) out vec2 v_uv;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  v_uv = position;\n"
        "}\n";
    static const char *tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "layout(location=0) in vec2 v_uv[];\n"
        "layout(location=0) out vec2 tc_uv[];\n"
        "layout(location=1) patch out vec3 patch_color;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  tc_uv[gl_InvocationID] = v_uv[gl_InvocationID];\n"
        "  if (gl_InvocationID == 0) {\n"
        "    float x = gl_in[0].gl_Position.x;\n"
        "    patch_color = (x < 0.0) ? vec3(0.0, 1.0, 0.0) : "
        "vec3(0.0, 0.0, 1.0);\n"
        "    gl_TessLevelOuter[0] = 3.0;\n"
        "    gl_TessLevelOuter[1] = 3.0;\n"
        "    gl_TessLevelOuter[2] = 3.0;\n"
        "    gl_TessLevelInner[0] = 2.0;\n"
        "  }\n"
        "}\n";
    static const char *tes =
        "#version 450 core\n"
        "layout(triangles, fractional_odd_spacing, ccw) in;\n"
        "layout(location=0) in vec2 tc_uv[];\n"
        "layout(location=1) patch in vec3 patch_color;\n"
        "layout(location=0) out vec3 te_color;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position * gl_TessCoord.x + "
        "gl_in[1].gl_Position * gl_TessCoord.y + "
        "gl_in[2].gl_Position * gl_TessCoord.z;\n"
        "  te_color = patch_color;\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec3 te_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(te_color, 1.0); }\n";

    /* Two patches: left (x<0 → green), right (x>0 → blue). */
    static const float verts[] = {
        -1.0f, -1.0f,
         0.0f, -1.0f,
        -0.5f,  0.9f,
         0.0f, -1.0f,
         1.0f, -1.0f,
         0.5f,  0.9f,
    };

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_with_tessellation(vs, tcs, tes, fs);
    GLuint vao = 0u, vbo = 0u;
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    make_pos2_vao(verts, sizeof(verts), &vao, &vbo);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 6);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_tessellation_patch_varying: draw failed\n");
        goto cleanup;
    }
    {
        const unsigned char *left =
            &pixels[((REG_H / 2) * REG_W + REG_W / 4) * 4];
        const unsigned char *right =
            &pixels[((REG_H / 2) * REG_W + 3 * REG_W / 4) * 4];
        if (left[0] > 20u || left[1] < 220u || left[2] > 20u ||
            right[0] > 20u || right[1] > 20u || right[2] < 220u) {
            fprintf(stderr,
                    "air_tessellation_patch_varying: expected green/blue "
                    "patches, got (%u,%u,%u)/(%u,%u,%u)\n",
                    left[0], left[1], left[2],
                    right[0], right[1], right[2]);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* TES resource binding: sampler2D + std140 UBO + std430 SSBO read inside
 * the evaluation shader (native TES path).  Each resource is re-bound /
 * mutated between draws to prove the TES stage actually re-reads it:
 * 1) white tex × green tint × white factor → green
 * 2) tint flipped to blue → blue   (UBO re-read)
 * 3) factor flipped to red → red    (SSBO re-read) */
static int test_air_tessellation_resources(unsigned char *pixels,
                                           const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=0) out vec2 v_uv;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  v_uv = position;\n"
        "}\n";
    static const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(location=0) in vec2 v_uv[];\n"
        "layout(location=0) out vec3 te_color;\n"
        "layout(binding=0) uniform sampler2D sampleTex;\n"
        "layout(std140, binding=1) uniform Params { vec4 tint; };\n"
        "layout(std430, binding=2) buffer Data { vec4 factor; } dataBuffer;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position * gl_TessCoord.x + "
        "gl_in[1].gl_Position * gl_TessCoord.y + "
        "gl_in[2].gl_Position * gl_TessCoord.z;\n"
        "  vec4 f = dataBuffer.factor;\n"
        "  te_color = texture(sampleTex, vec2(0.5, 0.5)).rgb * "
        "tint.rgb * f.rgb;\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec3 te_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(te_color, 1.0); }\n";

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint program = link_program_tess_eval_only(vs, tes, fs);
    GLuint vao = 0u, vbo = 0u;
    GLuint ubo = 0u, ssbo = 0u, sampled = 0u;
    const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    const float blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    const float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const unsigned char texel[4] = {255u, 255u, 255u, 255u};
    int result = 1;
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(TRI_VERTS), TRI_VERTS,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(green), green, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(white), white,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);

    glGenTextures(1, &sampled);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sampled);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, texel);

    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    /* Segment 1: green (white × green × white). */
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *center =
            &pixels[((REG_H / 2) * REG_W + REG_W / 2) * 4];
        if (center[0] > 20u || center[1] < 220u || center[2] > 20u) {
            fprintf(stderr,
                    "air_tessellation_resources: segment 1 expected green, "
                    "got (%u,%u,%u,%u)\n",
                    center[0], center[1], center[2], center[3]);
            goto cleanup;
        }
    }

    /* Segment 2: tint → blue via UBO re-read. */
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(blue), blue);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *center =
            &pixels[((REG_H / 2) * REG_W + REG_W / 2) * 4];
        if (center[0] > 20u || center[1] > 20u || center[2] < 220u) {
            fprintf(stderr,
                    "air_tessellation_resources: segment 2 expected blue, "
                    "got (%u,%u,%u,%u)\n",
                    center[0], center[1], center[2], center[3]);
            goto cleanup;
        }
    }

    /* Segment 3: tint back to white, factor → red via SSBO re-read.
     * (green tint × red factor would multiply to black, so tint is white
     * here to isolate the SSBO re-read.) */
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(white), white);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(red), red);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_tessellation_resources: segment 3 GL error\n");
        goto cleanup;
    }
    {
        const unsigned char *center =
            &pixels[((REG_H / 2) * REG_W + REG_W / 2) * 4];
        if (center[0] < 220u || center[1] > 20u || center[2] > 20u) {
            fprintf(stderr,
                    "air_tessellation_resources: segment 3 expected red, "
                    "got (%u,%u,%u,%u)\n",
                    center[0], center[1], center[2], center[3]);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    /* Unbind base bindings so later suite tests (single shared context)
     * do not observe stale UBO/SSBO/sampler references to deleted objects. */
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    if (sampled) glDeleteTextures(1, &sampled);
    if (ssbo) glDeleteBuffers(1, &ssbo);
    if (ubo) glDeleteBuffers(1, &ubo);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_isolines_point_mode(unsigned char *pixels,
                                                     const char *out_path)
{
    (void)out_path;
    /* Isolines and layout(point_mode) have no Metal-native tessellation
     * equivalent; they must run through the AIR TES compute expansion +
     * passthrough vertex (line/point rasterization).  Three draws:
     *
     *  1. isolines, outer = {4, 2} -> 2 isolines x 4 line segments each;
     *     segment midpoints are probed.
     *  2. quads point_mode, inner = {3, 3} -> 9 points at grid cell
     *     centres; all 9 exact locations are probed.
     *  3. triangles point_mode, inner = 2 -> 4 points at the cell
     *     centroids; all 4 are probed. */
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *iso_tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelOuter[0] = 4.0;\n"
        "    gl_TessLevelOuter[1] = 2.0;\n"
        "  }\n"
        "}\n";
    static const char *iso_tes =
        "#version 450 core\n"
        "layout(isolines, equal_spacing) in;\n"
        "void main() {\n"
        "  vec2 a = gl_in[0].gl_Position.xy;\n"
        "  vec2 b = gl_in[1].gl_Position.xy;\n"
        "  vec2 p = mix(a, b, gl_TessCoord.x);\n"
        "  p.y += gl_TessCoord.y * 0.6;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "}\n";
    static const char *quad_tcs =
        "#version 450 core\n"
        "layout(vertices=4) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelInner[0] = 3.0;\n"
        "    gl_TessLevelInner[1] = 3.0;\n"
        "  }\n"
        "}\n";
    static const char *quad_tes =
        "#version 450 core\n"
        "layout(quads, equal_spacing, point_mode) in;\n"
        "void main() {\n"
        "  vec2 p00 = gl_in[0].gl_Position.xy;\n"
        "  vec2 p10 = gl_in[1].gl_Position.xy;\n"
        "  vec2 p01 = gl_in[2].gl_Position.xy;\n"
        "  vec2 p11 = gl_in[3].gl_Position.xy;\n"
        "  vec2 p = mix(mix(p00, p10, gl_TessCoord.x),\n"
        "               mix(p01, p11, gl_TessCoord.x), gl_TessCoord.y);\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_PointSize = 8.0;\n"
        "}\n";
    static const char *tri_tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelInner[0] = 2.0;\n"
        "  }\n"
        "}\n";
    static const char *tri_tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, point_mode) in;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy * gl_TessCoord.x +\n"
        "           gl_in[1].gl_Position.xy * gl_TessCoord.y +\n"
        "           gl_in[2].gl_Position.xy * gl_TessCoord.z;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_PointSize = 8.0;\n"
        "}\n";
    /* Isoline control points A(-0.7,-0.3) B(0.7,-0.3); C is unused. */
    static const float iso_positions[6] = {
        -0.7f, -0.3f, 0.7f, -0.3f, 0.0f, -0.9f,
    };
    /* Quad corners: p00 p10 p01 p11. */
    static const float quad_positions[8] = {
        -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f,
    };
    /* Triangle A(-0.6,-0.4) B(0.6,-0.4) C(0,0.7). */
    static const float tri_positions[6] = {
        -0.6f, -0.4f, 0.6f, -0.4f, 0.0f, 0.7f,
    };

    GLuint color = 0u;
    GLuint fbo = make_fbo(REG_W, REG_H, &color);
    GLuint iso_program =
        link_program_with_tessellation(vs, iso_tcs, iso_tes, fs);
    GLuint quad_program =
        link_program_with_tessellation(vs, quad_tcs, quad_tes, fs);
    GLuint tri_program =
        link_program_with_tessellation(vs, tri_tcs, tri_tes, fs);
    GLuint vao = 0u, vbo = 0u;
    int result = 1;
    if (!fbo || !iso_program || !quad_program || !tri_program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);

    /* Draw 1: isolines, outer = {4, 2}.  Two lines at v = 0.25 / 0.75,
     * each with segments at u in {0.125, 0.375, 0.625, 0.875} midpoints. */
    clear_color(0.0f, 0.0f, 0.0f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(iso_positions), iso_positions,
                 GL_STATIC_DRAW);
    glUseProgram(iso_program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        static const float probes[4][2] = {
            {-0.175f, -0.15f}, { 0.175f, -0.15f},
            {-0.175f,  0.15f}, { 0.175f,  0.15f},
        };
        for (int i = 0; i < 4; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_point_mode: isoline "
                        "probe %d not drawn at (%d,%d)\n",
                        i, sx, sy);
                goto cleanup;
            }
        }
    }

    /* Draw 2: quads point_mode, inner = {3, 3} -> 9 points at the grid
     * cell centres x = -1/3, 0, 1/3; y = -1/3, 0, 1/3. */
    clear_color(0.0f, 0.0f, 0.0f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_positions), quad_positions,
                 GL_STATIC_DRAW);
    glUseProgram(quad_program);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    glDrawArrays(GL_PATCHES, 0, 4);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        static const float xs[3] = {-1.0f / 3.0f, 0.0f, 1.0f / 3.0f};
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
                const int sx = (int)((xs[i] + 1.0f) * 0.5f * REG_W);
                const int sy = (int)((xs[j] + 1.0f) * 0.5f * REG_H);
                const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
                if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                    fprintf(stderr,
                            "air_tessellation_isolines_point_mode: quad "
                            "point (%d,%d) expected green, got (%u,%u,%u)\n",
                            i, j, px[0], px[1], px[2]);
                    goto cleanup;
                }
            }
        }
    }

    /* Draw 3: triangles point_mode, inner = 2 -> 4 points at (u,v) =
     * (1/6,1/6), (2/3,1/6), (1/6,2/3), (2/3,2/3) with w = 1-u-v, mapped
     * via p = A*u + B*v + C*w (A(-0.6,-0.4) B(0.6,-0.4) C(0,0.7)):
     * (0,1/3), (-0.3,-13/60), (0.3,-13/60), (0,-23/30). */
    clear_color(0.0f, 0.0f, 0.0f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri_positions), tri_positions,
                 GL_STATIC_DRAW);
    glUseProgram(tri_program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        static const float probes[4][2] = {
            { 0.0f,  1.0f / 3.0f},
            {-0.3f, -13.0f / 60.0f},
            { 0.3f, -13.0f / 60.0f},
            { 0.0f, -23.0f / 30.0f},
        };
        for (int i = 0; i < 4; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
            if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                fprintf(stderr,
                        "air_tessellation_isolines_point_mode: triangle "
                        "point %d expected green, got (%u,%u,%u) at "
                        "(%d,%d)\n",
                        i, px[0], px[1], px[2], sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (iso_program) glDeleteProgram(iso_program);
    if (quad_program) glDeleteProgram(quad_program);
    if (tri_program) glDeleteProgram(tri_program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_isolines_variants(unsigned char *pixels,
                                                    const char *out_path)
{
    /* P2E combination coverage on top of air_tessellation_isolines_point_mode:
     * TES-only (no TCS) draws, instanced (VS shifts per gl_InstanceID),
     * multi-patch (two isoline patches, per-patch TCS factors) and an
     * indirect command, plus GL_PRIMITIVES_GENERATED query counts. */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position + vec2(float(gl_InstanceID) * 0.6, 0.0),\n"
        "                     0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *iso_tes =
        "#version 450 core\n"
        "layout(isolines, equal_spacing) in;\n"
        "void main() {\n"
        "  vec2 p = mix(gl_in[0].gl_Position.xy, gl_in[1].gl_Position.xy,\n"
        "               gl_TessCoord.x);\n"
        "  p.y += gl_TessCoord.y * 0.6;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "}\n";
    static const char *quad_tes =
        "#version 450 core\n"
        "layout(quads, equal_spacing, point_mode) in;\n"
        "void main() {\n"
        "  vec2 p = mix(mix(gl_in[0].gl_Position.xy, gl_in[1].gl_Position.xy,\n"
        "                   gl_TessCoord.x),\n"
        "               mix(gl_in[2].gl_Position.xy, gl_in[3].gl_Position.xy,\n"
        "                   gl_TessCoord.x), gl_TessCoord.y);\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_PointSize = 8.0;\n"
        "}\n";
    static const char *mp_tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    if (gl_PrimitiveID == 0) {\n"
        "      gl_TessLevelOuter[0] = 4.0;\n"
        "      gl_TessLevelOuter[1] = 2.0;\n"
        "    } else {\n"
        "      gl_TessLevelOuter[0] = 2.0;\n"
        "      gl_TessLevelOuter[1] = 1.0;\n"
        "    }\n"
        "  }\n"
        "}\n";

    /* Isoline base points A(-0.7,-0.3) B(0.7,-0.3); C unused.  Quad corners
     * p00 p10 p01 p11.  Second patch (multi-patch stage) uses its own
     * control points shifted +0.9. */
    static const float iso_positions[6] = {
        -0.7f, -0.3f, 0.7f, -0.3f, 0.0f, -0.9f,
    };
    static const float quad_positions[8] = {
        -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f,
    };
    static const float mp_positions[12] = {
        -0.7f, -0.3f, 0.7f, -0.3f, 0.0f, -0.9f,
        -0.7f, -0.3f, 0.7f, -0.3f, 0.0f, -0.9f,
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, cmd_buf = 0u, q = 0u;
    GLuint iso_program =
        link_program_tess_eval_only(vs, iso_tes, fs);
    GLuint quad_program =
        link_program_tess_eval_only(vs, quad_tes, fs);
    GLuint mp_program = link_program_with_tessellation(vs, mp_tcs, iso_tes, fs);
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo || !iso_program || !quad_program || !mp_program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &cmd_buf);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cmd_buf);
    glGenQueries(1, &q);

    /* Stage 1: TES-only instanced isolines, outer {4, 2}.  Two instances,
     * instance 1 shifted +0.6; 8 lines per instance -> 16 primitives. */
    clear_color(0.0f, 0.0f, 0.0f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(iso_positions), iso_positions,
                 GL_STATIC_DRAW);
    glUseProgram(iso_program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    {
        const GLfloat outer[4] = {4.0f, 2.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {1.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glDrawArraysInstanced(GL_PATCHES, 0, 3, 2);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    {
        GLuint written = 0u;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
        if (written != 16u) {
            fprintf(stderr,
                    "air_tessellation_isolines_variants: instanced isoline "
                    "query got %u primitives, expected 16\n", written);
            goto cleanup;
        }
    }
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Segment midpoints: u=0.375 -> x=-0.175 on instance 0, +0.6 on
         * instance 1; v=0.25/0.75 -> y=-0.15/+0.15. */
        static const float probes[4][2] = {
            {-0.175f, -0.15f}, { 0.425f, -0.15f},
            {-0.175f,  0.15f}, { 0.425f,  0.15f},
        };
        for (int i = 0; i < 4; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_variants: instanced "
                        "isoline probe %d not drawn at (%d,%d)\n",
                        i, sx, sy);
                goto cleanup;
            }
        }
    }

    /* Stage 2: TES-only quad point_mode via glDrawArraysIndirect, inner
     * {3, 3} -> 9 points (query count 9). */
    clear_color(0.0f, 0.0f, 0.0f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_positions), quad_positions,
                 GL_STATIC_DRAW);
    glUseProgram(quad_program);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    {
        const GLfloat outer[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {3.0f, 3.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    {
        /* DrawArraysIndirectCommand: {count, primCount, first, baseInstance} */
        const GLuint cmd[4] = {4u, 1u, 0u, 0u};
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmd), cmd, GL_STATIC_DRAW);
    }
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glDrawArraysIndirect(GL_PATCHES, 0);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    {
        GLuint written = 0u;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
        if (written != 9u) {
            fprintf(stderr,
                    "air_tessellation_isolines_variants: indirect quad point "
                    "query got %u primitives, expected 9\n", written);
            goto cleanup;
        }
    }
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        static const float xs[3] = {-1.0f / 3.0f, 0.0f, 1.0f / 3.0f};
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
                const int sx = (int)((xs[i] + 1.0f) * 0.5f * REG_W);
                const int sy = (int)((xs[j] + 1.0f) * 0.5f * REG_H);
                const unsigned char *px = &pixels[(sy * REG_W + sx) * 4];
                if (px[0] > 20u || px[1] < 220u || px[2] > 20u) {
                    fprintf(stderr,
                            "air_tessellation_isolines_variants: indirect "
                            "quad point (%d,%d) expected green, got "
                            "(%u,%u,%u)\n",
                            i, j, px[0], px[1], px[2]);
                    goto cleanup;
                }
            }
        }
    }

    /* Stage 3: TCS multi-patch isolines.  Patch 0: outer {4,2} (16 items,
     * 8 lines); patch 1: outer {2,1} (4 items, 2 lines) whose control
     * points sit at x+0.9 (mirrored base line). */
    clear_color(0.0f, 0.0f, 0.0f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(mp_positions), mp_positions,
                 GL_STATIC_DRAW);
    glUseProgram(mp_program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 6);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Patch 0 lines at y=-0.15: x=-0.175 (segment u=0.375).  Patch 1:
         * 2 segments -> vertices at u=0 and u=0.5; the segment midpoint is
         * u=0.25 -> x = (0.2+0.9) + ... control points at +0.9: A'(-0.7+0.9,
         * -0.3) B'(0.7+0.9, -0.3); u=0.25 -> x = -0.7+0.9+1.4*0.25 = 0.55;
         * y = -0.3 + 0.6*(line+0.5)/1 = -0.15. */
        static const float probes[2][2] = {
            {-0.175f, -0.15f},
            { 0.55f,  -0.15f},
        };
        for (int i = 0; i < 2; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_variants: multi-patch "
                        "isoline probe %d not drawn at (%d,%d)\n",
                        i, sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (q) glDeleteQueries(1, &q);
    if (cmd_buf) glDeleteBuffers(1, &cmd_buf);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (iso_program) glDeleteProgram(iso_program);
    if (quad_program) glDeleteProgram(quad_program);
    if (mp_program) glDeleteProgram(mp_program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_isolines_indexed(unsigned char *pixels,
                                                  const char *out_path)
{
    /* Indexed TES-only isolines through the AIR TES compute kernel's
     * gather path: a shuffled element stream (patch 0 reads gl_in in order
     * C,A,B) exercises the gather mapping, two instances exercise the
     * per-instance sparse-capture offset decomposition, and a restart
     * marker splits the stream into two patches with shared indices. */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position + vec2(float(gl_InstanceID) * 0.6, 0.0),\n"
        "                     0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *iso_tes =
        "#version 450 core\n"
        "layout(isolines, equal_spacing) in;\n"
        "void main() {\n"
        "  vec2 p = mix(gl_in[0].gl_Position.xy, gl_in[1].gl_Position.xy,\n"
        "               gl_TessCoord.x);\n"
        "  p.y += gl_TessCoord.y * 0.6;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "}\n";
    static const float positions[6] = {
        -0.7f, -0.3f, 0.7f, -0.3f, 0.0f, -0.9f, /* A B C */
    };
    /* Patch 0: {2,0,1} -> gl_in = (C,A,B); line 0 spans C->A.
     * Restart (0xFFFFFFFF) then patch 1: {1,0,2} -> gl_in = (B,A,C). */
    static const GLuint indices[8] = {
        2u, 0u, 1u, 0xFFFFFFFFu, 1u, 0u, 2u,
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, ebo = 0u, q = 0u;
    GLuint program = link_program_tess_eval_only(vs, iso_tes, fs);
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);
    glGenQueries(1, &q);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    {
        const GLfloat outer[4] = {4.0f, 2.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {1.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFFFFFFu);
    /* Restart splits 7 indices into two 3-vertex patches; two instances ->
     * 2 patches * 8 lines * 2 instances = 32 primitives. */
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glDrawElementsInstanced(GL_PATCHES, 7, GL_UNSIGNED_INT, 0, 2);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    {
        GLuint written = 0u;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
        if (written != 32u) {
            fprintf(stderr,
                    "air_tessellation_isolines_indexed: query got %u "
                    "primitives, expected 32\n", written);
            goto cleanup;
        }
    }
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Patch 0 line 0: gl_in=(C,A); u=0.375, v=0.25 ->
         * x = 0 + (-0.7)*0.375 = -0.2625 ; y = -0.9+0.225+0.15 = -0.525.
         * Patch 1 line 0: gl_in=(B,A) on instance 0 at x+0.0 and the whole
         * scene shifts +0.6 on instance 1. */
        static const float probes[3][2] = {
            {-0.2625f, -0.525f},
            { 0.3375f, -0.525f},
        };
        for (int i = 0; i < 2; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_indexed: probe %d not "
                        "drawn at (%d,%d)\n", i, sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (q) glDeleteQueries(1, &q);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_isolines_multidraw(unsigned char *pixels,
                                                    const char *out_path)
{
    /* glMultiDrawArrays over GL_PATCHES: each sub-draw is a separate
     * TES-only isolines dispatch with its own control points; the query
     * sums both sub-draws. */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *iso_tes =
        "#version 450 core\n"
        "layout(isolines, equal_spacing) in;\n"
        "void main() {\n"
        "  vec2 p = mix(gl_in[0].gl_Position.xy, gl_in[1].gl_Position.xy,\n"
        "               gl_TessCoord.x);\n"
        "  p.y += gl_TessCoord.y * 0.6;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "}\n";
    static const float positions[12] = {
        -0.7f, -0.3f, 0.7f, -0.3f, 0.0f, -0.9f,   /* patch A */
        0.2f,  -0.3f, 1.6f, -0.3f, 0.9f, -0.9f,   /* patch B */
    };
    static const GLuint ebo_data[6] = {0u, 1u, 2u, 3u, 4u, 5u};
    static const void *ebo_ranges[2] = {(void *)0u, (void *)(3u * 4u)};
    static const GLint firsts[2] = {0, 3};
    static const GLsizei counts[2] = {3, 3};

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, ebo = 0u, q = 0u;
    GLuint program = link_program_tess_eval_only(vs, iso_tes, fs);
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ebo_data), ebo_data,
                 GL_STATIC_DRAW);
    glGenQueries(1, &q);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    {
        const GLfloat outer[4] = {4.0f, 2.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {1.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }

    /* Stage 1: glMultiDrawArrays, two patches -> 16 lines. */
    clear_color(0.0f, 0.0f, 0.0f);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glMultiDrawArrays(GL_PATCHES, firsts, counts, 2);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    {
        GLuint written = 0u;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
        if (written != 16u) {
            fprintf(stderr,
                    "air_tessellation_isolines_multidraw: arrays query got "
                    "%u primitives, expected 16\n", written);
            goto cleanup;
        }
    }
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Line 0 midpoint (u=0.375, v=0.25): patch A
         * x=-0.7+1.4*0.375=-0.175, y=-0.15; patch B starts at +0.9 ->
         * x=-0.175+0.9=0.725. */
        static const float probes[2][2] = {
            {-0.175f, -0.15f},
            { 0.725f, -0.15f},
        };
        for (int i = 0; i < 2; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_multidraw: arrays probe "
                        "%d not drawn at (%d,%d)\n", i, sx, sy);
                goto cleanup;
            }
        }
    }

    /* Stage 2: glMultiDrawElements, same two patches via element ranges. */
    clear_color(0.0f, 0.0f, 0.0f);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glMultiDrawElements(GL_PATCHES, counts, GL_UNSIGNED_INT, ebo_ranges, 2);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    {
        GLuint written = 0u;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
        if (written != 16u) {
            fprintf(stderr,
                    "air_tessellation_isolines_multidraw: elements query got "
                    "%u primitives, expected 16\n", written);
            goto cleanup;
        }
    }
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        static const float probes[2][2] = {
            {-0.175f, -0.15f},
            { 0.725f, -0.15f},
        };
        for (int i = 0; i < 2; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_multidraw: elements probe "
                        "%d not drawn at (%d,%d)\n", i, sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (q) glDeleteQueries(1, &q);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_isolines_rasterdiscard(unsigned char *pixels,
                                                        const char *out_path)
{
    /* P2E rasterizer-discard coverage: with GL_RASTERIZER_DISCARD active the
     * TES compute expansion must still generate primitives (query counts)
     * but the passthrough draw must produce no pixels; disabling the cap
     * restores rasterization. */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *iso_tes =
        "#version 450 core\n"
        "layout(isolines, equal_spacing) in;\n"
        "void main() {\n"
        "  vec2 p = mix(gl_in[0].gl_Position.xy, gl_in[1].gl_Position.xy,\n"
        "               gl_TessCoord.x);\n"
        "  p.y += gl_TessCoord.y * 0.6;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "}\n";
    static const float positions[6] = {
        -0.7f, -0.3f, 0.7f, -0.3f, 0.0f, -0.9f,
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, q = 0u;
    GLuint program = link_program_tess_eval_only(vs, iso_tes, fs);
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenQueries(1, &q);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    {
        const GLfloat outer[4] = {4.0f, 2.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {1.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }

    /* Stage 1: rasterizer discard active.  Primitive query still counts
     * (8 lines), the framebuffer stays black. */
    glEnable(GL_RASTERIZER_DISCARD);
    clear_color(0.0f, 0.0f, 0.0f);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    {
        GLuint written = 0u;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
        if (written != 8u) {
            fprintf(stderr,
                    "air_tessellation_isolines_rasterdiscard: discard query "
                    "got %u primitives, expected 8\n", written);
            goto cleanup;
        }
    }
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    for (int py = 0; py < REG_H; py++) {
        for (int px = 0; px < REG_W; px++) {
            const unsigned char *c = &pixels[(py * REG_W + px) * 4];
            if (c[1] >= 220u) {
                fprintf(stderr,
                        "air_tessellation_isolines_rasterdiscard: discard "
                        "produced pixels at (%d,%d)\n", px, py);
                goto cleanup;
            }
        }
    }

    /* Stage 2: discard disabled again; the same draw must rasterize. */
    glDisable(GL_RASTERIZER_DISCARD);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Line 0 midpoint (u=0.375, v=0.25): x=-0.175, y=-0.15. */
        static const float probes[2][2] = {
            {-0.175f, -0.15f},
            {-0.175f,  0.15f},
        };
        for (int i = 0; i < 2; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_rasterdiscard: probe %d "
                        "not drawn at (%d,%d)\n", i, sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (q) glDeleteQueries(1, &q);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_isolines_tripoint_instanced(
    unsigned char *pixels, const char *out_path)
{
    /* P2E instanced coverage for triangle point-mode: two instances of a
     * triangle patch (inner {2, 2} -> 4 points per patch per instance),
     * shifted by the VS per gl_InstanceID.  Query counts 4 * 2 = 8. */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position + vec2(float(gl_InstanceID) * 0.6, 0.0),\n"
        "                     0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *tri_tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, point_mode) in;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy * gl_TessCoord.x +\n"
        "           gl_in[1].gl_Position.xy * gl_TessCoord.y +\n"
        "           gl_in[2].gl_Position.xy * gl_TessCoord.z;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_PointSize = 8.0;\n"
        "}\n";
    /* Triangle A(-0.6,-0.4) B(0.6,-0.4) C(0,0.7). */
    static const float positions[6] = {
        -0.6f, -0.4f, 0.6f, -0.4f, 0.0f, 0.7f,
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, q = 0u;
    GLuint program = link_program_tess_eval_only(vs, tri_tes, fs);
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo || !program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenQueries(1, &q);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    {
        const GLfloat outer[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {2.0f, 2.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glDrawArraysInstanced(GL_PATCHES, 0, 3, 2);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    {
        GLuint written = 0u;
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
        if (written != 8u) {
            fprintf(stderr,
                    "air_tessellation_isolines_tripoint_instanced: query got "
                    "%u primitives, expected 8\n", written);
            goto cleanup;
        }
    }
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* Points (u,v) = (1/6,1/6),(2/3,1/6),(1/6,2/3),(2/3,2/3) with
         * w=1-u-v mapped via A*u+B*v+C*w: (0,1/3), (-0.3,-13/60),
         * (0.3,-13/60), (0,-23/30); instance 1 shifted +0.6. */
        static const float probes[8][2] = {
            { 0.0f,  1.0f / 3.0f}, { 0.6f,  1.0f / 3.0f},
            {-0.3f, -13.0f / 60.0f}, { 0.3f, -13.0f / 60.0f},
            { 0.3f, -13.0f / 60.0f}, { 0.9f, -13.0f / 60.0f},
            { 0.0f, -23.0f / 30.0f}, { 0.6f, -23.0f / 30.0f},
        };
        for (int i = 0; i < 8; i++) {
            const int sx = (int)((probes[i][0] + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((probes[i][1] + 1.0f) * 0.5f * REG_H);
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                        found = 1;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_tripoint_instanced: probe "
                        "%d not drawn at (%d,%d)\n", i, sx, sy);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (q) glDeleteQueries(1, &q);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_isolines_xfb(unsigned char *pixels,
                                              const char *out_path)
{
    /* P2E transform-feedback coverage for the isolines TES compute path:
     * two patches, outer {4, 2} each.  Per GL 4.6 §11.2.2.3 the primitive
     * generator emits n=outer[0] isolines at v = {0, 1/n, ..., (n-1)/n},
     * each subdivided into m=outer[1] segments; each segment is one line
     * primitive of two vertices, so 2*n*m = 16 expanded vertices per
     * patch.  The kernel writes a complete stage-out record per work item
     * into a temporary stream; the renderer gathers the requested varying
     * into the compact GL XFB buffer.  The test verifies all 32 expected
     * tf_pos values are present, plus
     * PRIMITIVES_GENERATED (n*m*2 = 16 lines) / PRIMITIVES_WRITTEN
     * (32 vertices / 2 = 16 lines) query counts. */
    (void)out_path;
    (void)pixels;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";
    static const char *iso_tes =
        "#version 450 core\n"
        "layout(isolines, equal_spacing) in;\n"
        "layout(location=0) out vec2 tf_pos;\n"
        "void main() {\n"
        "  vec2 p = mix(gl_in[0].gl_Position.xy, gl_in[1].gl_Position.xy,\n"
        "               gl_TessCoord.x);\n"
        "  p.y += gl_TessCoord.y * 0.6;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  tf_pos = p;\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const float positions[12] = {
        -0.7f, -0.3f, 0.7f, -0.3f, 0.0f, -0.9f,   /* patch A */
        0.2f, -0.3f, 1.6f, -0.3f, 0.9f, -0.9f,    /* patch B (+0.9) */
    };
    static const char *tf_varying = "tf_pos";

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, tbo = 0u;
    GLuint program = 0u;
    GLuint gen_q = 0u, wr_q = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;

    program = glCreateProgram();
    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs),
        compile_shader(GL_TESS_EVALUATION_SHADER, iso_tes),
        compile_shader(GL_FRAGMENT_SHADER, fs),
    };
    if (!program || !shaders[0] || !shaders[1] || !shaders[2]) goto cleanup;
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glTransformFeedbackVaryings(program, 1, &tf_varying,
                                GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    {
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(program, sizeof(log), NULL, log);
            fprintf(stderr,
                    "air_tessellation_isolines_xfb: link FAIL: %s\n", log);
            goto cleanup;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenQueries(1, &gen_q);
    glGenQueries(1, &wr_q);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    {
        const GLfloat outer[4] = {4.0f, 2.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {1.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }

    glGenBuffers(1, &tbo);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL,
                 GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo);

    glBeginTransformFeedback(GL_LINES);
    glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, wr_q);
    glDrawArrays(GL_PATCHES, 0, 6);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndTransformFeedback();
    glFinish();
    {
        /* XFB without GL_RASTERIZER_DISCARD must also rasterize: the
         * passthrough draw emits the captured lines to the FBO. */
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const int sx = (int)((0.0f + 1.0f) * 0.5f * REG_W);
        const int sy = (int)((-0.15f + 1.0f) * 0.5f * REG_H);
        int drawn = 0;
        for (int dy = -2; dy <= 2 && !drawn; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                const int px = sx + dx, py = sy + dy;
                if (px < 0 || px >= REG_W || py < 0 || py >= REG_H) continue;
                const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u) {
                    drawn = 1;
                    break;
                }
            }
        }
        if (!drawn) {
            fprintf(stderr,
                    "air_tessellation_isolines_xfb: XFB with rasterization "
                    "enabled drew no lines at (%d,%d)\n", sx, sy);
            goto cleanup;
        }
    }
    {
        GLuint generated = 0u, written = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        glGetQueryObjectuiv(wr_q, GL_QUERY_RESULT, &written);
        if (generated != 16u || written != 16u) {
            fprintf(stderr,
                    "air_tessellation_isolines_xfb: query got generated=%u "
                    "written=%u, expected 16/16\n", generated, written);
            goto cleanup;
        }
    }
    {
        /* Expected records: per patch, 4 isolines at v={0,1/4,1/2,3/4}
         * (y=-0.3,-0.15,0,0.15), each with 2 segments = 4 vertices at
         * u={0,1/2,1/2,1} (x=-0.7,-0.35,-0.35,0 for patch A, +0.9 for
         * patch B) -> 16 records per patch, 32 in total.  Each compact
         * record contains only the requested vec2 tf_pos varying. */
        GLfloat data[4096 / 4];
        GLenum err = GL_NO_ERROR;
        while ((err = glGetError()) != GL_NO_ERROR) { }
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(data),
                           data);
        if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr,
                    "air_tessellation_isolines_xfb: readback FAIL\n");
            goto cleanup;
        }
        for (int r = 0; r < 32; r++) {
            const float tx = data[r * 2 + 0];
            const float ty = data[r * 2 + 1];
            int found = 0;
            for (int p = 0; p < 2 && !found; p++) {
                const float dx = p == 1 ? 0.9f : 0.0f;
                for (int row = 0; row < 4 && !found; row++) {
                    const float yv = (float)row / 4.0f * 0.6f;
                    const float exp_y = -0.3f + yv;
                    for (int seg = 0; seg < 2 && !found; seg++) {
                        for (int vtx = 0; vtx < 2 && !found; vtx++) {
                            const float u =
                                ((float)seg + (float)vtx) / 2.0f;
                            const float ex = -0.7f + 1.4f * u + dx;
                            if (tx - ex > -1e-3f && tx - ex < 1e-3f &&
                                ty - exp_y > -1e-3f &&
                                ty - exp_y < 1e-3f) {
                                found = 1;
                            }
                        }
                    }
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_isolines_xfb: record %d "
                        "tf=(%g,%g) not an expected vertex\n",
                        r, tx, ty);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (wr_q) glDeleteQueries(1, &wr_q);
    if (gen_q) glDeleteQueries(1, &gen_q);
    if (tbo) glDeleteBuffers(1, &tbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_factors_spacing(unsigned char *pixels,
                                                 const char *out_path)
{
    /* GL 4.6 §11.2.2.2 runtime verification of TES layout spacing, winding
     * and zero-outer-factor semantics:
     *  - native triangles: layout(cw) vs layout(ccw) under back-face
     *    culling (ccw front-facing visible, cw culled; the query counts the
     *    generated primitives either way), then outer factor 0 discards the
     *    patch (nothing rasterized);
     *  - point-mode quads: fractional_odd (inner 3 -> 9 points) and
     *    fractional_even (inner 3 -> 16 points) subdivision counts verified
     *    via GL_PRIMITIVES_GENERATED (the CPU-side item accounting uses
     *    mglTessRoundLevelForSpacing; the GPU kernel rounding was verified
     *    during bring-up, see AIR_M3_CPP_TODO item 345);
     *  - point-mode triangles: the same two spacings verified the same way.
     * Only the first two tessellation draws of the test may be raster
     * verified: the 3rd+ tessellation draw (native or point-mode) reads a
     * stale vertex capture (slot 24) -- the pre-existing stale-buffer
     * aliasing bug -- so the zero-outer and spacing segments after it are
     * query-only (the CPU-side query is immune). */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *tesQuadFmt =
        "#version 450 core\n"
        "layout(quads, point_mode, %s) in;\n"
        "void main() {\n"
        "  vec2 p = mix(mix(gl_in[0].gl_Position.xy,\n"
        "                    gl_in[1].gl_Position.xy, gl_TessCoord.x),\n"
        "                mix(gl_in[2].gl_Position.xy,\n"
        "                    gl_in[3].gl_Position.xy, gl_TessCoord.x),\n"
        "                gl_TessCoord.y);\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_PointSize = 8.0;\n"
        "}\n";
    static const char *tesTriFmt =
        "#version 450 core\n"
        "layout(triangles, point_mode, %s) in;\n"
        "void main() {\n"
        "  vec3 t = gl_TessCoord;\n"
        "  vec2 p = t.x * gl_in[0].gl_Position.xy\n"
        "         + t.y * gl_in[1].gl_Position.xy\n"
        "         + t.z * gl_in[2].gl_Position.xy;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_PointSize = 8.0;\n"
        "}\n";
    static const char *tesTriFill =
        "#version 450 core\n"
        "layout(triangles, %s) in;\n"
        "void main() {\n"
        "  vec3 t = gl_TessCoord;\n"
        "  vec2 p = t.x * gl_in[0].gl_Position.xy\n"
        "         + t.y * gl_in[1].gl_Position.xy\n"
        "         + t.z * gl_in[2].gl_Position.xy;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "}\n";
    static const char *spacingNames[3] = {
        "fractional_odd_spacing", "equal_spacing", "fractional_even_spacing",
    };
    /* Raster-verified modes (first two point-mode dispatches). */
    static const int modeIdx[2] = {0, 2}; /* odd, even */
    static const GLuint spacingQuery[2] = {9u, 16u};
    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, q = 0u;
    GLuint quadProg[3] = {0u, 0u, 0u};
    GLuint triProg[3] = {0u, 0u, 0u};
    GLuint fillCCW = 0u, fillCW = 0u;
    int result = 1;
    /* Quad corners then triangle vertices in one VBO. */
    static const float verts[14] = {
        -0.6f, -0.6f, 0.6f, -0.6f, -0.6f, 0.6f, 0.6f, 0.6f, /* quad */
        -0.6f, -0.6f, 0.6f, -0.6f, 0.0f, 0.6f,               /* tri  */
    };

    fbo = make_fbo(REG_W, REG_H, &color);
    for (int i = 0; i < 3; i++) {
        char quadSrc[512], triSrc[512];
        snprintf(quadSrc, sizeof(quadSrc), tesQuadFmt, spacingNames[i]);
        snprintf(triSrc, sizeof(triSrc), tesTriFmt, spacingNames[i]);
        quadProg[i] = link_program_tess_eval_only(vs, quadSrc, fs);
        triProg[i] = link_program_tess_eval_only(vs, triSrc, fs);
        if (!quadProg[i] || !triProg[i]) goto cleanup;
    }
    {
        char fillCCWSrc[512], fillCWSrc[512];
        snprintf(fillCCWSrc, sizeof(fillCCWSrc), tesTriFill, "ccw");
        snprintf(fillCWSrc, sizeof(fillCWSrc), tesTriFill, "cw");
        fillCCW = link_program_tess_eval_only(vs, fillCCWSrc, fs);
        fillCW = link_program_tess_eval_only(vs, fillCWSrc, fs);
    }
    if (!fbo || !fillCCW || !fillCW) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glGenQueries(1, &q);

    /* Segment 1: native triangles, layout(ccw) vs layout(cw) under back-face
     * culling.  The tessellated triangle is CCW in NDC; ccw-front + cull
     * back keeps it visible, cw-front culls it.  The query still counts the
     * generated primitives either way. */
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    {
        const GLfloat outer[4] = {2.0f, 2.0f, 2.0f, 1.0f};
        const GLfloat inner[2] = {2.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    for (int i = 0; i < 2; i++) {
        GLuint prims = 0u;
        clear_color(0.0f, 0.0f, 0.0f);
        glUseProgram(i == 0 ? fillCCW : fillCW);
        glBeginQuery(GL_PRIMITIVES_GENERATED, q);
        glDrawArrays(GL_PATCHES, 4, 3);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glFinish();
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &prims);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        {
            const unsigned char *c = &pixels[(51 * REG_W + 64) * 4];
            if (prims != 4u ||
                (i == 0 && (c[0] > 20u || c[1] < 220u || c[2] > 20u)) ||
                (i == 1 && (c[0] > 20u || c[1] > 20u || c[2] > 20u))) {
                fprintf(stderr,
                        "air_tessellation_factors_spacing: %s cull expected "
                        "4 prims + %s, got %u prims pixel=(%u,%u,%u)\n",
                        i == 0 ? "ccw" : "cw",
                        i == 0 ? "visible" : "culled",
                        prims, c[0], c[1], c[2]);
                goto cleanup;
            }
        }
    }
    glDisable(GL_CULL_FACE);

    /* Segment 2: zero outer factor discards the patch. */
    {
        const GLfloat outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const GLfloat inner[2] = {2.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    /* Segment 2: zero outer factor discards the patch.  The raster of
     * draws 3+ is scrambled by the pre-existing stale-buffer aliasing bug
     * (see the note above), so this segment is query-only: a discarded
     * patch must generate 0 primitives. */
    {
        const GLfloat outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const GLfloat inner[2] = {2.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    {
        GLuint prims = 12345u;
        glUseProgram(fillCCW);
        glBeginQuery(GL_PRIMITIVES_GENERATED, q);
        glDrawArrays(GL_PATCHES, 4, 3);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glFinish();
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &prims);
        if (prims != 0u) {
            fprintf(stderr,
                    "air_tessellation_factors_spacing: zero outer factor "
                    "expected 0 prims, got %u\n", prims);
            goto cleanup;
        }
    }

    /* Segment 3: point-mode quads, inner {3,3}: subdivision count follows
     * spacing (odd 9 / even 16).  Query-only: the CPU-side item accounting
     * (mglAIRTessEvalItemsPerPatch -> mglTessRoundLevelForSpacing) drives
     * the GL_PRIMITIVES_GENERATED query, and the GPU kernel's spacing
     * rounding was independently verified during bring-up (correct cell
     * tesscoords in the stage-out records; see AIR_M3_CPP_TODO item 345).
     * Raster probes are impossible here: the 3rd+ tessellation draw in a
     * row is scrambled by the pre-existing stale-buffer aliasing bug. */
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    {
        const GLfloat outer[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {3.0f, 3.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    for (int mi = 0; mi < 2; mi++) {
        const int i = modeIdx[mi];
        GLuint prims = 0u;
        glUseProgram(quadProg[i]);
        glBeginQuery(GL_PRIMITIVES_GENERATED, q);
        glDrawArrays(GL_PATCHES, 0, 4);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glFinish();
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &prims);
        if (prims != spacingQuery[mi]) {
            fprintf(stderr,
                    "air_tessellation_factors_spacing: quad point_mode %s "
                    "expected %u prims, got %u\n",
                    spacingNames[i], spacingQuery[mi], prims);
            goto cleanup;
        }
    }

    /* Segment 4: point-mode triangles, inner {3}: query-only (odd 9 /
     * even 16), for the same reason as segment 3. */
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    {
        const GLfloat outer[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {3.0f, 1.0f};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    }
    for (int mi = 0; mi < 2; mi++) {
        const int i = modeIdx[mi];
        GLuint prims = 0u;
        glUseProgram(triProg[i]);
        glBeginQuery(GL_PRIMITIVES_GENERATED, q);
        glDrawArrays(GL_PATCHES, 4, 3);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glFinish();
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &prims);
        if (prims != spacingQuery[mi]) {
            fprintf(stderr,
                    "air_tessellation_factors_spacing: tri point_mode %s "
                    "expected %u prims, got %u\n",
                    spacingNames[i], spacingQuery[mi], prims);
            goto cleanup;
        }
    }

    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_tessellation_factors_spacing: GL error\n");
        goto cleanup;
    }

    result = 0;

cleanup:
    if (q) glDeleteQueries(1, &q);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    for (int i = 0; i < 3; i++) {
        if (quadProg[i]) glDeleteProgram(quadProg[i]);
        if (triProg[i]) glDeleteProgram(triProg[i]);
    }
    if (fillCCW) glDeleteProgram(fillCCW);
    if (fillCW) glDeleteProgram(fillCW);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_tessellation_cull_distance(unsigned char *pixels,
                                               const char *out_path)
{
    /* TES-written gl_CullDistance post-tess culling (GL 4.6 §13.6.1) on
     * the AIR TES compute expansion:
     *  - point_mode quad: d[0] = 0.75 - u culls the u > 0.75 cell column
     *    (the rightmost 3x3 column is absent, the rest rasterizes);
     *  - isolines: d[0] = 0.5 - v culls the v > 0.5 isoline rows (the
     *    v = 0.75 row is absent, v <= 0.5 rows rasterize; both endpoints
     *    of a segment share v, so the two-endpoint rule is exercised).
     * The GL_PRIMITIVES_GENERATED query still counts the generated
     * primitives (culling happens after primitive generation).  Only two
     * tessellation draws run (both raster-verified): the 3rd+ tessellation
     * draw reads a stale vertex capture (slot 24) -- the pre-existing
     * stale-buffer aliasing bug documented in
     * air_tessellation_factors_spacing. */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *tesQuad =
        "#version 450 core\n"
        "layout(quads, point_mode, equal_spacing) in;\n"
        "void main() {\n"
        "  vec2 p = mix(mix(gl_in[0].gl_Position.xy,\n"
        "                    gl_in[1].gl_Position.xy, gl_TessCoord.x),\n"
        "                mix(gl_in[2].gl_Position.xy,\n"
        "                    gl_in[3].gl_Position.xy, gl_TessCoord.x),\n"
        "                gl_TessCoord.y);\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_PointSize = 8.0;\n"
        "  gl_CullDistance[0] = 0.75 - gl_TessCoord.x;\n"
        "}\n";
    static const char *tesIso =
        "#version 450 core\n"
        "layout(isolines) in;\n"
        "void main() {\n"
        "  vec2 p = mix(mix(gl_in[0].gl_Position.xy,\n"
        "                    gl_in[1].gl_Position.xy, gl_TessCoord.x),\n"
        "                mix(gl_in[2].gl_Position.xy,\n"
        "                    gl_in[3].gl_Position.xy, gl_TessCoord.x),\n"
        "                gl_TessCoord.y);\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_CullDistance[0] = 0.5 - gl_TessCoord.y;\n"
        "}\n";
    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, q = 0u;
    GLuint quadProg = 0u, isoProg = 0u;
    int result = 1;
    /* Quad corners (-0.6,-0.6) (0.6,-0.6) (-0.6,0.6) (0.6,0.6). */
    static const float verts[8] = {
        -0.6f, -0.6f, 0.6f, -0.6f, -0.6f, 0.6f, 0.6f, 0.6f,
    };

    fbo = make_fbo(REG_W, REG_H, &color);
    quadProg = link_program_tess_eval_only(vs, tesQuad, fs);
    isoProg = link_program_tess_eval_only(vs, tesIso, fs);
    if (!fbo || !quadProg || !isoProg) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glGenQueries(1, &q);
    glPatchParameteri(GL_PATCH_VERTICES, 4);

    /* Draw 1: point-mode quad, inner {3,3} -> 9 points at u,v in
     * {1/6, 3/6, 5/6}.  d[0] = 0.75 - u culls the u = 5/6 column (px 90);
     * the u = 1/6 (px 38) and u = 3/6 (px 64) columns rasterize. */
    {
        const GLfloat outer[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {3.0f, 3.0f};
        GLuint prims = 0u;
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
        clear_color(0.0f, 0.0f, 0.0f);
        glUseProgram(quadProg);
        glBeginQuery(GL_PRIMITIVES_GENERATED, q);
        glDrawArrays(GL_PATCHES, 0, 4);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glFinish();
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &prims);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        if (prims != 9u) {
            fprintf(stderr,
                    "air_tessellation_cull_distance: quad point_mode query "
                    "expected 9 prims, got %u\n", prims);
            goto cleanup;
        }
        /* u = 1/6 column at (38, 38): green; u = 3/6 at (64, 38): green;
         * u = 5/6 at (90, 38): culled -> black. */
        {
            static const float vis[2][2] = {{-0.4f, -0.4f}, {0.0f, -0.4f}};
            for (int i = 0; i < 2; i++) {
                const int sx = (int)((vis[i][0] + 1.0f) * 0.5f * REG_W);
                const int sy = (int)((vis[i][1] + 1.0f) * 0.5f * REG_H);
                int found = 0;
                for (int dy = -2; dy <= 2 && !found; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        const int px = sx + dx, py = sy + dy;
                        if (px < 0 || px >= REG_W || py < 0 || py >= REG_H)
                            continue;
                        const unsigned char *c =
                            &pixels[(py * REG_W + px) * 4];
                        if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u)
                            found = 1;
                    }
                }
                if (!found) {
                    fprintf(stderr,
                            "air_tessellation_cull_distance: quad visible "
                            "point %d missing at (%d,%d)\n", i, sx, sy);
                    goto cleanup;
                }
            }
        }
        {
            const int sx = (int)((0.4f + 1.0f) * 0.5f * REG_W);
            const int sy = (int)((-0.4f + 1.0f) * 0.5f * REG_H);
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = sx + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H)
                        continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] > 20u || c[1] > 20u || c[2] > 20u) {
                        fprintf(stderr,
                                "air_tessellation_cull_distance: quad culled "
                                "point visible at (%d,%d)\n", px, py);
                        goto cleanup;
                    }
                }
            }
        }
    }

    /* Draw 2: isolines, outer {4,2} -> 8 line primitives at v in
     * {0, 1/4, 1/2, 3/4}.  d[0] = 0.5 - v culls the v = 3/4 row (px 83);
     * the v = 1/2 row (px 64) rasterizes. */
    {
        const GLfloat outer[4] = {4.0f, 2.0f, 1.0f, 1.0f};
        const GLfloat inner[2] = {2.0f, 1.0f};
        GLuint prims = 0u;
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
        clear_color(0.0f, 0.0f, 0.0f);
        glUseProgram(isoProg);
        glBeginQuery(GL_PRIMITIVES_GENERATED, q);
        glDrawArrays(GL_PATCHES, 0, 4);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glFinish();
        glGetQueryObjectuiv(q, GL_QUERY_RESULT, &prims);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        if (prims != 8u) {
            fprintf(stderr,
                    "air_tessellation_cull_distance: isolines query expected "
                    "8 prims, got %u\n", prims);
            goto cleanup;
        }
        /* v = 1/2 row at y = 0 (px 64): green through the centre. */
        {
            int found = 0;
            for (int dy = -2; dy <= 2 && !found; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = 64 + dx, py = 64 + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H)
                        continue;
                    const unsigned char *c =
                        &pixels[(py * REG_W + px) * 4];
                    if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u)
                        found = 1;
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_tessellation_cull_distance: isoline v=1/2 row "
                        "missing\n");
                goto cleanup;
            }
        }
        /* v = 3/4 row at y = 0.3 (px 83): culled -> black. */
        {
            const int sy = (int)((0.3f + 1.0f) * 0.5f * REG_H);
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    const int px = 64 + dx, py = sy + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H)
                        continue;
                    const unsigned char *c = &pixels[(py * REG_W + px) * 4];
                    if (c[0] > 20u || c[1] > 20u || c[2] > 20u) {
                        fprintf(stderr,
                                "air_tessellation_cull_distance: culled "
                                "isoline v=3/4 visible at (%d,%d)\n",
                                px, py);
                        goto cleanup;
                    }
                }
            }
        }
    }

    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_tessellation_cull_distance: GL error\n");
        goto cleanup;
    }

    result = 0;

cleanup:
    if (q) glDeleteQueries(1, &q);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (quadProg) glDeleteProgram(quadProg);
    if (isoProg) glDeleteProgram(isoProg);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* TEMP accumulation-bug probe: remove before commit. */
static int test_air_tessellation_accumulation(unsigned char *pixels,
                                              const char *out_path)
{
    /* Accumulation-bug regression: the pre-existing stale vertex-capture
     * bug (documented in AIR_M3_CPP_TODO) deterministically scrambled the
     * 3rd+ consecutive tessellation draw (native or point-mode) and broke
     * air_tessellation_isolines_indexed whenever any extra test ran before
     * the isolines block.  The bug is no longer reproducible (fixed by the
     * spacing/zero-factor and compute-fallback work of 2026-08-14); this
     * test pins the two failure modes:
     *  - a third consecutive point-mode quad draw (n=4 after n=2, n=3)
     *    must rasterize all 16 cells at their correct positions (the
     *    stale-capture failure produced huge gl_in positions -> empty
     *    raster), and
     *  - an interleaved isolines sequence + a fifth/sixth quad draw must
     *    rasterize too (the accumulation counter was global across draws).
     * Registered before the isolines block so the suite-position breakage
     * (any pre-isolines tess test) is covered by the full suite run. */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *tes =
        "#version 450 core\n"
        "layout(quads, point_mode, equal_spacing) in;\n"
        "void main() {\n"
        "  vec2 p = mix(mix(gl_in[0].gl_Position.xy,\n"
        "                    gl_in[1].gl_Position.xy, gl_TessCoord.x),\n"
        "                mix(gl_in[2].gl_Position.xy,\n"
        "                    gl_in[3].gl_Position.xy, gl_TessCoord.x),\n"
        "                gl_TessCoord.y);\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  gl_PointSize = 8.0;\n"
        "}\n";
    static const char *tesIso =
        "#version 450 core\n"
        "layout(isolines) in;\n"
        "void main() {\n"
        "  vec2 p = mix(mix(gl_in[0].gl_Position.xy,\n"
        "                    gl_in[1].gl_Position.xy, gl_TessCoord.x),\n"
        "                mix(gl_in[2].gl_Position.xy,\n"
        "                    gl_in[3].gl_Position.xy, gl_TessCoord.x),\n"
        "                gl_TessCoord.y);\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const float verts[8] = {
        -0.6f, -0.6f, 0.6f, -0.6f, -0.6f, 0.6f, 0.6f, 0.6f,
    };
    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u;
    GLuint quadProg = 0u, isoProg = 0u;
    int result = 1;

    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;
    quadProg = link_program_tess_eval_only(vs, tes, fs);
    isoProg = link_program_tess_eval_only(vs, tesIso, fs);
    if (!quadProg || !isoProg) goto cleanup;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glPatchParameteri(GL_PATCH_VERTICES, 4);

    /* Draws 1-3: quad point-mode n=2, n=3, n=4.  The n=4 draw (3rd
     * consecutive) is the historical failure case. */
    for (int d = 0; d < 3; d++) {
        const GLfloat outer[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const float levels[3] = {2.0f, 3.0f, 4.0f};
        const GLfloat inner[2] = {levels[d], levels[d]};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
        clear_color(0.0f, 0.0f, 0.0f);
        glUseProgram(quadProg);
        glDrawArrays(GL_PATCHES, 0, 4);
        glFinish();
        if (d == 2) {
            glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE,
                         pixels);
            fprintf(stderr, "ACCUM n=4 green map:\n");
            for (int yy = 0; yy < REG_H; yy++) {
                for (int xx = 0; xx < REG_W; xx++) {
                    const unsigned char *cc = &pixels[(yy * REG_W + xx) * 4];
                    if (cc[0] <= 20u && cc[1] >= 200u && cc[2] <= 20u)
                        fprintf(stderr, "  g@(%d,%d)\n", xx, yy);
                }
            }
            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    const float u = ((float)i + 0.5f) / 4.0f;
                    const float v = ((float)j + 0.5f) / 4.0f;
                    const float x = -0.6f + 1.2f * u;
                    const float y = -0.6f + 1.2f * v;
                    const int sx = (int)((x + 1.0f) * 0.5f * REG_W);
                    const int sy = (int)((y + 1.0f) * 0.5f * REG_H);
                    int found = 0;
                    for (int dy = -2; dy <= 2 && !found; dy++) {
                        for (int dx = -2; dx <= 2; dx++) {
                            const int px = sx + dx, py = sy + dy;
                            if (px < 0 || px >= REG_W || py < 0 ||
                                py >= REG_H) continue;
                            const unsigned char *c =
                                &pixels[(py * REG_W + px) * 4];
                            if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u)
                                found = 1;
                        }
                    }
                    if (!found) {
                        fprintf(stderr,
                                "air_tessellation_accumulation: 3rd quad "
                                "draw cell (%d,%d) missing at (%d,%d)\n",
                                i, j, sx, sy);
                        goto cleanup;
                    }
                }
            }
        }
    }

    /* Draws 4-6: isolines outer {4,2}, {3,2}, {4,3}. */
    {
        static const float isoOuter[3][4] = {
            {4.0f, 2.0f, 1.0f, 1.0f}, {3.0f, 2.0f, 1.0f, 1.0f},
            {4.0f, 3.0f, 1.0f, 1.0f},
        };
        static const float rowProbes[4] = {-0.6f, -0.3f, 0.0f, 0.3f};
        for (int d = 0; d < 3; d++) {
            const GLfloat inner[2] = {2.0f, 1.0f};
            glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, isoOuter[d]);
            glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
            clear_color(0.0f, 0.0f, 0.0f);
            glUseProgram(isoProg);
            glDrawArrays(GL_PATCHES, 0, 4);
            glFinish();
            if (d == 2) {
                glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE,
                             pixels);
                for (int r = 0; r < 4; r++) {
                    const int sx = (int)((0.0f + 1.0f) * 0.5f * REG_W);
                    const int sy = (int)((rowProbes[r] + 1.0f) * 0.5f *
                                         REG_H);
                    int found = 0;
                    for (int dy = -2; dy <= 2 && !found; dy++) {
                        for (int dx = -2; dx <= 2; dx++) {
                            const int px = sx + dx, py = sy + dy;
                            if (px < 0 || px >= REG_W || py < 0 ||
                                py >= REG_H) continue;
                            const unsigned char *c =
                                &pixels[(py * REG_W + px) * 4];
                            if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u)
                                found = 1;
                        }
                    }
                    if (!found) {
                        fprintf(stderr,
                                "air_tessellation_accumulation: 3rd isoline "
                                "draw row v=%g missing at (%d,%d)\n",
                                (double)rowProbes[r], sx, sy);
                        goto cleanup;
                    }
                }
            }
        }
    }

    /* Draws 7-8: quad point-mode n=5, n=6 -- the accumulation counter was
     * global across draws; later draws must still rasterize. */
    for (int d = 0; d < 2; d++) {
        const GLfloat outer[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const float levels[2] = {5.0f, 6.0f};
        const GLfloat inner[2] = {levels[d], levels[d]};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
        clear_color(0.0f, 0.0f, 0.0f);
        glUseProgram(quadProg);
        glDrawArrays(GL_PATCHES, 0, 4);
        glFinish();
        if (d == 1) {
            glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE,
                         pixels);
            static const float probes[4][2] = {
                {-0.6f + 1.2f * 1.5f / 6.0f, -0.6f + 1.2f * 1.5f / 6.0f},
                {-0.6f + 1.2f * 4.5f / 6.0f, -0.6f + 1.2f * 2.5f / 6.0f},
                {-0.6f + 1.2f * 2.5f / 6.0f, -0.6f + 1.2f * 4.5f / 6.0f},
                {-0.6f + 1.2f * 5.5f / 6.0f, -0.6f + 1.2f * 5.5f / 6.0f},
            };
            for (int p = 0; p < 4; p++) {
                const int sx = (int)((probes[p][0] + 1.0f) * 0.5f * REG_W);
                const int sy = (int)((probes[p][1] + 1.0f) * 0.5f * REG_H);
                int found = 0;
                for (int dy = -2; dy <= 2 && !found; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        const int px = sx + dx, py = sy + dy;
                        if (px < 0 || px >= REG_W || py < 0 || py >= REG_H)
                            continue;
                        const unsigned char *c =
                            &pixels[(py * REG_W + px) * 4];
                        if (c[0] <= 20u && c[1] >= 220u && c[2] <= 20u)
                            found = 1;
                    }
                }
                if (!found) {
                    fprintf(stderr,
                            "air_tessellation_accumulation: 6th quad draw "
                            "probe %d missing at (%d,%d)\n", p, sx, sy);
                    goto cleanup;
                }
            }
        }
    }

    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "air_tessellation_accumulation: GL error\n");
        goto cleanup;
    }

    result = 0;

cleanup:
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (quadProg) glDeleteProgram(quadProg);
    if (isoProg) glDeleteProgram(isoProg);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}
static int test_air_geometry_xfb(unsigned char *pixels,
                                 const char *out_path)
{
    /* P1 GS transform-feedback coverage (mgl_air_gs_abi.h §5/§5b): points
     * in, triangle_strip out, one triangle per input point.  The GL4
     * ordered path captures compact per-buffer records: only the listed
     * varying (tf_pos, vec2, 2 floats per record) in strict emission
     * order.
     *
     * First segment (2 points): both triangles visible → 6 captured
     * records, queries PRIMITIVES_GENERATED=2 / PRIMITIVES_WRITTEN=2, and
     * without GL_RASTERIZER_DISCARD the triangles must also rasterize.
     *
     * Second segment (3 points): the third point (x < -0.2) emits a
     * fully culled triangle (gl_CullDistance=-1) → GL 4.6 §13.2.4 a
     * culled primitive contributes nothing: queries 3/2, only the two
     * visible triangles occupy the store head, the rest stays zeroed.
     */
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) out vec2 tf_pos;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  float cull = p.x < -0.2f ? -1.0f : 1.0f;\n"
        "  gl_CullDistance[0] = cull;\n"
        "  tf_pos = p;\n"
        "  gl_Position = vec4(p + vec2(-0.15, 0.0), 0.0, 1.0); EmitVertex();\n"
        "  gl_CullDistance[0] = cull;\n"
        "  tf_pos = p;\n"
        "  gl_Position = vec4(p + vec2( 0.15, 0.0), 0.0, 1.0); EmitVertex();\n"
        "  gl_CullDistance[0] = cull;\n"
        "  tf_pos = p;\n"
        "  gl_Position = vec4(p + vec2( 0.0, 0.3), 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *tf_varying = "tf_pos";

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, tbo = 0u;
    GLuint gen_q = 0u, wr_q = 0u, program = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;

    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs),
        compile_shader(GL_GEOMETRY_SHADER, gs),
        compile_shader(GL_FRAGMENT_SHADER, fs),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glTransformFeedbackVaryings(program, 1, &tf_varying,
                                GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    {
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(program, sizeof(log), NULL, log);
            fprintf(stderr, "air_geometry_xfb: link FAIL: %s\n", log);
            goto cleanup;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glGenQueries(1, &gen_q);
    glGenQueries(1, &wr_q);
    glUseProgram(program);
    glGenBuffers(1, &tbo);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL,
                 GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo);

    /* Segment 1: two visible triangles. */
    {
        static const float positions[4] = {
            -0.1f, -0.3f, 0.5f, -0.3f,
        };
        glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
        glBeginTransformFeedback(GL_TRIANGLES);
        glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, wr_q);
        glDrawArrays(GL_POINTS, 0, 2);
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glEndTransformFeedback();
        glFinish();
    }
    {
        /* XFB without GL_RASTERIZER_DISCARD must also rasterize. */
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const int probes[2][2] = {
            { (int)((-0.1f + 1.0f) * 0.5f * REG_W),
              (int)((-0.2f + 1.0f) * 0.5f * REG_H) },
            { (int)(( 0.5f + 1.0f) * 0.5f * REG_W),
              (int)((-0.2f + 1.0f) * 0.5f * REG_H) },
        };
        for (int i = 0; i < 2; i++) {
            const unsigned char *c =
                &pixels[(probes[i][1] * REG_W + probes[i][0]) * 4];
            if (c[0] <= 20u && c[1] <= 20u && c[2] <= 20u) {
                fprintf(stderr,
                        "air_geometry_xfb: triangle %d not green at "
                        "(%d,%d), got (%u,%u,%u)\n",
                        i, probes[i][0], probes[i][1], c[0], c[1], c[2]);
                goto cleanup;
            }
        }
    }
    {
        GLuint generated = 0u, written = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        glGetQueryObjectuiv(wr_q, GL_QUERY_RESULT, &written);
        if (generated != 2u || written != 2u) {
            fprintf(stderr,
                    "air_geometry_xfb: segment 1 query got generated=%u "
                    "written=%u, expected 2/2\n", generated, written);
            goto cleanup;
        }
    }
    {
        /* 2 triangles x 3 vertices, compact per-buffer records (GL 4.6
         * §13.2.4): only the listed varying (tf_pos, vec2) is captured,
         * tightly packed at 2 floats per record in strict emission order. */
        GLfloat data[4096 / 4];
        GLenum err = GL_NO_ERROR;
        while ((err = glGetError()) != GL_NO_ERROR) { }
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(data),
                           data);
        if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr, "air_geometry_xfb: readback FAIL\n");
            goto cleanup;
        }
        static const float expTf[2][2] = { {-0.1f, -0.3f}, {0.5f, -0.3f} };
        for (int r = 0; r < 6; r++) {
            const int tri = r / 3;
            const float tx = data[r * 2 + 0];
            const float ty = data[r * 2 + 1];
            if (!(tx - expTf[tri][0] > -1e-3f &&
                  tx - expTf[tri][0] < 1e-3f &&
                  ty - expTf[tri][1] > -1e-3f &&
                  ty - expTf[tri][1] < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_xfb: record %d tf=(%g,%g), "
                        "expected tri %d at (%g,%g)\n",
                        r, tx, ty, tri, expTf[tri][0], expTf[tri][1]);
                goto cleanup;
            }
        }
    }

    /* Segment 2: three points, the third triangle is fully culled
     * (gl_CullDistance=-1): it must not be captured. */
    {
        static const float positions[6] = {
            0.1f, -0.5f, 0.7f, -0.5f, -0.5f, -0.5f,
        };
        glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                     GL_STATIC_DRAW);
        GLfloat zeros[4096 / 4];
        memset(zeros, 0, sizeof(zeros));
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, zeros,
                     GL_STATIC_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo);
        glBeginTransformFeedback(GL_TRIANGLES);
        glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, wr_q);
        glDrawArrays(GL_POINTS, 0, 3);
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glEndTransformFeedback();
        glFinish();
    }
    {
        GLuint generated = 0u, written = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        glGetQueryObjectuiv(wr_q, GL_QUERY_RESULT, &written);
        if (generated != 3u || written != 2u) {
            fprintf(stderr,
                    "air_geometry_xfb: segment 2 query got generated=%u "
                    "written=%u, expected 3/2\n", generated, written);
            goto cleanup;
        }
    }
    {
        GLfloat seg2[4096 / 4];
        GLenum err = GL_NO_ERROR;
        while ((err = glGetError()) != GL_NO_ERROR) { }
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(seg2),
                           seg2);
        if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr, "air_geometry_xfb: segment 2 readback FAIL\n");
            goto cleanup;
        }
        for (int r = 0; r < 6; r++) {
            const int tri = r / 3;
            const float efx = tri == 0 ? 0.1f : 0.7f;
            const float efy = -0.5f;
            const float tx = seg2[r * 2 + 0];
            const float ty = seg2[r * 2 + 1];
            if (!(tx - efx > -1e-3f && tx - efx < 1e-3f &&
                  ty - efy > -1e-3f && ty - efy < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_xfb: segment 2 record %d "
                        "tf=(%g,%g), expected (%g,%g)\n",
                        r, tx, ty, efx, efy);
                goto cleanup;
            }
        }
        for (int w = 6 * 2; w < 4096 / 4; w++) {
            if (seg2[w] != 0.0f) {
                fprintf(stderr,
                        "air_geometry_xfb: segment 2 culled primitive "
                        "left bytes at float %d (%g)\n", w, seg2[w]);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (wr_q) glDeleteQueries(1, &wr_q);
    if (gen_q) glDeleteQueries(1, &gen_q);
    if (tbo) glDeleteBuffers(1, &tbo);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* Link-time validation for ARB_transform_feedback3 layouts.  This is kept
 * separate from the GS execution tests so unsupported SEPARATE_ATTRIBS
 * capture does not hide validation regressions. */
static int test_air_xfb_link_layout(unsigned char *pixels,
                                    const char *out_path)
{
    (void)pixels;
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "out vec4 x;\n"
        "out vec2 y;\n"
        "void main() {\n"
        "  x = vec4(1.0); y = vec2(0.5);\n"
        "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=2) out;\n"
        "layout(location=0) out vec4 s0_data;\n"
        "layout(stream=1, location=0) out vec4 s1_data;\n"
        "void main() {\n"
        "  s0_data = vec4(1.0);\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  EmitStreamVertex(0); EndStreamPrimitive(0);\n"
        "  s1_data = vec4(2.0);\n"
        "  EmitStreamVertex(1); EndStreamPrimitive(1);\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *interleaved_next[] = {"x", "gl_NextBuffer", "y"};
    static const char *interleaved_skip[] = {"x", "gl_SkipComponents2", "y"};
    static const char *interleaved_next_skip[] = {
        "x", "gl_NextBuffer", "gl_SkipComponents2", "y"
    };
    static const char *separate_varyings[] = {"x", "y"};
    static const char *separate_next[] = {"x", "gl_NextBuffer", "y"};
    static const char *separate_skip[] = {"x", "gl_SkipComponents2", "y"};
    static const char *duplicate[] = {"x", "x"};
    static const char *missing[] = {"does_not_exist"};
    static const char *stream_mismatch[] = {"s0_data", "s1_data"};

    struct XFBLinkCase {
        const char *name;
        const char *vertex;
        const char *geometry;
        GLsizei count;
        const char *const *varyings;
        GLenum mode;
        GLenum api_error;
        GLint link_status;
    } cases[] = {
        {"interleaved_next", vs, NULL, 3, interleaved_next,
         GL_INTERLEAVED_ATTRIBS, GL_NO_ERROR, GL_TRUE},
        {"interleaved_skip", vs, NULL, 3, interleaved_skip,
         GL_INTERLEAVED_ATTRIBS, GL_NO_ERROR, GL_TRUE},
        {"interleaved_next_skip", vs, NULL, 4, interleaved_next_skip,
         GL_INTERLEAVED_ATTRIBS, GL_NO_ERROR, GL_TRUE},
        {"separate_varyings", vs, NULL, 2, separate_varyings,
         GL_SEPARATE_ATTRIBS, GL_NO_ERROR, GL_TRUE},
        {"separate_next", vs, NULL, 3, separate_next,
         GL_SEPARATE_ATTRIBS, GL_INVALID_OPERATION, GL_TRUE},
        {"separate_skip", vs, NULL, 3, separate_skip,
         GL_SEPARATE_ATTRIBS, GL_INVALID_OPERATION, GL_TRUE},
        {"duplicate", vs, NULL, 2, duplicate,
         GL_INTERLEAVED_ATTRIBS, GL_NO_ERROR, GL_FALSE},
        {"missing", vs, NULL, 1, missing,
         GL_INTERLEAVED_ATTRIBS, GL_NO_ERROR, GL_FALSE},
        {"stream_mismatch", vs, gs, 2, stream_mismatch,
         GL_INTERLEAVED_ATTRIBS, GL_NO_ERROR, GL_FALSE},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        GLenum api_error = GL_NO_ERROR;
        GLint link_status = GL_FALSE;
        if (xfb_link_status(cases[i].vertex, cases[i].geometry, fs,
                            cases[i].count,
                            cases[i].varyings, cases[i].mode,
                            &api_error, &link_status) != 0) {
            fprintf(stderr, "air_xfb_link_layout: %s setup failed\n",
                    cases[i].name);
            return 1;
        }
        if (api_error != cases[i].api_error ||
            link_status != cases[i].link_status) {
            fprintf(stderr,
                    "air_xfb_link_layout: %s expected api=0x%x/link=%d, "
                    "got api=0x%x/link=%d\n",
                    cases[i].name, cases[i].api_error, cases[i].link_status,
                    api_error, link_status);
            return 1;
        }
    }
    return 0;
}

/* Transform-feedback reflection API surface (ARB_transform_feedback3):
 * glGetProgramiv(GL_TRANSFORM_FEEDBACK_VARYINGS / BUFFER_MODE /
 * VARYING_MAX_LENGTH) and glGetTransformFeedbackVarying, including the
 * special names gl_NextBuffer / gl_SkipComponentsN (counted and
 * indexable, reporting type NONE with sizes 0 / N) and the
 * default-stream program (no layout(stream) qualifier anywhere, so the
 * whole capture is stream 0). */
static int test_air_xfb_reflection(unsigned char *pixels,
                                   const char *out_path)
{
    (void)pixels;
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "out vec4 x;\n"
        "out vec2 y;\n"
        "void main() {\n"
        "  x = vec4(1.0); y = vec2(0.5);\n"
        "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *interleaved[] = {
        "x", "gl_NextBuffer", "gl_SkipComponents2", "y"
    };
    static const char *separate[] = { "x", "y" };

    GLuint program = 0u, vs_sh = 0u, fs_sh = 0u;
    int result = 1;

    /* Case 1: interleaved with special names. */
    vs_sh = compile_shader(GL_VERTEX_SHADER, vs);
    fs_sh = compile_shader(GL_FRAGMENT_SHADER, fs);
    if (!vs_sh || !fs_sh) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    glAttachShader(program, vs_sh);
    glAttachShader(program, fs_sh);
    glTransformFeedbackVaryings(program, 4, interleaved,
                                GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(program);
    glDeleteShader(vs_sh); vs_sh = 0u;
    glDeleteShader(fs_sh); fs_sh = 0u;
    {
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) goto cleanup;
    }
    {
        static const struct {
            const char *name;
            GLint size;
            GLenum type;
        } exp[4] = {
            {"x", 1, GL_FLOAT_VEC4},
            {"gl_NextBuffer", 0, GL_NONE},
            {"gl_SkipComponents2", 2, GL_NONE},
            {"y", 1, GL_FLOAT_VEC2},
        };
        GLint count = 0, mode = 0, maxLen = 0;
        glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_VARYINGS, &count);
        glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &mode);
        glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH,
                       &maxLen);
        if (count != 4 || mode != GL_INTERLEAVED_ATTRIBS || maxLen != 19) {
            fprintf(stderr,
                    "air_xfb_reflection: programiv got count=%d "
                    "mode=0x%x maxLen=%d, expected 4/0x8c8c/19\n",
                    count, mode, maxLen);
            goto cleanup;
        }
        for (GLuint i = 0; i < 4; i++) {
            char name[64] = {0};
            GLsizei length = -1, size = -1;
            GLenum type = 0;
            glGetTransformFeedbackVarying(program, i, sizeof(name),
                                          &length, &size, &type, name);
            if (strcmp(name, exp[i].name) != 0 ||
                length != (GLsizei)strlen(exp[i].name) ||
                size != exp[i].size || type != exp[i].type) {
                fprintf(stderr,
                        "air_xfb_reflection: varying %u got "
                        "\"%s\"/%d/0x%x, expected \"%s\"/%d/0x%x\n",
                        i, name, size, type,
                        exp[i].name, exp[i].size, exp[i].type);
                goto cleanup;
            }
        }
        /* index past the end -> GL_INVALID_VALUE, outputs unmodified */
        while (glGetError() != GL_NO_ERROR) { }
        {
            char name[8] = {'z', 0};
            GLsizei length = 77;
            glGetTransformFeedbackVarying(program, 4, sizeof(name),
                                          &length, NULL, NULL, name);
            if (glGetError() != GL_INVALID_VALUE || length != 77 ||
                name[0] != 'z') {
                fprintf(stderr,
                        "air_xfb_reflection: overflow index did not raise "
                        "INVALID_VALUE cleanly\n");
                goto cleanup;
            }
        }
        /* REFERENCED_BY_* (GL 4.6 §7.3.1): "x"/"y" are written by the VS
         * only; the FS does not consume them. */
        {
            static const GLenum props[3] = {
                GL_REFERENCED_BY_VERTEX_SHADER,
                GL_REFERENCED_BY_GEOMETRY_SHADER,
                GL_REFERENCED_BY_FRAGMENT_SHADER,
            };
            GLint vals[3] = {-1, -1, -1};
            glGetProgramResourceiv(program, GL_TRANSFORM_FEEDBACK_VARYING,
                                   0, 3, props, 3, NULL, vals);
            if (vals[0] != GL_TRUE || vals[1] != GL_FALSE ||
                vals[2] != GL_FALSE) {
                fprintf(stderr,
                        "air_xfb_reflection: referenced_by \"x\" got "
                        "%d/%d/%d, expected TRUE/FALSE/FALSE\n",
                        vals[0], vals[1], vals[2]);
                goto cleanup;
            }
        }
    }
    glDeleteProgram(program); program = 0u;

    /* Case 2: SEPARATE mode reported back. */
    vs_sh = compile_shader(GL_VERTEX_SHADER, vs);
    fs_sh = compile_shader(GL_FRAGMENT_SHADER, fs);
    if (!vs_sh || !fs_sh) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    glAttachShader(program, vs_sh);
    glAttachShader(program, fs_sh);
    glTransformFeedbackVaryings(program, 2, separate, GL_SEPARATE_ATTRIBS);
    glLinkProgram(program);
    glDeleteShader(vs_sh); vs_sh = 0u;
    glDeleteShader(fs_sh); fs_sh = 0u;
    {
        GLint ok = 0, count = 0, mode = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) goto cleanup;
        glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_VARYINGS, &count);
        glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &mode);
        if (count != 2 || mode != GL_SEPARATE_ATTRIBS) {
            fprintf(stderr,
                    "air_xfb_reflection: separate got count=%d mode=0x%x, "
                    "expected 2/0x8c8d\n", count, mode);
            goto cleanup;
        }
    }
    glDeleteProgram(program); program = 0u;

    /* Case 3: program with no transform feedback -> spec defaults. */
    vs_sh = compile_shader(GL_VERTEX_SHADER, vs);
    fs_sh = compile_shader(GL_FRAGMENT_SHADER, fs);
    if (!vs_sh || !fs_sh) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    glAttachShader(program, vs_sh);
    glAttachShader(program, fs_sh);
    glLinkProgram(program);
    glDeleteShader(vs_sh); vs_sh = 0u;
    glDeleteShader(fs_sh); fs_sh = 0u;
    {
        GLint ok = 0, count = -1, mode = -1, maxLen = -1;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) goto cleanup;
        glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_VARYINGS, &count);
        glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &mode);
        glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH,
                       &maxLen);
        if (count != 0 || mode != GL_INTERLEAVED_ATTRIBS || maxLen != 0) {
            fprintf(stderr,
                    "air_xfb_reflection: defaults got %d/0x%x/%d, "
                    "expected 0/0x8c8c/0\n", count, mode, maxLen);
            goto cleanup;
        }
        /* Unlinked program -> INVALID_OPERATION on both queries. */
        {
            GLuint raw = glCreateProgram();
            GLint scratch = 0;
            while (glGetError() != GL_NO_ERROR) { }
            glGetProgramiv(raw, GL_TRANSFORM_FEEDBACK_VARYINGS, &scratch);
            GLenum e1 = glGetError();
            glGetTransformFeedbackVarying(raw, 0, 0, NULL, NULL, NULL, NULL);
            GLenum e2 = glGetError();
            glDeleteProgram(raw);
            if (e1 != GL_INVALID_OPERATION ||
                e2 != GL_INVALID_OPERATION) {
                fprintf(stderr,
                        "air_xfb_reflection: unlinked errors 0x%x/0x%x, "
                        "expected INVALID_OPERATION twice\n", e1, e2);
                goto cleanup;
            }
        }
    }

    result = 0;

    /* Case 4: GS writes the captured varying with NO stream qualifier
     * (default stream 0); the FS consumes it.  REFERENCED_BY_VERTEX is
     * FALSE (the VS outputs only v_g), GEOMETRY TRUE, FRAGMENT TRUE. */
    {
        static const char *vs4 =
            "#version 450 core\n"
            "layout(location=0) out vec2 v_g;\n"
            "void main() {\n"
            "  v_g = vec2(0.5);\n"
            "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        static const char *gs4 =
            "#version 450 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "layout(location=0) in vec2 v_g[];\n"
            "layout(location=0) out vec2 g_out;\n"
            "void main() {\n"
            "  g_out = v_g[0];\n"
            "  gl_Position = gl_in[0].gl_Position;\n"
            "  EmitVertex();\n"
            "  EndPrimitive();\n"
            "}\n";
        static const char *fs4 =
            "#version 450 core\n"
            "layout(location=0) in vec2 g_out;\n"
            "layout(location=0) out vec4 frag;\n"
            "void main() { frag = vec4(g_out, 0.0, 1.0); }\n";
        static const char *vary4[] = { "g_out" };
        GLuint shaders4[3] = {
            compile_shader(GL_VERTEX_SHADER, vs4),
            compile_shader(GL_GEOMETRY_SHADER, gs4),
            compile_shader(GL_FRAGMENT_SHADER, fs4),
        };
        if (!shaders4[0] || !shaders4[1] || !shaders4[2]) goto case4_done;
        program = glCreateProgram();
        if (!program) goto case4_done;
        for (int i = 0; i < 3; i++) glAttachShader(program, shaders4[i]);
        glTransformFeedbackVaryings(program, 1, vary4,
                                    GL_INTERLEAVED_ATTRIBS);
        glLinkProgram(program);
        for (int i = 0; i < 3; i++) glDeleteShader(shaders4[i]);
        {
            GLint ok = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &ok);
            if (ok) {
                static const GLenum props[3] = {
                    GL_REFERENCED_BY_VERTEX_SHADER,
                    GL_REFERENCED_BY_GEOMETRY_SHADER,
                    GL_REFERENCED_BY_FRAGMENT_SHADER,
                };
                GLint vals[3] = {-1, -1, -1};
                glGetProgramResourceiv(program,
                                       GL_TRANSFORM_FEEDBACK_VARYING,
                                       0, 3, props, 3, NULL, vals);
                if (vals[0] != GL_FALSE || vals[1] != GL_TRUE ||
                    vals[2] != GL_TRUE) {
                    fprintf(stderr,
                            "air_xfb_reflection: GS referenced_by got "
                            "%d/%d/%d, expected FALSE/TRUE/TRUE\n",
                            vals[0], vals[1], vals[2]);
                    result = 1;
                }
            } else {
                fprintf(stderr,
                        "air_xfb_reflection: case 4 link failed\n");
                result = 1;
            }
        }
        glDeleteProgram(program);
        program = 0u;
    }
case4_done:
    if (result != 0) goto cleanup;

cleanup:
    if (program) glDeleteProgram(program);
    if (vs_sh) glDeleteShader(vs_sh);
    if (fs_sh) glDeleteShader(fs_sh);
    return result;
}

/* GS multi-stream transform feedback (P1, GL 4.6 §11.1.3.4): points in,
 * points out, two streams.  Stream 0 rasterizes and captures to XFB
 * buffer 0; stream 1 captures to XFB buffer 1 (no rasterization).
 *
 * Each input point emits one vertex to stream 0 (s0_data = p) and one to
 * stream 1 (s1_data = p + (0.5, 0)).  After drawing 3 points:
 *   - 3 green points rasterized (stream 0 only)
 *   - PRIMITIVES_GENERATED = 3 (stream 0 only, GL 4.6 §13.2.4)
 *   - non-indexed TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 3 (stream 0)
 *   - XFB buffer 0: 3 full stage-out records (80B each, stride 20 floats)
 *   - XFB buffer 1: 3 compact records (32B each, stride 8 floats:
 *     position + s1_data)
 *
 * XFB record ordering is validated order-agnostically: Metal compute
 * does not guarantee thread-group execution order, and the GS kernel
 * uses an atomic cursor to reserve XFB space, so records may appear in
 * any order.  GL 4.6 mandates primitive-order preservation; a prefix-sum
 * dispatch would be needed to satisfy that strictly and is left as a
 * future task. */
static int test_air_geometry_multi_stream_xfb(unsigned char *pixels,
                                              const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=2) out;\n"
        "layout(location=0) out vec2 s0_data;\n"
        "layout(stream=1, location=0) out vec2 s1_data;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  s0_data = p;\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  EmitStreamVertex(0);\n"
        "  EndStreamPrimitive(0);\n"
        "  s1_data = p + vec2(0.5, 0.0);\n"
        "  gl_Position = vec4(p + vec2(0.5, 0.0), 0.0, 1.0);\n"
        "  EmitStreamVertex(1);\n"
        "  EndStreamPrimitive(1);\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec2 s0_data;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *varyings[] = {
        "s0_data", "gl_NextBuffer", "s1_data"
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u;
    GLuint tbo0 = 0u, tbo1 = 0u;
    GLuint gen_q = 0u, wr_q = 0u;
    GLuint gen_q1 = 0u, wr_q1 = 0u, gen_no_xfb_q1 = 0u;
    GLuint program = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;

    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs),
        compile_shader(GL_GEOMETRY_SHADER, gs),
        compile_shader(GL_FRAGMENT_SHADER, fs),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glTransformFeedbackVaryings(program, 3, varyings,
                                GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    {
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(program, sizeof(log), NULL, log);
            fprintf(stderr,
                    "air_geometry_multi_stream_xfb: link FAIL: %s\n", log);
            goto cleanup;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glGenQueries(1, &gen_q);
    glGenQueries(1, &wr_q);
    glGenQueries(1, &gen_q1);
    glGenQueries(1, &wr_q1);
    glGenQueries(1, &gen_no_xfb_q1);
    glUseProgram(program);

    glGenBuffers(1, &tbo0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo0);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo0);

    glGenBuffers(1, &tbo1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo1);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, tbo1);

    /* Draw 3 points. */
    {
        static const float positions[6] = {
            -0.5f, -0.3f, 0.0f, -0.3f, 0.5f, -0.3f,
        };
        glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
        glBeginTransformFeedback(GL_POINTS);
        glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, wr_q);
        glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, 1u, gen_q1);
        glBeginQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, 1u,
                            wr_q1);
        glDrawArrays(GL_POINTS, 0, 3);
        glEndQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, 1u);
        glEndQueryIndexed(GL_PRIMITIVES_GENERATED, 1u);
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glEndTransformFeedback();
        glFinish();
    }

    /* Indexed PRIMITIVES_GENERATED is independent of transform-feedback
     * capture.  Exercise stream 1 again with XFB inactive so the query cannot
     * be satisfied accidentally from the written-byte counter. */
    glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, 1u, gen_no_xfb_q1);
    glDrawArrays(GL_POINTS, 0, 3);
    glEndQueryIndexed(GL_PRIMITIVES_GENERATED, 1u);
    glFinish();

    /* Verify rasterization (stream 0 only). */
    {
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        const int probes[3][2] = {
            { (int)((-0.5f + 1.0f) * 0.5f * REG_W),
              (int)((-0.3f + 1.0f) * 0.5f * REG_H) },
            { (int)(( 0.0f + 1.0f) * 0.5f * REG_W),
              (int)((-0.3f + 1.0f) * 0.5f * REG_H) },
            { (int)(( 0.5f + 1.0f) * 0.5f * REG_W),
              (int)((-0.3f + 1.0f) * 0.5f * REG_H) },
        };
        for (int i = 0; i < 3; i++) {
            /* Point rasterization may land on either of two adjacent
             * pixels depending on the rounding convention, so check a
             * 2-pixel neighborhood around the expected center. */
            int found = 0;
            for (int dy = 0; dy <= 1 && !found; dy++) {
                for (int dx = -1; dx <= 1 && !found; dx++) {
                    int px = probes[i][0] + dx;
                    int py = probes[i][1] + dy;
                    if (px < 0 || px >= REG_W || py < 0 || py >= REG_H)
                        continue;
                    const unsigned char *c =
                        &pixels[(py * REG_W + px) * 4];
                    if (c[1] >= 180u) found = 1;
                }
            }
            if (!found) {
                fprintf(stderr,
                        "air_geometry_multi_stream_xfb: stream 0 point %d "
                        "not green near (%d,%d)\n",
                        i, probes[i][0], probes[i][1]);
                goto cleanup;
            }
        }
    }

    /* Verify stream 0 and indexed stream 1 queries (GL 4.6 §13.2.4):
     *   PRIMITIVES_GENERATED = stream 0 primitives only (rasterizer-bound) = 3
     *   TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = stream 0 primitives = 3.
     *   Indexed stream 1 has three emitted points and three XFB records. */
    {
        GLuint generated = 0u, written = 0u;
        GLuint generated1 = 0u, written1 = 0u, generatedNoXfb1 = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        glGetQueryObjectuiv(wr_q, GL_QUERY_RESULT, &written);
        glGetQueryObjectuiv(gen_q1, GL_QUERY_RESULT, &generated1);
        glGetQueryObjectuiv(wr_q1, GL_QUERY_RESULT, &written1);
        glGetQueryObjectuiv(gen_no_xfb_q1, GL_QUERY_RESULT,
                            &generatedNoXfb1);
        if (generated != 3u || written != 3u ||
            generated1 != 3u || written1 != 3u || generatedNoXfb1 != 3u) {
            fprintf(stderr,
                    "air_geometry_multi_stream_xfb: query got generated=%u "
                    "written=%u indexed=%u/%u no_xfb=%u, expected 3/3, "
                    "3/3 and 3\n",
                    generated, written, generated1, written1,
                    generatedNoXfb1);
            goto cleanup;
        }
    }

    /* Stream 0 XFB: full stage-out records (80B = 20 floats per record).
     * position at float 0..3, s0_data at float 16..17. */
    {
        GLfloat data[4096 / 4];
        GLenum err = GL_NO_ERROR;
        while ((err = glGetError()) != GL_NO_ERROR) { }
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo0);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(data),
                           data);
        if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr,
                    "air_geometry_multi_stream_xfb: stream 0 readback FAIL\n");
            goto cleanup;
        }
        /* GL4 ordered XFB (mgl_air_gs_abi.h §5b): compact per-buffer
         * records in strict emission order — the pass-2 scatter writes
         * records of work item w before w+1.  Each record is exactly the
         * captured varying (s0_data, vec2): 2 floats. */
        static const float expPos[3][2] = {
            { -0.5f, -0.3f }, { 0.0f, -0.3f }, { 0.5f, -0.3f },
        };
        for (int r = 0; r < 3; r++) {
            float sx = data[r * 2 + 0];
            float sy = data[r * 2 + 1];
            if (!(sx - expPos[r][0] > -1e-3f &&
                  sx - expPos[r][0] < 1e-3f &&
                  sy - expPos[r][1] > -1e-3f &&
                  sy - expPos[r][1] < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_multi_stream_xfb: stream 0 record %d "
                        "s0_data=(%g,%g), expected (%g,%g)\n",
                        r, sx, sy, expPos[r][0], expPos[r][1]);
                goto cleanup;
            }
        }
    }

    /* Stream 1 XFB: compact records (s1_data, vec2 = 2 floats) in strict
     * emission order.  s1_data = p + (0.5, 0). */
    {
        GLfloat data[4096 / 4];
        GLenum err = GL_NO_ERROR;
        while ((err = glGetError()) != GL_NO_ERROR) { }
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo1);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(data),
                           data);
        if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr,
                    "air_geometry_multi_stream_xfb: stream 1 readback FAIL\n");
            goto cleanup;
        }
        static const float expPos[3][2] = {
            { 0.0f, -0.3f }, { 0.5f, -0.3f }, { 1.0f, -0.3f },
        };
        for (int r = 0; r < 3; r++) {
            float sx = data[r * 2 + 0];
            float sy = data[r * 2 + 1];
            if (!(sx - expPos[r][0] > -1e-3f &&
                  sx - expPos[r][0] < 1e-3f &&
                  sy - expPos[r][1] > -1e-3f &&
                  sy - expPos[r][1] < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_multi_stream_xfb: stream 1 record %d "
                        "s1_data=(%g,%g), expected (%g,%g)\n",
                        r, sx, sy, expPos[r][0], expPos[r][1]);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (wr_q) glDeleteQueries(1, &wr_q);
    if (gen_q) glDeleteQueries(1, &gen_q);
    if (wr_q1) glDeleteQueries(1, &wr_q1);
    if (gen_q1) glDeleteQueries(1, &gen_q1);
    if (gen_no_xfb_q1) glDeleteQueries(1, &gen_no_xfb_q1);
    if (tbo1) glDeleteBuffers(1, &tbo1);
    if (tbo0) glDeleteBuffers(1, &tbo0);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* GS gl_Layer / gl_ViewportIndex output (P1): a points-in/triangle-strip-out
 * GS writes gl_Layer=1 into a whole-level layered framebuffer, or
 * gl_ViewportIndex=1 into a regular framebuffer.  The expanded triangle must
 * land on framebuffer layer 1 / the second viewport respectively. */
/* GL4 whole-primitive atomic truncation (mgl_air_gs_abi.h §5b): a
 * primitive lands only if it fits in EVERY buffer it feeds (GL 4.6
 * §13.2.4).  GS emits one triangle per input point; tf_a -> buffer 0,
 * tf_b -> buffer 1 (gl_NextBuffer), so each primitive needs 24 bytes in
 * both buffers (3 vertices x vec2).
 * Segment 1: buffer 1 holds exactly one primitive while buffer 0 could
 * hold two -> only the first primitive may land in EITHER buffer
 * (cross-buffer atomicity), and PRIMITIVES_WRITTEN counts it once.
 * Segment 2: both buffers hold 1.5 primitives -> exactly one whole
 * primitive lands; the half-fitting tail must leave zero bytes. */
static int test_air_geometry_xfb_truncate(unsigned char *pixels,
                                          const char *out_path)
{
    (void)pixels;
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) out vec2 tf_a;\n"
        "layout(location=1) out vec2 tf_b;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    tf_a = p;\n"
        "    tf_b = p + vec2(0.1, 0.1);\n"
        "    gl_Position = vec4(p, 0.0, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *varyings[] = { "tf_a", "gl_NextBuffer", "tf_b" };
    static const float positions[6] = {
        -0.7f, -0.3f, 0.0f, -0.3f, 0.7f, -0.3f,
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u;
    GLuint tbo0 = 0u, tbo1 = 0u, gen_q = 0u, wr_q = 0u, program = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;

    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs),
        compile_shader(GL_GEOMETRY_SHADER, gs),
        compile_shader(GL_FRAGMENT_SHADER, fs),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glTransformFeedbackVaryings(program, 3, varyings,
                                GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    {
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(program, sizeof(log), NULL, log);
            fprintf(stderr, "air_geometry_xfb_truncate: link FAIL: %s\n",
                    log);
            goto cleanup;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenQueries(1, &gen_q);
    glGenQueries(1, &wr_q);
    glUseProgram(program);
    glEnable(GL_RASTERIZER_DISCARD);
    glGenBuffers(1, &tbo0);
    glGenBuffers(1, &tbo1);

    /* Segment 1: buffer 0 = 2 primitives, buffer 1 = 1 primitive. */
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo0);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 48, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo1);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 24, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, tbo1);
    glBeginTransformFeedback(GL_TRIANGLES);
    glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, wr_q);
    glDrawArrays(GL_POINTS, 0, 3);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndTransformFeedback();
    glFinish();
    {
        GLuint generated = 0u, written = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        glGetQueryObjectuiv(wr_q, GL_QUERY_RESULT, &written);
        if (generated != 3u || written != 1u) {
            fprintf(stderr,
                    "air_geometry_xfb_truncate: segment 1 query got "
                    "generated=%u written=%u, expected 3/1\n",
                    generated, written);
            goto cleanup;
        }
    }
    {
        /* Cross-buffer atomicity: buffer 0 had room for 2 primitives but
         * must hold only the first because buffer 1 is full. */
        GLfloat d0[12] = {0}, d1[6] = {0};
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo0);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(d0), d0);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo1);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(d1), d1);
        for (int r = 0; r < 3; r++) {
            if (!(d0[r * 2] - -0.7f > -1e-3f && d0[r * 2] - -0.7f < 1e-3f &&
                  d0[r * 2 + 1] - -0.3f > -1e-3f &&
                  d0[r * 2 + 1] - -0.3f < 1e-3f &&
                  d1[r * 2] - -0.6f > -1e-3f && d1[r * 2] - -0.6f < 1e-3f &&
                  d1[r * 2 + 1] - -0.2f > -1e-3f &&
                  d1[r * 2 + 1] - -0.2f < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_xfb_truncate: segment 1 record %d "
                        "a=(%g,%g) b=(%g,%g)\n",
                        r, d0[r * 2], d0[r * 2 + 1],
                        d1[r * 2], d1[r * 2 + 1]);
                goto cleanup;
            }
        }
        for (int w = 6; w < 12; w++) {
            if (d0[w] != 0.0f) {
                fprintf(stderr,
                        "air_geometry_xfb_truncate: segment 1 torn "
                        "primitive at buffer 0 float %d (%g)\n",
                        w, d0[w]);
                goto cleanup;
            }
        }
    }

    /* Segment 2: both buffers = 1.5 primitives -> one whole primitive. */
    {
        GLfloat zeros[9] = {0};
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo0);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 36, zeros,
                     GL_STATIC_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo0);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo1);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 36, zeros,
                     GL_STATIC_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, tbo1);
        glBeginTransformFeedback(GL_TRIANGLES);
        glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, wr_q);
        glDrawArrays(GL_POINTS, 0, 3);
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glEndTransformFeedback();
        glFinish();
    }
    {
        GLuint generated = 0u, written = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        glGetQueryObjectuiv(wr_q, GL_QUERY_RESULT, &written);
        if (generated != 3u || written != 1u) {
            fprintf(stderr,
                    "air_geometry_xfb_truncate: segment 2 query got "
                    "generated=%u written=%u, expected 3/1\n",
                    generated, written);
            goto cleanup;
        }
    }
    {
        GLfloat d0[9] = {0}, d1[9] = {0};
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo0);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(d0), d0);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo1);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(d1), d1);
        for (int r = 0; r < 3; r++) {
            if (!(d0[r * 2] - -0.7f > -1e-3f && d0[r * 2] - -0.7f < 1e-3f &&
                  d1[r * 2] - -0.6f > -1e-3f && d1[r * 2] - -0.6f < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_xfb_truncate: segment 2 record %d "
                        "a=(%g,%g) b=(%g,%g)\n",
                        r, d0[r * 2], d0[r * 2 + 1],
                        d1[r * 2], d1[r * 2 + 1]);
                goto cleanup;
            }
        }
        for (int w = 6; w < 9; w++) {
            if (d0[w] != 0.0f || d1[w] != 0.0f) {
                fprintf(stderr,
                        "air_geometry_xfb_truncate: segment 2 torn "
                        "primitive at float %d (%g/%g)\n",
                        w, d0[w], d1[w]);
                goto cleanup;
            }
        }
        for (int r = 0; r < 3; r++) {
            if (!(d0[r * 2 + 1] - -0.3f > -1e-3f &&
                  d0[r * 2 + 1] - -0.3f < 1e-3f &&
                  d1[r * 2 + 1] - -0.2f > -1e-3f &&
                  d1[r * 2 + 1] - -0.2f < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_xfb_truncate: segment 2 record %d "
                        "a=(%g,%g) b=(%g,%g)\n",
                        r, d0[r * 2], d0[r * 2 + 1],
                        d1[r * 2], d1[r * 2 + 1]);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    glDisable(GL_RASTERIZER_DISCARD);
    if (wr_q) glDeleteQueries(1, &wr_q);
    if (gen_q) glDeleteQueries(1, &gen_q);
    if (tbo1) glDeleteBuffers(1, &tbo1);
    if (tbo0) glDeleteBuffers(1, &tbo0);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* GL_SEPARATE_ATTRIBS execution coverage (mgl_air_gs_abi.h §5b): each
 * varying captures to its own buffer (varying i -> buffer i), packed at
 * offset 0.  This exercises the same ordered per-buffer scatter as the
 * interleaved path with one varying per buffer.  3 points in, one point
 * out each; records must land in strict emission order. */
static int test_air_geometry_separate_xfb(unsigned char *pixels,
                                          const char *out_path)
{
    (void)pixels;
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "layout(location=0) out vec2 sep_a;\n"
        "layout(location=1) out vec4 sep_b;\n"
        "void main() {\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  sep_a = p;\n"
        "  sep_b = vec4(p, 0.25, 0.75);\n"
        "  gl_Position = vec4(p, 0.0, 1.0);\n"
        "  EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *varyings[] = { "sep_a", "sep_b" };
    static const float positions[6] = {
        -0.6f, -0.2f, 0.1f, 0.4f, 0.8f, -0.5f,
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u;
    GLuint tbo0 = 0u, tbo1 = 0u, gen_q = 0u, wr_q = 0u, program = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;

    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs),
        compile_shader(GL_GEOMETRY_SHADER, gs),
        compile_shader(GL_FRAGMENT_SHADER, fs),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glTransformFeedbackVaryings(program, 2, varyings,
                                GL_SEPARATE_ATTRIBS);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    {
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(program, sizeof(log), NULL, log);
            fprintf(stderr, "air_geometry_separate_xfb: link FAIL: %s\n",
                    log);
            goto cleanup;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenQueries(1, &gen_q);
    glGenQueries(1, &wr_q);
    glUseProgram(program);
    glEnable(GL_RASTERIZER_DISCARD);

    glGenBuffers(1, &tbo0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo0);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 64, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo0);
    glGenBuffers(1, &tbo1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo1);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 128, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, tbo1);

    glBeginTransformFeedback(GL_POINTS);
    glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, wr_q);
    glDrawArrays(GL_POINTS, 0, 3);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndTransformFeedback();
    glFinish();
    {
        GLuint generated = 0u, written = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        glGetQueryObjectuiv(wr_q, GL_QUERY_RESULT, &written);
        if (generated != 3u || written != 3u) {
            fprintf(stderr,
                    "air_geometry_separate_xfb: query got generated=%u "
                    "written=%u, expected 3/3\n", generated, written);
            goto cleanup;
        }
    }
    {
        GLfloat d0[6] = {0}, d1[12] = {0};
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo0);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(d0), d0);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo1);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(d1), d1);
        for (int r = 0; r < 3; r++) {
            const float px = positions[r * 2], py = positions[r * 2 + 1];
            if (!(d0[r * 2] - px > -1e-3f && d0[r * 2] - px < 1e-3f &&
                  d0[r * 2 + 1] - py > -1e-3f &&
                  d0[r * 2 + 1] - py < 1e-3f &&
                  d1[r * 4] - px > -1e-3f && d1[r * 4] - px < 1e-3f &&
                  d1[r * 4 + 1] - py > -1e-3f &&
                  d1[r * 4 + 1] - py < 1e-3f &&
                  d1[r * 4 + 2] - 0.25f > -1e-3f &&
                  d1[r * 4 + 2] - 0.25f < 1e-3f &&
                  d1[r * 4 + 3] - 0.75f > -1e-3f &&
                  d1[r * 4 + 3] - 0.75f < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_separate_xfb: record %d a=(%g,%g) "
                        "b=(%g,%g,%g,%g), expected p=(%g,%g)\n",
                        r, d0[r * 2], d0[r * 2 + 1],
                        d1[r * 4], d1[r * 4 + 1], d1[r * 4 + 2],
                        d1[r * 4 + 3], px, py);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    glDisable(GL_RASTERIZER_DISCARD);
    if (wr_q) glDeleteQueries(1, &wr_q);
    if (gen_q) glDeleteQueries(1, &gen_q);
    if (tbo1) glDeleteBuffers(1, &tbo1);
    if (tbo0) glDeleteBuffers(1, &tbo0);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* Passthrough GS + XFB: a geometry shader that re-emits gl_in unchanged
 * still runs the GS compute expansion (no source-string bypass).  With
 * transform feedback active the expansion must capture the forwarded
 * varying.  One triangle in, three captured records. */
static int test_air_geometry_passthrough_xfb(unsigned char *pixels,
                                             const char *out_path)
{
    (void)pixels;
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=0) out vec2 v_data;\n"
        "void main() {\n"
        "  v_data = position + vec2(0.25, 0.25);\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) in vec2 v_data[];\n"
        "layout(location=0) out vec2 g_data;\n"
        "void main() {\n"
        "  for (int n_vertex_index = 0; n_vertex_index < 3;\n"
        "       n_vertex_index++) {\n"
        "    g_data = v_data[n_vertex_index];\n"
        "    gl_Position = gl_in[n_vertex_index].gl_Position;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const char *tf_varying = "g_data";
    static const float positions[6] = {
        -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f,
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, tbo = 0u;
    GLuint gen_q = 0u, wr_q = 0u, program = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;

    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs),
        compile_shader(GL_GEOMETRY_SHADER, gs),
        compile_shader(GL_FRAGMENT_SHADER, fs),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glTransformFeedbackVaryings(program, 1, &tf_varying,
                                GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    {
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(program, sizeof(log), NULL, log);
            fprintf(stderr,
                    "air_geometry_passthrough_xfb: link FAIL: %s\n", log);
            goto cleanup;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenQueries(1, &gen_q);
    glGenQueries(1, &wr_q);
    glUseProgram(program);
    glEnable(GL_RASTERIZER_DISCARD);

    glGenBuffers(1, &tbo);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 256, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo);

    glBeginTransformFeedback(GL_TRIANGLES);
    glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, wr_q);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndTransformFeedback();
    glFinish();
    {
        GLuint generated = 0u, written = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        glGetQueryObjectuiv(wr_q, GL_QUERY_RESULT, &written);
        if (generated != 1u || written != 1u) {
            fprintf(stderr,
                    "air_geometry_passthrough_xfb: query got generated=%u "
                    "written=%u, expected 1/1\n", generated, written);
            goto cleanup;
        }
    }
    {
        GLfloat data[6] = {0};
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(data),
                           data);
        if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr,
                    "air_geometry_passthrough_xfb: readback FAIL\n");
            goto cleanup;
        }
        for (int r = 0; r < 3; r++) {
            const float ex = positions[r * 2] + 0.25f;
            const float ey = positions[r * 2 + 1] + 0.25f;
            if (!(data[r * 2] - ex > -1e-3f && data[r * 2] - ex < 1e-3f &&
                  data[r * 2 + 1] - ey > -1e-3f &&
                  data[r * 2 + 1] - ey < 1e-3f)) {
                fprintf(stderr,
                        "air_geometry_passthrough_xfb: record %d "
                        "g_data=(%g,%g), expected (%g,%g)\n",
                        r, data[r * 2], data[r * 2 + 1], ex, ey);
                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    glDisable(GL_RASTERIZER_DISCARD);
    if (wr_q) glDeleteQueries(1, &wr_q);
    if (gen_q) glDeleteQueries(1, &gen_q);
    if (tbo) glDeleteBuffers(1, &tbo);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* A Minecraft/CTS-style passthrough GS (re-emits gl_in unchanged) must
 * still execute: GEOMETRY_SHADER_INVOCATIONS / PRIMITIVES_EMITTED count
 * the invocation and the emitted triangle. */
static int test_air_geometry_passthrough_queries(unsigned char *pixels,
                                                 const char *out_path)
{
    (void)pixels;
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "void main() {\n"
        "  for (int n_vertex_index = 0; n_vertex_index < 3;\n"
        "       n_vertex_index++) {\n"
        "    gl_Position = gl_in[n_vertex_index].gl_Position;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    static const float positions[6] = {
        -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f,
    };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u;
    GLuint inv_q = 0u, prim_q = 0u, program = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;
    program = link_program_with_geometry(vs, gs, fs);
    if (!program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenQueries(1, &inv_q);
    glGenQueries(1, &prim_q);
    glUseProgram(program);
    clear_color(0.0f, 0.0f, 0.0f);
    glBeginQuery(GL_GEOMETRY_SHADER_INVOCATIONS, inv_q);
    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, prim_q);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glEndQuery(GL_GEOMETRY_SHADER_INVOCATIONS);
    glFinish();
    {
        GLuint invocations = 0u, primitives = 0u;
        glGetQueryObjectuiv(inv_q, GL_QUERY_RESULT, &invocations);
        glGetQueryObjectuiv(prim_q, GL_QUERY_RESULT, &primitives);
        if (invocations != 1u || primitives != 1u) {
            fprintf(stderr,
                    "air_geometry_passthrough_queries: invocations=%u "
                    "primitives=%u; expected 1/1\n",
                    invocations, primitives);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    if (inv_q) glDeleteQueries(1, &inv_q);
    if (prim_q) glDeleteQueries(1, &prim_q);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* Layered GS repro (KHR-GL46.geometry_shader.layered_rendering shape):
 * points in, triangle_strip out, uniform-gated gl_Layer writes with
 * gl_Layer read-back into a flat int varying, one point emitting 24
 * vertices across 6 strip primitives.  Verifies the expansion emits the
 * full vertex count and the flat int varying reaches the FS through the
 * passthrough stage. */
static int test_air_geometry_layered_repro(unsigned char *pixels,
                                           const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=0) out vec2 v_pos;\n"
        "void main() { v_pos = position; gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=96) out;\n"
        "layout(location=0) in vec2 v_pos[];\n"
        "layout(location=0) flat out int layer_id;\n"
        "uniform int provoking_vertex_index;\n"
        "void main() {\n"
        "  for (int n = 0; n < 6; ++n) {\n"
        "    if (provoking_vertex_index == 0 ||\n"
        "        provoking_vertex_index == 1) gl_Layer = n;\n"
        "    layer_id = gl_Layer;\n"
        "    gl_Position = vec4(v_pos[0], 0.0, 1.0);\n"
        "    EmitVertex();\n"
        "    layer_id = gl_Layer;\n"
        "    gl_Position = vec4(v_pos[0] + 0.1, 0.0, 1.0);\n"
        "    EmitVertex();\n"
        "    layer_id = gl_Layer;\n"
        "    gl_Position = vec4(v_pos[0] + 0.2, 0.0, 1.0);\n"
        "    EmitVertex();\n"
        "    layer_id = gl_Layer;\n"
        "    gl_Position = vec4(v_pos[0] + 0.3, 0.0, 1.0);\n"
        "    EmitVertex();\n"
        "    EndPrimitive();\n"
        "  }\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) flat in int layer_id;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(float(layer_id) / 5.0, 1.0, 0.0, 1.0); }\n";
    static const float positions[2] = { 0.0f, 0.5f };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, program = 0u;
    GLuint gen_q = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;

    GLuint shaders[3] = {
        compile_shader(GL_VERTEX_SHADER, vs),
        compile_shader(GL_GEOMETRY_SHADER, gs),
        compile_shader(GL_FRAGMENT_SHADER, fs),
    };
    if (!shaders[0] || !shaders[1] || !shaders[2]) goto cleanup;
    program = glCreateProgram();
    if (!program) goto cleanup;
    for (int i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (int i = 0; i < 3; i++) glDeleteShader(shaders[i]);
    {
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(program, sizeof(log), NULL, log);
            fprintf(stderr, "air_geometry_layered_repro: link FAIL: %s\n",
                    log);
            goto cleanup;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenQueries(1, &gen_q);
    glUseProgram(program);
    {
        GLint pvi = glGetUniformLocation(program, "provoking_vertex_index");
        glUniform1i(pvi, 0);
    }
    glBeginQuery(GL_PRIMITIVES_GENERATED, gen_q);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    {
        GLuint generated = 0u;
        glGetQueryObjectuiv(gen_q, GL_QUERY_RESULT, &generated);
        /* 6 iterations x 1 strip of 4 vertices = 12 list triangles;
         * PRIMITIVES_GENERATED counts GS-emitted primitives including
         * any culled ones (GL 4.6), from the kernel-side counter. */
        if (generated != 12u) {
            fprintf(stderr,
                    "air_geometry_layered_repro: generated=%u, "
                    "expected 12\n", generated);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (gen_q) glDeleteQueries(1, &gen_q);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

static int test_air_geometry_layer_viewport(unsigned char *pixels,
                                            const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) out vec3 v_color;\n"
        "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); v_color = vec3(0.0, 1.0, 0.0); }\n";
    static const char *gs_layer =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) in vec3 v_color[];\n"
        "layout(location=0) out vec3 g_color;\n"
        "void main() {\n"
        "  gl_Layer = 1;\n"
        "  g_color = v_color[0];\n"
        "  gl_Position = vec4(-0.5, -0.5, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 0.1, -0.5, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.2,  0.3, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *gs_viewport =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) in vec3 v_color[];\n"
        "layout(location=0) out vec3 g_color;\n"
        "void main() {\n"
        "  gl_ViewportIndex = 1;\n"
        "  g_color = v_color[0];\n"
        "  gl_Position = vec4(-0.5, -0.5, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 0.1, -0.5, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.2,  0.3, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec3 g_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(g_color, 1.0); }\n";
    static const float position[2] = {-0.5f, -0.5f};
    static const float centroid[2] = {-0.2f, -0.2333f};
    GLuint color = 0u;
    GLuint vao = 0u, vbo = 0u;
    GLuint lfbo = 0u, vfbo = 0u;
    GLuint layerProgram = 0u, viewportProgram = 0u;
    int result = 1;

    lfbo = make_layer_fbo(REG_W, REG_H, &color);
    if (!lfbo) goto cleanup;
    glBindFramebuffer(GL_FRAMEBUFFER, lfbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, 0u);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "air_geometry_layer_viewport: whole-level probe FBO incomplete\n");
        goto cleanup;
    }
    /* Isolation probe: a plain VS writing gl_Layer=1 (no GS) must also land
     * on layer 1; this splits backend [[layer]] output from the GS chain. */
    {
        static const char *vs_layer =
            "#version 450 core\n"
            "layout(location=0) in vec2 position;\n"
            "layout(location=0) out vec3 g_color;\n"
            "void main() { gl_Position = vec4(position, 0.0, 1.0);\n"
            "  g_color = vec3(0.0, 1.0, 0.0); gl_Layer = 1; }\n";
        GLuint probeProgram = link_program(vs_layer, fs);
        if (!probeProgram) goto cleanup;
        static const float tri[6] = {
            -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f,
        };
        GLuint pvao = 0u, pvbo = 0u;
        glGenVertexArrays(1, &pvao);
        glBindVertexArray(pvao);
        glGenBuffers(1, &pvbo);
        glBindBuffer(GL_ARRAY_BUFFER, pvbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
        glBindFramebuffer(GL_FRAMEBUFFER, lfbo);
        glUseProgram(probeProgram);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0, 0);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        {
            int cx = REG_W / 2, cy = REG_H / 2;
            const unsigned char *pp = &pixels[(cy * REG_W + cx) * 4];
            if (pp[1] >= 180) {
                fprintf(stderr,
                        "air_geometry_layer_viewport: probe layer 0 got "
                        "(%d,%d,%d); expected untouched\n", pp[0], pp[1], pp[2]);
                goto cleanup;
            }
        }
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0, 1);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        {
            int cx = REG_W / 2, cy = REG_H / 2;
            const unsigned char *pp = &pixels[(cy * REG_W + cx) * 4];
            if (pp[1] < 180) {
                fprintf(stderr,
                        "air_geometry_layer_viewport: probe layer 1 center got "
                        "(%d,%d,%d); expected green\n", pp[0], pp[1], pp[2]);
            goto cleanup;
            }
        }
        /* Writing only gl_Layer must not alias onto gl_ViewportIndex.
         * Viewport 1 is the left half; viewport 0 is the full target.
         * NDC origin under viewport 0 is the framebuffer center. */
        glViewport(0, 0, REG_W, REG_H);
        glViewportIndexedf(1, 0.0f, 0.0f, (GLfloat)REG_W / 2.0f,
                           (GLfloat)REG_H);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0);
        clear_color(0.0f, 0.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0, 1);
        glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        {
            int cx = REG_W / 2, cy = REG_H / 2;
            const unsigned char *pp = &pixels[(cy * REG_W + cx) * 4];
            if (pp[1] < 180) {
                fprintf(stderr,
                        "air_geometry_layer_viewport: layer-only VS used "
                        "viewport 1 (center (%d,%d,%d)); expected viewport 0\n",
                        pp[0], pp[1], pp[2]);
                goto cleanup;
            }
        }
        glViewport(0, 0, REG_W, REG_H);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0u);

    /* Scene 1: restore the whole-level layered attachment after the probe's
     * per-layer readback bindings.  gl_Layer=1 must land the triangle on
     * framebuffer layer 1 and leave layer 0 untouched. */
    layerProgram = link_program_with_geometry(vs, gs_layer, fs);
    if (!layerProgram) goto cleanup;
    glBindFramebuffer(GL_FRAMEBUFFER, lfbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "air_geometry_layer_viewport: layered GS FBO incomplete\n");
        goto cleanup;
    }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glUseProgram(layerProgram);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_POINTS, 0, 1);
    glFinish();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0, 0);
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        int cx = (int)((centroid[0] + 1.0) * 0.5 * REG_W);
        int cy = (int)((centroid[1] + 1.0) * 0.5 * REG_H);
        const unsigned char *pp = &pixels[(cy * REG_W + cx) * 4];
        if (pp[1] >= 180 || pp[0] > 40 || pp[2] > 40) {
            fprintf(stderr,
                    "air_geometry_layer_viewport: layer 0 got color "
                    "(%d,%d,%d) at centroid; expected untouched\n",
                    pp[0], pp[1], pp[2]);
            goto cleanup;
        }
    }
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0, 1);
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        int cx = (int)((centroid[0] + 1.0) * 0.5 * REG_W);
        int cy = (int)((centroid[1] + 1.0) * 0.5 * REG_H);
        const unsigned char *pp = &pixels[(cy * REG_W + cx) * 4];
        if (pp[1] < 180 || pp[0] > 40 || pp[2] > 40) {
            fprintf(stderr,
                    "air_geometry_layer_viewport: layer 1 centroid got "
                    "(%d,%d,%d); expected green triangle (slice 1 draw)\n",
                    pp[0], pp[1], pp[2]);
            goto cleanup;
        }
    }

    /* Scene 1b: per-layer readback above changed the attachment back to a
     * single slice.  Reattaching the whole level must restore layered
     * rendering for the next draw. */
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr,
                "air_geometry_layer_viewport: layered GS repeat FBO incomplete\n");
        goto cleanup;
    }
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_POINTS, 0, 1);
    glFinish();
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0, 0);
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        int cx = (int)((centroid[0] + 1.0) * 0.5 * REG_W);
        int cy = (int)((centroid[1] + 1.0) * 0.5 * REG_H);
        const unsigned char *pp = &pixels[(cy * REG_W + cx) * 4];
        if (pp[1] >= 180 || pp[0] > 40 || pp[2] > 40) {
            fprintf(stderr,
                    "air_geometry_layer_viewport: layer 0 got color "
                    "(%d,%d,%d) at centroid; expected untouched (1b)\n",
                    pp[0], pp[1], pp[2]);
            goto cleanup;
        }
    }
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0, 1);
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        int cx = (int)((centroid[0] + 1.0) * 0.5 * REG_W);
        int cy = (int)((centroid[1] + 1.0) * 0.5 * REG_H);
        const unsigned char *pp = &pixels[(cy * REG_W + cx) * 4];
        if (pp[1] < 180 || pp[0] > 40 || pp[2] > 40) {
            fprintf(stderr,
                    "air_geometry_layer_viewport: layer 1 centroid got "
                    "(%d,%d,%d); expected green triangle (1b)\n",
                    pp[0], pp[1], pp[2]);
            goto cleanup;
        }
    }

    viewportProgram = link_program_with_geometry(vs, gs_viewport, fs);
    vfbo = make_fbo(REG_W, REG_H, &color);
    if (!vfbo || !viewportProgram) goto cleanup;
    glBindFramebuffer(GL_FRAMEBUFFER, vfbo);
    glUseProgram(viewportProgram);
    glViewport(0, 0, REG_W, REG_H);
    glViewportIndexedf(1, 0.0f, 0.0f, (GLfloat)REG_W / 2.0f,
                       (GLfloat)REG_H);
    clear_color(0.0f, 0.0f, 0.0f);
    glDrawArrays(GL_POINTS, 0, 1);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        /* NDC -> viewport 1 (left half, full height). */
        int cx = (int)((centroid[0] + 1.0) * 0.5 * (REG_W / 2));
        int cy = (int)((centroid[1] + 1.0) * 0.5 * REG_H);
        const unsigned char *pp = &pixels[(cy * REG_W + cx) * 4];
        if (pp[1] < 180 || pp[0] > 40 || pp[2] > 40) {
            fprintf(stderr,
                    "air_geometry_layer_viewport: viewport-1 centroid got "
                    "(%d,%d,%d); expected green in left half\n",
                    pp[0], pp[1], pp[2]);
            goto cleanup;
        }
        /* Mirror position in the right half (viewport 0) must be empty. */
        int rx = (int)((centroid[0] + 1.0) * 0.5 * (REG_W / 2)) + REG_W / 2;
        const unsigned char *rp = &pixels[(cy * REG_W + rx) * 4];
        if (rp[1] >= 180) {
            fprintf(stderr,
                    "air_geometry_layer_viewport: viewport-0 mirror got "
                    "(%d,%d,%d); expected empty\n", rp[0], rp[1], rp[2]);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (layerProgram) glDeleteProgram(layerProgram);
    if (viewportProgram) glDeleteProgram(viewportProgram);
    if (lfbo) glDeleteFramebuffers(1, &lfbo);
    if (vfbo) glDeleteFramebuffers(1, &vfbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* KHR-GL46.geometry_shader.rendering.points_input_points_output replica:
 * one input point, GS emits a 3x3 grid of points via gl_Position offsets
 * (no EndPrimitive; points topology ends at implicit primitive end). */
static int test_air_geometry_points_grid(unsigned char *pixels,
                                         const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 vs_gs_color[1];\n"
        "void main() { vs_gs_color[0] = color;\n"
        "  gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=9) out;\n"
        "in vec3 vs_gs_color[1];\n"
        "out vec3 gs_fs_color;\n"
        "uniform ivec2 renderingTargetSize;\n"
        "void main() {\n"
        "  float dx = 2.0 / float(renderingTargetSize.x);\n"
        "  float dy = 2.0 / float(renderingTargetSize.y);\n"
        "  for (int i = -1; i <= 1; ++i)\n"
        "    for (int j = -1; j <= 1; ++j) {\n"
        "      gs_fs_color = vs_gs_color[0];\n"
        "      gl_Position = gl_in[0].gl_Position\n"
        "        + vec4(i * dx, j * dy, 0, 0);\n"
        "      EmitVertex();\n"
        "    }\n"
        "}\n";
    /* Ablation variants: strip varyings / single point / no gl_in read. */
    static const char *gs_novary =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=9) out;\n"
        "void main() {\n"
        "  for (int i = -1; i <= 1; ++i)\n"
        "    for (int j = -1; j <= 1; ++j) {\n"
        "      gl_Position = gl_in[0].gl_Position\n"
        "        + vec4(float(i) * 0.1, float(j) * 0.1, 0.0, 1.0);\n"
        "      EmitVertex();\n"
        "    }\n"
        "}\n";
    static const char *fs_const =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(1.0, 0.0, 0.0, 1.0); }\n";
    static const char *vs_pass =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec3 gs_fs_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(gs_fs_color, 1.0); }\n";
    static const float positions[2] = { 0.0f, 0.5f };
    static const float colors[12] = { 0.25f, 0.5f, 0.75f, 0.25f, 0.5f, 0.75f,
                                      0.25f, 0.5f, 0.75f, 0.25f, 0.5f, 0.75f };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, vbo_c = 0u, program = 0u;
    int result = 1;
    int use_r32i_fbo = 0;
    if (getenv("MGL_PG_INTOUT") || getenv("MGL_PG_INTOUT_VSFS")) {
        fbo = make_fbo_r32i(REG_W, REG_H, &color);
    } else if (getenv("MGL_PG_CTSLINES")) {
        /* CTS runs in a 256x256 surface with a 45x45 viewport */
        fbo = make_fbo(256, 256, &color);
    } else {
        fbo = make_fbo(REG_W, REG_H, &color);
    }
    if (!fbo) goto cleanup;
    if (getenv("MGL_PG_NOVARY")) {
        program = link_program_with_geometry(vs_pass, gs_novary, fs_const);
    } else if (getenv("MGL_PG_ONEPT")) {
        program = link_program_with_geometry(vs_pass,
            "#version 450 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "void main() {\n"
            "  gl_Position = gl_in[0].gl_Position;\n"
            "  EmitVertex();\n"
            "}\n", fs_const);
    } else if (getenv("MGL_PG_FLATVEC")) {
        /* ONEPT + one flat vec4 varying, float FS output */
        program = link_program_with_geometry(
            "#version 450 core\n"
            "layout(location=0) in vec2 position;\n"
            "out vec4 v0;\n"
            "void main() { v0 = vec4(3.0, 0.0, 0.0, 0.0);\n"
            "  gl_Position = vec4(position, 0.0, 1.0); }\n",
            "#version 450 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "in vec4 v0[1];\n"
            "out vec4 w0;\n"
            "void main() {\n"
            "  gl_Position = gl_in[0].gl_Position;\n"
            "  w0 = v0[0];\n"
            "  EmitVertex();\n}\n",
            "#version 450 core\n"
            "flat in vec4 w0;\n"
            "layout(location=0) out vec4 frag;\n"
            "void main() { frag = vec4(w0.x, 0.0, 0.0, 1.0); }\n");
    } else if (getenv("MGL_PG_INTOUT")) {
        /* ONEPT + int FS output on an R32I attachment */
        program = link_program_with_geometry(
            "#version 450 core\n"
            "layout(location=0) in vec2 position;\n"
            "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n",
            "#version 450 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "void main() {\n"
            "  gl_Position = gl_in[0].gl_Position;\n"
            "  EmitVertex();\n}\n",
            "#version 450 core\n"
            "layout(location=0) out int fs_out;\n"
            "void main() { fs_out = 7; }\n");
    } else if (getenv("MGL_PG_MANYIV")) {
        /* ONEPT + N flat ivec4 varyings + int FS output */
        static char gsv[8192], fsv[8192], vsv[512];
        int nv = atoi(getenv("MGL_PG_MANYIV"));
        snprintf(vsv, sizeof(vsv),
                 "#version 450 core\n"
                 "layout(location=0) in vec2 position;\n"
                 "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n");
        snprintf(gsv, sizeof(gsv),
                 "#version 450 core\n"
                 "layout(points) in;\n"
                 "layout(points, max_vertices=1) out;\n");
        for (int i = 0; i < nv; i++)
            snprintf(gsv + strlen(gsv), sizeof(gsv) - strlen(gsv),
                     "flat out ivec4 v%d;\n", i);
        snprintf(gsv + strlen(gsv), sizeof(gsv) - strlen(gsv),
                 "void main() {\n"
                 "  gl_Position = gl_in[0].gl_Position;\n");
        for (int i = 0; i < nv; i++)
            snprintf(gsv + strlen(gsv), sizeof(gsv) - strlen(gsv),
                     "  v%d = ivec4(%d, 0, 0, 0);\n", i, i + 1);
        snprintf(gsv + strlen(gsv), sizeof(gsv) - strlen(gsv),
                 "  EmitVertex();\n}\n");
        snprintf(fsv, sizeof(fsv),
                 "#version 450 core\n");
        for (int i = 0; i < nv; i++)
            snprintf(fsv + strlen(fsv), sizeof(fsv) - strlen(fsv),
                     "flat in ivec4 v%d;\n", i);
        snprintf(fsv + strlen(fsv), sizeof(fsv) - strlen(fsv),
                 "layout(location=0) out int fs_out;\n"
                 "void main() { int sum = 0;\n");
        for (int i = 0; i < nv; i++)
            snprintf(fsv + strlen(fsv), sizeof(fsv) - strlen(fsv),
                     "  sum += v%d.x;\n", i);
        snprintf(fsv + strlen(fsv), sizeof(fsv) - strlen(fsv),
                 "  fs_out = sum;\n}\n");
        program = link_program_with_geometry(vsv, gsv, fsv);
    } else if (getenv("MGL_PG_CTSLINES")) {
        /* CTS lines_input_line_strip_output_line_strip_drawcall shape */
        program = link_program_with_geometry(
            "#version 450 core\n"
            "layout(location=0) in vec4 position;\n"
            "uniform ivec2 renderingTargetSize;\n"
            "out vec4 vs_gs_color[1];\n"
            "void main() {\n"
            "    gl_Position = position;\n"
            "    switch (gl_VertexID) {\n"
            "        case 0:\n"
            "        case 4: vs_gs_color[0] = vec4(1, 0, 0, 0); break;\n"
            "        case 1: vs_gs_color[0] = vec4(0, 1, 0, 0); break;\n"
            "        case 2: vs_gs_color[0] = vec4(0, 0, 1, 0); break;\n"
            "        case 3: vs_gs_color[0] = vec4(0, 0, 0, 1); break;\n"
            "        default: vs_gs_color[0] = vec4(0.0); break;\n"
            "    }\n"
            "}\n",
            "#version 450 core\n"
            "layout(lines) in;\n"
            "layout(line_strip, max_vertices=6) out;\n"
            "in vec4 vs_gs_color[2];\n"
            "out vec4 gs_fs_color;\n"
            "uniform ivec2 renderingTargetSize;\n"
            "void main() {\n"
            "    float dx = 2.0 / float(renderingTargetSize.x);\n"
            "    float dy = 2.0 / float(renderingTargetSize.y);\n"
            "    vec4 start_pos = gl_in[0].gl_Position;\n"
            "    vec4 end_pos   = gl_in[1].gl_Position;\n"
            "    vec4 mid_col   = mix(vs_gs_color[0], vs_gs_color[1], 0.5);\n"
            "    if (start_pos.x != end_pos.x) {\n"
            "        gl_Position = vec4(-1.0, start_pos.y + dy, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex();\n"
            "        gl_Position = vec4(1.0, end_pos.y + dy, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex(); EndPrimitive();\n"
            "        gl_Position = vec4(-1.0, start_pos.y, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex();\n"
            "        gl_Position = vec4(1.0, end_pos.y, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex(); EndPrimitive();\n"
            "        gl_Position = vec4(-1.0, start_pos.y - dy, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex();\n"
            "        gl_Position = vec4(1.0, end_pos.y - dy, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex(); EndPrimitive();\n"
            "    } else {\n"
            "        gl_Position = vec4(start_pos.x - dx, start_pos.y, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex();\n"
            "        gl_Position = vec4(end_pos.x - dx, end_pos.y, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex(); EndPrimitive();\n"
            "        gl_Position = vec4(start_pos.x, start_pos.y, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex();\n"
            "        gl_Position = vec4(end_pos.x, end_pos.y, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex(); EndPrimitive();\n"
            "        gl_Position = vec4(start_pos.x + dx, start_pos.y, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex();\n"
            "        gl_Position = vec4(end_pos.x + dx, end_pos.y, 0, 1);\n"
            "        gs_fs_color = mid_col; EmitVertex(); EndPrimitive();\n"
            "    }\n"
            "}\n",
            "#version 450 core\n"
            "layout(location = 0) in vec4 gs_fs_color;\n"
            "layout(location=0) out vec4 frag;\n"
                        "void main() { frag = vec4(1.0, 0.0, 0.0, 1.0); }\n");
    } else if (getenv("MGL_PG_NOATTR")) {
        /* same as ONEPT but the VS reads gl_VertexID, no attribute */
        program = link_program_with_geometry(
            "#version 450 core\n"
            "void main() {\n"
            "  gl_Position = vec4(float(gl_VertexID), 0.5, 0.0, 1.0);\n"
            "}\n",
            "#version 450 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "void main() {\n"
            "  gl_Position = gl_in[0].gl_Position;\n"
            "  EmitVertex();\n"
            "}\n", fs_const);
    } else if (getenv("MGL_PG_DECLONLY")) {
        program = link_program_with_geometry(
            "#version 450 core\n"
            "layout(location=0) in vec2 position;\n"
            "void main() {\n"
            "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "}\n",
            "#version 450 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "void main() {\n"
            "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "  EmitVertex();\n"
            "}\n", fs_const);
    } else if (getenv("MGL_PG_CONST")) {
        program = link_program_with_geometry(vs_pass,
            "#version 450 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "void main() {\n"
            "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "  EmitVertex();\n"
            "}\n", fs_const);
    } else {
        program = link_program_with_geometry(vs, gs, fs);
    }
    if (!program) goto cleanup;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    if (!program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    if (getenv("MGL_PG_CTSLINES")) {
        float cts_pos[8 * 4];
        for (int n = 0; n < 8; n++) {
            cts_pos[n * 4 + 0] =
                -1.0f + (((float)(3 + 7 * n)) + 0.5f) / 45.0f * 2.0f;
            cts_pos[n * 4 + 1] = -1.0f + (3.5f / 45.0f) * 2.0f;
            cts_pos[n * 4 + 2] = 0.0f;
            cts_pos[n * 4 + 3] = 1.0f;
        }
        glBufferData(GL_ARRAY_BUFFER, sizeof(cts_pos), cts_pos,
                     GL_STATIC_DRAW);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void *)0);
        glDisableVertexAttribArray(1);
    } else {
    glGenBuffers(1, &vbo_c);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    }
    if (getenv("MGL_PG_CTSLINES")) {
        glUseProgram(program);
        glUniform2i(glGetUniformLocation(program, "renderingTargetSize"),
                    45, 45);
        glViewport(0, 0, 45, 45);
        glDrawArrays(GL_LINE_STRIP, 0, 8);
        glFinish();
        unsigned char px[45 * 45 * 4];
        memset(px, 0, sizeof(px));
        glReadPixels(0, 0, 45, 45, GL_RGBA, GL_UNSIGNED_BYTE, px);
        int lit = 0, lit_a = 0, first = -1;
        for (int i = 0; i < 45 * 45; i++) {
            if (px[i * 4] || px[i * 4 + 1] || px[i * 4 + 2]) {
                lit++;
                if (first < 0) first = i;
            }
            if (px[i * 4 + 3] && (i % 7 == 0)) lit_a++;
        }
        fprintf(stderr,
                "cts_lines_shape: lit=%d lit_a~%d first=%d "
                "rgba=%d,%d,%d,%d\n",
                lit, lit_a, first,
                first >= 0 ? px[first * 4] : px[0],
                first >= 0 ? px[first * 4 + 1] : px[1],
                first >= 0 ? px[first * 4 + 2] : px[2],
                first >= 0 ? px[first * 4 + 3] : px[3]);
        {
            /* per-column lit histogram across all rows */
            int colhit[45] = {0};
            for (int yy = 0; yy < 45; yy++)
                for (int xx = 0; xx < 45; xx++) {
                    const unsigned char *q = &px[(yy * 45 + xx) * 4];
                    if (q[0] || q[1] || q[2]) colhit[xx]++;
                }
            fprintf(stderr, "cols:");
            for (int xx = 0; xx < 45; xx++)
                fprintf(stderr, " %d", colhit[xx]);
            fprintf(stderr, "\n");
        }
        result = 0;
        goto cleanup;
    }
    glUseProgram(program);
    glUniform2i(glGetUniformLocation(program, "renderingTargetSize"),
                REG_W, REG_H);
    if (getenv("MGL_PG_VSFS")) {
        /* ablation: same draw through a plain VS+FS program */
        static const char *vspg =
            "#version 450 core\n"
            "layout(location=0) in vec2 position;\n"
            "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
        static const char *fspg =
            "#version 450 core\n"
            "layout(location=0) out vec4 frag;\n"
            "void main() { frag = vec4(1.0, 0.0, 0.0, 1.0); }\n";
        GLuint p2 = link_program(vspg, fspg);
        if (p2) {
            glUseProgram(p2);
            glDrawArrays(GL_POINTS, 0, 1);
            glDeleteProgram(p2);
        }
    } else {
        glDrawArrays(GL_POINTS, 0, 1);
    }
    glFinish();
    if (getenv("MGL_PG_INTOUT_VSFS")) {
        /* ablation: same int output through plain VS+FS */
        GLuint p2 = link_program(
            "#version 450 core\n"
            "layout(location=0) in vec2 position;\n"
            "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n",
            "#version 450 core\n"
            "layout(location=0) out int fs_out;\n"
            "void main() { fs_out = 7; }\n");
        if (p2) {
            glUseProgram(p2);
            glDrawArrays(GL_POINTS, 0, 1);
            glFinish();
            GLint iv[4] = { -1, 0, 0, 0 };
            glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_INT, iv);
            fprintf(stderr, "points_grid INTOUT-VSFS: r=%d (expect 7)\n",
                    iv[0]);
            glDeleteProgram(p2);
        }
        result = 0;
        goto cleanup;
    }
    if (getenv("MGL_PG_INTOUT")) {
        GLint iv[4] = { -1, 0, 0, 0 };
        glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_INT, iv);
        fprintf(stderr, "points_grid INTOUT: r=%d (expect 7)\n", iv[0]);
        result = 0;
        goto cleanup;
    }
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    /* All 9 emitted points rasterize with the vertex color (0.25,0.5,0.75
     * -> 64,128,191): exactly 9 pixels, each matching within 1 LSB. */
    {
        int found = 0;
        for (int y = 0; y < REG_H; y++)
            for (int x = 0; x < REG_W; x++) {
                const unsigned char *q = &pixels[(y * REG_W + x) * 4];
                if (!q[0] && !q[1] && !q[2]) continue;
                found++;
                if (abs(q[0] - 64) > 1 || abs(q[1] - 128) > 1 ||
                    abs(q[2] - 191) > 1 || q[3] != 255) {
                    fprintf(stderr,
                            "air_geometry_points_grid: pixel (%d,%d) got "
                            "(%d,%d,%d,%d), expected (64,128,191,255)\n",
                            x, y, q[0], q[1], q[2], q[3]);
                    goto cleanup;
                }
            }
        if (found != 9) {
            fprintf(stderr,
                    "air_geometry_points_grid: %d nonzero pixels, "
                    "expected 9\n", found);
            goto cleanup;
        }
    }

    /* Control: the same center point drawn without a geometry shader
     * must land on the same pixel as the GS-emitted center point. */
    {
        int gx = -1, gy = -1;
        for (int y = 0; y < REG_H; y++)
            for (int x = 0; x < REG_W; x++) {
                const unsigned char *q = &pixels[(y * REG_W + x) * 4];
                if (q[0] && q[1]) { gx = x; gy = y; }
            }
        static const char *vs2 =
            "#version 450 core\n"
            "layout(location=0) in vec2 position;\n"
            "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
        static const char *fs2 =
            "#version 450 core\n"
            "layout(location=0) out vec4 frag;\n"
            "void main() { frag = vec4(1.0, 0.0, 0.0, 1.0); }\n";
        GLuint prog2 = link_program(vs2, fs2);
        if (prog2) {
            glUseProgram(prog2);
            glDrawArrays(GL_POINTS, 0, 1);
            glFinish();
            glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
            for (int y = 0; y < REG_H; y++)
                for (int x = 0; x < REG_W; x++) {
                    const unsigned char *q = &pixels[(y * REG_W + x) * 4];
                    if (q[0] > 200 && !q[1] && gx >= 0 &&
                        (abs(x - gx) > 1 || abs(y - gy) > 1))
                        {
                            fprintf(stderr,
                                    "points_grid: no-GS point (%d,%d) vs GS "
                                    "center (%d,%d)\n", x, y, gx, gy);
                            glDeleteProgram(prog2);
                            goto cleanup;
                        }
                }
        }
        if (prog2) glDeleteProgram(prog2);
    }

    result = 0;

cleanup:
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vbo_c) glDeleteBuffers(1, &vbo_c);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}


/* AIR argument metadata must be ordered by parameter index: a fragment
 * shader with both a plain uniform (buffer argument) and a varying
 * (fragment_input value argument) used to emit the buffer node first,
 * breaking the rasterizer varying link (colors read as 0). */
static int test_fs_varying_with_plain_uniform(unsigned char *pixels,
                                              const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=0) out vec3 c3;\n"
        "void main() { c3 = vec3(1.0, 0.0, 0.0);\n"
        "  gl_PointSize = 1.0;\n"
        "  gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *fs =
        "#version 450 core\n"
        "uniform ivec2 sz;\n"
        "layout(location=0) in vec3 c3;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(c3, 1.0) + float(sz.x) * 1e-7; }\n";
    static const float pos[2] = { 0.0f, 0.5f };

    GLuint fbo = 0u, tex = 0u, vao = 0u, vbo = 0u, program = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &tex);
    if (!fbo) goto cleanup;
    program = link_program(vs, fs);
    if (!program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glUseProgram(program);
    glUniform2i(glGetUniformLocation(program, "sz"), REG_W, REG_H);
    glDrawArrays(GL_POINTS, 0, 1);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    {
        int red = 0;
        for (int y = 0; y < REG_H; y++)
            for (int x = 0; x < REG_W; x++) {
                const unsigned char *q = &pixels[(y * REG_W + x) * 4];
                if (q[0] > 200 && q[1] < 50) red++;
            }
        if (red != 1) {
            fprintf(stderr,
                    "fs_varying_with_plain_uniform: %d red pixels, "
                    "expected 1\n", red);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (tex) glDeleteTextures(1, &tex);
    return result;
}

/* KHR-GL46.geometry_shader.rendering lines-in/line_strip-out replica:
 * one horizontal input line, GS expands it into three line strips
 * (CTS rendering family shape).  Verifies the expanded primitives
 * actually rasterize. */
/* CTS-derived regressions for GS link/query/XFB-builtin semantics. */
static int test_gs_link_semantics(unsigned char *pixels,
                                  const char *out_path)
{
    (void)pixels;
    (void)out_path;
    int result = 1;

    /* GLSL 4.60 §4.4.1.2: layout(invocations<=0) is a compile-time error. */
    {
        static const char *gs =
            "#version 460 core\n"
            "layout(points, invocations=0) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "void main() { gl_Position = gl_in[0].gl_Position; EmitVertex(); }\n";
        GLuint s = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(s, 1, &gs, NULL);
        glCompileShader(s);
        GLint ok = 1;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (ok) {
            fprintf(stderr,
                    "gs_link_semantics: layout(invocations=0) compiled\n");
            glDeleteShader(s);
            return 1;
        }
        glDeleteShader(s);
    }

    /* GL 4.6 §11.3.2: missing max_vertices fails link, not compile. */
    {
        static const char *vs =
            "#version 460 core\n"
            "void main() { gl_Position = vec4(1.0); }\n";
        static const char *gs =
            "#version 460 core\n"
            "layout(points) in;\n"
            "layout(points) out;\n"
            "void main() { gl_Position = gl_in[0].gl_Position; EmitVertex(); }\n";
        static const char *fs =
            "#version 460 core\n"
            "layout(location=0) out vec4 frag;\n"
            "void main() { frag = vec4(1.0); }\n";
        GLuint a = compile_shader(GL_VERTEX_SHADER, vs);
        GLuint b = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(b, 1, &gs, NULL);
        glCompileShader(b);
        GLint compiled = 0;
        glGetShaderiv(b, GL_COMPILE_STATUS, &compiled);
        GLuint c = compile_shader(GL_FRAGMENT_SHADER, fs);
        GLuint prog = glCreateProgram();
        int local_fail = 0;
        if (!a || !compiled || !c || !prog) {
            fprintf(stderr,
                    "gs_link_semantics: missing max_vertices compile failed "
                    "(compiled=%d)\n", compiled);
            local_fail = 1;
        } else {
            glAttachShader(prog, a);
            glAttachShader(prog, b);
            glAttachShader(prog, c);
            glLinkProgram(prog);
            GLint linked = 1;
            glGetProgramiv(prog, GL_LINK_STATUS, &linked);
            if (linked) {
                fprintf(stderr,
                        "gs_link_semantics: missing max_vertices linked\n");
                local_fail = 1;
            }
        }
        if (a) glDeleteShader(a);
        if (b) glDeleteShader(b);
        if (c) glDeleteShader(c);
        if (prog) glDeleteProgram(prog);
        if (local_fail) return 1;
    }

    /* layout(invocations=N) must reach GL_GEOMETRY_SHADER_INVOCATIONS. */
    {
        static const char *vs =
            "#version 460 core\n"
            "void main() { gl_Position = vec4(1.0); }\n";
        static const char *gs =
            "#version 460 core\n"
            "layout(points, invocations=10) in;\n"
            "layout(points, max_vertices=1) out;\n"
            "void main() {\n"
            "    gl_Position = gl_in[0].gl_Position;\n"
            "    EmitVertex(); EndPrimitive();\n"
            "}\n";
        static const char *fs =
            "#version 460 core\n"
            "layout(location=0) out vec4 frag;\n"
            "void main() { frag = vec4(1.0); }\n";
        GLuint prog = glCreateProgram();
        GLuint a = compile_shader(GL_VERTEX_SHADER, vs);
        GLuint b = compile_shader(GL_GEOMETRY_SHADER, gs);
        GLuint c = compile_shader(GL_FRAGMENT_SHADER, fs);
        if (!a || !b || !c) goto invocations_done;
        glAttachShader(prog, a);
        glAttachShader(prog, b);
        glAttachShader(prog, c);
        glLinkProgram(prog);
        {
            GLint ok = 0;
            GLint inv = -1;
            glGetProgramiv(prog, GL_LINK_STATUS, &ok);
            glGetProgramiv(prog, GL_GEOMETRY_SHADER_INVOCATIONS, &inv);
            if (!ok || inv != 10) {
                fprintf(stderr,
                        "gs_link_semantics: invocations query got %d\n",
                        inv);
                goto invocations_done;
            }
        }
        result = 0;
    invocations_done:
        if (a) glDeleteShader(a);
        if (b) glDeleteShader(b);
        if (c) glDeleteShader(c);
        if (prog) glDeleteProgram(prog);
    }
    if (result != 0) return result;

    /* Capturing gl_Position through a GS must write the TF buffer and
     * count primitives. */
    {
        static const char *vs =
            "#version 460 core\n"
            "void main() { gl_Position = vec4(1.0); }\n";
        static const char *gs =
            "#version 460 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices=8) out;\n"
            "void main() {\n"
            "    for (int n = 0; n < 8; ++n) {\n"
            "        gl_Position = vec4(1.0f / (float(n) + 1.0f), 0.0, 0.0, 1.0);\n"
            "        EmitVertex();\n"
            "    }\n"
            "    EndPrimitive();\n"
            "}\n";
        static const char *fs =
            "#version 460 core\n"
            "layout(location=0) out vec4 frag;\n"
            "void main() { frag = vec4(1.0); }\n";
        GLuint prog = glCreateProgram();
        GLuint a = compile_shader(GL_VERTEX_SHADER, vs);
        GLuint b = compile_shader(GL_GEOMETRY_SHADER, gs);
        GLuint c = compile_shader(GL_FRAGMENT_SHADER, fs);
        GLuint tfb = 0u, vao = 0u;
        GLuint query = 0u;
        if (!a || !b || !c) goto xfb_done;
        glAttachShader(prog, a);
        glAttachShader(prog, b);
        glAttachShader(prog, c);
        {
            const char *tfv = "gl_Position";
            glTransformFeedbackVaryings(prog, 1, &tfv,
                                        GL_SEPARATE_ATTRIBS);
        }
        glLinkProgram(prog);
        {
            GLint ok = 0;
            glGetProgramiv(prog, GL_LINK_STATUS, &ok);
            if (!ok) {
                fprintf(stderr,
                        "gs_link_semantics: gl_Position XFB link failed\n");
                goto xfb_done;
            }
        }
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &tfb);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfb);
        {
            GLfloat zeros[32] = {0};
            glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(zeros),
                         zeros, GL_STATIC_READ);
        }
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
        glGenQueries(1, &query);
        glUseProgram(prog);
        glBeginTransformFeedback(GL_POINTS);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query);
        glDrawArrays(GL_POINTS, 0, 1);
        glEndTransformFeedback();
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        {
            GLfloat got[8] = {0};
            GLuint wval = 0u;
            glGetQueryObjectuiv(query, GL_QUERY_RESULT, &wval);
            glGetNamedBufferSubData(tfb, 0, sizeof(got), got);
            /* One input point fans out to 8 emitted point primitives. */
            if (wval != 8u) {
                fprintf(stderr,
                        "gs_link_semantics: tf written=%u expect 8\n",
                        wval);
                result = 5;
                goto xfb_done;
            }
            /* Record n=0 wrote position (1, 0, 0, 1). */
            if (got[0] != 1.0f || got[3] != 1.0f) {
                fprintf(stderr,
                        "gs_link_semantics: tf r0={%.3f %.3f %.3f %.3f}\n",
                        got[0], got[1], got[2], got[3]);
                result = 6;
                goto xfb_done;
            }
        }
        result = 0;

xfb_done:
        if (query) glDeleteQueries(1, &query);
        if (tfb) glDeleteBuffers(1, &tfb);
        if (vao) glDeleteVertexArrays(1, &vao);
        if (a) glDeleteShader(a);
        if (b) glDeleteShader(b);
        if (c) glDeleteShader(c);
        if (prog) glDeleteProgram(prog);
    }

    /* Report component-related limits */
    {
        struct { const char *n; GLenum e; } q[] = {
            {"MAX_GEOMETRY_OUTPUT_COMPONENTS",
             GL_MAX_GEOMETRY_OUTPUT_COMPONENTS},
            {"MAX_FRAGMENT_INPUT_COMPONENTS",
             GL_MAX_FRAGMENT_INPUT_COMPONENTS},
            {"MAX_VERTEX_OUTPUT_COMPONENTS",
             GL_MAX_VERTEX_OUTPUT_COMPONENTS},
        };
        for (int i = 0; i < 3; i++) {
            GLint v = -1;
            glGetIntegerv(q[i].e, &v);
            fprintf(stderr, "limits: %s = %d\n", q[i].n, v);
        }
    }

cleanup:
    return result;
}

static int test_air_geometry_lines_expand(unsigned char *pixels,
                                          const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=1) in vec4 color;\n"
        "uniform ivec2 renderingTargetSize;\n"
        "out vec4 vs_gs_color[2];\n"
        "void main() { vs_gs_color[0] = color;\n"
        "  vs_gs_color[1] = color;\n"
        "  gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 450 core\n"
        "layout(lines) in;\n"
        "layout(line_strip, max_vertices=6) out;\n"
        "in vec4 vs_gs_color[2];\n"
        "out vec4 gs_fs_color;\n"
        "uniform ivec2 renderingTargetSize;\n"
        "void main() {\n"
        "  float dy = 2.0 / float(renderingTargetSize.y);\n"
        "  vec4 start_pos = gl_in[0].gl_Position;\n"
        "  vec4 end_pos   = gl_in[1].gl_Position;\n"
        "  vec4 col = mix(vs_gs_color[0], vs_gs_color[1], 0.5);\n"
        "  for (int k = -1; k <= 1; ++k) {\n"
        "    gs_fs_color = col;\n"
        "    gl_Position = vec4(start_pos.x, start_pos.y + k * dy, 0, 1);\n"
        "    EmitVertex();\n"
        "    gs_fs_color = col;\n"
        "    gl_Position = vec4(end_pos.x, end_pos.y + k * dy, 0, 1);\n"
        "    EmitVertex();\n"
        "    EndPrimitive();\n"
        "  }\n"
        "}\n";
    static const char *fs_le_a =
        "#version 450 core\n"
        "layout(location=0) in vec4 gs_fs_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = gs_fs_color; }\n";
    static const char *fs_le_v3 =
        "#version 450 core\n"
        "layout(location=0) in vec3 gs_fs_color;\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(gs_fs_color, 1.0); }\n";
    const char *fs = getenv("MGL_LE_FS_V3") ? fs_le_v3 : fs_le_a;
    static const float positions[4] = { -0.5f, 0.5f, 0.5f, 0.5f };
    static const float colors[8] = { 0.1f, 0.2f, 0.3f, 0.4f,
                                     0.1f, 0.2f, 0.3f, 0.4f };

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u, vbo_c = 0u, program = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;
    program = link_program_with_geometry(vs, gs, fs);
    if (!program) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    clear_color(0.0f, 0.0f, 0.0f);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glGenBuffers(1, &vbo_c);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glUseProgram(program);
    glUniform2i(glGetUniformLocation(program, "renderingTargetSize"),
                REG_W, REG_H);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    /* Three horizontal 1px lines across the middle half of the FBO. */
    {
        int rows[3] = { 95, 96, 97 };  /* readback rows are bottom-origin */
        for (int r = 0; r < 3; r++) {
            int y = rows[r];
            int lit = 0;
            for (int x = 0; x < REG_W; x++) {
                const unsigned char *q = &pixels[(y * REG_W + x) * 4];
                if (q[0] || q[1] || q[2]) lit++;
            }
            if (lit < 32) {
                fprintf(stderr,
                        "air_geometry_lines_expand: row %d has %d lit "
                        "pixels, expected >= 32\n", y, lit);

                goto cleanup;
            }
        }
    }

    result = 0;

cleanup:
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vbo_c) glDeleteBuffers(1, &vbo_c);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}
/* P3.2: safe-fallback pipeline built from the precompiled safe_fallback
 * metallib asset — no runtime source compilation in the emergency path.
 *
 * Phase A (flag unset): the AIR final pipeline renders the green triangle;
 * the safe assets must not leak into the normal path.
 * Phase B (MGL_FORCE_SAFE_FALLBACK_PIPELINE=1): pipeline creation throws a
 * synthetic exception before the real PSO build; the virtualized-AGX safe
 * branch must resolve functions from the embedded asset and create a usable
 * PSO.  Its shaders output a degenerate center triangle, so the probe keeps
 * the clear color and the draw completes without GL error.
 * Phase C (fresh program, flag unset): green again — the forced fallback left
 * no fallout in the shared pipeline cache.
 */
static int test_air_pipeline_safe_fallback(unsigned char *pixels,
                                           const char *out_path)
{
    (void)out_path;
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 frag;\n"
        "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";

    GLuint fbo = 0u, color = 0u, vao = 0u, vbo = 0u;
    GLuint programA = 0u, programB = 0u, programC = 0u;
    int result = 1;
    fbo = make_fbo(REG_W, REG_H, &color);
    if (!fbo) goto cleanup;

    programA = link_program(vs, fs);
    programB = link_program(vs, fs);
    programC = link_program(vs, fs);
    if (!programA || !programB || !programC) goto cleanup;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const float tri[6] = { -0.2f, -0.2f, 0.4f, -0.2f, -0.2f, 0.4f };
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);

    /* Probe inside the triangle (centroid) and outside it (clear color). */
    const int px[2][2] = {
        { (int)((0.0f + 1.0f) * 0.5f * REG_W),
          (int)((0.0f + 1.0f) * 0.5f * REG_H) },
        { (int)((-0.7f + 1.0f) * 0.5f * REG_W),
          (int)((0.7f + 1.0f) * 0.5f * REG_H) },
    };

    /* Phase A: normal AIR pipeline renders the green triangle. */
    clear_color(0.0f, 0.0f, 0.0f);
    glUseProgram(programA);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *c =
            &pixels[(px[0][1] * REG_W + px[0][0]) * 4];
        if (c[0] > 20u || c[1] < 220u || c[2] > 20u) {
            fprintf(stderr,
                    "air_pipeline_safe_fallback: phase A probe not green "
                    "(%u,%u,%u)\n", c[0], c[1], c[2]);
            goto cleanup;
        }
    }

    /* Phase B: forced pipeline-creation exception -> safe fallback PSO from
     * the precompiled asset.  Draw must complete with no GL error, and the
     * (degenerate) safe shaders must have replaced the green program. */
    clear_color(0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    setenv("MGL_FORCE_SAFE_FALLBACK_PIPELINE", "1", 1);
    glUseProgram(programB);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    unsetenv("MGL_FORCE_SAFE_FALLBACK_PIPELINE");
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *c =
            &pixels[(px[0][1] * REG_W + px[0][0]) * 4];
        if (c[0] > 20u || c[1] > 20u || c[2] < 220u) {
            fprintf(stderr,
                    "air_pipeline_safe_fallback: phase B probe not clear "
                    "blue (%u,%u,%u) — safe fallback did not take over\n",
                    c[0], c[1], c[2]);
            goto cleanup;
        }
        GLenum err = GL_NO_ERROR;
        while ((err = glGetError()) != GL_NO_ERROR) {
            fprintf(stderr,
                    "air_pipeline_safe_fallback: phase B GL error 0x%x\n",
                    err);
            goto cleanup;
        }
    }

    /* Phase C: fresh program draws green again (cache/pipeline state clean). */
    clear_color(0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(programC);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, REG_W, REG_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    {
        const unsigned char *c =
            &pixels[(px[0][1] * REG_W + px[0][0]) * 4];
        if (c[0] > 20u || c[1] < 220u || c[2] > 20u) {
            fprintf(stderr,
                    "air_pipeline_safe_fallback: phase C probe not green "
                    "(%u,%u,%u)\n", c[0], c[1], c[2]);
            goto cleanup;
        }
        const unsigned char *outside =
            &pixels[(px[1][1] * REG_W + px[1][0]) * 4];
        if (outside[0] > 20u || outside[1] > 20u || outside[2] > 20u) {
            fprintf(stderr,
                    "air_pipeline_safe_fallback: phase C outside probe "
                    "not black (%u,%u,%u)\n",
                    outside[0], outside[1], outside[2]);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (programA) glDeleteProgram(programA);
    if (programB) glDeleteProgram(programB);
    if (programC) glDeleteProgram(programC);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (color) glDeleteTextures(1, &color);
    return result;
}

/* Link-time guard for the GS compute ABI's reserved Metal buffer slots.
 * Twenty-four SSBOs occupy user slots 0..23 and remain valid.  Adding the
 * twenty-fifth assigns user slot 24, which is MGL_AIR_GS_SLOT_INPUT and must
 * fail the link before the renderer can silently overwrite either binding. */
static int test_air_geometry_buffer_slot_conflict(unsigned char *pixels,
                                                  const char *out_path)
{
    (void)pixels;
    (void)out_path;
    GLint control_status = GL_FALSE;
    GLint conflict_status = GL_TRUE;

    if (geometry_program_link_status_with_ssbo_count(
            24u, 0, &control_status) != 0 ||
        geometry_program_link_status_with_ssbo_count(
            25u, 0, &conflict_status) != 0) {
        fprintf(stderr,
                "air_geometry_buffer_slot_conflict: shader setup failed\n");
        return 1;
    }
    if (control_status != GL_TRUE || conflict_status != GL_FALSE) {
        fprintf(stderr,
                "air_geometry_buffer_slot_conflict: expected 24 SSBO link=1 "
                "and 25 SSBO link=0, got %d/%d\n",
                control_status, conflict_status);
        return 1;
    }

    /* GS runtime-array size metadata moves to hidden slot 23 so gather params
     * remain at slot 25.  User slots 0..22 are valid; a 24th resource would
     * occupy the hidden size-table slot and must fail at link time. */
    GLint gs_runtime_valid_status = GL_FALSE;
    GLint gs_runtime_conflict_status = GL_TRUE;
    if (geometry_program_link_status_with_ssbo_count(
            23u, 1, &gs_runtime_valid_status) != 0 ||
        geometry_program_link_status_with_ssbo_count(
            24u, 1, &gs_runtime_conflict_status) != 0 ||
        gs_runtime_valid_status != GL_TRUE ||
        gs_runtime_conflict_status != GL_FALSE) {
        fprintf(stderr,
                "air_geometry_buffer_slot_conflict: expected GS runtime-size "
                "23/24 SSBO link=1/0, got %d/%d\n",
                gs_runtime_valid_status, gs_runtime_conflict_status);
        return 1;
    }

    /* The runtime-array-size table occupies slot 25 for compute stages.  A
     * program with 25 user SSBOs reaches slots 0..24 and remains valid; the
     * 26th reaches slot 25 and must fail before link success. */
    GLint runtime_valid_status = GL_FALSE;
    GLint runtime_conflict_status = GL_TRUE;
    if (compute_program_link_status_with_ssbo_count(
            25u, 1, &runtime_valid_status) != 0 ||
        compute_program_link_status_with_ssbo_count(
            26u, 1, &runtime_conflict_status) != 0 ||
        runtime_valid_status != GL_TRUE || runtime_conflict_status != GL_FALSE) {
        fprintf(stderr,
                "air_geometry_buffer_slot_conflict: expected runtime-size "
                "25/26 SSBO link=1/0, got %d/%d\n",
                runtime_valid_status, runtime_conflict_status);
        return 1;
    }

    /* Renderer user-buffer tables expose [0,
     * kMGLMaxMetalUserBufferCount).  The physical compute ABI additionally
     * owns slot kMGLMaxMetalComputeBufferIndex, which must not be assigned to
     * a reflected user resource even when runtime sizing is inactive. */
    GLint max_valid_status = GL_FALSE;
    GLint max_conflict_status = GL_TRUE;
    if (compute_program_link_status_with_ssbo_count(
            31u, 0, &max_valid_status) != 0 ||
        compute_program_link_status_with_ssbo_count(
            32u, 0, &max_conflict_status) != 0 ||
        max_valid_status != GL_TRUE || max_conflict_status != GL_FALSE) {
        fprintf(stderr,
                "air_geometry_buffer_slot_conflict: expected user-slot "
                "31/32 SSBO link=1/0, got %d/%d\n",
                max_valid_status, max_conflict_status);
        return 1;
    }
    return 0;
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
    SELF_CHECK_TEST("gl_clip_planes",     test_gl_clip_planes),
    SELF_CHECK_TEST("legacy_clip_vertex", test_legacy_clip_vertex),
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
    SELF_CHECK_TEST("air_cull_distance", test_air_cull_distance),
    SELF_CHECK_TEST("air_geometry_varying", test_air_geometry_varying),
    SELF_CHECK_TEST("air_geometry_resources", test_air_geometry_resources),
    SELF_CHECK_TEST("air_geometry_buffer_slot_conflict",
                    test_air_geometry_buffer_slot_conflict),
    SELF_CHECK_TEST("air_geometry_instancing", test_air_geometry_instancing),
    SELF_CHECK_TEST("air_gs_unsupported", test_air_gs_unsupported),
    SELF_CHECK_TEST("air_geometry_indexed", test_air_geometry_indexed),
    SELF_CHECK_TEST("air_geometry_indirect", test_air_geometry_indirect),
    SELF_CHECK_TEST("air_geometry_multi_draw", test_air_geometry_multi_draw),
    SELF_CHECK_TEST("air_geometry_base_vertex_instance",
                    test_air_geometry_base_vertex_instance),
    SELF_CHECK_TEST("air_tessellation_indexed", test_air_tessellation_indexed),
    SELF_CHECK_TEST("air_tessellation_instanced",
                    test_air_tessellation_instanced),
    SELF_CHECK_TEST("air_tessellation_indirect",
                    test_air_tessellation_indirect),
    SELF_CHECK_TEST("air_tessellation_multipatch",
                    test_air_tessellation_multipatch),
    SELF_CHECK_TEST("air_tessellation_varying", test_air_tessellation_varying),
    SELF_CHECK_TEST("air_tessellation_accumulation",
                    test_air_tessellation_accumulation),
    SELF_CHECK_TEST("air_tessellation_isolines_point_mode",
                    test_air_tessellation_isolines_point_mode),
    SELF_CHECK_TEST("air_tessellation_isolines_variants",
                    test_air_tessellation_isolines_variants),
    SELF_CHECK_TEST("air_tessellation_isolines_indexed",
                    test_air_tessellation_isolines_indexed),
    SELF_CHECK_TEST("air_tessellation_isolines_multidraw",
                    test_air_tessellation_isolines_multidraw),
    SELF_CHECK_TEST("air_tessellation_isolines_rasterdiscard",
                    test_air_tessellation_isolines_rasterdiscard),
    SELF_CHECK_TEST("air_tessellation_isolines_tripoint_instanced",
                    test_air_tessellation_isolines_tripoint_instanced),
    SELF_CHECK_TEST("air_tessellation_isolines_xfb",
                    test_air_tessellation_isolines_xfb),
    SELF_CHECK_TEST("air_tessellation_patch_varying",
                    test_air_tessellation_patch_varying),
    SELF_CHECK_TEST("air_tessellation_resources",
                    test_air_tessellation_resources),
    SELF_CHECK_TEST("air_tessellation_factors_spacing",
                    test_air_tessellation_factors_spacing),
    SELF_CHECK_TEST("air_tessellation_cull_distance",
                    test_air_tessellation_cull_distance),
    SELF_CHECK_TEST("air_geometry_xfb", test_air_geometry_xfb),
    SELF_CHECK_TEST("air_xfb_link_layout", test_air_xfb_link_layout),
    SELF_CHECK_TEST("air_xfb_reflection", test_air_xfb_reflection),
    SELF_CHECK_TEST("air_geometry_xfb_truncate",
                    test_air_geometry_xfb_truncate),
    SELF_CHECK_TEST("air_geometry_separate_xfb",
                    test_air_geometry_separate_xfb),
    SELF_CHECK_TEST("air_geometry_passthrough_xfb",
                    test_air_geometry_passthrough_xfb),
    SELF_CHECK_TEST("air_geometry_passthrough_queries",
                    test_air_geometry_passthrough_queries),
    SELF_CHECK_TEST("air_geometry_layered_repro",
                    test_air_geometry_layered_repro),
    SELF_CHECK_TEST("air_geometry_multi_stream_xfb",
                    test_air_geometry_multi_stream_xfb),
    SELF_CHECK_TEST("air_geometry_layer_viewport",
                    test_air_geometry_layer_viewport),
    SELF_CHECK_TEST("air_geometry_cull_distance",
                    test_air_geometry_cull_distance),
    SELF_CHECK_TEST("air_geometry_ssbo_visibility",
                    test_air_geometry_ssbo_visibility),
    SELF_CHECK_TEST("compute_dispatch_ssbo", test_compute_dispatch_ssbo),
    SELF_CHECK_TEST("air_pipeline_safe_fallback",
                    test_air_pipeline_safe_fallback),
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
    SELF_CHECK_TEST("air_msaa_resolve",   test_air_msaa_resolve),
    SELF_CHECK_TEST("legacy_glsl_frontend", test_legacy_glsl_frontend),
    GOLDEN_TEST("rtt_sample",             test_render_to_texture_sample),
    SELF_CHECK_TEST("air_query_scissor_occluded",
                    test_air_query_scissor_occluded),
    SELF_CHECK_TEST("air_renderpass_layer_slice",
                    test_air_renderpass_layer_slice),
    SELF_CHECK_TEST("texture_mip_dimensions",
                    test_texture_mip_dimensions),
    SELF_CHECK_TEST("texture_storage_internalformat_validation",
                    test_texture_storage_internalformat_validation),
    SELF_CHECK_TEST("framebuffer_layer_targets",
                    test_framebuffer_layer_targets),
    SELF_CHECK_TEST("framebuffer_texture_layer_validation",
                    test_framebuffer_texture_layer_validation),
    SELF_CHECK_TEST("framebuffer_cube_layer_slice",
                    test_framebuffer_cube_layer_slice),
    SELF_CHECK_TEST("air_geometry_points_grid",
                    test_air_geometry_points_grid),
    SELF_CHECK_TEST("gs_link_semantics", test_gs_link_semantics),
    SELF_CHECK_TEST("air_geometry_lines_expand",
                    test_air_geometry_lines_expand),
    SELF_CHECK_TEST("fs_varying_with_plain_uniform",
                    test_fs_varying_with_plain_uniform),
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
