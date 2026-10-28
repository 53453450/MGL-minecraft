/*
 * mc_offline_harness.c — offline Minecraft 1.21.11 frame-shape harness for MGL.
 *
 * Replays the per-frame GL call stream that vanilla 1.21.11 produces through
 * com.mojang.blaze3d.opengl.GlCommandEncoder (+ DynamicUniforms /
 * VertexArrayCache), derived from the deobfuscated source tree, so MGL-side
 * costs (batch formation, state sync, pointer validation) can be measured and
 * A/B-ed without launching the game.
 *
 * Faithfulness notes (source lines of record in the 1.21.11 tree):
 *  - GlCommandEncoder.drawFromBuffers: bindVertexArray(format, vbo) per draw
 *    (VertexArrayCache hit -> glBindVertexArray, glBindVertexBuffer on VBO
 *    change), then an UNCONDITIONAL glBindBuffer(GL_ELEMENT_ARRAY_BUFFER)
 *    for indexed draws, then glDrawElements.
 *  - GlCommandEncoder.trySetup: per draw; sampler uniforms rebind
 *    UNCONDITIONALLY (glActiveTexture + glBindTexture + glBindSampler +
 *    2x glTexParameteri on GL_TEXTURE_MIN/MAX_LOD); UBO uniforms do
 *    glBindBufferRange per draw while dirty (DynamicUniforms hands every
 *    draw its own slice offset from a persistent-mapped ring buffer —
 *    DynamicUniformStorage.writeUniform).
 *  - applyPipelineState fires per pipeline change: depth test/func, cull,
 *    blend + blendFuncSeparate, polygon mode, depth/color mask.
 *  - createRenderPass per pass: glBindFramebuffer, optional clear, viewport.
 *
 * Visual mode (default) renders for real: a visible window, one unit quad per
 * draw placed on a grid by the per-draw UBO slice, an atlas in every texture
 * unit, and all six passes accumulating on one target that is blitted to the
 * drawable.  The frame therefore reaches the screen and can be inspected, and
 * frame_pixels / drawable painted report how much of it was actually drawn
 * (a readback at the end, after the timings, so it costs nothing).
 *
 * HARNESS_SUBMIT_ONLY=1 restores the older shape — hidden window, one target
 * per pass, empty payload, no blit — which rasterises nothing and keeps the
 * timing comparable with pre-visual runs.  Either way the measured stream is
 * the same call shape (binding counts, slice arithmetic, state churn); visual
 * mode adds ~3 ms/frame of GPU rasterisation to the wall number, not to the
 * CPU one.  HARNESS_FINISH turns the GPU drain into its own measurement.
 *
 * Invariants enforced at init (the harness fails fast instead of quietly
 * measuring a broken frame):
 *  - every program's DynamicTransforms block is mapped onto UBO_BINDING with
 *    glGetUniformBlockIndex + glUniformBlockBinding and read back, because
 *    GLSL 3.30 cannot spell layout(binding=) on a uniform block;
 *  - the block's std140 size comes from GL_UNIFORM_BLOCK_DATA_SIZE and must
 *    fit one aligned slice;
 *  - vertex inputs match the VAO's feed path: aLight is read back as
 *    GL_FLOAT_VEC2, since the VAO converts SHORT data with glVertexAttribFormat
 *    and a float-format attribute must not feed an integer input;
 *  - every sampler uniform is assigned its unit once and read back, so the
 *    entity pass really samples two units (Sampler1 -> 1, not the default 0);
 *  - the UBO ring is sized from draws_per_frame() — the same function that
 *    labels the run — and once a slice would fall outside it the draws stop and
 *    the run exits 2 instead of measuring ranges that do not exist.
 *
 * Measured: per-frame wall/CPU, getrusage user/sys split, CLOCK_PROCESS_CPUTIME
 * per-frame cross-check, and — when linked against an MGL build carrying the
 * mgl_vm_probe_count counter — exact vm_region probe counts per frame (dlsym,
 * optional).
 *
 * Knobs (numeric bounds are clamped, not silently truncated):
 *   HARNESS_FRAMES, HARNESS_WARMUP, HARNESS_TERRAIN_DRAWS,
 *   HARNESS_ENTITY_DRAWS, HARNESS_GUI_DRAWS,
 *   HARNESS_STARTUP_QUERIES, HARNESS_SKIP_LOD_PARAMS,
 *   HARNESS_SKIP_SAMPLER_BLOCK, HARNESS_NO_PRESENT, HARNESS_FINISH,
 *   HARNESS_CHECK_GL, HARNESS_SUBMIT_ONLY, HARNESS_VSYNC,
 *   HARNESS_WIDTH, HARNESS_HEIGHT, HARNESS_HOLD (seconds to keep presenting),
 *   HARNESS_DUMP (write the frame as a PPM)  (flags accept 0 = off)
 *
 * Scratch/optimization tool: linked against build/libmgl.dylib like
 * mgl_benchmark (see Makefile target mc-harness).
 */
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

/* ------------------------------------------------------------------ */
/* Scenario calibration                                                */
/* ------------------------------------------------------------------ */

/* Draw mix: same totals the VM-probe audit modeled (1216 draws/frame,
 * matching benchmark MC_TERRAIN/ENTITY/GUI constants) plus a small sky pass
 * of array draws. */
#define DEFAULT_TERRAIN_DRAWS 896
#define DEFAULT_ENTITY_DRAWS  224
#define DEFAULT_GUI_DRAWS      96
#define SKY_DRAWS              24

#define TERRAIN_LAYERS 3            /* solid / cutout / cutout_mipped programs */
#define PASS_COUNT 6                /* sky, terrain, entities, translucent, clouds+weather, GUI */
#define CLEAR_PASSES 2              /* createRenderPass clear count per frame (trace: ~2.2/frame) */
#define ENTITY_TEXTURE_CYCLE 5      /* per-entity texture alternation */
#define GUI_TEXTURE_GROUP 8         /* vanilla GUI batches glyphs per atlas page */
#define TRANSLUCENT_DIVISOR 8       /* translucent re-draws a fraction of terrain */
#define CLOUD_DRAWS 32              /* clouds + weather array draws */

#define VAO_CACHE_SIZE 12           /* VertexArrayCache keyed by format; few formats in practice */
#define UBO_RING_BUFFERS 2          /* MappableRingBuffer(…, 130, …) */
#define UBO_BLOCK_ALIGN 256         /* GlDevice uniform offset alignment */
#define TRANSFORM_UBO_SIZE 96       /* DynamicUniforms.TRANSFORM_UBO_SIZE (game-side constant) */
#define UBO_BINDING 1               /* binding point every DynamicTransforms block is mapped to */
/* Slices not attributable to a modelled pass (e.g. one-off passes drawn once
 * per frame).  Kept as explicit headroom so the ring is never exactly full: a
 * full ring hides an off-by-one behind a clamp at the driver. */
#define UBO_RING_HEADROOM 16

/* Safety rails. */
#define MAX_FRAMES 1000000
#define MAX_WARMUP 100000
#define MAX_DRAWS  100000

/* Visual mode (default): a visible window, real vertex/index/texture/UBO data,
 * and every pass accumulating on pass 0's target — the way the game's main pass
 * renders — then blitted to the window.  HARNESS_SUBMIT_ONLY=1 restores the old
 * shape (hidden window, one target per pass, empty payload, no blit) for timing
 * runs that should stay comparable with pre-visual numbers. */
#define QUAD_VERTS 6                /* two triangles per draw */
#define GRID_COLS 40                /* quads per row when the frame is tiled */
#define DEFAULT_WIDTH 960
#define DEFAULT_HEIGHT 540

/* Present every frame keeps MGL's in-flight work bounded at one frame.  With
 * HARNESS_NO_PRESENT=1 nothing drains the commit queue, so the frame count is
 * capped hard rather than letting the process grow without bound. */
#define MAX_NO_PRESENT_FRAMES 300

/* ------------------------------------------------------------------ */
/* Timing                                                              */
/* ------------------------------------------------------------------ */

static double now_ns(clockid_t c)
{
    struct timespec ts;
    clock_gettime(c, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

typedef struct {
    double wall_per_frame_ms;
    double cpu_per_frame_ms;        /* getrusage user+sys, per-frame differenced */
    double user_per_frame_ms;
    double sys_per_frame_ms;
    double pcpu_per_frame_ms;       /* CLOCK_PROCESS_CPUTIME_ID, no scheduler noise */
    double p50_ms;
    double p95_ms;
    double pcpu_p50_ms;
} FrameStats;

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* nearest-rank with clamping; n*q can land on n for q near 1.0 after fp
 * rounding, and n<=0 must not reach the divisions below. */
static int pct_index(int n, double q)
{
    int idx = (int)((double)n * q);
    if (idx < 0) idx = 0;
    if (idx > n - 1) idx = n - 1;
    return idx;
}

static int summarize(FrameStats *out, const double *wall_ms, const double *user_ms,
                     const double *sys_ms, const double *pcpu_ms, int n)
{
    double *sorted;
    double wall = 0.0, user = 0.0, sys = 0.0, pcpu = 0.0;

    memset(out, 0, sizeof(*out));
    if (n <= 0) return -1;
    sorted = malloc((size_t)n * sizeof(double));
    if (!sorted) return -1;

    for (int i = 0; i < n; i++) {
        wall += wall_ms[i];
        user += user_ms[i];
        sys += sys_ms[i];
        pcpu += pcpu_ms[i];
        sorted[i] = wall_ms[i];
    }
    qsort(sorted, (size_t)n, sizeof(double), cmp_double);
    out->wall_per_frame_ms = wall / (double)n;
    out->user_per_frame_ms = user / (double)n;
    out->sys_per_frame_ms = sys / (double)n;
    out->cpu_per_frame_ms = out->user_per_frame_ms + out->sys_per_frame_ms;
    out->pcpu_per_frame_ms = pcpu / (double)n;
    out->p50_ms = sorted[pct_index(n, 0.50)];
    out->p95_ms = sorted[pct_index(n, 0.95)];

    /* Same buffer, second distribution: the CPU-time clock is the low-noise
     * cross-check on the getrusage differencing above. */
    for (int i = 0; i < n; i++) sorted[i] = pcpu_ms[i];
    qsort(sorted, (size_t)n, sizeof(double), cmp_double);
    out->pcpu_p50_ms = sorted[pct_index(n, 0.50)];
    free(sorted);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Env parsing                                                         */
/* ------------------------------------------------------------------ */

/* atoi() maps "", "abc" and out-of-range text to 0 (or wraps), which turns a
 * typo into a 0-frame run and a divide by zero downstream. */
static int env_int(const char *name, int def, int lo, int hi)
{
    const char *raw = getenv(name);
    char *end = NULL;
    long v;

    if (!raw || !raw[0]) return def;
    errno = 0;
    v = strtol(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0') {
        fprintf(stderr, "harness: %s=\"%s\" is not an integer; using %d\n", name, raw, def);
        return def;
    }
    if (v < (long)lo) {
        fprintf(stderr, "harness: %s=%ld is below the minimum %d; clamping\n", name, v, lo);
        v = lo;
    }
    if (v > (long)hi) {
        fprintf(stderr, "harness: %s=%ld is above the maximum %d; clamping\n", name, v, hi);
        v = hi;
    }
    return (int)v;
}

/* Flags are presence-with-value so that FOO=0 disables, instead of the
 * getenv()!=NULL spelling that makes FOO=0 mean "on". */
static int env_flag(const char *name)
{
    const char *raw = getenv(name);
    if (!raw || !raw[0]) return 0;
    return strcmp(raw, "0") != 0;
}

/* ------------------------------------------------------------------ */
/* GL loader (same pattern as mgl_benchmark.c)                          */
/* ------------------------------------------------------------------ */

static PFNGLBINDBUFFERPROC p_glBindBuffer;
static PFNGLBINDBUFFERRANGEPROC p_glBindBufferRange;
static PFNGLBUFFERDATAPROC p_glBufferData;
static PFNGLDRAWARRAYSPROC p_glDrawArrays;
static PFNGLDRAWELEMENTSPROC p_glDrawElements;
static PFNGLBINDVERTEXARRAYPROC p_glBindVertexArray;
static PFNGLBINDVERTEXBUFFERPROC p_glBindVertexBuffer;
static PFNGLVERTEXATTRIBFORMATPROC p_glVertexAttribFormat;
static PFNGLVERTEXATTRIBBINDINGPROC p_glVertexAttribBinding;
static PFNGLENABLEVERTEXATTRIBARRAYPROC p_glEnableVertexAttribArray;
static PFNGLGENVERTEXARRAYSPROC p_glGenVertexArrays;
static PFNGLBINDTEXTUREPROC p_glBindTexture;
static PFNGLACTIVETEXTUREPROC p_glActiveTexture;
static PFNGLBINDSAMPLERPROC p_glBindSampler;
static PFNGLBINDFRAMEBUFFERPROC p_glBindFramebuffer;
static PFNGLBLITFRAMEBUFFERPROC p_glBlitFramebuffer;
static PFNGLVIEWPORTPROC p_glViewport;
static PFNGLUSEPROGRAMPROC p_glUseProgram;
static PFNGLGETUNIFORMLOCATIONPROC p_glGetUniformLocation;
static PFNGLUNIFORM1IPROC p_glUniform1i;
static PFNGLGETUNIFORMBLOCKINDEXPROC p_glGetUniformBlockIndex;
static PFNGLUNIFORMBLOCKBINDINGPROC p_glUniformBlockBinding;
static PFNGLGETACTIVEUNIFORMBLOCKIVPROC p_glGetActiveUniformBlockiv;
static PFNGLGENSAMPLERSPROC p_glGenSamplers;
static PFNGLSAMPLERPARAMETERIPROC p_glSamplerParameteri;
static PFNGLDELETESAMPLERSPROC p_glDeleteSamplers;

#define LOAD_GL(ptr, type, name) \
    do { \
        ptr = (type)glfwGetProcAddress(name); \
        if (!ptr) { fprintf(stderr, "missing GL entry %s\n", name); return 1; } \
    } while (0)

static int load_gl(void)
{
    LOAD_GL(p_glBindBuffer, PFNGLBINDBUFFERPROC, "glBindBuffer");
    LOAD_GL(p_glBindBufferRange, PFNGLBINDBUFFERRANGEPROC, "glBindBufferRange");
    LOAD_GL(p_glBufferData, PFNGLBUFFERDATAPROC, "glBufferData");
    LOAD_GL(p_glDrawArrays, PFNGLDRAWARRAYSPROC, "glDrawArrays");
    LOAD_GL(p_glDrawElements, PFNGLDRAWELEMENTSPROC, "glDrawElements");
    LOAD_GL(p_glBindVertexArray, PFNGLBINDVERTEXARRAYPROC, "glBindVertexArray");
    LOAD_GL(p_glBindVertexBuffer, PFNGLBINDVERTEXBUFFERPROC, "glBindVertexBuffer");
    LOAD_GL(p_glVertexAttribFormat, PFNGLVERTEXATTRIBFORMATPROC, "glVertexAttribFormat");
    LOAD_GL(p_glVertexAttribBinding, PFNGLVERTEXATTRIBBINDINGPROC, "glVertexAttribBinding");
    LOAD_GL(p_glEnableVertexAttribArray, PFNGLENABLEVERTEXATTRIBARRAYPROC, "glEnableVertexAttribArray");
    LOAD_GL(p_glGenVertexArrays, PFNGLGENVERTEXARRAYSPROC, "glGenVertexArrays");
    LOAD_GL(p_glBindTexture, PFNGLBINDTEXTUREPROC, "glBindTexture");
    LOAD_GL(p_glActiveTexture, PFNGLACTIVETEXTUREPROC, "glActiveTexture");
    LOAD_GL(p_glBindSampler, PFNGLBINDSAMPLERPROC, "glBindSampler");
    LOAD_GL(p_glBindFramebuffer, PFNGLBINDFRAMEBUFFERPROC, "glBindFramebuffer");
    LOAD_GL(p_glBlitFramebuffer, PFNGLBLITFRAMEBUFFERPROC, "glBlitFramebuffer");
    LOAD_GL(p_glViewport, PFNGLVIEWPORTPROC, "glViewport");
    LOAD_GL(p_glUseProgram, PFNGLUSEPROGRAMPROC, "glUseProgram");
    LOAD_GL(p_glGetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC, "glGetUniformLocation");
    LOAD_GL(p_glUniform1i, PFNGLUNIFORM1IPROC, "glUniform1i");
    LOAD_GL(p_glGetUniformBlockIndex, PFNGLGETUNIFORMBLOCKINDEXPROC, "glGetUniformBlockIndex");
    LOAD_GL(p_glUniformBlockBinding, PFNGLUNIFORMBLOCKBINDINGPROC, "glUniformBlockBinding");
    LOAD_GL(p_glGetActiveUniformBlockiv, PFNGLGETACTIVEUNIFORMBLOCKIVPROC, "glGetActiveUniformBlockiv");
    LOAD_GL(p_glGenSamplers, PFNGLGENSAMPLERSPROC, "glGenSamplers");
    LOAD_GL(p_glSamplerParameteri, PFNGLSAMPLERPARAMETERIPROC, "glSamplerParameteri");
    LOAD_GL(p_glDeleteSamplers, PFNGLDELETESAMPLERSPROC, "glDeleteSamplers");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Optional probe counter (instrumented baseline builds only)          */
/* ------------------------------------------------------------------ */

static unsigned long *g_probe_counter;

static unsigned long probe_count(void)
{
    return g_probe_counter ? *g_probe_counter : 0UL;
}

/* ------------------------------------------------------------------ */
/* Harness scene                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    /* GL objects */
    GLuint vaos[VAO_CACHE_SIZE];
    GLuint ebos[VAO_CACHE_SIZE];
    GLuint vertex_arena;        /* one big VBO, per-format slots like GlBuffer reuse */
    GLuint dynamic_ubo[UBO_RING_BUFFERS];  /* DynamicUniforms ring */
    GLuint fbos[PASS_COUNT];
    GLuint fbo_color[PASS_COUNT];   /* attachments owned so a re-init/teardown frees them */
    GLuint fbo_depth[PASS_COUNT];
    GLuint programs[TERRAIN_LAYERS + 2];   /* 3 terrain layers + entity + GUI */
    GLint  sampler_locs[TERRAIN_LAYERS + 2];
    GLint  entity_sampler1_loc;  /* entity FS's second sampler -> unit 1 */
    GLuint textures[8];
    GLuint samplers[3];

    /* geometry shape */
    GLsizei vertex_slice_bytes;
    GLsizei index_count;
    int width, height;          /* drawable size; FBO attachments match it */

    /* per-draw UBO allocator (DynamicUniformStorage) */
    size_t ubo_block;           /* rounded TRANSFORM_UBO_SIZE */
    size_t ubo_next_block;
    int    ubo_ring_index;
    size_t ubo_ring_bytes;      /* capacity of ONE ring buffer, in bytes */
    int    ubo_ring_slices;     /* capacity of one ring buffer, in slices */
    int    ubo_overflow;        /* a slice was requested past the ring: run invalid */
    int    ubo_block_data_size; /* std140 size reported by the driver */

    /* state mirror to decide when the game would switch program/pipeline */
    int terrain_draws, entity_draws, gui_draws, frames;
    int emit_startup_location_queries;
} HarnessScene;

/* Terrain/entity VS: POSITION_TEX_LIGHTMAP_COLOR exactly as vanilla declares
 * it — SHORT buffer data (UV2) converted to a *float* shader input by
 * glVertexAttribFormat.  That conversion is the MGL cost this file measures.
 * The integer spelling (uvec2 + glVertexAttribIFormat) is a different,
 * non-vanilla path: a float-format attribute must not feed an integer shader
 * input, so mixing the two leaves the attribute undefined.  Staying on the
 * Format path here means the VS must declare aLight as vec2, and
 * validate_attributes() fails the run if the input stops being active. */
static const char *kTerrainVS =
    "#version 330 core\n"
    "layout(location=0) in vec3 aPos;\n"
    "layout(location=1) in vec2 aUV;\n"
    "layout(location=2) in vec2 aLight;\n"
    "layout(std140) uniform DynamicTransforms { mat4 uModelView; vec4 uColor; vec3 uOffset; mat4 uTexMat; };\n"
    "uniform sampler2D Sampler0;\n"
    "out vec2 vUV;\n"
    "out vec2 vLight;\n"
    "void main(){ vUV = aUV; vLight = aLight; gl_Position = uModelView * vec4(aPos + uOffset, 1.0); }\n";
static const char *kTerrainFS =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "in vec2 vLight;\n"
"layout(std140) uniform DynamicTransforms { mat4 uModelView; vec4 uColor; vec3 uOffset; mat4 uTexMat; };\n"
    "uniform sampler2D Sampler0;\n"
    "out vec4 C;\n"
    "void main(){ C = texture(Sampler0, vUV) * uColor * vec4(vLight * (1.0 / 255.0), 1.0, 1.0); }\n";
static const char *kEntityFS =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "in vec2 vLight;\n"
"layout(std140) uniform DynamicTransforms { mat4 uModelView; vec4 uColor; vec3 uOffset; mat4 uTexMat; };\n"
    "uniform sampler2D Sampler0;\n"
    "uniform sampler2D Sampler1;\n"
    "out vec4 C;\n"
    "void main(){ C = (texture(Sampler0, vUV) * 0.5 + texture(Sampler1, vUV) * 0.5) * uColor * vec4(vLight * (1.0 / 255.0), 1.0, 1.0); }\n";
static const char *kGuiVS =
    "#version 330 core\n"
    "layout(location=0) in vec3 aPos;\n"
    "layout(location=1) in vec2 aUV;\n"
    "layout(std140) uniform DynamicTransforms { mat4 uModelView; vec4 uColor; vec3 uOffset; mat4 uTexMat; };\n"
    "uniform sampler2D Sampler0;\n"
    "out vec2 vUV;\n"
    "void main(){ vUV = aUV; gl_Position = uModelView * vec4(aPos + uOffset, 1.0); }\n";
/* GUI gets its own FS: kTerrainFS reads the vLight varying that only
 * kTerrainVS writes, and an unwritten FS input is an undefined value. */
static const char *kGuiFS =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D Sampler0;\n"
    "out vec4 C;\n"
    "void main(){ C = texture(Sampler0, vUV); }\n";

static int check_shader(GLuint sh, const char *stage, const char *src)
{
    GLint ok = 0;
    char log[2048];
    GLsizei len = 0;

    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (ok) return 0;
    glGetShaderInfoLog(sh, (GLsizei)sizeof log, &len, log);
    fprintf(stderr, "harness: %s shader failed to compile: %.*s\n", stage, (int)len, log);
    fprintf(stderr, "harness: source was:\n%s", src);
    return -1;
}

static GLuint compile_program(const char *vs_src, const char *fs_src)
{
    GLuint v = 0, f = 0, p = 0;
    GLint ok = 0;

    v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs_src, NULL);
    glCompileShader(v);
    if (check_shader(v, "vertex", vs_src) != 0) goto fail;

    f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs_src, NULL);
    glCompileShader(f);
    if (check_shader(f, "fragment", fs_src) != 0) goto fail;

    p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei len = 0;
        glGetProgramInfoLog(p, (GLsizei)sizeof log, &len, log);
        fprintf(stderr, "harness: program link failed: %.*s\n", (int)len, log);
        goto fail;
    }
    glDetachShader(p, v);
    glDetachShader(p, f);
    glDeleteShader(v);
    glDeleteShader(f);
    return p;

fail:
    if (p) glDeleteProgram(p);
    if (v) glDeleteShader(v);
    if (f) glDeleteShader(f);
    return 0;
}

/* VertexArrayCache.Separate: one VAO per (format), attribute formats bound
 * once, per-VBO glBindVertexBuffer on change.  Terrain format here:
 * position(3f)@0 + uv(2f)@12 + light(2s)@20 + color(4ub normalized)@24 =
 * 28 bytes, i.e. POSITION_TEX_LIGHTMAP_COLOR as vanilla declares it. */
static void setup_format_vao(GLuint vao)
{
    p_glBindVertexArray(vao);
    for (GLuint i = 0; i < 4; i++) p_glEnableVertexAttribArray(i);
    p_glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
    p_glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 12);
    /* Vanilla light (UV2) is Type.SHORT via Usage.UV — a non-integer SHORT
     * attrib, not IFormat.  The integer spelling makes MGL's dynamic vertex
     * path run a whole-buffer format conversion per draw. */
    p_glVertexAttribFormat(2, 2, GL_SHORT, GL_FALSE, 20);
    p_glVertexAttribFormat(3, 4, GL_UNSIGNED_BYTE, GL_TRUE, 24);
    for (GLuint i = 0; i < 4; i++) p_glVertexAttribBinding(i, 0);
}

/* Slices one frame actually issues.  Single source of truth: the ring is sized
 * from this and the run label prints it, so the two cannot drift apart. */
static int draws_per_frame(const HarnessScene *s)
{
    return SKY_DRAWS + s->terrain_draws + s->entity_draws +
           s->terrain_draws / TRANSLUCENT_DIVISOR + CLOUD_DRAWS + s->gui_draws;
}

/* Map each program's DynamicTransforms block onto UBO_BINDING and verify the
 * mapping, the std140 size and the slice capacity all agree.  Without this the
 * shader's block keeps its implicit binding 0 while the harness binds ranges to
 * UBO_BINDING, i.e. every slice is bound to a point nothing reads. */
static int setup_ubo_bindings(HarnessScene *s)
{
    int nprograms = TERRAIN_LAYERS + 2;

    for (int i = 0; i < nprograms; i++) {
        GLuint block = p_glGetUniformBlockIndex(s->programs[i], "DynamicTransforms");
        GLint  binding = -1, size = 0;

        if (block == GL_INVALID_INDEX) {
            fprintf(stderr, "harness: program %d has no active DynamicTransforms block\n", i);
            return 1;
        }
        p_glUniformBlockBinding(s->programs[i], block, UBO_BINDING);
        glGetActiveUniformBlockiv(s->programs[i], block, GL_UNIFORM_BLOCK_BINDING, &binding);
        glGetActiveUniformBlockiv(s->programs[i], block, GL_UNIFORM_BLOCK_DATA_SIZE, &size);
        if (binding != (GLint)UBO_BINDING) {
            fprintf(stderr, "harness: program %d block binding is %d, expected %d\n",
                    i, binding, UBO_BINDING);
            return 1;
        }
        if (size <= 0 || (size_t)size > s->ubo_block) {
            fprintf(stderr, "harness: program %d block needs %d bytes, slice is %zu\n",
                    i, size, s->ubo_block);
            return 1;
        }
        s->ubo_block_data_size = size;
    }
    return 0;
}

/* The VAO feeds aLight through glVertexAttribFormat (float path), so the shader
 * must consume it as float: a float-format attribute feeding an integer input
 * is undefined, and the two spellings measure different MGL code.  Checked from
 * the program's active-attribute list rather than trusted, because a shader
 * edit that flips the type is otherwise invisible at run time.  The location
 * check pins what the VAO enables: layout(location=) is only a promise. */
static int expect_attribute(HarnessScene *s, int prog, const char *name,
                            GLint want_location, GLenum want_type)
{
    GLint count = 0, found = 0;
    GLint loc = glGetAttribLocation(s->programs[prog], name);

    if (loc != want_location) {
        fprintf(stderr, "harness: program %d attribute %s is at %d, expected %d\n",
                prog, name, loc, want_location);
        return 1;
    }
    glGetProgramiv(s->programs[prog], GL_ACTIVE_ATTRIBUTES, &count);
    for (GLint i = 0; i < count; i++) {
        char buf[256];
        GLsizei len = 0;
        GLint size = 0;
        GLenum type = 0;

        glGetActiveAttrib(s->programs[prog], (GLuint)i, (GLsizei)sizeof buf, &len, &size, &type, buf);
        if (strcmp(buf, name) != 0) continue;
        found = 1;
        if (type != want_type) {
            fprintf(stderr, "harness: program %d attribute %s is type 0x%04x, expected 0x%04x; "
                    "a float-format vertex array must not feed an integer input\n",
                    prog, name, type, want_type);
            return 1;
        }
    }
    if (!found) {
        fprintf(stderr, "harness: program %d has no active attribute %s\n", prog, name);
        return 1;
    }
    return 0;
}

static int validate_attributes(HarnessScene *s)
{
    for (int i = 0; i <= TERRAIN_LAYERS; i++) {
        if (expect_attribute(s, i, "aPos", 0, GL_FLOAT_VEC3) != 0) return 1;
        if (expect_attribute(s, i, "aUV", 1, GL_FLOAT_VEC2) != 0) return 1;
        /* SHORT buffer data converted to float by glVertexAttribFormat. */
        if (expect_attribute(s, i, "aLight", 2, GL_FLOAT_VEC2) != 0) return 1;
    }
    if (expect_attribute(s, TERRAIN_LAYERS + 1, "aPos", 0, GL_FLOAT_VEC3) != 0) return 1;
    if (expect_attribute(s, TERRAIN_LAYERS + 1, "aUV", 1, GL_FLOAT_VEC2) != 0) return 1;
    return 0;
}

/* Vanilla's GlProgram assigns each sampler its unit once at load time.  Read
 * the value back: a glUniform1i that silently does nothing (dead uniform, wrong
 * location) leaves every sampler on unit 0, which still "works" — two binds,
 * one sampler state — and quietly stops testing the second unit. */
static int bind_sampler_units(HarnessScene *s)
{
    for (int i = 0; i < TERRAIN_LAYERS + 2; i++) {
        GLint got = -1;

        p_glUseProgram(s->programs[i]);
        p_glUniform1i(s->sampler_locs[i], 0);
        if (i == TERRAIN_LAYERS) p_glUniform1i(s->entity_sampler1_loc, 1);

        glGetUniformiv(s->programs[i], s->sampler_locs[i], &got);
        if (got != 0) {
            fprintf(stderr, "harness: program %d Sampler0 reads unit %d, expected 0\n", i, got);
            return 1;
        }
        if (i == TERRAIN_LAYERS) {
            glGetUniformiv(s->programs[i], s->entity_sampler1_loc, &got);
            if (got != 1) {
                fprintf(stderr, "harness: entity Sampler1 reads unit %d, expected 1\n", got);
                return 1;
            }
        }
    }
    p_glUseProgram(0);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Drawable payload                                                    */
/* ------------------------------------------------------------------ */

/* POSITION_TEX_LIGHTMAP_COLOR, 28 bytes: pos(3f) uv0(2f) uv2(2s) color(4ub). */
typedef struct {
    float    pos[3];
    float    uv[2];
    int16_t  light[2];
    uint8_t  color[4];
} QuadVertex;

/* One unit quad per vertex slot.  Per-draw placement comes from the UBO offset
 * (each draw owns a slice), so every draw can share these six vertices and the
 * per-draw glBindVertexBuffer stride/offset model stays exactly as before. */
static void build_quads(QuadVertex *v, int slots)
{
    static const float corner[QUAD_VERTS][2] = {
        { -0.5f, -0.5f }, { 0.5f, -0.5f }, { 0.5f, 0.5f },
        { -0.5f, -0.5f }, { 0.5f, 0.5f },  { -0.5f, 0.5f },
    };

    for (int s = 0; s < slots; s++) {
        for (int i = 0; i < QUAD_VERTS; i++) {
            QuadVertex *q = &v[s * QUAD_VERTS + i];
            memset(q, 0, sizeof(*q));
            q->pos[0] = corner[i][0];
            q->pos[1] = corner[i][1];
            q->uv[0] = corner[i][0] + 0.5f;
            q->uv[1] = corner[i][1] + 0.5f;
            q->light[0] = 255;      /* shader scales by 1/255, so: full light */
            q->light[1] = 255;
            q->color[0] = q->color[1] = q->color[2] = q->color[3] = 255;
        }
    }
}


/* std140 DynamicTransforms for one slice: an orthographic projection over the
 * grid, a per-draw offset (the quad's cell), and an identity texture matrix.
 * Written once per slice and uploaded as the ring's initial content — vanilla
 * memcpy's these into a persistent map without a GL call, so doing it at init
 * keeps the per-draw call stream the same shape. */
static void fill_ubo_image(unsigned char *image, size_t block, int slices)
{
    const int rows = (slices + GRID_COLS - 1) / GRID_COLS;
    const float sx = 2.0f / (float)GRID_COLS;
    const float sy = 2.0f / (float)rows;

    memset(image, 0, block * (size_t)slices);
    for (int k = 0; k < slices; k++) {
        float *f = (float *)(void *)(image + (size_t)k * block);
        int col = k % GRID_COLS, row = k / GRID_COLS;

        f[0] = sx;                                  /* uModelView: scale x */
        f[5] = sy;                                  /*                y */
        f[15] = 1.0f;                               /*                w */
        /* uColor: a vertical brightness gradient over the grid, so the frame
         * reads as a scene and per-draw uniform delivery is visible. */
        {
            float t = 0.62f + 0.38f * (float)row / (float)(rows > 1 ? rows - 1 : 1);

            f[16] = f[17] = f[18] = t;
        }
        f[19] = 1.0f;
        f[20] = (float)col + 0.5f - 0.5f * (float)GRID_COLS;
        f[21] = (float)row + 0.5f - 0.5f * (float)rows;
        f[24] = f[29] = f[34] = f[39] = 1.0f;       /* uTexMat = identity */
    }
}

static int init_scene(HarnessScene *s, int width, int height)
{
    memset(s, 0, sizeof(*s));
    s->terrain_draws = env_int("HARNESS_TERRAIN_DRAWS", DEFAULT_TERRAIN_DRAWS, 0, MAX_DRAWS);
    s->entity_draws = env_int("HARNESS_ENTITY_DRAWS", DEFAULT_ENTITY_DRAWS, 0, MAX_DRAWS);
    s->gui_draws = env_int("HARNESS_GUI_DRAWS", DEFAULT_GUI_DRAWS, 0, MAX_DRAWS);
    s->frames = env_int("HARNESS_FRAMES", 120, 1, MAX_FRAMES);
    s->emit_startup_location_queries = env_flag("HARNESS_STARTUP_QUERIES");

    /* POSITION_TEX_LIGHTMAP_COLOR: pos(3f) uv0(2f) uv2(2s) color(4ub-norm) */
    s->vertex_slice_bytes = 28;
    s->index_count = QUAD_VERTS;
    s->width = width;
    s->height = height;

    glGenVertexArrays(VAO_CACHE_SIZE, s->vaos);
    for (int i = 0; i < VAO_CACHE_SIZE; i++) setup_format_vao(s->vaos[i]);

    glGenBuffers(1, &s->vertex_arena);
    p_glBindBuffer(GL_ARRAY_BUFFER, s->vertex_arena);
    {
        QuadVertex quads[64 * QUAD_VERTS];

        build_quads(quads, 64);
        p_glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof quads, quads, GL_STATIC_DRAW);
    }

    glGenBuffers(VAO_CACHE_SIZE, s->ebos);
    for (int i = 0; i < VAO_CACHE_SIZE; i++) {
        /* build_quads writes six standalone vertices (two triangles, not a
         * shared four-vertex strip), so the indices are just 0..5. */
        static const GLuint indices[QUAD_VERTS] = { 0, 1, 2, 3, 4, 5 };

        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i]);
        p_glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)sizeof indices, indices, GL_STATIC_DRAW);
    }

    /* DynamicUniforms ring: persistent storage, TRANSFORM_UBO_SIZE blocks
     * rounded to the device uniform offset alignment (256 on Apple GL).  The
     * game can wrap mid-frame because it re-syncs the mapped range per frame;
     * a plain buffer cannot, so the ring must hold one frame's slices. */
    s->ubo_block = ((size_t)TRANSFORM_UBO_SIZE + UBO_BLOCK_ALIGN - 1) &
                   ~((size_t)UBO_BLOCK_ALIGN - 1);
    s->ubo_ring_slices = draws_per_frame(s) + UBO_RING_HEADROOM;
    s->ubo_ring_bytes = s->ubo_block * (size_t)s->ubo_ring_slices;
    if (s->ubo_ring_bytes > (size_t)1 << 31) {
        fprintf(stderr, "harness: UBO ring would be %zu bytes; lower HARNESS_*_DRAWS\n",
                s->ubo_ring_bytes);
        return 1;
    }
    glGenBuffers(UBO_RING_BUFFERS, s->dynamic_ubo);
    {
        unsigned char *image = calloc(1, s->ubo_ring_bytes);

        if (!image) {
            fprintf(stderr, "harness: out of memory for the %zu-byte UBO image\n",
                    s->ubo_ring_bytes);
            return 1;
        }
        fill_ubo_image(image, s->ubo_block, s->ubo_ring_slices);
        for (int i = 0; i < UBO_RING_BUFFERS; i++) {
            p_glBindBuffer(GL_UNIFORM_BUFFER, s->dynamic_ubo[i]);
            p_glBufferData(GL_UNIFORM_BUFFER, (GLsizeiptr)s->ubo_ring_bytes, image, GL_DYNAMIC_DRAW);
        }
        free(image);
    }

    /* Render-pass targets: every pass FBO gets color+depth like the real
     * GpuTextureView FBOs (createRenderPass would otherwise fail validation). */
    glGenFramebuffers(PASS_COUNT, s->fbos);
    glGenTextures(PASS_COUNT, s->fbo_color);
    glGenTextures(PASS_COUNT, s->fbo_depth);
    for (int i = 0; i < PASS_COUNT; i++) {
        p_glBindTexture(GL_TEXTURE_2D, s->fbo_color[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, s->width, s->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        p_glBindTexture(GL_TEXTURE_2D, s->fbo_depth[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, s->width, s->height, 0,
                     GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
        p_glBindFramebuffer(GL_FRAMEBUFFER, s->fbos[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s->fbo_color[i], 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, s->fbo_depth[i], 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "harness: fbo %d incomplete\n", i);
            return 1;
        }
    }
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);

    for (int i = 0; i < TERRAIN_LAYERS; i++) {
        s->programs[i] = compile_program(kTerrainVS, kTerrainFS);
        if (!s->programs[i]) return 1;
        s->sampler_locs[i] = p_glGetUniformLocation(s->programs[i], "Sampler0");
    }
    s->programs[TERRAIN_LAYERS] = compile_program(kTerrainVS, kEntityFS);
    if (!s->programs[TERRAIN_LAYERS]) return 1;
    s->sampler_locs[TERRAIN_LAYERS] = p_glGetUniformLocation(s->programs[TERRAIN_LAYERS], "Sampler0");
    s->entity_sampler1_loc = p_glGetUniformLocation(s->programs[TERRAIN_LAYERS], "Sampler1");

    s->programs[TERRAIN_LAYERS + 1] = compile_program(kGuiVS, kGuiFS);
    if (!s->programs[TERRAIN_LAYERS + 1]) return 1;
    s->sampler_locs[TERRAIN_LAYERS + 1] = p_glGetUniformLocation(s->programs[TERRAIN_LAYERS + 1], "Sampler0");

    for (int i = 0; i < TERRAIN_LAYERS + 2; i++) {
        if (s->sampler_locs[i] < 0) {
            fprintf(stderr, "harness: program %d has no active Sampler0 uniform\n", i);
            return 1;
        }
    }
    /* Without an explicit unit Sampler1 keeps its default 0, so the entity
     * pass would sample unit 0 twice — two binds, one sampler state. */
    if (s->entity_sampler1_loc < 0) {
        fprintf(stderr, "harness: entity program has no active Sampler1 uniform\n");
        return 1;
    }
    if (validate_attributes(s) != 0) return 1;
    if (bind_sampler_units(s) != 0) return 1;
    if (setup_ubo_bindings(s) != 0) return 1;

    glGenTextures(8, s->textures);
    {
        /* One atlas, uploaded into every texture name.  Terrain sections all
         * sample the same block atlas in the real game, and this keeps the
         * image correct no matter which unit's texture a backend ends up
         * resolving — per-draw variety comes from uColor instead. */
        static const uint8_t tile[8][3] = {
            {  96, 140,  64 }, { 148, 184, 100 }, {  72, 104, 160 }, { 168,  96,  72 },
            { 176, 160, 120 }, { 104,  88, 144 }, {  64, 120, 132 }, { 200, 208, 216 },
        };
        uint8_t px[64 * 64 * 4];

        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                const uint8_t *c = tile[((y / 32) * 2 + (x / 32)) % 8];
                uint8_t *d = px + ((size_t)y * 64 + x) * 4;

                d[0] = c[0];
                d[1] = c[1];
                d[2] = c[2];
                d[3] = 255;
            }
        }
        for (int i = 0; i < 8; i++) {
            p_glBindTexture(GL_TEXTURE_2D, s->textures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
            /* Minecraft always sets explicit filters on its textures; without
             * this the default mipmap filter makes the texture sampling-
             * incomplete and MGL substitutes its fallback texture per draw. */
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
    }
    p_glGenSamplers(3, s->samplers);
    /* RenderPipeline sampler descriptions always set explicit filters; a
     * default sampler object (mipmap min filter) would make every sampled
     * texture incomplete and push MGL into its fallback path. */
    for (int i = 0; i < 3; i++) {
        p_glSamplerParameteri(s->samplers[i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        p_glSamplerParameteri(s->samplers[i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        p_glSamplerParameteri(s->samplers[i], GL_TEXTURE_WRAP_S, GL_REPEAT);
        p_glSamplerParameteri(s->samplers[i], GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    return 0;
}

/* Balances init_scene; safe to call on a partially initialised scene (object
 * names are zero from the memset and glDelete* ignore 0). */
static void destroy_scene(HarnessScene *s)
{
    for (int i = 0; i < TERRAIN_LAYERS + 2; i++)
        if (s->programs[i]) glDeleteProgram(s->programs[i]);
    p_glDeleteSamplers(3, s->samplers);
    glDeleteTextures(8, s->textures);
    glDeleteFramebuffers(PASS_COUNT, s->fbos);
    glDeleteTextures(PASS_COUNT, s->fbo_color);
    glDeleteTextures(PASS_COUNT, s->fbo_depth);
    glDeleteVertexArrays(VAO_CACHE_SIZE, s->vaos);
    glDeleteBuffers(VAO_CACHE_SIZE, s->ebos);
    glDeleteBuffers(UBO_RING_BUFFERS, s->dynamic_ubo);
    if (s->vertex_arena) glDeleteBuffers(1, &s->vertex_arena);
    memset(s, 0, sizeof(*s));
}

/* GlRenderPass.setUniform + trySetup's per-draw UBO slice binding.
 * Returns non-zero once the ring is exhausted; the caller then skips the draw
 * instead of binding a range that does not exist. */
static int bind_draw_ubo_slice(HarnessScene *s, GLuint program_binding)
{
    size_t offset;

    if (s->ubo_overflow) return -1;
    offset = s->ubo_next_block * s->ubo_block;

    /* Running past the store is not an error MGL reports (it implements the
     * GL 4.2+ "range may exceed the store" reading and clamps at use time), so
     * an over-full ring silently measures nonsense.  Refuse to bind, and let
     * the run die with rc=2. */
    if (offset + s->ubo_block > s->ubo_ring_bytes) {
        fprintf(stderr,
                "harness: UBO ring exhausted at slice %zu of %d (%zu-byte ring); "
                "raise UBO_RING_HEADROOM or fix draws_per_frame()\n",
                s->ubo_next_block, s->ubo_ring_slices, s->ubo_ring_bytes);
        s->ubo_overflow = 1;
        return -1;
    }
    p_glBindBufferRange(GL_UNIFORM_BUFFER, program_binding,
                        s->dynamic_ubo[s->ubo_ring_index],
                        (GLintptr)offset, (GLsizeiptr)s->ubo_block);
    s->ubo_next_block++;
    return 0;
}

/* GlCommandEncoder.trySetup sampler block: runs for every sampler uniform on
 * every trySetup, unconditionally (bind + sampler + min/max LOD params).
 * HARNESS_SKIP_LOD_PARAMS=1 drops only the two glTexParameteri calls to
 * isolate their MGL-side cost; HARNESS_SKIP_SAMPLER_BLOCK=1 drops the whole
 * block (both are measurement knobs, vanilla keeps them). */
static int g_skip_lod_params;
static int g_skip_sampler_block;
/* Visual mode: visible window, shared pass target, blit to the drawable.
 * HARNESS_SUBMIT_ONLY=1 turns it off for pure submit timing. */
static int g_visual = 1;
/* Present every frame like the real game (default on).  Disabling is only
 * for pure submit stress; unbounded in-flight commits then accumulate, so the
 * frame count is capped (see MAX_NO_PRESENT_FRAMES). */
static int g_present_per_frame = 1;
/* Drain the GPU inside the frame (separately timed) instead of leaving the
 * queue in flight; without it the wall number is submit+driver, not completion. */
static int g_finish_per_frame;
static int g_check_gl;
static unsigned long g_gl_errors;
static GLFWwindow *g_window;

static void try_setup_samplers(int sampler_count, GLuint tex0, GLuint tex1, GLuint sampler)
{
    if (g_skip_sampler_block) return;
    for (int u = 0; u < sampler_count; u++) {
        p_glActiveTexture(GL_TEXTURE0 + u);
        p_glBindTexture(GL_TEXTURE_2D, u == 0 ? tex0 : tex1);
        p_glBindSampler(u, sampler);
        if (!g_skip_lod_params) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LOD, 4);
        }
    }
}

/* applyPipelineState on pipeline change (depth/cull/blend/polygon/masks). */
static void apply_pipeline_state(int depth, int cull, int blend)
{
    if (depth) {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    if (cull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    if (blend) {
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    } else {
        glDisable(GL_BLEND);
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
}

/* createRenderPass: FBO bind (+ optional clear) + viewport.  In visual mode
 * every pass accumulates on pass 0's target — that is how the game's main pass
 * renders, and it is what makes the window show the scene instead of one
 * pass's worth of it.  Submit-only keeps one target per pass. */
static GLuint pass_target(const HarnessScene *s, int pass)
{
    return g_visual ? s->fbos[0] : s->fbos[pass];
}

static void begin_pass(HarnessScene *s, int pass)
{
    /* Every target the frame draws into must start defined, or the window shows
     * uninitialised memory.  Submit-only keeps the traced ~2 clears/frame. */
    int clear = g_visual ? (pass == 0) : (pass < CLEAR_PASSES);

    p_glBindFramebuffer(GL_FRAMEBUFFER, pass_target(s, pass));
    if (clear) {
        glDisable(GL_SCISSOR_TEST);
        glDepthMask(GL_TRUE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glClearColor(0.5f, 0.7f, 1.0f, 1.0f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    p_glViewport(0, 0, s->width, s->height);
}

/* VertexArrayCache.bindVertexArray on the Separate path: VAO cache hit +
 * glBindVertexBuffer when the VBO changes (here: rarely — slot cycling). */
static void bind_vertex_source(HarnessScene *s, int slot, int *last_slot)
{
    /* VertexArrayCache keys by format: every terrain section shares ONE VAO;
     * per-section variation is the vertex-buffer binding (offset here, a
     * separate section GpuBuffer in the real game). */
    p_glBindVertexArray(s->vaos[0]);
    if (slot != *last_slot) {
        p_glBindVertexBuffer(0, s->vertex_arena,
                             (GLintptr)((size_t)(slot % 64) * QUAD_VERTS *
                                        (size_t)s->vertex_slice_bytes),
                             s->vertex_slice_bytes);
        *last_slot = slot;
    }
}

/* GlCommandEncoder.finishRenderPass + the present that reaches the window.
 * The FBO is the frame's real target; the drawable only ever sees a blit, which
 * is also what a game does when it resolves to the default framebuffer. */
static void present_frame(HarnessScene *s)
{
    if (!g_visual) {
        p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return;
    }
    p_glBindFramebuffer(GL_READ_FRAMEBUFFER, s->fbos[0]);
    p_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    p_glBlitFramebuffer(0, 0, s->width, s->height,
                        0, 0, s->width, s->height,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void draw_frame(HarnessScene *s, int frame)
{
    int last_vbo_slot = -1;
    int last_program = -1;

    /* DynamicUniforms.reset(): rotate ring, reset allocator. */
    s->ubo_ring_index = frame % UBO_RING_BUFFERS;
    s->ubo_next_block = 0;

    /* ---- pass 0: sky (array draws, no depth, own program) ---- */
    begin_pass(s, 0);
    apply_pipeline_state(0, 1, 0);
    if (last_program != 0) {
        p_glUseProgram(s->programs[0]);
        p_glUniform1i(s->sampler_locs[0], 0);
        last_program = 0;
    }
    for (int i = 0; i < SKY_DRAWS; i++) {
        bind_vertex_source(s, i, &last_vbo_slot);
        try_setup_samplers(1, s->textures[0], s->textures[0], s->samplers[0]);
        if (bind_draw_ubo_slice(s, UBO_BINDING) != 0) continue;
        p_glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    /* ---- pass 1: terrain (3 layer programs, indexed, atlas texture) ----
     * Vanilla renders layer-by-layer (SectionRenderDispatcher runs one
     * section sweep per layer), so the program changes per LAYER, not per
     * draw. */
    begin_pass(s, 1);
    apply_pipeline_state(1, 1, 0);
    for (int layer = 0; layer < TERRAIN_LAYERS; layer++) {
      for (int i = layer; i < s->terrain_draws; i += TERRAIN_LAYERS) {
        if (last_program != layer) {
            p_glUseProgram(s->programs[layer]);
            p_glUniform1i(s->sampler_locs[layer], 0);
            last_program = layer;
        }
        bind_vertex_source(s, i, &last_vbo_slot);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i % VAO_CACHE_SIZE]);
        try_setup_samplers(1, s->textures[0], s->textures[0], s->samplers[0]);
        if (bind_draw_ubo_slice(s, UBO_BINDING) != 0) continue;
        p_glDrawElements(GL_TRIANGLES, s->index_count, GL_UNSIGNED_INT, (void *)0);
      }
    }

    /* ---- pass 2: entities (per-draw texture alternation, 2 samplers) ---- */
    begin_pass(s, 2);
    apply_pipeline_state(1, 0, 0);
    for (int i = 0; i < s->entity_draws; i++) {
        if (last_program != TERRAIN_LAYERS) {
            p_glUseProgram(s->programs[TERRAIN_LAYERS]);
            p_glUniform1i(s->sampler_locs[TERRAIN_LAYERS], 0);
            /* Sampler1 -> unit 1, or both samplers read unit 0 and the pass
             * stops exercising a second sampler state. */
            p_glUniform1i(s->entity_sampler1_loc, 1);
            last_program = TERRAIN_LAYERS;
        }
        bind_vertex_source(s, i, &last_vbo_slot);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i % VAO_CACHE_SIZE]);
        try_setup_samplers(2, s->textures[1 + (i % ENTITY_TEXTURE_CYCLE)],
                           s->textures[1 + ((i + 1) % ENTITY_TEXTURE_CYCLE)], s->samplers[1]);
        if (bind_draw_ubo_slice(s, UBO_BINDING) != 0) continue;
        p_glDrawElements(GL_TRIANGLES, s->index_count, GL_UNSIGNED_INT, (void *)0);
    }

    /* ---- pass 3: translucent (blend on) ---- */
    begin_pass(s, 3);
    apply_pipeline_state(1, 0, 1);
    for (int i = 0; i < s->terrain_draws / TRANSLUCENT_DIVISOR; i++) {
        int layer = i % TERRAIN_LAYERS;
        if (last_program != layer) {
            p_glUseProgram(s->programs[layer]);
            p_glUniform1i(s->sampler_locs[layer], 0);
            last_program = layer;
        }
        bind_vertex_source(s, i, &last_vbo_slot);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i % VAO_CACHE_SIZE]);
        try_setup_samplers(1, s->textures[0], s->textures[0], s->samplers[2]);
        if (bind_draw_ubo_slice(s, UBO_BINDING) != 0) continue;
        p_glDrawElements(GL_TRIANGLES, s->index_count, GL_UNSIGNED_INT, (void *)0);
    }

    /* ---- pass 4: clouds + weather (small, array draws, blend) ---- */
    begin_pass(s, 4);
    apply_pipeline_state(0, 0, 1);
    for (int i = 0; i < CLOUD_DRAWS; i++) {
        bind_vertex_source(s, i, &last_vbo_slot);
        try_setup_samplers(1, s->textures[7], s->textures[7], s->samplers[2]);
        if (bind_draw_ubo_slice(s, UBO_BINDING) != 0) continue;
        p_glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    /* ---- pass 5: GUI (blend, texture per 8 draws, scissor sometimes) ---- */
    begin_pass(s, 5);
    apply_pipeline_state(0, 0, 1);
    if (last_program != TERRAIN_LAYERS + 1) {
        p_glUseProgram(s->programs[TERRAIN_LAYERS + 1]);
        p_glUniform1i(s->sampler_locs[TERRAIN_LAYERS + 1], 0);
        last_program = TERRAIN_LAYERS + 1;
    }
    for (int i = 0; i < s->gui_draws; i++) {
        if ((i % 16) == 0) {
            glEnable(GL_SCISSOR_TEST);
            /* Vanilla scissor rects track GUI widget bounds; keeping the call but
             * covering the whole target means the visual pass is not clipped away. */
            glScissor(0, 0, s->width, s->height);
        }
        bind_vertex_source(s, 0, &last_vbo_slot);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->ebos[i % VAO_CACHE_SIZE]);
        try_setup_samplers(1, s->textures[(i / GUI_TEXTURE_GROUP) % 4],
                           s->textures[(i / GUI_TEXTURE_GROUP) % 4], s->samplers[0]);
        if (bind_draw_ubo_slice(s, UBO_BINDING) != 0) continue;
        p_glDrawElements(GL_TRIANGLES, s->index_count, GL_UNSIGNED_INT, (void *)0);
    }
    glDisable(GL_SCISSOR_TEST);

    /* GlCommandEncoder.finishRenderPass, then put the frame on the drawable */
    present_frame(s);

    /* The slice count is what sizes the ring, so a disagreement here means the
     * modelled draw mix and the emitted stream have drifted apart — the next
     * run would overflow silently.  Flag it instead of measuring on. */
    if (s->ubo_next_block != (size_t)draws_per_frame(s)) {
        if (!s->ubo_overflow) {
            fprintf(stderr,
                    "harness: frame %d issued %zu UBO slices but draws_per_frame() says %d\n",
                    frame, s->ubo_next_block, draws_per_frame(s));
        }
        s->ubo_overflow = 1;
    }
}

/* Startup phase stand-in: a shaderpack-style program whose uniform locations
 * get resolved once, in a window clean enough to time. */
static void run_startup_location_queries(void)
{
    char fs_src[2048];
    size_t off = 0;
    const char *fsrc = NULL, *vs1 = "#version 330 core\nvoid main(){gl_Position=vec4(0.0);}\n";
    GLuint v = 0, f = 0, p = 0;
    GLint ok = 0;

    off += (size_t)snprintf(fs_src + off, sizeof fs_src - off, "#version 330 core\nout vec4 C;\n");
    for (int i = 0; i < 24; i++)
        off += (size_t)snprintf(fs_src + off, sizeof fs_src - off, "uniform vec4 uV%d;\n", i);
    off += (size_t)snprintf(fs_src + off, sizeof fs_src - off, "void main(){C=uV0;}\n");
    fsrc = fs_src;

    f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fsrc, NULL);
    glCompileShader(f);
    if (check_shader(f, "startup fragment", fsrc) != 0) goto out;

    v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs1, NULL);
    glCompileShader(v);
    if (check_shader(v, "startup vertex", vs1) != 0) goto out;

    p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        fprintf(stderr, "harness: startup program link failed\n");
        goto out;
    }
    p_glUseProgram(p);
    for (int rep = 0; rep < 300; rep++) {
        for (int i = 0; i < 24; i++) {
            char name[16];
            snprintf(name, sizeof name, "uV%d", i);
            (void)p_glGetUniformLocation(p, name);
        }
    }
    p_glUseProgram(0);

out:
    if (p) glDeleteProgram(p);
    if (v) glDeleteShader(v);
    if (f) glDeleteShader(f);
}

static void drain_gl_errors(int frame)
{
    GLenum e;
    for (int k = 0; k < 64; k++) {
        e = glGetError();
        if (e == GL_NO_ERROR) return;
        g_gl_errors++;
        if (g_gl_errors <= 8)
            fprintf(stderr, "harness: GL error 0x%04x at frame %d\n", e, frame);
    }
}

/* Proof that the stream put something on the target: read pass 0's colour
 * attachment back once, at the end, and report how much of it stopped being the
 * clear colour.  A harness that issues 1384 draws and rasterises nothing looks
 * identical in the timings, so this is the only way to know the window is real.
 * Runs after the measurement, so the readback stall costs nothing. */
static void report_target_content(HarnessScene *s)
{
    const size_t pixels = (size_t)s->width * (size_t)s->height;
    unsigned char *px = malloc(pixels * 4);
    uint8_t seen[4096 / 8];
    size_t painted = 0;
    int distinct = 0;

    if (!px) return;
    memset(seen, 0, sizeof seen);

    p_glBindFramebuffer(GL_READ_FRAMEBUFFER, s->fbos[0]);
    glReadPixels(0, 0, s->width, s->height, GL_RGBA, GL_UNSIGNED_BYTE, px);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);

    for (size_t i = 0; i < pixels; i++) {
        const unsigned char *p = px + i * 4;
        unsigned hash;

        /* clear colour is (128, 178, 255); anything far from it was drawn */
        if (abs((int)p[0] - 128) > 8 || abs((int)p[1] - 178) > 8 || abs((int)p[2] - 255) > 8)
            painted++;
        hash = ((unsigned)p[0] * 257u ^ (unsigned)p[1] * 1299709u ^ (unsigned)p[2] * 65537u) & 4095u;
        if (!(seen[hash >> 3] & (1u << (hash & 7)))) {
            seen[hash >> 3] |= (uint8_t)(1u << (hash & 7));
            distinct++;
        }
    }
    printf("frame_pixels=%zu painted=%.1f%% distinct_colors=%d image=%dx%d\n",
           pixels, 100.0 * (double)painted / (double)pixels, distinct, s->width, s->height);

    /* Same question for the drawable: blit once more and read the window's back
     * buffer, so "the window shows the frame" is measured, not assumed. */
    {
        size_t in_window = 0;

        present_frame(s);
        glReadPixels(0, 0, s->width, s->height, GL_RGBA, GL_UNSIGNED_BYTE, px);
        for (size_t i = 0; i < pixels; i++) {
            const unsigned char *p = px + i * 4;

            if (abs((int)p[0] - 128) > 8 || abs((int)p[1] - 178) > 8 || abs((int)p[2] - 255) > 8)
                in_window++;
        }
        printf("drawable painted=%.1f%%\n", 100.0 * (double)in_window / (double)pixels);
    }

    /* HARNESS_DUMP=<path.ppm>: leave the frame on disk so the image can be
     * looked at instead of inferred from a percentage. */
    {
        const char *dump = getenv("HARNESS_DUMP");

        if (dump && dump[0]) {
            FILE *f = fopen(dump, "wb");

            if (!f) {
                fprintf(stderr, "harness: cannot write %s: %s\n", dump, strerror(errno));
            } else {
                fprintf(f, "P6\n%d %d\n255\n", s->width, s->height);
                for (size_t i = 0; i < pixels; i++) fwrite(px + i * 4, 1, 3, f);
                fclose(f);
                printf("dump=%s\n", dump);
            }
        }
    }
    free(px);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    int rc = 0;
    int glfw_ready = 0, scene_ready = 0;
    const char *json_path = NULL;
    FILE *json_fp = NULL;
    GLFWwindow *win = NULL;
    HarnessScene scene;
    double *wall_ms = NULL, *user_ms = NULL, *sys_ms = NULL, *swap_ms = NULL;
    double *pcpu_ms = NULL, *finish_ms = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "harness: --json needs a path\n");
                return 1;
            }
            json_path = argv[++i];
        } else {
            fprintf(stderr, "harness: ignoring unknown argument \"%s\"\n", argv[i]);
        }
    }
    /* Open the report up front: a bad path must fail before the frames run,
     * not be discovered (and swallowed) after them. */
    if (json_path) {
        json_fp = fopen(json_path, "w");
        if (!json_fp) {
            fprintf(stderr, "harness: cannot write JSON to %s: %s\n", json_path, strerror(errno));
            return 1;
        }
    }

    g_probe_counter = (unsigned long *)dlsym(RTLD_DEFAULT, "mgl_vm_probe_count");
    g_skip_lod_params = env_flag("HARNESS_SKIP_LOD_PARAMS");
    g_skip_sampler_block = env_flag("HARNESS_SKIP_SAMPLER_BLOCK");
    g_present_per_frame = !env_flag("HARNESS_NO_PRESENT");
    g_finish_per_frame = env_flag("HARNESS_FINISH");
    g_check_gl = env_flag("HARNESS_CHECK_GL");
    g_visual = !env_flag("HARNESS_SUBMIT_ONLY");

    int width = env_int("HARNESS_WIDTH", DEFAULT_WIDTH, 16, 8192);
    int height = env_int("HARNESS_HEIGHT", DEFAULT_HEIGHT, 16, 8192);

    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        rc = 1;
        goto done;
    }
    glfw_ready = 1;
    glfwWindowHint(GLFW_VISIBLE, g_visual ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    win = glfwCreateWindow(width, height, "mc_offline_harness", NULL, NULL);
    g_window = win;
    if (!win) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        rc = 1;
        goto done;
    }
    glfwMakeContextCurrent(win);
    /* Uncapped by default: a 60 Hz vsync would hide the cost this tool
     * measures.  HARNESS_VSYNC=1 gives a smooth window instead. */
    glfwSwapInterval(env_flag("HARNESS_VSYNC") ? 1 : 0);
    if (load_gl() != 0) {
        rc = 1;
        goto done;
    }

    /* init_scene memsets first, so a partial scene is still safe to destroy. */
    scene_ready = 1;
    if (init_scene(&scene, width, height) != 0) {
        rc = 1;
        goto done;
    }

    int warmup = env_int("HARNESS_WARMUP", 10, 0, MAX_WARMUP);

    /* Optional startup phase (shaderpack-style location resolution). */
    unsigned long probes_before_startup = probe_count();
    double startup_wall_0 = now_ns(CLOCK_MONOTONIC);
    if (scene.emit_startup_location_queries) run_startup_location_queries();
    double startup_wall_ms = (now_ns(CLOCK_MONOTONIC) - startup_wall_0) / 1e6;
    unsigned long startup_probes = probe_count() - probes_before_startup;

    /* A visible window must service its event loop or macOS marks the process
     * unresponsive; it also lets the user close the window to end the run. */
    for (int i = 0; i < warmup; i++) {
        draw_frame(&scene, i);
        if (g_visual) glfwPollEvents();
    }
    glFinish();

    int frames = scene.frames;
    if (!g_present_per_frame && frames > MAX_NO_PRESENT_FRAMES) {
        fprintf(stderr,
                "harness: HARNESS_NO_PRESENT=1 with %d frames; capping at %d "
                "(nothing drains the in-flight commits without a present)\n",
                frames, MAX_NO_PRESENT_FRAMES);
        frames = MAX_NO_PRESENT_FRAMES;
    }

    wall_ms = malloc((size_t)frames * sizeof(double));
    user_ms = malloc((size_t)frames * sizeof(double));
    sys_ms = malloc((size_t)frames * sizeof(double));
    swap_ms = malloc((size_t)frames * sizeof(double));
    pcpu_ms = malloc((size_t)frames * sizeof(double));
    finish_ms = malloc((size_t)frames * sizeof(double));
    if (!wall_ms || !user_ms || !sys_ms || !swap_ms || !pcpu_ms || !finish_ms) {
        fprintf(stderr, "harness: out of memory for %d frame samples\n", frames);
        rc = 1;
        goto done;
    }

    unsigned long probes_before = probe_count();
    struct rusage r0, r1;
    getrusage(RUSAGE_SELF, &r0);
    double wall0 = now_ns(CLOCK_MONOTONIC);
    int measured = 0;

    for (int f = 0; f < frames; f++) {
        struct rusage rf0, rf1;
        getrusage(RUSAGE_SELF, &rf0);
        double c0 = now_ns(CLOCK_PROCESS_CPUTIME_ID);
        double f0 = now_ns(CLOCK_MONOTONIC);
        draw_frame(&scene, f);
        double f1 = now_ns(CLOCK_MONOTONIC);
        double c1 = now_ns(CLOCK_PROCESS_CPUTIME_ID);
        /* Present like the real game (RenderSystem/Glfw swap every frame).
         * Without a per-frame present MGL's in-flight commits and CoW
         * snapshots are never drained and accumulate without bound — at
         * HARNESS_FRAMES=3000 that was tens of GB.  Present keeps in-flight
         * work bounded at one frame, exactly as the game does.  Swap time is
         * reported separately and not folded into the submit stats. */
        double s0 = now_ns(CLOCK_MONOTONIC);
        if (g_present_per_frame) glfwSwapBuffers(g_window);
        double s1 = now_ns(CLOCK_MONOTONIC);
        /* Optional in-frame drain: without it the wall number covers
         * submission and driver work, not completion.  Timed on its own so
         * the submit cost stays comparable across runs. */
        double q0 = 0.0, q1 = 0.0;
        if (g_finish_per_frame) {
            q0 = now_ns(CLOCK_MONOTONIC);
            glFinish();
            q1 = now_ns(CLOCK_MONOTONIC);
        }
        if (g_check_gl) drain_gl_errors(f);
        getrusage(RUSAGE_SELF, &rf1);

        wall_ms[f] = (f1 - f0) / 1e6;
        swap_ms[f] = (s1 - s0) / 1e6;
        finish_ms[f] = (q1 - q0) / 1e6;
        pcpu_ms[f] = (c1 - c0) / 1e6;
        user_ms[f] = ((double)(rf1.ru_utime.tv_sec - rf0.ru_utime.tv_sec) * 1e6 +
                      (double)(rf1.ru_utime.tv_usec - rf0.ru_utime.tv_usec)) / 1e3;
        sys_ms[f] = ((double)(rf1.ru_stime.tv_sec - rf0.ru_stime.tv_sec) * 1e6 +
                     (double)(rf1.ru_stime.tv_usec - rf0.ru_stime.tv_usec)) / 1e3;

        measured++;
        /* An exhausted ring means the frame no longer matches the model, so
         * further frames measure a different (shorter) stream.  Stop. */
        if (scene.ubo_overflow) break;
        if (g_visual) {
            glfwPollEvents();
            if (glfwWindowShouldClose(g_window)) break;
        }
    }

    if (measured > 0 && measured < frames) {
        fprintf(stderr, "harness: stopped after %d of %d frames (%s)\n", measured, frames,
                scene.ubo_overflow ? "UBO ring exhausted" : "window closed");
    }
    if (measured <= 0) {
        fprintf(stderr, "harness: no frame completed\n");
        rc = scene.ubo_overflow ? 2 : 1;
        goto done;
    }
    /* Every statistic below covers what actually ran, not what was asked for. */
    frames = measured;

    getrusage(RUSAGE_SELF, &r1);
    double wall_total_ms = (now_ns(CLOCK_MONOTONIC) - wall0) / 1e6;
    unsigned long probes_total = probe_count() - probes_before;

    FrameStats st;
    if (summarize(&st, wall_ms, user_ms, sys_ms, pcpu_ms, frames) != 0) {
        fprintf(stderr, "harness: summarize failed (frames=%d)\n", frames);
        rc = 1;
        goto done;
    }
    double *sorted = malloc((size_t)frames * sizeof(double));
    if (!sorted) {
        fprintf(stderr, "harness: out of memory for percentile sort\n");
        rc = 1;
        goto done;
    }
    memcpy(sorted, swap_ms, (size_t)frames * sizeof(double));
    qsort(sorted, (size_t)frames, sizeof(double), cmp_double);
    double swap_p50 = sorted[pct_index(frames, 0.50)];
    memcpy(sorted, finish_ms, (size_t)frames * sizeof(double));
    qsort(sorted, (size_t)frames, sizeof(double), cmp_double);
    double finish_p50 = sorted[pct_index(frames, 0.50)];
    free(sorted);

    double total_user_ms = ((double)(r1.ru_utime.tv_sec - r0.ru_utime.tv_sec) * 1e6 +
                            (double)(r1.ru_utime.tv_usec - r0.ru_utime.tv_usec)) / 1e3;
    double total_sys_ms = ((double)(r1.ru_stime.tv_sec - r0.ru_stime.tv_sec) * 1e6 +
                           (double)(r1.ru_stime.tv_usec - r0.ru_stime.tv_usec)) / 1e3;

    int per_frame = draws_per_frame(&scene);
    printf("backend=mgl frames=%d warmup=%d draws_per_frame=%d visual=%d image=%dx%d\n",
           frames, warmup, per_frame, g_visual, scene.width, scene.height);
    printf("frame_wall_ms=%.3f frame_cpu_ms=%.3f frame_user_ms=%.3f frame_sys_ms=%.3f "
           "frame_pcpu_ms=%.3f p50_ms=%.3f p95_ms=%.3f pcpu_p50_ms=%.3f "
           "swap_p50_ms=%.3f finish_p50_ms=%.3f present=%d\n",
           st.wall_per_frame_ms, st.cpu_per_frame_ms, st.user_per_frame_ms, st.sys_per_frame_ms,
           st.pcpu_per_frame_ms, st.p50_ms, st.p95_ms, st.pcpu_p50_ms,
           swap_p50, finish_p50, g_present_per_frame);
    printf("window_wall_ms=%.1f window_user_ms=%.1f window_sys_ms=%.1f\n",
           wall_total_ms, total_user_ms, total_sys_ms);
    printf("ubo slices_per_frame=%d ring_slices=%d ring_bytes=%zu block_bytes=%zu "
           "block_data_size=%d binding=%d overflow=%d\n",
           per_frame, scene.ubo_ring_slices, scene.ubo_ring_bytes, scene.ubo_block,
           scene.ubo_block_data_size, UBO_BINDING, scene.ubo_overflow);
    printf("probes_available=%d probes_per_frame=%.1f startup_queries_ms=%.3f startup_probes=%lu\n",
           g_probe_counter != NULL,
           g_probe_counter ? (double)probes_total / frames : 0.0,
           startup_wall_ms, startup_probes);
    printf("gl_errors=%lu checked=%d\n", g_gl_errors, g_check_gl);
    if (g_visual) report_target_content(&scene);

    if (json_fp) {
        int wrote = fprintf(json_fp,
                "{\"schema_version\":2,\"tool\":\"mc_offline_harness\",\"frames\":%d,"
                "\"warmup\":%d,\"draws_per_frame\":%d,"
                "\"frame_wall_ms\":%.3f,\"frame_cpu_ms\":%.3f,\"frame_user_ms\":%.3f,"
                "\"frame_sys_ms\":%.3f,\"frame_pcpu_ms\":%.3f,"
                "\"p50_ms\":%.3f,\"p95_ms\":%.3f,\"pcpu_p50_ms\":%.3f,"
                "\"swap_p50_ms\":%.3f,\"finish_p50_ms\":%.3f,\"present\":%s,"
                "\"ubo_slices_per_frame\":%d,\"ubo_ring_slices\":%d,\"ubo_ring_bytes\":%zu,"
                "\"ubo_block_data_size\":%d,\"ubo_overflow\":%s,"
                "\"window_user_ms\":%.1f,\"window_sys_ms\":%.1f,"
                "\"probes_available\":%s,\"probes_per_frame\":%.1f,"
                "\"startup_queries_ms\":%.3f,\"startup_probes\":%lu,\"gl_errors\":%lu,"
                "\"visual\":%s,\"width\":%d,\"height\":%d}\n",
                frames, warmup, per_frame,
                st.wall_per_frame_ms, st.cpu_per_frame_ms, st.user_per_frame_ms,
                st.sys_per_frame_ms, st.pcpu_per_frame_ms,
                st.p50_ms, st.p95_ms, st.pcpu_p50_ms,
                swap_p50, finish_p50, g_present_per_frame ? "true" : "false",
                per_frame, scene.ubo_ring_slices, scene.ubo_ring_bytes,
                scene.ubo_block_data_size, scene.ubo_overflow ? "true" : "false",
                total_user_ms, total_sys_ms,
                g_probe_counter ? "true" : "false",
                g_probe_counter ? (double)probes_total / frames : 0.0,
                startup_wall_ms, startup_probes, g_gl_errors,
                g_visual ? "true" : "false", scene.width, scene.height);
        if (wrote < 0 || fclose(json_fp) != 0) {
            fprintf(stderr, "harness: writing JSON to %s failed: %s\n", json_path, strerror(errno));
            json_fp = NULL;
            rc = 1;
        } else {
            json_fp = NULL;
        }
    }

    /* HARNESS_HOLD=<seconds>: keep presenting after the measurement so the
     * frame can actually be looked at.  Closing the window ends it early. */
    {
        int hold = env_int("HARNESS_HOLD", 0, 0, 3600);

        if (hold > 0 && g_visual) {
            double deadline = now_ns(CLOCK_MONOTONIC) + (double)hold * 1e9;
            int f = 0;

            printf("holding=%ds width=%d height=%d (close the window to end)\n",
                   hold, scene.width, scene.height);
            while (now_ns(CLOCK_MONOTONIC) < deadline) {
                draw_frame(&scene, f++);
                glfwSwapBuffers(g_window);
                glfwPollEvents();
                if (glfwWindowShouldClose(g_window)) break;
            }
        }
    }

    /* A run whose ring overflowed measured draws against slices that do not
     * exist; the numbers must not be mistaken for a valid baseline. */
    if (scene.ubo_overflow) {
        fprintf(stderr, "harness: UBO ring overflowed; this run is not a valid measurement\n");
        rc = 2;
    }
    if (g_check_gl && g_gl_errors) {
        fprintf(stderr, "harness: %lu GL errors were raised; measurement is suspect\n", g_gl_errors);
        if (rc == 0) rc = 2;
    }

done:
    if (scene_ready) destroy_scene(&scene);
    if (win) glfwDestroyWindow(win);
    g_window = NULL;
    if (glfw_ready) glfwTerminate();
    if (json_fp) fclose(json_fp);
    free(wall_ms);
    free(user_ms);
    free(sys_ms);
    free(swap_ms);
    free(pcpu_ms);
    free(finish_ms);
    return rc;
}
